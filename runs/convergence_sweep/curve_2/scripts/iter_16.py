import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
np.random.seed(SEED)

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)


def chronological_context(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    st = times[order]
    sd = dates[order]

    positions = np.arange(n, dtype=np.int64)

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = su[1:] != su[:-1]
    user_start = np.maximum.accumulate(
        np.where(new_user, positions, 0)
    )
    user_position_sorted = positions - user_start

    new_day = new_user.copy()
    new_day[1:] |= sd[1:] != sd[:-1]
    day_start = np.maximum.accumulate(
        np.where(new_day, positions, 0)
    )
    day_position_sorted = positions - day_start

    gaps = np.zeros(n, dtype=np.float64)
    gaps[1:] = (st[1:] - st[:-1]) / 60000.0
    gaps[new_user] = 1e6

    new_session = new_user | (gaps > 30.0)
    session_start = np.maximum.accumulate(
        np.where(new_session, positions, 0)
    )
    session_position_sorted = positions - session_start

    new_batch = new_user.copy()
    new_batch[1:] |= st[1:] != st[:-1]
    batch_start = np.maximum.accumulate(
        np.where(new_batch, positions, 0)
    )
    batch_position_sorted = positions - batch_start

    previous_gap_sorted = np.where(
        new_user, 1e6, np.maximum(gaps, 0.0)
    )

    def restore(x, dtype=np.float32):
        result = np.empty(n, dtype=dtype)
        result[order] = x.astype(dtype, copy=False)
        return result

    context = {
        "user_position": restore(user_position_sorted),
        "day_position": restore(day_position_sorted),
        "session_position": restore(session_position_sorted),
        "batch_position": restore(batch_position_sorted),
        "previous_gap_minutes": restore(previous_gap_sorted),
    }

    def causal_repeat_count(values):
        values = np.asarray(values, dtype=np.int64)
        base = int(np.max(values)) + 1
        pair_key = users * np.int64(base) + values
        pair_order = np.lexsort((rows, times, pair_key))
        sorted_key = pair_key[pair_order]

        new_pair = np.empty(n, dtype=bool)
        new_pair[0] = True
        new_pair[1:] = sorted_key[1:] != sorted_key[:-1]

        pair_positions = np.arange(n, dtype=np.int64)
        pair_start = np.maximum.accumulate(
            np.where(new_pair, pair_positions, 0)
        )
        prior_count = pair_positions - pair_start

        result = np.empty(n, dtype=np.float32)
        result[pair_order] = prior_count.astype(np.float32)
        return result

    context["video_repeats"] = causal_repeat_count(split.X["video_id"])
    context["author_repeats"] = causal_repeat_count(split.X["author_id"])
    context["tag_repeats"] = causal_repeat_count(split.X["tag"])

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    context["hour"] = hour
    context["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    context["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)

    context["is_repeat_video"] = (
        context["video_repeats"] > 0
    ).astype(np.float32)
    context["is_repeat_author"] = (
        context["author_repeats"] > 0
    ).astype(np.float32)
    context["is_repeat_tag"] = (
        context["tag_repeats"] > 0
    ).astype(np.float32)

    return context


tr_ctx = chronological_context(train)
va_ctx = chronological_context(valid)
te_ctx = chronological_context(test)

train_dates = np.asarray(train.date, dtype=np.int32)
last_train_day = int(np.max(train_dates)) % 100
ages = last_train_day - (train_dates % 100)
recency_weight = np.exp(
    -np.log(2.0) * ages.astype(np.float32) / 5.0
).astype(np.float32)
recency_weight /= max(float(np.mean(recency_weight)), 1e-6)

global_rate = float(
    np.sum(recency_weight * y_train) / np.sum(recency_weight)
)
global_rate = float(np.clip(global_rate, 1e-5, 1.0 - 1e-5))
global_logit = float(np.log(global_rate / (1.0 - global_rate)))

# ------------------------------------------------------------------
# Family 1: additive empirical presentation-hazard model.
# ------------------------------------------------------------------

GAM_CONTEXT = [
    "user_position",
    "day_position",
    "session_position",
    "batch_position",
    "previous_gap_minutes",
    "video_repeats",
    "author_repeats",
    "tag_repeats",
    "hour",
]


def discretize_context(name, x):
    x = np.asarray(x, dtype=np.float32)
    if name == "previous_gap_minutes":
        return np.minimum(
            np.floor(np.log2(1.0 + np.maximum(x, 0.0))), 15
        ).astype(np.int64)
    if name == "hour":
        return np.minimum(np.maximum(x.astype(np.int64), 0), 31)
    if name == "batch_position":
        return np.minimum(x.astype(np.int64), 15)
    if "repeats" in name:
        return np.minimum(x.astype(np.int64), 12)
    return np.minimum(x.astype(np.int64), 40)


hazard_tables = {}
for name in GAM_CONTEXT:
    ids = discretize_context(name, tr_ctx[name])
    size = int(np.max(ids)) + 1
    count = np.bincount(
        ids, weights=recency_weight, minlength=size
    ).astype(np.float64)
    positive = np.bincount(
        ids, weights=recency_weight * y_train, minlength=size
    ).astype(np.float64)

    smoothing = 250.0 if name == "hour" else 120.0
    rate = (
        positive + smoothing * global_rate
    ) / np.maximum(count + smoothing, 1e-12)
    rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
    residual = np.log(rate / (1.0 - rate)) - global_logit
    reliability = count / (count + smoothing)
    hazard_tables[name] = (
        residual * np.sqrt(reliability)
    ).astype(np.float32)


def predict_hazard(context):
    result = np.full(
        len(context["hour"]), global_logit, dtype=np.float32
    )
    for name in GAM_CONTEXT:
        ids = discretize_context(name, context[name])
        table = hazard_tables[name]
        ids = np.minimum(ids, len(table) - 1)
        result += table[ids] / np.sqrt(float(len(GAM_CONTEXT)))
    return result


hazard_valid = predict_hazard(va_ctx)
hazard_test = predict_hazard(te_ctx)

# ------------------------------------------------------------------
# Family 2: globally fitted linear fatigue model.
# ------------------------------------------------------------------

LINEAR_CONTEXT = [
    "user_position",
    "day_position",
    "session_position",
    "batch_position",
    "previous_gap_minutes",
    "video_repeats",
    "author_repeats",
    "tag_repeats",
    "hour_sin",
    "hour_cos",
    "is_repeat_video",
    "is_repeat_author",
    "is_repeat_tag",
]


def raw_linear_matrix(context):
    columns = []
    for name in LINEAR_CONTEXT:
        x = np.asarray(context[name], dtype=np.float32)
        if name in (
            "user_position",
            "day_position",
            "session_position",
            "batch_position",
            "previous_gap_minutes",
            "video_repeats",
            "author_repeats",
            "tag_repeats",
        ):
            x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        columns.append(x)
    return np.column_stack(columns).astype(np.float32)


Ltr = raw_linear_matrix(tr_ctx)
Lva = raw_linear_matrix(va_ctx)
Lte = raw_linear_matrix(te_ctx)

linear_center = np.mean(Ltr, axis=0, dtype=np.float64).astype(np.float32)
linear_scale = np.std(Ltr, axis=0, dtype=np.float64).astype(np.float32)
linear_scale = np.maximum(linear_scale, 1e-3)

Ltr = ((Ltr - linear_center) / linear_scale).astype(np.float32)
Lva = ((Lva - linear_center) / linear_scale).astype(np.float32)
Lte = ((Lte - linear_center) / linear_scale).astype(np.float32)

Ltr = np.column_stack([
    np.ones(len(Ltr), dtype=np.float32), Ltr
]).astype(np.float32)
Lva = np.column_stack([
    np.ones(len(Lva), dtype=np.float32), Lva
]).astype(np.float32)
Lte = np.column_stack([
    np.ones(len(Lte), dtype=np.float32), Lte
]).astype(np.float32)

coef = np.zeros(Ltr.shape[1], dtype=np.float64)
coef[0] = global_logit
weight_sum = float(np.sum(recency_weight))

for _ in range(8):
    logits = np.clip(Ltr @ coef, -20.0, 20.0)
    probability = 1.0 / (1.0 + np.exp(-logits))
    gradient = (
        Ltr.T @ (recency_weight * (probability - y_train))
    ) / weight_sum

    curvature_weight = (
        recency_weight * probability * (1.0 - probability)
    ).astype(np.float64)
    hessian = (
        Ltr.T @ (Ltr * curvature_weight[:, None])
    ) / weight_sum

    regularization = np.full(len(coef), 2e-3, dtype=np.float64)
    regularization[0] = 1e-7
    gradient += regularization * coef
    hessian.flat[::len(coef) + 1] += regularization + 1e-6

    step = np.linalg.solve(hessian, gradient)
    coef -= step
    if float(np.max(np.abs(step))) < 1e-5:
        break

linear_valid = (Lva @ coef).astype(np.float32)
linear_test = (Lte @ coef).astype(np.float32)

del Ltr, Lva, Lte
gc.collect()

# ------------------------------------------------------------------
# Family 3: nonlinear content-by-presentation interaction boosting.
# ------------------------------------------------------------------

CAT_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "music_type",
    "upload_type",
    "video_type",
    "user_active_degree",
    "register_days_bucket",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

BOOST_CONTEXT = [
    "user_position",
    "day_position",
    "session_position",
    "batch_position",
    "previous_gap_minutes",
    "video_repeats",
    "author_repeats",
    "tag_repeats",
    "hour_sin",
    "hour_cos",
    "is_repeat_video",
    "is_repeat_author",
    "is_repeat_tag",
]


def selected_histories(split_name):
    columns = []
    for key in ("video_id", "author_id"):
        histories = historical_features(split_name, key=key)
        selected = []
        for name in sorted(histories):
            if (
                "train_count_log1p" in name
                or "long_view_rate" in name
                or "is_like_rate" in name
                or "is_follow_rate" in name
            ):
                selected.append(name)
        for name in selected:
            x = np.asarray(histories[name], dtype=np.float32)
            columns.append(
                np.where(np.isfinite(x), x, 0.0).astype(np.float32)
            )
    return np.column_stack(columns).astype(np.float32)


tr_hist = selected_histories("train")
va_hist = selected_histories("valid")
te_hist = selected_histories("test")


def boost_matrix(split, context, histories):
    cats = np.column_stack([
        np.asarray(split.X[name], dtype=np.float32)
        for name in CAT_FIELDS
    ]).astype(np.float32)

    ctx_columns = []
    for name in BOOST_CONTEXT:
        x = np.asarray(context[name], dtype=np.float32)
        if name in (
            "user_position",
            "day_position",
            "session_position",
            "batch_position",
            "previous_gap_minutes",
            "video_repeats",
            "author_repeats",
            "tag_repeats",
        ):
            x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        ctx_columns.append(x)
    ctx = np.column_stack(ctx_columns).astype(np.float32)

    return np.column_stack([cats, ctx, histories]).astype(np.float32)


Xtr = boost_matrix(train, tr_ctx, tr_hist)
Xva = boost_matrix(valid, va_ctx, va_hist)
Xte = boost_matrix(test, te_ctx, te_hist)

dataset = lgb.Dataset(
    Xtr,
    label=y_train,
    weight=recency_weight,
    categorical_feature=list(range(len(CAT_FIELDS))),
    free_raw_data=False,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 55,
    "max_depth": 9,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.3,
    "lambda_l2": 10.0,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 40.0,
    "cat_l2": 15.0,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": SEED,
    "bagging_seed": SEED + 1,
    "feature_fraction_seed": SEED + 2,
    "verbose": -1,
}

model = lgb.train(
    params,
    dataset,
    num_boost_round=210,
)

boost_valid = model.predict(Xva).astype(np.float32)
boost_test = model.predict(Xte).astype(np.float32)

del model, dataset, Xtr, Xva, Xte, tr_hist, va_hist, te_hist
gc.collect()

# ------------------------------------------------------------------
# Compare standalone families and incumbent blends.
# ------------------------------------------------------------------

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float32,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float32,
)


def standardize_pair(valid_scores, test_scores):
    center = float(np.mean(valid_scores))
    scale = max(float(np.std(valid_scores)), 1e-6)
    return (
        ((valid_scores - center) / scale).astype(np.float32),
        ((test_scores - center) / scale).astype(np.float32),
    )


families = {
    "presentation_hazard_gam": (hazard_valid, hazard_test),
    "linear_fatigue_logistic": (linear_valid, linear_test),
    "context_content_gbdt": (boost_valid, boost_test),
}

inc_valid_z, inc_test_z = standardize_pair(inc_valid, inc_test)
inc_metric = evaluate(valid.user_id, y_valid, inc_valid)

best_name = "incumbent"
best_metric = inc_metric
best_valid = inc_valid.copy()
best_test = inc_test.copy()
best_raw_valid = boost_valid.copy()

candidate_log = {
    "incumbent": float(inc_metric["primary"])
}

blend_alphas = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70]

for family_name, (raw_valid, raw_test) in families.items():
    raw_metric = evaluate(valid.user_id, y_valid, raw_valid)
    candidate_log[family_name] = float(raw_metric["primary"])

    if float(raw_metric["primary"]) > float(best_metric["primary"]):
        best_name = family_name
        best_metric = raw_metric
        best_valid = raw_valid.copy()
        best_test = raw_test.copy()
        best_raw_valid = raw_valid.copy()

    raw_valid_z, raw_test_z = standardize_pair(raw_valid, raw_test)

    best_family_blend = -np.inf
    for alpha in blend_alphas:
        blended_valid = (
            (1.0 - alpha) * inc_valid_z + alpha * raw_valid_z
        ).astype(np.float32)
        metric = evaluate(valid.user_id, y_valid, blended_valid)
        primary = float(metric["primary"])
        best_family_blend = max(best_family_blend, primary)

        if primary > float(best_metric["primary"]):
            blended_test = (
                (1.0 - alpha) * inc_test_z + alpha * raw_test_z
            ).astype(np.float32)
            best_name = "{}_blend_{:.2f}".format(family_name, alpha)
            best_metric = metric
            best_valid = blended_valid.copy()
            best_test = blended_test.copy()
            best_raw_valid = raw_valid.copy()

    candidate_log[family_name + "_best_blend"] = best_family_blend

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS selected={} improvement_over_incumbent={:.6f}".format(
        best_name,
        float(best_metric["primary"]) - float(inc_metric["primary"]),
    )
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if "blend_" in best_name or best_name == "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metric["primary"]),
            "gauc": float(best_metric["gauc"]),
            "ndcg@5": float(best_metric["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)