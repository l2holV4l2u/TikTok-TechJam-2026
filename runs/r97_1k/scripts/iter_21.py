import os
import gc
import json
import time
import warnings
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

np.random.seed(314159)

MAX_SELECTED_FIELDS = 12
HALF_LIFE_DAYS = 4.0
NUM_BOOST_ROUND = 520


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_pos = (
        np.arange(n, dtype=np.float64) - start_pos.astype(np.float64)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    sizes = np.diff(np.r_[-1, np.flatnonzero(ends)]).astype(np.float64)
    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = (local_pos / denom).astype(np.float32)
    return result


def safe_ids(split, name):
    card = int(FEATURE_CARDINALITIES[name])
    x = np.asarray(split.X[name], dtype=np.int64)
    return np.where((x >= 0) & (x < card), x, 0).astype(np.int32)


def js_divergence_from_counts(a, b):
    a = np.asarray(a, dtype=np.float64) + 0.5
    b = np.asarray(b, dtype=np.float64) + 0.5
    a /= a.sum()
    b /= b.sum()
    m = 0.5 * (a + b)
    return float(
        0.5 * np.sum(a * np.log(a / m))
        + 0.5 * np.sum(b * np.log(b / m))
    )


def association_score(ids, labels, card):
    count = np.bincount(ids, minlength=card).astype(np.float64)
    pos = np.bincount(
        ids, weights=labels.astype(np.float64), minlength=card
    )
    global_rate = float(labels.mean())
    rate = (pos + 20.0 * global_rate) / (count + 20.0)
    between = np.sum(
        count * np.square(rate - global_rate)
    ) / max(float(count.sum()), 1.0)
    return float(between)


def choose_stationary_fields(train, labels):
    date = np.asarray(train.date, dtype=np.int32)
    unique_dates = np.unique(date)
    midpoint = unique_dates[len(unique_dates) // 2]
    early = date < midpoint
    late = ~early

    excluded = {
        "video_id",
        "author_id",
        "user_id",
        "is_lowactive_period",
    }
    rows = []

    for name in train.X:
        card = int(FEATURE_CARDINALITIES[name])
        if name in excluded or card > 2500 or card <= 1:
            continue

        ids = safe_ids(train, name)
        early_counts = np.bincount(
            ids[early], minlength=card
        ).astype(np.float64)
        late_counts = np.bincount(
            ids[late], minlength=card
        ).astype(np.float64)

        js = js_divergence_from_counts(early_counts, late_counts)
        assoc = association_score(ids, labels, card)

        # Label association supplies predictive relevance, while the
        # denominator suppresses fields whose marginal distribution is
        # changing rapidly even within the training period.
        utility = assoc / (1.0 + 80.0 * js)
        nonzero = int(np.count_nonzero(
            np.bincount(ids, minlength=card)
        ))
        rows.append((utility, assoc, js, nonzero, name))

    rows.sort(reverse=True)

    chosen = [r[4] for r in rows[:MAX_SELECTED_FIELDS]]
    mandatory = [
        "tag",
        "tab",
        "duration_bucket",
        "upload_type",
        "onehot_feat3",
        "onehot_feat8",
    ]
    for name in mandatory:
        if name in train.X and name not in chosen:
            if len(chosen) >= MAX_SELECTED_FIELDS:
                chosen[-1] = name
            else:
                chosen.append(name)

    # Preserve order while removing any replacement duplicates.
    chosen = list(dict.fromkeys(chosen))

    for utility, assoc, js, nonzero, name in rows[:20]:
        print(
            "FINDINGS stationarity field=%s utility=%.8g "
            "association=%.8g js=%.8g observed_ids=%d selected=%d"
            % (
                name,
                utility,
                assoc,
                js,
                nonzero,
                int(name in chosen),
            ),
            flush=True,
        )

    print(
        "FINDINGS selected_stationary_fields=" + ",".join(chosen),
        flush=True,
    )
    return chosen


def fit_field_statistics(train, labels, sample_weight, fields):
    statistics = {}
    global_weight = float(sample_weight.sum())
    global_pos = float(np.dot(sample_weight, labels))
    global_rate = global_pos / max(global_weight, 1e-12)

    for name in fields:
        ids = safe_ids(train, name)
        card = int(FEATURE_CARDINALITIES[name])

        total = np.bincount(
            ids, weights=sample_weight, minlength=card
        ).astype(np.float64)
        positive = np.bincount(
            ids,
            weights=sample_weight * labels,
            minlength=card,
        ).astype(np.float64)
        negative = total - positive

        # Empirical-Bayes rate used by the discriminative tree.
        strength = 30.0
        smoothed_rate = (
            positive + strength * global_rate
        ) / (total + strength)

        # Class-conditional likelihood ratio used by the generative model.
        alpha = 1.0
        p_pos = (positive + alpha) / (
            positive.sum() + alpha * card
        )
        p_neg = (negative + alpha) / (
            negative.sum() + alpha * card
        )
        log_ratio = np.log(p_pos) - np.log(p_neg)
        log_ratio = np.clip(log_ratio, -5.0, 5.0)

        statistics[name] = {
            "total": total,
            "positive": positive,
            "rate": smoothed_rate.astype(np.float32),
            "log_ratio": log_ratio.astype(np.float32),
        }

    statistics["_global_rate"] = global_rate
    return statistics


def history_matrix(split_name):
    columns = []
    names = []
    for key in ("video_id", "author_id"):
        histories = historical_features(split_name, key=key)
        preferred = [
            k for k in histories
            if (
                "long_view_rate" in k
                or "count_log1p" in k
                or "is_click_rate" in k
                or "play_time_ms_logmean" in k
                or "comment_stay_time_logmean" in k
            )
        ]
        if not preferred:
            preferred = sorted(histories)[:6]

        for name in sorted(preferred):
            x = np.asarray(histories[name], dtype=np.float32)
            finite = np.isfinite(x)
            if not finite.all():
                x = np.where(finite, x, 0.0).astype(np.float32)
            columns.append(x)
            names.append(name)

    if not columns:
        return np.empty((len(load(split_name)), 0), dtype=np.float32), []
    return np.column_stack(columns).astype(np.float32), names


def build_tree_matrix(
    split,
    split_name,
    fields,
    statistics,
    train_labels=None,
    train_weight=None,
):
    n = len(split)
    columns = []
    names = []

    for name in fields:
        ids = safe_ids(split, name)
        stat = statistics[name]

        if train_labels is not None:
            # Weighted leave-one-out target encoding for training rows.
            own_w = train_weight.astype(np.float64)
            own_pos = own_w * train_labels.astype(np.float64)
            total = stat["total"][ids] - own_w
            positive = stat["positive"][ids] - own_pos
            strength = 30.0
            rate = (
                positive
                + strength * float(statistics["_global_rate"])
            ) / (total + strength)
            rate = rate.astype(np.float32)
        else:
            rate = stat["rate"][ids]

        columns.append(rate)
        names.append(name + "_stable_te")

        # Raw low-cardinality state permits nonlinear category combinations
        # without exposing brittle user/video/author identities.
        columns.append(ids.astype(np.float32))
        names.append(name + "_raw")

    hist, hist_names = history_matrix(split_name)
    for j, name in enumerate(hist_names):
        columns.append(hist[:, j])
        names.append(name)

    for name in sorted(split.num):
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.where(np.isfinite(x), x, 0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        columns.append(x.astype(np.float32))
        names.append(name + "_signed_log1p")

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    angle = (2.0 * np.pi / 24.0) * hour
    columns.append(np.sin(angle).astype(np.float32))
    names.append("hour_sin")
    columns.append(np.cos(angle).astype(np.float32))
    names.append("hour_cos")

    matrix = np.column_stack(columns).astype(np.float32)
    return matrix, names


def generative_score(split, fields, statistics):
    prior = float(statistics["_global_rate"])
    prior = np.clip(prior, 1e-5, 1.0 - 1e-5)
    score = np.full(
        len(split),
        np.log(prior / (1.0 - prior)),
        dtype=np.float32,
    )

    # Averaging correlated evidence avoids the extreme overconfidence of
    # ordinary Naive Bayes while retaining its low-variance prediction rule.
    scale = np.float32(1.0 / np.sqrt(max(len(fields), 1)))
    for name in fields:
        ids = safe_ids(split, name)
        score += scale * statistics[name]["log_ratio"][ids]
    return score


inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)
if not (
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent predictions are required")

train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
train_date = np.asarray(train.date, dtype=np.int32)
max_date = int(train_date.max())

# This is applied to the main model, not merely to a side statistic.
age_days = (max_date - train_date).astype(np.float32)
train_weight = np.exp(
    -np.log(2.0) * age_days / HALF_LIFE_DAYS
).astype(np.float32)
train_weight /= np.mean(train_weight)

fields = choose_stationary_fields(train, train_y)
statistics = fit_field_statistics(
    train, train_y, train_weight, fields
)

X_train, feature_names = build_tree_matrix(
    train,
    "train",
    fields,
    statistics,
    train_labels=train_y,
    train_weight=train_weight,
)
X_valid, valid_feature_names = build_tree_matrix(
    valid,
    "valid",
    fields,
    statistics,
)
if feature_names != valid_feature_names:
    raise RuntimeError("Train/validation feature mismatch")

print(
    "FINDINGS tree_matrix rows=%d columns=%d half_life=%.1f"
    % (X_train.shape[0], X_train.shape[1], HALF_LIFE_DAYS),
    flush=True,
)

dtrain = lgb.Dataset(
    X_train,
    label=train_y,
    weight=train_weight,
    feature_name=feature_names,
    free_raw_data=False,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 1200,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 3.0,
    "max_bin": 127,
    "min_gain_to_split": 1e-5,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": 314159,
    "feature_fraction_seed": 314159,
    "bagging_seed": 271828,
    "verbose": -1,
}

booster = lgb.train(
    params,
    dtrain,
    num_boost_round=NUM_BOOST_ROUND,
)

tree_valid = booster.predict(
    X_valid, num_iteration=NUM_BOOST_ROUND
).astype(np.float32)
nb_valid = generative_score(valid, fields, statistics)

valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)
inc_valid = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"), dtype=np.float32
)

inc_rank = within_user_rank(valid_uid, inc_valid)
tree_rank = within_user_rank(valid_uid, tree_valid)
nb_rank = within_user_rank(valid_uid, nb_valid)
hybrid_rank = (
    0.82 * tree_rank + 0.18 * nb_rank
).astype(np.float32)

family_ranks = {
    "stationary_lgb": tree_rank,
    "generative_nb": nb_rank,
    "tree_nb_hybrid": hybrid_rank,
}

candidate_results = {}
control_metrics = evaluate(valid_uid, valid_y, inc_rank)
candidate_results["trusted_incumbent"] = float(
    control_metrics["primary"]
)

best_primary = float(control_metrics["primary"])
best_scores = inc_rank.copy()
best_raw = None
best_family = "trusted_incumbent"
best_alpha = 0.0

blend_alphas = [0.03, 0.06, 0.10, 0.15, 0.25, 0.40, 1.00]

for family_name, family_rank in family_ranks.items():
    standalone = evaluate(valid_uid, valid_y, family_rank)
    candidate_results[family_name + "_standalone"] = float(
        standalone["primary"]
    )
    print(
        "FINDINGS family=%s standalone_primary=%.6f "
        "gauc=%.6f ndcg5=%.6f"
        % (
            family_name,
            float(standalone["primary"]),
            float(standalone["gauc"]),
            float(standalone["ndcg@5"]),
        ),
        flush=True,
    )

    for alpha in blend_alphas:
        blended = (
            (1.0 - alpha) * inc_rank + alpha * family_rank
        ).astype(np.float32)
        metrics = evaluate(valid_uid, valid_y, blended)
        candidate_name = "%s_blend_%.2f" % (family_name, alpha)
        candidate_results[candidate_name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_scores = blended.copy()
            best_raw = family_rank.copy()
            best_family = family_name
            best_alpha = float(alpha)

final_metrics = evaluate(valid_uid, valid_y, best_scores)

print(
    "FINDINGS winner=%s alpha=%.2f control_primary=%.6f "
    "winner_primary=%.6f"
    % (
        best_family,
        best_alpha,
        float(control_metrics["primary"]),
        float(final_metrics["primary"]),
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_results, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_raw is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

del X_train
del X_valid
del dtrain
del tree_valid
del nb_valid
del train
del valid
del train_y
del train_date
del train_weight
del inc_valid
gc.collect()

test = load("test")
inc_test = np.asarray(
    np.load(inc_test_path, mmap_mode="r"), dtype=np.float32
)
inc_test_rank = within_user_rank(test.user_id, inc_test)

if best_family == "trusted_incumbent":
    test_scores = inc_test_rank
else:
    X_test, test_feature_names = build_tree_matrix(
        test,
        "test",
        fields,
        statistics,
    )
    if test_feature_names != feature_names:
        raise RuntimeError("Train/test feature mismatch")

    if best_family in ("stationary_lgb", "tree_nb_hybrid"):
        tree_test = booster.predict(
            X_test, num_iteration=NUM_BOOST_ROUND
        ).astype(np.float32)
        tree_test_rank = within_user_rank(test.user_id, tree_test)

    if best_family in ("generative_nb", "tree_nb_hybrid"):
        nb_test = generative_score(test, fields, statistics)
        nb_test_rank = within_user_rank(test.user_id, nb_test)

    if best_family == "stationary_lgb":
        raw_test_rank = tree_test_rank
    elif best_family == "generative_nb":
        raw_test_rank = nb_test_rank
    elif best_family == "tree_nb_hybrid":
        raw_test_rank = (
            0.82 * tree_test_rank + 0.18 * nb_test_rank
        ).astype(np.float32)
    else:
        raise RuntimeError("Unknown winning family")

    test_scores = (
        (1.0 - best_alpha) * inc_test_rank
        + best_alpha * raw_test_rank
    ).astype(np.float32)

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)