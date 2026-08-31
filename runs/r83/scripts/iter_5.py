import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 2026
THREADS = min(8, os.cpu_count() or 1)
MAX_QUERY = 200

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type", "hour",
    "onehot_feat3", "onehot_feat7", "onehot_feat8",
    "video_type", "user_active_degree", "register_days_bucket",
]
NUM_FIELDS = [
    "duration_ms", "user_follow_user_num", "user_fans_user_num",
    "user_friend_user_num", "user_register_days",
]
AGG_FIELDS = [
    ("video_id", 30.0),
    ("author_id", 50.0),
    ("tag", 100.0),
]

np.random.seed(SEED)


def date_ord(date):
    date = np.asarray(date, dtype=np.int64)
    unique = np.unique(date)
    ords = np.array(
        [np.datetime64(str(int(x)), "D").astype(np.int64) for x in unique],
        dtype=np.int64,
    )
    return ords[np.searchsorted(unique, date)]


def safe_logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


def rank_within_user(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((scores, users))
    sorted_users = users[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]

    group_starts = np.maximum.accumulate(
        np.where(new_group, np.arange(n, dtype=np.int64), 0)
    )
    _, counts = np.unique(sorted_users, return_counts=True)
    row_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - group_starts
    denom = np.maximum(row_counts - 1, 1)

    ranked = positions / denom
    ranked[row_counts == 1] = 0.5

    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def make_groups(users, max_query=MAX_QUERY):
    users = np.asarray(users)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    _, counts = np.unique(sorted_users, return_counts=True)

    groups = []
    for c in counts:
        c = int(c)
        while c > max_query:
            groups.append(max_query)
            c -= max_query
        if c:
            groups.append(c)
    return order, np.asarray(groups, dtype=np.int32)


def aggregate_features(fit, fit_y, ev=None):
    """
    Fit rows receive leave-one-out aggregate features.
    Evaluation rows receive aggregates from all fit rows.
    """
    fit_y = np.asarray(fit_y, dtype=np.float64)
    prior = float(fit_y.mean())

    fit_day = date_ord(fit.date)
    max_day = int(fit_day.max())
    fit_age = np.maximum(max_day - fit_day, 0)
    recency_w = np.exp2(-fit_age.astype(np.float64) / 4.0)

    fit_parts = []
    ev_parts = []
    fit_rate_columns = []
    ev_rate_columns = []

    for field, smoothing in AGG_FIELDS:
        ids = np.asarray(fit.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])

        counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
        positives = np.bincount(
            ids, weights=fit_y, minlength=cardinality
        ).astype(np.float64)
        wcounts = np.bincount(
            ids, weights=recency_w, minlength=cardinality
        ).astype(np.float64)
        wpositives = np.bincount(
            ids, weights=recency_w * fit_y, minlength=cardinality
        ).astype(np.float64)

        loo_count = np.maximum(counts[ids] - 1.0, 0.0)
        loo_rate = (
            positives[ids] - fit_y + smoothing * prior
        ) / (loo_count + smoothing)

        self_w = recency_w
        loo_wcount = np.maximum(wcounts[ids] - self_w, 0.0)
        loo_recent = (
            wpositives[ids] - self_w * fit_y + smoothing * prior
        ) / (loo_wcount + smoothing)

        fit_parts.extend([
            np.log1p(loo_count).astype(np.float32),
            loo_rate.astype(np.float32),
            loo_recent.astype(np.float32),
        ])
        fit_rate_columns.append(
            0.65 * safe_logit(loo_rate) + 0.35 * safe_logit(loo_recent)
        )

        if ev is not None:
            ev_ids = np.asarray(ev.X[field], dtype=np.int64)
            ev_count = counts[ev_ids]
            ev_rate = (
                positives[ev_ids] + smoothing * prior
            ) / (ev_count + smoothing)
            ev_recent = (
                wpositives[ev_ids] + smoothing * prior
            ) / (wcounts[ev_ids] + smoothing)

            ev_parts.extend([
                np.log1p(ev_count).astype(np.float32),
                ev_rate.astype(np.float32),
                ev_recent.astype(np.float32),
            ])
            ev_rate_columns.append(
                0.65 * safe_logit(ev_rate) + 0.35 * safe_logit(ev_recent)
            )

    fit_eb = (
        0.52 * fit_rate_columns[0]
        + 0.30 * fit_rate_columns[1]
        + 0.18 * fit_rate_columns[2]
    ).astype(np.float64)

    if ev is None:
        return fit_parts, fit_eb

    ev_eb = (
        0.52 * ev_rate_columns[0]
        + 0.30 * ev_rate_columns[1]
        + 0.18 * ev_rate_columns[2]
    ).astype(np.float64)
    return fit_parts, ev_parts, fit_eb, ev_eb


def build_matrices(fit, fit_y, ev=None):
    fit_cols = []
    ev_cols = []

    for field in CAT_FIELDS:
        fit_cols.append(np.asarray(fit.X[field], dtype=np.float32))
        if ev is not None:
            ev_cols.append(np.asarray(ev.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        a = np.asarray(fit.num[field], dtype=np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        fit_cols.append(np.log1p(np.maximum(a, 0.0)).astype(np.float32))

        if ev is not None:
            b = np.asarray(ev.num[field], dtype=np.float64)
            b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
            ev_cols.append(np.log1p(np.maximum(b, 0.0)).astype(np.float32))

    fit_ord = date_ord(fit.date)
    fit_max = int(fit_ord.max())
    fit_cols.append((fit_ord - fit_max).astype(np.float32))

    if ev is not None:
        ev_ord = date_ord(ev.date)
        # Absolute ordinal difference remains correct across month boundaries.
        offset = fit_max - int(fit_ord.max())
        del offset
        fit_date_unique = np.unique(np.asarray(fit.date, dtype=np.int64))
        fit_last_ord = max(
            np.datetime64(str(int(x)), "D").astype(np.int64)
            for x in fit_date_unique
        )
        ev_abs = np.array(
            [
                np.datetime64(str(int(x)), "D").astype(np.int64)
                for x in np.asarray(ev.date, dtype=np.int64)
            ],
            dtype=np.int64,
        )
        ev_cols.append((ev_abs - fit_last_ord).astype(np.float32))

    if ev is None:
        agg_fit, fit_eb = aggregate_features(fit, fit_y, None)
        fit_cols.extend(agg_fit)
        X_fit = np.ascontiguousarray(np.column_stack(fit_cols), dtype=np.float32)
        return X_fit, fit_eb

    agg_fit, agg_ev, fit_eb, ev_eb = aggregate_features(fit, fit_y, ev)
    fit_cols.extend(agg_fit)
    ev_cols.extend(agg_ev)

    X_fit = np.ascontiguousarray(np.column_stack(fit_cols), dtype=np.float32)
    X_ev = np.ascontiguousarray(np.column_stack(ev_cols), dtype=np.float32)
    return X_fit, X_ev, fit_eb, ev_eb


def train_binary(Xtr, ytr, Xva, yva):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.3,
        "lambda_l2": 2.0,
        "max_bin": 63,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "verbosity": -1,
        "verbose": -1,
        "seed": SEED,
        "num_threads": THREADS,
        "force_col_wise": True,
    }
    dtr = lgb.Dataset(
        Xtr, label=ytr, categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False
    )
    dva = lgb.Dataset(
        Xva, label=yva, categorical_feature=list(range(len(CAT_FIELDS))),
        reference=dtr, free_raw_data=False
    )
    model = lgb.train(
        params,
        dtr,
        num_boost_round=180,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    pred = model.predict(
        Xva, num_iteration=model.best_iteration, raw_score=True
    )
    return model, params, int(model.best_iteration), np.asarray(pred, dtype=np.float64)


def train_ranker(Xtr, ytr, utr, Xva, yva, uva):
    tr_order, tr_groups = make_groups(utr)
    va_order, va_groups = make_groups(uva)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "label_gain": [0, 1],
        "learning_rate": 0.045,
        "num_leaves": 63,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_bin": 63,
        "lambdarank_truncation_level": 10,
        "lambdarank_norm": True,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "verbosity": -1,
        "verbose": -1,
        "seed": SEED + 11,
        "num_threads": THREADS,
        "force_col_wise": True,
    }

    dtr = lgb.Dataset(
        Xtr[tr_order],
        label=np.asarray(ytr)[tr_order],
        group=tr_groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    dva = lgb.Dataset(
        Xva[va_order],
        label=np.asarray(yva)[va_order],
        group=va_groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        reference=dtr,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        dtr,
        num_boost_round=170,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    pred = model.predict(Xva, num_iteration=model.best_iteration)
    return model, params, int(model.best_iteration), np.asarray(pred, dtype=np.float64)


def refit_lgb(params, rounds, X, y, users, family):
    if family == "rank":
        order, groups = make_groups(users)
        ds = lgb.Dataset(
            X[order],
            label=np.asarray(y)[order],
            group=groups,
            categorical_feature=list(range(len(CAT_FIELDS))),
            free_raw_data=True,
        )
    else:
        ds = lgb.Dataset(
            X,
            label=y,
            categorical_feature=list(range(len(CAT_FIELDS))),
            free_raw_data=True,
        )
    return lgb.train(params, ds, num_boost_round=max(1, int(rounds)))


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
u_train = np.asarray(train.user_id)
u_valid = np.asarray(valid.user_id)

X_train, X_valid, eb_train, eb_valid = build_matrices(
    train, y_train, valid
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)

predictions = {"empirical_bayes": eb_valid}
models = {}
errors = []

try:
    bm, bp, bi, bpred = train_binary(X_train, y_train, X_valid, y_valid)
    predictions["binary_gbdt"] = bpred
    models["binary"] = (bm, bp, bi)
except Exception as exc:
    errors.append("binary=" + repr(exc)[:300])

try:
    rm, rp, ri, rpred = train_ranker(
        X_train, y_train, u_train, X_valid, y_valid, u_valid
    )
    predictions["rank"] = rpred
    models["rank"] = (rm, rp, ri)
except Exception as exc:
    errors.append("rank=" + repr(exc)[:300])

candidate_scores = {}
candidate_specs = {}
best_primary = -np.inf
best_spec = ("incumbent", 0.0)
best_valid_scores = inc_valid.copy()

inc_rank_valid = rank_within_user(u_valid, inc_valid)
inc_metrics = evaluate(u_valid, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])

if float(inc_metrics["primary"]) > best_primary:
    best_primary = float(inc_metrics["primary"])

alphas = [0.25, 0.50, 0.75, 1.0]
for name, raw_pred in predictions.items():
    raw_metrics = evaluate(u_valid, y_valid, raw_pred)
    candidate_scores[name + "_raw"] = float(raw_metrics["primary"])

    new_rank = rank_within_user(u_valid, raw_pred)
    for alpha in alphas:
        blend = (1.0 - alpha) * inc_rank_valid + alpha * new_rank
        m = evaluate(u_valid, y_valid, blend)
        cname = "%s_blend_%.2f" % (name, alpha)
        candidate_scores[cname] = float(m["primary"])
        candidate_specs[cname] = (name, alpha)
        if float(m["primary"]) > best_primary:
            best_primary = float(m["primary"])
            best_spec = (name, alpha)
            best_valid_scores = blend.copy()

if errors:
    print("FINDINGS " + json.dumps({"training_errors": errors}))

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

chosen_name, chosen_alpha = best_spec
metrics = evaluate(u_valid, y_valid, best_valid_scores)

# Release train-only models and matrices before constructing the combined refit.
for value in models.values():
    del value
models.clear()
gc.collect()

test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)

if chosen_name == "incumbent":
    test_scores = inc_test
else:
    combined_y = np.concatenate(
        [y_train, y_valid.astype(np.float32)], axis=0
    )

    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {
        field: np.concatenate(
            [np.asarray(train.X[field]), np.asarray(valid.X[field])]
        )
        for field in set(CAT_FIELDS + [x[0] for x in AGG_FIELDS])
    }
    combined.num = {
        field: np.concatenate(
            [np.asarray(train.num[field]), np.asarray(valid.num[field])]
        )
        for field in NUM_FIELDS
    }
    combined.date = np.concatenate(
        [np.asarray(train.date), np.asarray(valid.date)]
    )
    combined.user_id = np.concatenate([u_train, u_valid])

    del X_train, X_valid
    gc.collect()

    X_combined, X_test, eb_combined, eb_test = build_matrices(
        combined, combined_y, test
    )

    if chosen_name == "empirical_bayes":
        new_test_raw = eb_test
    elif chosen_name == "binary_gbdt":
        # Recover the selected train-only iteration count with a short repeat only
        # if the model object was released.
        _, binary_params, binary_rounds, _ = train_binary(
            np.ascontiguousarray(X_combined[:len(y_train)]),
            y_train,
            np.ascontiguousarray(X_combined[len(y_train):]),
            y_valid,
        )
        final_model = refit_lgb(
            binary_params, binary_rounds, X_combined, combined_y,
            combined.user_id, "binary"
        )
        new_test_raw = final_model.predict(
            X_test, num_iteration=binary_rounds, raw_score=True
        )
    else:
        # Select the same recipe's iteration count using train -> validation,
        # then refit that fixed recipe on train + validation.
        _, rank_params, rank_rounds, _ = train_ranker(
            np.ascontiguousarray(X_combined[:len(y_train)]),
            y_train,
            u_train,
            np.ascontiguousarray(X_combined[len(y_train):]),
            y_valid,
            u_valid,
        )
        final_model = refit_lgb(
            rank_params, rank_rounds, X_combined, combined_y,
            combined.user_id, "rank"
        )
        new_test_raw = final_model.predict(
            X_test, num_iteration=rank_rounds
        )

    test_users = np.asarray(test.user_id)
    new_test_rank = rank_within_user(test_users, new_test_raw)
    inc_test_rank = rank_within_user(test_users, inc_test)
    test_scores = (
        (1.0 - chosen_alpha) * inc_test_rank
        + chosen_alpha * new_test_rank
    )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "FINDINGS " + json.dumps({
        "selected_family": chosen_name,
        "selected_new_model_weight": float(chosen_alpha),
        "feature_count": int(len(CAT_FIELDS) + len(NUM_FIELDS) + 1 + 3 * len(AGG_FIELDS)),
    })
)

wall = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(wall),
    })
)