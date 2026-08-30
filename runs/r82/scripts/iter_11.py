import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
EPS = 1e-6
NUM_ROUNDS = 120
BLEND_WEIGHTS = [0.15, 0.30, 0.45, 0.60, 0.75]

BASE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "hour",
    "upload_type",
    "music_type",
    "video_type",
    "is_video_author",
    "is_live_streamer",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

ENTITY_FIELDS = [
    ("video_id", 20.0),
    ("author_id", 30.0),
    ("tag", 80.0),
    ("duration_bucket", 100.0),
]

PAIR_FIELDS = [
    ("user_id", "author_id", 8.0),
    ("user_id", "tag", 12.0),
    ("user_id", "video_id", 5.0),
]


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.float64)
    if half_life is None:
        return np.ones(len(dates), dtype=np.float64)
    age = float(np.max(dates)) - dates
    weights = np.exp2(-age / float(half_life))
    weights /= max(float(weights.mean()), EPS)
    return weights


def entity_statistics(fit_ids, query_ids, y, weights, cardinality, smoothing):
    fit_ids = np.asarray(fit_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    count = np.bincount(
        fit_ids, weights=weights, minlength=int(cardinality)
    ).astype(np.float64)
    positive = np.bincount(
        fit_ids, weights=weights * y, minlength=int(cardinality)
    ).astype(np.float64)

    global_rate = float(np.sum(weights * y) / max(np.sum(weights), EPS))

    fit_count_loo = np.maximum(count[fit_ids] - weights, 0.0)
    fit_positive_loo = positive[fit_ids] - weights * y
    fit_rate = (
        fit_positive_loo + float(smoothing) * global_rate
    ) / (fit_count_loo + float(smoothing))

    clipped_query = np.clip(query_ids, 0, len(count) - 1)
    query_count = count[clipped_query]
    query_rate = (
        positive[clipped_query] + float(smoothing) * global_rate
    ) / (query_count + float(smoothing))

    return (
        fit_rate.astype(np.float32),
        np.log1p(fit_count_loo).astype(np.float32),
        query_rate.astype(np.float32),
        np.log1p(query_count).astype(np.float32),
    )


def pair_statistics(
    fit_left,
    fit_right,
    query_left,
    query_right,
    right_cardinality,
    y,
    weights,
    smoothing,
):
    fit_left = np.asarray(fit_left, dtype=np.int64)
    fit_right = np.asarray(fit_right, dtype=np.int64)
    query_left = np.asarray(query_left, dtype=np.int64)
    query_right = np.asarray(query_right, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    fit_key = fit_left * np.int64(right_cardinality) + fit_right
    query_key = query_left * np.int64(right_cardinality) + query_right

    unique_key, inverse = np.unique(fit_key, return_inverse=True)
    count = np.bincount(inverse, weights=weights).astype(np.float64)
    positive = np.bincount(inverse, weights=weights * y).astype(np.float64)
    global_rate = float(np.sum(weights * y) / max(np.sum(weights), EPS))

    fit_count_loo = np.maximum(count[inverse] - weights, 0.0)
    fit_positive_loo = positive[inverse] - weights * y
    fit_rate = (
        fit_positive_loo + float(smoothing) * global_rate
    ) / (fit_count_loo + float(smoothing))

    positions = np.searchsorted(unique_key, query_key)
    clipped = np.minimum(positions, len(unique_key) - 1)
    found = positions < len(unique_key)
    found &= unique_key[clipped] == query_key

    query_count = np.zeros(len(query_key), dtype=np.float64)
    query_positive = np.zeros(len(query_key), dtype=np.float64)
    query_count[found] = count[clipped[found]]
    query_positive[found] = positive[clipped[found]]
    query_rate = (
        query_positive + float(smoothing) * global_rate
    ) / (query_count + float(smoothing))

    return (
        fit_rate.astype(np.float32),
        np.log1p(fit_count_loo).astype(np.float32),
        query_rate.astype(np.float32),
        np.log1p(query_count).astype(np.float32),
    )


def numeric_transform(values):
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros(len(x), dtype=np.float32)
    out[finite] = np.log1p(np.maximum(x[finite], 0.0)).astype(np.float32)
    return out


def build_matrices(fit, query):
    y = np.asarray(fit.y, dtype=np.float64)
    fit_columns = []
    query_columns = []
    names = []

    for field in BASE_FIELDS:
        fit_columns.append(np.asarray(fit.X[field], dtype=np.float32))
        query_columns.append(np.asarray(query.X[field], dtype=np.float32))
        names.append(field)

    fit_columns.append(numeric_transform(fit.num["duration_ms"]))
    query_columns.append(numeric_transform(query.num["duration_ms"]))
    names.append("duration_ms_log1p")

    uniform_weights = recency_weights(fit.date, None)
    recent_weights = recency_weights(fit.date, 4.0)

    for suffix, weights in [
        ("uniform", uniform_weights),
        ("recent4", recent_weights),
    ]:
        for field, smoothing in ENTITY_FIELDS:
            fr, fc, qr, qc = entity_statistics(
                fit.X[field],
                query.X[field],
                y,
                weights,
                FEATURE_CARDINALITIES[field],
                smoothing,
            )
            fit_columns.extend([fr, fc])
            query_columns.extend([qr, qc])
            names.extend([
                "%s_rate_%s" % (field, suffix),
                "%s_count_%s" % (field, suffix),
            ])

        for left, right, smoothing in PAIR_FIELDS:
            fr, fc, qr, qc = pair_statistics(
                fit.X[left],
                fit.X[right],
                query.X[left],
                query.X[right],
                FEATURE_CARDINALITIES[right],
                y,
                weights,
                smoothing,
            )
            fit_columns.extend([fr, fc])
            query_columns.extend([qr, qc])
            names.extend([
                "%s_%s_rate_%s" % (left, right, suffix),
                "%s_%s_count_%s" % (left, right, suffix),
            ])

    X_fit = np.column_stack(fit_columns).astype(np.float32, copy=False)
    X_query = np.column_stack(query_columns).astype(np.float32, copy=False)
    categorical_indices = list(range(len(BASE_FIELDS)))
    return X_fit, X_query, names, categorical_indices


def user_sort_and_groups(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    if len(sorted_users) == 0:
        return order, np.empty(0, dtype=np.int32)

    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_users)]
    groups = (ends - starts).astype(np.int32)
    return order, groups


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(starts_flag, np.arange(n, dtype=np.int64), 0)
    )
    positions = np.arange(n, dtype=np.float64) - starts

    start_indices = np.flatnonzero(starts_flag)
    end_indices = np.r_[start_indices[1:], n]
    sizes = end_indices - start_indices
    repeated_sizes = np.repeat(sizes, sizes).astype(np.float64)

    ranked = positions / np.maximum(repeated_sizes - 1.0, 1.0)
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def train_model(
    X,
    y,
    user_ids,
    feature_names,
    categorical_indices,
    objective,
    half_life,
    dates,
):
    if objective == "binary":
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.07,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 2.0,
            "max_bin": 127,
            "verbose": -1,
            "seed": 2026,
            "num_threads": min(16, os.cpu_count() or 8),
            "force_col_wise": True,
        }
    elif objective == "lambdarank":
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "lambdarank_truncation_level": 10,
            "learning_rate": 0.07,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 2.0,
            "max_bin": 127,
            "verbose": -1,
            "seed": 2026,
            "num_threads": min(16, os.cpu_count() or 8),
            "force_col_wise": True,
        }
    elif objective == "rank_xendcg":
        params = {
            "objective": "rank_xendcg",
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "learning_rate": 0.07,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 2.0,
            "max_bin": 127,
            "verbose": -1,
            "seed": 2026,
            "num_threads": min(16, os.cpu_count() or 8),
            "force_col_wise": True,
        }
    else:
        raise ValueError("Unknown objective: %s" % objective)

    sample_weight = recency_weights(dates, half_life).astype(np.float32)

    if objective == "binary":
        dataset = lgb.Dataset(
            X,
            label=np.asarray(y, dtype=np.float32),
            weight=sample_weight,
            feature_name=feature_names,
            categorical_feature=categorical_indices,
            free_raw_data=False,
        )
    else:
        order, groups = user_sort_and_groups(user_ids)
        X_sorted = X[order]
        y_sorted = np.asarray(y, dtype=np.float32)[order]
        weights_sorted = sample_weight[order]
        dataset = lgb.Dataset(
            X_sorted,
            label=y_sorted,
            weight=weights_sorted,
            group=groups,
            feature_name=feature_names,
            categorical_feature=categorical_indices,
            free_raw_data=False,
        )

    model = lgb.train(
        params,
        dataset,
        num_boost_round=NUM_ROUNDS,
    )
    return model


train = load("train")
valid = load("valid")

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

X_train, X_valid, feature_names, categorical_indices = build_matrices(
    train, valid
)

recipes = [
    {
        "name": "pointwise_binary_hl4",
        "objective": "binary",
        "half_life": 4.0,
    },
    {
        "name": "pairwise_lambdarank_uniform",
        "objective": "lambdarank",
        "half_life": None,
    },
    {
        "name": "pairwise_lambdarank_hl4",
        "objective": "lambdarank",
        "half_life": 4.0,
    },
    {
        "name": "listwise_xendcg_hl4",
        "objective": "rank_xendcg",
        "half_life": 4.0,
    },
]

candidate_scores = {"incumbent": float(inc_metrics["primary"])}
candidate_predictions = {"incumbent": inc_valid}
candidate_recipes = {
    "incumbent": {
        "base_recipe": None,
        "mode": "incumbent",
        "blend_weight": 0.0,
    }
}

standalone_findings = []

for recipe in recipes:
    model = train_model(
        X_train,
        train.y,
        train.user_id,
        feature_names,
        categorical_indices,
        recipe["objective"],
        recipe["half_life"],
        train.date,
    )
    raw_valid = np.asarray(
        model.predict(X_valid, num_iteration=model.best_iteration),
        dtype=np.float64,
    )
    raw_rank = within_user_rank(valid.user_id, raw_valid)

    raw_metrics = evaluate(valid.user_id, valid.y, raw_valid)
    candidate_scores[recipe["name"]] = float(raw_metrics["primary"])
    candidate_predictions[recipe["name"]] = raw_valid
    candidate_recipes[recipe["name"]] = {
        "base_recipe": recipe,
        "mode": "standalone",
        "blend_weight": 1.0,
    }

    standalone_findings.append(
        "%s primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            recipe["name"],
            float(raw_metrics["primary"]),
            float(raw_metrics["gauc"]),
            float(raw_metrics["ndcg@5"]),
        )
    )

    for alpha in BLEND_WEIGHTS:
        name = "%s_rankblend%.2f" % (recipe["name"], alpha)
        blended = alpha * raw_rank + (1.0 - alpha) * inc_valid_rank
        blend_metrics = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[name] = float(blend_metrics["primary"])
        candidate_predictions[name] = blended
        candidate_recipes[name] = {
            "base_recipe": recipe,
            "mode": "rankblend",
            "blend_weight": float(alpha),
        }

    del model, raw_valid, raw_rank
    gc.collect()

winner = max(candidate_scores, key=candidate_scores.get)
winner_recipe = candidate_recipes[winner]
valid_scores = np.asarray(candidate_predictions[winner], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
for finding in standalone_findings:
    print("FINDINGS " + finding)
print(
    "FINDINGS winner=%s mode=%s delta_incumbent=%+.6f features=%d"
    % (
        winner,
        winner_recipe["mode"],
        float(metrics["primary"] - inc_metrics["primary"]),
        X_train.shape[1],
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

del X_train, X_valid, candidate_predictions
gc.collect()

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winner_recipe["mode"] == "incumbent":
    test_scores = inc_test
else:
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {}
    for field in set(BASE_FIELDS + [x[0] for x in ENTITY_FIELDS]):
        combined.X[field] = np.concatenate([
            np.asarray(train.X[field]),
            np.asarray(valid.X[field]),
        ])
    for left, right, _ in PAIR_FIELDS:
        if left not in combined.X:
            combined.X[left] = np.concatenate([
                np.asarray(train.X[left]),
                np.asarray(valid.X[left]),
            ])
        if right not in combined.X:
            combined.X[right] = np.concatenate([
                np.asarray(train.X[right]),
                np.asarray(valid.X[right]),
            ])

    combined.y = np.concatenate([
        np.asarray(train.y, dtype=np.int8),
        np.asarray(valid.y, dtype=np.int8),
    ])
    combined.user_id = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    combined.date = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])
    combined.num = {
        "duration_ms": np.concatenate([
            np.asarray(train.num["duration_ms"]),
            np.asarray(valid.num["duration_ms"]),
        ])
    }

    X_combined, X_test, final_names, final_categorical = build_matrices(
        combined, test
    )

    base_recipe = winner_recipe["base_recipe"]
    final_model = train_model(
        X_combined,
        combined.y,
        combined.user_id,
        final_names,
        final_categorical,
        base_recipe["objective"],
        base_recipe["half_life"],
        combined.date,
    )
    raw_test = np.asarray(
        final_model.predict(X_test, num_iteration=final_model.best_iteration),
        dtype=np.float64,
    )

    if winner_recipe["mode"] == "standalone":
        test_scores = raw_test
    elif winner_recipe["mode"] == "rankblend":
        alpha = float(winner_recipe["blend_weight"])
        raw_test_rank = within_user_rank(test.user_id, raw_test)
        inc_test_rank = within_user_rank(test.user_id, inc_test)
        test_scores = alpha * raw_test_rank + (1.0 - alpha) * inc_test_rank
    else:
        raise ValueError("Unknown winner mode: %s" % winner_recipe["mode"])

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)