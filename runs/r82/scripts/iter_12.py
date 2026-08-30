import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from sklearn.ensemble import ExtraTreesClassifier
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
EPS = 1e-6

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "hour",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "video_type",
]

NUM_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]

TE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
]

TE_SMOOTHING = {
    "video_id": 18.0,
    "author_id": 28.0,
    "tag": 75.0,
    "duration_bucket": 100.0,
    "upload_type": 100.0,
}

BLEND_WEIGHTS = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70]
HALF_LIFE = 4.0


def logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(x) - np.log1p(-x)


def recency_weights(dates, half_life=HALF_LIFE):
    dates = np.asarray(dates, dtype=np.float64)
    age = float(np.max(dates)) - dates
    weights = np.exp2(-age / float(half_life))
    weights /= max(float(np.mean(weights)), EPS)
    return weights.astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]

    starts = np.maximum.accumulate(
        np.where(starts_mask, np.arange(n, dtype=np.int64), 0)
    )
    positions = np.arange(n, dtype=np.float64) - starts

    start_idx = np.flatnonzero(starts_mask)
    end_idx = np.r_[start_idx[1:], n]
    sizes = end_idx - start_idx
    repeated_sizes = np.repeat(sizes, sizes).astype(np.float64)

    ranked = positions / np.maximum(repeated_sizes - 1.0, 1.0)
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def group_order_and_sizes(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    if len(sorted_users) == 0:
        return order, np.empty(0, dtype=np.int32)

    boundaries = np.r_[
        0,
        np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
        len(sorted_users),
    ]
    groups = np.diff(boundaries).astype(np.int32)
    return order, groups


def concatenate_splits(train, valid):
    class Combined:
        pass

    combined = Combined()
    combined.X = {}
    for field in CAT_FIELDS:
        combined.X[field] = np.concatenate([
            np.asarray(train.X[field], dtype=np.int64),
            np.asarray(valid.X[field], dtype=np.int64),
        ])

    combined.num = {}
    for field in NUM_FIELDS:
        combined.num[field] = np.concatenate([
            np.asarray(train.num[field], dtype=np.float32),
            np.asarray(valid.num[field], dtype=np.float32),
        ])

    combined.y = np.concatenate([
        np.asarray(train.y, dtype=np.int8),
        np.asarray(valid.y, dtype=np.int8),
    ])
    combined.date = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])
    combined.user_id = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    return combined


def build_features(fit, query, fit_weights, fit_is_training=False):
    y = np.asarray(fit.y, dtype=np.float64)
    weights = np.asarray(fit_weights, dtype=np.float64)
    n = len(y)

    columns = []

    for field in CAT_FIELDS:
        if fit_is_training:
            values = np.asarray(fit.X[field], dtype=np.float32)
        else:
            values = np.asarray(query.X[field], dtype=np.float32)
        columns.append(values)

    for field in NUM_FIELDS:
        if fit_is_training:
            values = np.asarray(fit.num[field], dtype=np.float64)
        else:
            values = np.asarray(query.num[field], dtype=np.float64)

        finite = np.isfinite(values)
        clean = np.zeros(values.shape[0], dtype=np.float32)
        clean[finite] = np.log1p(
            np.maximum(values[finite], 0.0)
        ).astype(np.float32)
        missing = (~finite).astype(np.float32)
        columns.append(clean)
        columns.append(missing)

    global_rate = float(np.sum(weights * y) / max(np.sum(weights), EPS))

    for field in TE_FIELDS:
        ids_fit = np.asarray(fit.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])

        counts = np.bincount(
            ids_fit,
            weights=weights,
            minlength=cardinality,
        ).astype(np.float64)
        positives = np.bincount(
            ids_fit,
            weights=weights * y,
            minlength=cardinality,
        ).astype(np.float64)

        smoothing = float(TE_SMOOTHING[field])

        if fit_is_training:
            row_count = counts[ids_fit] - weights
            row_positive = positives[ids_fit] - weights * y
            rates = (
                row_positive + smoothing * global_rate
            ) / np.maximum(row_count + smoothing, EPS)
            count_feature = np.log1p(np.maximum(row_count, 0.0))
        else:
            ids_query = np.asarray(query.X[field], dtype=np.int64)
            safe_ids = np.clip(ids_query, 0, cardinality - 1)
            row_count = counts[safe_ids]
            row_positive = positives[safe_ids]
            rates = (
                row_positive + smoothing * global_rate
            ) / np.maximum(row_count + smoothing, EPS)
            count_feature = np.log1p(np.maximum(row_count, 0.0))

        columns.append(logit(rates).astype(np.float32))
        columns.append(count_feature.astype(np.float32))

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    assert matrix.shape[0] == (n if fit_is_training else len(query.user_id))
    return matrix


def train_binary(X_train, y_train, weights, X_valid, y_valid):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "cat_smooth": 20.0,
        "cat_l2": 12.0,
        "force_col_wise": True,
        "num_threads": -1,
        "seed": 2026,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=list(range(len(CAT_FIELDS))),
        reference=dtrain,
        free_raw_data=True,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=240,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    iteration = int(model.best_iteration or 240)
    predictions = model.predict(X_valid, num_iteration=iteration)
    return predictions, iteration


def train_lambdarank(
    X_train,
    y_train,
    user_train,
    weights,
    X_valid,
    y_valid,
    user_valid,
):
    train_order, train_groups = group_order_and_sizes(user_train)
    valid_order, valid_groups = group_order_and_sizes(user_valid)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "label_gain": [0, 1],
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.045,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.5,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "cat_smooth": 20.0,
        "cat_l2": 12.0,
        "force_col_wise": True,
        "num_threads": -1,
        "seed": 2027,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(
        X_train[train_order],
        label=y_train[train_order],
        weight=weights[train_order],
        group=train_groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        X_valid[valid_order],
        label=y_valid[valid_order],
        group=valid_groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        reference=dtrain,
        free_raw_data=True,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    iteration = int(model.best_iteration or 200)

    pred_sorted = model.predict(
        X_valid[valid_order],
        num_iteration=iteration,
    )
    predictions = np.empty(len(y_valid), dtype=np.float64)
    predictions[valid_order] = pred_sorted
    return predictions, iteration


def train_extra_trees(X_train, y_train, weights, X_valid):
    model = ExtraTreesClassifier(
        n_estimators=48,
        criterion="entropy",
        max_depth=20,
        min_samples_leaf=24,
        max_features=0.72,
        bootstrap=False,
        class_weight=None,
        n_jobs=-1,
        random_state=2028,
    )
    model.fit(X_train, y_train, sample_weight=weights)
    predictions = model.predict_proba(X_valid)[:, 1]
    return predictions


def fit_final_binary(X_fit, y_fit, weights, X_test, rounds):
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "cat_smooth": 20.0,
        "cat_l2": 12.0,
        "force_col_wise": True,
        "num_threads": -1,
        "seed": 2026,
        "verbose": -1,
    }
    dfit = lgb.Dataset(
        X_fit,
        label=y_fit,
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    model = lgb.train(params, dfit, num_boost_round=int(rounds))
    return model.predict(X_test, num_iteration=int(rounds))


def fit_final_lambdarank(
    X_fit,
    y_fit,
    user_fit,
    weights,
    X_test,
    rounds,
):
    order, groups = group_order_and_sizes(user_fit)
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "label_gain": [0, 1],
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.045,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.5,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "cat_smooth": 20.0,
        "cat_l2": 12.0,
        "force_col_wise": True,
        "num_threads": -1,
        "seed": 2027,
        "verbose": -1,
    }
    dfit = lgb.Dataset(
        X_fit[order],
        label=y_fit[order],
        weight=weights[order],
        group=groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    model = lgb.train(params, dfit, num_boost_round=int(rounds))
    return model.predict(X_test, num_iteration=int(rounds))


train = load("train")
valid = load("valid")

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_metrics = evaluate(valid.user_id, valid.y, inc_valid)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_valid_logit = logit(inc_valid)

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_weights = recency_weights(train.date)

X_train = build_features(
    train, train, train_weights, fit_is_training=True
)
X_valid = build_features(
    train, valid, train_weights, fit_is_training=False
)

candidate_scores = {
    "incumbent": float(inc_valid_metrics["primary"])
}
candidate_predictions = {
    "incumbent": inc_valid
}
candidate_recipes = {
    "incumbent": {
        "family": "incumbent",
        "blend_mode": "none",
        "blend_weight": 0.0,
        "rounds": 0,
    }
}


def register_family(name, predictions, rounds, probability_like):
    predictions = np.asarray(predictions, dtype=np.float64)

    standalone_metrics = evaluate(
        valid.user_id, valid.y, predictions
    )
    candidate_scores[name] = float(standalone_metrics["primary"])
    candidate_predictions[name] = predictions
    candidate_recipes[name] = {
        "family": name,
        "blend_mode": "standalone",
        "blend_weight": 1.0,
        "rounds": int(rounds),
    }

    family_rank = within_user_rank(valid.user_id, predictions)
    for weight in BLEND_WEIGHTS:
        rank_name = "%s_rankblend_%.2f" % (name, weight)
        rank_prediction = (
            weight * family_rank
            + (1.0 - weight) * inc_valid_rank
        )
        metrics = evaluate(
            valid.user_id, valid.y, rank_prediction
        )
        candidate_scores[rank_name] = float(metrics["primary"])
        candidate_predictions[rank_name] = rank_prediction
        candidate_recipes[rank_name] = {
            "family": name,
            "blend_mode": "rank",
            "blend_weight": float(weight),
            "rounds": int(rounds),
        }

        if probability_like:
            logit_name = "%s_logitblend_%.2f" % (name, weight)
            logit_prediction = (
                weight * logit(predictions)
                + (1.0 - weight) * inc_valid_logit
            )
            metrics = evaluate(
                valid.user_id, valid.y, logit_prediction
            )
            candidate_scores[logit_name] = float(metrics["primary"])
            candidate_predictions[logit_name] = logit_prediction
            candidate_recipes[logit_name] = {
                "family": name,
                "blend_mode": "logit",
                "blend_weight": float(weight),
                "rounds": int(rounds),
            }


binary_valid, binary_rounds = train_binary(
    X_train,
    y_train,
    train_weights,
    X_valid,
    y_valid,
)
register_family(
    "pointwise_lgbm",
    binary_valid,
    binary_rounds,
    probability_like=True,
)
del binary_valid
gc.collect()

rank_valid, rank_rounds = train_lambdarank(
    X_train,
    y_train,
    train.user_id,
    train_weights,
    X_valid,
    y_valid,
    valid.user_id,
)
register_family(
    "querywise_lambdamart",
    rank_valid,
    rank_rounds,
    probability_like=False,
)
del rank_valid
gc.collect()

extra_valid = train_extra_trees(
    X_train,
    y_train,
    train_weights,
    X_valid,
)
register_family(
    "bagged_extra_trees",
    extra_valid,
    rounds=48,
    probability_like=True,
)
del extra_valid
gc.collect()

winner = max(candidate_scores, key=candidate_scores.get)
recipe = candidate_recipes[winner]
valid_scores = np.asarray(
    candidate_predictions[winner], dtype=np.float64
)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS recency_half_life=%.1f train_weight_min=%.4f "
    "train_weight_max=%.4f feature_count=%d binary_rounds=%d "
    "rank_rounds=%d"
    % (
        HALF_LIFE,
        float(np.min(train_weights)),
        float(np.max(train_weights)),
        int(X_train.shape[1]),
        int(binary_rounds),
        int(rank_rounds),
    )
)
print(
    "FINDINGS winner=%s family=%s mode=%s weight=%.2f "
    "delta_incumbent=%+.6f"
    % (
        winner,
        recipe["family"],
        recipe["blend_mode"],
        float(recipe["blend_weight"]),
        float(metrics["primary"] - inc_valid_metrics["primary"]),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if recipe["family"] == "incumbent":
    test_scores = inc_test
else:
    combined = concatenate_splits(train, valid)
    combined_weights = recency_weights(combined.date)

    del X_train, X_valid
    gc.collect()

    X_combined = build_features(
        combined,
        combined,
        combined_weights,
        fit_is_training=True,
    )
    X_test = build_features(
        combined,
        test,
        combined_weights,
        fit_is_training=False,
    )

    y_combined = np.asarray(combined.y, dtype=np.int8)
    family = recipe["family"]

    if family == "pointwise_lgbm":
        raw_test = fit_final_binary(
            X_combined,
            y_combined,
            combined_weights,
            X_test,
            recipe["rounds"],
        )
    elif family == "querywise_lambdamart":
        raw_test = fit_final_lambdarank(
            X_combined,
            y_combined,
            combined.user_id,
            combined_weights,
            X_test,
            recipe["rounds"],
        )
    elif family == "bagged_extra_trees":
        final_extra = ExtraTreesClassifier(
            n_estimators=48,
            criterion="entropy",
            max_depth=20,
            min_samples_leaf=24,
            max_features=0.72,
            bootstrap=False,
            class_weight=None,
            n_jobs=-1,
            random_state=2028,
        )
        final_extra.fit(
            X_combined,
            y_combined,
            sample_weight=combined_weights,
        )
        raw_test = final_extra.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Unknown final family: %s" % family)

    weight = float(recipe["blend_weight"])
    mode = recipe["blend_mode"]

    if mode == "standalone":
        test_scores = raw_test
    elif mode == "rank":
        test_scores = (
            weight * within_user_rank(test.user_id, raw_test)
            + (1.0 - weight)
            * within_user_rank(test.user_id, inc_test)
        )
    elif mode == "logit":
        test_scores = (
            weight * logit(raw_test)
            + (1.0 - weight) * logit(inc_test)
        )
    else:
        raise ValueError("Unknown blend mode: %s" % mode)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, '
    '"ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)