import os
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate


SEED = 2024
NUM_BOOST_ROUND = 350
CHECKPOINT_STEP = 25


def make_matrix(split, fields):
    return np.column_stack(
        [
            np.asarray(split.X[field], dtype=np.int32)
            for field in fields
        ]
    )


def main():
    train = load("train")
    valid = load("valid")

    fields = list(train.X.keys())
    x_train = make_matrix(train, fields)
    x_valid = make_matrix(valid, fields)
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    categorical_features = list(range(len(fields)))

    dtrain = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=fields,
        categorical_feature=categorical_features,
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        x_valid,
        label=y_valid,
        reference=dtrain,
        feature_name=fields,
        categorical_feature=categorical_features,
        free_raw_data=True,
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 200,
        "min_gain_to_split": 1e-4,
        "lambda_l1": 0.05,
        "lambda_l2": 1.0,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "max_bin": 127,
        "max_cat_threshold": 64,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "verbosity": -1,
        "verbose": -1,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED + 1,
        "data_random_seed": SEED + 2,
        "deterministic": True,
        "force_col_wise": True,
        "first_metric_only": True,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=40,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )

    max_iteration = int(model.best_iteration or model.current_iteration())
    candidates = list(
        range(CHECKPOINT_STEP, max_iteration + 1, CHECKPOINT_STEP)
    )
    if max_iteration not in candidates:
        candidates.append(max_iteration)
    candidates = sorted(set(max(1, value) for value in candidates))

    valid_users = np.asarray(valid.user_id, dtype=np.int64)
    best_primary = -np.inf
    best_iteration = max_iteration
    best_metrics = None

    for iteration in candidates:
        scores = model.predict(
            x_valid,
            num_iteration=iteration,
        )
        metrics = evaluate(valid_users, y_valid, scores)
        print(
            "iteration=%d primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (
                iteration,
                metrics["primary"],
                metrics["gauc"],
                metrics["ndcg@5"],
            ),
            flush=True,
        )

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_iteration = iteration
            best_metrics = dict(metrics)

    valid_scores = model.predict(
        x_valid,
        num_iteration=best_iteration,
    )
    best_metrics = evaluate(valid_users, y_valid, valid_scores)

    del x_train, x_valid, y_train, y_valid, dtrain, dvalid
    gc.collect()

    out = os.environ.get("ITER_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        test = load("test")
        x_test = make_matrix(test, fields)
        test_scores = model.predict(
            x_test,
            num_iteration=best_iteration,
        )
        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    result = {
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": 0.0,
    }
    print("METRICS " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()