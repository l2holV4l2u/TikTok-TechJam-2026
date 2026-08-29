import os
import time
import json
import gc

import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


SEED = 2024

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "hour",
    "tag",
    "upload_type",
    "music_type",
    "video_type",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

HISTORY_KEYS = ("video_id", "author_id")


def group_positions_and_sizes(sorted_group_start):
    n = sorted_group_start.shape[0]
    indices = np.arange(n, dtype=np.int64)
    start_indices = np.where(sorted_group_start, indices, 0)
    start_indices = np.maximum.accumulate(start_indices)
    positions = indices - start_indices

    boundaries = np.flatnonzero(sorted_group_start)
    ends = np.r_[boundaries[1:], n]
    lengths = ends - boundaries
    sizes = np.repeat(lengths, lengths)
    return positions, sizes


def temporal_features(split):
    n = len(split.user_id)
    row = np.arange(n, dtype=np.int64)
    uid = np.asarray(split.user_id, dtype=np.int64)
    date = np.asarray(split.date, dtype=np.int64)
    timestamp = np.asarray(split.time_ms, dtype=np.int64)
    video = np.asarray(split.video_id, dtype=np.int64)
    author = np.asarray(split.X["author_id"], dtype=np.int64)

    order = np.lexsort((row, timestamp, date, uid))

    suid = uid[order]
    sdate = date[order]
    stime = timestamp[order]
    svideo = video[order]
    sauthor = author[order]

    new_day = np.empty(n, dtype=bool)
    new_day[0] = True
    new_day[1:] = (
        (suid[1:] != suid[:-1])
        | (sdate[1:] != sdate[:-1])
    )
    day_pos, day_size = group_positions_and_sizes(new_day)

    new_batch = np.empty(n, dtype=bool)
    new_batch[0] = True
    new_batch[1:] = (
        (suid[1:] != suid[:-1])
        | (stime[1:] != stime[:-1])
    )
    batch_pos, batch_size = group_positions_and_sizes(new_batch)

    gap_ms = np.zeros(n, dtype=np.float64)
    same_day_previous = ~new_day
    gap_ms[1:] = np.where(
        same_day_previous[1:],
        np.maximum(stime[1:] - stime[:-1], 0),
        0,
    )
    gap_seconds_log = np.log1p(
        np.minimum(gap_ms / 1000.0, 86400.0)
    ).astype(np.float32)

    adjacent_video_repeat = np.zeros(n, dtype=np.float32)
    adjacent_author_repeat = np.zeros(n, dtype=np.float32)
    adjacent_video_repeat[1:] = (
        same_day_previous[1:]
        & (svideo[1:] == svideo[:-1])
    ).astype(np.float32)
    adjacent_author_repeat[1:] = (
        same_day_previous[1:]
        & (sauthor[1:] == sauthor[:-1])
    ).astype(np.float32)

    day_fraction = np.divide(
        day_pos,
        np.maximum(day_size - 1, 1),
        dtype=np.float64,
    ).astype(np.float32)
    batch_fraction = np.divide(
        batch_pos,
        np.maximum(batch_size - 1, 1),
        dtype=np.float64,
    ).astype(np.float32)

    sorted_features = np.column_stack(
        [
            np.minimum(day_pos, 63).astype(np.float32),
            np.log1p(day_pos).astype(np.float32),
            np.log1p(day_size).astype(np.float32),
            day_fraction,
            np.minimum(batch_pos, 15).astype(np.float32),
            np.log1p(batch_size).astype(np.float32),
            batch_fraction,
            gap_seconds_log,
            adjacent_video_repeat,
            adjacent_author_repeat,
        ]
    ).astype(np.float32, copy=False)

    result = np.empty_like(sorted_features)
    result[order] = sorted_features
    return result


def build_matrix(split_name, split):
    blocks = []

    categorical = np.column_stack(
        [
            np.asarray(split.X[name], dtype=np.float32)
            for name in CAT_FIELDS
        ]
    )
    blocks.append(categorical)

    numeric_columns = []
    for name in NUM_FIELDS:
        values = np.asarray(split.num[name], dtype=np.float32)
        finite = np.isfinite(values)
        cleaned = np.where(finite, values, 0.0)
        cleaned = np.maximum(cleaned, 0.0)
        numeric_columns.append(
            np.log1p(cleaned).astype(np.float32)
        )
        numeric_columns.append(
            (~finite).astype(np.float32)
        )
    blocks.append(np.column_stack(numeric_columns))

    blocks.append(temporal_features(split))

    history_columns = []
    history_names = []
    for key in HISTORY_KEYS:
        histories = historical_features(split_name, key=key)
        for name in sorted(histories):
            values = np.asarray(histories[name], dtype=np.float32)
            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            history_columns.append(values)
            history_names.append(name)

    if history_columns:
        blocks.append(np.column_stack(history_columns))

    matrix = np.ascontiguousarray(
        np.column_stack(blocks),
        dtype=np.float32,
    )
    return matrix, history_names


def standardization(scores):
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return mean, std


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    positions, sizes = group_positions_and_sizes(starts)
    ranked_sorted = np.divide(
        positions,
        np.maximum(sizes - 1, 1),
        dtype=np.float64,
    )
    ranked_sorted[sizes == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def main():
    start_time = time.perf_counter()

    requested_device = os.environ["AGENT_DEVICE"].lower()
    if requested_device != "cpu":
        raise RuntimeError(
            "This LightGBM experiment requires AGENT_DEVICE=cpu"
        )

    num_threads = max(
        1,
        int(
            os.environ.get(
                "OMP_NUM_THREADS",
                os.cpu_count() or 1,
            )
        ),
    )

    train = load("train")
    valid = load("valid")

    x_train, train_history_names = build_matrix("train", train)
    x_valid, valid_history_names = build_matrix("valid", valid)

    if train_history_names != valid_history_names:
        raise RuntimeError("Historical feature layouts differ")

    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.float32)

    categorical_indices = list(range(len(CAT_FIELDS)))

    train_dataset = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )
    valid_dataset = lgb.Dataset(
        x_valid,
        label=y_valid,
        categorical_feature=categorical_indices,
        reference=train_dataset,
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "min_data_per_group": 200,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "max_cat_threshold": 32,
        "verbosity": -1,
        "verbose": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "data_random_seed": SEED + 3,
        "num_threads": num_threads,
        "force_col_wise": True,
    }

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=700,
        valid_sets=[valid_dataset],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(60, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    tree_valid_scores = np.asarray(
        model.predict(
            x_valid,
            num_iteration=model.best_iteration,
        ),
        dtype=np.float64,
    )
    tree_metrics = evaluate(
        valid.user_id,
        valid.y,
        tree_valid_scores,
    )

    artifacts = os.environ.get("RUN_ARTIFACTS", "")
    incumbent_valid_path = os.path.join(
        artifacts,
        "incumbent_valid_scores.npy",
    )
    incumbent_test_path = os.path.join(
        artifacts,
        "incumbent_test_scores.npy",
    )

    if not (
        os.path.isfile(incumbent_valid_path)
        and os.path.isfile(incumbent_test_path)
    ):
        raise RuntimeError("Trusted incumbent predictions are unavailable")

    incumbent_valid = np.asarray(
        np.load(incumbent_valid_path),
        dtype=np.float64,
    )
    if incumbent_valid.shape != tree_valid_scores.shape:
        raise RuntimeError("Incumbent validation shape mismatch")

    incumbent_metrics = evaluate(
        valid.user_id,
        valid.y,
        incumbent_valid,
    )

    tree_mean, tree_std = standardization(tree_valid_scores)
    incumbent_mean, incumbent_std = standardization(incumbent_valid)

    tree_z = (tree_valid_scores - tree_mean) / tree_std
    incumbent_z = (
        incumbent_valid - incumbent_mean
    ) / incumbent_std

    tree_rank = within_user_rank(
        valid.user_id,
        tree_valid_scores,
    )
    incumbent_rank = within_user_rank(
        valid.user_id,
        incumbent_valid,
    )

    candidate_results = {
        "lightgbm_raw": float(tree_metrics["primary"]),
        "incumbent": float(incumbent_metrics["primary"]),
    }

    selected_primary = float(incumbent_metrics["primary"])
    selected_method = "incumbent"
    selected_alpha = 0.0
    valid_scores = incumbent_valid.copy()

    alphas = np.linspace(0.0, 1.0, 21)

    for alpha in alphas:
        alpha = float(alpha)

        if alpha == 0.0:
            z_blend = incumbent_valid
        elif alpha == 1.0:
            z_blend = tree_valid_scores
        else:
            z_blend = (
                alpha * tree_z
                + (1.0 - alpha) * incumbent_z
            )

        z_metrics = evaluate(
            valid.user_id,
            valid.y,
            z_blend,
        )
        z_name = "zblend_tree_%0.2f" % alpha
        candidate_results[z_name] = float(
            z_metrics["primary"]
        )

        if float(z_metrics["primary"]) > selected_primary:
            selected_primary = float(z_metrics["primary"])
            selected_method = "zblend"
            selected_alpha = alpha
            valid_scores = np.asarray(
                z_blend,
                dtype=np.float64,
            ).copy()

        if alpha == 0.0:
            rank_blend = incumbent_valid
        elif alpha == 1.0:
            rank_blend = tree_valid_scores
        else:
            rank_blend = (
                alpha * tree_rank
                + (1.0 - alpha) * incumbent_rank
            )

        rank_metrics = evaluate(
            valid.user_id,
            valid.y,
            rank_blend,
        )
        rank_name = "rankblend_tree_%0.2f" % alpha
        candidate_results[rank_name] = float(
            rank_metrics["primary"]
        )

        if float(rank_metrics["primary"]) > selected_primary:
            selected_primary = float(rank_metrics["primary"])
            selected_method = "rankblend"
            selected_alpha = alpha
            valid_scores = np.asarray(
                rank_blend,
                dtype=np.float64,
            ).copy()

    metrics = evaluate(
        valid.user_id,
        valid.y,
        valid_scores,
    )

    print(
        "FINDINGS best_iteration=%d n_features=%d "
        "history_features=%d lightgbm_primary=%.6f "
        "incumbent_primary=%.6f selected_method=%s "
        "selected_tree_weight=%.2f selected_primary=%.6f"
        % (
            int(model.best_iteration),
            int(x_train.shape[1]),
            len(train_history_names),
            float(tree_metrics["primary"]),
            float(incumbent_metrics["primary"]),
            selected_method,
            selected_alpha,
            float(metrics["primary"]),
        ),
        flush=True,
    )
    print(
        "CANDIDATES "
        + json.dumps(candidate_results, sort_keys=True),
        flush=True,
    )

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(valid_scores, dtype=np.float64),
        )

    del train_dataset
    del valid_dataset
    del x_train
    del x_valid
    del y_train
    del y_valid
    del train
    gc.collect()

    test = load("test")
    x_test, test_history_names = build_matrix("test", test)

    if test_history_names != train_history_names:
        raise RuntimeError("Test historical feature layout differs")

    tree_test_scores = np.asarray(
        model.predict(
            x_test,
            num_iteration=model.best_iteration,
        ),
        dtype=np.float64,
    )

    incumbent_test = np.asarray(
        np.load(incumbent_test_path),
        dtype=np.float64,
    )
    if incumbent_test.shape != tree_test_scores.shape:
        raise RuntimeError("Incumbent test shape mismatch")

    if selected_method == "incumbent" or selected_alpha == 0.0:
        test_scores = incumbent_test
    elif selected_method == "zblend":
        tree_test_z = (
            tree_test_scores - tree_mean
        ) / tree_std
        incumbent_test_z = (
            incumbent_test - incumbent_mean
        ) / incumbent_std
        test_scores = (
            selected_alpha * tree_test_z
            + (1.0 - selected_alpha) * incumbent_test_z
        )
    elif selected_method == "rankblend":
        tree_test_rank = within_user_rank(
            test.user_id,
            tree_test_scores,
        )
        incumbent_test_rank = within_user_rank(
            test.user_id,
            incumbent_test,
        )
        test_scores = (
            selected_alpha * tree_test_rank
            + (1.0 - selected_alpha) * incumbent_test_rank
        )
    else:
        raise RuntimeError("Unknown selected blend method")

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    elapsed = time.perf_counter() - start_time
    final = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
        "device": "cpu",
    }
    print(
        "METRICS "
        + json.dumps(final, separators=(", ", ": "))
    )


if __name__ == "__main__":
    main()