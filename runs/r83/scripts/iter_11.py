import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()
SEED = 41731

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
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat3",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

HALF_LIVES = [2.5, 5.0, 10.0, None]
BLEND_WEIGHTS = [0.10, 0.20, 0.35, 0.50, 0.70]


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(float(x.std()), 1e-9)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    sizes = np.diff(np.r_[starts, n])
    positions = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    denominators = np.maximum(np.repeat(sizes, sizes) - 1, 1)
    result = np.empty(n, dtype=np.float64)
    result[order] = positions / denominators
    return result


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = day_index.max() - day_index
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    return np.exp2(-age.astype(np.float32) / float(half_life)).astype(
        np.float32
    )


def raw_numeric(split):
    columns = []
    for name in NUM_FIELDS:
        value = np.asarray(split.num[name], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(value, 0.0)))
    return np.column_stack(columns).astype(np.float32)


def sequence_features(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    st = times[order]

    user_start = np.r_[True, su[1:] != su[:-1]]
    starts = np.flatnonzero(user_start)
    sizes = np.diff(np.r_[starts, n])
    pos = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    reverse_pos = np.repeat(sizes, sizes) - 1 - pos

    gap = np.zeros(n, dtype=np.float32)
    same_user = su[1:] == su[:-1]
    raw_gap = np.maximum(st[1:] - st[:-1], 0)
    gap[1:] = np.where(same_user, raw_gap, 0).astype(np.float32)
    gap = np.log1p(gap / 1000.0)

    batch_pos = np.zeros(n, dtype=np.int64)
    same_batch = np.r_[
        False, (su[1:] == su[:-1]) & (st[1:] == st[:-1])
    ]
    batch_starts = np.flatnonzero(~same_batch)
    batch_sizes = np.diff(np.r_[batch_starts, n])
    batch_pos[:] = (
        np.arange(n, dtype=np.int64)
        - np.repeat(batch_starts, batch_sizes)
    )

    result_sorted = np.column_stack(
        [
            np.log1p(pos).astype(np.float32),
            np.log1p(reverse_pos).astype(np.float32),
            gap.astype(np.float32),
            np.log1p(batch_pos).astype(np.float32),
        ]
    )
    result = np.empty_like(result_sorted)
    result[order] = result_sorted

    day_key = users * np.int64(100000000) + dates
    day_order = np.lexsort((rows, times, day_key))
    sorted_key = day_key[day_order]
    day_starts = np.flatnonzero(
        np.r_[True, sorted_key[1:] != sorted_key[:-1]]
    )
    day_sizes = np.diff(np.r_[day_starts, n])
    day_pos = (
        np.arange(n, dtype=np.int64)
        - np.repeat(day_starts, day_sizes)
    )
    day_result = np.empty(n, dtype=np.float32)
    day_result[day_order] = np.log1p(day_pos).astype(np.float32)

    return np.column_stack([result, day_result]).astype(np.float32)


def key_arrays(split):
    user = np.asarray(split.X["user_id"], dtype=np.int64)
    video = np.asarray(split.X["video_id"], dtype=np.int64)
    author = np.asarray(split.X["author_id"], dtype=np.int64)
    tag = np.asarray(split.X["tag"], dtype=np.int64)
    duration = np.asarray(split.X["duration_bucket"], dtype=np.int64)

    return [
        video,
        author,
        tag,
        duration,
        user * np.int64(10000) + video,
        user * np.int64(10000) + author,
        user * np.int64(128) + tag,
    ]


def aggregate_rate_features(
    fit_keys,
    query_keys,
    y_fit,
    fit_weights,
    training_query=False,
    smoothing=12.0,
):
    y_fit = np.asarray(y_fit, dtype=np.float64)
    fit_weights = np.asarray(fit_weights, dtype=np.float64)
    prior = float(np.sum(fit_weights * y_fit) / np.sum(fit_weights))

    train_columns = []
    query_columns = []

    for fit_key, query_key in zip(fit_keys, query_keys):
        unique_key, inverse = np.unique(fit_key, return_inverse=True)
        sum_weight = np.bincount(
            inverse, weights=fit_weights, minlength=len(unique_key)
        )
        sum_positive = np.bincount(
            inverse,
            weights=fit_weights * y_fit,
            minlength=len(unique_key),
        )

        if training_query:
            denominator = (
                sum_weight[inverse] - fit_weights + smoothing
            )
            numerator = (
                sum_positive[inverse]
                - fit_weights * y_fit
                + smoothing * prior
            )
            rate_train = numerator / np.maximum(denominator, 1e-8)
            count_train = np.maximum(
                sum_weight[inverse] - fit_weights, 0.0
            )
            train_columns.extend(
                [
                    rate_train.astype(np.float32),
                    np.log1p(count_train).astype(np.float32),
                ]
            )

        location = np.searchsorted(unique_key, query_key)
        found = location < len(unique_key)
        safe_location = np.minimum(location, len(unique_key) - 1)
        found &= unique_key[safe_location] == query_key

        query_rate = np.full(len(query_key), prior, dtype=np.float64)
        query_count = np.zeros(len(query_key), dtype=np.float64)
        if np.any(found):
            loc = safe_location[found]
            query_rate[found] = (
                sum_positive[loc] + smoothing * prior
            ) / (sum_weight[loc] + smoothing)
            query_count[found] = sum_weight[loc]

        query_columns.extend(
            [
                query_rate.astype(np.float32),
                np.log1p(query_count).astype(np.float32),
            ]
        )

        del unique_key, inverse, sum_weight, sum_positive

    train_matrix = None
    if training_query:
        train_matrix = np.column_stack(train_columns).astype(np.float32)
    query_matrix = np.column_stack(query_columns).astype(np.float32)
    return train_matrix, query_matrix


def base_matrix(split):
    categorical = np.column_stack(
        [np.asarray(split.X[name], dtype=np.float32) for name in CAT_FIELDS]
    )
    numeric = raw_numeric(split)
    sequence = sequence_features(split)
    return np.ascontiguousarray(
        np.column_stack([categorical, numeric, sequence]),
        dtype=np.float32,
    )


def build_matrices(fit_split, query_split, y_fit):
    fit_base = base_matrix(fit_split)
    query_base = base_matrix(query_split)

    fit_keys = key_arrays(fit_split)
    query_keys = key_arrays(query_split)
    history_weights = recency_weights(fit_split.date, 5.0)

    fit_hist, query_hist = aggregate_rate_features(
        fit_keys,
        query_keys,
        y_fit,
        history_weights,
        training_query=True,
        smoothing=12.0,
    )

    fit_matrix = np.ascontiguousarray(
        np.column_stack([fit_base, fit_hist]), dtype=np.float32
    )
    query_matrix = np.ascontiguousarray(
        np.column_stack([query_base, query_hist]), dtype=np.float32
    )

    del fit_base, query_base, fit_keys, query_keys
    gc.collect()
    return fit_matrix, query_matrix, fit_hist, query_hist


def empirical_bayes_score(history_matrix):
    rates = np.clip(
        np.asarray(history_matrix[:, 0::2], dtype=np.float64),
        1e-4,
        1.0 - 1e-4,
    )
    logits = np.log(rates / (1.0 - rates))

    content = logits[:, :4].mean(axis=1)
    user_affinity = logits[:, 4:].mean(axis=1)
    return 0.35 * content + 0.65 * user_affinity


def train_lgb(
    x_train,
    y_train,
    x_valid,
    y_valid,
    sample_weight,
    num_boost_round=260,
    early_stop=True,
):
    categorical_indices = list(range(len(CAT_FIELDS)))
    dtrain = lgb.Dataset(
        x_train,
        label=y_train,
        weight=sample_weight,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 1.5,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "cat_smooth": 20.0,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "verbose": -1,
    }

    if early_stop:
        dvalid = lgb.Dataset(
            x_valid,
            label=y_valid,
            reference=dtrain,
            categorical_feature=categorical_indices,
            free_raw_data=False,
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(35, verbose=False)],
        )
    else:
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
        )
    return model


class CombinedSplit:
    pass


def combine_splits(train, valid):
    combined = CombinedSplit()
    combined.X = {}
    for name in train.X:
        combined.X[name] = np.concatenate(
            [
                np.asarray(train.X[name]),
                np.asarray(valid.X[name]),
            ]
        )

    combined.num = {}
    for name in train.num:
        combined.num[name] = np.concatenate(
            [
                np.asarray(train.num[name]),
                np.asarray(valid.num[name]),
            ]
        )

    combined.user_id = np.concatenate(
        [np.asarray(train.user_id), np.asarray(valid.user_id)]
    )
    combined.video_id = np.concatenate(
        [np.asarray(train.video_id), np.asarray(valid.video_id)]
    )
    combined.date = np.concatenate(
        [np.asarray(train.date), np.asarray(valid.date)]
    )
    combined.time_ms = np.concatenate(
        [np.asarray(train.time_ms), np.asarray(valid.time_ms)]
    )
    return combined


def main():
    train = load("train")
    valid = load("valid")
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid = np.load(
        os.path.join(shared, "incumbent_valid_scores.npy")
    ).astype(np.float64)
    incumbent_test_path = os.path.join(
        shared, "incumbent_test_scores.npy"
    )

    incumbent_metrics = evaluate(
        valid.user_id, y_valid, incumbent_valid
    )
    incumbent_rank = within_user_rank(valid.user_id, incumbent_valid)
    incumbent_z = zscore(incumbent_valid)

    candidates = {
        "trusted_incumbent": float(incumbent_metrics["primary"])
    }

    best_primary = float(incumbent_metrics["primary"])
    best_valid = incumbent_valid.copy()
    best_descriptor = {
        "family": "incumbent",
        "blend": "standalone",
        "weight": 0.0,
        "half_life": None,
        "rounds": 0,
    }

    x_train, x_valid, hist_train, hist_valid = build_matrices(
        train, valid, y_train
    )

    def consider_family(name, prediction, descriptor):
        nonlocal best_primary, best_valid, best_descriptor

        prediction = np.asarray(prediction, dtype=np.float64)
        standalone = float(
            evaluate(valid.user_id, y_valid, prediction)["primary"]
        )
        candidates[name + "_standalone"] = standalone

        if standalone > best_primary:
            best_primary = standalone
            best_valid = prediction.copy()
            best_descriptor = dict(descriptor)
            best_descriptor.update(
                {"blend": "standalone", "weight": 1.0}
            )

        pred_rank = within_user_rank(valid.user_id, prediction)
        pred_z = zscore(prediction)

        local_rank_best = -np.inf
        local_rank_weight = 0.0
        local_raw_best = -np.inf
        local_raw_weight = 0.0

        for weight in BLEND_WEIGHTS:
            rank_blend = (
                (1.0 - weight) * incumbent_rank
                + weight * pred_rank
            )
            rank_score = float(
                evaluate(
                    valid.user_id, y_valid, rank_blend
                )["primary"]
            )
            if rank_score > local_rank_best:
                local_rank_best = rank_score
                local_rank_weight = weight
            if rank_score > best_primary:
                best_primary = rank_score
                best_valid = rank_blend.copy()
                best_descriptor = dict(descriptor)
                best_descriptor.update(
                    {"blend": "rank", "weight": weight}
                )

            raw_blend = (
                (1.0 - weight) * incumbent_z
                + weight * pred_z
            )
            raw_score = float(
                evaluate(
                    valid.user_id, y_valid, raw_blend
                )["primary"]
            )
            if raw_score > local_raw_best:
                local_raw_best = raw_score
                local_raw_weight = weight
            if raw_score > best_primary:
                best_primary = raw_score
                best_valid = raw_blend.copy()
                best_descriptor = dict(descriptor)
                best_descriptor.update(
                    {"blend": "raw", "weight": weight}
                )

        candidates[name + "_best_rank_blend"] = local_rank_best
        candidates[name + "_best_rank_weight"] = local_rank_weight
        candidates[name + "_best_raw_blend"] = local_raw_best
        candidates[name + "_best_raw_weight"] = local_raw_weight

    eb_valid = empirical_bayes_score(hist_valid)
    consider_family(
        "empirical_bayes_affinity",
        eb_valid,
        {
            "family": "empirical_bayes",
            "half_life": 5.0,
            "rounds": 0,
        },
    )

    fitted_rounds = {}
    for half_life in HALF_LIVES:
        weights = recency_weights(train.date, half_life)
        model = train_lgb(
            x_train,
            y_train,
            x_valid,
            y_valid,
            weights,
            num_boost_round=260,
            early_stop=True,
        )
        rounds = int(model.best_iteration or 260)
        prediction = model.predict(
            x_valid, num_iteration=rounds
        ).astype(np.float64)

        label = (
            "uniform"
            if half_life is None
            else ("hl_" + str(half_life).replace(".", "_"))
        )
        fitted_rounds[label] = rounds
        consider_family(
            "lgb_" + label,
            prediction,
            {
                "family": "lgb",
                "half_life": half_life,
                "rounds": rounds,
            },
        )
        del model, prediction, weights
        gc.collect()

    metrics = evaluate(valid.user_id, y_valid, best_valid)

    print(
        "FINDINGS "
        + json.dumps(
            {
                "winner": best_descriptor,
                "incumbent_primary": float(
                    incumbent_metrics["primary"]
                ),
                "selected_primary": float(metrics["primary"]),
                "lgb_best_iterations": fitted_rounds,
            },
            sort_keys=True,
        )
    )
    print(
        "CANDIDATES "
        + json.dumps(candidates, sort_keys=True)
    )

    out = os.environ.get("ITER_OUT")
    if out:
        np.save(
            os.path.join(out, "scores_valid.npy"),
            np.asarray(best_valid, dtype=np.float64),
        )

    test = load("test")
    incumbent_test = np.load(incumbent_test_path).astype(np.float64)

    if best_descriptor["family"] == "incumbent":
        test_scores = incumbent_test

    else:
        combined = combine_splits(train, valid)
        y_combined = np.concatenate(
            [
                y_train,
                y_valid.astype(np.float32),
            ]
        )

        del x_train, x_valid, hist_train, hist_valid
        gc.collect()

        x_combined, x_test, combined_hist, test_hist = build_matrices(
            combined, test, y_combined
        )

        if best_descriptor["family"] == "empirical_bayes":
            component_test = empirical_bayes_score(test_hist)
        else:
            half_life = best_descriptor["half_life"]
            combined_weights = recency_weights(
                combined.date, half_life
            )
            rounds = max(20, int(best_descriptor["rounds"]))
            final_model = train_lgb(
                x_combined,
                y_combined,
                None,
                None,
                combined_weights,
                num_boost_round=rounds,
                early_stop=False,
            )
            component_test = final_model.predict(
                x_test, num_iteration=rounds
            ).astype(np.float64)
            del final_model, combined_weights

        blend_kind = best_descriptor["blend"]
        weight = float(best_descriptor["weight"])

        if blend_kind == "standalone":
            test_scores = component_test
        elif blend_kind == "rank":
            test_scores = (
                (1.0 - weight)
                * within_user_rank(test.user_id, incumbent_test)
                + weight
                * within_user_rank(test.user_id, component_test)
            )
        else:
            test_scores = (
                (1.0 - weight) * zscore(incumbent_test)
                + weight * zscore(component_test)
            )

    if out:
        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    elapsed = time.time() - START
    print(
        "METRICS "
        + json.dumps(
            {
                "primary": float(metrics["primary"]),
                "gauc": float(metrics["gauc"]),
                "ndcg@5": float(metrics["ndcg@5"]),
                "gpu_seconds": float(elapsed),
            }
        )
    )


if __name__ == "__main__":
    main()