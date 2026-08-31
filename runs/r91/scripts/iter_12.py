import os
import time
import json
import numpy as np

from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()
SMOOTHING = 200.0

# Feature cardinalities for the sequence-context bins built below.
CARDS = np.asarray([10, 10, 7, 5, 6, 6, 6, 6], dtype=np.int64)

FAMILIES = {
    # Position/hazard model: propensity changes over a user's feed trajectory.
    "position_hazard": [0, 1, 2, 3, 4],
    # Satiation model: repeated exposure to the same content entity may reduce relevance.
    "repeat_satiation": [5, 6, 7],
    # A broader slate-context additive model.
    "context_gam": [0, 1, 2, 3, 4, 5, 6, 7],
}

BLEND_WEIGHTS = [0.025, 0.05, 0.10, 0.175, 0.25, 0.35, 0.50]


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(float(x.std()), 1e-10)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    positions = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    denominators = np.maximum(np.repeat(sizes, sizes) - 1, 1)

    result = np.empty(n, dtype=np.float64)
    result[order] = positions / denominators
    return result


def group_position_total(keys, time_ms=None):
    """
    Causal position and total group size, vectorized. If time_ms is supplied,
    rows are ordered chronologically within each key group; row index breaks
    timestamp ties.
    """
    arrays = [np.asarray(k, dtype=np.int64) for k in keys]
    n = len(arrays[0])
    rows = np.arange(n, dtype=np.int64)

    # np.lexsort uses the final key as the primary key.
    sort_keys = [rows]
    if time_ms is not None:
        sort_keys.append(np.asarray(time_ms, dtype=np.int64))
    sort_keys.extend(arrays[::-1])
    order = np.lexsort(tuple(sort_keys))

    changed = np.zeros(n, dtype=bool)
    changed[0] = True
    for arr in arrays:
        sorted_arr = arr[order]
        changed[1:] |= sorted_arr[1:] != sorted_arr[:-1]

    starts = np.flatnonzero(changed)
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    pos_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    total_sorted = np.repeat(sizes, sizes)

    position = np.empty(n, dtype=np.int64)
    total = np.empty(n, dtype=np.int64)
    position[order] = pos_sorted
    total[order] = total_sorted
    return position, total


def context_features(split):
    user = np.asarray(split.user_id, dtype=np.int64)
    date = np.asarray(split.date, dtype=np.int64)
    time_ms = np.asarray(split.time_ms, dtype=np.int64)

    day_pos, day_total = group_position_total(
        [user, date], time_ms=time_ms
    )
    window_pos, window_total = group_position_total(
        [user], time_ms=time_ms
    )
    batch_pos, batch_total = group_position_total(
        [user, time_ms], time_ms=None
    )

    author_pos, _ = group_position_total(
        [user, date, np.asarray(split.X["author_id"], dtype=np.int64)],
        time_ms=time_ms,
    )
    video_pos, _ = group_position_total(
        [user, date, np.asarray(split.X["video_id"], dtype=np.int64)],
        time_ms=time_ms,
    )
    tag_pos, _ = group_position_total(
        [user, date, np.asarray(split.X["tag"], dtype=np.int64)],
        time_ms=time_ms,
    )

    day_quantile = np.minimum(
        9, (10 * day_pos) // np.maximum(day_total, 1)
    )
    window_quantile = np.minimum(
        9, (10 * window_pos) // np.maximum(window_total, 1)
    )

    # Seven bins: 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64+.
    day_size_bin = np.digitize(
        day_total, np.asarray([1, 3, 7, 15, 31, 63])
    ).astype(np.int64)

    # Causal position in a simultaneous feed batch and batch multiplicity.
    batch_position_bin = np.minimum(batch_pos, 4)
    batch_size_bin = np.minimum(batch_total - 1, 5)

    # Number of prior same-entity exposures during the current user-day.
    author_repeat = np.minimum(author_pos, 5)
    video_repeat = np.minimum(video_pos, 5)
    tag_repeat = np.minimum(tag_pos, 5)

    return np.ascontiguousarray(
        np.column_stack(
            [
                day_quantile,
                window_quantile,
                day_size_bin,
                batch_position_bin,
                batch_size_bin,
                author_repeat,
                video_repeat,
                tag_repeat,
            ]
        ),
        dtype=np.int16,
    )


def fit_gam(features, labels):
    labels = np.asarray(labels, dtype=np.float64)
    global_rate = float(labels.mean())
    global_rate = np.clip(global_rate, 1e-6, 1.0 - 1e-6)
    global_logit = np.log(global_rate / (1.0 - global_rate))

    tables = []
    for j, card in enumerate(CARDS):
        ids = np.asarray(features[:, j], dtype=np.int64)
        count = np.bincount(ids, minlength=int(card)).astype(np.float64)
        positive = np.bincount(
            ids, weights=labels, minlength=int(card)
        ).astype(np.float64)

        rate = (
            positive + SMOOTHING * global_rate
        ) / (count + SMOOTHING)
        rate = np.clip(rate, 1e-6, 1.0 - 1e-6)
        effect = np.log(rate / (1.0 - rate)) - global_logit

        # Weighted centering keeps each feature a pure ranking residual.
        effect -= np.sum(effect * count) / max(float(count.sum()), 1.0)
        tables.append(effect)
    return tables


def gam_score(features, tables, columns):
    score = np.zeros(len(features), dtype=np.float64)
    for j in columns:
        score += tables[j][np.asarray(features[:, j], dtype=np.int64)]
    score /= np.sqrt(max(len(columns), 1))
    return score


def novelty_score(features):
    # Label-free diversity/satiation prior, structurally distinct from the GAM:
    # penalize prior duplicate entities, emphasizing exact video duplication.
    author_repeat = features[:, 5].astype(np.float64)
    video_repeat = features[:, 6].astype(np.float64)
    tag_repeat = features[:, 7].astype(np.float64)
    return -(
        0.55 * np.log1p(author_repeat)
        + 1.00 * np.log1p(video_repeat)
        + 0.20 * np.log1p(tag_repeat)
    )


def make_signals(features, tables):
    signals = {
        name: gam_score(features, tables, columns)
        for name, columns in FAMILIES.items()
    }
    signals["duplicate_diversity"] = novelty_score(features)

    # A hazard formulation combines a learned position model and a separate
    # non-parametric duplicate penalty rather than fitting another CTR model.
    signals["hazard_plus_diversity"] = (
        signals["position_hazard"] + 0.5 * signals["duplicate_diversity"]
    )
    return signals


def main():
    train = load("train")
    valid = load("valid")

    y_train = np.asarray(train.y, dtype=np.float64)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid = np.load(
        os.path.join(shared, "incumbent_valid_scores.npy")
    ).astype(np.float64)
    incumbent_test_path = os.path.join(
        shared, "incumbent_test_scores.npy"
    )

    train_features = context_features(train)
    valid_features = context_features(valid)
    train_tables = fit_gam(train_features, y_train)
    valid_signals = make_signals(valid_features, train_tables)

    incumbent_metrics = evaluate(
        valid.user_id, y_valid, incumbent_valid
    )
    incumbent_primary = float(incumbent_metrics["primary"])

    candidates = {
        "trusted_incumbent": incumbent_primary,
    }

    best_primary = incumbent_primary
    best_valid = incumbent_valid.copy()
    best_descriptor = ("incumbent", "raw", 0.0)

    incumbent_z = zscore(incumbent_valid)
    incumbent_rank = within_user_rank(valid.user_id, incumbent_valid)

    for family_name, signal in valid_signals.items():
        signal = np.asarray(signal, dtype=np.float64)

        standalone_metrics = evaluate(valid.user_id, y_valid, signal)
        candidates[
            family_name + "_standalone"
        ] = float(standalone_metrics["primary"])

        if float(signal.std()) < 1e-12:
            continue

        signal_z = zscore(signal)
        signal_rank = within_user_rank(valid.user_id, signal)

        for weight in BLEND_WEIGHTS:
            raw_blend = (
                (1.0 - weight) * incumbent_z + weight * signal_z
            )
            raw_primary = float(
                evaluate(valid.user_id, y_valid, raw_blend)["primary"]
            )
            candidates[
                f"{family_name}_zblend_{weight:.3f}"
            ] = raw_primary
            if raw_primary > best_primary:
                best_primary = raw_primary
                best_valid = raw_blend.copy()
                best_descriptor = (family_name, "zblend", weight)

            rank_blend = (
                (1.0 - weight) * incumbent_rank + weight * signal_rank
            )
            rank_primary = float(
                evaluate(valid.user_id, y_valid, rank_blend)["primary"]
            )
            candidates[
                f"{family_name}_rankblend_{weight:.3f}"
            ] = rank_primary
            if rank_primary > best_primary:
                best_primary = rank_primary
                best_valid = rank_blend.copy()
                best_descriptor = (family_name, "rankblend", weight)

    metrics = evaluate(valid.user_id, y_valid, best_valid)

    print(
        "FINDINGS "
        + json.dumps(
            {
                "winner_family": best_descriptor[0],
                "winner_blend": best_descriptor[1],
                "winner_weight": float(best_descriptor[2]),
                "incumbent_primary": incumbent_primary,
                "winner_primary": float(metrics["primary"]),
                "position_signal_std": float(
                    valid_signals["position_hazard"].std()
                ),
                "repeat_signal_std": float(
                    valid_signals["repeat_satiation"].std()
                ),
                "valid_repeated_author_share": float(
                    np.mean(valid_features[:, 5] > 0)
                ),
                "valid_repeated_video_share": float(
                    np.mean(valid_features[:, 6] > 0)
                ),
                "valid_mult-row_batch_share": float(
                    np.mean(valid_features[:, 4] > 0)
                ),
            },
            sort_keys=True,
        )
    )
    print("CANDIDATES " + json.dumps(candidates, sort_keys=True))

    out = os.environ.get("ITER_OUT")
    if out:
        np.save(
            os.path.join(out, "scores_valid.npy"),
            np.asarray(best_valid, dtype=np.float64),
        )

    test = load("test")
    incumbent_test = np.load(incumbent_test_path).astype(np.float64)

    if best_descriptor[0] == "incumbent":
        test_scores = incumbent_test
    else:
        # Refit the identical target-effect recipe using all labels available
        # before the test period. Context features remain split-local and causal.
        valid_labels_float = y_valid.astype(np.float64)
        refit_features = np.concatenate(
            [train_features, valid_features], axis=0
        )
        refit_labels = np.concatenate(
            [y_train, valid_labels_float], axis=0
        )
        refit_tables = fit_gam(refit_features, refit_labels)

        test_features = context_features(test)
        test_signals = make_signals(test_features, refit_tables)
        selected_signal = test_signals[best_descriptor[0]]
        weight = float(best_descriptor[2])

        if best_descriptor[1] == "zblend":
            test_scores = (
                (1.0 - weight) * zscore(incumbent_test)
                + weight * zscore(selected_signal)
            )
        elif best_descriptor[1] == "rankblend":
            test_scores = (
                (1.0 - weight)
                * within_user_rank(test.user_id, incumbent_test)
                + weight
                * within_user_rank(test.user_id, selected_signal)
            )
        else:
            test_scores = selected_signal

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