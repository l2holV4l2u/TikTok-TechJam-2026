import os
import time
import json
import gc
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float64)
yv = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def per_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    su = users[order]
    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = su[1:] != su[:-1]

    starts = np.where(starts_flag, np.arange(n), 0)
    starts = np.maximum.accumulate(starts)
    within = np.arange(n) - starts

    group_starts = np.flatnonzero(starts_flag)
    group_ends = np.r_[group_starts[1:], n]
    sizes = group_ends - group_starts
    repeated_sizes = np.repeat(sizes, sizes)

    ranked = np.where(
        repeated_sizes > 1,
        within / np.maximum(repeated_sizes - 1, 1),
        0.5,
    ).astype(np.float64)

    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def user_centered_target(users, labels):
    users = np.asarray(users, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    cardinality = int(users.max()) + 1
    count = np.bincount(users, minlength=cardinality).astype(np.float64)
    positive = np.bincount(
        users, weights=labels, minlength=cardinality
    ).astype(np.float64)
    means = positive / np.maximum(count, 1.0)
    return labels - means[users]


# ----------------------------------------------------------------------
# Family 1: user-fixed-effect conditional ridge.
#
# The target is demeaned inside each training user. Thus the model cannot
# spend capacity predicting whether a user is generally likely to long-view;
# it estimates only item/context utility relative to alternatives shown to
# the same user. Ridge pooling keeps rare item and content effects stable.
# ----------------------------------------------------------------------

ridge_fields = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "hour",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

offsets = {}
total_columns = 0
for field in ridge_fields:
    offsets[field] = total_columns
    total_columns += int(FEATURE_CARDINALITIES[field])


def make_sparse(split):
    n = len(split.user_id)
    k = len(ridge_fields)
    rows = np.tile(np.arange(n, dtype=np.int32), k)
    cols = np.empty(n * k, dtype=np.int32)

    for j, field in enumerate(ridge_fields):
        a = j * n
        b = (j + 1) * n
        cols[a:b] = (
            np.asarray(split.X[field], dtype=np.int32) + offsets[field]
        )

    data = np.ones(n * k, dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n, total_columns),
        dtype=np.float32,
    )


Xtr = make_sparse(train)
Xva = make_sparse(valid)
Xte = make_sparse(test)

centered_y = user_centered_target(train.user_id, y)

dates = np.asarray(train.date, dtype=np.int32)
last_date = int(dates.max())
age = (last_date - dates).astype(np.float64)

# A mild decay changes the effective target toward the split boundary without
# throwing away the high-volume early training days.
ridge_weight = np.exp(-np.log(2.0) * age / 10.0)
ridge_weight /= np.mean(ridge_weight)
sqrt_weight = np.sqrt(ridge_weight)

weighted_Xtr = Xtr.multiply(sqrt_weight[:, None]).tocsr()
weighted_target = centered_y * sqrt_weight

ridge_solution = lsqr(
    weighted_Xtr,
    weighted_target,
    damp=7.0,
    atol=2e-4,
    btol=2e-4,
    iter_lim=45,
    show=False,
)
ridge_coef = ridge_solution[0]

ridge_valid = np.asarray(Xva @ ridge_coef).ravel().astype(np.float64)
ridge_test = np.asarray(Xte @ ridge_coef).ravel().astype(np.float64)

print(
    "FINDINGS conditional_ridge_cols=%d nnz=%d lsqr_iters=%d residual=%.6f"
    % (
        total_columns,
        Xtr.nnz,
        int(ridge_solution[2]),
        float(ridge_solution[3]),
    )
)

del weighted_Xtr, weighted_target, Xtr, Xva, Xte
gc.collect()


# ----------------------------------------------------------------------
# Family 2: metadata-graph diffusion.
#
# Each video starts from its recency-weighted empirical long-view posterior.
# Repeated diffusion replaces part of that estimate with count-weighted
# averages from videos sharing author, tag, or stable content descriptors.
# This transfers signal to sparse videos while retaining identity-specific
# quality for well-observed videos.
# ----------------------------------------------------------------------

video_card = int(FEATURE_CARDINALITIES["video_id"])
train_video = np.asarray(train.video_id, dtype=np.int64)

global_rate = float(np.average(y, weights=ridge_weight))

video_count = np.bincount(
    train_video,
    weights=ridge_weight,
    minlength=video_card,
).astype(np.float64)
video_positive = np.bincount(
    train_video,
    weights=ridge_weight * y,
    minlength=video_card,
).astype(np.float64)

video_base_rate = (
    video_positive + 24.0 * global_rate
) / np.maximum(video_count + 24.0, 1e-12)
video_base_logit = logit(video_base_rate)

graph_fields = [
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

# Metadata is effectively video-side data. Last assignment is deterministic
# and avoids any use of validation/test distributions.
video_metadata = {}
for field in graph_fields:
    values = np.asarray(train.X[field], dtype=np.int64)
    metadata = np.zeros(video_card, dtype=np.int64)
    metadata[train_video] = values
    video_metadata[field] = metadata

graph_score = video_base_logit.copy()
node_weight = video_count + 8.0

for _ in range(6):
    neighbor_sum = np.zeros(video_card, dtype=np.float64)

    for field in graph_fields:
        group = video_metadata[field]
        card = int(FEATURE_CARDINALITIES[field])

        group_weight = np.bincount(
            group,
            weights=node_weight,
            minlength=card,
        ).astype(np.float64)
        group_signal = np.bincount(
            group,
            weights=node_weight * graph_score,
            minlength=card,
        ).astype(np.float64)

        group_mean = group_signal / np.maximum(group_weight, 1e-12)
        neighbor_sum += group_mean[group]

    neighbor_mean = neighbor_sum / len(graph_fields)

    # Direct evidence remains dominant; diffusion only regularizes it.
    reliability = video_count / (video_count + 35.0)
    propagated = 0.62 * video_base_logit + 0.38 * neighbor_mean
    graph_score = (
        reliability * video_base_logit
        + (1.0 - reliability) * propagated
    )

graph_valid = graph_score[
    np.asarray(valid.video_id, dtype=np.int64)
]
graph_test = graph_score[
    np.asarray(test.video_id, dtype=np.int64)
]

print(
    "FINDINGS graph_observed_videos=%d sparse_videos=%d diffusion_fields=%d"
    % (
        int(np.sum(video_count > 0)),
        int(np.sum((video_count > 0) & (video_count < 20))),
        len(graph_fields),
    )
)


# ----------------------------------------------------------------------
# Family 3: median-across-days entity consensus.
#
# Pooled popularity can be dominated by one high-volume or anomalous day.
# Here each training day contributes one shrunk entity posterior and the
# final entity utility is the median logit across days. Author consensus is
# added as a lower-variance backoff to video consensus.
# ----------------------------------------------------------------------

unique_dates = np.sort(np.unique(dates))


def temporal_consensus(entity_ids, cardinality, prior):
    entity_ids = np.asarray(entity_ids, dtype=np.int64)
    day_logits = np.empty(
        (len(unique_dates), cardinality), dtype=np.float32
    )

    for j, day in enumerate(unique_dates):
        mask = dates == day
        ids = entity_ids[mask]
        labels = y[mask]

        counts = np.bincount(
            ids, minlength=cardinality
        ).astype(np.float64)
        positives = np.bincount(
            ids, weights=labels, minlength=cardinality
        ).astype(np.float64)

        day_global = float(labels.mean())
        rates = (
            positives + prior * day_global
        ) / np.maximum(counts + prior, 1e-12)
        day_logits[j] = logit(rates).astype(np.float32)

    consensus = np.median(day_logits, axis=0).astype(np.float64)
    dispersion = np.median(
        np.abs(day_logits - consensus[None, :]),
        axis=0,
    ).astype(np.float64)
    return consensus, dispersion


video_consensus, video_dispersion = temporal_consensus(
    train.video_id,
    video_card,
    prior=18.0,
)

author_card = int(FEATURE_CARDINALITIES["author_id"])
author_consensus, author_dispersion = temporal_consensus(
    train.X["author_id"],
    author_card,
    prior=25.0,
)

valid_video = np.asarray(valid.video_id, dtype=np.int64)
test_video = np.asarray(test.video_id, dtype=np.int64)
valid_author = np.asarray(valid.X["author_id"], dtype=np.int64)
test_author = np.asarray(test.X["author_id"], dtype=np.int64)

# Entities with unstable daily estimates are automatically pulled more toward
# their lower-variance author consensus.
valid_reliability = 1.0 / (1.0 + video_dispersion[valid_video])
test_reliability = 1.0 / (1.0 + video_dispersion[test_video])

consensus_valid = (
    valid_reliability * video_consensus[valid_video]
    + (1.0 - valid_reliability) * author_consensus[valid_author]
)
consensus_test = (
    test_reliability * video_consensus[test_video]
    + (1.0 - test_reliability) * author_consensus[test_author]
)

print(
    "FINDINGS temporal_days=%d median_video_dispersion=%.5f "
    "median_author_dispersion=%.5f"
    % (
        len(unique_dates),
        float(np.median(video_dispersion[video_count > 0])),
        float(np.median(author_dispersion)),
    )
)


# ----------------------------------------------------------------------
# Family 4: heterogeneous rank aggregation.
#
# This combines the conditional utility, graph-smoothed quality, and robust
# temporal consensus. They make errors for different reasons, so averaging
# within-user percentile ranks can reduce family-specific variance without
# relying on their incompatible score calibrations.
# ----------------------------------------------------------------------

rv = per_user_rank(valid.user_id, ridge_valid)
rt = per_user_rank(test.user_id, ridge_test)
gv = per_user_rank(valid.user_id, graph_valid)
gt = per_user_rank(test.user_id, graph_test)
cv = per_user_rank(valid.user_id, consensus_valid)
ct = per_user_rank(test.user_id, consensus_test)

aggregate_valid = 0.46 * rv + 0.29 * gv + 0.25 * cv
aggregate_test = 0.46 * rt + 0.29 * gt + 0.25 * ct

families = {
    "conditional_fixed_effect_ridge": (ridge_valid, ridge_test),
    "metadata_graph_diffusion": (graph_valid, graph_test),
    "median_day_entity_consensus": (consensus_valid, consensus_test),
    "heterogeneous_rank_aggregation": (
        aggregate_valid,
        aggregate_test,
    ),
}

inc_valid_rank = per_user_rank(valid.user_id, inc_valid)
inc_test_rank = per_user_rank(test.user_id, inc_test)

candidate_log = {}
best_primary = -np.inf
best_valid = None
best_test = None
best_raw_valid = None
best_name = None
best_is_blend = False

# Include the incumbent as a safety candidate. All new-family blend weights
# are selected only from validation and then transferred unchanged to test.
inc_metrics = evaluate(valid.user_id, yv, inc_valid)
candidate_log["trusted_incumbent"] = float(inc_metrics["primary"])

if inc_metrics["primary"] > best_primary:
    best_primary = float(inc_metrics["primary"])
    best_valid = inc_valid.copy()
    best_test = inc_test.copy()
    best_raw_valid = aggregate_valid.copy()
    best_name = "trusted_incumbent"
    best_is_blend = True

own_weights = [0.08, 0.14, 0.22, 0.32, 0.45, 0.60, 1.0]

for family_name, (raw_valid, raw_test) in families.items():
    raw_metrics = evaluate(valid.user_id, yv, raw_valid)
    candidate_log[family_name] = float(raw_metrics["primary"])

    raw_valid_rank = per_user_rank(valid.user_id, raw_valid)
    raw_test_rank = per_user_rank(test.user_id, raw_test)

    for own_weight in own_weights:
        if own_weight == 1.0:
            blended_valid = raw_valid
            blended_test = raw_test
            blend_name = family_name + "_standalone"
        else:
            blended_valid = (
                (1.0 - own_weight) * inc_valid_rank
                + own_weight * raw_valid_rank
            )
            blended_test = (
                (1.0 - own_weight) * inc_test_rank
                + own_weight * raw_test_rank
            )
            blend_name = "%s_blend_%.2f" % (
                family_name,
                own_weight,
            )

        metrics = evaluate(valid.user_id, yv, blended_valid)
        primary = float(metrics["primary"])
        candidate_log[blend_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_valid = np.asarray(
                blended_valid, dtype=np.float64
            ).copy()
            best_test = np.asarray(
                blended_test, dtype=np.float64
            ).copy()
            best_raw_valid = np.asarray(
                raw_valid, dtype=np.float64
            ).copy()
            best_name = blend_name
            best_is_blend = own_weight < 1.0

final_metrics = evaluate(valid.user_id, yv, best_valid)

print(
    "CANDIDATES "
    + json.dumps(
        candidate_log,
        sort_keys=True,
        separators=(",", ":"),
    )
)
print(
    "FINDINGS selected=%s primary=%.8f gauc=%.8f ndcg5=%.8f"
    % (
        best_name,
        float(final_metrics["primary"]),
        float(final_metrics["gauc"]),
        float(final_metrics["ndcg@5"]),
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
    if best_is_blend:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.6f}'
    % (
        float(final_metrics["primary"]),
        float(final_metrics["gauc"]),
        float(final_metrics["ndcg@5"]),
        float(elapsed),
    )
)