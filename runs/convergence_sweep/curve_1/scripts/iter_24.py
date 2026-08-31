import os
import time
import json
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
warnings.filterwarnings("ignore")
np.random.seed(271828)
torch.manual_seed(271828)
torch.set_num_threads(min(8, os.cpu_count() or 8))

train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.int8)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]
    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_flag)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranks = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranks[multi] = (
        positions[multi]
        / (repeated_lengths[multi].astype(np.float64) - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def rank_blend(users, incumbent, candidate, alpha):
    return (
        (1.0 - alpha) * within_user_rank(users, incumbent)
        + alpha * within_user_rank(users, candidate)
    )


def recency_weights(dates, half_life=5.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    ages = unique_dates.size - 1 - np.searchsorted(unique_dates, dates)
    return np.exp2(-ages.astype(np.float32) / float(half_life))


# ---------------------------------------------------------------------
# Family 1: popularity-corrected sampled-softmax target representation.
# The target representation combines video, author, tag and duration.
# Negatives are other positive events sampled in proportion to recency;
# subtracting log positive-video frequency counters sampling/popularity bias.
# ---------------------------------------------------------------------

TARGET_FIELDS = ("video_id", "author_id", "tag", "duration_bucket")


class SampledSoftmaxTargetModel(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.user = nn.Embedding(FEATURE_CARDINALITIES["user_id"], dim)
        self.video = nn.Embedding(FEATURE_CARDINALITIES["video_id"], dim)
        self.author = nn.Embedding(FEATURE_CARDINALITIES["author_id"], dim)
        self.tag = nn.Embedding(FEATURE_CARDINALITIES["tag"], dim)
        self.duration = nn.Embedding(
            FEATURE_CARDINALITIES["duration_bucket"], dim
        )
        self.video_bias = nn.Embedding(
            FEATURE_CARDINALITIES["video_id"], 1
        )
        self.user_bias = nn.Embedding(
            FEATURE_CARDINALITIES["user_id"], 1
        )

        for emb in (
            self.user,
            self.video,
            self.author,
            self.tag,
            self.duration,
        ):
            nn.init.normal_(emb.weight, std=0.06)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.user_bias.weight)

    def target_vector(self, video, author, tag, duration):
        return (
            self.video(video)
            + 0.55 * self.author(author)
            + 0.30 * self.tag(tag)
            + 0.20 * self.duration(duration)
        )

    def point_scores(self, user, video, author, tag, duration):
        u = self.user(user)
        target = self.target_vector(video, author, tag, duration)
        score = (u * target).sum(dim=1) / np.sqrt(u.shape[1])
        score = score + self.video_bias(video).squeeze(1)
        score = score + self.user_bias(user).squeeze(1)
        return score


positive_rows = np.flatnonzero(train_y > 0)
positive_dates = np.asarray(train.date, dtype=np.int32)[positive_rows]
positive_weights = recency_weights(positive_dates, half_life=5.0)
positive_prob = positive_weights.astype(np.float64)
positive_prob /= positive_prob.sum()

positive_video = np.asarray(
    train.X["video_id"], dtype=np.int64
)[positive_rows]
video_positive_count = np.bincount(
    positive_video,
    weights=positive_weights,
    minlength=FEATURE_CARDINALITIES["video_id"],
).astype(np.float64)
video_positive_prob = (
    video_positive_count + 0.25
) / (
    video_positive_count.sum()
    + 0.25 * video_positive_count.size
)
log_video_q = np.log(video_positive_prob + 1e-12).astype(np.float32)

train_arrays = {
    name: np.asarray(train.X[name], dtype=np.int64)
    for name in TARGET_FIELDS
}
train_user = np.asarray(train.user_id, dtype=np.int64)

softmax_model = SampledSoftmaxTargetModel(dim=32)
optimizer = torch.optim.AdamW(
    softmax_model.parameters(), lr=0.004, weight_decay=2e-6
)

batch_size = 768
num_negatives = 96
rng = np.random.default_rng(271828)

softmax_model.train()
for epoch in range(3):
    epoch_rows = rng.permutation(positive_rows)
    epoch_loss = 0.0
    epoch_batches = 0

    for start in range(0, epoch_rows.size, batch_size):
        pos_rows = epoch_rows[start:start + batch_size]
        if pos_rows.size < 32:
            continue

        neg_positive_indices = rng.choice(
            positive_rows.size,
            size=num_negatives,
            replace=True,
            p=positive_prob,
        )
        neg_rows = positive_rows[neg_positive_indices]

        user_t = torch.from_numpy(train_user[pos_rows]).long()
        pos_video_t = torch.from_numpy(
            train_arrays["video_id"][pos_rows]
        ).long()
        pos_author_t = torch.from_numpy(
            train_arrays["author_id"][pos_rows]
        ).long()
        pos_tag_t = torch.from_numpy(
            train_arrays["tag"][pos_rows]
        ).long()
        pos_duration_t = torch.from_numpy(
            train_arrays["duration_bucket"][pos_rows]
        ).long()

        neg_video_t = torch.from_numpy(
            train_arrays["video_id"][neg_rows]
        ).long()
        neg_author_t = torch.from_numpy(
            train_arrays["author_id"][neg_rows]
        ).long()
        neg_tag_t = torch.from_numpy(
            train_arrays["tag"][neg_rows]
        ).long()
        neg_duration_t = torch.from_numpy(
            train_arrays["duration_bucket"][neg_rows]
        ).long()

        user_vector = softmax_model.user(user_t)
        pos_vector = softmax_model.target_vector(
            pos_video_t, pos_author_t, pos_tag_t, pos_duration_t
        )
        neg_vector = softmax_model.target_vector(
            neg_video_t, neg_author_t, neg_tag_t, neg_duration_t
        )

        scale = np.sqrt(user_vector.shape[1])
        pos_logit = (
            (user_vector * pos_vector).sum(dim=1) / scale
            + softmax_model.video_bias(pos_video_t).squeeze(1)
        )
        neg_logits = (
            torch.matmul(user_vector, neg_vector.T) / scale
            + softmax_model.video_bias(neg_video_t).view(1, -1)
        )

        pos_correction = torch.from_numpy(
            log_video_q[train_arrays["video_id"][pos_rows]]
        )
        neg_correction = torch.from_numpy(
            log_video_q[train_arrays["video_id"][neg_rows]]
        ).view(1, -1)

        logits = torch.cat(
            [
                (pos_logit - pos_correction).view(-1, 1),
                neg_logits - neg_correction,
            ],
            dim=1,
        )
        labels = torch.zeros(pos_rows.size, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            softmax_model.parameters(), max_norm=5.0
        )
        optimizer.step()

        epoch_loss += float(loss.detach())
        epoch_batches += 1

    print(
        "FINDINGS sampled_softmax_epoch=%d mean_loss=%.6f"
        % (epoch + 1, epoch_loss / max(epoch_batches, 1))
    )


@torch.no_grad()
def predict_sampled_softmax(split, model, batch=65536):
    model.eval()
    n = len(split.user_id)
    result = np.empty(n, dtype=np.float32)

    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    authors = np.asarray(split.X["author_id"], dtype=np.int64)
    tags = np.asarray(split.X["tag"], dtype=np.int64)
    durations = np.asarray(
        split.X["duration_bucket"], dtype=np.int64
    )

    for start in range(0, n, batch):
        end = min(start + batch, n)
        score = model.point_scores(
            torch.from_numpy(users[start:end]).long(),
            torch.from_numpy(videos[start:end]).long(),
            torch.from_numpy(authors[start:end]).long(),
            torch.from_numpy(tags[start:end]).long(),
            torch.from_numpy(durations[start:end]).long(),
        )
        result[start:end] = score.cpu().numpy()

    return result.astype(np.float64)


softmax_valid = predict_sampled_softmax(valid, softmax_model)


# ---------------------------------------------------------------------
# Family 2: positive-only nonnegative matrix factorization.
# Unlike signed SVD, multiplicative NMF represents each user's interests as
# additive nonnegative components and does not force disliked-item evidence
# into the same linear subspace.
# ---------------------------------------------------------------------

def fit_positive_nmf(split, labels, rank=28, iterations=14):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    positive = labels > 0

    weights = recency_weights(split.date, half_life=6.0)
    values = weights[positive].astype(np.float32)

    matrix = sp.coo_matrix(
        (
            values,
            (users[positive], videos[positive]),
        ),
        shape=(
            FEATURE_CARDINALITIES["user_id"],
            FEATURE_CARDINALITIES["video_id"],
        ),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    generator = np.random.default_rng(161803)
    user_factor = (
        0.05
        + generator.random(
            (matrix.shape[0], rank), dtype=np.float32
        )
    )
    item_factor = (
        0.05
        + generator.random(
            (matrix.shape[1], rank), dtype=np.float32
        )
    )

    eps = np.float32(1e-6)
    for iteration in range(iterations):
        numerator_u = matrix.dot(item_factor)
        gram_i = item_factor.T.dot(item_factor)
        denominator_u = user_factor.dot(gram_i) + eps
        user_factor *= numerator_u / denominator_u
        user_factor = np.maximum(user_factor, eps)

        numerator_i = matrix.T.dot(user_factor)
        gram_u = user_factor.T.dot(user_factor)
        denominator_i = item_factor.dot(gram_u) + eps
        item_factor *= numerator_i / denominator_i
        item_factor = np.maximum(item_factor, eps)

        # Resolve NMF's scale ambiguity to keep arithmetic stable.
        scale = np.sqrt(
            np.maximum(np.mean(user_factor * user_factor, axis=0), eps)
            / np.maximum(np.mean(item_factor * item_factor, axis=0), eps)
        )
        user_factor /= scale[None, :]
        item_factor *= scale[None, :]

    return user_factor.astype(np.float32), item_factor.astype(np.float32)


def predict_nmf(split, user_factor, item_factor):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    result = np.einsum(
        "ij,ij->i",
        user_factor[users],
        item_factor[videos],
        optimize=True,
    )
    return result.astype(np.float64)


nmf_user, nmf_item = fit_positive_nmf(
    train, train_y, rank=28, iterations=14
)
nmf_valid = predict_nmf(valid, nmf_user, nmf_item)


# ---------------------------------------------------------------------
# Family 3: recency-weighted hierarchical empirical Bayes.
# Entity rates are shrunk toward their parent's rate, rather than only a
# global rate: video -> author -> global and tag/duration -> global.
# ---------------------------------------------------------------------

def weighted_counts(ids, labels, weights, cardinality):
    count = np.bincount(
        ids, weights=weights, minlength=cardinality
    ).astype(np.float64)
    positive = np.bincount(
        ids, weights=weights * labels, minlength=cardinality
    ).astype(np.float64)
    return count, positive


def fit_hierarchical_bayes(split, labels):
    labels = np.asarray(labels, dtype=np.float64)
    weights = recency_weights(split.date, half_life=4.5).astype(
        np.float64
    )
    global_rate = float(np.sum(weights * labels) / np.sum(weights))

    author_ids = np.asarray(split.X["author_id"], dtype=np.int64)
    video_ids = np.asarray(split.X["video_id"], dtype=np.int64)
    tag_ids = np.asarray(split.X["tag"], dtype=np.int64)
    duration_ids = np.asarray(
        split.X["duration_bucket"], dtype=np.int64
    )
    tab_ids = np.asarray(split.X["tab"], dtype=np.int64)

    author_n, author_p = weighted_counts(
        author_ids,
        labels,
        weights,
        FEATURE_CARDINALITIES["author_id"],
    )
    author_rate = (
        author_p + 35.0 * global_rate
    ) / (author_n + 35.0)

    # Estimate a stable parent author for each video from train only.
    pair_key = (
        video_ids * FEATURE_CARDINALITIES["author_id"] + author_ids
    )
    unique_pair, pair_count = np.unique(pair_key, return_counts=True)
    pair_video = unique_pair // FEATURE_CARDINALITIES["author_id"]
    pair_author = unique_pair % FEATURE_CARDINALITIES["author_id"]
    ordering = np.lexsort((-pair_count, pair_video))
    ordered_video = pair_video[ordering]
    ordered_author = pair_author[ordering]
    first = np.r_[True, ordered_video[1:] != ordered_video[:-1]]
    video_parent = np.zeros(
        FEATURE_CARDINALITIES["video_id"], dtype=np.int64
    )
    video_parent[ordered_video[first]] = ordered_author[first]

    video_n, video_p = weighted_counts(
        video_ids,
        labels,
        weights,
        FEATURE_CARDINALITIES["video_id"],
    )
    video_prior = author_rate[video_parent]
    video_rate = (
        video_p + 22.0 * video_prior
    ) / (video_n + 22.0)

    tables = {
        "video_id": video_rate,
        "author_id": author_rate,
    }

    for name, ids, strength in (
        ("tag", tag_ids, 45.0),
        ("duration_bucket", duration_ids, 60.0),
        ("tab", tab_ids, 90.0),
    ):
        n, p = weighted_counts(
            ids,
            labels,
            weights,
            FEATURE_CARDINALITIES[name],
        )
        tables[name] = (
            p + strength * global_rate
        ) / (n + strength)

    for name in tables:
        rate = np.clip(tables[name], 1e-5, 1.0 - 1e-5)
        tables[name] = np.log(rate / (1.0 - rate)).astype(
            np.float32
        )

    return tables


def predict_hierarchical_bayes(split, tables):
    coefficients = {
        "video_id": 0.48,
        "author_id": 0.22,
        "tag": 0.12,
        "duration_bucket": 0.10,
        "tab": 0.08,
    }
    result = np.zeros(len(split.user_id), dtype=np.float64)
    for name, coefficient in coefficients.items():
        ids = np.asarray(split.X[name], dtype=np.int64)
        result += coefficient * tables[name][ids]
    return result


bayes_tables = fit_hierarchical_bayes(train, train_y)
bayes_valid = predict_hierarchical_bayes(valid, bayes_tables)


# ---------------------------------------------------------------------
# Validation comparison. Each family is scored alone and in rank space
# against the trusted incumbent.
# ---------------------------------------------------------------------

raw_valid = {
    "sampled_softmax": softmax_valid,
    "positive_nmf": nmf_valid,
    "hierarchical_bayes": bayes_valid,
}

candidate_scores = {}
candidate_arrays = {}
candidate_spec = {}

inc_metric = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
candidate_arrays["trusted_incumbent"] = inc_valid
candidate_spec["trusted_incumbent"] = ("incumbent", 0.0)

for family, scores in raw_valid.items():
    metric = evaluate(valid_users, valid_y, scores)
    raw_name = family + "_raw"
    candidate_scores[raw_name] = float(metric["primary"])
    candidate_arrays[raw_name] = scores
    candidate_spec[raw_name] = (family, 1.0)

    for alpha in (0.05, 0.10, 0.20, 0.35, 0.50, 0.70):
        blended = rank_blend(
            valid_users, inc_valid, scores, alpha
        )
        name = "%s_incblend_%.2f" % (family, alpha)
        metric = evaluate(valid_users, valid_y, blended)
        candidate_scores[name] = float(metric["primary"])
        candidate_arrays[name] = blended
        candidate_spec[name] = (family, alpha)

# Also test a structurally diverse equal-rank ensemble before incumbent blend.
cross_family_valid = (
    0.40 * within_user_rank(valid_users, softmax_valid)
    + 0.35 * within_user_rank(valid_users, nmf_valid)
    + 0.25 * within_user_rank(valid_users, bayes_valid)
)
raw_valid["cross_family"] = cross_family_valid
cross_metric = evaluate(valid_users, valid_y, cross_family_valid)
candidate_scores["cross_family_raw"] = float(cross_metric["primary"])
candidate_arrays["cross_family_raw"] = cross_family_valid
candidate_spec["cross_family_raw"] = ("cross_family", 1.0)

for alpha in (0.05, 0.10, 0.20, 0.35, 0.50):
    blended = rank_blend(
        valid_users, inc_valid, cross_family_valid, alpha
    )
    name = "cross_family_incblend_%.2f" % alpha
    metric = evaluate(valid_users, valid_y, blended)
    candidate_scores[name] = float(metric["primary"])
    candidate_arrays[name] = blended
    candidate_spec[name] = ("cross_family", alpha)

winner_name = max(candidate_scores, key=candidate_scores.get)
winner_valid = candidate_arrays[winner_name]
winner_metrics = evaluate(valid_users, valid_y, winner_valid)

raw_names = [
    name for name in candidate_scores
    if name.endswith("_raw")
]
best_raw_name = max(raw_names, key=lambda x: candidate_scores[x])
best_raw_valid = candidate_arrays[best_raw_name]

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: round(score, 6)
            for name, score in sorted(candidate_scores.items())
        },
        sort_keys=True,
    )
)
print(
    "FINDINGS winner=%s best_raw=%s incumbent=%.6f"
    % (
        winner_name,
        best_raw_name,
        candidate_scores["trusted_incumbent"],
    )
)


# ---------------------------------------------------------------------
# Test inference occurs only after validation selection.
# ---------------------------------------------------------------------

test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
inc_test = np.load(inc_test_path).astype(np.float64)

softmax_test = predict_sampled_softmax(test, softmax_model)
nmf_test = predict_nmf(test, nmf_user, nmf_item)
bayes_test = predict_hierarchical_bayes(test, bayes_tables)

raw_test = {
    "sampled_softmax": softmax_test,
    "positive_nmf": nmf_test,
    "hierarchical_bayes": bayes_test,
}
raw_test["cross_family"] = (
    0.40 * within_user_rank(test_users, softmax_test)
    + 0.35 * within_user_rank(test_users, nmf_test)
    + 0.25 * within_user_rank(test_users, bayes_test)
)

winner_family, winner_alpha = candidate_spec[winner_name]
if winner_family == "incumbent":
    winner_test = inc_test
elif winner_alpha >= 0.999:
    winner_test = raw_test[winner_family]
else:
    winner_test = rank_blend(
        test_users,
        inc_test,
        raw_test[winner_family],
        winner_alpha,
    )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(winner_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(winner_test, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(winner_metrics["primary"]),
            "gauc": float(winner_metrics["gauc"]),
            "ndcg@5": float(winner_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)