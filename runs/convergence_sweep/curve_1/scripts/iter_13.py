import os
import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 8))

LIKELIHOOD_FIELDS = [
    "video_id", "author_id", "tab", "duration_bucket", "tag",
    "music_type", "upload_type", "video_type", "hour",
    "user_active_degree", "is_live_streamer", "is_video_author",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_bucket",
    "onehot_feat0", "onehot_feat1", "onehot_feat2",
    "onehot_feat3", "onehot_feat4", "onehot_feat6",
    "onehot_feat7", "onehot_feat8", "onehot_feat9",
    "onehot_feat10", "onehot_feat11", "onehot_feat12",
    "onehot_feat16",
]

ROCCHIO_FIELDS = [
    "video_id", "author_id", "duration_bucket", "tag", "music_type",
    "upload_type", "video_type", "onehot_feat0", "onehot_feat1",
    "onehot_feat2", "onehot_feat3", "onehot_feat4", "onehot_feat6",
    "onehot_feat7", "onehot_feat8", "onehot_feat12",
]

RANK_INDIVIDUAL_FIELDS = [
    "video_id", "author_id", "tab", "duration_bucket", "tag",
    "music_type", "upload_type", "hour", "onehot_feat1",
    "onehot_feat3", "onehot_feat7", "onehot_feat8",
]

RANK_CROSSES = [
    ("user_id", "video_id"),
    ("user_id", "author_id"),
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("tab", "video_id"),
    ("tab", "author_id"),
    ("user_active_degree", "video_id"),
    ("onehot_feat3", "author_id"),
]

ROCCHIO_DIM = 384
RANK_HASH_SIZE = 1 << 19
PRED_BATCH = 65536
TRAIN_BATCH = 32768


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    weights = np.exp2(-age.astype(np.float32) / np.float32(half_life))
    return (weights / weights.mean()).astype(np.float32)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)
    if n == 0:
        return values.copy()

    row_ids = np.arange(n, dtype=np.int64)
    order = np.lexsort((row_ids, values, users))
    ordered_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = ordered_users[1:] != ordered_users[:-1]
    starts = np.flatnonzero(starts_flag)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = positions[multi] / (repeated_lengths[multi] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


class CategoricalLikelihood:
    """
    Recency-weighted class-conditional categorical likelihood ratios.
    It contains no learned cross-feature representation and therefore forms
    predictions differently from FM/deep interaction models.
    """

    def __init__(self, fields, smoothing=18.0):
        self.fields = list(fields)
        self.smoothing = float(smoothing)
        self.tables = {}
        self.prior_log_odds = 0.0

    def fit(self, split, labels, weights):
        labels = np.asarray(labels, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        positive_total = float(np.sum(weights * labels))
        negative_total = float(np.sum(weights * (1.0 - labels)))
        self.prior_log_odds = np.log(
            (positive_total + 1.0) / (negative_total + 1.0)
        )

        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            cardinality = FEATURE_CARDINALITIES[field]
            pos = np.bincount(
                ids, weights=weights * labels, minlength=cardinality
            ).astype(np.float64)
            neg = np.bincount(
                ids, weights=weights * (1.0 - labels), minlength=cardinality
            ).astype(np.float64)

            # Symmetric Dirichlet smoothing in each class distribution.
            alpha = self.smoothing / max(cardinality, 1)
            log_ratio = (
                np.log(pos + alpha)
                - np.log(positive_total + self.smoothing)
                - np.log(neg + alpha)
                + np.log(negative_total + self.smoothing)
            )
            self.tables[field] = np.clip(log_ratio, -5.0, 5.0).astype(
                np.float32
            )
        return self

    def predict(self, split):
        score = np.full(
            len(split.user_id), self.prior_log_odds, dtype=np.float64
        )
        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            score += self.tables[field][ids]
        return score


def rocchio_hash(ids, namespace, dimension):
    ids = np.asarray(ids, dtype=np.uint64)
    salt = np.uint64(0x9E3779B1 * (namespace + 1))
    mixed = ids * np.uint64(0x85EBCA6B) + salt
    buckets = (mixed % np.uint64(dimension)).astype(np.int64)
    signs = np.where(
        ((mixed >> np.uint64(17)) & np.uint64(1)) == 0, 1.0, -1.0
    ).astype(np.float32)
    return buckets, signs


class SparseRocchio:
    """
    For each user, accumulates a signed hashed centroid of content attributes
    that occurred more often on positive than negative impressions.
    """

    def __init__(self, fields, dimension):
        self.fields = list(fields)
        self.dimension = int(dimension)
        self.preference = None

    def fit(self, split, labels, weights):
        users = np.asarray(split.X["user_id"], dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        n_users = FEATURE_CARDINALITIES["user_id"]

        global_rate = float(np.sum(weights * labels) / np.sum(weights))
        user_weight = np.bincount(
            users, weights=weights, minlength=n_users
        ).astype(np.float32)
        user_positive = np.bincount(
            users, weights=weights * labels, minlength=n_users
        ).astype(np.float32)
        user_rate = (
            user_positive + np.float32(6.0 * global_rate)
        ) / (user_weight + np.float32(6.0))

        residual = weights * (labels - user_rate[users])
        preference = np.zeros(
            (n_users, self.dimension), dtype=np.float32
        )

        for namespace, field in enumerate(self.fields):
            buckets, signs = rocchio_hash(
                split.X[field], namespace, self.dimension
            )
            np.add.at(
                preference,
                (users, buckets),
                residual * signs,
            )

        # Per-user scaling does not change ranking, but bounds numerical range.
        norms = np.sqrt(np.sum(preference * preference, axis=1))
        preference /= np.maximum(norms[:, None], 1.0)
        self.preference = preference
        return self

    def predict(self, split):
        users = np.asarray(split.X["user_id"], dtype=np.int64)
        score = np.zeros(len(users), dtype=np.float64)

        for namespace, field in enumerate(self.fields):
            buckets, signs = rocchio_hash(
                split.X[field], namespace, self.dimension
            )
            score += (
                self.preference[users, buckets].astype(np.float64)
                * signs.astype(np.float64)
            )
        return score


def rank_hash_tokens(split):
    tokens = []
    mask = np.uint64(RANK_HASH_SIZE - 1)

    for namespace, field in enumerate(RANK_INDIVIDUAL_FIELDS):
        values = np.asarray(split.X[field], dtype=np.uint64)
        mixed = (
            values * np.uint64(0x9E3779B185EBCA87)
            + np.uint64((namespace + 1) * 0xC2B2AE3D)
        )
        tokens.append((mixed & mask).astype(np.uint32))

    offset = len(RANK_INDIVIDUAL_FIELDS)
    for index, (left, right) in enumerate(RANK_CROSSES):
        a = np.asarray(split.X[left], dtype=np.uint64)
        b = np.asarray(split.X[right], dtype=np.uint64)
        mixed = (
            a * np.uint64(0xD6E8FEB86659FD93)
            + b * np.uint64(0xA5A3564E27F8862B)
            + np.uint64((offset + index + 1) * 0x9E3779B1)
        )
        tokens.append((mixed & mask).astype(np.uint32))

    return np.column_stack(tokens).astype(np.uint32, copy=False)


class HashedPairwiseRanker(nn.Module):
    def __init__(self, hash_size):
        super().__init__()
        self.embedding = nn.Embedding(hash_size, 1, sparse=True)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, token_ids):
        return self.embedding(token_ids).squeeze(-1).sum(dim=1)


def fit_pairwise_ranker(tokens, users, labels, rounds=4):
    model = HashedPairwiseRanker(RANK_HASH_SIZE)
    optimizer = torch.optim.SparseAdam(
        model.parameters(), lr=0.035, betas=(0.9, 0.99)
    )
    users = np.asarray(users, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    rng = np.random.RandomState(SEED + 77)

    for round_index in range(rounds):
        random_key = rng.randint(
            0, np.iinfo(np.int32).max, size=len(users), dtype=np.int32
        )
        order = np.lexsort((random_key, users))
        left = order[:-1]
        right = order[1:]
        keep = (
            (users[left] == users[right])
            & (labels[left] != labels[right])
        )
        left = left[keep]
        right = right[keep]

        shuffle = rng.permutation(len(left))
        left = left[shuffle]
        right = right[shuffle]

        total_loss = 0.0
        total_pairs = 0
        model.train()

        for start in range(0, len(left), TRAIN_BATCH):
            li = left[start:start + TRAIN_BATCH]
            ri = right[start:start + TRAIN_BATCH]

            left_tokens = torch.from_numpy(
                tokens[li].astype(np.int64, copy=False)
            )
            right_tokens = torch.from_numpy(
                tokens[ri].astype(np.int64, copy=False)
            )
            direction = torch.from_numpy(
                np.where(labels[li] > labels[ri], 1.0, -1.0).astype(
                    np.float32
                )
            )

            optimizer.zero_grad(set_to_none=True)
            difference = model(left_tokens) - model(right_tokens)

            # Smooth large-margin objective. Unlike pointwise BCE, only the
            # ordering of impressions belonging to the same user is trained.
            loss = F.softplus(1.0 - direction * difference).mean()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(li)
            total_pairs += len(li)

        print(
            "FINDINGS family=hashed_pairwise round={} pairs={} loss={:.6f}".format(
                round_index + 1,
                total_pairs,
                total_loss / max(total_pairs, 1),
            ),
            flush=True,
        )
    return model


@torch.no_grad()
def predict_pairwise(model, tokens):
    model.eval()
    result = np.empty(tokens.shape[0], dtype=np.float64)
    for start in range(0, tokens.shape[0], PRED_BATCH):
        end = min(start + PRED_BATCH, tokens.shape[0])
        token_tensor = torch.from_numpy(
            tokens[start:end].astype(np.int64, copy=False)
        )
        result[start:end] = model(token_tensor).numpy().astype(np.float64)
    return result


def metric(users, labels, scores):
    return evaluate(users, labels, scores)


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
weights = recency_weights(train.date, half_life=4.0)

# Family 1: generative categorical likelihood ratios.
likelihood = CategoricalLikelihood(
    LIKELIHOOD_FIELDS, smoothing=18.0
).fit(train, train_y, weights)
likelihood_valid = likelihood.predict(valid)

# Family 2: non-parametric personalized sparse content centroids.
rocchio = SparseRocchio(
    ROCCHIO_FIELDS, ROCCHIO_DIM
).fit(train, train_y, weights)
rocchio_valid = rocchio.predict(valid)

# Family 3: discriminative within-user pairwise large-margin learning.
train_rank_tokens = rank_hash_tokens(train)
valid_rank_tokens = rank_hash_tokens(valid)
pairwise_model = fit_pairwise_ranker(
    train_rank_tokens,
    np.asarray(train.user_id, dtype=np.int64),
    np.asarray(train.y, dtype=np.int8),
)
pairwise_valid = predict_pairwise(pairwise_model, valid_rank_tokens)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation prediction length mismatch")

valid_users = np.asarray(valid.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)

raw_predictions = {
    "categorical_likelihood": likelihood_valid,
    "sparse_rocchio": rocchio_valid,
    "hashed_pairwise": pairwise_valid,
}
valid_ranks = {
    name: within_user_rank(valid_users, prediction)
    for name, prediction in raw_predictions.items()
}
inc_rank = within_user_rank(valid_users, inc_valid)

candidate_scores = {"trusted_incumbent": inc_rank}
candidate_recipes = {
    "trusted_incumbent": ("incumbent", None, 0.0)
}

for name, rank_score in valid_ranks.items():
    candidate_scores[name] = rank_score
    candidate_recipes[name] = ("standalone", name, 1.0)

    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50, 0.65):
        candidate_name = "{}_inc_blend_{:.2f}".format(name, alpha)
        candidate_scores[candidate_name] = (
            (1.0 - alpha) * inc_rank + alpha * rank_score
        )
        candidate_recipes[candidate_name] = ("blend", name, alpha)

# Also test one equal-weight heterogeneous aggregate. It is fixed rather than
# tuned over a large weight grid.
family_average = (
    valid_ranks["categorical_likelihood"]
    + valid_ranks["sparse_rocchio"]
    + valid_ranks["hashed_pairwise"]
) / 3.0
candidate_scores["three_family_average"] = family_average
candidate_recipes["three_family_average"] = (
    "family_average", None, 1.0
)
for alpha in (0.15, 0.30, 0.45):
    name = "three_family_inc_blend_{:.2f}".format(alpha)
    candidate_scores[name] = (
        (1.0 - alpha) * inc_rank + alpha * family_average
    )
    candidate_recipes[name] = ("family_average_blend", None, alpha)

candidate_metrics = {}
for name, scores in candidate_scores.items():
    candidate_metrics[name] = metric(
        valid_users, valid_labels, scores
    )

candidate_primary = {
    name: float(values["primary"])
    for name, values in candidate_metrics.items()
}
winner_name = max(candidate_primary, key=candidate_primary.get)
winner_valid = candidate_scores[winner_name]
winner_metrics = candidate_metrics[winner_name]
winner_recipe = candidate_recipes[winner_name]

best_raw_name = max(
    raw_predictions,
    key=lambda name: candidate_primary[name],
)
best_raw_valid = raw_predictions[best_raw_name]

print(
    "FINDINGS standalone_best={} primary={:.6f}".format(
        best_raw_name, candidate_primary[best_raw_name]
    ),
    flush=True,
)
print(
    "FINDINGS selected={} recipe={}".format(
        winner_name, json.dumps(winner_recipe)
    ),
    flush=True,
)
print(
    "CANDIDATES {}".format(
        json.dumps(candidate_primary, sort_keys=True)
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(winner_valid, dtype=np.float64),
    )
    if winner_recipe[0] not in ("standalone",):
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

# Test is used only for feature-based inference. Its labels are never accessed.
test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

likelihood_test = likelihood.predict(test)
rocchio_test = rocchio.predict(test)
test_rank_tokens = rank_hash_tokens(test)
pairwise_test = predict_pairwise(pairwise_model, test_rank_tokens)

test_users = np.asarray(test.user_id, dtype=np.int64)
test_raw = {
    "categorical_likelihood": likelihood_test,
    "sparse_rocchio": rocchio_test,
    "hashed_pairwise": pairwise_test,
}
test_ranks = {
    name: within_user_rank(test_users, prediction)
    for name, prediction in test_raw.items()
}
inc_test_rank = within_user_rank(test_users, inc_test)

recipe_type, family_name, alpha = winner_recipe
if recipe_type == "incumbent":
    winner_test = inc_test_rank
elif recipe_type == "standalone":
    winner_test = test_ranks[family_name]
elif recipe_type == "blend":
    winner_test = (
        (1.0 - alpha) * inc_test_rank
        + alpha * test_ranks[family_name]
    )
elif recipe_type == "family_average":
    winner_test = (
        test_ranks["categorical_likelihood"]
        + test_ranks["sparse_rocchio"]
        + test_ranks["hashed_pairwise"]
    ) / 3.0
elif recipe_type == "family_average_blend":
    test_average = (
        test_ranks["categorical_likelihood"]
        + test_ranks["sparse_rocchio"]
        + test_ranks["hashed_pairwise"]
    ) / 3.0
    winner_test = (
        (1.0 - alpha) * inc_test_rank + alpha * test_average
    )
else:
    raise RuntimeError("Unknown selected recipe")

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(winner_test, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {{"primary": {:.10f}, "gauc": {:.10f}, '
    '"ndcg@5": {:.10f}, "gpu_seconds": {:.4f}}}'.format(
        float(winner_metrics["primary"]),
        float(winner_metrics["gauc"]),
        float(winner_metrics["ndcg@5"]),
        elapsed,
    )
)