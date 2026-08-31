import os
import time
import json
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 8))

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
USER_INDEX = 0
VIDEO_INDEX = 1
AUTHOR_INDEX = 2
TAB_INDEX = 3
DURATION_INDEX = 4

DIM = 12
BATCH_SIZE = 16384
EPOCHS = 2
HASH_SIZE = 131071
DEVICE = torch.device("cpu")


def stack_fields(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64)
        for name in FIELDS
    ])


def recency_weights(dates, half_life=2.5):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    day_position = np.searchsorted(unique_dates, dates)
    age = unique_dates.size - 1 - day_position
    weights = np.exp2(-age.astype(np.float32) / np.float32(half_life))
    weights /= weights.mean()
    return weights.astype(np.float32)


def init_embedding(embedding, std=0.025):
    nn.init.normal_(embedding.weight, mean=0.0, std=std)


class HashedWideDeep(nn.Module):
    """
    The wide branch memorizes individual categories and typed pair crosses
    through a shared hash table. The deep branch generalizes through dense
    embeddings and nonlinear interactions.
    """

    def __init__(self, cardinalities):
        super().__init__()
        self.cardinalities = cardinalities
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, DIM)
            for cardinality in cardinalities
        ])
        self.linear = nn.ModuleList([
            nn.Embedding(cardinality, 1)
            for cardinality in cardinalities
        ])
        self.cross_linear = nn.Embedding(HASH_SIZE, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        self.deep = nn.Sequential(
            nn.Linear(len(cardinalities) * DIM, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

        for embedding in self.embeddings:
            init_embedding(embedding)
        for embedding in self.linear:
            init_embedding(embedding)
        init_embedding(self.cross_linear, std=0.01)

        self.pairs = [
            (left, right)
            for left in range(len(cardinalities))
            for right in range(left + 1, len(cardinalities))
        ]

    def forward(self, x):
        embedded = torch.cat([
            embedding(x[:, index])
            for index, embedding in enumerate(self.embeddings)
        ], dim=1)

        wide = self.bias.expand(x.shape[0])
        for index, embedding in enumerate(self.linear):
            wide = wide + embedding(x[:, index]).squeeze(1)

        for pair_number, (left, right) in enumerate(self.pairs):
            hashed = (
                x[:, left] * 1000003
                + x[:, right] * 9176
                + pair_number * 611953
            ) % HASH_SIZE
            wide = wide + self.cross_linear(hashed).squeeze(1)

        return wide + self.deep(embedded).squeeze(1)


class PairwiseTwoTower(nn.Module):
    """
    A context/user tower is matched to a video-side tower by a dot product.
    Its pairwise BPR training directly orders a user's observed positive row
    above one of that user's observed negative rows.
    """

    def __init__(self, cardinalities):
        super().__init__()
        self.user_embedding = nn.Embedding(cardinalities[USER_INDEX], DIM)
        self.tab_embedding = nn.Embedding(cardinalities[TAB_INDEX], DIM)
        self.video_embedding = nn.Embedding(cardinalities[VIDEO_INDEX], DIM)
        self.author_embedding = nn.Embedding(cardinalities[AUTHOR_INDEX], DIM)
        self.duration_embedding = nn.Embedding(
            cardinalities[DURATION_INDEX], DIM
        )
        self.linear = nn.ModuleList([
            nn.Embedding(cardinality, 1)
            for cardinality in cardinalities
        ])
        self.context_gate = nn.Sequential(
            nn.Linear(2 * DIM, DIM),
            nn.Tanh(),
        )
        self.item_gate = nn.Sequential(
            nn.Linear(3 * DIM, DIM),
            nn.Tanh(),
        )
        self.bias = nn.Parameter(torch.zeros(()))

        for embedding in [
            self.user_embedding,
            self.tab_embedding,
            self.video_embedding,
            self.author_embedding,
            self.duration_embedding,
        ]:
            init_embedding(embedding)
        for embedding in self.linear:
            init_embedding(embedding)

    def forward(self, x):
        context = self.context_gate(torch.cat([
            self.user_embedding(x[:, USER_INDEX]),
            self.tab_embedding(x[:, TAB_INDEX]),
        ], dim=1))
        item = self.item_gate(torch.cat([
            self.video_embedding(x[:, VIDEO_INDEX]),
            self.author_embedding(x[:, AUTHOR_INDEX]),
            self.duration_embedding(x[:, DURATION_INDEX]),
        ], dim=1))

        score = torch.sum(context * item, dim=1) / math.sqrt(DIM)
        score = score + self.bias
        for index, embedding in enumerate(self.linear):
            score = score + embedding(x[:, index]).squeeze(1)
        return score


def train_wide_deep(model, x, labels, weights):
    model.to(DEVICE)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    w_tensor = torch.from_numpy(weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0018, weight_decay=1e-6
    )
    rng = np.random.RandomState(SEED + 11)
    n = x.shape[0]

    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        weighted_loss = 0.0
        total_weight = 0.0

        model.train()
        for start in range(0, n, BATCH_SIZE):
            index = torch.from_numpy(
                permutation[start:start + BATCH_SIZE]
            )
            xb = x_tensor.index_select(0, index)
            yb = y_tensor.index_select(0, index)
            wb = w_tensor.index_select(0, index)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(row_loss * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            weighted_loss += float(torch.sum(row_loss * wb).detach())
            total_weight += float(torch.sum(wb))

        print(
            "FINDINGS family=wide_deep epoch={} weighted_loss={:.6f}".format(
                epoch + 1, weighted_loss / max(total_weight, 1e-8)
            ),
            flush=True,
        )
    return model


def prepare_pairwise_indices(x, labels):
    users = x[:, USER_INDEX]
    positive_indices = np.flatnonzero(labels == 1).astype(np.int64)
    negative_indices = np.flatnonzero(labels == 0).astype(np.int64)

    negative_order = np.argsort(
        users[negative_indices], kind="stable"
    )
    negative_indices = negative_indices[negative_order]
    negative_users = users[negative_indices]

    user_cardinality = FEATURE_CARDINALITIES["user_id"]
    negative_counts = np.bincount(
        negative_users, minlength=user_cardinality
    ).astype(np.int64)
    negative_starts = np.zeros(user_cardinality, dtype=np.int64)
    if user_cardinality > 1:
        negative_starts[1:] = np.cumsum(negative_counts[:-1])

    positive_users = users[positive_indices]
    eligible = negative_counts[positive_users] > 0
    return (
        positive_indices[eligible],
        negative_indices,
        negative_counts,
        negative_starts,
    )


def train_pairwise(model, x, labels, weights):
    (
        positive_indices,
        negative_indices,
        negative_counts,
        negative_starts,
    ) = prepare_pairwise_indices(x, labels)

    x_tensor = torch.from_numpy(x)
    weight_tensor = torch.from_numpy(weights)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.002, weight_decay=2e-6
    )
    rng = np.random.RandomState(SEED + 29)

    for epoch in range(EPOCHS):
        permutation = rng.permutation(positive_indices.size)
        epoch_loss = 0.0
        epoch_weight = 0.0
        model.train()

        for start in range(0, permutation.size, BATCH_SIZE):
            selected = positive_indices[
                permutation[start:start + BATCH_SIZE]
            ]
            selected_users = x[selected, USER_INDEX]
            counts = negative_counts[selected_users]

            random_offsets = (
                rng.random(selected.size) * counts
            ).astype(np.int64)
            selected_negative = negative_indices[
                negative_starts[selected_users] + random_offsets
            ]

            positive_tensor = torch.from_numpy(selected)
            negative_tensor = torch.from_numpy(selected_negative)
            xb_positive = x_tensor.index_select(0, positive_tensor)
            xb_negative = x_tensor.index_select(0, negative_tensor)
            wb = weight_tensor.index_select(0, positive_tensor)

            optimizer.zero_grad(set_to_none=True)
            positive_score = model(xb_positive)
            negative_score = model(xb_negative)
            row_loss = F.softplus(-(positive_score - negative_score))
            loss = torch.sum(row_loss * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += float(torch.sum(row_loss * wb).detach())
            epoch_weight += float(torch.sum(wb))

        print(
            "FINDINGS family=pairwise_two_tower epoch={} "
            "weighted_pair_loss={:.6f} pairs={}".format(
                epoch + 1,
                epoch_loss / max(epoch_weight, 1e-8),
                positive_indices.size,
            ),
            flush=True,
        )
    return model


@torch.no_grad()
def predict_torch(model, x):
    model.eval()
    x_tensor = torch.from_numpy(x)
    result = np.empty(x.shape[0], dtype=np.float64)
    prediction_batch = BATCH_SIZE * 2

    for start in range(0, x.shape[0], prediction_batch):
        end = min(start + prediction_batch, x.shape[0])
        result[start:end] = (
            model(x_tensor[start:end])
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    return result


class EmpiricalBayesCrosses:
    """
    Smoothed, recency-weighted target statistics for stable entity fields and
    hashed context crosses. This forms predictions non-parametrically rather
    than learning latent interactions.
    """

    def __init__(self, cardinalities, hash_size=HASH_SIZE):
        self.cardinalities = cardinalities
        self.hash_size = hash_size
        self.global_rate = None
        self.field_tables = []
        self.cross_tables = []
        self.cross_pairs = [
            (VIDEO_INDEX, TAB_INDEX),
            (AUTHOR_INDEX, TAB_INDEX),
            (VIDEO_INDEX, DURATION_INDEX),
            (AUTHOR_INDEX, DURATION_INDEX),
        ]

    @staticmethod
    def _smooth_logit(sums, counts, global_rate, prior):
        rates = (sums + prior * global_rate) / (counts + prior)
        rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
        return np.log(rates / (1.0 - rates)).astype(np.float32)

    def _hash(self, x, pair_number, left, right):
        return (
            x[:, left] * 1000003
            + x[:, right] * 9176
            + pair_number * 611953
        ) % self.hash_size

    def fit(self, x, labels, weights):
        labels = labels.astype(np.float64)
        weights64 = weights.astype(np.float64)
        self.global_rate = float(
            np.sum(labels * weights64) / np.sum(weights64)
        )

        self.field_tables = []
        priors = [100.0, 24.0, 32.0, 80.0, 60.0]
        for index, cardinality in enumerate(self.cardinalities):
            counts = np.bincount(
                x[:, index],
                weights=weights64,
                minlength=cardinality,
            )
            sums = np.bincount(
                x[:, index],
                weights=weights64 * labels,
                minlength=cardinality,
            )
            self.field_tables.append(
                self._smooth_logit(
                    sums, counts, self.global_rate, priors[index]
                )
            )

        self.cross_tables = []
        for pair_number, (left, right) in enumerate(self.cross_pairs):
            hashed = self._hash(x, pair_number, left, right)
            counts = np.bincount(
                hashed,
                weights=weights64,
                minlength=self.hash_size,
            )
            sums = np.bincount(
                hashed,
                weights=weights64 * labels,
                minlength=self.hash_size,
            )
            self.cross_tables.append(
                self._smooth_logit(
                    sums, counts, self.global_rate, prior=45.0
                )
            )
        return self

    def predict(self, x):
        base = math.log(
            self.global_rate / (1.0 - self.global_rate)
        )
        score = np.full(x.shape[0], base, dtype=np.float64)

        # User propensity is constant within most evaluation groups, so it is
        # downweighted; video, author, tab and duration drive the ranking.
        field_weights = [0.10, 0.95, 0.75, 0.45, 0.45]
        for index, coefficient in enumerate(field_weights):
            score += coefficient * (
                self.field_tables[index][x[:, index]] - base
            )

        cross_weights = [0.45, 0.35, 0.30, 0.25]
        for pair_number, ((left, right), coefficient) in enumerate(
            zip(self.cross_pairs, cross_weights)
        ):
            hashed = self._hash(x, pair_number, left, right)
            score += coefficient * (
                self.cross_tables[pair_number][hashed] - base
            )
        return score


def within_user_average_rank(user_ids, scores):
    """
    Percentile rank within each user, preserving exact ties by assigning their
    average rank. This makes blends insensitive to family-specific logit scale.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = users.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, values, users))
    sorted_users = users[order]
    sorted_values = values[order]

    user_start_flags = np.r_[
        True, sorted_users[1:] != sorted_users[:-1]
    ]
    user_starts = np.flatnonzero(user_start_flags)
    user_ends = np.r_[user_starts[1:], n]
    user_lengths = user_ends - user_starts

    row_user_starts = np.repeat(user_starts, user_lengths)
    row_user_lengths = np.repeat(user_lengths, user_lengths)

    tie_start_flags = np.r_[
        True,
        (sorted_users[1:] != sorted_users[:-1])
        | (sorted_values[1:] != sorted_values[:-1]),
    ]
    tie_starts = np.flatnonzero(tie_start_flags)
    tie_ends = np.r_[tie_starts[1:], n]
    tie_lengths = tie_ends - tie_starts
    tie_midpoints = (tie_starts + tie_ends - 1) / 2.0
    row_tie_midpoints = np.repeat(tie_midpoints, tie_lengths)

    sorted_rank = (
        row_tie_midpoints - row_user_starts
    ) / np.maximum(row_user_lengths - 1, 1)

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_rank
    return result


train = load("train")
valid = load("valid")

x_train = stack_fields(train)
x_valid = stack_fields(valid)
labels_train = np.asarray(train.y, dtype=np.int8)
labels_valid = np.asarray(valid.y, dtype=np.int8)
weights_train = recency_weights(train.date, half_life=2.5)
cardinalities = [FEATURE_CARDINALITIES[name] for name in FIELDS]

wide_deep = HashedWideDeep(cardinalities)
wide_deep = train_wide_deep(
    wide_deep, x_train, labels_train, weights_train
)
valid_wide = predict_torch(wide_deep, x_valid)

pairwise = PairwiseTwoTower(cardinalities)
pairwise = train_pairwise(
    pairwise, x_train, labels_train, weights_train
)
valid_pairwise = predict_torch(pairwise, x_valid)

empirical_bayes = EmpiricalBayesCrosses(cardinalities).fit(
    x_train, labels_train, weights_train
)
valid_eb = empirical_bayes.predict(x_valid)

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not os.path.exists(incumbent_valid_path):
    raise RuntimeError("Trusted incumbent validation scores are unavailable")

incumbent_valid = np.load(incumbent_valid_path).astype(np.float64)
incumbent_valid_rank = within_user_average_rank(
    valid.user_id, incumbent_valid
)

family_valid = {
    "wide_deep": valid_wide,
    "pairwise_two_tower": valid_pairwise,
    "empirical_bayes_crosses": valid_eb,
}

candidate_scores = {}
candidate_arrays = {}
candidate_raw_family = {}
candidate_alpha = {}
blend_alphas = [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]

incumbent_metrics = evaluate(
    valid.user_id, labels_valid, incumbent_valid
)
candidate_scores["trusted_incumbent"] = float(
    incumbent_metrics["primary"]
)
candidate_arrays["trusted_incumbent"] = incumbent_valid
candidate_raw_family["trusted_incumbent"] = "wide_deep"
candidate_alpha["trusted_incumbent"] = 0.0

for family_name, raw_scores in family_valid.items():
    raw_metrics = evaluate(valid.user_id, labels_valid, raw_scores)
    candidate_scores[family_name] = float(raw_metrics["primary"])
    candidate_arrays[family_name] = raw_scores
    candidate_raw_family[family_name] = family_name
    candidate_alpha[family_name] = 1.0

    family_rank = within_user_average_rank(valid.user_id, raw_scores)
    for alpha in blend_alphas:
        blend_name = "{}_blend_{:.2f}".format(family_name, alpha)
        blended = (
            (1.0 - alpha) * incumbent_valid_rank
            + alpha * family_rank
        )
        metrics = evaluate(valid.user_id, labels_valid, blended)
        candidate_scores[blend_name] = float(metrics["primary"])
        candidate_arrays[blend_name] = blended
        candidate_raw_family[blend_name] = family_name
        candidate_alpha[blend_name] = alpha

best_name = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_arrays[best_name]
best_metrics = evaluate(valid.user_id, labels_valid, valid_scores)

print(
    "CANDIDATES " + json.dumps(
        candidate_scores, sort_keys=True
    ),
    flush=True,
)
print(
    "FINDINGS selected={} family={} incumbent_weight={:.2f} "
    "new_family_weight={:.2f}".format(
        best_name,
        candidate_raw_family[best_name],
        1.0 - candidate_alpha[best_name],
        candidate_alpha[best_name],
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if best_name != candidate_raw_family[best_name]:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(
                family_valid[candidate_raw_family[best_name]],
                dtype=np.float64,
            ),
        )

test = load("test")
x_test = stack_fields(test)

test_family = {
    "wide_deep": predict_torch(wide_deep, x_test),
    "pairwise_two_tower": predict_torch(pairwise, x_test),
    "empirical_bayes_crosses": empirical_bayes.predict(x_test),
}

selected_family = candidate_raw_family[best_name]
alpha = candidate_alpha[best_name]

if best_name == "trusted_incumbent":
    if not os.path.exists(incumbent_test_path):
        raise RuntimeError("Trusted incumbent test scores are unavailable")
    test_scores = np.load(incumbent_test_path).astype(np.float64)
elif alpha >= 1.0:
    test_scores = test_family[selected_family]
else:
    if not os.path.exists(incumbent_test_path):
        raise RuntimeError("Trusted incumbent test scores are unavailable")
    incumbent_test = np.load(incumbent_test_path).astype(np.float64)
    incumbent_test_rank = within_user_average_rank(
        test.user_id, incumbent_test
    )
    family_test_rank = within_user_average_rank(
        test.user_id, test_family[selected_family]
    )
    test_scores = (
        (1.0 - alpha) * incumbent_test_rank
        + alpha * family_test_rank
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)