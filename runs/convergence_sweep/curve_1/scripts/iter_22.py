import os
import time
import json
import random

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
THREADS = min(8, os.cpu_count() or 8)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    age = unique_dates.size - 1 - np.searchsorted(unique_dates, dates)
    weights = np.exp2(-age.astype(np.float64) / half_life)
    weights /= np.mean(weights)
    return weights.astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
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


def rank_blend(user_ids, left, right, alpha):
    return (
        (1.0 - alpha) * within_user_rank(user_ids, left)
        + alpha * within_user_rank(user_ids, right)
    )


class ContextualTensorFactorization(nn.Module):
    """
    Two CP-style three-way tensors:
      user x video x tab
      user x author x duration_bucket

    Additive video/author/tag biases provide a stable population backoff,
    while the tensor products form predictions differently from an FM's
    sum of pairwise interactions.
    """

    def __init__(self, rank=20):
        super().__init__()
        self.user = nn.Embedding(FEATURE_CARDINALITIES["user_id"], rank)
        self.video = nn.Embedding(FEATURE_CARDINALITIES["video_id"], rank)
        self.author = nn.Embedding(FEATURE_CARDINALITIES["author_id"], rank)
        self.tab = nn.Embedding(FEATURE_CARDINALITIES["tab"], rank)
        self.duration = nn.Embedding(
            FEATURE_CARDINALITIES["duration_bucket"], rank
        )

        self.video_bias = nn.Embedding(
            FEATURE_CARDINALITIES["video_id"], 1
        )
        self.author_bias = nn.Embedding(
            FEATURE_CARDINALITIES["author_id"], 1
        )
        self.tag_bias = nn.Embedding(FEATURE_CARDINALITIES["tag"], 1)
        self.duration_bias = nn.Embedding(
            FEATURE_CARDINALITIES["duration_bucket"], 1
        )
        self.tab_bias = nn.Embedding(FEATURE_CARDINALITIES["tab"], 1)
        self.global_bias = nn.Parameter(torch.zeros(()))

        for module in (
            self.user,
            self.video,
            self.author,
            self.tab,
            self.duration,
        ):
            nn.init.normal_(module.weight, std=0.08)

        for module in (
            self.video_bias,
            self.author_bias,
            self.tag_bias,
            self.duration_bias,
            self.tab_bias,
        ):
            nn.init.zeros_(module.weight)

    def forward(self, user, video, author, tab, duration, tag):
        u = self.user(user)
        v = self.video(video)
        a = self.author(author)
        t = self.tab(tab)
        d = self.duration(duration)

        video_context = torch.sum(u * v * t, dim=1)
        author_context = torch.sum(u * a * d, dim=1)

        additive = (
            self.video_bias(video).squeeze(1)
            + 0.65 * self.author_bias(author).squeeze(1)
            + 0.35 * self.tag_bias(tag).squeeze(1)
            + 0.45 * self.duration_bias(duration).squeeze(1)
            + 0.30 * self.tab_bias(tab).squeeze(1)
        )

        return self.global_bias + video_context + author_context + additive


TENSOR_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
]


def fit_tensor_model(train, labels, weights):
    arrays = {
        field: torch.from_numpy(
            np.asarray(train.X[field], dtype=np.int64)
        )
        for field in TENSOR_FIELDS
    }
    target = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    row_weights = torch.from_numpy(np.asarray(weights, dtype=np.float32))

    model = ContextualTensorFactorization(rank=20)
    prevalence = float(np.mean(labels))
    model.global_bias.data.fill_(
        np.log(prevalence / max(1.0 - prevalence, 1e-8))
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.012, weight_decay=2e-5
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + 11)

    n = len(labels)
    batch_size = 65536

    for epoch in range(5):
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0
        total_weight = 0.0

        model.train()
        for start in range(0, n, batch_size):
            selected = permutation[start:start + batch_size]

            logits = model(
                arrays["user_id"][selected],
                arrays["video_id"][selected],
                arrays["author_id"][selected],
                arrays["tab"][selected],
                arrays["duration_bucket"][selected],
                arrays["tag"][selected],
            )
            y = target[selected]
            w = row_weights[selected]

            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = torch.sum(losses * w) / torch.sum(w)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(torch.sum(losses * w))
            total_weight += float(torch.sum(w))

        print(
            "FINDINGS tensor_epoch=%d weighted_logloss=%.6f"
            % (epoch + 1, total_loss / max(total_weight, 1e-12))
        )

    return model


def predict_tensor(model, split, batch_size=131072):
    arrays = {
        field: np.asarray(split.X[field], dtype=np.int64)
        for field in TENSOR_FIELDS
    }
    n = len(split.user_id)
    result = np.empty(n, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            logits = model(
                torch.from_numpy(arrays["user_id"][start:end]),
                torch.from_numpy(arrays["video_id"][start:end]),
                torch.from_numpy(arrays["author_id"][start:end]),
                torch.from_numpy(arrays["tab"][start:end]),
                torch.from_numpy(arrays["duration_bucket"][start:end]),
                torch.from_numpy(arrays["tag"][start:end]),
            )
            result[start:end] = logits.numpy().astype(np.float64)

    return result


def make_binary_history(train, labels, positive):
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)

    keep = labels == (1 if positive else 0)
    data = np.ones(np.sum(keep), dtype=np.float32)

    matrix = sp.coo_matrix(
        (data, (users[keep], videos[keep])),
        shape=(
            FEATURE_CARDINALITIES["user_id"],
            FEATURE_CARDINALITIES["video_id"],
        ),
        dtype=np.float32,
    ).tocsr()
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


def build_item_knn(positive_history, top_k=80, shrinkage=6.0):
    counts = (positive_history.T @ positive_history).tocsr()
    counts.setdiag(0.0)
    counts.eliminate_zeros()

    popularity = np.asarray(
        positive_history.sum(axis=0)
    ).reshape(-1).astype(np.float64)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(popularity, 1.0))

    row_parts = []
    col_parts = []
    value_parts = []

    for item in range(counts.shape[0]):
        start, end = counts.indptr[item], counts.indptr[item + 1]
        columns = counts.indices[start:end]
        values = counts.data[start:end].astype(np.float64, copy=False)

        if values.size == 0:
            continue

        similarities = (
            values
            * inv_sqrt[item]
            * inv_sqrt[columns]
            * (values / (values + shrinkage))
        )

        if similarities.size > top_k:
            chosen = np.argpartition(similarities, -top_k)[-top_k:]
            columns = columns[chosen]
            similarities = similarities[chosen]

        row_parts.append(np.full(columns.size, item, dtype=np.int32))
        col_parts.append(columns.astype(np.int32, copy=False))
        value_parts.append(similarities.astype(np.float32))

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    values = np.concatenate(value_parts)

    similarity = sp.coo_matrix(
        (values, (rows, cols)),
        shape=counts.shape,
        dtype=np.float32,
    ).tocsr()

    print(
        "FINDINGS item_knn_edges=%d mean_degree=%.2f"
        % (
            similarity.nnz,
            similarity.nnz / max(similarity.shape[0], 1),
        )
    )
    return similarity


def predict_sparse_history(split, history, similarity, batch_users=512):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    result = np.zeros(users.size, dtype=np.float64)

    unique_users, inverse = np.unique(users, return_inverse=True)
    valid_user_limit = history.shape[0]

    for start in range(0, unique_users.size, batch_users):
        end = min(start + batch_users, unique_users.size)
        batch = unique_users[start:end]
        safe = batch < valid_user_limit

        dense_scores = np.zeros(
            (batch.size, similarity.shape[1]), dtype=np.float32
        )
        if np.any(safe):
            product = history[batch[safe]] @ similarity
            dense_scores[safe] = product.toarray()

        mask = (inverse >= start) & (inverse < end)
        local_users = inverse[mask] - start
        result[mask] = dense_scores[local_users, videos[mask]]

    return result


def fit_duration_density(train, labels, n_bins=32, prior=12.0):
    users = np.asarray(train.user_id, dtype=np.int64)
    duration = np.asarray(train.num["duration_ms"], dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)

    finite = np.isfinite(duration)
    fallback = float(np.nanmedian(duration))
    duration = np.where(finite, duration, fallback)
    transformed = np.log1p(np.maximum(duration, 0.0))

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(transformed, quantiles))
    bins = np.searchsorted(edges, transformed, side="right")
    actual_bins = edges.size + 1

    n_users = FEATURE_CARDINALITIES["user_id"]
    flat = users * actual_bins + bins

    positive = np.bincount(
        flat,
        weights=(labels == 1).astype(np.float64),
        minlength=n_users * actual_bins,
    ).reshape(n_users, actual_bins)
    negative = np.bincount(
        flat,
        weights=(labels == 0).astype(np.float64),
        minlength=n_users * actual_bins,
    ).reshape(n_users, actual_bins)

    global_positive = positive.sum(axis=0) + 1.0
    global_negative = negative.sum(axis=0) + 1.0
    global_positive /= global_positive.sum()
    global_negative /= global_negative.sum()

    positive_total = positive.sum(axis=1, keepdims=True)
    negative_total = negative.sum(axis=1, keepdims=True)

    positive_probability = (
        positive + prior * global_positive[None, :]
    ) / (positive_total + prior)
    negative_probability = (
        negative + prior * global_negative[None, :]
    ) / (negative_total + prior)

    table = np.log(
        np.maximum(positive_probability, 1e-9)
        / np.maximum(negative_probability, 1e-9)
    ).astype(np.float32)

    return edges, fallback, table


def predict_duration_density(split, edges, fallback, table):
    users = np.asarray(split.user_id, dtype=np.int64)
    duration = np.asarray(split.num["duration_ms"], dtype=np.float64)
    duration = np.where(np.isfinite(duration), duration, fallback)
    transformed = np.log1p(np.maximum(duration, 0.0))
    bins = np.searchsorted(edges, transformed, side="right")

    safe_users = np.minimum(users, table.shape[0] - 1)
    safe_bins = np.minimum(bins, table.shape[1] - 1)
    return table[safe_users, safe_bins].astype(np.float64)


train = load("train")
valid = load("valid")

train_labels = np.asarray(train.y, dtype=np.int8)
valid_labels = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
train_weights = recency_weights(train.date, half_life=4.0)

# Family 1: supervised contextual CP tensor.
tensor_model = fit_tensor_model(
    train, train_labels, train_weights
)
tensor_valid = predict_tensor(tensor_model, valid)

# Family 2: item-item collaborative filtering from positive and negative
# histories. The contrastive score penalizes candidates resembling a user's
# explicitly unsuccessful historical exposures.
positive_history = make_binary_history(
    train, train_labels, positive=True
)
negative_history = make_binary_history(
    train, train_labels, positive=False
)
item_similarity = build_item_knn(
    positive_history, top_k=80, shrinkage=6.0
)

knn_positive_valid = predict_sparse_history(
    valid, positive_history, item_similarity
)
knn_negative_valid = predict_sparse_history(
    valid, negative_history, item_similarity
)
knn_valid = knn_positive_valid - 0.30 * knn_negative_valid

# Family 3: personalized generative density ratio over raw duration.
duration_edges, duration_fallback, duration_table = fit_duration_density(
    train, train_labels, n_bins=32, prior=12.0
)
duration_valid = predict_duration_density(
    valid, duration_edges, duration_fallback, duration_table
)

own_ensemble_valid = (
    0.50 * within_user_rank(valid_users, tensor_valid)
    + 0.35 * within_user_rank(valid_users, knn_valid)
    + 0.15 * within_user_rank(valid_users, duration_valid)
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

candidate_scores = {}
candidate_metrics = {}
candidate_specs = {}
candidate_raw = {}


def register(name, scores, spec, raw):
    scores = np.asarray(scores, dtype=np.float64)
    candidate_scores[name] = scores
    candidate_metrics[name] = evaluate(
        valid_users, valid_labels, scores
    )
    candidate_specs[name] = spec
    candidate_raw[name] = np.asarray(raw, dtype=np.float64)


register(
    "incumbent",
    incumbent_valid,
    ("incumbent",),
    own_ensemble_valid,
)
register(
    "contextual_tensor",
    tensor_valid,
    ("tensor",),
    tensor_valid,
)
register(
    "contrastive_item_knn",
    knn_valid,
    ("knn",),
    knn_valid,
)
register(
    "personalized_duration_density",
    duration_valid,
    ("duration",),
    duration_valid,
)
register(
    "own_structural_ensemble",
    own_ensemble_valid,
    ("own_ensemble",),
    own_ensemble_valid,
)

families = {
    "tensor": tensor_valid,
    "knn": knn_valid,
    "duration": duration_valid,
    "own_ensemble": own_ensemble_valid,
}

for family_name, family_scores in families.items():
    for alpha in (0.05, 0.10, 0.20, 0.35, 0.50, 0.70):
        name = "%s_incumbent_blend_%.2f" % (
            family_name, alpha
        )
        blended = rank_blend(
            valid_users,
            incumbent_valid,
            family_scores,
            alpha,
        )
        register(
            name,
            blended,
            ("blend", family_name, alpha),
            family_scores,
        )

winner = max(
    candidate_metrics,
    key=lambda name: candidate_metrics[name]["primary"],
)
winner_spec = candidate_specs[winner]
valid_scores = candidate_scores[winner]
metrics = candidate_metrics[winner]

compact = {
    name: round(float(result["primary"]), 6)
    for name, result in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(compact, sort_keys=True))
print(
    "FINDINGS winner=%s spec=%s tensor=%.6f knn=%.6f duration=%.6f own_ensemble=%.6f incumbent=%.6f"
    % (
        winner,
        repr(winner_spec),
        candidate_metrics["contextual_tensor"]["primary"],
        candidate_metrics["contrastive_item_knn"]["primary"],
        candidate_metrics["personalized_duration_density"]["primary"],
        candidate_metrics["own_structural_ensemble"]["primary"],
        candidate_metrics["incumbent"]["primary"],
    )
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_spec[0] in ("incumbent", "blend"):
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
        )

test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)

needed_family = None
if winner_spec[0] == "tensor":
    needed_family = "tensor"
elif winner_spec[0] == "knn":
    needed_family = "knn"
elif winner_spec[0] == "duration":
    needed_family = "duration"
elif winner_spec[0] == "own_ensemble":
    needed_family = "own_ensemble"
elif winner_spec[0] == "blend":
    needed_family = winner_spec[1]

tensor_test = None
knn_test = None
duration_test = None
own_ensemble_test = None


def get_family_test(name):
    global tensor_test, knn_test, duration_test, own_ensemble_test

    if name == "tensor":
        if tensor_test is None:
            tensor_test = predict_tensor(tensor_model, test)
        return tensor_test

    if name == "knn":
        if knn_test is None:
            positive = predict_sparse_history(
                test, positive_history, item_similarity
            )
            negative = predict_sparse_history(
                test, negative_history, item_similarity
            )
            knn_test = positive - 0.30 * negative
        return knn_test

    if name == "duration":
        if duration_test is None:
            duration_test = predict_duration_density(
                test,
                duration_edges,
                duration_fallback,
                duration_table,
            )
        return duration_test

    if name == "own_ensemble":
        if own_ensemble_test is None:
            t = get_family_test("tensor")
            k = get_family_test("knn")
            d = get_family_test("duration")
            own_ensemble_test = (
                0.50 * within_user_rank(test_users, t)
                + 0.35 * within_user_rank(test_users, k)
                + 0.15 * within_user_rank(test_users, d)
            )
        return own_ensemble_test

    raise ValueError("Unknown family: %s" % name)


if winner_spec[0] == "incumbent":
    test_scores = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)
elif winner_spec[0] in ("tensor", "knn", "duration", "own_ensemble"):
    test_scores = get_family_test(winner_spec[0])
elif winner_spec[0] == "blend":
    _, family_name, alpha = winner_spec
    incumbent_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)
    family_test = get_family_test(family_name)
    test_scores = rank_blend(
        test_users, incumbent_test, family_test, alpha
    )
else:
    raise RuntimeError("Unrecognized winning specification")

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