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
SEED = 8675309
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "hour",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_matrix(split):
    cols = []
    for name, card, offset in zip(FIELDS, cards, offsets):
        values = np.asarray(split.X[name], dtype=np.int64)
        if values.size and (values.min() < 0 or values.max() >= card):
            raise ValueError("Out-of-range category for " + name)
        cols.append(values + offset)
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int64)


x_train = make_matrix(train)
x_valid = make_matrix(valid)
x_test = make_matrix(test)

dates = np.asarray(train.date, dtype=np.int64)
sample_weight = np.exp2(
    (dates - dates.max()).astype(np.float32) / 4.0
)
sample_weight /= sample_weight.mean()
sample_weight = sample_weight.astype(np.float32)

rng = np.random.default_rng(SEED)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ordered_ranks = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ordered_ranks[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ordered_ranks
    return ranks


def train_binary(model, epochs, learning_rate):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    n = len(y_train)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)

        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:lo + BATCH_SIZE]
            xb = torch.from_numpy(x_train[idx])
            target = torch.from_numpy(y_train[idx])
            weights = torch.from_numpy(sample_weight[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


@torch.inference_mode()
def predict_torch(model, matrix):
    model.eval()
    result = np.empty(len(matrix), dtype=np.float64)

    for lo in range(0, len(matrix), PRED_BATCH_SIZE):
        hi = min(lo + PRED_BATCH_SIZE, len(matrix))
        xb = torch.from_numpy(matrix[lo:hi])
        result[lo:hi] = (
            model(xb).detach().cpu().numpy().astype(np.float64)
        )

    return result


class ProductNetwork(nn.Module):
    """
    A PNN-style predictor. It exposes all pairwise embedding inner products
    directly to the nonlinear tower instead of requiring an MLP to discover
    multiplicative category interactions from concatenated embeddings.
    """

    def __init__(self, n_features, n_fields, emb_dim=12):
        super().__init__()
        self.n_fields = n_fields
        self.embedding = nn.Embedding(n_features, emb_dim)
        self.linear = nn.Embedding(n_features, 1)
        self.pairs = [
            (i, j)
            for i in range(n_fields)
            for j in range(i + 1, n_fields)
        ]

        input_dim = n_fields * emb_dim + len(self.pairs)
        self.tower = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        emb = self.embedding(x)
        products = [
            (emb[:, i, :] * emb[:, j, :]).sum(dim=1, keepdim=True)
            for i, j in self.pairs
        ]
        product_features = torch.cat(products, dim=1)
        deep_input = torch.cat(
            [emb.flatten(1), product_features], dim=1
        )
        wide = self.linear(x).squeeze(-1).sum(dim=1)
        return self.bias + wide + self.tower(deep_input).squeeze(1)


class MultiRelationalMF(nn.Module):
    """
    A latent collaborative model rather than a generic FM. The user vector
    separately matches video, author and tag representations, while side
    fields enter only through stable additive biases.
    """

    def __init__(self, n_features, rank=24):
        super().__init__()
        self.user_embedding = nn.Embedding(cards[0], rank)
        self.video_embedding = nn.Embedding(cards[1], rank)
        self.author_embedding = nn.Embedding(cards[2], rank)
        self.tag_embedding = nn.Embedding(cards[4], rank)

        self.field_bias = nn.Embedding(n_features, 1)
        self.user_bias = nn.Embedding(cards[0], 1)
        self.video_bias = nn.Embedding(cards[1], 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        for emb in [
            self.user_embedding,
            self.video_embedding,
            self.author_embedding,
            self.tag_embedding,
        ]:
            nn.init.normal_(emb.weight, std=0.025)

        nn.init.zeros_(self.field_bias.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.video_bias.weight)

    def forward(self, x):
        user = x[:, 0] - offsets[0]
        video = x[:, 1] - offsets[1]
        author = x[:, 2] - offsets[2]
        tag = x[:, 4] - offsets[4]

        u = self.user_embedding(user)
        video_match = (u * self.video_embedding(video)).sum(dim=1)
        author_match = (u * self.author_embedding(author)).sum(dim=1)
        tag_match = (u * self.tag_embedding(tag)).sum(dim=1)

        side_bias = self.field_bias(x).squeeze(-1).sum(dim=1)
        identity_bias = (
            self.user_bias(user).squeeze(1)
            + self.video_bias(video).squeeze(1)
        )

        return (
            self.global_bias
            + identity_bias
            + side_bias
            + video_match
            + 0.65 * author_match
            + 0.35 * tag_match
        )


def fit_naive_bayes():
    """
    Recency-weighted categorical Naive Bayes. The score is a sum of
    class-conditional log likelihood ratios, structurally different from
    embedding interaction models.
    """
    positive_weight = sample_weight * y_train
    negative_weight = sample_weight * (1.0 - y_train)

    pos_total = float(positive_weight.sum())
    neg_total = float(negative_weight.sum())
    prior_log_odds = np.log(
        (pos_total + 1.0) / (neg_total + 1.0)
    )

    tables = []
    smoothing = 2.0

    for field_index, card in enumerate(cards):
        local_ids = x_train[:, field_index] - offsets[field_index]

        pos_counts = np.bincount(
            local_ids,
            weights=positive_weight,
            minlength=card,
        ).astype(np.float64)
        neg_counts = np.bincount(
            local_ids,
            weights=negative_weight,
            minlength=card,
        ).astype(np.float64)

        pos_prob = (
            pos_counts + smoothing
        ) / (pos_total + smoothing * card)
        neg_prob = (
            neg_counts + smoothing
        ) / (neg_total + smoothing * card)

        tables.append(np.log(pos_prob) - np.log(neg_prob))

    return prior_log_odds, tables


def predict_naive_bayes(matrix, fitted):
    prior_log_odds, tables = fitted
    scores = np.full(len(matrix), prior_log_odds, dtype=np.float64)

    # Identity fields receive slightly less weight because their temporal
    # target rates are less stationary than tab/tag/duration context.
    field_weights = [0.55, 0.90, 0.85, 1.15, 1.10, 1.05, 0.75]

    for j, (table, field_weight) in enumerate(
        zip(tables, field_weights)
    ):
        local_ids = matrix[:, j] - offsets[j]
        scores += field_weight * table[local_ids]

    return scores


valid_predictions = {}
test_predictions = {}

torch.manual_seed(SEED + 1)
pnn = ProductNetwork(
    total_cardinality, len(FIELDS), emb_dim=12
)
train_binary(pnn, epochs=3, learning_rate=0.0014)
valid_predictions["product_network"] = predict_torch(pnn, x_valid)
test_predictions["product_network"] = predict_torch(pnn, x_test)
del pnn

torch.manual_seed(SEED + 2)
latent = MultiRelationalMF(total_cardinality, rank=24)
train_binary(latent, epochs=4, learning_rate=0.0013)
valid_predictions["multi_relational_mf"] = predict_torch(
    latent, x_valid
)
test_predictions["multi_relational_mf"] = predict_torch(
    latent, x_test
)
del latent

naive_bayes = fit_naive_bayes()
valid_predictions["categorical_naive_bayes"] = predict_naive_bayes(
    x_valid, naive_bayes
)
test_predictions["categorical_naive_bayes"] = predict_naive_bayes(
    x_test, naive_bayes
)

shared_dir = os.environ.get("SHARED_ARTIFACTS")
if not shared_dir:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared_dir, "incumbent_test_scores.npy")
).astype(np.float64)

if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_scores = {}
candidate_valid = {}
candidate_test = {}
candidate_raw = {}
candidate_blended = {}

correlations = {}
blend_alphas = [0.10, 0.20, 0.35, 0.50]

for family in valid_predictions:
    raw_valid = valid_predictions[family]
    raw_test = test_predictions[family]

    raw_metrics = evaluate(valid.user_id, valid.y, raw_valid)
    candidate_scores[family] = float(raw_metrics["primary"])
    candidate_valid[family] = raw_valid
    candidate_test[family] = raw_test
    candidate_raw[family] = raw_valid
    candidate_blended[family] = False

    valid_rank = within_user_rank(valid.user_id, raw_valid)
    test_rank = within_user_rank(test.user_id, raw_test)

    if np.std(valid_rank) > 0 and np.std(inc_valid_rank) > 0:
        correlations[family] = float(
            np.corrcoef(valid_rank, inc_valid_rank)[0, 1]
        )
    else:
        correlations[family] = 0.0

    for alpha in blend_alphas:
        name = family + "_blend_" + str(alpha)
        blend_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * valid_rank
        )
        blend_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * test_rank
        )

        metrics = evaluate(
            valid.user_id, valid.y, blend_valid
        )
        candidate_scores[name] = float(metrics["primary"])
        candidate_valid[name] = blend_valid
        candidate_test[name] = blend_test
        candidate_raw[name] = raw_valid
        candidate_blended[name] = True

winner = max(candidate_scores, key=candidate_scores.get)
winner_valid = candidate_valid[winner]
winner_test = candidate_test[winner]
winner_metrics = evaluate(valid.user_id, valid.y, winner_valid)

print(
    "FINDINGS "
    + json.dumps(
        {
            "within_user_rank_correlation_with_incumbent": correlations,
            "winner": winner,
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(candidate_scores, sort_keys=True)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(winner_test, dtype=np.float64),
    )

    if candidate_blended[winner]:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
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