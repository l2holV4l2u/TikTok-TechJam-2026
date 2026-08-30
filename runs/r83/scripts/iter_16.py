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
SEED = 91827
BATCH_SIZE = 4096
PRED_BATCH = 8192
EPOCHS = 1
HALF_LIFE_DAYS = 7.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "hour",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
]
N_FIELDS = len(FIELDS)
FAMILIES = ["field_aware_fm", "fibinet", "nfm"]
FAMILY_SEEDS = {
    "field_aware_fm": 101,
    "fibinet": 307,
    "nfm": 509,
}


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64)
            for field in FIELDS
        ]),
        dtype=np.int64,
    )


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates).astype(np.float32)
    age = float(day_index.max()) - day_index
    weights = np.exp2(-age / HALF_LIFE_DAYS).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    position = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    denominator = np.repeat(np.maximum(lengths - 1, 1), lengths)

    ranked_sorted = position.astype(np.float64) / denominator
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def zscore(scores):
    scores = np.asarray(scores, dtype=np.float64)
    std = float(scores.std())
    if std < 1e-12:
        return np.zeros_like(scores)
    return (scores - float(scores.mean())) / std


class FirstOrder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.weights = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[field]), 1)
            for field in FIELDS
        ])
        for embedding in self.weights:
            nn.init.zeros_(embedding.weight)

    def forward(self, x):
        result = self.bias.expand(x.shape[0])
        for j, embedding in enumerate(self.weights):
            result = result + embedding(x[:, j]).squeeze(1)
        return result


class FieldAwareFM(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.dim = dim
        self.first = FirstOrder()
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[field]),
                N_FIELDS * dim,
            )
            for field in FIELDS
        ])
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.025)

        left, right = np.triu_indices(N_FIELDS, k=1)
        self.register_buffer(
            "left_index", torch.as_tensor(left, dtype=torch.long)
        )
        self.register_buffer(
            "right_index", torch.as_tensor(right, dtype=torch.long)
        )

    def forward(self, x):
        # values[:, source_field, target_field, latent_dimension]
        values = torch.stack([
            embedding(x[:, source]).view(
                x.shape[0], N_FIELDS, self.dim
            )
            for source, embedding in enumerate(self.embeddings)
        ], dim=1)

        left_vectors = values[
            :, self.left_index, self.right_index, :
        ]
        right_vectors = values[
            :, self.right_index, self.left_index, :
        ]
        interaction = (left_vectors * right_vectors).sum(dim=(1, 2))
        return self.first(x) + interaction


class FiBiNET(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim
        self.first = FirstOrder()
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[field]), dim)
            for field in FIELDS
        ])
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.025)

        hidden = max(4, N_FIELDS // 3)
        self.excitation = nn.Sequential(
            nn.Linear(N_FIELDS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_FIELDS),
            nn.Sigmoid(),
        )

        left, right = np.triu_indices(N_FIELDS, k=1)
        self.register_buffer(
            "left_index", torch.as_tensor(left, dtype=torch.long)
        )
        self.register_buffer(
            "right_index", torch.as_tensor(right, dtype=torch.long)
        )
        n_pairs = len(left)

        self.bilinear = nn.Parameter(
            torch.empty(n_pairs, dim, dim)
        )
        nn.init.xavier_uniform_(self.bilinear)

        self.network = nn.Sequential(
            nn.Linear(n_pairs * dim, 128),
            nn.SiLU(),
            nn.Linear(128, 40),
            nn.SiLU(),
            nn.Linear(40, 1),
        )

    def forward(self, x):
        embedded = torch.stack([
            embedding(x[:, j])
            for j, embedding in enumerate(self.embeddings)
        ], dim=1)

        squeeze = embedded.mean(dim=2)
        gates = self.excitation(squeeze).unsqueeze(2)
        recalibrated = embedded * (0.5 + gates)

        left = recalibrated[:, self.left_index, :]
        right = recalibrated[:, self.right_index, :]
        transformed = torch.einsum(
            "bpd,pde->bpe", left, self.bilinear
        )
        products = transformed * right
        deep_logit = self.network(
            products.flatten(start_dim=1)
        ).squeeze(1)
        return self.first(x) + deep_logit


class NFM(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.first = FirstOrder()
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[field]), dim)
            for field in FIELDS
        ])
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.025)

        self.norm = nn.LayerNorm(dim)
        self.network = nn.Sequential(
            nn.Linear(dim, 64),
            nn.SiLU(),
            nn.Linear(64, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x):
        embedded = torch.stack([
            embedding(x[:, j])
            for j, embedding in enumerate(self.embeddings)
        ], dim=1)

        summed = embedded.sum(dim=1)
        bi_interaction = 0.5 * (
            summed.square() - embedded.square().sum(dim=1)
        )
        deep_logit = self.network(
            self.norm(bi_interaction)
        ).squeeze(1)
        return self.first(x) + deep_logit


def construct_model(family):
    if family == "field_aware_fm":
        return FieldAwareFM()
    if family == "fibinet":
        return FiBiNET()
    if family == "nfm":
        return NFM()
    raise ValueError(family)


def fit_model(X, y, weights, family):
    seed = SEED + FAMILY_SEEDS[family]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = construct_model(family)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=3e-5,
    )

    xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.int64))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    wt = torch.from_numpy(np.asarray(weights, dtype=np.float32))

    for _ in range(EPOCHS):
        order = rng.permutation(len(X))
        model.train()

        for start in range(0, len(order), BATCH_SIZE):
            index_np = order[start:start + BATCH_SIZE]
            index = torch.from_numpy(index_np)

            logits = model(xt[index])
            batch_weights = wt[index]
            losses = F.binary_cross_entropy_with_logits(
                logits,
                yt[index],
                reduction="none",
            )
            loss = (
                losses * batch_weights
            ).sum() / batch_weights.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.inference_mode()
def predict_model(model, X):
    xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.int64))
    scores = np.empty(len(X), dtype=np.float32)
    model.eval()

    for start in range(0, len(X), PRED_BATCH):
        end = min(start + PRED_BATCH, len(X))
        scores[start:end] = model(xt[start:end]).cpu().numpy()

    return scores.astype(np.float64)


def combine_scores(raw, incumbent, users, mode, alpha):
    raw = np.asarray(raw, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)

    if mode == "raw":
        return raw
    if mode == "zblend":
        return (
            alpha * zscore(incumbent)
            + (1.0 - alpha) * zscore(raw)
        )
    if mode == "rankblend":
        return (
            alpha * within_user_rank(users, incumbent)
            + (1.0 - alpha) * within_user_rank(users, raw)
        )
    raise ValueError(mode)


train = load("train")
valid = load("valid")

X_train = make_matrix(train)
X_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
weights_train = recency_weights(train.date)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

candidate_predictions = {}
candidate_specs = {}
candidate_metrics = {}
raw_metrics = {}

for family in FAMILIES:
    model = fit_model(
        X_train,
        y_train,
        weights_train,
        family,
    )
    raw_scores = predict_model(model, X_valid)

    raw_name = family + "_raw"
    candidate_predictions[raw_name] = raw_scores
    candidate_specs[raw_name] = (family, "raw", 0.0)

    for alpha in (0.25, 0.50, 0.75):
        zname = family + "_zblend_inc%.2f" % alpha
        candidate_predictions[zname] = combine_scores(
            raw_scores,
            inc_valid,
            valid_users,
            "zblend",
            alpha,
        )
        candidate_specs[zname] = (
            family, "zblend", alpha
        )

        rname = family + "_rankblend_inc%.2f" % alpha
        candidate_predictions[rname] = combine_scores(
            raw_scores,
            inc_valid,
            valid_users,
            "rankblend",
            alpha,
        )
        candidate_specs[rname] = (
            family, "rankblend", alpha
        )

    del model

best_name = None
best_result = None

for name, scores in candidate_predictions.items():
    result = evaluate(valid_users, y_valid, scores)
    score = float(result["primary"])
    candidate_metrics[name] = score

    if name.endswith("_raw"):
        raw_metrics[name[:-4]] = score

    if (
        best_result is None
        or score > float(best_result["primary"])
    ):
        best_name = name
        best_result = result

valid_scores = np.asarray(
    candidate_predictions[best_name],
    dtype=np.float64,
)
winning_family, winning_mode, winning_alpha = (
    candidate_specs[best_name]
)

print("CANDIDATES " + json.dumps(
    candidate_metrics, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "winner": best_name,
    "raw_family_primary": raw_metrics,
    "winning_family": winning_family,
    "winning_mode": winning_mode,
    "winning_incumbent_weight": winning_alpha,
    "test_labels_read": False,
    "auxiliary_outcomes_read": False,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )

# Refit the exact selected family and training recipe on train+validation.
X_combined = np.concatenate([X_train, X_valid], axis=0)
y_combined = np.concatenate([
    y_train,
    np.asarray(valid.y, dtype=np.float32),
])
combined_dates = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])
combined_weights = recency_weights(combined_dates)

final_model = fit_model(
    X_combined,
    y_combined,
    combined_weights,
    winning_family,
)

test = load("test")
X_test = make_matrix(test)
raw_test_scores = predict_model(final_model, X_test)

inc_test = np.asarray(
    np.load(inc_test_path),
    dtype=np.float64,
)
test_scores = combine_scores(
    raw_test_scores,
    inc_test,
    np.asarray(test.user_id),
    winning_mode,
    winning_alpha,
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_result["primary"]),
    "gauc": float(best_result["gauc"]),
    "ndcg@5": float(best_result["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))