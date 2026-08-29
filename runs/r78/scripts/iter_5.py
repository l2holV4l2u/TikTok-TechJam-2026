import os
import time
import json
import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 8675309
RANK = 8
EPOCHS = 2
BATCH_SIZE = 8192
PRED_BATCH = 32768

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]
N_FIELDS = len(FIELDS)

OFFSETS = {}
TOTAL_CARDINALITY = 0
for field in FIELDS:
    OFFSETS[field] = TOTAL_CARDINALITY
    TOTAL_CARDINALITY += int(FEATURE_CARDINALITIES[field])


def make_matrix(split):
    n = len(split.user_id)
    result = np.empty((n, N_FIELDS), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        result[:, j] = (
            np.asarray(split.X[field], dtype=np.int64) + OFFSETS[field]
        )
    return result


def rank_transform(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    if n == 0:
        return scores.copy()

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    repeated_starts = np.repeat(starts, sizes)
    repeated_sizes = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ranked_sorted = np.where(
        repeated_sizes > 1,
        positions.astype(np.float64) /
        np.maximum(repeated_sizes.astype(np.float64) - 1.0, 1.0),
        0.5,
    )
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def initial_intercept(y):
    p = float(np.mean(y))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class FieldAwareFM(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.ffm = nn.Embedding(TOTAL_CARDINALITY, N_FIELDS * RANK)
        self.bias = nn.Parameter(torch.tensor(float(intercept)))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.ffm.weight, mean=0.0, std=0.025)

        pair_i = []
        pair_j = []
        for i in range(N_FIELDS):
            for j in range(i + 1, N_FIELDS):
                pair_i.append(i)
                pair_j.append(j)
        self.register_buffer("pair_i", torch.tensor(pair_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(pair_j, dtype=torch.long))

    def forward(self, x):
        linear_score = self.linear(x).sum(dim=1).squeeze(-1)

        all_ffm = self.ffm(x).reshape(
            x.shape[0], N_FIELDS, N_FIELDS, RANK
        )
        left = all_ffm[:, self.pair_i, self.pair_j, :]
        right = all_ffm[:, self.pair_j, self.pair_i, :]
        interaction = (left * right).sum(dim=(1, 2))

        return self.bias + linear_score + interaction


class XDeepFM(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, RANK)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.linear.weight)

        cin_widths = [16, 16]
        cin_layers = []
        previous_width = N_FIELDS
        for width in cin_widths:
            layer = nn.Linear(previous_width * N_FIELDS, width)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            cin_layers.append(layer)
            previous_width = width
        self.cin_layers = nn.ModuleList(cin_layers)

        deep_dim = N_FIELDS * RANK
        self.deep = nn.Sequential(
            nn.Linear(deep_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.cin_head = nn.Linear(sum(cin_widths), 1)
        self.bias = nn.Parameter(torch.tensor(float(intercept)))

    def forward(self, x):
        x0 = self.embedding(x)
        hidden = x0
        cin_outputs = []

        for layer in self.cin_layers:
            outer = torch.einsum("bhk,bfk->bkhf", hidden, x0)
            outer = outer.reshape(
                x.shape[0], RANK, hidden.shape[1] * N_FIELDS
            )
            hidden = torch.relu(layer(outer)).transpose(1, 2)
            cin_outputs.append(hidden.sum(dim=2))

        cin_vector = torch.cat(cin_outputs, dim=1)
        cin_score = self.cin_head(cin_vector).squeeze(1)
        deep_score = self.deep(x0.reshape(x.shape[0], -1)).squeeze(1)
        linear_score = self.linear(x).sum(dim=1).squeeze(-1)

        return self.bias + linear_score + cin_score + deep_score


class FiBiNET(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, RANK)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.linear.weight)

        squeeze_hidden = max(4, N_FIELDS // 2)
        self.se = nn.Sequential(
            nn.Linear(N_FIELDS, squeeze_hidden),
            nn.ReLU(),
            nn.Linear(squeeze_hidden, N_FIELDS),
            nn.Sigmoid(),
        )

        pair_i = []
        pair_j = []
        for i in range(N_FIELDS):
            for j in range(i + 1, N_FIELDS):
                pair_i.append(i)
                pair_j.append(j)
        n_pairs = len(pair_i)
        self.register_buffer("pair_i", torch.tensor(pair_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(pair_j, dtype=torch.long))

        identity = torch.eye(RANK).unsqueeze(0).repeat(n_pairs, 1, 1)
        identity += 0.01 * torch.randn_like(identity)
        self.bilinear = nn.Parameter(identity)

        interaction_dim = n_pairs * RANK
        self.interaction_net = nn.Sequential(
            nn.Linear(interaction_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.tensor(float(intercept)))

    def forward(self, x):
        embeddings = self.embedding(x)

        squeeze = embeddings.mean(dim=2)
        field_weights = self.se(squeeze)
        recalibrated = embeddings * field_weights.unsqueeze(2)

        left = recalibrated[:, self.pair_i, :]
        right = recalibrated[:, self.pair_j, :]
        transformed_left = torch.einsum(
            "bpk,pkd->bpd", left, self.bilinear
        )
        interactions = transformed_left * right

        interaction_score = self.interaction_net(
            interactions.reshape(x.shape[0], -1)
        ).squeeze(1)
        linear_score = self.linear(x).sum(dim=1).squeeze(-1)

        return self.bias + linear_score + interaction_score


def make_model(family, intercept):
    if family == "field_aware_fm":
        return FieldAwareFM(intercept)
    if family == "xdeepfm_cin":
        return XDeepFM(intercept)
    if family == "fibinet":
        return FiBiNET(intercept)
    raise ValueError(f"Unknown family: {family}")


def fit_model(family, split, labels, seed):
    torch.manual_seed(seed)
    x_np = make_matrix(split)
    y_np = np.asarray(labels, dtype=np.float32)

    x = torch.from_numpy(np.ascontiguousarray(x_np))
    y = torch.from_numpy(y_np)

    model = make_model(family, initial_intercept(y_np))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=1e-6
    )

    generator = torch.Generator()
    generator.manual_seed(seed + 137)
    n = x.shape[0]

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, split):
    model.eval()
    x_np = make_matrix(split)
    x = torch.from_numpy(np.ascontiguousarray(x_np))
    result = np.empty(x.shape[0], dtype=np.float64)

    for start in range(0, x.shape[0], PRED_BATCH):
        end = min(start + PRED_BATCH, x.shape[0])
        result[start:end] = (
            model(x[start:end]).cpu().numpy().astype(np.float64)
        )
    return result


class JoinedSplit:
    pass


def join_splits(a, b):
    result = JoinedSplit()
    result.X = {
        field: np.concatenate([
            np.asarray(a.X[field]),
            np.asarray(b.X[field]),
        ])
        for field in a.X
    }
    result.user_id = np.concatenate([
        np.asarray(a.user_id), np.asarray(b.user_id)
    ])
    result.video_id = np.concatenate([
        np.asarray(a.video_id), np.asarray(b.video_id)
    ])
    result.date = np.concatenate([
        np.asarray(a.date), np.asarray(b.date)
    ])
    result.time_ms = np.concatenate([
        np.asarray(a.time_ms), np.asarray(b.time_ms)
    ])
    return result


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)

artifact_dir = os.environ["RUN_ARTIFACTS"]
inc_valid = np.load(
    os.path.join(artifact_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_valid_rank = rank_transform(valid.user_id, inc_valid)

families = ["field_aware_fm", "xdeepfm_cin", "fibinet"]
blend_alphas = [0.15, 0.25, 0.40, 0.60, 0.80, 1.00]

candidate_scores = {}
best_primary = -np.inf
best_family = None
best_alpha = None
best_valid_scores = None

for family_index, family in enumerate(families):
    model = fit_model(
        family, train, train_y, SEED + 1000 * family_index
    )
    raw_scores = predict_model(model, valid)
    raw_rank = rank_transform(valid.user_id, raw_scores)

    raw_metric = evaluate(valid.user_id, valid_y, raw_scores)
    candidate_scores[family + "_standalone"] = float(
        raw_metric["primary"]
    )

    for alpha in blend_alphas:
        if alpha == 1.0:
            blended = raw_scores
            name = family + "_raw"
        else:
            blended = (
                alpha * raw_rank + (1.0 - alpha) * inc_valid_rank
            )
            name = family + "_blend_" + str(alpha)

        metric = evaluate(valid.user_id, valid_y, blended)
        primary = float(metric["primary"])
        candidate_scores[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_family = family
            best_alpha = alpha
            best_valid_scores = np.asarray(
                blended, dtype=np.float64
            ).copy()

    del model, raw_scores, raw_rank
    gc.collect()

final_metrics = evaluate(
    valid.user_id, valid_y, best_valid_scores
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Refit the selected recipe on all labels available before the test period.
test = load("test")
joined = join_splits(train, valid)
joined_y = np.concatenate([
    train_y, valid_y.astype(np.float32)
])

winner_index = families.index(best_family)
test_model = fit_model(
    best_family,
    joined,
    joined_y,
    SEED + 1000 * winner_index,
)
new_test_scores = predict_model(test_model, test)

if best_alpha == 1.0:
    final_test_scores = new_test_scores
else:
    incumbent_test = np.load(
        os.path.join(artifact_dir, "incumbent_test_scores.npy")
    ).astype(np.float64)
    new_test_rank = rank_transform(test.user_id, new_test_scores)
    incumbent_test_rank = rank_transform(test.user_id, incumbent_test)
    final_test_scores = (
        best_alpha * new_test_rank
        + (1.0 - best_alpha) * incumbent_test_rank
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(final_test_scores, dtype=np.float64),
    )

candidate_scores["SELECTED_" + best_family + "_alpha_" + str(best_alpha)] = (
    float(final_metrics["primary"])
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))