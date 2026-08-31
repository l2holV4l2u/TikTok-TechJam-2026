import os
import time
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
FIELD_COUNT = len(FIELDS)
EMBED_DIM = 8
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
EPOCHS = 4
HALF_LIFE_DAYS = 4.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def make_offsets():
    offsets = []
    total = 0
    for name in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def encode(split):
    n = len(split.user_id)
    result = np.empty((n, FIELD_COUNT), dtype=np.int64)
    for j, name in enumerate(FIELDS):
        values = np.asarray(split.X[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        if values.size and (values.min() < 0 or values.max() >= card):
            raise ValueError("Out-of-range categorical value for " + name)
        result[:, j] = values + OFFSETS[j]
    return result


class SparseDenseModel(nn.Module):
    def sparse_parameters(self):
        raise NotImplementedError

    def dense_parameters(self):
        sparse_ids = {id(p) for p in self.sparse_parameters()}
        return [p for p in self.parameters() if id(p) not in sparse_ids]


class FieldAwareFM(SparseDenseModel):
    """Field-aware factors: each token has a different vector for each target field."""
    def __init__(self, cardinality, n_fields, rank):
        super().__init__()
        self.n_fields = n_fields
        self.rank = rank
        self.linear = nn.Embedding(cardinality, 1, sparse=True)
        self.factors = nn.Embedding(
            cardinality, n_fields * rank, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.025)

    def sparse_parameters(self):
        return [self.linear.weight, self.factors.weight]

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        field_vectors = []
        for i in range(self.n_fields):
            v = self.factors(x[:, i]).view(
                x.shape[0], self.n_fields, self.rank
            )
            field_vectors.append(v)

        interaction = torch.zeros(
            x.shape[0], dtype=torch.float32, device=x.device
        )
        for i in range(self.n_fields):
            for j in range(i + 1, self.n_fields):
                interaction = interaction + (
                    field_vectors[i][:, j, :] *
                    field_vectors[j][:, i, :]
                ).sum(dim=1)
        return self.bias + linear + interaction


class CrossNetwork(SparseDenseModel):
    """A wide linear term plus explicit bounded-degree DCN cross layers."""
    def __init__(self, cardinality, n_fields, rank, n_cross=3):
        super().__init__()
        self.n_fields = n_fields
        self.rank = rank
        width = n_fields * rank

        self.embedding = nn.Embedding(cardinality, rank, sparse=True)
        self.linear = nn.Embedding(cardinality, 1, sparse=True)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.linear.weight)

        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(width)) for _ in range(n_cross)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(width)) for _ in range(n_cross)
        ])
        for w in self.cross_w:
            nn.init.normal_(w, mean=0.0, std=0.015)

        self.deep = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(width + 32, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def sparse_parameters(self):
        return [self.embedding.weight, self.linear.weight]

    def forward(self, x):
        x0 = self.embedding(x).reshape(x.shape[0], -1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = (cross * w).sum(dim=1, keepdim=True)
            cross = x0 * scalar + b + cross
        deep = self.deep(x0)
        nonlinear = self.output(torch.cat([cross, deep], dim=1)).squeeze(-1)
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + wide + nonlinear


class AsymmetricLatentMF(SparseDenseModel):
    """
    User vector is matched only against a composed item-side vector.
    This differs from an FM because it does not form all field pairs.
    """
    def __init__(self, cardinality, rank):
        super().__init__()
        self.user_vec = nn.Embedding(cardinality, rank, sparse=True)
        self.video_vec = nn.Embedding(cardinality, rank, sparse=True)
        self.author_vec = nn.Embedding(cardinality, rank, sparse=True)
        self.context_vec = nn.Embedding(cardinality, rank, sparse=True)
        self.linear = nn.Embedding(cardinality, 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))

        for emb in [
            self.user_vec, self.video_vec, self.author_vec, self.context_vec
        ]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.linear.weight)

    def sparse_parameters(self):
        return [
            self.user_vec.weight,
            self.video_vec.weight,
            self.author_vec.weight,
            self.context_vec.weight,
            self.linear.weight,
        ]

    def forward(self, x):
        u = self.user_vec(x[:, 0])
        item = (
            self.video_vec(x[:, 1])
            + 0.7 * self.author_vec(x[:, 2])
            + 0.25 * self.context_vec(x[:, 3])
            + 0.25 * self.context_vec(x[:, 4])
        )
        affinity = (u * item).sum(dim=1) / math.sqrt(u.shape[1])
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + wide + affinity


def fit_model(model, x_train, y_train, weights, lr, seed):
    model.train()
    sparse_params = model.sparse_parameters()
    dense_params = model.dense_parameters()

    sparse_opt = torch.optim.SparseAdam(sparse_params, lr=lr)
    dense_opt = torch.optim.Adam(
        dense_params, lr=lr, weight_decay=1e-6
    ) if dense_params else None

    n = x_train.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed)

    for epoch in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = x_train[idx]
            yb = y_train[idx]
            wb = weights[idx]

            sparse_opt.zero_grad(set_to_none=True)
            if dense_opt is not None:
                dense_opt.zero_grad(set_to_none=True)

            logits = model(xb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (row_loss * wb).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(dense_params, 5.0)
            sparse_opt.step()
            if dense_opt is not None:
                dense_opt.step()

    return model


def predict(model, encoded):
    model.eval()
    result = np.empty(encoded.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, encoded.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, encoded.shape[0])
            xb = torch.from_numpy(encoded[start:end])
            result[start:end] = (
                model(xb).detach().cpu().numpy().astype(np.float64)
            )
    return result


def within_user_rank(user_ids, scores):
    """
    Scale-free percentile rank inside each logged impression set.
    Row index provides deterministic ordering for exact score ties.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = values.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, values, users))
    sorted_users = users[order]

    starts = np.empty(n, dtype=np.int64)
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]

    starts[new_group] = np.flatnonzero(new_group)
    starts[~new_group] = 0
    starts = np.maximum.accumulate(starts)

    group_start_positions = np.flatnonzero(new_group)
    group_ends = np.r_[group_start_positions[1:], n]
    counts = group_ends - group_start_positions
    sorted_counts = np.repeat(counts, counts)

    positions = np.arange(n, dtype=np.int64) - starts
    ranked_sorted = (positions.astype(np.float64) + 0.5) / sorted_counts

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


train = load("train")
valid = load("valid")

x_train_np = encode(train)
x_valid_np = encode(valid)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

train_dates = np.asarray(train.date, dtype=np.int64)
age_days = train_dates.max() - train_dates
recency = np.exp(
    -math.log(2.0) * age_days.astype(np.float64) / HALF_LIFE_DAYS
).astype(np.float32)
recency /= recency.mean()
train_weights = torch.from_numpy(recency)

models = {
    "recency_ffm": FieldAwareFM(
        TOTAL_CARDINALITY, FIELD_COUNT, EMBED_DIM
    ),
    "recency_dcn": CrossNetwork(
        TOTAL_CARDINALITY, FIELD_COUNT, EMBED_DIM, n_cross=3
    ),
    "recency_latent_mf": AsymmetricLatentMF(
        TOTAL_CARDINALITY, rank=16
    ),
}
learning_rates = {
    "recency_ffm": 0.0020,
    "recency_dcn": 0.0015,
    "recency_latent_mf": 0.0025,
}

valid_predictions = {}
candidate_scores = {}
candidate_specs = {}

for model_index, (name, model) in enumerate(models.items()):
    fit_model(
        model,
        x_train,
        y_train,
        train_weights,
        learning_rates[name],
        SEED + 101 * model_index,
    )
    pred = predict(model, x_valid_np)
    valid_predictions[name] = pred
    metric = evaluate(valid.user_id, valid.y, pred)
    candidate_scores[name] = float(metric["primary"])
    candidate_specs[name] = ("raw", name, 1.0)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not (os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.shape[0] != len(valid.user_id):
    raise ValueError("Incumbent validation prediction length mismatch")

inc_metric = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
candidate_specs["trusted_incumbent"] = ("incumbent", None, 0.0)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
blend_weights = [0.20, 0.35, 0.50, 0.65, 0.80]

for name, pred in valid_predictions.items():
    own_rank = within_user_rank(valid.user_id, pred)
    for alpha in blend_weights:
        blend = alpha * own_rank + (1.0 - alpha) * inc_valid_rank
        blend_name = "%s_rankblend_%.2f" % (name, alpha)
        metric = evaluate(valid.user_id, valid.y, blend)
        candidate_scores[blend_name] = float(metric["primary"])
        candidate_specs[blend_name] = ("blend", name, alpha)

winner = max(candidate_scores, key=candidate_scores.get)
winner_kind, winner_model_name, winner_alpha = candidate_specs[winner]

if winner_kind == "raw":
    valid_scores = valid_predictions[winner_model_name]
elif winner_kind == "blend":
    valid_scores = (
        winner_alpha * within_user_rank(
            valid.user_id, valid_predictions[winner_model_name]
        )
        + (1.0 - winner_alpha) * inc_valid_rank
    )
else:
    valid_scores = inc_valid.copy()

metrics = evaluate(valid.user_id, valid.y, valid_scores)

best_own_name = max(
    valid_predictions,
    key=lambda key: candidate_scores[key]
)

print("FINDINGS " + json.dumps({
    "half_life_days": HALF_LIFE_DAYS,
    "winner": winner,
    "best_standalone_family": best_own_name,
    "best_standalone_primary": candidate_scores[best_own_name],
    "incumbent_primary": candidate_scores["trusted_incumbent"],
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_kind in ("blend", "incumbent"):
        raw_name = winner_model_name if winner_model_name else best_own_name
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(valid_predictions[raw_name], dtype=np.float64),
        )

test = load("test")
x_test_np = encode(test)

test_predictions = {}
for name, model in models.items():
    test_predictions[name] = predict(model, x_test_np)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if inc_test.shape[0] != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

if winner_kind == "raw":
    test_scores = test_predictions[winner_model_name]
elif winner_kind == "blend":
    own_test_rank = within_user_rank(
        test.user_id, test_predictions[winner_model_name]
    )
    inc_test_rank = within_user_rank(test.user_id, inc_test)
    test_scores = (
        winner_alpha * own_test_rank
        + (1.0 - winner_alpha) * inc_test_rank
    )
else:
    test_scores = inc_test.copy()

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))