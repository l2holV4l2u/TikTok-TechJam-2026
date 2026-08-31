import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 27183
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 8))

DEVICE = torch.device("cpu")
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "hour",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
CARDS = [int(FEATURE_CARDINALITIES[x]) for x in FIELDS]
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 16384
HALF_LIFE_DAYS = 4.0
EPOCHS = 2


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    first = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((np.array([-1]), end_positions)))
    row_sizes = np.repeat(sizes, sizes)

    position = np.arange(n, dtype=np.int64) - first
    ranked = (position.astype(np.float64) + 0.5) / row_sizes
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def make_categorical(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64) for name in FIELDS
    ])


def fit_numeric_transform(train):
    columns = []
    centers = []
    scales = []
    for name in NUM_FIELDS:
        raw = np.asarray(train.num[name], dtype=np.float64)
        transformed = np.log1p(np.maximum(np.nan_to_num(raw, nan=0.0), 0.0))
        center = float(np.median(transformed))
        q25, q75 = np.percentile(transformed, [25.0, 75.0])
        scale = float(max(q75 - q25, 0.25))
        columns.append(transformed)
        centers.append(center)
        scales.append(scale)
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


def make_numeric(split, centers, scales):
    result = np.empty((len(split), len(NUM_FIELDS)), dtype=np.float32)
    for j, name in enumerate(NUM_FIELDS):
        raw = np.asarray(split.num[name], dtype=np.float64)
        value = np.log1p(np.maximum(np.nan_to_num(raw, nan=0.0), 0.0))
        value = (value - float(centers[j])) / float(scales[j])
        result[:, j] = np.clip(value, -8.0, 8.0).astype(np.float32)
    return result


class FieldAwareFM(nn.Module):
    """Each source field has a different embedding when paired with each target field."""

    def __init__(self, cards, n_numeric, dim=6):
        super().__init__()
        self.n_fields = len(cards)
        self.dim = dim
        self.interaction_embeddings = nn.ModuleList([
            nn.Embedding(card, self.n_fields * dim) for card in cards
        ])
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(card, 1) for card in cards
        ])
        self.numeric = nn.Linear(n_numeric, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.interaction_embeddings:
            nn.init.normal_(emb.weight, std=0.025)
        for emb in self.linear_embeddings:
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.numeric.weight)

    def forward(self, xcat, xnum):
        batch = xcat.shape[0]
        field_views = [
            self.interaction_embeddings[i](xcat[:, i]).view(
                batch, self.n_fields, self.dim
            )
            for i in range(self.n_fields)
        ]
        z = torch.stack(field_views, dim=1)

        linear = self.bias.expand(batch)
        for i, emb in enumerate(self.linear_embeddings):
            linear = linear + emb(xcat[:, i]).squeeze(-1)
        linear = linear + self.numeric(xnum).squeeze(-1)

        interactions = torch.zeros(batch, device=xcat.device)
        for i in range(self.n_fields):
            for j in range(i + 1, self.n_fields):
                interactions = interactions + (
                    z[:, i, j, :] * z[:, j, i, :]
                ).sum(dim=-1)
        return linear + interactions


class ProductNetwork(nn.Module):
    """A PNN forms explicit pairwise inner products and learns nonlinear combinations."""

    def __init__(self, cards, n_numeric, dim=8):
        super().__init__()
        self.n_fields = len(cards)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card in cards
        ])
        pair_count = self.n_fields * (self.n_fields - 1) // 2
        input_dim = self.n_fields * dim + pair_count + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.025)

    def forward(self, xcat, xnum):
        z = torch.stack([
            emb(xcat[:, i]) for i, emb in enumerate(self.embeddings)
        ], dim=1)
        products = []
        for i in range(self.n_fields):
            for j in range(i + 1, self.n_fields):
                products.append((z[:, i, :] * z[:, j, :]).sum(dim=-1))
        products = torch.stack(products, dim=1)
        features = torch.cat([
            z.flatten(1),
            products,
            xnum,
        ], dim=1)
        return self.mlp(features).squeeze(-1)


class FeatureRecalibratedBilinear(nn.Module):
    """
    FiBiNET-style squeeze/excitation recalibrates whole feature fields before
    preserving vector-valued bilinear interactions.
    """

    def __init__(self, cards, n_numeric, dim=8):
        super().__init__()
        self.n_fields = len(cards)
        self.dim = dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card in cards
        ])
        hidden = max(4, self.n_fields // 2)
        self.excitation = nn.Sequential(
            nn.Linear(self.n_fields, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_fields),
            nn.Sigmoid(),
        )
        pair_count = self.n_fields * (self.n_fields - 1) // 2
        input_dim = self.n_fields * dim + pair_count * dim + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.025)

    def forward(self, xcat, xnum):
        z = torch.stack([
            emb(xcat[:, i]) for i, emb in enumerate(self.embeddings)
        ], dim=1)
        squeeze = z.mean(dim=-1)
        gates = 0.5 + self.excitation(squeeze)
        rz = z * gates.unsqueeze(-1)

        bilinear = []
        for i in range(self.n_fields):
            for j in range(i + 1, self.n_fields):
                bilinear.append(rz[:, i, :] * rz[:, j, :])
        bilinear = torch.cat(bilinear, dim=1)
        features = torch.cat([
            rz.flatten(1),
            bilinear,
            xnum,
        ], dim=1)
        return self.mlp(features).squeeze(-1)


def train_model(model, xcat, xnum, labels, sample_weights, epochs=EPOCHS):
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=2e-6
    )

    n = len(labels)
    xcat_t = torch.from_numpy(xcat)
    xnum_t = torch.from_numpy(xnum)
    y_t = torch.from_numpy(labels.astype(np.float32, copy=False))
    w_t = torch.from_numpy(sample_weights.astype(np.float32, copy=False))

    epoch_losses = []
    generator = torch.Generator()
    generator.manual_seed(SEED)

    for epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        weighted_loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb_cat = xcat_t[idx].to(DEVICE)
            xb_num = xnum_t[idx].to(DEVICE)
            yb = y_t[idx].to(DEVICE)
            wb = w_t[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_cat, xb_num)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.clamp(
                torch.sum(wb), min=1e-6
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            weighted_loss_sum += float(torch.sum(losses * wb).detach())
            weight_sum += float(torch.sum(wb))

        epoch_losses.append(weighted_loss_sum / max(weight_sum, 1e-12))

    del optimizer, permutation
    gc.collect()
    return epoch_losses


@torch.no_grad()
def predict_model(model, xcat, xnum):
    model.eval()
    xcat_t = torch.from_numpy(xcat)
    xnum_t = torch.from_numpy(xnum)
    output = np.empty(len(xcat), dtype=np.float64)

    for start in range(0, len(xcat), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(xcat))
        logits = model(
            xcat_t[start:end].to(DEVICE),
            xnum_t[start:end].to(DEVICE),
        )
        output[start:end] = logits.cpu().numpy().astype(np.float64)
    return output


train = load("train")
valid = load("valid")
test = load("test")

centers, scales = fit_numeric_transform(train)
train_cat = make_categorical(train)
valid_cat = make_categorical(valid)
test_cat = make_categorical(test)
train_num = make_numeric(train, centers, scales)
valid_num = make_numeric(valid, centers, scales)
test_num = make_numeric(test, centers, scales)

train_labels = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int32)
age = np.maximum(int(train_dates.max()) - train_dates, 0).astype(np.float32)
train_weights = np.power(0.5, age / HALF_LIFE_DAYS).astype(np.float32)
train_weights /= max(float(train_weights.mean()), 1e-8)

family_constructors = {
    "recency_ffm": lambda: FieldAwareFM(
        CARDS, len(NUM_FIELDS), dim=6
    ),
    "recency_pnn": lambda: ProductNetwork(
        CARDS, len(NUM_FIELDS), dim=8
    ),
    "recency_fibinet": lambda: FeatureRecalibratedBilinear(
        CARDS, len(NUM_FIELDS), dim=8
    ),
}

raw_valid = {}
raw_test = {}
training_losses = {}
parameter_counts = {}

for family_index, (name, constructor) in enumerate(family_constructors.items()):
    torch.manual_seed(SEED + family_index * 101)
    model = constructor()
    parameter_counts[name] = int(sum(
        p.numel() for p in model.parameters()
    ))

    training_losses[name] = train_model(
        model,
        train_cat,
        train_num,
        train_labels,
        train_weights,
    )
    raw_valid[name] = predict_model(model, valid_cat, valid_num)
    raw_test[name] = predict_model(model, test_cat, test_num)

    del model
    gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

if len(inc_valid) != len(valid) or len(inc_test) != len(test):
    raise RuntimeError("Trusted incumbent prediction lengths do not match")

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

family_valid_ranks = {
    name: rank_percentile(valid.user_id, values)
    for name, values in raw_valid.items()
}
family_test_ranks = {
    name: rank_percentile(test.user_id, values)
    for name, values in raw_test.items()
}

candidate_valid = {"incumbent": inc_valid}
candidate_test = {"incumbent": inc_test}
candidate_raw_source = {"incumbent": inc_valid}

for name in raw_valid:
    candidate_valid[name + "_standalone"] = raw_valid[name]
    candidate_test[name + "_standalone"] = raw_test[name]
    candidate_raw_source[name + "_standalone"] = raw_valid[name]

    for alpha in (0.05, 0.10, 0.20, 0.35, 0.50, 0.75):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * family_valid_ranks[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * family_test_ranks[name]
        )
        candidate_raw_source[key] = raw_valid[name]

# A cross-family rank ensemble tests whether field-aware memorization,
# scalar product interactions, and recalibrated vector interactions make
# sufficiently different errors to complement one another.
ensemble_specs = {
    "ffm_pnn_equal": {
        "recency_ffm": 0.50,
        "recency_pnn": 0.50,
    },
    "pnn_fibinet_equal": {
        "recency_pnn": 0.50,
        "recency_fibinet": 0.50,
    },
    "three_family_equal": {
        "recency_ffm": 1.0 / 3.0,
        "recency_pnn": 1.0 / 3.0,
        "recency_fibinet": 1.0 / 3.0,
    },
}

for ensemble_name, weights in ensemble_specs.items():
    ensemble_valid = np.zeros(len(valid), dtype=np.float64)
    ensemble_test = np.zeros(len(test), dtype=np.float64)
    for family, weight in weights.items():
        ensemble_valid += weight * family_valid_ranks[family]
        ensemble_test += weight * family_test_ranks[family]

    candidate_valid[ensemble_name + "_standalone"] = ensemble_valid
    candidate_test[ensemble_name + "_standalone"] = ensemble_test
    candidate_raw_source[ensemble_name + "_standalone"] = ensemble_valid

    for alpha in (0.10, 0.20, 0.35, 0.50, 0.75):
        key = f"{ensemble_name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank + alpha * ensemble_valid
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank + alpha * ensemble_test
        )
        candidate_raw_source[key] = ensemble_valid

candidate_metrics = {}
for key, scores in candidate_valid.items():
    candidate_metrics[key] = evaluate(
        valid.user_id,
        valid.y,
        scores,
    )

best_key = max(
    candidate_metrics,
    key=lambda key: float(candidate_metrics[key]["primary"]),
)
best_valid = candidate_valid[best_key]
best_test = candidate_test[best_key]
best_metrics = candidate_metrics[best_key]

candidate_summary = {
    key: float(metrics["primary"])
    for key, metrics in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(candidate_summary, sort_keys=True))

correlations = {}
names = list(family_valid_ranks)
for i, left in enumerate(names):
    correlations[left + "__incumbent"] = float(np.corrcoef(
        family_valid_ranks[left], inc_valid_rank
    )[0, 1])
    for right in names[i + 1:]:
        correlations[left + "__" + right] = float(np.corrcoef(
            family_valid_ranks[left], family_valid_ranks[right]
        )[0, 1])

standalone_metrics = {
    name: {
        "primary": float(candidate_metrics[name + "_standalone"]["primary"]),
        "gauc": float(candidate_metrics[name + "_standalone"]["gauc"]),
        "ndcg@5": float(candidate_metrics[name + "_standalone"]["ndcg@5"]),
    }
    for name in raw_valid
}
print("FINDINGS " + json.dumps({
    "best_candidate": best_key,
    "half_life_days": HALF_LIFE_DAYS,
    "epochs": EPOCHS,
    "parameter_counts": parameter_counts,
    "training_losses": training_losses,
    "standalone_metrics": standalone_metrics,
    "within_user_rank_correlations": correlations,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_key != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw_source[best_key], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))