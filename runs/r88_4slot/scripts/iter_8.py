import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 19427
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
EMBED_DIM = 8
BATCH_SIZE = 8192
LR = 0.002
CHECKPOINTS = (2, 4)
MAX_EPOCHS = max(CHECKPOINTS)
HALF_LIFE = 7.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)

pair_i, pair_j = np.triu_indices(n_fields, k=1)
pair_i_t = torch.as_tensor(pair_i, dtype=torch.long)
pair_j_t = torch.as_tensor(pair_j, dtype=torch.long)
n_pairs = len(pair_i)


def encode(split):
    return np.stack(
        [
            np.asarray(split.X[name], dtype=np.int64) + offsets[j]
            for j, name in enumerate(FIELDS)
        ],
        axis=1,
    )


def recency_weights(dates, half_life=HALF_LIFE):
    dates = np.asarray(dates)
    _, inverse = np.unique(dates, return_inverse=True)
    age = inverse.max() - inverse
    weights = np.exp2(-age.astype(np.float32) / float(half_life))
    weights /= max(float(weights.mean()), 1e-7)
    return weights.astype(np.float32)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids)
    values = np.asarray(scores)
    n = len(values)

    order = np.lexsort((np.arange(n, dtype=np.int64), values, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    expanded_starts = np.repeat(starts, counts)
    expanded_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - expanded_starts

    sorted_ranks = np.full(n, 0.5, dtype=np.float64)
    mask = expanded_counts > 1
    sorted_ranks[mask] = (
        positions[mask] / (expanded_counts[mask].astype(np.float64) - 1.0)
    )

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


class FieldAwareFM(nn.Module):
    """Each feature has a different embedding for every partner field."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.ffm = nn.Embedding(
            total_cardinality, n_fields * EMBED_DIM
        )
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.ffm.weight, mean=0.0, std=0.025)

    def forward(self, x):
        batch = x.shape[0]
        linear = self.linear(x).sum(dim=1).squeeze(1) + self.bias

        e = self.ffm(x).reshape(
            batch, n_fields, n_fields, EMBED_DIM
        )
        left = e[:, pair_i_t, pair_j_t, :]
        right = e[:, pair_j_t, pair_i_t, :]
        interaction = (left * right).sum(dim=(1, 2))
        return linear + interaction


class AutoInt(nn.Module):
    """Self-attention learns data-dependent feature interaction weights."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.linear = nn.Embedding(total_cardinality, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.linear.weight)

        self.attn1 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, dropout=0.0, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.output = nn.Linear(n_fields * EMBED_DIM, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        e = self.embedding(x)

        attended, _ = self.attn1(
            e, e, e, need_weights=False
        )
        h = self.norm1(e + torch.relu(attended))

        attended, _ = self.attn2(
            h, h, h, need_weights=False
        )
        h = self.norm2(h + torch.relu(attended))

        first_order = self.linear(x).sum(dim=1).squeeze(1)
        interaction = self.output(h.reshape(x.shape[0], -1)).squeeze(1)
        return first_order + interaction + self.bias


class XDeepFM(nn.Module):
    """CIN explicitly forms bounded-degree vector-wise interactions."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.linear = nn.Embedding(total_cardinality, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.linear.weight)

        cin_sizes = (20, 20)
        self.cin_layers = nn.ModuleList()
        previous_channels = n_fields
        for size in cin_sizes:
            self.cin_layers.append(
                nn.Conv1d(
                    previous_channels * n_fields,
                    size,
                    kernel_size=1,
                )
            )
            previous_channels = size

        flat_dim = n_fields * EMBED_DIM
        self.deep = nn.Sequential(
            nn.Linear(flat_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(sum(cin_sizes) + 32, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        x0 = self.embedding(x)  # B,F,D
        h = x0
        cin_outputs = []

        for layer in self.cin_layers:
            # B,H,F,D -> B,(H*F),D
            outer = torch.einsum("bhd,bfd->bhfd", h, x0)
            outer = outer.reshape(
                x.shape[0], h.shape[1] * n_fields, EMBED_DIM
            )
            h = torch.relu(layer(outer))
            cin_outputs.append(h.sum(dim=2))

        cin = torch.cat(cin_outputs, dim=1)
        deep = self.deep(x0.reshape(x.shape[0], -1))
        first_order = self.linear(x).sum(dim=1).squeeze(1)
        interaction = self.output(torch.cat([cin, deep], dim=1)).squeeze(1)
        return first_order + interaction + self.bias


FAMILIES = {
    "field_aware_fm": FieldAwareFM,
    "autoint": AutoInt,
    "xdeepfm_cin": XDeepFM,
}


def predict(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    prediction_batch = BATCH_SIZE * 2

    with torch.inference_mode():
        for lo in range(0, len(x_np), prediction_batch):
            hi = min(lo + prediction_batch, len(x_np))
            xb = torch.from_numpy(x_np[lo:hi])
            result[lo:hi] = model(xb).cpu().numpy()

    return result


def train_select(
    model_class,
    x_train,
    y_train,
    dates_train,
    x_valid,
    y_valid,
    valid_users,
):
    torch.manual_seed(SEED)
    model = model_class()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-6,
    )

    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(
        y_train.astype(np.float32, copy=False)
    )
    w_tensor = torch.from_numpy(recency_weights(dates_train))
    n = len(x_train)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_epoch = None
    best_scores = None
    best_metrics = None
    best_primary = -np.inf

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=generator)

        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_tensor[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_tensor[idx],
                reduction="none",
            )
            loss = (losses * w_tensor[idx]).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if epoch in CHECKPOINTS:
            scores = predict(model, x_valid)
            metrics = evaluate(valid_users, y_valid, scores)
            primary = float(metrics["primary"])

            if primary > best_primary:
                best_primary = primary
                best_epoch = epoch
                best_scores = scores.copy()
                best_metrics = metrics

    return best_epoch, best_scores, best_metrics


def fit_fixed(model_class, x_fit, y_fit, dates_fit, epochs):
    torch.manual_seed(SEED)
    model = model_class()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-6,
    )

    x_tensor = torch.from_numpy(x_fit)
    y_tensor = torch.from_numpy(
        y_fit.astype(np.float32, copy=False)
    )
    w_tensor = torch.from_numpy(recency_weights(dates_fit))
    n = len(x_fit)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(int(epochs)):
        model.train()
        order = torch.randperm(n, generator=generator)

        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_tensor[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_tensor[idx],
                reduction="none",
            )
            loss = (losses * w_tensor[idx]).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


train = load("train")
valid = load("valid")

x_train = encode(train)
x_valid = encode(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
dates_train = np.asarray(train.date)
valid_users = np.asarray(valid.user_id)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)

if not os.path.exists(inc_valid_path):
    raise RuntimeError("Trusted incumbent validation scores unavailable")
if not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent test scores unavailable")

inc_valid = np.asarray(np.load(inc_valid_path))
inc_valid_rank = within_user_rank(valid_users, inc_valid)

family_results = {}
candidate_log = {}

for family_name, model_class in FAMILIES.items():
    epoch, raw_scores, standalone_metrics = train_select(
        model_class=model_class,
        x_train=x_train,
        y_train=y_train,
        dates_train=dates_train,
        x_valid=x_valid,
        y_valid=y_valid,
        valid_users=valid_users,
    )

    family_results[family_name] = {
        "epoch": int(epoch),
        "raw_scores": raw_scores,
        "standalone_metrics": standalone_metrics,
    }
    candidate_log[family_name + "_standalone"] = float(
        standalone_metrics["primary"]
    )
    candidate_log[family_name + "_epoch"] = int(epoch)


alphas = np.linspace(0.0, 1.0, 11)
best_primary = -np.inf
best_family = None
best_alpha = None
best_valid_scores = None
best_raw_scores = None
best_metrics = None

for family_name, result in family_results.items():
    model_rank = within_user_rank(
        valid_users, result["raw_scores"]
    )
    local_best_primary = -np.inf
    local_best_alpha = None

    for alpha in alphas:
        blended = (
            (1.0 - float(alpha)) * inc_valid_rank
            + float(alpha) * model_rank
        )
        metrics = evaluate(valid_users, y_valid, blended)
        primary = float(metrics["primary"])

        if primary > local_best_primary:
            local_best_primary = primary
            local_best_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_family = family_name
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()
            best_raw_scores = result["raw_scores"].copy()
            best_metrics = metrics

    candidate_log[family_name + "_best_blend"] = float(
        local_best_primary
    )
    candidate_log[family_name + "_blend_alpha"] = float(
        local_best_alpha
    )

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_raw_scores, dtype=np.float64),
    )


# Refit the validation-selected recipe on train plus validation.
x_fit = np.concatenate([x_train, x_valid], axis=0)
y_fit = np.concatenate(
    [
        y_train,
        np.asarray(valid.y, dtype=np.float32),
    ],
    axis=0,
)
dates_fit = np.concatenate(
    [
        dates_train,
        np.asarray(valid.date),
    ],
    axis=0,
)

selected_model = fit_fixed(
    model_class=FAMILIES[best_family],
    x_fit=x_fit,
    y_fit=y_fit,
    dates_fit=dates_fit,
    epochs=family_results[best_family]["epoch"],
)

test = load("test")
x_test = encode(test)
raw_test_scores = predict(selected_model, x_test)
test_users = np.asarray(test.user_id)

inc_test = np.asarray(np.load(inc_test_path))
inc_test_rank = within_user_rank(test_users, inc_test)
model_test_rank = within_user_rank(test_users, raw_test_scores)

test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * model_test_rank
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": best_family,
            "selected_epoch": int(
                family_results[best_family]["epoch"]
            ),
            "selected_model_weight": float(best_alpha),
            "recency_half_life_days": float(HALF_LIFE),
        },
        sort_keys=True,
    )
)

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(", ", ": "),
    )
)