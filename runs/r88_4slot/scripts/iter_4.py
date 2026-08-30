import os
import time
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
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
CHECKPOINTS = (2, 4, 6)
MAX_EPOCHS = max(CHECKPOINTS)
LR = 0.002

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)
flat_dim = n_fields * EMBED_DIM

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


def recency_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    day_index = {int(d): i for i, d in enumerate(unique_dates)}
    idx = np.fromiter(
        (day_index[int(d)] for d in dates),
        dtype=np.int16,
        count=len(dates),
    )
    age = int(idx.max()) - idx.astype(np.float32)
    w = np.exp2(-age / float(half_life)).astype(np.float32)
    w /= max(float(w.mean()), 1e-6)
    return w


class BaseCTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)

    def embedded(self, x):
        return self.embedding(x)


class DCNCross(BaseCTR):
    def __init__(self):
        super().__init__()
        self.cross_w = nn.ParameterList(
            [nn.Parameter(torch.empty(flat_dim)) for _ in range(3)]
        )
        self.cross_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(flat_dim)) for _ in range(3)]
        )
        for w in self.cross_w:
            nn.init.normal_(w, mean=0.0, std=0.02)
        self.deep = nn.Sequential(
            nn.Linear(flat_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(flat_dim + 32, 1)

    def forward(self, x):
        x0 = self.embedded(x).reshape(x.shape[0], -1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scale = torch.sum(xl * w, dim=1, keepdim=True)
            xl = x0 * scale + b + xl
        deep = self.deep(x0)
        return self.output(torch.cat([xl, deep], dim=1)).squeeze(1)


class ProductNetwork(BaseCTR):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(flat_dim + n_pairs, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        e = self.embedded(x)
        products = (e[:, pair_i_t] * e[:, pair_j_t]).sum(dim=2)
        z = torch.cat([e.reshape(x.shape[0], -1), products], dim=1)
        return self.network(z).squeeze(1)


class FiBiNet(BaseCTR):
    def __init__(self):
        super().__init__()
        squeeze_dim = max(3, n_fields // 3)
        self.senet = nn.Sequential(
            nn.Linear(n_fields, squeeze_dim),
            nn.ReLU(),
            nn.Linear(squeeze_dim, n_fields),
            nn.Sigmoid(),
        )
        self.bilinear_scale = nn.Parameter(
            torch.ones(n_pairs, EMBED_DIM)
        )
        self.network = nn.Sequential(
            nn.Linear(flat_dim + n_pairs * EMBED_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        e = self.embedded(x)
        field_summary = e.mean(dim=2)
        gates = self.senet(field_summary).unsqueeze(2)
        weighted = e * gates
        interactions = (
            weighted[:, pair_i_t]
            * weighted[:, pair_j_t]
            * self.bilinear_scale.unsqueeze(0)
        )
        z = torch.cat(
            [
                weighted.reshape(x.shape[0], -1),
                interactions.reshape(x.shape[0], -1),
            ],
            dim=1,
        )
        return self.network(z).squeeze(1)


FAMILIES = {
    "dcn_uniform": (DCNCross, None),
    "pnn_recency7": (ProductNetwork, 7.0),
    "fibinet_recency3": (FiBiNet, 3.0),
}


def predict(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0, len(x_np), BATCH_SIZE * 2):
            hi = min(lo + BATCH_SIZE * 2, len(x_np))
            xb = torch.from_numpy(x_np[lo:hi])
            result[lo:hi] = model(xb).cpu().numpy()
    return result


def train_select(
    model_class,
    half_life,
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
        model.parameters(), lr=LR, weight_decay=1e-6
    )

    x_t = torch.from_numpy(x_train)
    y_t = torch.from_numpy(y_train.astype(np.float32, copy=False))
    w_t = torch.from_numpy(recency_weights(dates_train, half_life))
    n = len(x_train)

    best_primary = -np.inf
    best_epoch = None
    best_scores = None
    best_metrics = None

    generator = torch.Generator()
    generator.manual_seed(SEED)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=generator)

        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_t[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx], reduction="none"
            )
            loss = (losses * w_t[idx]).mean()

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


def fit_fixed(
    model_class,
    half_life,
    x_fit,
    y_fit,
    dates_fit,
    epochs,
):
    torch.manual_seed(SEED)
    model = model_class()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-6
    )

    x_t = torch.from_numpy(x_fit)
    y_t = torch.from_numpy(y_fit.astype(np.float32, copy=False))
    w_t = torch.from_numpy(recency_weights(dates_fit, half_life))
    n = len(x_fit)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_t[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx], reduction="none"
            )
            loss = (losses * w_t[idx]).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    nonsingle = repeated_counts > 1
    ranked_sorted[nonsingle] = (
        positions[nonsingle] / (repeated_counts[nonsingle] - 1.0)
    )

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


train = load("train")
valid = load("valid")

x_train = encode(train)
x_valid = encode(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
dates_train = np.asarray(train.date)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

candidate_scores = {}
candidate_metrics = {}
candidate_epochs = {}

for name, (model_class, half_life) in FAMILIES.items():
    epoch, scores, metrics = train_select(
        model_class=model_class,
        half_life=half_life,
        x_train=x_train,
        y_train=y_train,
        dates_train=dates_train,
        x_valid=x_valid,
        y_valid=y_valid,
        valid_users=valid_users,
    )
    candidate_scores[name] = scores
    candidate_metrics[name] = metrics
    candidate_epochs[name] = epoch

recorded = {}
best_primary = -np.inf
best_name = None
best_alpha = None
best_valid_scores = None
best_raw_scores = None
best_metrics = None

alphas = np.linspace(0.0, 1.0, 11)

for name, raw_scores in candidate_scores.items():
    standalone = candidate_metrics[name]
    recorded[name + "_standalone"] = float(standalone["primary"])

    model_rank = within_user_rank(valid_users, raw_scores)
    local_best = -np.inf
    local_alpha = None

    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid_rank + alpha * model_rank
        metrics = evaluate(valid_users, y_valid, blended)
        primary = float(metrics["primary"])
        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()
            best_raw_scores = raw_scores.copy()
            best_metrics = metrics

    recorded[name + "_best_blend"] = float(local_best)
    recorded[name + "_blend_alpha"] = float(local_alpha)
    recorded[name + "_epoch"] = int(candidate_epochs[name])

print("CANDIDATES " + json.dumps(recorded, sort_keys=True))

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

# Refit the selected architecture on train + validation.
x_fit = np.concatenate([x_train, x_valid], axis=0)
y_fit = np.concatenate(
    [y_train, np.asarray(valid.y, dtype=np.float32)],
    axis=0,
)
dates_fit = np.concatenate(
    [dates_train, np.asarray(valid.date)],
    axis=0,
)

selected_class, selected_half_life = FAMILIES[best_name]
test_model = fit_fixed(
    model_class=selected_class,
    half_life=selected_half_life,
    x_fit=x_fit,
    y_fit=y_fit,
    dates_fit=dates_fit,
    epochs=candidate_epochs[best_name],
)

test = load("test")
x_test = encode(test)
raw_test_scores = predict(test_model, x_test)

inc_test = np.load(inc_test_path)
test_users = np.asarray(test.user_id)
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

elapsed = time.time() - START
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": best_name,
            "selected_epoch": int(candidate_epochs[best_name]),
            "selected_model_weight": float(best_alpha),
        },
        sort_keys=True,
    )
)
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