import os
import gc
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024
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
K = 16
FFM_K = 8
LR = 0.001
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = {
    "wide_additive": 8,
    "expanded_fm": 8,
    "deepfm": 8,
    "field_aware_fm": 8,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, N_FIELDS), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + offsets[j]
    return x


def make_combined_matrix(a, b):
    na = len(a.user_id)
    nb = len(b.user_id)
    x = np.empty((na + nb, N_FIELDS), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:na, j] = np.asarray(a.X[field], dtype=np.int64) + offsets[j]
        x[na:, j] = np.asarray(b.X[field], dtype=np.int64) + offsets[j]
    return x


class WideAdditive(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(-1)


class ExpandedFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, K)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, K)
        self.deep = nn.Sequential(
            nn.Linear(N_FIELDS * K, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(v.reshape(v.shape[0], -1)).squeeze(-1)
        return self.bias + linear + fm + deep


class FieldAwareFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality * N_FIELDS, FFM_K)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        interaction = torch.zeros(
            x.shape[0], dtype=self.embedding.weight.dtype, device=x.device
        )
        for i in range(N_FIELDS):
            xi = x[:, i]
            for j in range(i + 1, N_FIELDS):
                xj = x[:, j]
                vi_for_j = self.embedding(xi * N_FIELDS + j)
                vj_for_i = self.embedding(xj * N_FIELDS + i)
                interaction = interaction + (vi_for_j * vj_for_i).sum(dim=1)
        return self.bias + linear + interaction


def make_model(name, seed):
    seed_everything(seed)
    if name == "wide_additive":
        return WideAdditive()
    if name == "expanded_fm":
        return ExpandedFM()
    if name == "deepfm":
        return DeepFM()
    if name == "field_aware_fm":
        return FieldAwareFM()
    raise ValueError(name)


def train_epoch(model, optimizer, x_tensor, y_tensor, generator):
    model.train()
    n = x_tensor.shape[0]
    order = torch.randperm(n, generator=generator)
    total_loss = 0.0

    for start in range(0, n, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x_tensor[idx]
        yb = y_tensor[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(idx)

    return total_loss / n


@torch.no_grad()
def predict(model, x):
    model.eval()
    xt = torch.from_numpy(x)
    scores = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x.shape[0])
        scores[start:end] = (
            model(xt[start:end]).cpu().numpy().astype(np.float32, copy=False)
        )
    return scores


def fit_family(name, x_train_t, y_train_t, x_valid, valid, y_valid):
    model = make_model(name, SEED + 101)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator()
    generator.manual_seed(SEED + 911)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    best_state = None

    for epoch in range(1, MAX_EPOCHS[name] + 1):
        train_epoch(model, optimizer, x_train_t, y_train_t, generator)
        scores = predict(model, x_valid)
        metrics = evaluate(valid.user_id, y_valid, scores)
        primary = float(metrics["primary"])

        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    del model, optimizer
    gc.collect()
    return best_scores, best_state, best_epoch, best_primary


train = load("train")
valid = load("valid")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train_t = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train)

shared_dir = os.environ.get("SHARED_ARTIFACTS")
if not shared_dir:
    raise RuntimeError("SHARED_ARTIFACTS is required for incumbent blending")

inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float32, copy=False)
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")

family_names = [
    "wide_additive",
    "expanded_fm",
    "deepfm",
    "field_aware_fm",
]

family_results = {}
candidate_log = {}
winner = None
winner_primary = -np.inf

blend_grid = np.linspace(0.0, 1.0, 11)

for family_index, name in enumerate(family_names):
    scores, state, best_epoch, standalone_primary = fit_family(
        name, x_train_t, y_train_t, x_valid, valid, y_valid
    )

    family_results[name] = {
        "scores": scores,
        "state": state,
        "epoch": best_epoch,
    }
    candidate_log[name] = float(standalone_primary)

    best_blend_primary = -np.inf
    best_alpha = 1.0
    best_blend_scores = scores

    for alpha in blend_grid:
        blended = alpha * scores + (1.0 - alpha) * inc_valid
        blend_metrics = evaluate(valid.user_id, y_valid, blended)
        blend_primary = float(blend_metrics["primary"])
        if blend_primary > best_blend_primary:
            best_blend_primary = blend_primary
            best_alpha = float(alpha)
            best_blend_scores = blended.copy()

    candidate_log[name + "_incumbent_blend"] = float(best_blend_primary)

    if best_blend_primary > winner_primary:
        winner_primary = best_blend_primary
        winner = {
            "family": name,
            "alpha": best_alpha,
            "epoch": best_epoch,
            "valid_scores": best_blend_scores,
        }

valid_scores = np.asarray(winner["valid_scores"], dtype=np.float32)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected_family": winner["family"],
    "selected_epoch": int(winner["epoch"]),
    "new_model_blend_weight": float(winner["alpha"]),
}))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Release all validation models and large training matrices before the refit.
selected_family = winner["family"]
selected_epoch = int(winner["epoch"])
selected_alpha = float(winner["alpha"])

family_results.clear()
del x_train_t, y_train_t, x_train, y_train
gc.collect()

# Refit the identical selected family on train + validation, stopping at the
# epoch selected using the train-only validation experiment.
test = load("test")
x_combined = make_combined_matrix(train, valid)
y_combined = np.concatenate([
    np.asarray(train.y, dtype=np.float32),
    np.asarray(valid.y, dtype=np.float32),
])
x_test = make_matrix(test)

combined_x_t = torch.from_numpy(x_combined)
combined_y_t = torch.from_numpy(y_combined)

combined_model = make_model(selected_family, SEED + 101)
combined_optimizer = torch.optim.Adam(combined_model.parameters(), lr=LR)
combined_generator = torch.Generator()
combined_generator.manual_seed(SEED + 911)

for _ in range(selected_epoch):
    train_epoch(
        combined_model,
        combined_optimizer,
        combined_x_t,
        combined_y_t,
        combined_generator,
    )

new_test_scores = predict(combined_model, x_test)

if selected_alpha < 1.0:
    inc_test = np.load(inc_test_path).astype(np.float32, copy=False)
    test_scores = (
        selected_alpha * new_test_scores
        + (1.0 - selected_alpha) * inc_test
    )
else:
    test_scores = new_test_scores

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))