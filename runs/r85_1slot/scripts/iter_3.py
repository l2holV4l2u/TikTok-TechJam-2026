import os
import gc
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
EPOCHS = {
    "additive": 7,
    "fm": 7,
    "deepfm": 7,
}

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


def extract(split_name, with_labels):
    s = load(split_name)
    x = np.column_stack([s.X[name] for name in FIELDS]).astype(
        np.int64, copy=False
    )
    users = np.asarray(s.user_id, dtype=np.int64)
    if with_labels:
        y = np.asarray(s.y, dtype=np.float32)
        del s
        gc.collect()
        return x, y, users
    del s
    gc.collect()
    return x, users


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


class AdditiveModel(nn.Module):
    """Wide categorical logistic model: prediction is a sum of field biases."""

    def __init__(self, base_logit):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.register_buffer(
            "offsets", torch.as_tensor(offsets_np, dtype=torch.long)
        )
        self.register_buffer(
            "base_logit",
            torch.tensor(float(base_logit), dtype=torch.float32),
        )
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        ids = x + self.offsets
        return self.base_logit + self.linear(ids).squeeze(-1).sum(dim=1)


class FMModel(nn.Module):
    """Second-order low-rank interactions between all expanded fields."""

    def __init__(self, base_logit, rank=16):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.factors = nn.Embedding(total_cardinality, rank)
        self.register_buffer(
            "offsets", torch.as_tensor(offsets_np, dtype=torch.long)
        )
        self.register_buffer(
            "base_logit",
            torch.tensor(float(base_logit), dtype=torch.float32),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(self, x):
        ids = x + self.offsets
        wide = self.linear(ids).squeeze(-1).sum(dim=1)
        v = self.factors(ids)
        summed = v.sum(dim=1)
        pairwise = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.base_logit + wide + pairwise


class DeepFMModel(nn.Module):
    """Wide + FM pair interactions + nonlinear cross-field prediction."""

    def __init__(self, base_logit, rank=12):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.factors = nn.Embedding(total_cardinality, rank)
        self.register_buffer(
            "offsets", torch.as_tensor(offsets_np, dtype=torch.long)
        )
        self.register_buffer(
            "base_logit",
            torch.tensor(float(base_logit), dtype=torch.float32),
        )
        input_dim = len(FIELDS) * rank
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)
        for layer in self.deep:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        ids = x + self.offsets
        wide = self.linear(ids).squeeze(-1).sum(dim=1)
        v = self.factors(ids)
        summed = v.sum(dim=1)
        pairwise = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(v.reshape(v.shape[0], -1)).squeeze(1)
        return self.base_logit + wide + pairwise + deep


def make_model(family, y):
    p = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    base_logit = float(np.log(p / (1.0 - p)))
    if family == "additive":
        return AdditiveModel(base_logit)
    if family == "fm":
        return FMModel(base_logit, rank=16)
    if family == "deepfm":
        return DeepFMModel(base_logit, rank=12)
    raise ValueError(f"Unknown family: {family}")


@torch.no_grad()
def predict(model, x):
    model.eval()
    out = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x.shape[0])
        xb = torch.from_numpy(x[start:end])
        out[start:end] = model(xb).cpu().numpy()
    return out


def fit_model(family, x, y, epochs):
    family_seed = {
        "additive": SEED + 11,
        "fm": SEED + 23,
        "deepfm": SEED + 37,
    }[family]
    torch.manual_seed(family_seed)

    model = make_model(family, y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    generator = torch.Generator()
    generator.manual_seed(family_seed)

    for _ in range(epochs):
        permutation = torch.randperm(x_tensor.shape[0], generator=generator)
        model.train()
        for start in range(0, x_tensor.shape[0], BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x_tensor[idx])
            loss = F.binary_cross_entropy_with_logits(
                logits, y_tensor[idx]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


x_train, y_train, _ = extract("train", with_labels=True)
x_valid, y_valid, valid_users = extract("valid", with_labels=True)
valid_labels = y_valid.astype(np.int8)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation predictions missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.shape[0] != x_valid.shape[0]:
    raise ValueError("Incumbent validation prediction length mismatch")

inc_valid_scale = max(float(np.std(inc_valid)), 1e-8)
inc_valid_norm = inc_valid / inc_valid_scale

families = ["additive", "fm", "deepfm"]
valid_predictions = {}
candidate_scores = {}
family_blend_weights = {}
family_scales = {}
best_name = None
best_family = None
best_weight = None
best_scores = None
best_metrics = None
best_raw_scores = None

for family in families:
    model = fit_model(
        family, x_train, y_train, EPOCHS[family]
    )
    raw = predict(model, x_valid).astype(np.float64)
    valid_predictions[family] = raw
    del model
    gc.collect()

    raw_metrics = evaluate(valid_users, valid_labels, raw)
    candidate_scores[f"{family}_raw"] = float(raw_metrics["primary"])

    own_scale = max(float(np.std(raw)), 1e-8)
    family_scales[family] = own_scale
    own_norm = raw / own_scale

    local_best_primary = -np.inf
    local_best_weight = 1.0
    local_best_metrics = None
    local_best_scores = None

    # Weight is the contribution of the new family. The exact selected
    # weight is subsequently reused for test.
    for w in np.linspace(0.0, 1.0, 21):
        blended = w * own_norm + (1.0 - w) * inc_valid_norm
        metrics_w = evaluate(valid_users, valid_labels, blended)
        if float(metrics_w["primary"]) > local_best_primary:
            local_best_primary = float(metrics_w["primary"])
            local_best_weight = float(w)
            local_best_metrics = metrics_w
            local_best_scores = blended.copy()

    family_blend_weights[family] = local_best_weight
    candidate_scores[f"{family}_blend"] = local_best_primary

    if best_metrics is None or local_best_primary > float(
        best_metrics["primary"]
    ):
        best_name = f"{family}_blend"
        best_family = family
        best_weight = local_best_weight
        best_scores = local_best_scores
        best_metrics = local_best_metrics
        best_raw_scores = raw.copy()

    if float(raw_metrics["primary"]) > float(best_metrics["primary"]):
        best_name = f"{family}_raw"
        best_family = family
        best_weight = 1.0
        best_scores = raw.copy()
        best_metrics = raw_metrics
        best_raw_scores = raw.copy()


print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "selected_family": best_family,
            "own_weight": best_weight,
            "blend_weights": family_blend_weights,
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_weight < 1.0 - 1e-12:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw_scores, dtype=np.float64),
        )

# Refit the selected structural family on train + validation using the
# identical epoch count, then form test predictions with the fixed blend.
x_refit = np.concatenate([x_train, x_valid], axis=0)
y_refit = np.concatenate([y_train, y_valid], axis=0)
del x_train, y_train, x_valid, y_valid
del valid_predictions
gc.collect()

test_model = fit_model(
    best_family,
    x_refit,
    y_refit,
    EPOCHS[best_family],
)
del x_refit, y_refit
gc.collect()

x_test, _ = extract("test", with_labels=False)
own_test = predict(test_model, x_test).astype(np.float64)
del test_model, x_test
gc.collect()

if best_weight < 1.0 - 1e-12:
    if not os.path.exists(inc_test_path):
        raise FileNotFoundError("Trusted incumbent test predictions missing")
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    if inc_test.shape[0] != own_test.shape[0]:
        raise ValueError("Incumbent test prediction length mismatch")

    own_scale = family_scales[best_family]
    test_scores = (
        best_weight * (own_test / own_scale)
        + (1.0 - best_weight) * (inc_test / inc_valid_scale)
    )
else:
    test_scores = own_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
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
        }
    )
)