import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 314159

# Targeted change: retain the complete modeling and selection recipe, but expose
# all safe categorical fields to the interaction models.
FIELDS = [
    "author_id",
    "duration_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "hour",
    "is_live_streamer",
    "is_lowactive_period",
    "is_video_author",
    "music_type",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "register_days_bucket",
    "register_days_range",
    "tab",
    "tag",
    "upload_type",
    "user_active_degree",
    "user_id",
    "video_id",
    "video_type",
]
EB_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "upload_type",
    "music_type",
    "duration_bucket",
    "tab",
    "hour",
]
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 65536
FAMILY_EPOCHS = {
    "wide": 5,
    "fm": 8,
    "deepfm": 5,
    "dcn": 5,
}
BLEND_WEIGHTS = [0.2, 0.4, 0.6, 0.8]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))
n_fields = len(FIELDS)


def make_features(split):
    x = np.column_stack([
        np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
    ])
    x += offsets_np[None, :]
    return np.ascontiguousarray(x, dtype=np.int64)


class WideModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(1)


class FMModel(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, k)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - v.square().sum(dim=(1, 2))
        )
        return self.bias + linear + interaction


class DeepFMModel(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, k)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(n_fields * k, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square().sum(dim=1)
            - v.square().sum(dim=(1, 2))
        )
        deep = self.mlp(v.flatten(start_dim=1)).squeeze(1)
        return self.bias + linear + fm + deep


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x0, x):
        scalar = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scalar + self.bias + x


class DCNModel(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        dim = n_fields * k
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, k)
        self.cross1 = CrossLayer(dim)
        self.cross2 = CrossLayer(dim)
        self.output = nn.Linear(dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        wide = self.linear(x).sum(dim=1).squeeze(1)
        x0 = self.embedding(x).flatten(start_dim=1)
        z = self.cross1(x0, x0)
        z = self.cross2(x0, z)
        crossed = self.output(z).squeeze(1)
        return self.bias + wide + crossed


def make_model(name):
    torch.manual_seed(SEED)
    if name == "wide":
        return WideModel()
    if name == "fm":
        return FMModel()
    if name == "deepfm":
        return DeepFMModel()
    if name == "dcn":
        return DCNModel()
    raise ValueError(name)


@torch.no_grad()
def predict_torch(model, x_np):
    model.eval()
    out = np.empty(x_np.shape[0], dtype=np.float64)
    for begin in range(0, x_np.shape[0], PRED_BATCH_SIZE):
        end = min(begin + PRED_BATCH_SIZE, x_np.shape[0])
        xb = torch.from_numpy(x_np[begin:end])
        out[begin:end] = model(xb).cpu().numpy().astype(np.float64)
    return out


def train_one_epoch(model, optimizer, x, y, generator):
    model.train()
    order = torch.randperm(x.shape[0], generator=generator)
    for begin in range(0, x.shape[0], BATCH_SIZE):
        idx = order[begin:begin + BATCH_SIZE]
        xb = x[idx]
        yb = y[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()


def fit_family_select_epoch(name, x_train_np, y_train_np, valid, x_valid_np):
    model = make_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(np.asarray(y_train_np, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None

    for epoch in range(1, FAMILY_EPOCHS[name] + 1):
        train_one_epoch(model, optimizer, x_train, y_train, generator)
        scores = predict_torch(model, x_valid_np)
        primary = float(evaluate(valid.user_id, valid.y, scores)["primary"])
        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()

    del model, optimizer, x_train, y_train
    gc.collect()
    return best_scores, best_epoch


def fit_family_fixed(name, x_np, y_np, epochs):
    model = make_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(epochs):
        train_one_epoch(model, optimizer, x, y, generator)

    return model


def fit_eb(split, labels, smoothing=30.0):
    y = np.asarray(labels, dtype=np.float64)
    global_rate = float(np.mean(y))
    global_logit = np.log(
        np.clip(global_rate, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - global_rate, 1e-6, 1.0)
    )
    tables = {}

    for field in EB_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, minlength=card).astype(np.float64)
        positives = np.bincount(
            ids, weights=y, minlength=card
        ).astype(np.float64)
        rate = (positives + smoothing * global_rate) / (
            counts + smoothing
        )
        table = np.log(
            np.clip(rate, 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - rate, 1e-6, 1.0)
        )
        tables[field] = table.astype(np.float64)

    return global_logit, tables


def predict_eb(split, model):
    global_logit, tables = model
    score = np.zeros(len(split.user_id), dtype=np.float64)
    for field in EB_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        score += tables[field][ids] - global_logit
    return score


train = load("train")
valid = load("valid")
x_train = make_features(train)
x_valid = make_features(valid)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

candidate_scores = {}
candidate_primary = {}
family_epochs = {}

for family in ["wide", "fm", "deepfm", "dcn"]:
    scores, epoch = fit_family_select_epoch(
        family, x_train, train.y, valid, x_valid
    )
    candidate_scores[family] = scores
    family_epochs[family] = epoch
    candidate_primary[family] = float(
        evaluate(valid.user_id, valid.y, scores)["primary"]
    )

eb_model = fit_eb(train, train.y)
eb_valid = predict_eb(valid, eb_model)
candidate_scores["empirical_bayes"] = eb_valid
candidate_primary["empirical_bayes"] = float(
    evaluate(valid.user_id, valid.y, eb_valid)["primary"]
)

best_name = None
best_family = None
best_alpha = 1.0
best_primary = -np.inf
best_valid_scores = None

for family, scores in candidate_scores.items():
    p = candidate_primary[family]
    if p > best_primary:
        best_primary = p
        best_name = family
        best_family = family
        best_alpha = 1.0
        best_valid_scores = scores.copy()

    for alpha in BLEND_WEIGHTS:
        blended = alpha * scores + (1.0 - alpha) * inc_valid
        blend_name = f"{family}_blend_{alpha:.1f}"
        blend_primary = float(
            evaluate(valid.user_id, valid.y, blended)["primary"]
        )
        candidate_primary[blend_name] = blend_primary
        if blend_primary > best_primary:
            best_primary = blend_primary
            best_name = blend_name
            best_family = family
            best_alpha = alpha
            best_valid_scores = blended.copy()

metrics = evaluate(valid.user_id, valid.y, best_valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Validation selection is complete. Refit the identical selected recipe on
# train plus validation and then score test without accessing test labels.
test = load("test")
x_test = make_features(test)

if best_family == "empirical_bayes":
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {
        f: np.concatenate([
            np.asarray(train.X[f], dtype=np.int64),
            np.asarray(valid.X[f], dtype=np.int64),
        ])
        for f in EB_FIELDS
    }
    combined.user_id = np.concatenate([
        np.asarray(train.user_id),
        np.asarray(valid.user_id),
    ])
    y_combined = np.concatenate([
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ])
    refit_eb = fit_eb(combined, y_combined)
    new_test_scores = predict_eb(test, refit_eb)
else:
    x_combined = np.concatenate([x_train, x_valid], axis=0)
    y_combined = np.concatenate([
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ])
    refit_model = fit_family_fixed(
        best_family,
        x_combined,
        y_combined,
        family_epochs[best_family],
    )
    new_test_scores = predict_torch(refit_model, x_test)

if best_alpha < 1.0:
    inc_test = np.asarray(
        np.load(inc_test_path),
        dtype=np.float64,
    )
    test_scores = (
        best_alpha * new_test_scores
        + (1.0 - best_alpha) * inc_test
    )
else:
    test_scores = new_test_scores

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(
    {k: float(v) for k, v in candidate_primary.items()},
    sort_keys=True,
))
print("FINDINGS " + json.dumps({
    "selected": best_name,
    "selected_family": best_family,
    "selected_new_model_weight": float(best_alpha),
    "best_epochs": {
        k: int(v) for k, v in family_epochs.items()
    },
    "categorical_field_count": int(n_fields),
}, sort_keys=True))

elapsed = time.time() - START_TIME
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))