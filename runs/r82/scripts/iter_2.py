import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 42
BATCH_SIZE = 2048
PRED_BATCH_SIZE = 16384
LR = 0.001

# These additional fields have meaningful within-user variation and relatively
# stable semantics across the date boundary.
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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
device = torch.device("cpu")

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
n_fields = len(FIELDS)
n_features = int(sum(cards))


def make_features(split):
    arr = np.stack(
        [
            np.asarray(split.X[name], dtype=np.int64) + offsets[j]
            for j, name in enumerate(FIELDS)
        ],
        axis=1,
    )
    return torch.from_numpy(arr)


class WideAdditive(nn.Module):
    """Independent categorical effects: no learned feature interactions."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(-1)


class ExpandedFM(nn.Module):
    """Shared low-rank pairwise interactions."""

    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interactions = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interactions


class DeepFM(nn.Module):
    """FM prediction plus a nonlinear tower over all field embeddings."""

    def __init__(self, dim=12):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(n_fields * dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        fm = 0.5 * (summed.square() - v.square().sum(dim=1)).sum(dim=1)
        deep = self.mlp(v.reshape(v.shape[0], -1)).squeeze(-1)
        return self.bias + linear + fm + deep


class FieldAwareFM(nn.Module):
    """
    Each categorical value has a different embedding when interacting with
    each partner field, unlike the shared embedding used by an ordinary FM.
    """

    def __init__(self, dim=8):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, n_fields * dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.dim = dim
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        # [batch, source_field, partner_field, dim]
        v = self.embedding(x).reshape(
            x.shape[0], n_fields, n_fields, self.dim
        )
        interaction = torch.zeros(
            x.shape[0], dtype=v.dtype, device=v.device
        )
        for i in range(n_fields):
            for j in range(i + 1, n_fields):
                interaction = interaction + (
                    v[:, i, j, :] * v[:, j, i, :]
                ).sum(dim=1)
        return self.bias + linear + interaction


RECIPES = {
    "wide_additive": {
        "factory": lambda: WideAdditive(),
        "epochs": 4,
        "half_life": None,
    },
    "expanded_fm": {
        "factory": lambda: ExpandedFM(16),
        "epochs": 5,
        "half_life": None,
    },
    "recency_fm_h6": {
        "factory": lambda: ExpandedFM(16),
        "epochs": 5,
        "half_life": 6.0,
    },
    "deepfm": {
        "factory": lambda: DeepFM(12),
        "epochs": 4,
        "half_life": None,
    },
    "field_aware_fm": {
        "factory": lambda: FieldAwareFM(8),
        "epochs": 4,
        "half_life": None,
    },
}


def date_weights(dates, half_life):
    if half_life is None:
        return None
    d = np.asarray(dates, dtype=np.int64)
    age = np.max(d) - d
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    # Preserve approximately the same average gradient scale as uniform BCE.
    w /= max(float(w.mean()), 1e-8)
    return torch.from_numpy(w.astype(np.float32))


def fit_model(x, y, recipe, dates, seed):
    torch.manual_seed(seed)
    model = recipe["factory"]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    weights = date_weights(dates, recipe["half_life"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1000)
    n = x.shape[0]

    model.train()
    for _ in range(recipe["epochs"]):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = x[idx].to(device)
            yb = y[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            if weights is not None:
                loss = (losses * weights[idx].to(device)).mean()
            else:
                loss = losses.mean()
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict(model, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x.shape[0])
        logits = model(x[start:end].to(device))
        result[start:end] = torch.sigmoid(logits).cpu().numpy()
    return result


shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required for incumbent blending")

inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

train = load("train")
valid = load("valid")

x_train = make_features(train)
x_valid = make_features(valid)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
train_dates = np.asarray(train.date, dtype=np.int64)

candidate_scores = {}
candidate_predictions = {}
candidate_alphas = {}

best_name = None
best_primary = -np.inf
best_valid_scores = None

inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metrics["primary"])
best_name = "trusted_incumbent"
best_primary = float(inc_metrics["primary"])
best_valid_scores = inc_valid.copy()

blend_grid = np.linspace(0.0, 1.0, 21)

for model_index, (name, recipe) in enumerate(RECIPES.items()):
    model = fit_model(
        x_train,
        y_train,
        recipe,
        train_dates,
        SEED + 17 * model_index,
    )
    pred = predict(model, x_valid)
    standalone = evaluate(valid.user_id, valid.y, pred)
    candidate_scores[name] = float(standalone["primary"])

    local_best_primary = -np.inf
    local_best_alpha = 1.0
    local_best_pred = pred

    for alpha in blend_grid:
        blended = alpha * pred + (1.0 - alpha) * inc_valid
        bm = evaluate(valid.user_id, valid.y, blended)
        p = float(bm["primary"])
        if p > local_best_primary:
            local_best_primary = p
            local_best_alpha = float(alpha)
            local_best_pred = blended.copy()

    blend_name = name + "_blend"
    candidate_scores[blend_name] = local_best_primary
    candidate_alphas[name] = local_best_alpha
    candidate_predictions[name] = pred

    if local_best_primary > best_primary:
        best_primary = local_best_primary
        best_name = name
        best_valid_scores = local_best_pred

    del model
    gc.collect()

metrics = evaluate(valid.user_id, valid.y, best_valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "selected_primary": float(metrics["primary"]),
            "blend_alphas_new_model": candidate_alphas,
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Produce hidden-test scores. If validation retained the trusted incumbent,
# reuse its already valid test prediction. Otherwise refit the selected family
# with the identical recipe on the allowed train+validation table.
if best_name == "trusted_incumbent":
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
else:
    selected_recipe = RECIPES[best_name]
    selected_index = list(RECIPES.keys()).index(best_name)
    alpha = candidate_alphas[best_name]

    if alpha <= 0.0:
        test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
    else:
        y_valid = torch.from_numpy(np.asarray(valid.y, dtype=np.float32))
        x_combined = torch.cat([x_train, x_valid], dim=0)
        y_combined = torch.cat([y_train, y_valid], dim=0)
        combined_dates = np.concatenate(
            [
                np.asarray(train.date, dtype=np.int64),
                np.asarray(valid.date, dtype=np.int64),
            ]
        )

        combined_model = fit_model(
            x_combined,
            y_combined,
            selected_recipe,
            combined_dates,
            SEED + 17 * selected_index,
        )

        test = load("test")
        x_test = make_features(test)
        new_test_pred = predict(combined_model, x_test)

        if alpha < 1.0:
            inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
            test_scores = alpha * new_test_pred + (1.0 - alpha) * inc_test
        else:
            test_scores = new_test_pred

        del combined_model, x_combined, y_combined, x_test
        gc.collect()

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)