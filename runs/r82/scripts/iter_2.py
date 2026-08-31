import os
import time
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2026
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
K = 16

FAMILY_CONFIGS = {
    "additive": {"epochs": 5, "lr": 0.003},
    "fm_expanded": {"epochs": 8, "lr": 0.001},
    "deepfm": {"epochs": 5, "lr": 0.001},
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
    x = np.empty((n, len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + offsets[j]
    return x


def make_combined_matrix(a, b):
    na = len(a.user_id)
    nb = len(b.user_id)
    x = np.empty((na + nb, len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:na, j] = np.asarray(a.X[field], dtype=np.int64) + offsets[j]
        x[na:, j] = np.asarray(b.X[field], dtype=np.int64) + offsets[j]
    return x


class AdditiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(-1)


class FMModel(nn.Module):
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
        interactions = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interactions


class DeepFMModel(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 12
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(len(FIELDS) * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.mlp(v.reshape(v.shape[0], -1)).squeeze(-1)
        return self.bias + linear + fm + deep


def make_model(name):
    if name == "additive":
        return AdditiveModel()
    if name == "fm_expanded":
        return FMModel()
    if name == "deepfm":
        return DeepFMModel()
    raise ValueError(name)


@torch.no_grad()
def predict(model, x):
    model.eval()
    scores = np.empty(x.shape[0], dtype=np.float32)
    x_tensor = torch.from_numpy(x)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x.shape[0])
        scores[start:end] = (
            model(x_tensor[start:end]).cpu().numpy().astype(np.float32, copy=False)
        )
    return scores


def fit_family(name, x, y, epochs, lr, valid_x=None, valid_split=None):
    seed_everything(SEED + {"additive": 11, "fm_expanded": 23, "deepfm": 37}[name])
    model = make_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))

    generator = torch.Generator()
    generator.manual_seed(SEED + {"additive": 101, "fm_expanded": 203, "deepfm": 307}[name])

    best_primary = -np.inf
    best_state = None
    best_scores = None
    best_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(x.shape[0], generator=generator)

        for start in range(0, x.shape[0], BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[idx])
            loss = loss_fn(logits, y_tensor[idx])
            loss.backward()
            optimizer.step()

        if valid_x is not None:
            epoch_scores = predict(model, valid_x)
            epoch_metrics = evaluate(
                valid_split.user_id,
                np.asarray(valid_split.y, dtype=np.int8),
                epoch_scores,
            )
            primary = float(epoch_metrics["primary"])
            if primary > best_primary:
                best_primary = primary
                best_epoch = epoch
                best_scores = epoch_scores.copy()
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if valid_x is not None:
        model.load_state_dict(best_state)
        return model, best_scores, best_epoch

    return model, None, epochs


def fit_empirical_bayes(split, smoothing=30.0):
    y = np.asarray(split.y, dtype=np.float64)
    global_rate = float(np.clip(y.mean(), 1e-5, 1.0 - 1e-5))
    tables = []
    for field in FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
        positives = np.bincount(
            ids, weights=y, minlength=cardinality
        ).astype(np.float64)
        rates = (positives + smoothing * global_rate) / (counts + smoothing)
        rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
        logits = np.log(rates / (1.0 - rates))
        support = counts / (counts + smoothing)
        tables.append((logits, support))
    global_logit = np.log(global_rate / (1.0 - global_rate))
    return global_logit, tables


def predict_empirical_bayes(split, fitted):
    global_logit, tables = fitted
    score = np.full(len(split.user_id), global_logit, dtype=np.float64)
    total_weight = np.ones(len(split.user_id), dtype=np.float64)

    for field, (logits, support) in zip(FIELDS, tables):
        ids = np.asarray(split.X[field], dtype=np.int64)
        w = support[ids]
        score += w * (logits[ids] - global_logit)
        total_weight += w

    return global_logit + (score - global_logit) / total_weight


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
x_train = make_matrix(train)
x_valid = make_matrix(valid)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

family_valid_scores = {}
family_best_epochs = {}
candidate_scores = {}

for family_name, config in FAMILY_CONFIGS.items():
    model, scores, best_epoch = fit_family(
        family_name,
        x_train,
        y_train,
        config["epochs"],
        config["lr"],
        valid_x=x_valid,
        valid_split=valid,
    )
    family_valid_scores[family_name] = np.asarray(scores, dtype=np.float64)
    family_best_epochs[family_name] = int(best_epoch)
    standalone_metrics = evaluate(valid.user_id, y_valid, scores)
    candidate_scores[family_name] = float(standalone_metrics["primary"])
    del model

eb_fit = fit_empirical_bayes(train, smoothing=30.0)
eb_valid_scores = predict_empirical_bayes(valid, eb_fit)
family_valid_scores["empirical_bayes"] = eb_valid_scores
family_best_epochs["empirical_bayes"] = 0
candidate_scores["empirical_bayes"] = float(
    evaluate(valid.user_id, y_valid, eb_valid_scores)["primary"]
)

blend_alphas = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0]
best_primary = -np.inf
winner_family = None
winner_alpha = None
winner_valid_scores = None

for family_name, scores in family_valid_scores.items():
    family_best_blend = -np.inf
    family_best_alpha = None
    for alpha in blend_alphas:
        blended = alpha * scores + (1.0 - alpha) * inc_valid
        blend_metrics = evaluate(valid.user_id, y_valid, blended)
        primary = float(blend_metrics["primary"])
        if primary > family_best_blend:
            family_best_blend = primary
            family_best_alpha = alpha
        if primary > best_primary:
            best_primary = primary
            winner_family = family_name
            winner_alpha = float(alpha)
            winner_valid_scores = blended.copy()

    candidate_scores[family_name + "_blend"] = family_best_blend
    candidate_scores[family_name + "_blend_alpha"] = family_best_alpha

metrics = evaluate(valid.user_id, y_valid, winner_valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner_valid_scores, dtype=np.float64),
    )

# Refit the selected family on train + validation, then apply the exact
# validation-selected blend weight to the trusted incumbent's test scores.
test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winner_family == "empirical_bayes":
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.user_id = np.concatenate([
        np.asarray(train.user_id),
        np.asarray(valid.user_id),
    ])
    combined.y = np.concatenate([
        np.asarray(train.y, dtype=np.int8),
        np.asarray(valid.y, dtype=np.int8),
    ])
    combined.X = {}
    for field in FIELDS:
        combined.X[field] = np.concatenate([
            np.asarray(train.X[field], dtype=np.int64),
            np.asarray(valid.X[field], dtype=np.int64),
        ])

    combined_eb = fit_empirical_bayes(combined, smoothing=30.0)
    new_test_scores = predict_empirical_bayes(test, combined_eb)
else:
    x_combined = make_combined_matrix(train, valid)
    y_combined = np.concatenate([
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ])
    x_test = make_matrix(test)

    config = FAMILY_CONFIGS[winner_family]
    selected_epoch = family_best_epochs[winner_family]
    combined_model, _, _ = fit_family(
        winner_family,
        x_combined,
        y_combined,
        selected_epoch,
        config["lr"],
        valid_x=None,
        valid_split=None,
    )
    new_test_scores = predict(combined_model, x_test)

test_scores = (
    winner_alpha * np.asarray(new_test_scores, dtype=np.float64)
    + (1.0 - winner_alpha) * inc_test
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("FINDINGS " + json.dumps({
    "winner_family": winner_family,
    "winner_alpha_new_family": winner_alpha,
    "best_epochs": family_best_epochs,
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

elapsed = time.time() - START_TIME
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))