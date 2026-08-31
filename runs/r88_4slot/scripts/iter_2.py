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
SEED = 2026
BATCH_SIZE = 8192
RANK = 16
LR = 0.001

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
STAT_FIELDS = [
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
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def encode(split):
    return np.stack(
        [
            np.asarray(split.X[name], dtype=np.int64) + offsets[j]
            for j, name in enumerate(FIELDS)
        ],
        axis=1,
    )


class ExpandedFM(nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1, sparse=True)
        self.latent = nn.Embedding(total_cardinality, RANK, sparse=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)
        prior = float(np.clip(prior, 1e-5, 1.0 - 1e-5))
        self.register_buffer(
            "intercept",
            torch.tensor(math.log(prior / (1.0 - prior)), dtype=torch.float32),
        )

    def forward(self, x):
        linear = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.latent(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.intercept + linear + interaction

    def optimizers(self):
        return [torch.optim.SparseAdam(self.parameters(), lr=LR)]


class NFM(nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1, sparse=True)
        self.latent = nn.Embedding(total_cardinality, RANK, sparse=True)
        self.mlp = nn.Sequential(
            nn.Linear(RANK, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.015)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        prior = float(np.clip(prior, 1e-5, 1.0 - 1e-5))
        self.register_buffer(
            "intercept",
            torch.tensor(math.log(prior / (1.0 - prior)), dtype=torch.float32),
        )

    def forward(self, x):
        linear = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.latent(x)
        pooled = 0.5 * (v.sum(dim=1).square() - v.square().sum(dim=1))
        nonlinear = self.mlp(pooled).squeeze(1)
        return self.intercept + linear + nonlinear

    def optimizers(self):
        sparse_parameters = [self.linear.weight, self.latent.weight]
        dense_parameters = list(self.mlp.parameters())
        return [
            torch.optim.SparseAdam(sparse_parameters, lr=LR),
            torch.optim.Adam(dense_parameters, lr=LR),
        ]


def predict_torch(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0, len(x_np), BATCH_SIZE * 2):
            hi = min(lo + BATCH_SIZE * 2, len(x_np))
            result[lo:hi] = model(
                torch.from_numpy(x_np[lo:hi])
            ).cpu().numpy()
    return result


def train_model(model_class, x_train, y_train, x_valid, valid, checkpoints):
    torch.manual_seed(SEED)
    model = model_class(float(y_train.mean()))
    optimizers = model.optimizers()
    criterion = nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train.astype(np.float32, copy=False))
    n = len(x_train)

    best_primary = -np.inf
    best_epoch = None
    best_scores = None
    best_state = None
    best_metrics = None

    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        order = torch.randperm(n)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(xt[idx])
            loss = criterion(logits, yt[idx])
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

        if epoch in checkpoints:
            scores = predict_torch(model, x_valid)
            metrics = evaluate(
                np.asarray(valid.user_id),
                np.asarray(valid.y, dtype=np.int8),
                scores,
            )
            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_epoch = int(epoch)
                best_scores = scores.copy()
                best_metrics = metrics
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    model.load_state_dict(best_state)
    return {
        "model": model,
        "epoch": best_epoch,
        "scores": best_scores,
        "metrics": best_metrics,
        "model_class": model_class,
    }


def fit_fixed_model(model_class, x_fit, y_fit, epochs):
    torch.manual_seed(SEED)
    model = model_class(float(y_fit.mean()))
    optimizers = model.optimizers()
    criterion = nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(x_fit)
    yt = torch.from_numpy(y_fit.astype(np.float32, copy=False))
    n = len(x_fit)

    for _ in range(int(epochs)):
        model.train()
        order = torch.randperm(n)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(xt[idx])
            loss = criterion(logits, yt[idx])
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

    return model


def fit_target_statistics(split, y, recency_half_life=5.0, smoothing=30.0):
    dates = np.asarray(split.date, dtype=np.int64)
    day_number = dates % 100
    newest = int(day_number.max())
    weights = np.exp2(-(newest - day_number) / recency_half_life).astype(np.float64)

    y64 = np.asarray(y, dtype=np.float64)
    global_rate = float(np.sum(weights * y64) / np.sum(weights))
    tables = {}

    for field in STAT_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])
        count = np.bincount(ids, weights=weights, minlength=card)
        positive = np.bincount(ids, weights=weights * y64, minlength=card)
        rate = (positive + smoothing * global_rate) / (count + smoothing)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        tables[field] = np.log(rate / (1.0 - rate)).astype(np.float32)

    return global_rate, tables


def predict_target_statistics(split, global_rate, tables):
    prior_logit = math.log(global_rate / (1.0 - global_rate))
    score = np.zeros(len(np.asarray(split.user_id)), dtype=np.float32)
    for field in STAT_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        table = tables[field]
        safe = np.clip(ids, 0, len(table) - 1)
        score += table[safe] - prior_logit
    score /= float(len(STAT_FIELDS))
    return score


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        return x - float(x.mean())
    return (x - float(x.mean())) / sd


def rank_percentile(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    if len(x) > 1:
        ranks /= float(len(x) - 1)
    return ranks


train = load("train")
valid = load("valid")

x_train = encode(train)
x_valid = encode(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

fm_result = train_model(
    ExpandedFM, x_train, y_train, x_valid, valid, checkpoints={4, 8, 12, 16, 20}
)
nfm_result = train_model(
    NFM, x_train, y_train, x_valid, valid, checkpoints={2, 4, 6, 8}
)

stat_prior, stat_tables = fit_target_statistics(train, y_train)
stat_valid = predict_target_statistics(valid, stat_prior, stat_tables)
stat_metrics = evaluate(valid_users, y_valid, stat_valid)

families = {
    "expanded_fm": {
        "scores": fm_result["scores"],
        "epoch": fm_result["epoch"],
        "model_class": ExpandedFM,
    },
    "nfm": {
        "scores": nfm_result["scores"],
        "epoch": nfm_result["epoch"],
        "model_class": NFM,
    },
    "recency_empirical_bayes": {
        "scores": stat_valid,
        "epoch": None,
        "model_class": None,
    },
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

candidate_scores = {}
candidate_details = {}
best_primary = -np.inf
winner = None

for family_name, info in families.items():
    own = np.asarray(info["scores"], dtype=np.float64)
    own_metrics = evaluate(valid_users, y_valid, own)
    standalone_name = family_name + "_standalone"
    candidate_scores[standalone_name] = float(own_metrics["primary"])
    candidate_details[standalone_name] = {
        "family": family_name,
        "mode": "standalone",
        "alpha": 1.0,
        "scores": own,
        "metrics": own_metrics,
    }

    if float(own_metrics["primary"]) > best_primary:
        best_primary = float(own_metrics["primary"])
        winner = standalone_name

    representations = {
        "raw": (own, inc_valid),
        "zscore": (zscore(own), zscore(inc_valid)),
        "rank": (rank_percentile(own), rank_percentile(inc_valid)),
    }

    family_best_blend = None
    family_best_blend_primary = -np.inf

    for mode, (own_rep, inc_rep) in representations.items():
        for alpha in np.linspace(0.1, 0.9, 9):
            blended = alpha * own_rep + (1.0 - alpha) * inc_rep
            metrics = evaluate(valid_users, y_valid, blended)
            primary = float(metrics["primary"])
            if primary > family_best_blend_primary:
                family_best_blend_primary = primary
                family_best_blend = {
                    "family": family_name,
                    "mode": mode,
                    "alpha": float(alpha),
                    "scores": blended.copy(),
                    "metrics": metrics,
                }

    blend_name = family_name + "_best_incumbent_blend"
    candidate_scores[blend_name] = family_best_blend_primary
    candidate_details[blend_name] = family_best_blend

    if family_best_blend_primary > best_primary:
        best_primary = family_best_blend_primary
        winner = blend_name

chosen = candidate_details[winner]
valid_scores = np.asarray(chosen["scores"], dtype=np.float64)
metrics = chosen["metrics"]
winning_family = chosen["family"]

print(
    "FINDINGS "
    + json.dumps(
        {
            "fm_epoch": int(fm_result["epoch"]),
            "nfm_epoch": int(nfm_result["epoch"]),
            "winner": winner,
            "blend_mode": chosen["mode"],
            "blend_alpha_new_model": float(chosen["alpha"]),
        },
        separators=(", ", ": "),
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, separators=(", ", ": ")))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )
    if chosen["mode"] != "standalone":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(families[winning_family]["scores"], dtype=np.float64),
        )

# Refit the selected family on train + validation, then score test.
test = load("test")
x_test = encode(test)
y_fit = np.concatenate(
    [y_train, np.asarray(valid.y, dtype=np.float32)],
    axis=0,
)

if winning_family in ("expanded_fm", "nfm"):
    x_fit = np.concatenate([x_train, x_valid], axis=0)
    selected_info = families[winning_family]
    test_model = fit_fixed_model(
        selected_info["model_class"],
        x_fit,
        y_fit,
        selected_info["epoch"],
    )
    own_test = predict_torch(test_model, x_test).astype(np.float64)
else:
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {
        field: np.concatenate(
            [
                np.asarray(train.X[field], dtype=np.int64),
                np.asarray(valid.X[field], dtype=np.int64),
            ]
        )
        for field in STAT_FIELDS
    }
    combined.date = np.concatenate(
        [
            np.asarray(train.date, dtype=np.int32),
            np.asarray(valid.date, dtype=np.int32),
        ]
    )
    combined.user_id = np.concatenate(
        [
            np.asarray(train.user_id),
            np.asarray(valid.user_id),
        ]
    )
    combined_prior, combined_tables = fit_target_statistics(combined, y_fit)
    own_test = predict_target_statistics(
        test, combined_prior, combined_tables
    ).astype(np.float64)

if chosen["mode"] == "standalone":
    test_scores = own_test
else:
    incumbent_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    alpha = float(chosen["alpha"])
    if chosen["mode"] == "raw":
        own_rep = own_test
        inc_rep = incumbent_test
    elif chosen["mode"] == "zscore":
        own_rep = zscore(own_test)
        inc_rep = zscore(incumbent_test)
    else:
        own_rep = rank_percentile(own_test)
        inc_rep = rank_percentile(incumbent_test)
    test_scores = alpha * own_rep + (1.0 - alpha) * inc_rep

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
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(", ", ": "),
    )
)