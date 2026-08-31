import os
import time
import json
import math
import random
import datetime
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 2026
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
BATCH_SIZE = 8192
MAX_EPOCHS = 12
CHECK_EPOCHS = {6, 8, 10, 12}
LEARNING_RATE = 0.001
HALF_LIVES = [None, 3.0, 7.0, 14.0]
BLEND_ALPHAS = [0.25, 0.50, 0.75, 1.0]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) + offsets[j]
            for j, f in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def date_ordinals(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique = np.unique(dates)
    lut = {}
    for value in unique:
        x = int(value)
        lut[x] = datetime.date(x // 10000, (x // 100) % 100, x % 100).toordinal()
    return np.asarray([lut[int(x)] for x in dates], dtype=np.int32)


def temporal_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    ords = date_ordinals(dates)
    age = ords.max() - ords
    w = np.exp2(-age.astype(np.float64) / float(half_life))
    w /= w.mean()
    return w.astype(np.float32)


class AdditiveModel(nn.Module):
    def __init__(self, initial_rate):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        p = float(np.clip(initial_rate, 1e-5, 1 - 1e-5))
        self.bias = nn.Parameter(torch.tensor(math.log(p / (1 - p)), dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(-1)


class FMModel(nn.Module):
    def __init__(self, initial_rate):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, K)
        p = float(np.clip(initial_rate, 1e-5, 1 - 1e-5))
        self.bias = nn.Parameter(torch.tensor(math.log(p / (1 - p)), dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        e = self.embedding(x)
        summed = e.sum(dim=1)
        interactions = 0.5 * (
            summed.square() - e.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interactions


class FieldWeightedFM(nn.Module):
    def __init__(self, initial_rate):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, K)
        self.pair_weight = nn.Parameter(torch.ones(n_fields, n_fields))
        p = float(np.clip(initial_rate, 1e-5, 1 - 1e-5))
        self.bias = nn.Parameter(torch.tensor(math.log(p / (1 - p)), dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        e = self.embedding(x)
        interaction = torch.zeros(x.shape[0], dtype=e.dtype, device=e.device)
        for i in range(n_fields):
            for j in range(i + 1, n_fields):
                interaction = interaction + self.pair_weight[i, j] * (
                    e[:, i, :] * e[:, j, :]
                ).sum(dim=1)
        return self.bias + linear + interaction


def create_model(family, initial_rate):
    if family == "additive":
        return AdditiveModel(initial_rate)
    if family == "fm":
        return FMModel(initial_rate)
    if family == "fwfm":
        return FieldWeightedFM(initial_rate)
    raise ValueError(family)


@torch.inference_mode()
def predict(model, X):
    model.eval()
    xt = torch.from_numpy(X)
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), 32768):
        end = min(start + 32768, len(X))
        out[start:end] = model(xt[start:end]).cpu().numpy()
    return out


def best_score_variant(raw_scores, incumbent_scores, users, labels):
    best = None
    all_results = {}
    for alpha in BLEND_ALPHAS:
        scores = alpha * raw_scores + (1.0 - alpha) * incumbent_scores
        met = evaluate(users, labels, scores)
        all_results[alpha] = (float(met["primary"]), scores, met)
        if best is None or met["primary"] > best[0]:
            best = (
                float(met["primary"]),
                float(alpha),
                np.asarray(scores, dtype=np.float64),
                met,
            )
    return best, all_results


def fit_and_select(family, half_life, X, y, dates, Xv, yv, uv, incumbent_valid):
    torch.manual_seed(SEED)
    weights = temporal_weights(dates, half_life)
    weighted_rate = float(np.sum(weights * y) / np.sum(weights))
    model = create_model(family, weighted_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    xt = torch.from_numpy(X)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    wt = torch.from_numpy(weights)
    generator = torch.Generator().manual_seed(SEED)

    best = None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        for start in range(0, len(X), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(xt[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            loss = (losses * wt[idx]).sum() / wt[idx].sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch in CHECK_EPOCHS:
            raw = predict(model, Xv).astype(np.float64)
            blend_best, _ = best_score_variant(raw, incumbent_valid, uv, yv)
            if best is None or blend_best[0] > best["primary"]:
                best = {
                    "primary": blend_best[0],
                    "alpha": blend_best[1],
                    "scores": blend_best[2].copy(),
                    "metrics": blend_best[3],
                    "epoch": epoch,
                    "state": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "raw": raw.copy(),
                }

    model.load_state_dict(best["state"])
    return model, best


def fit_fixed(family, half_life, epochs, X, y, dates):
    torch.manual_seed(SEED)
    weights = temporal_weights(dates, half_life)
    weighted_rate = float(np.sum(weights * y) / np.sum(weights))
    model = create_model(family, weighted_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    xt = torch.from_numpy(X)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    wt = torch.from_numpy(weights)
    generator = torch.Generator().manual_seed(SEED)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        for start in range(0, len(X), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(xt[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            loss = (losses * wt[idx]).sum() / wt[idx].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def empirical_bayes_fit_predict(train_split, y, target_split, half_life):
    weights = temporal_weights(train_split.date, half_life).astype(np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    global_rate = float(np.sum(weights * y64) / np.sum(weights))
    global_logit = math.log(
        np.clip(global_rate, 1e-6, 1 - 1e-6) /
        np.clip(1 - global_rate, 1e-6, 1)
    )

    score = np.zeros(len(target_split.user_id), dtype=np.float64)
    used = 0
    for field, smoothing, coefficient in [
        ("video_id", 25.0, 1.0),
        ("author_id", 35.0, 0.7),
        ("tab", 80.0, 0.5),
        ("duration_bucket", 100.0, 0.35),
    ]:
        n = int(FEATURE_CARDINALITIES[field])
        ids = np.asarray(train_split.X[field], dtype=np.int64)
        target_ids = np.asarray(target_split.X[field], dtype=np.int64)
        count = np.bincount(ids, weights=weights, minlength=n)
        positive = np.bincount(ids, weights=weights * y64, minlength=n)
        rate = (positive + smoothing * global_rate) / (count + smoothing)
        rate = np.clip(rate, 1e-5, 1 - 1e-5)
        logits = np.log(rate / (1.0 - rate)) - global_logit
        score += coefficient * logits[target_ids]
        used += 1
    return score / max(used, 1)


shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

train = load("train")
valid = load("valid")
X_train = make_matrix(train)
X_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
train_dates = np.asarray(train.date)

candidate_scores = {}
records = []

# Broad temporal sweep for the factorization family.
for half_life in HALF_LIVES:
    model, result = fit_and_select(
        "fm", half_life, X_train, y_train, train_dates,
        X_valid, y_valid, valid_users, incumbent_valid
    )
    name = "fm_hl_" + ("none" if half_life is None else str(int(half_life)))
    candidate_scores[name] = result["primary"]
    records.append({
        "name": name,
        "family": "fm",
        "half_life": half_life,
        **result,
    })
    del model

# Structurally different cheap families, each with a stationary and recency-weighted version.
for family in ["additive", "fwfm"]:
    for half_life in [None, 7.0]:
        model, result = fit_and_select(
            family, half_life, X_train, y_train, train_dates,
            X_valid, y_valid, valid_users, incumbent_valid
        )
        name = family + "_hl_" + ("none" if half_life is None else "7")
        candidate_scores[name] = result["primary"]
        records.append({
            "name": name,
            "family": family,
            "half_life": half_life,
            **result,
        })
        del model

# Non-parametric family under the same temporal weighting hypothesis.
for half_life in [None, 3.0, 7.0, 14.0]:
    raw = empirical_bayes_fit_predict(train, y_train, valid, half_life)
    blend_best, _ = best_score_variant(
        raw, incumbent_valid, valid_users, y_valid
    )
    name = "empirical_bayes_hl_" + (
        "none" if half_life is None else str(int(half_life))
    )
    candidate_scores[name] = blend_best[0]
    records.append({
        "name": name,
        "family": "empirical_bayes",
        "half_life": half_life,
        "primary": blend_best[0],
        "alpha": blend_best[1],
        "scores": blend_best[2],
        "metrics": blend_best[3],
        "epoch": 0,
    })

winner = max(records, key=lambda z: z["primary"])
valid_scores = np.asarray(winner["scores"], dtype=np.float64)
metrics = evaluate(valid_users, y_valid, valid_scores)

print("CANDIDATES " + json.dumps({
    k: float(v) for k, v in candidate_scores.items()
}, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": winner["name"],
    "blend_alpha_new_family": float(winner["alpha"]),
    "selected_epoch": int(winner["epoch"]),
    "half_life_days": winner["half_life"],
}, sort_keys=True))

# Refit the selected recipe on train + validation and score test.
test = load("test")
incumbent_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

if winner["family"] == "empirical_bayes":
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {
        f: np.concatenate([
            np.asarray(train.X[f], dtype=np.int64),
            np.asarray(valid.X[f], dtype=np.int64)
        ])
        for f in FIELDS
    }
    combined.date = np.concatenate([
        np.asarray(train.date), np.asarray(valid.date)
    ])
    combined.user_id = np.concatenate([
        np.asarray(train.user_id), np.asarray(valid.user_id)
    ])
    y_combined = np.concatenate([
        y_train, y_valid.astype(np.float32)
    ])
    raw_test = empirical_bayes_fit_predict(
        combined, y_combined, test, winner["half_life"]
    )
else:
    X_combined = np.concatenate([X_train, X_valid], axis=0)
    y_combined = np.concatenate([
        y_train, y_valid.astype(np.float32)
    ])
    dates_combined = np.concatenate([
        np.asarray(train.date), np.asarray(valid.date)
    ])
    final_model = fit_fixed(
        winner["family"],
        winner["half_life"],
        int(winner["epoch"]),
        X_combined,
        y_combined,
        dates_combined,
    )
    X_test = make_matrix(test)
    raw_test = predict(final_model, X_test).astype(np.float64)

alpha = float(winner["alpha"])
test_scores = alpha * raw_test + (1.0 - alpha) * incumbent_test

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "scores_valid.npy"), valid_scores)
    np.save(os.path.join(out, "scores_test.npy"), np.asarray(test_scores, dtype=np.float64))

wall = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(wall),
}))