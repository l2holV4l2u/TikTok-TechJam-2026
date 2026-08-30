import os
import time
import json
import gc
import math
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 24029
BATCH_SIZE = 8192
DEVICE = torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

DEEP_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
MF_FIELDS = ["user_id", "video_id", "author_id"]
HALF_LIFE = 7.0


def day_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    # All fitting windows end in April, so day-of-month differences are exact.
    day = dates % 100
    age = day.max() - day
    w = np.exp2(-age.astype(np.float32) / HALF_LIFE)
    return w / np.mean(w)


def matrix(split, fields):
    return np.stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in fields], axis=1
    )


def predict_torch(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float64)
    x = torch.from_numpy(np.ascontiguousarray(x_np))
    with torch.inference_mode():
        for lo in range(0, len(x), BATCH_SIZE * 2):
            hi = min(len(x), lo + BATCH_SIZE * 2)
            result[lo:hi] = model(x[lo:hi]).cpu().numpy().astype(np.float64)
    return result


def fit_torch(model, x_np, y_np, weights, epochs, seed):
    torch.manual_seed(seed)
    model.to(DEVICE)
    model.train()
    x = torch.from_numpy(np.ascontiguousarray(x_np))
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    w = torch.from_numpy(np.asarray(weights, dtype=np.float32))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-6
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = len(x)

    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:lo + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            loss = (losses * w[idx]).sum() / w[idx].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


class LatentMF(nn.Module):
    """Separate user-video and user-author latent preference channels."""
    def __init__(self, cards, dim=24):
        super().__init__()
        uc, vc, ac = cards
        self.user_video = nn.Embedding(uc, dim)
        self.video = nn.Embedding(vc, dim)
        self.user_author = nn.Embedding(uc, dim)
        self.author = nn.Embedding(ac, dim)
        self.user_bias = nn.Embedding(uc, 1)
        self.video_bias = nn.Embedding(vc, 1)
        self.author_bias = nn.Embedding(ac, 1)
        self.bias = nn.Parameter(torch.zeros(()))

        for emb in [self.user_video, self.video, self.user_author, self.author]:
            nn.init.normal_(emb.weight, std=0.03)
        for emb in [self.user_bias, self.video_bias, self.author_bias]:
            nn.init.zeros_(emb.weight)

    def forward(self, x):
        u, v, a = x[:, 0], x[:, 1], x[:, 2]
        uv = (self.user_video(u) * self.video(v)).sum(dim=1)
        ua = (self.user_author(u) * self.author(a)).sum(dim=1)
        b = (
            self.user_bias(u).squeeze(1)
            + self.video_bias(v).squeeze(1)
            + self.author_bias(a).squeeze(1)
        )
        return self.bias + b + uv + ua


class NFM(nn.Module):
    """Nonlinear transformation of pooled pairwise field interactions."""
    def __init__(self, cards, dim=16):
        super().__init__()
        offsets = np.cumsum([0] + cards[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        total = int(sum(cards))
        self.linear = nn.Embedding(total, 1)
        self.embedding = nn.Embedding(total, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        z = x + self.offsets
        linear = self.linear(z).sum(dim=1).squeeze(1)
        e = self.embedding(z)
        pooled = 0.5 * (e.sum(dim=1).square() - e.square().sum(dim=1))
        nonlinear = self.mlp(pooled).squeeze(1)
        return self.bias + linear + nonlinear


def clipped_logit(p):
    return np.log(np.clip(p, 1e-4, 1.0 - 1e-4) /
                  np.clip(1.0 - p, 1e-4, 1.0))


class DirectRate:
    def __init__(self, cardinality, prior, smoothing):
        self.cardinality = int(cardinality)
        self.prior = float(prior)
        self.smoothing = float(smoothing)
        self.rate = None

    def fit(self, ids, y, w):
        ids = np.asarray(ids, dtype=np.int64)
        sw = np.bincount(ids, weights=w, minlength=self.cardinality)
        sy = np.bincount(ids, weights=w * y, minlength=self.cardinality)
        self.rate = (
            sy + self.smoothing * self.prior
        ) / (sw + self.smoothing)
        return self

    def predict(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        ok = (ids >= 0) & (ids < len(self.rate))
        out = np.full(len(ids), self.prior, dtype=np.float64)
        out[ok] = self.rate[ids[ok]]
        return out


class PairRate:
    def __init__(self, right_cardinality, prior, smoothing):
        self.right_cardinality = int(right_cardinality)
        self.prior = float(prior)
        self.smoothing = float(smoothing)
        self.keys = None
        self.rates = None

    def fit(self, left, right, y, w):
        keys = (
            np.asarray(left, dtype=np.int64) * self.right_cardinality
            + np.asarray(right, dtype=np.int64)
        )
        unique, inverse = np.unique(keys, return_inverse=True)
        sw = np.bincount(inverse, weights=w)
        sy = np.bincount(inverse, weights=w * y)
        self.keys = unique
        self.rates = (
            sy + self.smoothing * self.prior
        ) / (sw + self.smoothing)
        return self

    def predict(self, left, right):
        query = (
            np.asarray(left, dtype=np.int64) * self.right_cardinality
            + np.asarray(right, dtype=np.int64)
        )
        pos = np.searchsorted(self.keys, query)
        safe = np.minimum(pos, len(self.keys) - 1)
        found = (pos < len(self.keys)) & (self.keys[safe] == query)
        out = np.full(len(query), self.prior, dtype=np.float64)
        out[found] = self.rates[pos[found]]
        return out


class EmpiricalBayesRanker:
    def fit(self, split, y, w):
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        self.prior = float(np.sum(w * y) / np.sum(w))

        self.user = DirectRate(
            FEATURE_CARDINALITIES["user_id"], self.prior, 30.0
        ).fit(split.X["user_id"], y, w)
        self.video = DirectRate(
            FEATURE_CARDINALITIES["video_id"], self.prior, 40.0
        ).fit(split.X["video_id"], y, w)
        self.author = DirectRate(
            FEATURE_CARDINALITIES["author_id"], self.prior, 60.0
        ).fit(split.X["author_id"], y, w)
        self.tag = DirectRate(
            FEATURE_CARDINALITIES["tag"], self.prior, 150.0
        ).fit(split.X["tag"], y, w)

        self.user_author = PairRate(
            FEATURE_CARDINALITIES["author_id"], self.prior, 12.0
        ).fit(split.X["user_id"], split.X["author_id"], y, w)
        self.user_tag = PairRate(
            FEATURE_CARDINALITIES["tag"], self.prior, 18.0
        ).fit(split.X["user_id"], split.X["tag"], y, w)
        return self

    def predict(self, split):
        base = clipped_logit(self.prior)
        user = clipped_logit(self.user.predict(split.X["user_id"])) - base
        video = clipped_logit(self.video.predict(split.X["video_id"])) - base
        author = clipped_logit(self.author.predict(split.X["author_id"])) - base
        tag = clipped_logit(self.tag.predict(split.X["tag"])) - base
        ua = clipped_logit(
            self.user_author.predict(split.X["user_id"], split.X["author_id"])
        ) - base
        ut = clipped_logit(
            self.user_tag.predict(split.X["user_id"], split.X["tag"])
        ) - base

        return (
            base
            + 0.25 * user
            + 0.80 * video
            + 0.35 * author
            + 0.15 * tag
            + 0.65 * ua
            + 0.35 * ut
        )


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n), scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    group = np.cumsum(starts_mask) - 1
    positions = np.arange(n) - starts[group]
    sizes = np.diff(np.r_[starts, n])
    denom = np.maximum(sizes[group] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denom
    ranked_sorted[sizes[group] == 1] = 0.5
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def best_incumbent_blend(user_ids, labels, own, incumbent):
    own_rank = within_user_rank(user_ids, own)
    inc_rank = within_user_rank(user_ids, incumbent)
    best = None
    for alpha in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]:
        score = alpha * own_rank + (1.0 - alpha) * inc_rank
        met = evaluate(user_ids, labels, score)
        if best is None or met["primary"] > best[0]:
            best = (float(met["primary"]), float(alpha), score, met)
    return best


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
train_weights = day_weights(train.date)

x_mf_train = matrix(train, MF_FIELDS)
x_mf_valid = matrix(valid, MF_FIELDS)
mf = LatentMF(
    [int(FEATURE_CARDINALITIES[f]) for f in MF_FIELDS], dim=24
)
mf = fit_torch(
    mf, x_mf_train, y_train, train_weights, epochs=4, seed=SEED
)
pred_mf = predict_torch(mf, x_mf_valid)
del mf
gc.collect()

x_nfm_train = matrix(train, DEEP_FIELDS)
x_nfm_valid = matrix(valid, DEEP_FIELDS)
nfm = NFM(
    [int(FEATURE_CARDINALITIES[f]) for f in DEEP_FIELDS], dim=16
)
nfm = fit_torch(
    nfm, x_nfm_train, y_train, train_weights, epochs=3, seed=SEED + 1
)
pred_nfm = predict_torch(nfm, x_nfm_valid)
del nfm
gc.collect()

eb = EmpiricalBayesRanker().fit(train, y_train, train_weights)
pred_eb = eb.predict(valid)
del eb
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)

family_predictions = {
    "recency_latent_mf": pred_mf,
    "recency_nfm": pred_nfm,
    "recency_empirical_bayes": pred_eb,
}

candidate_summary = {}
raw_summary = {}
best_choice = None

for name, prediction in family_predictions.items():
    raw_met = evaluate(valid.user_id, valid.y, prediction)
    raw_summary[name] = float(raw_met["primary"])
    blended = best_incumbent_blend(
        valid.user_id, valid.y, prediction, inc_valid
    )
    candidate_summary[name + "_blend"] = float(blended[0])
    if best_choice is None or blended[0] > best_choice["primary"]:
        best_choice = {
            "family": name,
            "primary": blended[0],
            "alpha": blended[1],
            "scores": blended[2],
            "metrics": blended[3],
            "raw": prediction,
        }

print("FINDINGS raw_family_primary=" + json.dumps(raw_summary, sort_keys=True))
print(
    "FINDINGS selected_family=%s own_rank_weight=%.2f"
    % (best_choice["family"], best_choice["alpha"])
)
print("CANDIDATES " + json.dumps(candidate_summary, sort_keys=True))

valid_scores = best_choice["scores"]
metrics = best_choice["metrics"]
out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_choice["raw"], dtype=np.float64),
    )

# Permitted refit on train + validation using the selected family's identical recipe.
y_valid = np.asarray(valid.y, dtype=np.float32)
combined_y = np.concatenate([y_train, y_valid])
combined_dates = np.concatenate([
    np.asarray(train.date), np.asarray(valid.date)
])
combined_weights = day_weights(combined_dates)

test = load("test")
selected = best_choice["family"]

if selected == "recency_latent_mf":
    x_combined = np.concatenate([x_mf_train, x_mf_valid], axis=0)
    model = LatentMF(
        [int(FEATURE_CARDINALITIES[f]) for f in MF_FIELDS], dim=24
    )
    model = fit_torch(
        model, x_combined, combined_y, combined_weights,
        epochs=4, seed=SEED
    )
    own_test = predict_torch(model, matrix(test, MF_FIELDS))

elif selected == "recency_nfm":
    x_combined = np.concatenate([x_nfm_train, x_nfm_valid], axis=0)
    model = NFM(
        [int(FEATURE_CARDINALITIES[f]) for f in DEEP_FIELDS], dim=16
    )
    model = fit_torch(
        model, x_combined, combined_y, combined_weights,
        epochs=3, seed=SEED + 1
    )
    own_test = predict_torch(model, matrix(test, DEEP_FIELDS))

else:
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {
        name: np.concatenate([
            np.asarray(train.X[name]), np.asarray(valid.X[name])
        ])
        for name in [
            "user_id", "video_id", "author_id", "tag"
        ]
    }
    model = EmpiricalBayesRanker().fit(
        combined, combined_y, combined_weights
    )
    own_test = model.predict(test)

inc_test = np.load(inc_test_path).astype(np.float64)
alpha = best_choice["alpha"]
test_scores = (
    alpha * within_user_rank(test.user_id, own_test)
    + (1.0 - alpha) * within_user_rank(test.user_id, inc_test)
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)