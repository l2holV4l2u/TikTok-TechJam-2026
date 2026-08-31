import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

BATCH_SIZE = 8192
EPOCHS = 5
LR = 0.002

WIDE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "hour",
    "user_active_degree",
    "video_type",
    "is_video_author",
]

GLOBAL_RATE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat8",
]

GLOBAL_SMOOTHING = {
    "video_id": 20.0,
    "author_id": 40.0,
    "tag": 300.0,
    "duration_bucket": 1000.0,
    "upload_type": 800.0,
    "onehot_feat8": 200.0,
}

GLOBAL_WEIGHTS = {
    "video_id": 0.45,
    "author_id": 0.25,
    "tag": 0.12,
    "duration_bucket": 0.08,
    "upload_type": 0.05,
    "onehot_feat8": 0.05,
}

AFFINITY_FIELDS = [
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "tab",
]

AFFINITY_WEIGHTS = {
    "tag": 0.30,
    "duration_bucket": 0.20,
    "upload_type": 0.15,
    "music_type": 0.15,
    "tab": 0.20,
}


def sigmoid_np(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit_np(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def make_wide_matrix(split, offsets):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) + off
            for f, off in zip(WIDE_FIELDS, offsets)
        ]),
        dtype=np.int64,
    )


class WideModel(nn.Module):
    def __init__(self, n_tokens):
        super().__init__()
        self.weight = nn.Embedding(n_tokens, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.weight.weight)

    def forward(self, x):
        return self.bias + self.weight(x).sum(dim=1).squeeze(-1)


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_videos, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.video_emb = nn.Embedding(n_videos, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.video_bias = nn.Embedding(n_videos, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight, std=0.025)
        nn.init.normal_(self.video_emb.weight, std=0.025)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.video_bias.weight)

    def forward(self, users, videos):
        dot = (self.user_emb(users) * self.video_emb(videos)).sum(dim=1)
        ub = self.user_bias(users).squeeze(-1)
        vb = self.video_bias(videos).squeeze(-1)
        return self.bias + ub + vb + dot


def predict_wide(model, matrix):
    model.eval()
    out = np.empty(len(matrix), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(matrix), 32768):
            end = min(start + 32768, len(matrix))
            xb = torch.from_numpy(matrix[start:end])
            out[start:end] = torch.sigmoid(model(xb)).numpy()
    return out


def predict_mf(model, users, videos):
    model.eval()
    out = np.empty(len(users), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(users), 32768):
            end = min(start + 32768, len(users))
            ub = torch.from_numpy(users[start:end])
            vb = torch.from_numpy(videos[start:end])
            out[start:end] = torch.sigmoid(model(ub, vb)).numpy()
    return out


def build_smoothed_rate(values, labels, cardinality, prior, smoothing):
    values = np.asarray(values, dtype=np.int64)
    count = np.bincount(values, minlength=cardinality).astype(np.float64)
    positive = np.bincount(
        values, weights=labels, minlength=cardinality
    ).astype(np.float64)
    return ((positive + smoothing * prior) /
            (count + smoothing)).astype(np.float32)


def build_affinity_rate(users, context, labels, user_card, context_card,
                        prior, smoothing=8.0):
    keys = (
        np.asarray(users, dtype=np.int64) * int(context_card)
        + np.asarray(context, dtype=np.int64)
    )
    total_card = int(user_card) * int(context_card)
    count = np.bincount(keys, minlength=total_card).astype(np.float32)
    positive = np.bincount(
        keys, weights=labels, minlength=total_card
    ).astype(np.float32)
    rate = (positive + np.float32(smoothing * prior)) / (
        count + np.float32(smoothing)
    )
    return rate.astype(np.float32, copy=False)


train = load("train")
y_train = np.asarray(train.y, dtype=np.float32)
prior = float(y_train.mean())

wide_cards = [int(FEATURE_CARDINALITIES[f]) for f in WIDE_FIELDS]
wide_offsets = np.cumsum([0] + wide_cards[:-1], dtype=np.int64)
wide_tokens = int(sum(wide_cards))
x_wide_train = make_wide_matrix(train, wide_offsets)

u_train = np.asarray(train.X["user_id"], dtype=np.int64)
v_train = np.asarray(train.X["video_id"], dtype=np.int64)
n_users = int(FEATURE_CARDINALITIES["user_id"])
n_videos = int(FEATURE_CARDINALITIES["video_id"])

wide_model = WideModel(wide_tokens)
mf_model = MatrixFactorization(n_users, n_videos, dim=32)

wide_optimizer = torch.optim.Adam(
    wide_model.parameters(), lr=LR, weight_decay=1e-7
)
mf_optimizer = torch.optim.Adam(
    mf_model.parameters(), lr=LR, weight_decay=2e-6
)

generator = torch.Generator()
generator.manual_seed(SEED)
n_train = len(y_train)

for epoch in range(EPOCHS):
    permutation = torch.randperm(n_train, generator=generator)
    wide_model.train()
    mf_model.train()

    for start in range(0, n_train, BATCH_SIZE):
        idx_t = permutation[start:start + BATCH_SIZE]
        idx = idx_t.numpy()

        yb = torch.from_numpy(y_train[idx])

        wide_optimizer.zero_grad(set_to_none=True)
        wide_logits = wide_model(torch.from_numpy(x_wide_train[idx]))
        wide_loss = F.binary_cross_entropy_with_logits(wide_logits, yb)
        wide_loss.backward()
        wide_optimizer.step()

        mf_optimizer.zero_grad(set_to_none=True)
        mf_logits = mf_model(
            torch.from_numpy(u_train[idx]),
            torch.from_numpy(v_train[idx]),
        )
        mf_loss = F.binary_cross_entropy_with_logits(mf_logits, yb)
        mf_loss.backward()
        mf_optimizer.step()


global_rates = {}
for field in GLOBAL_RATE_FIELDS:
    global_rates[field] = build_smoothed_rate(
        train.X[field],
        y_train,
        int(FEATURE_CARDINALITIES[field]),
        prior,
        GLOBAL_SMOOTHING[field],
    )

affinity_rates = {}
for field in AFFINITY_FIELDS:
    affinity_rates[field] = build_affinity_rate(
        u_train,
        train.X[field],
        y_train,
        n_users,
        int(FEATURE_CARDINALITIES[field]),
        prior,
        smoothing=8.0,
    )


def empirical_bayes_scores(split):
    score_logit = np.zeros(len(split.user_id), dtype=np.float64)
    for field in GLOBAL_RATE_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        score_logit += GLOBAL_WEIGHTS[field] * logit_np(
            global_rates[field][ids]
        )
    return sigmoid_np(score_logit).astype(np.float32)


def affinity_scores(split):
    users = np.asarray(split.X["user_id"], dtype=np.int64)

    context_logit = np.zeros(len(users), dtype=np.float64)
    for field in AFFINITY_FIELDS:
        context_card = int(FEATURE_CARDINALITIES[field])
        context = np.asarray(split.X[field], dtype=np.int64)
        keys = users * context_card + context
        context_logit += AFFINITY_WEIGHTS[field] * logit_np(
            affinity_rates[field][keys]
        )

    video_ids = np.asarray(split.X["video_id"], dtype=np.int64)
    author_ids = np.asarray(split.X["author_id"], dtype=np.int64)
    item_logit = (
        0.65 * logit_np(global_rates["video_id"][video_ids])
        + 0.35 * logit_np(global_rates["author_id"][author_ids])
    )

    return sigmoid_np(0.58 * item_logit + 0.42 * context_logit).astype(
        np.float32
    )


valid = load("valid")
x_wide_valid = make_wide_matrix(valid, wide_offsets)
valid_users = np.asarray(valid.X["user_id"], dtype=np.int64)
valid_videos = np.asarray(valid.X["video_id"], dtype=np.int64)

own_valid = {
    "wide_additive": predict_wide(wide_model, x_wide_valid),
    "latent_mf": predict_mf(mf_model, valid_users, valid_videos),
    "empirical_bayes": empirical_bayes_scores(valid),
    "user_context_affinity": affinity_scores(valid),
}
own_valid["wide_mf_ensemble"] = (
    0.50 * own_valid["wide_additive"]
    + 0.50 * own_valid["latent_mf"]
).astype(np.float32)
own_valid["bayes_affinity_ensemble"] = (
    0.55 * own_valid["empirical_bayes"]
    + 0.45 * own_valid["user_context_affinity"]
).astype(np.float32)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")
inc_valid_raw = np.load(inc_valid_path)
inc_valid_prob = sigmoid_np(inc_valid_raw)

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_spec = None
best_valid_scores = None
best_own_valid = None

blend_weights = [0.15, 0.30, 0.45, 0.60, 0.75]

for family, scores in own_valid.items():
    standalone_metrics = evaluate(valid.user_id, valid.y, scores)
    standalone_name = family + "_standalone"
    candidate_scores[standalone_name] = float(standalone_metrics["primary"])

    if standalone_metrics["primary"] > best_primary:
        best_primary = float(standalone_metrics["primary"])
        best_metrics = standalone_metrics
        best_spec = (family, 1.0)
        best_valid_scores = np.asarray(scores, dtype=np.float64)
        best_own_valid = np.asarray(scores, dtype=np.float64)

    for own_weight in blend_weights:
        blended = (
            own_weight * np.asarray(scores, dtype=np.float64)
            + (1.0 - own_weight) * inc_valid_prob
        )
        metrics = evaluate(valid.user_id, valid.y, blended)
        name = family + "_blend_" + str(own_weight)
        candidate_scores[name] = float(metrics["primary"])

        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_spec = (family, own_weight)
            best_valid_scores = blended
            best_own_valid = np.asarray(scores, dtype=np.float64)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "winner_family": best_spec[0],
        "winner_own_weight": float(best_spec[1]),
        "prior": prior,
    }, sort_keys=True)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    if best_spec[1] < 1.0:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_own_valid, dtype=np.float64),
        )

del x_wide_valid
test = load("test")
x_wide_test = make_wide_matrix(test, wide_offsets)
test_users = np.asarray(test.X["user_id"], dtype=np.int64)
test_videos = np.asarray(test.X["video_id"], dtype=np.int64)

winner_family, winner_weight = best_spec

if winner_family == "wide_additive":
    own_test_scores = predict_wide(wide_model, x_wide_test)
elif winner_family == "latent_mf":
    own_test_scores = predict_mf(mf_model, test_users, test_videos)
elif winner_family == "empirical_bayes":
    own_test_scores = empirical_bayes_scores(test)
elif winner_family == "user_context_affinity":
    own_test_scores = affinity_scores(test)
elif winner_family == "wide_mf_ensemble":
    wide_test = predict_wide(wide_model, x_wide_test)
    mf_test = predict_mf(mf_model, test_users, test_videos)
    own_test_scores = (0.50 * wide_test + 0.50 * mf_test).astype(np.float32)
elif winner_family == "bayes_affinity_ensemble":
    bayes_test = empirical_bayes_scores(test)
    affinity_test = affinity_scores(test)
    own_test_scores = (
        0.55 * bayes_test + 0.45 * affinity_test
    ).astype(np.float32)
else:
    raise RuntimeError("Unknown winner family: " + winner_family)

if winner_weight < 1.0:
    inc_test_raw = np.load(inc_test_path)
    inc_test_prob = sigmoid_np(inc_test_raw)
    test_scores = (
        winner_weight * np.asarray(own_test_scores, dtype=np.float64)
        + (1.0 - winner_weight) * inc_test_prob
    )
else:
    test_scores = np.asarray(own_test_scores, dtype=np.float64)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))