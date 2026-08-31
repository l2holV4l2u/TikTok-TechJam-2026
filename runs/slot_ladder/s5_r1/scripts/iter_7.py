import os
import time
import json
import random
import numpy as np
import torch
from torch import nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
]
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets(fields):
    offsets = []
    total = 0
    for name in fields:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets(FIELDS)


def build_matrix(split):
    return np.ascontiguousarray(
        np.column_stack(
            [
                np.asarray(split.X[name], dtype=np.int64) + OFFSETS[j]
                for j, name in enumerate(FIELDS)
            ]
        ),
        dtype=np.int64,
    )


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    weights = np.exp2(-age.astype(np.float64) / float(half_life))
    weights /= weights.mean()
    return weights.astype(np.float32)


class FM(nn.Module):
    def __init__(self, cardinality, rank=16):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.factor = nn.Embedding(cardinality, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factor.weight, mean=0.0, std=0.015)

    def forward(self, x):
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.factor(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + wide + interaction


class NFM(nn.Module):
    """Neural FM: pooled second-order interaction vector is transformed nonlinearly."""

    def __init__(self, cardinality, rank=16):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.factor = nn.Embedding(cardinality, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        self.deep = nn.Sequential(
            nn.BatchNorm1d(rank),
            nn.Linear(rank, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factor.weight, mean=0.0, std=0.015)

    def forward(self, x):
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.factor(x)
        summed = v.sum(dim=1)
        bi_vector = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        )
        deep = self.deep(bi_vector).squeeze(-1)
        return self.bias + wide + deep


class BPRMatrixFactorization(nn.Module):
    """Collaborative scorer trained only through within-user pair comparisons."""

    def __init__(self, n_users, n_videos, rank=40):
        super().__init__()
        self.user_factor = nn.Embedding(n_users, rank)
        self.video_factor = nn.Embedding(n_videos, rank)
        self.video_bias = nn.Embedding(n_videos, 1)
        nn.init.normal_(self.user_factor.weight, mean=0.0, std=0.025)
        nn.init.normal_(self.video_factor.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.video_bias.weight)

    def score(self, users, videos):
        return (
            self.user_factor(users) * self.video_factor(videos)
        ).sum(dim=-1) + self.video_bias(videos).squeeze(-1)

    def forward(self, users, videos):
        return self.score(users, videos)


def train_pointwise(model, x_np, y_np, weights_np, epochs, lr, seed):
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    weights = torch.from_numpy(np.asarray(weights_np, dtype=np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    n = len(y)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for begin in range(0, n, BATCH_SIZE):
            idx = order[begin:begin + BATCH_SIZE]
            logits = model(x[idx])
            per_row = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            loss = (per_row * weights[idx]).sum() / weights[idx].sum().clamp_min(1e-8)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def make_bpr_pairs(train, n_pairs, seed):
    rng = np.random.default_rng(seed)
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    dates = np.asarray(train.date, dtype=np.int64)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    pos_count = np.bincount(users[labels == 1], minlength=n_users).astype(np.int64)
    neg_count = np.bincount(users[labels == 0], minlength=n_users).astype(np.int64)
    total_count = pos_count + neg_count

    valid_users = np.flatnonzero((pos_count > 0) & (neg_count > 0))
    sampled_users = rng.choice(valid_users, size=n_pairs, replace=True)

    # Sorting by (user, label) puts each user's negatives before positives.
    sorted_rows = np.lexsort((labels, users))
    user_starts = np.cumsum(
        np.r_[0, total_count[:-1]], dtype=np.int64
    )

    neg_offsets = (
        rng.random(n_pairs) * neg_count[sampled_users]
    ).astype(np.int64)
    pos_offsets = (
        rng.random(n_pairs) * pos_count[sampled_users]
    ).astype(np.int64)

    neg_rows = sorted_rows[user_starts[sampled_users] + neg_offsets]
    pos_rows = sorted_rows[
        user_starts[sampled_users]
        + neg_count[sampled_users]
        + pos_offsets
    ]

    max_date = dates.max()
    pair_age = max_date - np.maximum(dates[pos_rows], dates[neg_rows])
    pair_weights = np.exp2(-pair_age.astype(np.float64) / 4.0)
    pair_weights /= pair_weights.mean()

    return (
        sampled_users.astype(np.int64),
        videos[pos_rows].astype(np.int64),
        videos[neg_rows].astype(np.int64),
        pair_weights.astype(np.float32),
    )


def train_bpr(model, pair_arrays, epochs, lr, seed):
    users_np, positives_np, negatives_np, weights_np = pair_arrays
    users = torch.from_numpy(users_np)
    positives = torch.from_numpy(positives_np)
    negatives = torch.from_numpy(negatives_np)
    weights = torch.from_numpy(weights_np)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    n = len(users)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for begin in range(0, n, BATCH_SIZE):
            idx = order[begin:begin + BATCH_SIZE]
            positive_score = model.score(users[idx], positives[idx])
            negative_score = model.score(users[idx], negatives[idx])
            per_pair = nn.functional.softplus(
                -(positive_score - negative_score)
            )
            loss = (
                per_pair * weights[idx]
            ).sum() / weights[idx].sum().clamp_min(1e-8)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def predict_pointwise(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(len(x_np), dtype=np.float64)
    for begin in range(0, len(x_np), PRED_BATCH_SIZE):
        end = min(begin + PRED_BATCH_SIZE, len(x_np))
        result[begin:end] = model(x[begin:end]).cpu().numpy()
    return result


@torch.inference_mode()
def predict_bpr(model, split):
    model.eval()
    users_np = np.asarray(split.X["user_id"], dtype=np.int64)
    videos_np = np.asarray(split.X["video_id"], dtype=np.int64)
    result = np.empty(len(users_np), dtype=np.float64)

    users = torch.from_numpy(users_np)
    videos = torch.from_numpy(videos_np)
    for begin in range(0, len(users_np), PRED_BATCH_SIZE):
        end = min(begin + PRED_BATCH_SIZE, len(users_np))
        result[begin:end] = model(
            users[begin:end], videos[begin:end]
        ).cpu().numpy()
    return result


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Row index is a deterministic final tie breaker.
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    repeated_starts = np.repeat(starts, sizes)
    denominators = np.repeat(np.maximum(sizes - 1, 1), sizes)
    sorted_ranks = (
        np.arange(n, dtype=np.float64) - repeated_starts
    ) / denominators

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_ranks
    return result


train = load("train")
valid = load("valid")
x_train = build_matrix(train)
x_valid = build_matrix(valid)

models = {}
valid_predictions = {}

# The half-lives are deliberately broad enough to test strong versus mild drift correction.
fm_specs = [
    ("fm_half2_5", 2.5),
    ("fm_half4_5", 4.5),
    ("fm_half8", 8.0),
]
for model_index, (name, half_life) in enumerate(fm_specs):
    torch.manual_seed(SEED + 100 * model_index)
    model = FM(TOTAL_CARDINALITY, rank=16)
    model = train_pointwise(
        model,
        x_train,
        train.y,
        recency_weights(train.date, half_life),
        epochs=4,
        lr=0.0015,
        seed=SEED + 1000 + model_index,
    )
    models[name] = model
    valid_predictions[name] = predict_pointwise(model, x_valid)

torch.manual_seed(SEED + 500)
nfm = NFM(TOTAL_CARDINALITY, rank=16)
nfm = train_pointwise(
    nfm,
    x_train,
    train.y,
    recency_weights(train.date, 4.5),
    epochs=3,
    lr=0.0012,
    seed=SEED + 1500,
)
models["nfm_half4_5"] = nfm
valid_predictions["nfm_half4_5"] = predict_pointwise(nfm, x_valid)

# Uniform-user pair sampling aligns training with within-user evaluation rather
# than allowing heavy users to dominate pair generation.
bpr_pairs = make_bpr_pairs(train, n_pairs=1200000, seed=SEED + 2000)
torch.manual_seed(SEED + 600)
bpr = BPRMatrixFactorization(
    int(FEATURE_CARDINALITIES["user_id"]),
    int(FEATURE_CARDINALITIES["video_id"]),
    rank=40,
)
bpr = train_bpr(
    bpr,
    bpr_pairs,
    epochs=4,
    lr=0.0020,
    seed=SEED + 2500,
)
models["bpr_recency"] = bpr
valid_predictions["bpr_recency"] = predict_bpr(bpr, valid)

del x_train
del bpr_pairs

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores are missing")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
incumbent_valid_rank = within_user_rank(
    valid.user_id, incumbent_valid
)

candidate_scores = {}
candidate_metrics = {}
candidate_recipes = {}
candidate_raw = {}

for name, scores in valid_predictions.items():
    standalone_metrics = evaluate(valid.user_id, valid.y, scores)
    candidate_scores[name] = scores
    candidate_metrics[name] = float(standalone_metrics["primary"])
    candidate_recipes[name] = ("standalone", name, 1.0)
    candidate_raw[name] = scores

    own_rank = within_user_rank(valid.user_id, scores)
    for own_weight in (0.20, 0.35, 0.50, 0.65):
        candidate_name = f"{name}_rankblend_w{own_weight:.2f}"
        blended = (
            own_weight * own_rank
            + (1.0 - own_weight) * incumbent_valid_rank
        )
        blended_metrics = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[candidate_name] = blended
        candidate_metrics[candidate_name] = float(
            blended_metrics["primary"]
        )
        candidate_recipes[candidate_name] = (
            "rankblend",
            name,
            own_weight,
        )
        candidate_raw[candidate_name] = scores

# A cross-loss ensemble tests whether pairwise collaborative ordering supplies
# residual information beyond the best pointwise recency models.
pointwise_rank = np.mean(
    np.column_stack(
        [
            within_user_rank(valid.user_id, valid_predictions["fm_half2_5"]),
            within_user_rank(valid.user_id, valid_predictions["fm_half4_5"]),
            within_user_rank(valid.user_id, valid_predictions["fm_half8"]),
            within_user_rank(valid.user_id, valid_predictions["nfm_half4_5"]),
        ]
    ),
    axis=1,
)
bpr_rank = within_user_rank(
    valid.user_id, valid_predictions["bpr_recency"]
)
cross_loss_ensemble = 0.75 * pointwise_rank + 0.25 * bpr_rank
cross_loss_raw = np.mean(
    np.column_stack(
        [
            valid_predictions["fm_half2_5"],
            valid_predictions["fm_half4_5"],
            valid_predictions["fm_half8"],
            valid_predictions["nfm_half4_5"],
            valid_predictions["bpr_recency"],
        ]
    ),
    axis=1,
)

m = evaluate(valid.user_id, valid.y, cross_loss_ensemble)
candidate_scores["cross_loss_ensemble"] = cross_loss_ensemble
candidate_metrics["cross_loss_ensemble"] = float(m["primary"])
candidate_recipes["cross_loss_ensemble"] = (
    "cross_loss",
    "ensemble",
    1.0,
)
candidate_raw["cross_loss_ensemble"] = cross_loss_raw

for own_weight in (0.20, 0.35, 0.50, 0.65):
    name = f"cross_loss_incumbent_rankblend_w{own_weight:.2f}"
    blended = (
        own_weight * cross_loss_ensemble
        + (1.0 - own_weight) * incumbent_valid_rank
    )
    m = evaluate(valid.user_id, valid.y, blended)
    candidate_scores[name] = blended
    candidate_metrics[name] = float(m["primary"])
    candidate_recipes[name] = (
        "cross_loss_incumbent",
        "ensemble",
        own_weight,
    )
    candidate_raw[name] = cross_loss_raw

winner_name = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = candidate_scores[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_metrics.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "best_candidate": winner_name,
            "best_standalone": max(
                valid_predictions,
                key=lambda k: candidate_metrics[k],
            ),
            "best_standalone_primary": max(
                candidate_metrics[k] for k in valid_predictions
            ),
        },
        separators=(",", ":"),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if candidate_recipes[winner_name][0] != "standalone":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner_name], dtype=np.float64),
        )

test = load("test")
x_test = build_matrix(test)

recipe_type, recipe_model, own_weight = candidate_recipes[winner_name]

if recipe_model != "ensemble":
    if recipe_model == "bpr_recency":
        own_test_raw = predict_bpr(models[recipe_model], test)
    else:
        own_test_raw = predict_pointwise(
            models[recipe_model], x_test
        )
else:
    test_raw_predictions = {
        "fm_half2_5": predict_pointwise(models["fm_half2_5"], x_test),
        "fm_half4_5": predict_pointwise(models["fm_half4_5"], x_test),
        "fm_half8": predict_pointwise(models["fm_half8"], x_test),
        "nfm_half4_5": predict_pointwise(models["nfm_half4_5"], x_test),
        "bpr_recency": predict_bpr(models["bpr_recency"], test),
    }
    pointwise_test_rank = np.mean(
        np.column_stack(
            [
                within_user_rank(test.user_id, test_raw_predictions["fm_half2_5"]),
                within_user_rank(test.user_id, test_raw_predictions["fm_half4_5"]),
                within_user_rank(test.user_id, test_raw_predictions["fm_half8"]),
                within_user_rank(test.user_id, test_raw_predictions["nfm_half4_5"]),
            ]
        ),
        axis=1,
    )
    bpr_test_rank = within_user_rank(
        test.user_id, test_raw_predictions["bpr_recency"]
    )
    own_test_rank = (
        0.75 * pointwise_test_rank + 0.25 * bpr_test_rank
    )
    own_test_raw = np.mean(
        np.column_stack(list(test_raw_predictions.values())),
        axis=1,
    )

if recipe_type == "standalone":
    test_scores = own_test_raw
elif recipe_type == "rankblend":
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError("Trusted incumbent test scores are missing")
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    test_scores = (
        own_weight * within_user_rank(test.user_id, own_test_raw)
        + (1.0 - own_weight)
        * within_user_rank(test.user_id, incumbent_test)
    )
elif recipe_type == "cross_loss":
    test_scores = own_test_rank
elif recipe_type == "cross_loss_incumbent":
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError("Trusted incumbent test scores are missing")
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    test_scores = (
        own_weight * own_test_rank
        + (1.0 - own_weight)
        * within_user_rank(test.user_id, incumbent_test)
    )
else:
    raise ValueError("Unknown candidate recipe")

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
        separators=(",", ":"),
    )
)