import os
import time
import json
import math
import random
import numpy as np
import torch
from torch import nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
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
USER_POS = FIELDS.index("user_id")
VIDEO_POS = FIELDS.index("video_id")
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
    columns = [
        np.asarray(split.X[name], dtype=np.int64) + OFFSETS[j]
        for j, name in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


class FieldWeightedFM(nn.Module):
    """FM whose contribution from each pair of fields is independently learned."""

    def __init__(self, cardinality, n_fields, rank=16):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.factor = nn.Embedding(cardinality, rank)
        self.bias = nn.Parameter(torch.zeros(1))

        pi, pj = np.triu_indices(n_fields, k=1)
        self.register_buffer("pair_i", torch.from_numpy(pi.astype(np.int64)))
        self.register_buffer("pair_j", torch.from_numpy(pj.astype(np.int64)))
        self.pair_weight = nn.Parameter(torch.ones(len(pi)))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factor.weight, mean=0.0, std=0.015)

    def forward(self, x):
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.factor(x)
        pair_scores = (
            v[:, self.pair_i, :] * v[:, self.pair_j, :]
        ).sum(dim=-1)
        interactions = (pair_scores * self.pair_weight).sum(dim=1)
        return self.bias + wide + interactions


class AutoIntBlock(nn.Module):
    def __init__(self, dim=12, heads=3):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch, fields, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.reshape(batch, fields, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, fields, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, fields, self.heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = torch.softmax(attention / math.sqrt(self.head_dim), dim=-1)
        z = torch.matmul(attention, v)
        z = z.transpose(1, 2).contiguous().reshape(batch, fields, dim)
        return self.norm(x + self.out(z))


class AutoIntModel(nn.Module):
    """Self-attention forms context-dependent interactions among feature fields."""

    def __init__(self, cardinality, n_fields, dim=12):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, dim)
        self.linear = nn.Embedding(cardinality, 1)
        self.block1 = AutoIntBlock(dim=dim, heads=3)
        self.block2 = AutoIntBlock(dim=dim, heads=3)
        self.output = nn.Sequential(
            nn.Linear(n_fields * dim, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = self.embedding(x)
        e = self.block1(e)
        e = self.block2(e)
        deep = self.output(e.flatten(start_dim=1)).squeeze(-1)
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + wide + deep


class CollaborativeMF(nn.Module):
    """A deliberately restricted user-video latent model."""

    def __init__(self, cardinality, rank=32):
        super().__init__()
        self.factor = nn.Embedding(cardinality, rank)
        self.bias = nn.Embedding(cardinality, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.factor.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x):
        user = x[:, USER_POS]
        video = x[:, VIDEO_POS]
        interaction = (
            self.factor(user) * self.factor(video)
        ).sum(dim=-1)
        biases = (
            self.bias(user).squeeze(-1)
            + self.bias(video).squeeze(-1)
        )
        return self.global_bias + biases + interaction


def train_model(model, x_np, y_np, epochs, batch_size, lr, seed):
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    generator = torch.Generator()
    generator.manual_seed(seed)
    n = x.shape[0]

    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for begin in range(0, n, batch_size):
            idx = order[begin:begin + batch_size]
            logits = model(x[idx])
            loss = criterion(logits, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(x.shape[0], dtype=np.float64)
    for begin in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(begin + PRED_BATCH_SIZE, x.shape[0])
        result[begin:end] = (
            model(x[begin:end]).cpu().numpy().astype(np.float64)
        )
    return result


def within_user_rank(user_ids, scores):
    """Map scores to [0,1] ranks separately within each logged user set."""
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)

    order = np.lexsort((scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    repeated_starts = np.repeat(starts, sizes)
    repeated_denoms = np.repeat(np.maximum(sizes - 1, 1), sizes)
    sorted_ranks = (
        np.arange(n, dtype=np.float64) - repeated_starts
    ) / repeated_denoms

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_ranks
    return result


train = load("train")
valid = load("valid")
x_train = build_matrix(train)
x_valid = build_matrix(valid)

model_specs = [
    (
        "fwfm",
        FieldWeightedFM(TOTAL_CARDINALITY, len(FIELDS), rank=16),
        4,
        8192,
        0.0015,
    ),
    (
        "autoint",
        AutoIntModel(TOTAL_CARDINALITY, len(FIELDS), dim=12),
        3,
        4096,
        0.0010,
    ),
    (
        "collaborative_mf",
        CollaborativeMF(TOTAL_CARDINALITY, rank=32),
        5,
        8192,
        0.0020,
    ),
]

models = {}
valid_predictions = {}

for model_index, (name, model, epochs, batch_size, lr) in enumerate(model_specs):
    torch.manual_seed(SEED + 101 * model_index)
    model = train_model(
        model,
        x_train,
        train.y,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=SEED + 1000 + model_index,
    )
    models[name] = model
    valid_predictions[name] = predict(model, x_valid)

del x_train

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError("Trusted incumbent validation predictions are missing")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
incumbent_valid_rank = within_user_rank(valid.user_id, incumbent_valid)

candidate_scores = {}
candidate_metrics = {}
candidate_own_raw = {}
candidate_recipe = {}

for name, scores in valid_predictions.items():
    metric = evaluate(valid.user_id, valid.y, scores)
    candidate_scores[name] = scores
    candidate_metrics[name] = float(metric["primary"])
    candidate_own_raw[name] = scores
    candidate_recipe[name] = ("standalone", name, 1.0)

    own_rank = within_user_rank(valid.user_id, scores)
    for own_weight in (0.20, 0.35, 0.50, 0.65, 0.80):
        raw_name = f"{name}_rawblend_w{own_weight:.2f}"
        raw_blend = (
            own_weight * scores
            + (1.0 - own_weight) * incumbent_valid
        )
        raw_metric = evaluate(valid.user_id, valid.y, raw_blend)
        candidate_scores[raw_name] = raw_blend
        candidate_metrics[raw_name] = float(raw_metric["primary"])
        candidate_own_raw[raw_name] = scores
        candidate_recipe[raw_name] = ("rawblend", name, own_weight)

        rank_name = f"{name}_rankblend_w{own_weight:.2f}"
        rank_blend = (
            own_weight * own_rank
            + (1.0 - own_weight) * incumbent_valid_rank
        )
        rank_metric = evaluate(valid.user_id, valid.y, rank_blend)
        candidate_scores[rank_name] = rank_blend
        candidate_metrics[rank_name] = float(rank_metric["primary"])
        candidate_own_raw[rank_name] = scores
        candidate_recipe[rank_name] = ("rankblend", name, own_weight)

own_ensemble_valid = np.mean(
    np.column_stack(
        [
            valid_predictions["fwfm"],
            valid_predictions["autoint"],
            valid_predictions["collaborative_mf"],
        ]
    ),
    axis=1,
)
own_ensemble_rank = within_user_rank(valid.user_id, own_ensemble_valid)

ensemble_metric = evaluate(valid.user_id, valid.y, own_ensemble_valid)
candidate_scores["own_family_ensemble"] = own_ensemble_valid
candidate_metrics["own_family_ensemble"] = float(ensemble_metric["primary"])
candidate_own_raw["own_family_ensemble"] = own_ensemble_valid
candidate_recipe["own_family_ensemble"] = ("standalone", "ensemble", 1.0)

for own_weight in (0.20, 0.35, 0.50, 0.65, 0.80):
    raw_name = f"family_ensemble_rawblend_w{own_weight:.2f}"
    raw_blend = (
        own_weight * own_ensemble_valid
        + (1.0 - own_weight) * incumbent_valid
    )
    raw_metric = evaluate(valid.user_id, valid.y, raw_blend)
    candidate_scores[raw_name] = raw_blend
    candidate_metrics[raw_name] = float(raw_metric["primary"])
    candidate_own_raw[raw_name] = own_ensemble_valid
    candidate_recipe[raw_name] = ("rawblend", "ensemble", own_weight)

    rank_name = f"family_ensemble_rankblend_w{own_weight:.2f}"
    rank_blend = (
        own_weight * own_ensemble_rank
        + (1.0 - own_weight) * incumbent_valid_rank
    )
    rank_metric = evaluate(valid.user_id, valid.y, rank_blend)
    candidate_scores[rank_name] = rank_blend
    candidate_metrics[rank_name] = float(rank_metric["primary"])
    candidate_own_raw[rank_name] = own_ensemble_valid
    candidate_recipe[rank_name] = ("rankblend", "ensemble", own_weight)

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

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if candidate_recipe[winner_name][0] != "standalone":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_own_raw[winner_name], dtype=np.float64),
        )

test = load("test")
x_test = build_matrix(test)
test_predictions = {
    name: predict(model, x_test)
    for name, model in models.items()
}

recipe_type, recipe_model, own_weight = candidate_recipe[winner_name]
if recipe_model == "ensemble":
    own_test = np.mean(
        np.column_stack(
            [
                test_predictions["fwfm"],
                test_predictions["autoint"],
                test_predictions["collaborative_mf"],
            ]
        ),
        axis=1,
    )
else:
    own_test = test_predictions[recipe_model]

if recipe_type == "standalone":
    test_scores = own_test
else:
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError("Trusted incumbent test predictions are missing")
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    if recipe_type == "rawblend":
        test_scores = (
            own_weight * own_test
            + (1.0 - own_weight) * incumbent_test
        )
    elif recipe_type == "rankblend":
        own_test_rank = within_user_rank(test.user_id, own_test)
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        test_scores = (
            own_weight * own_test_rank
            + (1.0 - own_weight) * incumbent_test_rank
        )
    else:
        raise ValueError("Unknown winner recipe")

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