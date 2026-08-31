import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260831

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "upload_type",
    "tab",
    "hour",
]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    sorted_rank = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_rank
    return result


def within_user_zscore(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    _, inverse = np.unique(user_ids, return_inverse=True)
    ng = int(inverse.max()) + 1

    count = np.bincount(inverse, minlength=ng).astype(np.float64)
    total = np.bincount(
        inverse, weights=scores, minlength=ng
    ).astype(np.float64)
    total2 = np.bincount(
        inverse, weights=scores * scores, minlength=ng
    ).astype(np.float64)

    mean = total / np.maximum(count, 1.0)
    variance = total2 / np.maximum(count, 1.0) - mean * mean
    scale = np.sqrt(np.maximum(variance, 1e-8))

    result = (scores - mean[inverse]) / scale[inverse]
    result[count[inverse] <= 1] = 0.0
    return np.clip(result, -5.0, 5.0)


class AdditiveListNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.tables = nn.ModuleList(
            [nn.Embedding(FEATURE_CARDINALITIES[f], 1) for f in FIELDS]
        )
        for table in self.tables:
            nn.init.zeros_(table.weight)

    def forward(self, xs):
        score = self.bias.expand(xs[0].shape[0])
        for table, x in zip(self.tables, xs):
            score = score + table(x).squeeze(1)
        return score


class FMListNet(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.ModuleList(
            [nn.Embedding(FEATURE_CARDINALITIES[f], 1) for f in FIELDS]
        )
        self.latent = nn.ModuleList(
            [nn.Embedding(FEATURE_CARDINALITIES[f], dim) for f in FIELDS]
        )
        for table in self.linear:
            nn.init.zeros_(table.weight)
        for table in self.latent:
            nn.init.normal_(table.weight, std=0.025)

    def forward(self, xs):
        score = self.bias.expand(xs[0].shape[0])
        for table, x in zip(self.linear, xs):
            score = score + table(x).squeeze(1)

        embeddings = torch.stack(
            [table(x) for table, x in zip(self.latent, xs)], dim=1
        )
        summed = embeddings.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - embeddings.square().sum(dim=(1, 2))
        )
        return score + interaction


class UserTowerListNet(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.user = nn.Embedding(
            FEATURE_CARDINALITIES["user_id"], dim
        )
        self.context = nn.ModuleList(
            [
                nn.Embedding(FEATURE_CARDINALITIES[f], dim)
                for f in FIELDS[1:]
            ]
        )
        self.context_linear = nn.ModuleList(
            [
                nn.Embedding(FEATURE_CARDINALITIES[f], 1)
                for f in FIELDS[1:]
            ]
        )
        nn.init.normal_(self.user.weight, std=0.03)
        for table in self.context:
            nn.init.normal_(table.weight, std=0.03)
        for table in self.context_linear:
            nn.init.zeros_(table.weight)

    def forward(self, xs):
        user_vector = self.user(xs[0])
        context_vectors = torch.stack(
            [table(x) for table, x in zip(self.context, xs[1:])],
            dim=1,
        )
        personalized = (
            context_vectors * user_vector.unsqueeze(1)
        ).sum(dim=2).sum(dim=1) / np.sqrt(user_vector.shape[1])

        score = self.bias + personalized
        for table, x in zip(self.context_linear, xs[1:]):
            score = score + table(x).squeeze(1)
        return score


def listnet_loss(scores, group_ids, labels, positive_weights, n_groups):
    maxima = torch.full(
        (n_groups,), -1.0e30, dtype=scores.dtype, device=scores.device
    )
    maxima.scatter_reduce_(
        0, group_ids, scores, reduce="amax", include_self=True
    )

    exponentials = torch.exp(scores - maxima[group_ids])
    denominators = torch.zeros(
        n_groups, dtype=scores.dtype, device=scores.device
    )
    denominators.scatter_add_(0, group_ids, exponentials)
    log_partition = (
        maxima + torch.log(torch.clamp(denominators, min=1e-12))
    )

    target_mass_rows = labels * positive_weights
    target_mass = torch.zeros(
        n_groups, dtype=scores.dtype, device=scores.device
    )
    target_mass.scatter_add_(0, group_ids, target_mass_rows)

    normalized_target = target_mass_rows / torch.clamp(
        target_mass[group_ids], min=1e-12
    )
    row_loss = normalized_target * (
        -scores + log_partition[group_ids]
    )

    active = (target_mass > 0).sum().clamp(min=1)
    return row_loss.sum() / active


def prepare_training(train):
    order = np.argsort(
        np.asarray(train.user_id, dtype=np.int64), kind="stable"
    )
    sorted_users = np.asarray(train.user_id)[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]

    tensors = {
        field: torch.from_numpy(
            np.asarray(train.X[field], dtype=np.int64)
        )
        for field in FIELDS
    }
    labels = torch.from_numpy(
        np.asarray(train.y, dtype=np.float32)
    )

    max_date = int(np.max(train.date))
    age = max_date - np.asarray(train.date, dtype=np.int64)
    recent = np.power(0.5, age.astype(np.float32) / 6.0).astype(
        np.float32
    )
    recent = torch.from_numpy(recent)

    return order, starts, ends, tensors, labels, recent


def train_listnet(
    model,
    prepared,
    learning_rate,
    epochs,
    recency_targets,
    seed,
):
    order, starts, ends, tensors, labels, recent = prepared
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-7
    )

    user_batch = 384
    n_users = len(starts)
    epoch_losses = []

    model.train()
    for epoch in range(epochs):
        user_order = rng.permutation(n_users)
        losses = []

        for b0 in range(0, n_users, user_batch):
            selected_users = np.sort(user_order[b0:b0 + user_batch])

            row_parts = [
                order[starts[u]:ends[u]] for u in selected_users
            ]
            if not row_parts:
                continue
            rows_np = np.concatenate(row_parts)
            lengths = np.asarray(
                [ends[u] - starts[u] for u in selected_users],
                dtype=np.int64,
            )
            group_np = np.repeat(
                np.arange(len(selected_users), dtype=np.int64),
                lengths,
            )

            rows = torch.from_numpy(rows_np)
            group = torch.from_numpy(group_np)
            xs = [tensors[f][rows] for f in FIELDS]
            y = labels[rows]
            if recency_targets:
                positive_weights = recent[rows]
            else:
                positive_weights = torch.ones_like(y)

            optimizer.zero_grad(set_to_none=True)
            scores = model(xs)
            loss = listnet_loss(
                scores,
                group,
                y,
                positive_weights,
                len(selected_users),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        epoch_losses.append(float(np.mean(losses)))

    return epoch_losses


@torch.no_grad()
def predict_model(model, split, batch_size=131072):
    model.eval()
    arrays = {
        field: np.asarray(split.X[field], dtype=np.int64)
        for field in FIELDS
    }
    result = np.empty(len(split.user_id), dtype=np.float64)

    for start in range(0, len(result), batch_size):
        end = min(start + batch_size, len(result))
        xs = [
            torch.from_numpy(arrays[field][start:end])
            for field in FIELDS
        ]
        result[start:end] = model(xs).cpu().numpy().astype(
            np.float64
        )
    return result


train = load("train")
valid = load("valid")
prepared = prepare_training(train)

models = {
    "additive_listnet_uniform": AdditiveListNet(),
    "fm_listnet_recent": FMListNet(dim=12),
    "user_tower_listnet_recent": UserTowerListNet(dim=16),
}

training_logs = {}
training_logs["additive_listnet_uniform"] = train_listnet(
    models["additive_listnet_uniform"],
    prepared,
    learning_rate=0.025,
    epochs=3,
    recency_targets=False,
    seed=SEED + 1,
)
training_logs["fm_listnet_recent"] = train_listnet(
    models["fm_listnet_recent"],
    prepared,
    learning_rate=0.006,
    epochs=3,
    recency_targets=True,
    seed=SEED + 2,
)
training_logs["user_tower_listnet_recent"] = train_listnet(
    models["user_tower_listnet_recent"],
    prepared,
    learning_rate=0.007,
    epochs=3,
    recency_targets=True,
    seed=SEED + 3,
)

valid_logits = {
    name: predict_model(model, valid)
    for name, model in models.items()
}
valid_rank = {
    name: within_user_rank(valid.user_id, score)
    for name, score in valid_logits.items()
}
valid_z = {
    name: within_user_zscore(valid.user_id, score)
    for name, score in valid_logits.items()
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores missing")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")

incumbent_rank = within_user_rank(
    valid.user_id, incumbent_valid
)
incumbent_z = within_user_zscore(
    valid.user_id, incumbent_valid
)

candidate_scores = {}
candidate_primary = {}
candidate_recipes = {}
candidate_raw = {}

for family in models:
    standalone = family + "_standalone"
    candidate_scores[standalone] = valid_rank[family]
    candidate_recipes[standalone] = (
        "standalone", family, 1.0
    )
    candidate_raw[standalone] = valid_rank[family]

    for weight in (0.10, 0.20, 0.30, 0.40, 0.50):
        rank_name = f"{family}_rankblend_w{weight:.2f}"
        rank_blend = (
            weight * valid_rank[family]
            + (1.0 - weight) * incumbent_rank
        )
        candidate_scores[rank_name] = rank_blend
        candidate_recipes[rank_name] = (
            "rankblend", family, weight
        )
        candidate_raw[rank_name] = valid_rank[family]

        z_name = f"{family}_zblend_w{weight:.2f}"
        z_blend = (
            weight * valid_z[family]
            + (1.0 - weight) * incumbent_z
        )
        candidate_scores[z_name] = z_blend
        candidate_recipes[z_name] = (
            "zblend", family, weight
        )
        candidate_raw[z_name] = valid_logits[family]

for name, scores in candidate_scores.items():
    candidate_primary[name] = float(
        evaluate(valid.user_id, valid.y, scores)["primary"]
    )

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

standalone_results = {
    family: candidate_primary[family + "_standalone"]
    for family in models
}
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "standalone": standalone_results,
            "training_losses": training_logs,
            "incumbent_check": float(
                evaluate(
                    valid.user_id, valid.y, incumbent_valid
                )["primary"]
            ),
        },
        separators=(",", ":"),
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        candidate_primary,
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
    if candidate_recipes[winner][0] != "standalone":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
        )

test = load("test")
recipe_type, winning_family, winning_weight = candidate_recipes[
    winner
]
test_logits = predict_model(models[winning_family], test)
test_rank = within_user_rank(test.user_id, test_logits)

if recipe_type == "standalone":
    test_scores = test_rank
else:
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError("Trusted incumbent test scores missing")
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    if len(incumbent_test) != len(test.user_id):
        raise ValueError("Incumbent test length mismatch")

    if recipe_type == "rankblend":
        incumbent_test_component = within_user_rank(
            test.user_id, incumbent_test
        )
        own_test_component = test_rank
    elif recipe_type == "zblend":
        incumbent_test_component = within_user_zscore(
            test.user_id, incumbent_test
        )
        own_test_component = within_user_zscore(
            test.user_id, test_logits
        )
    else:
        raise ValueError("Unknown recipe type")

    test_scores = (
        winning_weight * own_test_component
        + (1.0 - winning_weight) * incumbent_test_component
    )

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