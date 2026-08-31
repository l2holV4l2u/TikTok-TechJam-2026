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
SEED = 8675309
PRED_BATCH = 32768

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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets():
    offsets = []
    total = 0
    for name in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def build_categorical(split):
    return np.ascontiguousarray(
        np.column_stack(
            [
                np.asarray(split.X[name], dtype=np.int64) + OFFSETS[j]
                for j, name in enumerate(FIELDS)
            ]
        ),
        dtype=np.int64,
    )


def group_codes(user_ids):
    _, inverse, counts = np.unique(
        np.asarray(user_ids), return_inverse=True, return_counts=True
    )
    return (
        np.asarray(inverse, dtype=np.int64),
        np.asarray(counts, dtype=np.float32),
    )


def within_group_rank(user_ids, values):
    user_ids = np.asarray(user_ids)
    values = np.asarray(values)
    n = len(values)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, values, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    rank_sorted = (
        np.arange(n, dtype=np.float32) - np.repeat(starts, sizes)
    )
    denom = np.repeat(np.maximum(sizes - 1, 1), sizes).astype(np.float32)
    rank_sorted /= denom

    result = np.empty(n, dtype=np.float32)
    result[order] = rank_sorted
    return result


def build_numeric(split, counts):
    duration = np.asarray(split.num["duration_ms"], dtype=np.float32)
    duration = np.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0)
    log_duration = np.log1p(np.maximum(duration, 0.0)).astype(np.float32)

    time_values = np.asarray(split.time_ms, dtype=np.float64)
    duration_rank = within_group_rank(split.user_id, duration)
    sequence_rank = within_group_rank(split.user_id, time_values)

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    hour = np.maximum(hour - 1.0, 0.0)
    angle = 2.0 * np.pi * hour / 24.0

    group_n = counts[np.asarray(group_codes(split.user_id)[0], dtype=np.int64)]
    log_group_n = np.log1p(group_n).astype(np.float32)

    features = np.column_stack(
        [
            (log_duration - 11.0) / 2.0,
            duration_rank - 0.5,
            sequence_rank - 0.5,
            (sequence_rank - 0.5) ** 2,
            np.sin(angle),
            np.cos(angle),
            log_group_n / 4.0,
        ]
    )
    return np.ascontiguousarray(features, dtype=np.float32)


def prepare(split):
    x = build_categorical(split)
    groups, counts = group_codes(split.user_id)
    z = build_numeric(split, counts)
    return x, groups, counts, z


class AdditiveSlateWide(nn.Module):
    """A generalized additive baseline over IDs and fixed slate-relative features."""

    needs_context = False

    def __init__(self, cardinality, n_numeric):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.numeric = nn.Sequential(
            nn.Linear(n_numeric, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, z, context=None):
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + wide + self.numeric(z).squeeze(-1)


class ContextBilinear(nn.Module):
    """
    A candidate competes against the mean item-side representation of the
    user's logged slate, in addition to user-item latent affinity.
    """

    needs_context = True

    def __init__(self, cardinality, n_numeric, dim=16):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, dim)
        self.linear = nn.Embedding(cardinality, 1)
        self.context_projection = nn.Linear(dim, dim, bias=False)
        self.numeric = nn.Linear(n_numeric, 1)
        self.context_distance = nn.Linear(dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def item_repr(self, x):
        e = self.embedding(x)
        side = torch.cat([e[:, :USER_POS], e[:, USER_POS + 1:]], dim=1)
        return side.mean(dim=1)

    def forward(self, x, z, context):
        item = self.item_repr(x)
        user = self.embedding(x[:, USER_POS])
        affinity = (user * item).sum(dim=-1)
        competition = (
            item * self.context_projection(context)
        ).sum(dim=-1)
        distance = self.context_distance(
            torch.abs(item - context)
        ).squeeze(-1)
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return (
            self.bias
            + wide
            + affinity
            + competition
            + distance
            + self.numeric(z).squeeze(-1)
        )


class SlateDeepSets(nn.Module):
    """
    A nonlinear permutation-invariant slate model. The score depends on the
    candidate, the slate prototype, and candidate-prototype interactions.
    """

    needs_context = True

    def __init__(self, cardinality, n_numeric, dim=16):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, dim)
        self.linear = nn.Embedding(cardinality, 1)
        input_dim = 5 * dim + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def item_repr(self, x):
        e = self.embedding(x)
        side = torch.cat([e[:, :USER_POS], e[:, USER_POS + 1:]], dim=1)
        return side.mean(dim=1)

    def forward(self, x, z, context):
        item = self.item_repr(x)
        user = self.embedding(x[:, USER_POS])
        deep_input = torch.cat(
            [
                user,
                item,
                context,
                item * context,
                item - context,
                z,
            ],
            dim=1,
        )
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + wide + self.mlp(deep_input).squeeze(-1)


@torch.inference_mode()
def calculate_context(model, x, groups, counts):
    model.eval()
    n_groups = len(counts)
    dim = model.embedding.embedding_dim
    sums = torch.zeros((n_groups, dim), dtype=torch.float32)

    for begin in range(0, len(x), PRED_BATCH):
        end = min(begin + PRED_BATCH, len(x))
        xb = torch.from_numpy(x[begin:end])
        gb = torch.from_numpy(groups[begin:end])
        item = model.item_repr(xb)
        sums.index_add_(0, gb, item)

    divisor = torch.from_numpy(counts).clamp_min(1.0).unsqueeze(1)
    return sums / divisor


def train_model(model, x_np, z_np, groups_np, counts_np, y_np,
                epochs, batch_size, lr, seed):
    x = torch.from_numpy(x_np)
    z = torch.from_numpy(z_np)
    groups = torch.from_numpy(groups_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    n = len(x)

    for epoch in range(epochs):
        context_table = None
        if model.needs_context:
            context_table = calculate_context(
                model, x_np, groups_np, counts_np
            ).detach()

        model.train()
        order = torch.randperm(n, generator=generator)
        for begin in range(0, n, batch_size):
            idx = order[begin:begin + batch_size]
            context = (
                context_table[groups[idx]]
                if context_table is not None else None
            )
            logits = model(x[idx], z[idx], context)
            loss = criterion(logits, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict(model, x_np, z_np, groups_np, counts_np):
    model.eval()
    context_table = None
    if model.needs_context:
        context_table = calculate_context(
            model, x_np, groups_np, counts_np
        )

    result = np.empty(len(x_np), dtype=np.float64)
    for begin in range(0, len(x_np), PRED_BATCH):
        end = min(begin + PRED_BATCH, len(x_np))
        xb = torch.from_numpy(x_np[begin:end])
        zb = torch.from_numpy(z_np[begin:end])
        if context_table is None:
            context = None
        else:
            gb = torch.from_numpy(groups_np[begin:end])
            context = context_table[gb]
        result[begin:end] = (
            model(xb, zb, context).cpu().numpy().astype(np.float64)
        )
    return result


def normalized_within_user_rank(user_ids, scores):
    return within_group_rank(user_ids, np.asarray(scores, dtype=np.float64)).astype(
        np.float64
    )


train = load("train")
valid = load("valid")

x_train, g_train, c_train, z_train = prepare(train)
x_valid, g_valid, c_valid, z_valid = prepare(valid)

specifications = [
    (
        "slate_additive_wide",
        AdditiveSlateWide(TOTAL_CARDINALITY, z_train.shape[1]),
        4,
        8192,
        0.0020,
    ),
    (
        "context_bilinear",
        ContextBilinear(
            TOTAL_CARDINALITY, z_train.shape[1], dim=16
        ),
        4,
        8192,
        0.0015,
    ),
    (
        "slate_deepsets",
        SlateDeepSets(
            TOTAL_CARDINALITY, z_train.shape[1], dim=16
        ),
        3,
        4096,
        0.0010,
    ),
]

models = {}
valid_raw = {}

for model_index, (name, model, epochs, batch_size, lr) in enumerate(
    specifications
):
    torch.manual_seed(SEED + 97 * model_index)
    model = train_model(
        model,
        x_train,
        z_train,
        g_train,
        c_train,
        train.y,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=SEED + 1000 + model_index,
    )
    models[name] = model
    valid_raw[name] = predict(
        model, x_valid, z_valid, g_valid, c_valid
    )

del x_train, z_train, g_train, c_train

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores are missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = normalized_within_user_rank(valid.user_id, inc_valid)

candidate_scores = {}
candidate_metrics = {}
candidate_recipe = {}
candidate_own = {}

inc_metric = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["trusted_incumbent"] = inc_valid
candidate_metrics["trusted_incumbent"] = float(inc_metric["primary"])
candidate_recipe["trusted_incumbent"] = ("incumbent", None, 0.0)
candidate_own["trusted_incumbent"] = None

for name, scores in valid_raw.items():
    own_metric = evaluate(valid.user_id, valid.y, scores)
    candidate_scores[name] = scores
    candidate_metrics[name] = float(own_metric["primary"])
    candidate_recipe[name] = ("standalone", name, 1.0)
    candidate_own[name] = scores

    own_rank = normalized_within_user_rank(valid.user_id, scores)
    for weight in (0.15, 0.25, 0.35, 0.50, 0.65):
        blend_name = f"{name}_rankblend_{weight:.2f}"
        blend = weight * own_rank + (1.0 - weight) * inc_valid_rank
        metric = evaluate(valid.user_id, valid.y, blend)
        candidate_scores[blend_name] = blend
        candidate_metrics[blend_name] = float(metric["primary"])
        candidate_recipe[blend_name] = ("rankblend", name, weight)
        candidate_own[blend_name] = scores

model_names = [spec[0] for spec in specifications]
family_rank = np.mean(
    np.column_stack(
        [
            normalized_within_user_rank(valid.user_id, valid_raw[name])
            for name in model_names
        ]
    ),
    axis=1,
)
family_raw = np.mean(
    np.column_stack([valid_raw[name] for name in model_names]),
    axis=1,
)

family_metric = evaluate(valid.user_id, valid.y, family_rank)
candidate_scores["slate_family_rank_ensemble"] = family_rank
candidate_metrics["slate_family_rank_ensemble"] = float(
    family_metric["primary"]
)
candidate_recipe["slate_family_rank_ensemble"] = (
    "family_standalone", "family", 1.0
)
candidate_own["slate_family_rank_ensemble"] = family_raw

for weight in (0.15, 0.25, 0.35, 0.50, 0.65):
    name = f"slate_family_incumbent_rankblend_{weight:.2f}"
    blend = weight * family_rank + (1.0 - weight) * inc_valid_rank
    metric = evaluate(valid.user_id, valid.y, blend)
    candidate_scores[name] = blend
    candidate_metrics[name] = float(metric["primary"])
    candidate_recipe[name] = ("family_rankblend", "family", weight)
    candidate_own[name] = family_raw

winner = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = candidate_scores[winner]
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
            "winner": winner,
            "incumbent_primary": float(inc_metric["primary"]),
            "best_standalone": max(
                (candidate_metrics[n], n) for n in model_names
            )[1],
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
    if candidate_recipe[winner][0] in (
        "rankblend", "family_rankblend"
    ):
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_own[winner], dtype=np.float64),
        )

test = load("test")
x_test, g_test, c_test, z_test = prepare(test)
test_raw = {
    name: predict(model, x_test, z_test, g_test, c_test)
    for name, model in models.items()
}

recipe_type, recipe_model, weight = candidate_recipe[winner]

if recipe_type == "incumbent":
    if not os.path.exists(inc_test_path):
        raise FileNotFoundError("Trusted incumbent test scores are missing")
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)

elif recipe_type == "standalone":
    test_scores = test_raw[recipe_model]

elif recipe_type == "family_standalone":
    test_scores = np.mean(
        np.column_stack(
            [
                normalized_within_user_rank(test.user_id, test_raw[name])
                for name in model_names
            ]
        ),
        axis=1,
    )

else:
    if not os.path.exists(inc_test_path):
        raise FileNotFoundError("Trusted incumbent test scores are missing")
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    inc_test_rank = normalized_within_user_rank(test.user_id, inc_test)

    if recipe_type == "rankblend":
        own_test_rank = normalized_within_user_rank(
            test.user_id, test_raw[recipe_model]
        )
    elif recipe_type == "family_rankblend":
        own_test_rank = np.mean(
            np.column_stack(
                [
                    normalized_within_user_rank(
                        test.user_id, test_raw[name]
                    )
                    for name in model_names
                ]
            ),
            axis=1,
        )
    else:
        raise ValueError("Unknown winning recipe")

    test_scores = weight * own_test_rank + (1.0 - weight) * inc_test_rank

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