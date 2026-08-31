import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 94731
THREADS = min(16, os.cpu_count() or 1)
torch.set_num_threads(THREADS)
torch.manual_seed(SEED)
np.random.seed(SEED)

HALF_LIFE = 4.0
SMOOTH = 24.0
BATCH_SIZE = 8192
EPOCHS = 3

TE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "hour",
    "upload_type",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "register_days_range",
    "onehot_feat3",
    "onehot_feat8",
]

RAW_FIELDS = [
    "tab",
    "duration_bucket",
    "tag",
    "hour",
    "upload_type",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "is_video_author",
    "is_live_streamer",
    "video_type",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = ordered_users[:-1] != ordered_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.asarray([-1], dtype=np.int64),
        end_positions,
    )))
    row_sizes = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - start_positions

    ranked = (
        positions.astype(np.float64) + 0.5
    ) / np.maximum(row_sizes, 1).astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    weights = np.power(
        0.5,
        age.astype(np.float32) / HALF_LIFE,
    )
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def add_target_evidence(train, valid, test, labels, weights,
                        train_columns, valid_columns, test_columns):
    weighted_global = float(
        np.sum(weights * labels) / np.maximum(np.sum(weights), 1e-12)
    )

    for field in TE_FIELDS:
        cardinality = int(FEATURE_CARDINALITIES[field])
        tr_key = np.asarray(train.X[field], dtype=np.int64)
        va_key = np.asarray(valid.X[field], dtype=np.int64)
        te_key = np.asarray(test.X[field], dtype=np.int64)

        counts = np.bincount(
            tr_key,
            weights=weights,
            minlength=cardinality,
        ).astype(np.float64)
        positives = np.bincount(
            tr_key,
            weights=weights * labels,
            minlength=cardinality,
        ).astype(np.float64)

        # Exact leave-one-out encodings prevent a training row's outcome
        # from entering its own evidence representation.
        loo_count = np.maximum(counts[tr_key] - weights, 0.0)
        loo_positive = positives[tr_key] - weights * labels
        loo_rate = (
            loo_positive + SMOOTH * weighted_global
        ) / (loo_count + SMOOTH)

        def full_apply(keys):
            entity_count = counts[keys]
            entity_rate = (
                positives[keys] + SMOOTH * weighted_global
            ) / (entity_count + SMOOTH)
            return (
                entity_rate.astype(np.float32),
                np.log1p(entity_count).astype(np.float32),
            )

        valid_rate, valid_count = full_apply(va_key)
        test_rate, test_count = full_apply(te_key)

        train_columns.extend([
            (loo_rate - weighted_global).astype(np.float32),
            np.log1p(loo_count).astype(np.float32),
        ])
        valid_columns.extend([
            (valid_rate - weighted_global).astype(np.float32),
            valid_count,
        ])
        test_columns.extend([
            (test_rate - weighted_global).astype(np.float32),
            test_count,
        ])

    return weighted_global


def add_history(columns, split_name, entity):
    history = historical_features(split_name, key=entity)
    for name in sorted(history):
        values = np.asarray(history[name], dtype=np.float32)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        columns.append(values)


def build_features(train, valid, test):
    labels = np.asarray(train.y, dtype=np.float32)
    weights = recency_weights(train.date)

    train_columns = []
    valid_columns = []
    test_columns = []

    weighted_global = add_target_evidence(
        train,
        valid,
        test,
        labels,
        weights,
        train_columns,
        valid_columns,
        test_columns,
    )

    for entity in ("video_id", "author_id"):
        add_history(train_columns, "train", entity)
        add_history(valid_columns, "valid", entity)
        add_history(test_columns, "test", entity)

    for field in NUM_FIELDS:
        for split, columns in (
            (train, train_columns),
            (valid, valid_columns),
            (test, test_columns),
        ):
            values = np.asarray(split.num[field], dtype=np.float32)
            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=1e7,
                neginf=0.0,
            )
            columns.append(
                np.sign(values) * np.log1p(np.abs(values))
            )

    for field in RAW_FIELDS:
        denominator = max(
            float(FEATURE_CARDINALITIES[field] - 1), 1.0
        )
        train_columns.append(
            np.asarray(train.X[field], dtype=np.float32) / denominator
        )
        valid_columns.append(
            np.asarray(valid.X[field], dtype=np.float32) / denominator
        )
        test_columns.append(
            np.asarray(test.X[field], dtype=np.float32) / denominator
        )

    x_train = np.ascontiguousarray(
        np.column_stack(train_columns), dtype=np.float32
    )
    x_valid = np.ascontiguousarray(
        np.column_stack(valid_columns), dtype=np.float32
    )
    x_test = np.ascontiguousarray(
        np.column_stack(test_columns), dtype=np.float32
    )

    x_train = np.nan_to_num(
        x_train, nan=0.0, posinf=20.0, neginf=-20.0
    )
    x_valid = np.nan_to_num(
        x_valid, nan=0.0, posinf=20.0, neginf=-20.0
    )
    x_test = np.nan_to_num(
        x_test, nan=0.0, posinf=20.0, neginf=-20.0
    )

    mean = np.mean(x_train, axis=0, dtype=np.float64)
    std = np.std(x_train, axis=0, dtype=np.float64)
    std = np.maximum(std, 0.05)

    x_train = np.clip(
        (x_train - mean[None, :]) / std[None, :],
        -8.0,
        8.0,
    ).astype(np.float32)
    x_valid = np.clip(
        (x_valid - mean[None, :]) / std[None, :],
        -8.0,
        8.0,
    ).astype(np.float32)
    x_test = np.clip(
        (x_test - mean[None, :]) / std[None, :],
        -8.0,
        8.0,
    ).astype(np.float32)

    print("FINDINGS " + json.dumps({
        "feature_dimension": int(x_train.shape[1]),
        "weighted_global_rate": weighted_global,
        "unweighted_global_rate": float(labels.mean()),
        "recency_weight_min": float(weights.min()),
        "recency_weight_max": float(weights.max()),
        "half_life_days": HALF_LIFE,
    }, sort_keys=True))

    return x_train, x_valid, x_test, labels, weights


class ObliviousTreeEnsemble(nn.Module):
    def __init__(self, dimension, trees=24, depth=4):
        super().__init__()
        self.dimension = dimension
        self.trees = trees
        self.depth = depth
        self.leaf_count = 2 ** depth

        self.feature_logits = nn.Parameter(
            0.02 * torch.randn(trees, depth, dimension)
        )
        self.thresholds = nn.Parameter(
            0.25 * torch.randn(trees, depth)
        )
        self.log_temperatures = nn.Parameter(
            torch.full((trees, depth), -0.4)
        )
        self.leaf_values = nn.Parameter(
            0.04 * torch.randn(trees, self.leaf_count)
        )
        self.tree_weights = nn.Parameter(
            torch.full((trees,), 1.0 / trees)
        )
        self.bias = nn.Parameter(torch.zeros(()))

        bits = np.zeros((self.leaf_count, depth), dtype=np.float32)
        for leaf in range(self.leaf_count):
            for level in range(depth):
                bits[leaf, level] = float((leaf >> level) & 1)
        self.register_buffer("leaf_bits", torch.from_numpy(bits))

    def forward(self, x):
        selectors = torch.softmax(
            self.feature_logits / 0.55, dim=-1
        )
        selected = torch.einsum("bd,tkd->btk", x, selectors)
        temperature = torch.clamp(
            torch.exp(self.log_temperatures), 0.15, 3.0
        )
        probabilities = torch.sigmoid(
            (selected - self.thresholds[None, :, :])
            / temperature[None, :, :]
        )
        probabilities = torch.clamp(
            probabilities, 1e-5, 1.0 - 1e-5
        )

        log_p = torch.log(probabilities)
        log_q = torch.log1p(-probabilities)
        bits = self.leaf_bits

        log_paths = (
            torch.einsum("btd,ld->btl", log_p, bits)
            + torch.einsum("btd,ld->btl", log_q, 1.0 - bits)
        )
        paths = torch.exp(log_paths)
        tree_outputs = torch.sum(
            paths * self.leaf_values[None, :, :], dim=-1
        )
        return (
            torch.sum(
                tree_outputs * self.tree_weights[None, :], dim=-1
            )
            + self.bias
        )


class AttentiveMaskNetwork(nn.Module):
    def __init__(self, dimension, hidden=64, steps=3):
        super().__init__()
        self.dimension = dimension
        self.steps = steps

        self.initial_context = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mask_layers = nn.ModuleList([
            nn.Linear(hidden, dimension) for _ in range(steps)
        ])
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dimension, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            for _ in range(steps)
        ])
        self.context_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            for _ in range(steps)
        ])
        self.output = nn.Linear(hidden, 1)
        self.direct = nn.Linear(dimension, 1)

    def forward(self, x):
        context = self.initial_context(x)
        accumulated = torch.zeros_like(context)
        prior = torch.ones_like(x)

        for mask_layer, feature_layer, context_layer in zip(
            self.mask_layers,
            self.feature_layers,
            self.context_layers,
        ):
            mask = torch.softmax(
                mask_layer(context) + torch.log(prior + 1e-4),
                dim=-1,
            ) * self.dimension
            masked = x * mask
            representation = feature_layer(masked)
            accumulated = accumulated + F.relu(representation)
            context = context_layer(representation)
            prior = torch.clamp(
                prior * (1.35 - torch.sigmoid(mask)),
                min=0.05,
                max=2.0,
            )

        return (
            self.output(accumulated / self.steps).squeeze(-1)
            + 0.25 * self.direct(x).squeeze(-1)
        )


class PrototypeInterpolator(nn.Module):
    def __init__(self, dimension, prototypes=64):
        super().__init__()
        self.dimension = dimension
        self.prototypes = nn.Parameter(
            0.45 * torch.randn(prototypes, dimension)
        )
        self.prototype_values = nn.Parameter(
            0.05 * torch.randn(prototypes)
        )
        self.log_bandwidth = nn.Parameter(torch.tensor(0.0))
        self.feature_scale = nn.Parameter(torch.zeros(dimension))
        self.linear = nn.Linear(dimension, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        scale = 0.35 + F.softplus(self.feature_scale)
        scaled_x = x * scale[None, :]
        scaled_p = self.prototypes * scale[None, :]

        x_norm = torch.sum(scaled_x * scaled_x, dim=1, keepdim=True)
        p_norm = torch.sum(
            scaled_p * scaled_p, dim=1
        )[None, :]
        distances = torch.clamp(
            x_norm + p_norm - 2.0 * scaled_x @ scaled_p.t(),
            min=0.0,
        )
        bandwidth = 0.5 + F.softplus(self.log_bandwidth)
        attention = torch.softmax(
            -distances / (bandwidth * self.dimension), dim=1
        )
        local_score = attention @ self.prototype_values
        return (
            local_score
            + 0.20 * self.linear(x).squeeze(-1)
            + self.bias
        )


def train_model(name, model, x_train, labels, weights, seed):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.6e-3,
        weight_decay=2.0e-5,
    )

    n = len(x_train)
    generator = np.random.default_rng(seed)
    indices = np.arange(n, dtype=np.int64)

    for epoch in range(EPOCHS):
        generator.shuffle(indices)
        epoch_loss = 0.0
        epoch_weight = 0.0

        for start in range(0, n, BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]
            xb = torch.from_numpy(x_train[batch_indices])
            yb = torch.from_numpy(labels[batch_indices])
            wb = torch.from_numpy(weights[batch_indices])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(row_loss * wb) / torch.clamp(
                torch.sum(wb), min=1e-6
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            batch_weight = float(torch.sum(wb).item())
            epoch_loss += float(loss.item()) * batch_weight
            epoch_weight += batch_weight

        print("FINDINGS " + json.dumps({
            "family": name,
            "epoch": epoch + 1,
            "weighted_train_logloss": epoch_loss / max(epoch_weight, 1e-8),
        }, sort_keys=True))

    return model


def predict_model(model, x):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x))
            xb = torch.from_numpy(x[start:end])
            result[start:end] = model(xb).numpy().astype(np.float32)
    return result


train = load("train")
valid = load("valid")
test = load("test")

x_train, x_valid, x_test, labels, weights = build_features(
    train, valid, test
)
dimension = x_train.shape[1]

models = {
    "neural_oblivious_trees": ObliviousTreeEnsemble(
        dimension, trees=24, depth=4
    ),
    "attentive_feature_masks": AttentiveMaskNetwork(
        dimension, hidden=64, steps=3
    ),
    "prototype_interpolator": PrototypeInterpolator(
        dimension, prototypes=64
    ),
}

family_valid = {}
family_test = {}

for model_index, (name, model) in enumerate(models.items()):
    torch.manual_seed(SEED + 1009 * (model_index + 1))
    model = train_model(
        name,
        model,
        x_train,
        labels,
        weights,
        SEED + 2003 * (model_index + 1),
    )
    family_valid[name] = predict_model(model, x_valid)
    family_test[name] = predict_model(model, x_test)
    del model
    gc.collect()

valid_ranks = {
    name: within_user_rank(valid.user_id, scores)
    for name, scores in family_valid.items()
}
test_ranks = {
    name: within_user_rank(test.user_id, scores)
    for name, scores in family_test.items()
}

family_valid["three_family_ensemble"] = (
    valid_ranks["neural_oblivious_trees"]
    + valid_ranks["attentive_feature_masks"]
    + valid_ranks["prototype_interpolator"]
) / 3.0
family_test["three_family_ensemble"] = (
    test_ranks["neural_oblivious_trees"]
    + test_ranks["attentive_feature_masks"]
    + test_ranks["prototype_interpolator"]
) / 3.0

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_valid = {
    "trusted_incumbent": inc_valid,
}
candidate_test = {
    "trusted_incumbent": inc_test,
}
candidate_source = {
    "trusted_incumbent": None,
}
candidate_metrics = {
    "trusted_incumbent": evaluate(
        valid.user_id, valid.y, inc_valid
    )
}

for name in family_valid:
    raw_valid = np.asarray(family_valid[name], dtype=np.float64)
    raw_test = np.asarray(family_test[name], dtype=np.float64)

    candidate_valid[name] = raw_valid
    candidate_test[name] = raw_test
    candidate_source[name] = name
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, raw_valid
    )

    own_valid_rank = within_user_rank(valid.user_id, raw_valid)
    own_test_rank = within_user_rank(test.user_id, raw_test)

    for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        blend_name = f"{name}_incumbent_{alpha:.2f}"
        blend_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_valid_rank
        )
        blend_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * own_test_rank
        )
        candidate_valid[blend_name] = blend_valid
        candidate_test[blend_name] = blend_test
        candidate_source[blend_name] = name
        candidate_metrics[blend_name] = evaluate(
            valid.user_id, valid.y, blend_valid
        )

best_name = max(
    candidate_metrics,
    key=lambda key: float(candidate_metrics[key]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(candidate_valid[best_name], dtype=np.float64)
best_test = np.asarray(candidate_test[best_name], dtype=np.float64)

own_names = list(family_valid.keys())
best_own_name = max(
    own_names,
    key=lambda key: float(candidate_metrics[key]["primary"]),
)

audit_source = candidate_source[best_name]
if audit_source is None:
    audit_source = best_own_name
raw_valid_for_audit = np.asarray(
    family_valid[audit_source], dtype=np.float64
)

correlations = {}
base_names = [
    "neural_oblivious_trees",
    "attentive_feature_masks",
    "prototype_interpolator",
]
for i in range(len(base_names)):
    for j in range(i + 1, len(base_names)):
        left = base_names[i]
        right = base_names[j]
        correlations[f"{left}__{right}"] = float(
            np.corrcoef(valid_ranks[left], valid_ranks[right])[0, 1]
        )

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "best_standalone_family": best_own_name,
    "family_rank_correlations": correlations,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        best_valid,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        best_test,
    )
    if best_name == "trusted_incumbent" or "_incumbent_" in best_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            raw_valid_for_audit,
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))