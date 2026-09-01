import os
import gc
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtri

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
torch.manual_seed(8317)
np.random.seed(8317)

BATCH_SIZE = 32768
HALF_LIFE_DAYS = 4.0

TE_FIELDS = [
    "user_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "music_type",
]

RAW_NUMERIC = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

HISTORY_SUFFIXES = (
    "train_count_log1p",
    "long_view_rate",
    "is_click_rate",
    "play_time_ms_logmean",
    "comment_stay_time_logmean",
)

PRIOR_STRENGTHS = {
    "user_id": 150.0,
    "tag": 1000.0,
    "tab": 1000.0,
    "duration_bucket": 1000.0,
    "upload_type": 700.0,
    "onehot_feat3": 200.0,
    "onehot_feat8": 200.0,
    "user_active_degree": 700.0,
    "music_type": 1000.0,
}


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = (
        np.arange(n, dtype=np.float64)
        - start_positions.astype(np.float64)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]

    sizes = np.diff(
        np.r_[-1, np.flatnonzero(ends)]
    ).astype(np.float64)
    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = (local_rank / denom).astype(np.float32)
    return result


def copula_score(rank):
    p = np.clip(np.asarray(rank, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return ndtri(p).astype(np.float32)


def load_selected_history(split_name):
    columns = []
    names = []

    for key in ("video_id", "author_id"):
        history = historical_features(split_name, key=key)
        for name in sorted(history):
            if any(name.endswith(suffix) for suffix in HISTORY_SUFFIXES):
                x = np.asarray(history[name], dtype=np.float32)
                x = np.nan_to_num(
                    x, nan=0.0, posinf=0.0, neginf=0.0
                )
                columns.append(x)
                names.append(name)

    if not columns:
        raise RuntimeError("No historical features found")

    return np.column_stack(columns).astype(np.float32), names


def load_raw_numeric(split):
    columns = []
    for name in RAW_NUMERIC:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        columns.append(x)

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    hour = np.mod(hour, 24.0)
    angle = (2.0 * np.pi / 24.0) * hour
    columns.append(np.sin(angle).astype(np.float32))
    columns.append(np.cos(angle).astype(np.float32))

    return np.column_stack(columns).astype(np.float32)


def fit_target_encodings(train, labels):
    prior = float(np.mean(labels))
    tables = {}

    for field in TE_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        ids = np.asarray(train.X[field], dtype=np.int64)
        ids = np.where((ids >= 0) & (ids < card), ids, 0)

        counts = np.bincount(ids, minlength=card).astype(np.float32)
        sums = np.bincount(
            ids, weights=labels, minlength=card
        ).astype(np.float32)

        tables[field] = {
            "count": counts,
            "sum": sums,
            "prior": prior,
            "strength": float(PRIOR_STRENGTHS[field]),
        }

    return tables


def target_encoding_matrix(split, tables, labels=None):
    columns = []

    for field in TE_FIELDS:
        table = tables[field]
        counts = table["count"]
        sums = table["sum"]
        prior = table["prior"]
        strength = table["strength"]
        card = len(counts)

        ids = np.asarray(split.X[field], dtype=np.int64)
        ids = np.where((ids >= 0) & (ids < card), ids, 0)

        c = counts[ids]
        s = sums[ids]

        if labels is not None:
            c = np.maximum(c - 1.0, 0.0)
            s = s - labels

        rate = (s + strength * prior) / (c + strength)
        log_count = np.log1p(c)

        columns.append(rate.astype(np.float32))
        columns.append(log_count.astype(np.float32))

    return np.column_stack(columns).astype(np.float32)


def make_features(split, split_name, tables, labels=None,
                  expected_history_names=None):
    history, history_names = load_selected_history(split_name)

    if (
        expected_history_names is not None
        and history_names != expected_history_names
    ):
        raise RuntimeError("Historical feature order mismatch")

    raw = load_raw_numeric(split)
    te = target_encoding_matrix(split, tables, labels=labels)

    matrix = np.column_stack([history, raw, te]).astype(np.float32)
    matrix = np.nan_to_num(
        matrix, nan=0.0, posinf=0.0, neginf=0.0
    )

    del history, raw, te
    gc.collect()
    return matrix, history_names


def fit_scaler(matrix):
    mean = np.mean(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)
    return mean, std


def apply_scaler(matrix, mean, std):
    matrix -= mean
    matrix /= std
    np.clip(matrix, -8.0, 8.0, out=matrix)
    return matrix.astype(np.float32, copy=False)


class RandomFourierKernel(nn.Module):
    """
    Fixed random Fourier mapping followed by logistic regression. This forms
    predictions through a stationary nonlinear kernel rather than learned
    feature crosses or tree partitions.
    """
    def __init__(self, input_dim, n_frequencies=96, bandwidth=1.5):
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(9109)

        omega = torch.randn(
            input_dim, n_frequencies, generator=generator
        ) / bandwidth
        phase = (
            2.0 * np.pi
            * torch.rand(n_frequencies, generator=generator)
        )

        self.register_buffer("omega", omega)
        self.register_buffer("phase", phase)
        self.output = nn.Linear(n_frequencies, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        z = torch.cos(x @ self.omega + self.phase)
        z = z * np.sqrt(2.0 / self.omega.shape[1])
        return self.output(z).squeeze(1)


class InteractionMLP(nn.Module):
    """
    Learned dense interaction model over all continuous, historical, and
    leakage-safe encoding inputs.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x).squeeze(1)


class SoftDecisionForest(nn.Module):
    """
    An ensemble of differentiable binary trees. Each tree partitions feature
    space with learned soft tests and assigns an independent score to every
    leaf, producing a piecewise rule-based prediction.
    """
    def __init__(self, input_dim, n_trees=24, depth=4):
        super().__init__()
        self.n_trees = n_trees
        self.depth = depth
        self.n_internal = (1 << depth) - 1
        self.n_leaves = 1 << depth

        self.node_weight = nn.Parameter(
            torch.empty(n_trees, self.n_internal, input_dim)
        )
        self.node_bias = nn.Parameter(
            torch.zeros(n_trees, self.n_internal)
        )
        self.leaf_value = nn.Parameter(
            torch.empty(n_trees, self.n_leaves)
        )
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(
            self.node_weight, mean=0.0, std=0.12
        )
        nn.init.normal_(
            self.leaf_value, mean=0.0, std=0.02
        )

    def forward(self, x):
        gates = torch.sigmoid(
            torch.einsum("bd,tnd->btn", x, self.node_weight)
            + self.node_bias.unsqueeze(0)
        )

        probabilities = torch.ones(
            x.shape[0], self.n_trees, 1,
            device=x.device, dtype=x.dtype
        )

        node_start = 0
        for level in range(self.depth):
            width = 1 << level
            level_gates = gates[
                :, :, node_start:node_start + width
            ]
            probabilities = torch.cat(
                [
                    probabilities * level_gates,
                    probabilities * (1.0 - level_gates),
                ],
                dim=2,
            ).reshape(x.shape[0], self.n_trees, width * 2)
            node_start += width

        tree_scores = torch.sum(
            probabilities * self.leaf_value.unsqueeze(0),
            dim=2,
        )
        return tree_scores.mean(dim=1) + self.global_bias


def train_model(model, features, labels, sample_weights,
                epochs, learning_rate, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    model.train()
    n = len(labels)

    for epoch in range(epochs):
        permutation = rng.permutation(n)
        loss_numerator = 0.0
        weight_denominator = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]

            x = torch.from_numpy(
                np.asarray(features[idx], dtype=np.float32)
            )
            y = torch.from_numpy(
                np.asarray(labels[idx], dtype=np.float32)
            )
            w = torch.from_numpy(
                np.asarray(sample_weights[idx], dtype=np.float32)
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            losses = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = torch.sum(losses * w) / torch.sum(w).clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_numerator += float(
                torch.sum(losses * w).detach()
            )
            weight_denominator += float(torch.sum(w))

        print(
            "FINDINGS family=%s epoch=%d weighted_logloss=%.6f"
            % (
                model.__class__.__name__,
                epoch + 1,
                loss_numerator / max(weight_denominator, 1.0),
            ),
            flush=True,
        )

    return model


def predict_model(model, features):
    model.eval()
    result = np.empty(len(features), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(features))
            x = torch.from_numpy(
                np.asarray(features[start:end], dtype=np.float32)
            )
            result[start:end] = (
                model(x).cpu().numpy().astype(np.float32)
            )

    return result


inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)

if not (
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent artifacts are required")

train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y)
valid_uid = np.asarray(valid.user_id, dtype=np.int64)

tables = fit_target_encodings(train, train_y)

train_features, history_names = make_features(
    train,
    "train",
    tables,
    labels=train_y,
)
valid_features, valid_history_names = make_features(
    valid,
    "valid",
    tables,
    labels=None,
    expected_history_names=history_names,
)

feature_mean, feature_std = fit_scaler(train_features)
train_features = apply_scaler(
    train_features, feature_mean, feature_std
)
valid_features = apply_scaler(
    valid_features, feature_mean, feature_std
)

train_dates = np.asarray(train.date, dtype=np.int32)
latest_train_date = int(train_dates.max())
age_days = (latest_train_date - train_dates).astype(np.float32)

train_weights = np.exp(
    -np.log(2.0) * age_days / HALF_LIFE_DAYS
).astype(np.float32)
train_weights /= np.mean(train_weights)

input_dim = train_features.shape[1]

models = {
    "rff_kernel": RandomFourierKernel(
        input_dim=input_dim,
        n_frequencies=96,
        bandwidth=1.5,
    ),
    "interaction_mlp": InteractionMLP(input_dim=input_dim),
    "soft_decision_forest": SoftDecisionForest(
        input_dim=input_dim,
        n_trees=24,
        depth=4,
    ),
}

training_config = {
    "rff_kernel": (2, 0.0040, 11003),
    "interaction_mlp": (2, 0.0020, 12007),
    "soft_decision_forest": (2, 0.0030, 13001),
}

inc_valid_raw = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_valid_rank = within_user_rank(valid_uid, inc_valid_raw)
inc_valid_copula = copula_score(inc_valid_rank)
inc_metrics = evaluate(valid_uid, valid_y, inc_valid_rank)

candidate_results = {
    "trusted_incumbent": float(inc_metrics["primary"])
}

valid_raw_by_family = {}
valid_rank_by_family = {}
standalone_primary = {}

for family, model in models.items():
    epochs, lr, seed = training_config[family]
    train_model(
        model,
        train_features,
        train_y,
        train_weights,
        epochs=epochs,
        learning_rate=lr,
        seed=seed,
    )

    raw = predict_model(model, valid_features)
    rank = within_user_rank(valid_uid, raw)
    metrics = evaluate(valid_uid, valid_y, rank)

    valid_raw_by_family[family] = raw
    valid_rank_by_family[family] = rank
    standalone_primary[family] = float(metrics["primary"])
    candidate_results[family + "_standalone"] = float(
        metrics["primary"]
    )

    disagreement = float(
        np.mean(np.abs(rank - inc_valid_rank))
    )
    correlation = float(
        np.corrcoef(rank, inc_valid_rank)[0, 1]
    )

    print(
        "FINDINGS family=%s standalone_primary=%.6f "
        "gauc=%.6f ndcg5=%.6f rank_corr=%.6f "
        "mean_abs_disagreement=%.6f"
        % (
            family,
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
            correlation,
            disagreement,
        ),
        flush=True,
    )

best_scores = inc_valid_rank.copy()
best_primary = float(inc_metrics["primary"])
best_family = None
best_transform = "rank"
best_alpha = 0.0
best_own_rank = None

alphas = [0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.32]

for family, family_rank in valid_rank_by_family.items():
    family_copula = copula_score(family_rank)

    for transform, incumbent_base, family_base in [
        ("rank", inc_valid_rank, family_rank),
        ("copula", inc_valid_copula, family_copula),
    ]:
        for alpha in alphas:
            blended = (
                (1.0 - alpha) * incumbent_base
                + alpha * family_base
            ).astype(np.float32)

            metrics = evaluate(valid_uid, valid_y, blended)
            name = "%s_%s_blend_%.2f" % (
                family, transform, alpha
            )
            candidate_results[name] = float(metrics["primary"])

            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_scores = blended.copy()
                best_family = family
                best_transform = transform
                best_alpha = float(alpha)
                best_own_rank = family_rank.copy()

# Also test a breadth ensemble before blending it with the incumbent.
family_names = list(valid_rank_by_family.keys())
mean_family_rank = np.mean(
    np.column_stack(
        [valid_rank_by_family[name] for name in family_names]
    ),
    axis=1,
).astype(np.float32)
mean_family_metrics = evaluate(
    valid_uid, valid_y, mean_family_rank
)
candidate_results["three_family_mean_standalone"] = float(
    mean_family_metrics["primary"]
)

mean_family_copula = copula_score(mean_family_rank)

for transform, incumbent_base, family_base in [
    ("rank", inc_valid_rank, mean_family_rank),
    ("copula", inc_valid_copula, mean_family_copula),
]:
    for alpha in alphas:
        blended = (
            (1.0 - alpha) * incumbent_base
            + alpha * family_base
        ).astype(np.float32)

        metrics = evaluate(valid_uid, valid_y, blended)
        name = "three_family_mean_%s_blend_%.2f" % (
            transform, alpha
        )
        candidate_results[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_scores = blended.copy()
            best_family = "three_family_mean"
            best_transform = transform
            best_alpha = float(alpha)
            best_own_rank = mean_family_rank.copy()

final_metrics = evaluate(valid_uid, valid_y, best_scores)

best_standalone_family = max(
    standalone_primary,
    key=standalone_primary.get,
)
fallback_raw_rank = valid_rank_by_family[
    best_standalone_family
].copy()

print(
    "FINDINGS feature_dim=%d history_dim=%d "
    "half_life_days=%.1f te_fields=%s"
    % (
        input_dim,
        len(history_names),
        HALF_LIFE_DAYS,
        ",".join(TE_FIELDS),
    ),
    flush=True,
)
print(
    "FINDINGS winner=%s transform=%s alpha=%.2f "
    "incumbent_primary=%.6f final_primary=%.6f"
    % (
        best_family if best_family is not None else "trusted_incumbent",
        best_transform,
        best_alpha,
        float(inc_metrics["primary"]),
        float(final_metrics["primary"]),
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_results, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(OUT, "scores_valid_raw.npy"),
        np.asarray(
            best_own_rank
            if best_own_rank is not None
            else fallback_raw_rank,
            dtype=np.float64,
        ),
    )

del train_features
del valid_features
del train_y
del train_weights
del train_dates
del inc_valid_raw
del valid_raw_by_family
gc.collect()

test = load("test")
test_uid = np.asarray(test.user_id, dtype=np.int64)

inc_test_raw = np.asarray(
    np.load(inc_test_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_test_rank = within_user_rank(test_uid, inc_test_raw)

if best_family is None:
    test_scores = inc_test_rank
else:
    test_features, test_history_names = make_features(
        test,
        "test",
        tables,
        labels=None,
        expected_history_names=history_names,
    )
    test_features = apply_scaler(
        test_features, feature_mean, feature_std
    )

    if best_family == "three_family_mean":
        family_test_ranks = []
        for family in family_names:
            raw = predict_model(models[family], test_features)
            family_test_ranks.append(
                within_user_rank(test_uid, raw)
            )
        selected_test_rank = np.mean(
            np.column_stack(family_test_ranks),
            axis=1,
        ).astype(np.float32)
    else:
        selected_test_raw = predict_model(
            models[best_family], test_features
        )
        selected_test_rank = within_user_rank(
            test_uid, selected_test_raw
        )

    if best_transform == "copula":
        incumbent_base = copula_score(inc_test_rank)
        selected_base = copula_score(selected_test_rank)
    else:
        incumbent_base = inc_test_rank
        selected_base = selected_test_rank

    test_scores = (
        (1.0 - best_alpha) * incumbent_base
        + best_alpha * selected_base
    ).astype(np.float32)

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)

print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)