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
torch.manual_seed(27183)
np.random.seed(27183)

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

TE_PRIORS = {
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

    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = np.arange(n, dtype=np.float64) - group_start

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]

    sizes = np.diff(np.r_[-1, np.flatnonzero(ends)]).astype(np.float64)
    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denominator = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = (local_rank / denominator).astype(np.float32)
    return result


def copula_score(rank):
    p = np.clip(np.asarray(rank, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return ndtri(p).astype(np.float32)


def load_history(split_name):
    columns = []
    names = []

    for key in ("video_id", "author_id"):
        history = historical_features(split_name, key=key)
        for name in sorted(history):
            if any(name.endswith(suffix) for suffix in HISTORY_SUFFIXES):
                x = np.asarray(history[name], dtype=np.float32)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                columns.append(x)
                names.append(name)

    if not columns:
        raise RuntimeError("No historical features were returned")

    return np.column_stack(columns).astype(np.float32), names


def fit_target_tables(train, labels):
    global_prior = float(np.mean(labels))
    tables = {}

    for field in TE_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        ids = np.asarray(train.X[field], dtype=np.int64)
        ids = np.where((ids >= 0) & (ids < card), ids, 0)

        counts = np.bincount(ids, minlength=card).astype(np.float32)
        sums = np.bincount(
            ids, weights=labels, minlength=card
        ).astype(np.float32)

        tables[field] = (
            counts,
            sums,
            global_prior,
            float(TE_PRIORS[field]),
        )

    return tables


def target_features(split, tables, labels=None):
    columns = []

    for field in TE_FIELDS:
        counts, sums, prior, strength = tables[field]
        ids = np.asarray(split.X[field], dtype=np.int64)
        ids = np.where((ids >= 0) & (ids < len(counts)), ids, 0)

        c = counts[ids]
        s = sums[ids]

        if labels is not None:
            c = np.maximum(c - 1.0, 0.0)
            s = s - labels

        rate = (s + strength * prior) / (c + strength)
        columns.append(rate.astype(np.float32))
        columns.append(np.log1p(c).astype(np.float32))

    return np.column_stack(columns).astype(np.float32)


def raw_numeric_features(split):
    columns = []

    for name in RAW_NUMERIC:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    hour = np.mod(hour, 24.0)
    angle = hour * (2.0 * np.pi / 24.0)
    columns.append(np.sin(angle).astype(np.float32))
    columns.append(np.cos(angle).astype(np.float32))

    return np.column_stack(columns).astype(np.float32)


def make_features(split, split_name, tables, labels=None,
                  expected_history_names=None):
    history, history_names = load_history(split_name)

    if (
        expected_history_names is not None
        and history_names != expected_history_names
    ):
        raise RuntimeError("Historical feature ordering changed across splits")

    raw = raw_numeric_features(split)
    target = target_features(split, tables, labels=labels)

    matrix = np.column_stack([history, raw, target]).astype(np.float32)
    matrix = np.nan_to_num(
        matrix, nan=0.0, posinf=0.0, neginf=0.0
    )

    del history, raw, target
    gc.collect()
    return matrix, history_names


def fit_scaler(x):
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)
    return mean, std


def apply_scaler(x, mean, std):
    x -= mean
    x /= std
    np.clip(x, -8.0, 8.0, out=x)
    return x.astype(np.float32, copy=False)


class NeuralAdditiveSpline(nn.Module):
    """
    A neural additive model. Each input is expanded into fixed Gaussian
    spline-like basis functions and receives its own learned shape function.
    Prediction is the sum of those univariate functions.
    """
    def __init__(self, input_dim, n_knots=13):
        super().__init__()
        knots = torch.linspace(-4.5, 4.5, n_knots)
        self.register_buffer("knots", knots)
        self.log_width = nn.Parameter(torch.tensor(np.log(0.85)))
        self.feature_weight = nn.Parameter(
            torch.zeros(input_dim, n_knots)
        )
        self.linear_weight = nn.Parameter(torch.zeros(input_dim))
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.feature_weight, mean=0.0, std=0.01)

    def forward(self, x):
        width = torch.exp(self.log_width).clamp(0.35, 2.0)
        basis = torch.exp(
            -0.5 * ((x.unsqueeze(2) - self.knots) / width) ** 2
        )
        basis = basis / basis.sum(dim=2, keepdim=True).clamp_min(1e-6)
        shaped = torch.sum(
            basis * self.feature_weight.unsqueeze(0), dim=2
        )
        return (
            shaped.sum(dim=1)
            + torch.sum(x * self.linear_weight, dim=1)
            + self.bias
        )


class AttentiveTabularNetwork(nn.Module):
    """
    Each decision step forms a row-dependent sparse feature mask, transforms
    the selected inputs, and adds a residual decision representation.
    """
    def __init__(self, input_dim, hidden_dim=64, n_steps=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps

        self.initial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.mask_layers = nn.ModuleList([
            nn.Linear(hidden_dim, input_dim) for _ in range(n_steps)
        ])
        self.transform_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            for _ in range(n_steps)
        ])
        self.output = nn.Linear(hidden_dim, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        state = self.initial(x)
        aggregate = torch.zeros_like(state)

        for mask_layer, transform in zip(
            self.mask_layers, self.transform_layers
        ):
            mask = torch.softmax(mask_layer(state), dim=1)
            mask = mask * self.input_dim
            decision = transform(x * mask)
            aggregate = aggregate + decision
            state = state + 0.5 * decision

        return self.output(aggregate / self.n_steps).squeeze(1)


class MaxoutExpertNetwork(nn.Module):
    """
    A bank of affine experts defines piecewise-linear regional responses.
    Maxout routing chooses an expert independently for every hidden unit.
    """
    def __init__(self, input_dim, hidden_dim=64, pieces=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pieces = pieces
        self.experts = nn.Linear(input_dim, hidden_dim * pieces)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        expert_scores = self.experts(x).reshape(
            x.shape[0], self.hidden_dim, self.pieces
        )
        routed = torch.max(expert_scores, dim=2).values
        return self.output(routed).squeeze(1)


def train_model(model, features, labels, sample_weights,
                epochs, learning_rate, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=2e-5,
    )

    n = len(labels)
    model.train()

    for epoch in range(epochs):
        permutation = rng.permutation(n)
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]

            xb = torch.from_numpy(
                np.asarray(features[idx], dtype=np.float32)
            )
            yb = torch.from_numpy(
                np.asarray(labels[idx], dtype=np.float32)
            )
            wb = torch.from_numpy(
                np.asarray(sample_weights[idx], dtype=np.float32)
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb).clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(torch.sum(losses * wb).detach())
            total_weight += float(torch.sum(wb))

        print(
            "FINDINGS family=%s epoch=%d weighted_logloss=%.6f"
            % (
                model.__class__.__name__,
                epoch + 1,
                total_loss / max(total_weight, 1.0),
            ),
            flush=True,
        )

    return model


def predict_model(model, features):
    result = np.empty(len(features), dtype=np.float32)
    model.eval()

    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(features))
            xb = torch.from_numpy(
                np.asarray(features[start:end], dtype=np.float32)
            )
            result[start:end] = (
                model(xb).cpu().numpy().astype(np.float32)
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
    raise RuntimeError("Trusted incumbent predictions are required")

train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y)
valid_uid = np.asarray(valid.user_id, dtype=np.int64)

tables = fit_target_tables(train, train_y)

train_features, history_names = make_features(
    train, "train", tables, labels=train_y
)
valid_features, _ = make_features(
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
latest_date = int(train_dates.max())
age_days = (latest_date - train_dates).astype(np.float32)
train_weights = np.exp(
    -np.log(2.0) * age_days / HALF_LIFE_DAYS
).astype(np.float32)
train_weights /= np.mean(train_weights)

input_dim = train_features.shape[1]

models = {
    "neural_additive_spline": NeuralAdditiveSpline(
        input_dim=input_dim,
        n_knots=13,
    ),
    "attentive_tabular": AttentiveTabularNetwork(
        input_dim=input_dim,
        hidden_dim=64,
        n_steps=3,
    ),
    "maxout_experts": MaxoutExpertNetwork(
        input_dim=input_dim,
        hidden_dim=64,
        pieces=4,
    ),
}

configs = {
    "neural_additive_spline": (3, 0.0040, 31013),
    "attentive_tabular": (2, 0.0015, 32003),
    "maxout_experts": (2, 0.0015, 33013),
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

valid_ranks = {}
standalone_scores = {}

for name, model in models.items():
    epochs, learning_rate, seed = configs[name]
    train_model(
        model,
        train_features,
        train_y,
        train_weights,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )

    raw = predict_model(model, valid_features)
    rank = within_user_rank(valid_uid, raw)
    metrics = evaluate(valid_uid, valid_y, rank)

    valid_ranks[name] = rank
    standalone_scores[name] = float(metrics["primary"])
    candidate_results[name + "_standalone"] = float(metrics["primary"])

    correlation = float(np.corrcoef(rank, inc_valid_rank)[0, 1])
    disagreement = float(np.mean(np.abs(rank - inc_valid_rank)))

    print(
        "FINDINGS family=%s primary=%.6f gauc=%.6f ndcg5=%.6f "
        "incumbent_rank_corr=%.6f mean_abs_disagreement=%.6f"
        % (
            name,
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

alphas = [0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.30]

for name, rank in valid_ranks.items():
    family_copula = copula_score(rank)

    for transform, incumbent_base, family_base in [
        ("rank", inc_valid_rank, rank),
        ("copula", inc_valid_copula, family_copula),
    ]:
        for alpha in alphas:
            blended = (
                (1.0 - alpha) * incumbent_base
                + alpha * family_base
            ).astype(np.float32)
            metrics = evaluate(valid_uid, valid_y, blended)

            candidate_name = "%s_%s_blend_%.2f" % (
                name, transform, alpha
            )
            candidate_results[candidate_name] = float(
                metrics["primary"]
            )

            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_scores = blended.copy()
                best_family = name
                best_transform = transform
                best_alpha = float(alpha)
                best_own_rank = rank.copy()

family_names = list(valid_ranks)
mean_family_rank = np.mean(
    np.column_stack([valid_ranks[name] for name in family_names]),
    axis=1,
).astype(np.float32)
mean_metrics = evaluate(valid_uid, valid_y, mean_family_rank)
candidate_results["three_family_mean_standalone"] = float(
    mean_metrics["primary"]
)

for transform, incumbent_base, family_base in [
    ("rank", inc_valid_rank, mean_family_rank),
    ("copula", inc_valid_copula, copula_score(mean_family_rank)),
]:
    for alpha in alphas:
        blended = (
            (1.0 - alpha) * incumbent_base
            + alpha * family_base
        ).astype(np.float32)
        metrics = evaluate(valid_uid, valid_y, blended)
        candidate_name = "three_family_mean_%s_blend_%.2f" % (
            transform, alpha
        )
        candidate_results[candidate_name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_scores = blended.copy()
            best_family = "three_family_mean"
            best_transform = transform
            best_alpha = float(alpha)
            best_own_rank = mean_family_rank.copy()

final_metrics = evaluate(valid_uid, valid_y, best_scores)

best_standalone_name = max(
    standalone_scores, key=standalone_scores.get
)
fallback_own_rank = valid_ranks[best_standalone_name]

print(
    "FINDINGS feature_dim=%d history_dim=%d half_life_days=%.1f"
    % (input_dim, len(history_names), HALF_LIFE_DAYS),
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
            else fallback_own_rank,
            dtype=np.float64,
        ),
    )

del train_features
del valid_features
del train_y
del train_weights
del train_dates
del inc_valid_raw
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
    test_features, _ = make_features(
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
        test_family_ranks = []
        for name in family_names:
            raw = predict_model(models[name], test_features)
            test_family_ranks.append(
                within_user_rank(test_uid, raw)
            )
        selected_test_rank = np.mean(
            np.column_stack(test_family_ranks),
            axis=1,
        ).astype(np.float32)
    else:
        selected_raw = predict_model(
            models[best_family], test_features
        )
        selected_test_rank = within_user_rank(
            test_uid, selected_raw
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