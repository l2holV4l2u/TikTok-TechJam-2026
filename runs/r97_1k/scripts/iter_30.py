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

torch.set_num_threads(min(12, os.cpu_count() or 8))
np.random.seed(73129)
torch.manual_seed(73129)

HISTORY_SUFFIXES = (
    "train_count_log1p",
    "long_view_rate",
    "is_click_rate",
    "play_time_ms_logmean",
    "comment_stay_time_logmean",
)

TE_FIELDS = [
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "music_type",
    "author_id",
    "video_id",
]

TE_PRIORS = {
    "tag": 800.0,
    "tab": 800.0,
    "duration_bucket": 800.0,
    "upload_type": 500.0,
    "onehot_feat3": 150.0,
    "onehot_feat8": 150.0,
    "user_active_degree": 500.0,
    "music_type": 800.0,
    "author_id": 120.0,
    "video_id": 80.0,
}

RAW_NUMERIC = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

TRAIN_STEPS = 210
PAIR_BATCH = 65536
PRED_BATCH = 262144


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

    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = np.arange(n, dtype=np.float64) - start_pos

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]

    sizes = np.diff(np.r_[-1, np.flatnonzero(ends)]).astype(np.float64)
    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = (local / denom).astype(np.float32)
    return result


def copula(rank):
    p = np.clip(np.asarray(rank, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return ndtri(p).astype(np.float32)


def load_history(split_name):
    columns = []
    names = []

    for key in ("video_id", "author_id"):
        history = historical_features(split_name, key=key)
        for name in sorted(history):
            if any(name.endswith(s) for s in HISTORY_SUFFIXES):
                x = np.asarray(history[name], dtype=np.float32)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                columns.append(x)
                names.append(name)

    return np.column_stack(columns).astype(np.float32), names


def fit_te_tables(train, y):
    prior = float(np.mean(y))
    tables = {}

    for field in TE_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        ids = np.asarray(train.X[field], dtype=np.int64)
        ids = np.where((ids >= 0) & (ids < card), ids, 0)

        count = np.bincount(ids, minlength=card).astype(np.float32)
        total = np.bincount(
            ids, weights=y, minlength=card
        ).astype(np.float32)

        tables[field] = (
            count,
            total,
            prior,
            float(TE_PRIORS[field]),
        )

    return tables


def make_features(split, split_name, tables, labels=None,
                  expected_history_names=None):
    hist, hist_names = load_history(split_name)
    if expected_history_names is not None and hist_names != expected_history_names:
        raise RuntimeError("Historical feature order mismatch")

    columns = [hist]

    numeric = []
    for name in RAW_NUMERIC:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        numeric.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    hour = np.asarray(split.X["hour"], dtype=np.float32) % 24.0
    angle = 2.0 * np.pi * hour / 24.0
    numeric.append(np.sin(angle).astype(np.float32))
    numeric.append(np.cos(angle).astype(np.float32))
    columns.append(np.column_stack(numeric).astype(np.float32))

    encoded = []
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
        encoded.append(rate.astype(np.float32))
        encoded.append(np.log1p(c).astype(np.float32))

    columns.append(np.column_stack(encoded).astype(np.float32))

    matrix = np.column_stack(columns).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    del hist, columns, numeric, encoded
    gc.collect()
    return matrix, hist_names


def fit_scaler(x):
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)
    return mean, std


def scale_inplace(x, mean, std):
    x -= mean
    x /= std
    np.clip(x, -7.0, 7.0, out=x)
    return x


def build_pair_sampler(user_ids):
    order = np.argsort(
        np.asarray(user_ids, dtype=np.int64),
        kind="stable",
    )
    sorted_users = np.asarray(user_ids, dtype=np.int64)[order]
    n = len(order)
    position = np.arange(n, dtype=np.int64)

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(np.where(starts_flag, position, 0))

    ends_flag = np.empty(n, dtype=bool)
    ends_flag[-1] = True
    ends_flag[:-1] = sorted_users[:-1] != sorted_users[1:]
    ends = np.minimum.accumulate(
        np.where(ends_flag, position, n - 1)[::-1]
    )[::-1]

    return order, starts, ends


class LinearRanker(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


class LowRankQuadraticRanker(nn.Module):
    def __init__(self, dim, rank=16):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.left = nn.Linear(dim, rank, bias=False)
        self.right = nn.Linear(dim, rank, bias=False)
        self.output_scale = nn.Parameter(torch.full((rank,), 0.05))

    def forward(self, x):
        interaction = (
            self.left(x) * self.right(x) * self.output_scale
        ).sum(dim=1)
        return self.linear(x).squeeze(1) + interaction


class MLPRanker(nn.Module):
    def __init__(self, dim, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(dim, 80),
            nn.SiLU(),
            nn.LayerNorm(80),
            nn.Linear(80, 40),
            nn.SiLU(),
            nn.Linear(40, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class PrototypeRanker(nn.Module):
    def __init__(self, positive_init, negative_init):
        super().__init__()
        self.positive = nn.Parameter(positive_init.clone())
        self.negative = nn.Parameter(negative_init.clone())
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.linear = nn.Linear(positive_init.shape[1], 1)

    def _density(self, x, prototypes):
        scale = F.softplus(self.log_scale) + 0.25
        x2 = (x * x).sum(dim=1, keepdim=True)
        p2 = (prototypes * prototypes).sum(dim=1).unsqueeze(0)
        distance = (
            x2 + p2 - 2.0 * (x @ prototypes.t())
        ) / x.shape[1]
        return torch.logsumexp(-distance / scale, dim=1)

    def forward(self, x):
        prototype_score = (
            self._density(x, self.positive)
            - self._density(x, self.negative)
        )
        return prototype_score + 0.20 * self.linear(x).squeeze(1)


def train_pairwise(model, x, y, pair_data, seed, lr):
    rng = np.random.default_rng(seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=2e-5
    )

    order, starts, ends = pair_data
    n = len(order)
    accepted = 0
    running = 0.0

    for step in range(TRAIN_STEPS):
        p = rng.integers(0, n, size=PAIR_BATCH, dtype=np.int64)
        widths = ends[p] - starts[p] + 1
        q = starts[p] + (
            rng.random(PAIR_BATCH) * widths
        ).astype(np.int64)

        ia = order[p]
        ib = order[q]
        mask = y[ia] != y[ib]
        ia = ia[mask]
        ib = ib[mask]

        if len(ia) < 256:
            continue

        xa = torch.from_numpy(x[ia])
        xb = torch.from_numpy(x[ib])
        target = torch.from_numpy(y[ia].astype(np.float32))

        optimizer.zero_grad(set_to_none=True)
        logits = model(xa) - model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        accepted += len(ia)
        running += float(loss.detach())

        if (step + 1) % 70 == 0:
            print(
                "FINDINGS model=%s step=%d pair_loss=%.6f accepted_pairs=%d"
                % (
                    model.__class__.__name__,
                    step + 1,
                    running / 70.0,
                    accepted,
                ),
                flush=True,
            )
            running = 0.0

    model.eval()
    return model


def predict_model(model, x):
    result = np.empty(len(x), dtype=np.float32)
    model.eval()

    with torch.no_grad():
        for start in range(0, len(x), PRED_BATCH):
            end = min(start + PRED_BATCH, len(x))
            xb = torch.from_numpy(x[start:end])
            result[start:end] = model(xb).cpu().numpy().astype(np.float32)

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
train_uid = np.asarray(train.user_id, dtype=np.int64)
valid_uid = np.asarray(valid.user_id, dtype=np.int64)

tables = fit_te_tables(train, train_y)

train_x, history_names = make_features(
    train, "train", tables, labels=train_y
)
valid_x, _ = make_features(
    valid,
    "valid",
    tables,
    labels=None,
    expected_history_names=history_names,
)

feature_mean, feature_std = fit_scaler(train_x)
scale_inplace(train_x, feature_mean, feature_std)
scale_inplace(valid_x, feature_mean, feature_std)

pair_data = build_pair_sampler(train_uid)
dim = train_x.shape[1]

rng = np.random.default_rng(88117)
positive_indices = np.flatnonzero(train_y == 1)
negative_indices = np.flatnonzero(train_y == 0)
pos_init_idx = rng.choice(positive_indices, size=12, replace=False)
neg_init_idx = rng.choice(negative_indices, size=12, replace=False)

models = {
    "pairwise_linear": LinearRanker(dim),
    "pairwise_quadratic": LowRankQuadraticRanker(dim, rank=16),
    "pairwise_prototype": PrototypeRanker(
        torch.from_numpy(train_x[pos_init_idx].copy()),
        torch.from_numpy(train_x[neg_init_idx].copy()),
    ),
    "pairwise_mlp_seed1": MLPRanker(dim, seed=19001),
    "pairwise_mlp_seed2": MLPRanker(dim, seed=59009),
}

learning_rates = {
    "pairwise_linear": 0.008,
    "pairwise_quadratic": 0.0025,
    "pairwise_prototype": 0.003,
    "pairwise_mlp_seed1": 0.0015,
    "pairwise_mlp_seed2": 0.0015,
}

seeds = {
    "pairwise_linear": 11731,
    "pairwise_quadratic": 21737,
    "pairwise_prototype": 31741,
    "pairwise_mlp_seed1": 41761,
    "pairwise_mlp_seed2": 51769,
}

for name in models:
    print(
        "FINDINGS fitting=%s rows=%d dim=%d steps=%d"
        % (name, len(train_x), dim, TRAIN_STEPS),
        flush=True,
    )
    train_pairwise(
        models[name],
        train_x,
        train_y,
        pair_data,
        seed=seeds[name],
        lr=learning_rates[name],
    )

inc_valid_raw = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_valid_rank = within_user_rank(valid_uid, inc_valid_raw)
inc_valid_copula = copula(inc_valid_rank)

candidate_results = {}
inc_metrics = evaluate(valid_uid, valid_y, inc_valid_rank)
candidate_results["trusted_incumbent"] = float(inc_metrics["primary"])

valid_ranks = {}
standalone_primary = {}

for name, model in models.items():
    raw = predict_model(model, valid_x)
    rank = within_user_rank(valid_uid, raw)
    metrics = evaluate(valid_uid, valid_y, rank)

    valid_ranks[name] = rank
    standalone_primary[name] = float(metrics["primary"])
    candidate_results[name + "_standalone"] = float(metrics["primary"])

    print(
        "FINDINGS family=%s primary=%.6f gauc=%.6f ndcg5=%.6f "
        "incumbent_corr=%.6f"
        % (
            name,
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
            float(np.corrcoef(rank, inc_valid_rank)[0, 1]),
        ),
        flush=True,
    )

mlp_ensemble = (
    0.5 * valid_ranks["pairwise_mlp_seed1"]
    + 0.5 * valid_ranks["pairwise_mlp_seed2"]
).astype(np.float32)
valid_ranks["pairwise_mlp_seed_ensemble"] = mlp_ensemble
mlp_metrics = evaluate(valid_uid, valid_y, mlp_ensemble)
standalone_primary["pairwise_mlp_seed_ensemble"] = float(
    mlp_metrics["primary"]
)
candidate_results["pairwise_mlp_seed_ensemble_standalone"] = float(
    mlp_metrics["primary"]
)

structural_consensus = np.mean(
    np.column_stack([
        valid_ranks["pairwise_linear"],
        valid_ranks["pairwise_quadratic"],
        valid_ranks["pairwise_prototype"],
        mlp_ensemble,
    ]),
    axis=1,
).astype(np.float32)
valid_ranks["structural_consensus"] = structural_consensus
consensus_metrics = evaluate(valid_uid, valid_y, structural_consensus)
standalone_primary["structural_consensus"] = float(
    consensus_metrics["primary"]
)
candidate_results["structural_consensus_standalone"] = float(
    consensus_metrics["primary"]
)

best_scores = inc_valid_rank.copy()
best_metrics = inc_metrics
best_family = None
best_alpha = 0.0
best_transform = "rank"
best_gamma = 1.0

alphas = [0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.32]
gammas = [1.0, 2.0, 4.0]

for family, rank in valid_ranks.items():
    family_best = -1.0

    for gamma in gammas:
        shaped = np.power(
            np.clip(rank, 0.0, 1.0), gamma
        ).astype(np.float32)

        options = [
            ("rank", inc_valid_rank, shaped),
            ("copula", inc_valid_copula, copula(shaped)),
        ]

        for transform, incumbent_base, family_base in options:
            for alpha in alphas:
                blended = (
                    (1.0 - alpha) * incumbent_base
                    + alpha * family_base
                ).astype(np.float32)
                metrics = evaluate(valid_uid, valid_y, blended)
                primary = float(metrics["primary"])
                family_best = max(family_best, primary)

                if primary > float(best_metrics["primary"]):
                    best_scores = blended.copy()
                    best_metrics = metrics
                    best_family = family
                    best_alpha = float(alpha)
                    best_transform = transform
                    best_gamma = float(gamma)

    candidate_results[family + "_best_incumbent_blend"] = family_best

best_raw_family = max(standalone_primary, key=standalone_primary.get)
best_raw_scores = valid_ranks[best_raw_family]

print(
    "FINDINGS mlp_seed_rank_corr=%.6f seed1_primary=%.6f "
    "seed2_primary=%.6f ensemble_primary=%.6f"
    % (
        float(np.corrcoef(
            valid_ranks["pairwise_mlp_seed1"],
            valid_ranks["pairwise_mlp_seed2"],
        )[0, 1]),
        standalone_primary["pairwise_mlp_seed1"],
        standalone_primary["pairwise_mlp_seed2"],
        standalone_primary["pairwise_mlp_seed_ensemble"],
    ),
    flush=True,
)
print(
    "FINDINGS winner=%s alpha=%.2f transform=%s gamma=%.1f "
    "incumbent_primary=%.6f final_primary=%.6f"
    % (
        best_family if best_family is not None else "trusted_incumbent",
        best_alpha,
        best_transform,
        best_gamma,
        float(inc_metrics["primary"]),
        float(best_metrics["primary"]),
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
            valid_ranks[best_family]
            if best_family is not None
            else best_raw_scores,
            dtype=np.float64,
        ),
    )

del valid_x, train_x, pair_data
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
    test_x, _ = make_features(
        test,
        "test",
        tables,
        labels=None,
        expected_history_names=history_names,
    )
    scale_inplace(test_x, feature_mean, feature_std)

    if best_family == "pairwise_mlp_seed_ensemble":
        rank1 = within_user_rank(
            test_uid,
            predict_model(models["pairwise_mlp_seed1"], test_x),
        )
        rank2 = within_user_rank(
            test_uid,
            predict_model(models["pairwise_mlp_seed2"], test_x),
        )
        selected_rank = (0.5 * rank1 + 0.5 * rank2).astype(np.float32)

    elif best_family == "structural_consensus":
        component_ranks = []
        for name in (
            "pairwise_linear",
            "pairwise_quadratic",
            "pairwise_prototype",
        ):
            component_ranks.append(
                within_user_rank(
                    test_uid,
                    predict_model(models[name], test_x),
                )
            )

        rank1 = within_user_rank(
            test_uid,
            predict_model(models["pairwise_mlp_seed1"], test_x),
        )
        rank2 = within_user_rank(
            test_uid,
            predict_model(models["pairwise_mlp_seed2"], test_x),
        )
        component_ranks.append(
            (0.5 * rank1 + 0.5 * rank2).astype(np.float32)
        )
        selected_rank = np.mean(
            np.column_stack(component_ranks), axis=1
        ).astype(np.float32)

    else:
        selected_rank = within_user_rank(
            test_uid,
            predict_model(models[best_family], test_x),
        )

    shaped_test = np.power(
        np.clip(selected_rank, 0.0, 1.0),
        best_gamma,
    ).astype(np.float32)

    if best_transform == "copula":
        incumbent_base = copula(inc_test_rank)
        selected_base = copula(shaped_test)
    else:
        incumbent_base = inc_test_rank
        selected_base = shaped_test

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
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)