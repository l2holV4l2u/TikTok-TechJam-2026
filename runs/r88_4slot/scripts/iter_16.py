import os
import time
import json
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
EPS = 1e-8

ENTITY_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "upload_type",
    "duration_bucket",
    "onehot_feat3",
    "onehot_feat8",
]

CAT_NUM_FIELDS = [
    "hour",
    "tab",
    "tag",
    "upload_type",
    "duration_bucket",
    "video_type",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def rank_within_user(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, values, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    good = repeated_counts > 1
    ranked[good] = positions[good] / (repeated_counts[good] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def date_weights(splits, half_life=8.0):
    dates = np.concatenate([np.asarray(s.date, dtype=np.int64) for s in splits])
    unique = np.unique(dates)
    index = np.searchsorted(unique, dates)
    age = (len(unique) - 1 - index).astype(np.float64)
    weights = np.exp2(-age / half_life)
    weights /= max(weights.mean(), EPS)
    return weights


def chronological_features(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    st = times[order]

    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts
    rep_starts = np.repeat(starts, counts)
    rep_counts = np.repeat(counts, counts)

    pos_sorted = np.arange(n, dtype=np.float64) - rep_starts
    pos_pct_sorted = np.zeros(n, dtype=np.float64)
    mask = rep_counts > 1
    pos_pct_sorted[mask] = pos_sorted[mask] / (rep_counts[mask] - 1.0)

    prev_gap_sorted = np.zeros(n, dtype=np.float64)
    next_gap_sorted = np.zeros(n, dtype=np.float64)
    same_prev = np.r_[False, su[1:] == su[:-1]]
    same_next = np.r_[su[:-1] == su[1:], False]
    prev_gap_sorted[same_prev] = (
        st[same_prev] - np.roll(st, 1)[same_prev]
    ) / 1000.0
    next_gap_sorted[same_next] = (
        np.roll(st, -1)[same_next] - st[same_next]
    ) / 1000.0

    pos_pct = np.empty(n, dtype=np.float64)
    user_size = np.empty(n, dtype=np.float64)
    prev_gap = np.empty(n, dtype=np.float64)
    next_gap = np.empty(n, dtype=np.float64)

    pos_pct[order] = pos_pct_sorted
    user_size[order] = rep_counts
    prev_gap[order] = np.log1p(np.maximum(prev_gap_sorted, 0.0))
    next_gap[order] = np.log1p(np.maximum(next_gap_sorted, 0.0))

    # Feed batches are identified by equal (user, timestamp), with row
    # position providing the documented deterministic tie ordering.
    batch_order = np.lexsort((rows, times, users))
    bu = users[batch_order]
    bt = times[batch_order]
    boundary = np.r_[
        True,
        (bu[1:] != bu[:-1]) | (bt[1:] != bt[:-1])
    ]
    bstarts = np.flatnonzero(boundary)
    bends = np.r_[bstarts[1:], n]
    bcounts = bends - bstarts
    brep_starts = np.repeat(bstarts, bcounts)
    brep_counts = np.repeat(bcounts, bcounts)
    bpos_sorted = np.arange(n, dtype=np.float64) - brep_starts
    bpos_pct_sorted = np.zeros(n, dtype=np.float64)
    bmask = brep_counts > 1
    bpos_pct_sorted[bmask] = (
        bpos_sorted[bmask] / (brep_counts[bmask] - 1.0)
    )

    batch_pos = np.empty(n, dtype=np.float64)
    batch_size = np.empty(n, dtype=np.float64)
    batch_pos[batch_order] = bpos_pct_sorted
    batch_size[batch_order] = brep_counts

    return np.column_stack([
        pos_pct,
        np.log1p(user_size),
        prev_gap,
        next_gap,
        batch_pos,
        np.log1p(batch_size),
    ]).astype(np.float32)


def entity_tables(splits, y, weights, smoothing=20.0):
    prior = float(np.sum(weights * y) / max(np.sum(weights), EPS))
    tables = {}
    offset = 0

    for field in ENTITY_FIELDS + ["user_id"]:
        ids = np.concatenate([
            np.asarray(s.X[field], dtype=np.int64) for s in splits
        ])
        card = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(
            ids, weights=weights, minlength=card
        ).astype(np.float64)
        positives = np.bincount(
            ids, weights=weights * y, minlength=card
        ).astype(np.float64)
        rates = (positives + smoothing * prior) / (counts + smoothing)
        reliability = counts / (counts + smoothing)
        tables[field] = (rates, reliability)

    return prior, tables


def raw_base_features(split):
    cols = []

    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float64)
        finite = np.isfinite(x)
        filled = np.where(finite, x, 0.0)
        cols.append(np.log1p(np.maximum(filled, 0.0)))
        cols.append((~finite).astype(np.float64))

    for field in CAT_NUM_FIELDS:
        x = np.asarray(split.X[field], dtype=np.float64)
        card = max(float(FEATURE_CARDINALITIES[field] - 1), 1.0)
        cols.append(x / card)

    hour = np.asarray(split.X["hour"], dtype=np.float64)
    angle = 2.0 * np.pi * hour / 24.0
    cols.append(np.sin(angle))
    cols.append(np.cos(angle))

    date = np.asarray(split.date, dtype=np.int64)
    date_rank = np.searchsorted(np.unique(date), date).astype(np.float64)
    if date_rank.max() > 0:
        date_rank /= date_rank.max()
    cols.append(date_rank)

    return np.column_stack(cols).astype(np.float32)


def assemble_features(fit_splits, target_split, training_part=False):
    y = np.concatenate([
        np.asarray(s.y, dtype=np.float64) for s in fit_splits
    ])
    weights = date_weights(fit_splits)
    prior, tables = entity_tables(fit_splits, y, weights)

    if training_part:
        splits_to_build = fit_splits
    else:
        splits_to_build = [target_split]

    outputs = []
    global_offset = 0

    all_fit_ids = {}
    if training_part:
        for field in ENTITY_FIELDS + ["user_id"]:
            all_fit_ids[field] = np.concatenate([
                np.asarray(s.X[field], dtype=np.int64) for s in fit_splits
            ])

    for split in splits_to_build:
        base = raw_base_features(split)
        context = chronological_features(split)
        n = len(split.user_id)
        extra = []

        if training_part:
            local_y = y[global_offset:global_offset + n]
            local_w = weights[global_offset:global_offset + n]

        entity_rate_for_relative = None

        for field in ENTITY_FIELDS + ["user_id"]:
            ids = np.asarray(split.X[field], dtype=np.int64)
            rates, reliability = tables[field]

            if training_part:
                fit_ids = all_fit_ids[field][global_offset:global_offset + n]
                card = int(FEATURE_CARDINALITIES[field])
                full_counts = np.bincount(
                    all_fit_ids[field],
                    weights=weights,
                    minlength=card,
                ).astype(np.float64)
                full_pos = np.bincount(
                    all_fit_ids[field],
                    weights=weights * y,
                    minlength=card,
                ).astype(np.float64)
                loo_count = np.maximum(
                    full_counts[fit_ids] - local_w, 0.0
                )
                loo_pos = np.maximum(
                    full_pos[fit_ids] - local_w * local_y, 0.0
                )
                local_rate = (
                    loo_pos + 20.0 * prior
                ) / (loo_count + 20.0)
                local_rel = loo_count / (loo_count + 20.0)
            else:
                local_rate = rates[ids]
                local_rel = reliability[ids]

            extra.append(safe_logit(local_rate) - safe_logit(prior))
            extra.append(local_rel)

            if field == "video_id":
                entity_rate_for_relative = local_rate

        duration = np.asarray(split.num["duration_ms"], dtype=np.float64)
        duration = np.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0)
        duration_rank = rank_within_user(split.user_id, duration)
        video_rate_rank = rank_within_user(
            split.user_id, entity_rate_for_relative
        )
        extra.append(duration_rank)
        extra.append(video_rate_rank)
        extra.append(duration_rank - 0.5)
        extra.append(video_rate_rank - 0.5)

        matrix = np.column_stack(
            [base, context, np.column_stack(extra)]
        ).astype(np.float32)
        outputs.append(matrix)
        global_offset += n

    if training_part:
        return np.concatenate(outputs, axis=0), y.astype(np.float32), weights
    return outputs[0]


class RandomForestModel:
    def fit(self, X, y, weights):
        dataset = lgb.Dataset(X, label=y, weight=weights, free_raw_data=True)
        params = {
            "objective": "binary",
            "boosting_type": "rf",
            "num_leaves": 63,
            "max_depth": 10,
            "min_data_in_leaf": 120,
            "feature_fraction": 0.72,
            "bagging_fraction": 0.70,
            "bagging_freq": 1,
            "learning_rate": 0.08,
            "lambda_l2": 4.0,
            "max_bin": 127,
            "num_threads": -1,
            "seed": 8191,
            "feature_fraction_seed": 8192,
            "bagging_seed": 8193,
            "verbose": -1,
        }
        self.model = lgb.train(params, dataset, num_boost_round=160)
        return self

    def predict(self, X):
        return self.model.predict(X).astype(np.float64)


class RFFKernelModel:
    def __init__(self, seed=3203, n_rff=72):
        self.seed = seed
        self.n_rff = n_rff

    def fit(self, X, y, weights):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        sample_n = min(n, 350000)
        sample_idx = rng.choice(n, size=sample_n, replace=False)

        xs = X[sample_idx].astype(np.float64)
        ys = y[sample_idx].astype(np.float64)
        ws = weights[sample_idx].astype(np.float64)

        self.mean = np.average(xs, axis=0, weights=ws)
        var = np.average((xs - self.mean) ** 2, axis=0, weights=ws)
        self.scale = np.sqrt(np.maximum(var, 1e-5))

        z = np.clip((xs - self.mean) / self.scale, -8.0, 8.0)
        self.proj = rng.normal(
            0.0, 0.42, size=(d, self.n_rff)
        ).astype(np.float64)
        self.phase = rng.uniform(
            0.0, 2.0 * np.pi, size=self.n_rff
        ).astype(np.float64)

        phi = np.sqrt(2.0 / self.n_rff) * np.cos(
            z @ self.proj + self.phase
        )
        design = np.column_stack([
            np.ones(sample_n, dtype=np.float64),
            z,
            phi,
        ])

        root_w = np.sqrt(ws / max(ws.mean(), EPS))
        dw = design * root_w[:, None]
        target = (ys - np.average(ys, weights=ws)) * root_w
        self.base = float(np.average(ys, weights=ws))

        gram = dw.T @ dw
        ridge = 18.0
        gram.flat[::gram.shape[0] + 1] += ridge
        rhs = dw.T @ target
        self.coef = np.linalg.solve(gram, rhs)
        return self

    def predict(self, X):
        n = len(X)
        result = np.empty(n, dtype=np.float64)
        chunk = 100000
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            z = np.clip(
                (X[start:stop].astype(np.float64) - self.mean) / self.scale,
                -8.0,
                8.0,
            )
            phi = np.sqrt(2.0 / self.n_rff) * np.cos(
                z @ self.proj + self.phase
            )
            design = np.column_stack([
                np.ones(stop - start, dtype=np.float64),
                z,
                phi,
            ])
            result[start:stop] = self.base + design @ self.coef
        return result


class LSHNeighborhoodModel:
    def __init__(self, seed=7717, n_tables=7, n_bits=13):
        self.seed = seed
        self.n_tables = n_tables
        self.n_bits = n_bits

    def fit(self, X, y, weights):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        sample_n = min(n, 400000)
        idx = rng.choice(n, size=sample_n, replace=False)
        xs = X[idx].astype(np.float64)
        ws = weights[idx].astype(np.float64)

        self.mean = np.average(xs, axis=0, weights=ws)
        var = np.average((xs - self.mean) ** 2, axis=0, weights=ws)
        self.scale = np.sqrt(np.maximum(var, 1e-5))
        self.prior = float(np.sum(weights * y) / np.sum(weights))

        z = np.clip(
            (X.astype(np.float32) - self.mean.astype(np.float32))
            / self.scale.astype(np.float32),
            -6.0,
            6.0,
        )

        self.projections = []
        self.rate_tables = []
        powers = (1 << np.arange(self.n_bits, dtype=np.int64))

        for _ in range(self.n_tables):
            proj = rng.normal(
                0.0, 1.0, size=(d, self.n_bits)
            ).astype(np.float32)
            key = ((z @ proj) > 0).astype(np.int64) @ powers

            size = 1 << self.n_bits
            counts = np.bincount(
                key, weights=weights, minlength=size
            ).astype(np.float64)
            positives = np.bincount(
                key, weights=weights * y, minlength=size
            ).astype(np.float64)
            rates = (positives + 35.0 * self.prior) / (counts + 35.0)
            reliability = counts / (counts + 35.0)

            self.projections.append(proj)
            self.rate_tables.append((rates, reliability))
        return self

    def predict(self, X):
        z = np.clip(
            (X.astype(np.float32) - self.mean.astype(np.float32))
            / self.scale.astype(np.float32),
            -6.0,
            6.0,
        )
        powers = (1 << np.arange(self.n_bits, dtype=np.int64))
        numerator = np.zeros(len(X), dtype=np.float64)
        denominator = np.zeros(len(X), dtype=np.float64)

        for proj, (rates, reliability) in zip(
            self.projections, self.rate_tables
        ):
            key = ((z @ proj) > 0).astype(np.int64) @ powers
            rel = reliability[key]
            numerator += rel * (safe_logit(rates[key]) - safe_logit(self.prior))
            denominator += rel

        return numerator / np.maximum(denominator, 0.25)


def make_model(name):
    if name == "bagged_random_forest":
        return RandomForestModel()
    if name == "rff_kernel_ridge":
        return RFFKernelModel()
    if name == "lsh_neighborhood":
        return LSHNeighborhoodModel()
    raise ValueError(name)


train = load("train")
valid = load("valid")
valid_users = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = rank_within_user(valid_users, inc_valid)

X_train, y_train, w_train = assemble_features(
    [train], train, training_part=True
)
X_valid = assemble_features(
    [train], valid, training_part=False
)

families = [
    "bagged_random_forest",
    "rff_kernel_ridge",
    "lsh_neighborhood",
]
alphas = np.linspace(0.0, 0.75, 13)

candidate_log = {}
raw_valid = {}
best_primary = -np.inf
best_family = None
best_alpha = None
best_scores = None
best_raw = None
best_metrics = None

for family in families:
    model = make_model(family).fit(X_train, y_train, w_train)
    raw = model.predict(X_valid)
    raw_valid[family] = raw

    standalone = evaluate(valid_users, valid_y, raw)
    candidate_log[family + "_standalone"] = float(standalone["primary"])

    raw_rank = rank_within_user(valid_users, raw)
    local_best = -np.inf
    local_alpha = 0.0

    for alpha in alphas:
        blended = (
            (1.0 - float(alpha)) * inc_valid_rank
            + float(alpha) * raw_rank
        )
        metrics = evaluate(valid_users, valid_y, blended)
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_family = family
            best_alpha = float(alpha)
            best_scores = blended.copy()
            best_raw = raw.copy()
            best_metrics = metrics

    candidate_log[family + "_best_blend"] = float(local_best)
    candidate_log[family + "_alpha"] = float(local_alpha)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS winner=%s alpha=%.4f standalone=%.6f"
    % (
        best_family,
        best_alpha,
        candidate_log[best_family + "_standalone"],
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )

# Refit the selected recipe on train + validation and score test.
test = load("test")
X_fit_all, y_fit_all, w_fit_all = assemble_features(
    [train, valid], train, training_part=True
)
X_test = assemble_features(
    [train, valid], test, training_part=False
)

final_model = make_model(best_family).fit(
    X_fit_all, y_fit_all, w_fit_all
)
test_raw = final_model.predict(X_test)
test_raw_rank = rank_within_user(test.user_id, test_raw)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
inc_test_rank = rank_within_user(test.user_id, inc_test)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * test_raw_rank
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
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)