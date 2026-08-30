import os
import time
import json
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
EPS = 1e-9
HALF_LIFE = 9.0
USER_SMOOTH = 8.0
MAIN_SMOOTH = 25.0
PAIR_SMOOTH = 18.0

MAIN_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "hour",
    "user_active_degree",
    "is_video_author",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat11",
    "register_days_bucket",
]

PAIR_SPECS = [
    ("author_id", "tag"),
    ("author_id", "tab"),
    ("video_id", "tab"),
    ("tag", "tab"),
    ("tag", "duration_bucket"),
    ("tag", "upload_type"),
    ("upload_type", "tab"),
    ("onehot_feat3", "tag"),
    ("onehot_feat8", "tag"),
    ("duration_bucket", "tab"),
]

LGB_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "hour",
    "user_active_degree",
    "is_video_author",
    "is_live_streamer",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat16",
]

NUM_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def rank_within_user(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    mask = repeated_counts > 1
    ranked[mask] = positions[mask] / (
        repeated_counts[mask] - 1.0
    )

    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def concatenate_cat(splits, field):
    return np.concatenate([
        np.asarray(s.X[field], dtype=np.int64)
        for s in splits
    ])


def concatenate_labels(splits):
    return np.concatenate([
        np.asarray(s.y, dtype=np.float64)
        for s in splits
    ])


def recency_weights(splits):
    dates = np.concatenate([
        np.asarray(s.date, dtype=np.int64)
        for s in splits
    ])
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = (len(unique_dates) - 1) - day_index
    w = np.exp2(-age.astype(np.float64) / HALF_LIFE)
    w /= np.mean(w)
    return w


def conditional_targets(splits):
    y = concatenate_labels(splits)
    w = recency_weights(splits)
    users = concatenate_cat(splits, "user_id")
    n_users = int(FEATURE_CARDINALITIES["user_id"])

    global_rate = float(np.sum(w * y) / np.sum(w))
    user_weight = np.bincount(
        users, weights=w, minlength=n_users
    ).astype(np.float64)
    user_positive = np.bincount(
        users, weights=w * y, minlength=n_users
    ).astype(np.float64)

    user_rate = (
        user_positive + USER_SMOOTH * global_rate
    ) / (user_weight + USER_SMOOTH)

    # The transformed target is candidate utility after removing the
    # user's slowly varying response propensity.
    target = y - user_rate[users]

    # Equalize prolific users moderately without discarding the positive
    # weighting implicit in the benchmark's GAUC.
    frequency_weight = 1.0 / np.sqrt(
        np.maximum(user_weight[users], 1.0)
    )
    fit_weight = w * frequency_weight
    fit_weight /= np.mean(fit_weight)

    return target, fit_weight, global_rate


def fit_dense_effect(ids, target, weight, cardinality, smoothing):
    count = np.bincount(
        ids, weights=weight, minlength=cardinality
    ).astype(np.float64)
    total = np.bincount(
        ids, weights=weight * target, minlength=cardinality
    ).astype(np.float64)
    effect = total / (count + smoothing)
    reliability = count / (count + smoothing)
    return effect, reliability


def fit_sparse_effect(keys, target, weight, smoothing):
    order = np.argsort(keys, kind="mergesort")
    sk = keys[order]
    st = target[order]
    sw = weight[order]

    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    unique_keys = sk[starts]
    count = np.add.reduceat(sw, starts)
    total = np.add.reduceat(sw * st, starts)
    effect = total / (count + smoothing)
    reliability = count / (count + smoothing)
    return unique_keys, effect, reliability


def sparse_lookup(query, keys, values):
    pos = np.searchsorted(keys, query)
    valid = pos < len(keys)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        valid[valid_indices] = (
            keys[pos[valid_indices]] == query[valid_indices]
        )
    out = np.zeros(len(query), dtype=np.float64)
    out[valid] = values[pos[valid]]
    return out


class ConditionalAdditiveGAM:
    def __init__(self, use_pairs=False):
        self.use_pairs = use_pairs
        self.main_tables = {}
        self.pair_tables = {}

    def fit(self, splits):
        target, weight, _ = conditional_targets(splits)

        for field in MAIN_FIELDS:
            ids = concatenate_cat(splits, field)
            effect, reliability = fit_dense_effect(
                ids,
                target,
                weight,
                int(FEATURE_CARDINALITIES[field]),
                MAIN_SMOOTH,
            )
            self.main_tables[field] = (effect, reliability)

        if self.use_pairs:
            for left, right in PAIR_SPECS:
                left_ids = concatenate_cat(splits, left)
                right_ids = concatenate_cat(splits, right)
                right_card = int(FEATURE_CARDINALITIES[right])
                keys = (
                    left_ids.astype(np.int64) * np.int64(right_card)
                    + right_ids.astype(np.int64)
                )
                table_keys, effect, reliability = fit_sparse_effect(
                    keys, target, weight, PAIR_SMOOTH
                )
                self.pair_tables[(left, right)] = (
                    right_card,
                    table_keys,
                    effect,
                    reliability,
                )
        return self

    def predict(self, split):
        n = len(split.user_id)
        numerator = np.zeros(n, dtype=np.float64)
        denominator = np.zeros(n, dtype=np.float64)

        main_strength = {
            "video_id": 1.30,
            "author_id": 1.20,
            "tag": 1.00,
            "tab": 0.90,
            "duration_bucket": 0.65,
            "upload_type": 0.60,
            "music_type": 0.45,
            "video_type": 0.35,
            "hour": 0.50,
            "user_active_degree": 0.30,
            "is_video_author": 0.30,
            "onehot_feat1": 0.40,
            "onehot_feat2": 0.50,
            "onehot_feat3": 0.65,
            "onehot_feat4": 0.35,
            "onehot_feat7": 0.45,
            "onehot_feat8": 0.60,
            "onehot_feat11": 0.30,
            "register_days_bucket": 0.35,
        }

        for field in MAIN_FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            effect, reliability = self.main_tables[field]
            strength = main_strength[field] * (
                0.15 + 0.85 * reliability[ids]
            )
            numerator += strength * effect[ids]
            denominator += strength

        if self.use_pairs:
            for left, right in PAIR_SPECS:
                left_ids = np.asarray(
                    split.X[left], dtype=np.int64
                )
                right_ids = np.asarray(
                    split.X[right], dtype=np.int64
                )
                right_card, keys, effect, reliability = (
                    self.pair_tables[(left, right)]
                )
                query = (
                    left_ids.astype(np.int64) * np.int64(right_card)
                    + right_ids.astype(np.int64)
                )
                pair_effect = sparse_lookup(query, keys, effect)
                pair_rel = sparse_lookup(query, keys, reliability)
                strength = 0.70 * pair_rel
                numerator += strength * pair_effect
                denominator += strength

        return numerator / np.maximum(denominator, 0.2)


def make_lgb_matrix(splits):
    blocks = []
    for split in splits:
        columns = []
        for field in LGB_FIELDS:
            columns.append(
                np.asarray(split.X[field], dtype=np.float32)
            )
        for field in NUM_FIELDS:
            x = np.asarray(split.num[field], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.log1p(np.maximum(x, 0.0))
            columns.append(x)
        blocks.append(np.column_stack(columns).astype(np.float32))
    return np.concatenate(blocks, axis=0)


class ConditionalBoostedTrees:
    def fit(self, splits):
        target, weight, _ = conditional_targets(splits)
        X = make_lgb_matrix(splits)

        dataset = lgb.Dataset(
            X,
            label=target.astype(np.float32),
            weight=weight.astype(np.float32),
            categorical_feature=list(range(len(LGB_FIELDS))),
            free_raw_data=True,
        )
        params = {
            "objective": "regression_l2",
            "metric": "l2",
            "learning_rate": 0.055,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 180,
            "lambda_l1": 0.15,
            "lambda_l2": 3.0,
            "feature_fraction": 0.82,
            "bagging_fraction": 0.82,
            "bagging_freq": 1,
            "max_bin": 127,
            "num_threads": max(1, min(16, os.cpu_count() or 1)),
            "seed": 20260830,
            "feature_fraction_seed": 20260831,
            "bagging_seed": 20260832,
            "verbose": -1,
        }
        self.model = lgb.train(
            params,
            dataset,
            num_boost_round=210,
        )
        return self

    def predict(self, split):
        X = make_lgb_matrix([split])
        return self.model.predict(X)


def fit_family(name, splits):
    if name == "conditional_additive_gam":
        return ConditionalAdditiveGAM(use_pairs=False).fit(splits)
    if name == "conditional_interaction_gam":
        return ConditionalAdditiveGAM(use_pairs=True).fit(splits)
    if name == "conditional_boosted_trees":
        return ConditionalBoostedTrees().fit(splits)
    raise ValueError(name)


train = load("train")
valid = load("valid")
valid_users = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent predictions unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = rank_within_user(valid_users, inc_valid)

family_names = [
    "conditional_additive_gam",
    "conditional_interaction_gam",
    "conditional_boosted_trees",
]

# Conservative blend grid: the incumbent is already strong, while these
# models are intended as complementary conditional-utility residuals.
alphas = np.asarray(
    [0.0, 0.08, 0.16, 0.25, 0.35, 0.50, 0.70, 1.0],
    dtype=np.float64,
)

candidate_log = {}
best_primary = -np.inf
best_name = None
best_alpha = None
best_scores = None
best_raw = None
best_metrics = None

for name in family_names:
    model = fit_family(name, [train])
    raw = np.asarray(model.predict(valid), dtype=np.float64)
    raw_rank = rank_within_user(valid_users, raw)

    standalone = evaluate(valid_users, valid_y, raw)
    candidate_log[name + "_standalone"] = float(
        standalone["primary"]
    )

    local_best = -np.inf
    local_alpha = 0.0
    for alpha in alphas:
        blended = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * raw_rank
        )
        metrics = evaluate(valid_users, valid_y, blended)
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_alpha = float(alpha)
            best_scores = blended.copy()
            best_raw = raw.copy()
            best_metrics = metrics

    candidate_log[name + "_best_blend"] = float(local_best)
    candidate_log[name + "_alpha"] = float(local_alpha)

print("CANDIDATES " + json.dumps(
    candidate_log, sort_keys=True
))
print(
    "FINDINGS selected_family=%s alpha=%.3f standalone=%.6f"
    % (
        best_name,
        best_alpha,
        candidate_log[best_name + "_standalone"],
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )

# Refit the identical selected family on all labels available before test.
# The selected blend coefficient is fixed from validation.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
final_model = fit_family(best_name, [train, valid])
test_raw = np.asarray(final_model.predict(test), dtype=np.float64)
test_raw_rank = rank_within_user(test_users, test_raw)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
inc_test_rank = rank_within_user(test_users, inc_test)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * test_raw_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))