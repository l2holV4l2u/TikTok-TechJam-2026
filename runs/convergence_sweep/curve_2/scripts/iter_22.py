import os
import time
import json
import gc
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float64)
yv = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def per_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]

    starts = np.where(starts_flag, np.arange(n), 0)
    starts = np.maximum.accumulate(starts)
    within = np.arange(n) - starts

    group_starts = np.flatnonzero(starts_flag)
    group_ends = np.r_[group_starts[1:], n]
    group_sizes = group_ends - group_starts
    repeated_sizes = np.repeat(group_sizes, group_sizes)

    ranked = np.where(
        repeated_sizes > 1,
        within / np.maximum(repeated_sizes - 1, 1),
        0.5,
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


# Training-only recency weights. Dates are all in April, so integer
# subtraction is equal to elapsed days here.
dates = np.asarray(train.date, dtype=np.int32)
last_date = int(dates.max())
age = (last_date - dates).astype(np.float64)
recency_weight = np.exp(-np.log(2.0) * age / 9.0)
recency_weight /= recency_weight.mean()

weighted_positive = recency_weight * y
weighted_negative = recency_weight * (1.0 - y)
global_rate = float(weighted_positive.sum() / recency_weight.sum())
global_logit = float(logit(global_rate))

# A user-centered target removes between-user prevalence, which cannot affect
# ranking inside an evaluation user's impression set.
user_card = int(FEATURE_CARDINALITIES["user_id"])
user_count = np.bincount(
    train_users, minlength=user_card
).astype(np.float64)
user_positive = np.bincount(
    train_users, weights=y, minlength=user_card
).astype(np.float64)
user_mean = (
    user_positive + 8.0 * global_rate
) / np.maximum(user_count + 8.0, 1e-12)
centered_y = y - user_mean[train_users]


# ----------------------------------------------------------------------
# Family 1: generative categorical likelihood-ratio model.
#
# For every categorical value, estimate its probability conditional on the
# positive and negative classes. Adding the resulting log likelihood ratios
# forms a Naive-Bayes discriminant rather than a pointwise discriminative
# model. It can remain useful under drift when class-conditional metadata
# relationships are more stationary than marginal identity frequencies.
# ----------------------------------------------------------------------

nb_fields = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "hour",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
    "fans_user_num_range",
    "register_days_bucket",
]

nb_valid = np.zeros(len(valid.user_id), dtype=np.float64)
nb_test = np.zeros(len(test.user_id), dtype=np.float64)

total_pos = float(weighted_positive.sum())
total_neg = float(weighted_negative.sum())

for field in nb_fields:
    card = int(FEATURE_CARDINALITIES[field])
    tr_ids = np.asarray(train.X[field], dtype=np.int64)

    pos = np.bincount(
        tr_ids, weights=weighted_positive, minlength=card
    ).astype(np.float64)
    neg = np.bincount(
        tr_ids, weights=weighted_negative, minlength=card
    ).astype(np.float64)

    # Jeffreys-like smoothing, strengthened for high-cardinality identities.
    alpha = 1.5 if card < 100 else 3.0
    log_ratio = (
        np.log((pos + alpha) / (total_pos + alpha * card))
        - np.log((neg + alpha) / (total_neg + alpha * card))
    )

    # Limit any single sparse identity from dominating the sum.
    counts = pos + neg
    reliability = counts / (counts + (12.0 if card < 100 else 35.0))
    contribution = np.clip(log_ratio, -3.5, 3.5) * reliability

    nb_valid += contribution[
        np.asarray(valid.X[field], dtype=np.int64)
    ]
    nb_test += contribution[
        np.asarray(test.X[field], dtype=np.int64)
    ]

nb_valid /= np.sqrt(len(nb_fields))
nb_test /= np.sqrt(len(nb_fields))


# ----------------------------------------------------------------------
# Family 2: crossed empirical-Bayes interaction table ensemble.
#
# Exact categorical conjunctions capture effects such as a video's utility
# changing by feed tab or hour. Hashing bounds memory, while posterior
# shrinkage suppresses collision and sparse-cell noise. This is a collection
# of non-parametric interaction tables, not a linear wide model.
# ----------------------------------------------------------------------

cross_pairs = [
    ("video_id", "tab"),
    ("video_id", "hour"),
    ("video_id", "duration_bucket"),
    ("author_id", "tab"),
    ("author_id", "tag"),
    ("author_id", "user_active_degree"),
    ("tag", "duration_bucket"),
    ("tag", "tab"),
    ("upload_type", "tab"),
    ("onehot_feat3", "tab"),
    ("onehot_feat8", "duration_bucket"),
    ("onehot_feat7", "user_active_degree"),
]

HASH_SIZE = 1 << 19
HASH_MASK = HASH_SIZE - 1


def pair_hash(a, b, seed):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return (
        a * np.int64(1000003)
        + b * np.int64(9176)
        + np.int64(seed * 2654435761)
    ) & HASH_MASK


cross_valid = np.zeros(len(valid.user_id), dtype=np.float64)
cross_test = np.zeros(len(test.user_id), dtype=np.float64)

for j, (field_a, field_b) in enumerate(cross_pairs):
    htr = pair_hash(
        train.X[field_a], train.X[field_b], j + 1
    )
    hva = pair_hash(
        valid.X[field_a], valid.X[field_b], j + 1
    )
    hte = pair_hash(
        test.X[field_a], test.X[field_b], j + 1
    )

    count = np.bincount(
        htr, weights=recency_weight, minlength=HASH_SIZE
    ).astype(np.float64)
    positive = np.bincount(
        htr, weights=weighted_positive, minlength=HASH_SIZE
    ).astype(np.float64)

    prior = 28.0
    rate = (
        positive + prior * global_rate
    ) / np.maximum(count + prior, 1e-12)
    effect = np.clip(logit(rate) - global_logit, -2.5, 2.5)
    effect *= count / (count + 24.0)

    cross_valid += effect[hva]
    cross_test += effect[hte]

cross_valid /= np.sqrt(len(cross_pairs))
cross_test /= np.sqrt(len(cross_pairs))


# ----------------------------------------------------------------------
# Family 3: online recency-decayed state-space preference model.
#
# Each training day supplies user-centered innovations to entity states.
# Older states decay before the next day's evidence is incorporated. Unlike
# one pooled target statistic, this estimates the utility state at the train
# boundary and therefore follows changing item/content quality.
# ----------------------------------------------------------------------

state_fields = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]

state_valid = np.zeros(len(valid.user_id), dtype=np.float64)
state_test = np.zeros(len(test.user_id), dtype=np.float64)
unique_dates = np.sort(np.unique(dates))
daily_decay = np.exp(-np.log(2.0) / 4.5)

for field in state_fields:
    card = int(FEATURE_CARDINALITIES[field])
    ids = np.asarray(train.X[field], dtype=np.int64)

    state_sum = np.zeros(card, dtype=np.float64)
    state_mass = np.zeros(card, dtype=np.float64)

    for day in unique_dates:
        mask = dates == day
        day_ids = ids[mask]

        innovation = np.bincount(
            day_ids,
            weights=centered_y[mask],
            minlength=card,
        ).astype(np.float64)
        mass = np.bincount(
            day_ids, minlength=card
        ).astype(np.float64)

        state_sum *= daily_decay
        state_mass *= daily_decay
        state_sum += innovation
        state_mass += mass

    prior_mass = 18.0 if card < 100 else 32.0
    state_effect = state_sum / (state_mass + prior_mass)
    state_effect = np.clip(state_effect, -0.45, 0.45)

    state_valid += state_effect[
        np.asarray(valid.X[field], dtype=np.int64)
    ]
    state_test += state_effect[
        np.asarray(test.X[field], dtype=np.int64)
    ]

state_valid /= np.sqrt(len(state_fields))
state_test /= np.sqrt(len(state_fields))


# ----------------------------------------------------------------------
# Family 4: nonlinear random-feature ridge over leave-one-out evidence.
#
# Each training row receives leakage-controlled, leave-one-out categorical
# evidence. Random tanh projections approximate a nonlinear kernel over those
# signals and numeric attributes; ridge then learns smooth interactions among
# them. This forms predictions differently from both explicit crossed tables
# and neural embedding models.
# ----------------------------------------------------------------------

kernel_fields = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
    "fans_user_num_range",
    "register_days_bucket",
]
numeric_fields = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

base_dim = len(kernel_fields) + len(numeric_fields)
ntr = len(train.user_id)
nva = len(valid.user_id)
nte = len(test.user_id)

base_train = np.empty((ntr, base_dim), dtype=np.float32)
base_valid = np.empty((nva, base_dim), dtype=np.float32)
base_test = np.empty((nte, base_dim), dtype=np.float32)

for j, field in enumerate(kernel_fields):
    card = int(FEATURE_CARDINALITIES[field])
    tr_ids = np.asarray(train.X[field], dtype=np.int64)

    count = np.bincount(
        tr_ids, weights=recency_weight, minlength=card
    ).astype(np.float64)
    positive = np.bincount(
        tr_ids, weights=weighted_positive, minlength=card
    ).astype(np.float64)

    prior = 25.0 if card < 100 else 45.0

    loo_count = np.maximum(count[tr_ids] - recency_weight, 0.0)
    loo_positive = np.maximum(
        positive[tr_ids] - weighted_positive, 0.0
    )
    loo_rate = (
        loo_positive + prior * global_rate
    ) / np.maximum(loo_count + prior, 1e-12)

    full_rate = (
        positive + prior * global_rate
    ) / np.maximum(count + prior, 1e-12)

    base_train[:, j] = np.clip(
        logit(loo_rate) - global_logit, -3.0, 3.0
    ).astype(np.float32)
    base_valid[:, j] = np.clip(
        logit(full_rate[
            np.asarray(valid.X[field], dtype=np.int64)
        ]) - global_logit,
        -3.0,
        3.0,
    ).astype(np.float32)
    base_test[:, j] = np.clip(
        logit(full_rate[
            np.asarray(test.X[field], dtype=np.int64)
        ]) - global_logit,
        -3.0,
        3.0,
    ).astype(np.float32)

for k, field in enumerate(numeric_fields):
    j = len(kernel_fields) + k

    tr_raw = np.asarray(train.num[field], dtype=np.float64)
    va_raw = np.asarray(valid.num[field], dtype=np.float64)
    te_raw = np.asarray(test.num[field], dtype=np.float64)

    tr_trans = np.log1p(np.maximum(np.nan_to_num(tr_raw), 0.0))
    va_trans = np.log1p(np.maximum(np.nan_to_num(va_raw), 0.0))
    te_trans = np.log1p(np.maximum(np.nan_to_num(te_raw), 0.0))

    mean = float(tr_trans.mean())
    std = float(tr_trans.std() + 1e-6)

    base_train[:, j] = np.clip(
        (tr_trans - mean) / std, -5.0, 5.0
    ).astype(np.float32)
    base_valid[:, j] = np.clip(
        (va_trans - mean) / std, -5.0, 5.0
    ).astype(np.float32)
    base_test[:, j] = np.clip(
        (te_trans - mean) / std, -5.0, 5.0
    ).astype(np.float32)

# Normalize target-evidence columns using training-only moments.
base_mean = base_train.mean(axis=0, dtype=np.float64)
base_std = base_train.std(axis=0, dtype=np.float64) + 1e-5
base_train = ((base_train - base_mean) / base_std).astype(np.float32)
base_valid = ((base_valid - base_mean) / base_std).astype(np.float32)
base_test = ((base_test - base_mean) / base_std).astype(np.float32)

rng = np.random.default_rng(20260429)
n_random = 48
projection = rng.normal(
    0.0, 0.55 / np.sqrt(base_dim), size=(base_dim, n_random)
).astype(np.float32)
bias = rng.uniform(-1.0, 1.0, size=n_random).astype(np.float32)

design_dim = 1 + base_dim + n_random
xtx = np.zeros((design_dim, design_dim), dtype=np.float64)
xty = np.zeros(design_dim, dtype=np.float64)

chunk = 50000
for start in range(0, ntr, chunk):
    end = min(start + chunk, ntr)
    b = base_train[start:end]
    nonlinear = np.tanh(b @ projection + bias).astype(np.float32)

    design = np.empty((end - start, design_dim), dtype=np.float32)
    design[:, 0] = 1.0
    design[:, 1:1 + base_dim] = b
    design[:, 1 + base_dim:] = nonlinear

    sw = np.sqrt(recency_weight[start:end]).astype(np.float32)
    weighted_design = design * sw[:, None]
    target = centered_y[start:end] * sw

    xtx += weighted_design.T @ weighted_design
    xty += weighted_design.T @ target

ridge_penalty = np.full(design_dim, 18.0, dtype=np.float64)
ridge_penalty[0] = 1e-3
xtx[np.diag_indices(design_dim)] += ridge_penalty
kernel_coef = np.linalg.solve(xtx, xty)


def kernel_predict(base):
    result = np.empty(len(base), dtype=np.float64)
    for start in range(0, len(base), chunk):
        end = min(start + chunk, len(base))
        b = base[start:end]
        nonlinear = np.tanh(b @ projection + bias).astype(np.float32)

        result[start:end] = (
            kernel_coef[0]
            + b @ kernel_coef[1:1 + base_dim]
            + nonlinear @ kernel_coef[1 + base_dim:]
        )
    return result


kernel_valid = kernel_predict(base_valid)
kernel_test = kernel_predict(base_test)

del base_train, base_valid, base_test, xtx, xty
gc.collect()

print(
    "FINDINGS families=4 nb_fields=%d crossed_tables=%d "
    "state_fields=%d kernel_base=%d kernel_random=%d"
    % (
        len(nb_fields),
        len(cross_pairs),
        len(state_fields),
        base_dim,
        n_random,
    )
)


# ----------------------------------------------------------------------
# Compare standalone families and rank blends with the trusted incumbent.
# The same validation-selected blend coefficient is transferred to test.
# ----------------------------------------------------------------------

families = {
    "generative_likelihood_ratio": (nb_valid, nb_test),
    "crossed_empirical_bayes": (cross_valid, cross_test),
    "online_state_space": (state_valid, state_test),
    "random_feature_kernel_ridge": (kernel_valid, kernel_test),
}

inc_valid_rank = per_user_rank(valid.user_id, inc_valid)
inc_test_rank = per_user_rank(test.user_id, inc_test)

candidate_log = {}
inc_metrics = evaluate(valid.user_id, yv, inc_valid)
candidate_log["trusted_incumbent"] = float(inc_metrics["primary"])

best_primary = float(inc_metrics["primary"])
best_valid = inc_valid.copy()
best_test = inc_test.copy()
best_raw_valid = kernel_valid.copy()
best_name = "trusted_incumbent"

blend_weights = [0.06, 0.10, 0.15, 0.22, 0.30, 0.42, 0.58]

for family_name, (raw_valid, raw_test) in families.items():
    raw_metrics = evaluate(valid.user_id, yv, raw_valid)
    candidate_log[family_name] = float(raw_metrics["primary"])

    raw_valid_rank = per_user_rank(valid.user_id, raw_valid)
    raw_test_rank = per_user_rank(test.user_id, raw_test)

    if float(raw_metrics["primary"]) > best_primary:
        best_primary = float(raw_metrics["primary"])
        best_valid = raw_valid.copy()
        best_test = raw_test.copy()
        best_raw_valid = raw_valid.copy()
        best_name = family_name

    for own_weight in blend_weights:
        blended_valid = (
            (1.0 - own_weight) * inc_valid_rank
            + own_weight * raw_valid_rank
        )
        blended_test = (
            (1.0 - own_weight) * inc_test_rank
            + own_weight * raw_test_rank
        )

        blend_name = "%s_blend_%.2f" % (
            family_name, own_weight
        )
        metrics = evaluate(valid.user_id, yv, blended_valid)
        primary = float(metrics["primary"])
        candidate_log[blend_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_valid = blended_valid.copy()
            best_test = blended_test.copy()
            best_raw_valid = raw_valid.copy()
            best_name = blend_name

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS selected=%s selected_primary=%.6f incumbent_primary=%.6f"
    % (
        best_name,
        best_primary,
        float(inc_metrics["primary"]),
    )
)

final_metrics = evaluate(valid.user_id, yv, best_valid)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)