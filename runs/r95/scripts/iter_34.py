import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
np.random.seed(SEED)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float64)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if len(inc_valid) != len(valid.user_id) or len(inc_test) != len(test.user_id):
    raise RuntimeError("Trusted incumbent prediction lengths do not match splits")

# ----------------------------------------------------------------------
# Train-only recency-weighted empirical Bayes relevance estimates.
# This supplies a structurally non-parametric quality family and also
# estimates temporal instability used by the DPP diversity kernel.
# ----------------------------------------------------------------------
train_dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(train_dates)
day_index = np.searchsorted(unique_dates, train_dates)
age = (len(unique_dates) - 1 - day_index).astype(np.float64)
recency_weight = np.exp2(-age / 4.0)
global_rate = float(np.average(ytr, weights=recency_weight))
global_logit = np.log(
    np.clip(global_rate, 1e-6, 1 - 1e-6)
    / np.clip(1.0 - global_rate, 1e-6, 1 - 1e-6)
)

EB_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "tab",
]

eb_tables = {}
instability_tables = {}


def logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


for field in EB_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])

    weighted_count = np.bincount(
        ids, weights=recency_weight, minlength=card
    ).astype(np.float64)
    weighted_positive = np.bincount(
        ids, weights=recency_weight * ytr, minlength=card
    ).astype(np.float64)

    smooth = 30.0 if field in ("video_id", "author_id") else 80.0
    rate = (
        weighted_positive + smooth * global_rate
    ) / (weighted_count + smooth)
    eb_tables[field] = logit(rate) - global_logit

    if field in ("video_id", "author_id"):
        recent_mask = age <= 3.0
        older_mask = ~recent_mask

        recent_count = np.bincount(
            ids[recent_mask],
            minlength=card
        ).astype(np.float64)
        recent_pos = np.bincount(
            ids[recent_mask],
            weights=ytr[recent_mask],
            minlength=card
        ).astype(np.float64)

        older_count = np.bincount(
            ids[older_mask],
            minlength=card
        ).astype(np.float64)
        older_pos = np.bincount(
            ids[older_mask],
            weights=ytr[older_mask],
            minlength=card
        ).astype(np.float64)

        temporal_smooth = 25.0
        recent_rate = (
            recent_pos + temporal_smooth * global_rate
        ) / (recent_count + temporal_smooth)
        older_rate = (
            older_pos + temporal_smooth * global_rate
        ) / (older_count + temporal_smooth)

        support = np.sqrt(
            np.minimum(recent_count, older_count)
            / (np.minimum(recent_count, older_count) + 20.0)
        )
        instability = np.abs(recent_rate - older_rate) * support
        instability_tables[field] = instability.astype(np.float64)


def empirical_bayes_scores(sample):
    components = []
    for field in EB_FIELDS:
        ids = np.asarray(sample.X[field], dtype=np.int64)
        components.append(eb_tables[field][ids])

    # Identity fields receive greater weight; side fields stabilize sparse
    # or temporally missing identities.
    result = (
        0.34 * components[0]
        + 0.25 * components[1]
        + 0.13 * components[2]
        + 0.10 * components[3]
        + 0.08 * components[4]
        + 0.10 * components[5]
    )
    return result.astype(np.float64)


def row_instability(sample):
    video = instability_tables["video_id"][
        np.asarray(sample.X["video_id"], dtype=np.int64)
    ]
    author = instability_tables["author_id"][
        np.asarray(sample.X["author_id"], dtype=np.int64)
    ]
    raw = 0.55 * video + 0.45 * author
    return raw.astype(np.float64)


eb_valid = empirical_bayes_scores(valid)
eb_test = empirical_bayes_scores(test)
unc_valid_raw = row_instability(valid)
unc_test_raw = row_instability(test)

# Instability scaling is chosen only from the train-derived entity table.
train_unc_values = np.concatenate([
    instability_tables["video_id"],
    instability_tables["author_id"],
])
unc_lo = float(np.quantile(train_unc_values, 0.50))
unc_hi = float(np.quantile(train_unc_values, 0.95))
unc_scale = max(unc_hi - unc_lo, 1e-8)


def normalize_uncertainty(raw):
    return np.clip((raw - unc_lo) / unc_scale, 0.0, 1.0)


unc_valid = normalize_uncertainty(unc_valid_raw)
unc_test = normalize_uncertainty(unc_test_raw)

print(
    "FINDINGS temporal_instability_train_q50=%.6f q95=%.6f "
    "valid_mean=%.6f test_mean=%.6f"
    % (
        unc_lo,
        unc_hi,
        float(np.mean(unc_valid)),
        float(np.mean(unc_test)),
    )
)

# ----------------------------------------------------------------------
# Ranking utilities. All blending occurs in within-user rank space, making
# the blend insensitive to calibration differences between model families.
# ----------------------------------------------------------------------
def rank_scores_by_user(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, -scores, user_ids))
    ordered_users = user_ids[order]

    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]

    result = np.empty(n, dtype=np.float64)
    for start, end in zip(starts, ends):
        group_order = order[start:end]
        size = end - start
        # Tiny quality term preserves deterministic ordering if blended ranks
        # happen to become equal.
        result[group_order] = (
            (size - np.arange(size, dtype=np.float64)) / max(size, 1)
            + 1e-10 * scores[group_order]
        )
    return result


inc_valid_rank = rank_scores_by_user(valid.user_id, inc_valid)
inc_test_rank = rank_scores_by_user(test.user_id, inc_test)
eb_valid_rank = rank_scores_by_user(valid.user_id, eb_valid)
eb_test_rank = rank_scores_by_user(test.user_id, eb_test)

# ----------------------------------------------------------------------
# Greedy MAP inference for a quality-weighted DPP.
#
# S_ij is an average of categorical equality kernels and is therefore PSD.
# With row-specific rho, S becomes
#   diag(sqrt(rho)) K diag(sqrt(rho)) + diag(1-rho),
# which remains PSD. Temporally unstable entities receive larger rho and
# therefore a stronger redundancy penalty.
# ----------------------------------------------------------------------
KERNEL_FIELDS = {
    "content": [
        "author_id",
        "tag",
        "duration_bucket",
        "upload_type",
        "video_type",
    ],
    "context": [
        "author_id",
        "tag",
        "tab",
        "hour",
        "music_type",
        "onehot_feat3",
        "onehot_feat8",
    ],
}


def dpp_rank_scores(sample, base_scores, uncertainty, kernel_name, beta,
                    uncertainty_conditioned):
    users = np.asarray(sample.user_id, dtype=np.int64)
    base_scores = np.asarray(base_scores, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    # Groups are ordered by descending quality before DPP inference.
    sorted_rows = np.lexsort((rows, -base_scores, users))
    sorted_users = users[sorted_rows]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]

    field_arrays = {
        field: np.asarray(sample.X[field], dtype=np.int64)
        for field in KERNEL_FIELDS[kernel_name]
    }

    output = np.empty(n, dtype=np.float64)
    pool_limit = 15

    for start, end in zip(starts, ends):
        quality_order = sorted_rows[start:end]
        group_size = end - start

        if group_size <= 1:
            output[quality_order] = 1.0 + 1e-10 * base_scores[quality_order]
            continue

        pool_size = min(pool_limit, group_size)
        pool = quality_order[:pool_size]

        # Rank-based quality avoids dependence on incumbent calibration.
        quality_rank = (
            pool_size - np.arange(pool_size, dtype=np.float64)
        ) / pool_size
        q = np.exp(float(beta) * quality_rank)

        similarity = np.zeros((pool_size, pool_size), dtype=np.float64)
        for field in KERNEL_FIELDS[kernel_name]:
            values = field_arrays[field][pool]
            similarity += (values[:, None] == values[None, :]).astype(
                np.float64
            )
        similarity /= float(len(KERNEL_FIELDS[kernel_name]))

        if uncertainty_conditioned:
            rho = 0.25 + 0.65 * uncertainty[pool]
        else:
            rho = np.full(pool_size, 0.62, dtype=np.float64)

        sqrt_rho = np.sqrt(np.clip(rho, 0.0, 0.98))
        similarity = (
            sqrt_rho[:, None] * similarity * sqrt_rho[None, :]
        )
        similarity[np.diag_indices(pool_size)] += 1.0 - rho

        kernel = q[:, None] * similarity * q[None, :]

        # Fast greedy Cholesky DPP MAP selection.
        residual = np.diag(kernel).copy()
        coefficients = np.zeros(
            (pool_size, pool_size), dtype=np.float64
        )
        selected = np.zeros(pool_size, dtype=bool)
        chosen_positions = []

        for step in range(pool_size):
            available_gain = np.where(selected, -np.inf, residual)
            chosen = int(np.argmax(available_gain))
            chosen_positions.append(chosen)
            selected[chosen] = True

            pivot = np.sqrt(max(residual[chosen], 1e-12))
            remaining = np.flatnonzero(~selected)
            if len(remaining) == 0:
                break

            if step == 0:
                projection = kernel[chosen, remaining]
            else:
                projection = (
                    kernel[chosen, remaining]
                    - coefficients[:step, chosen].dot(
                        coefficients[:step, remaining]
                    )
                )

            new_coefficient = projection / pivot
            coefficients[step, remaining] = new_coefficient
            residual[remaining] = np.maximum(
                residual[remaining] - new_coefficient ** 2,
                1e-12
            )
            residual[chosen] = -np.inf

        dpp_pool_order = pool[np.asarray(chosen_positions, dtype=np.int64)]

        # Only the candidate pool is reranked. Lower-quality impressions keep
        # their incumbent ordering, protecting GAUC outside the top slate.
        final_order = np.concatenate([
            dpp_pool_order,
            quality_order[pool_size:],
        ])

        output[final_order] = (
            (group_size - np.arange(group_size, dtype=np.float64))
            / group_size
            + 1e-10 * base_scores[final_order]
        )

    return output


# Construct fixed, mechanism-distinct slate predictors on validation.
dpp_valid = {}
dpp_valid["dpp_content_fixed"] = dpp_rank_scores(
    valid, inc_valid, unc_valid, "content", beta=3.2,
    uncertainty_conditioned=False
)
dpp_valid["dpp_content_uncertainty"] = dpp_rank_scores(
    valid, inc_valid, unc_valid, "content", beta=3.2,
    uncertainty_conditioned=True
)
dpp_valid["dpp_context_uncertainty"] = dpp_rank_scores(
    valid, inc_valid, unc_valid, "context", beta=3.6,
    uncertainty_conditioned=True
)

candidate_scores = {}
candidate_recipes = {}

candidate_scores["incumbent"] = inc_valid
candidate_recipes["incumbent"] = ("incumbent", None, None)

candidate_scores["empirical_bayes"] = eb_valid_rank
candidate_recipes["empirical_bayes"] = ("eb", None, None)

# Also compare each DPP family alone and conservatively blended with the
# incumbent ranking. Fixed blend weights are shared across all families.
for family, transformed in dpp_valid.items():
    candidate_scores[family] = transformed
    candidate_recipes[family] = ("dpp", family, 1.0)

    for alpha in (0.25, 0.50, 0.75):
        name = "%s_blend_%02d" % (family, int(alpha * 100))
        score = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * transformed
        )
        candidate_scores[name] = score
        candidate_recipes[name] = ("dpp", family, alpha)

# A distinct non-parametric quality blend checks whether the DPP gain, if
# any, is merely due to adding train-only entity histories.
for alpha in (0.15, 0.30):
    name = "empirical_bayes_blend_%02d" % int(alpha * 100)
    candidate_scores[name] = (
        (1.0 - alpha) * inc_valid_rank
        + alpha * eb_valid_rank
    )
    candidate_recipes[name] = ("eb_blend", None, alpha)

candidate_metrics = {}
for name, scores in candidate_scores.items():
    candidate_metrics[name] = evaluate(uva, yva, scores)

best_name = max(
    candidate_metrics,
    key=lambda name: candidate_metrics[name]["primary"]
)
best_valid_scores = candidate_scores[best_name]
best_metrics = candidate_metrics[best_name]

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: round(float(metric["primary"]), 7)
            for name, metric in candidate_metrics.items()
        },
        sort_keys=True
    )
)

base_top = np.argsort(-inc_valid, kind="stable")[:1000]
chosen_top = np.argsort(-best_valid_scores, kind="stable")[:1000]
top_overlap = len(np.intersect1d(base_top, chosen_top)) / 1000.0
print(
    "FINDINGS selected=%s top1000_global_overlap=%.4f"
    % (best_name, top_overlap)
)

# ----------------------------------------------------------------------
# Apply the exact validation-selected recipe to test features, without
# reading test labels.
# ----------------------------------------------------------------------
recipe_type, recipe_family, recipe_alpha = candidate_recipes[best_name]

if recipe_type == "incumbent":
    best_test_scores = inc_test
elif recipe_type == "eb":
    best_test_scores = eb_test_rank
elif recipe_type == "eb_blend":
    best_test_scores = (
        (1.0 - recipe_alpha) * inc_test_rank
        + recipe_alpha * eb_test_rank
    )
elif recipe_type == "dpp":
    family_to_parameters = {
        "dpp_content_fixed": ("content", 3.2, False),
        "dpp_content_uncertainty": ("content", 3.2, True),
        "dpp_context_uncertainty": ("context", 3.6, True),
    }
    kernel_name, beta, conditioned = family_to_parameters[recipe_family]
    transformed_test = dpp_rank_scores(
        test,
        inc_test,
        unc_test,
        kernel_name,
        beta=beta,
        uncertainty_conditioned=conditioned,
    )
    if recipe_alpha >= 1.0:
        best_test_scores = transformed_test
    else:
        best_test_scores = (
            (1.0 - recipe_alpha) * inc_test_rank
            + recipe_alpha * transformed_test
        )
else:
    raise RuntimeError("Unknown selected recipe")

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    # The empirical-Bayes scorer is the script's independent train-fitted
    # model; save it whenever the reported result may use the incumbent.
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(eb_valid_rank, dtype=np.float64),
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