import os
import time
import json
import gc
import random
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260829
random.seed(SEED)
np.random.seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
    "onehot_feat1", "onehot_feat3", "onehot_feat7", "onehot_feat8",
    "user_active_degree", "video_type",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HALF_LIVES = {
    "binary_uniform": None,
    "binary_recent": 4.0,
    "lambdarank_recent": 4.0,
}
NUM_ROUNDS = {
    "binary_uniform": 260,
    "binary_recent": 300,
    "lambdarank_recent": 220,
}


def age_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    if half_life is None:
        return np.ones(len(dates), dtype=np.float64)
    # All fitting windows used here are within April 2022.
    endpoint = int(dates.max())
    age = endpoint - dates
    return np.exp2(-age.astype(np.float64) / float(half_life))


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def within_user_rank(user_ids, scores):
    """Vectorized ascending percentile rank; ranking is all that metrics use."""
    u = np.asarray(user_ids)
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, s, u))
    us = u[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = us[1:] != us[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    group_start = np.repeat(starts, sizes)
    group_size = np.repeat(sizes, sizes)
    position = np.arange(n, dtype=np.int64) - group_start

    ranked_sorted = (position + 0.5) / group_size
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def aggregate_rate(src_key, eval_key, y, w, cardinality, smoothing):
    src_key = np.asarray(src_key, dtype=np.int64)
    eval_key = np.asarray(eval_key, dtype=np.int64)
    y64 = np.asarray(y, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)

    prior = float(np.sum(w64 * y64) / np.sum(w64))
    count = np.bincount(
        src_key, weights=w64, minlength=cardinality
    ).astype(np.float64, copy=False)
    positive = np.bincount(
        src_key, weights=w64 * y64, minlength=cardinality
    ).astype(np.float64, copy=False)

    # Leave-one-out features prevent target leakage in the fitting table.
    loo_count = count[src_key] - w64
    loo_positive = positive[src_key] - w64 * y64
    src_rate = (
        loo_positive + smoothing * prior
    ) / (loo_count + smoothing)

    eval_rate = (
        positive[eval_key] + smoothing * prior
    ) / (count[eval_key] + smoothing)

    src_log_count = np.log1p(np.maximum(loo_count, 0.0))
    eval_log_count = np.log1p(count[eval_key])
    return (
        src_rate.astype(np.float32),
        eval_rate.astype(np.float32),
        src_log_count.astype(np.float32),
        eval_log_count.astype(np.float32),
    )


def make_stat_features(source, target, y_source, weights):
    specs = [
        ("video", source.X["video_id"], target.X["video_id"], 8000, 25.0),
        ("author", source.X["author_id"], target.X["author_id"], 7000, 35.0),
        ("tag", source.X["tag"], target.X["tag"], 64, 60.0),
        (
            "user_tag",
            source.X["user_id"] * 64 + source.X["tag"],
            target.X["user_id"] * 64 + target.X["tag"],
            30000 * 64,
            8.0,
        ),
        (
            "user_tab",
            source.X["user_id"] * 20 + source.X["tab"],
            target.X["user_id"] * 20 + target.X["tab"],
            30000 * 20,
            10.0,
        ),
        (
            "user_duration",
            source.X["user_id"] * 12 + source.X["duration_bucket"],
            target.X["user_id"] * 12 + target.X["duration_bucket"],
            30000 * 12,
            10.0,
        ),
    ]

    src_cols = []
    tgt_cols = []
    target_rates = {}
    for name, sk, tk, cardinality, smoothing in specs:
        sr, tr, sc, tc = aggregate_rate(
            sk, tk, y_source, weights, cardinality, smoothing
        )
        src_cols.extend([sr, sc])
        tgt_cols.extend([tr, tc])
        target_rates[name] = tr

    return (
        np.column_stack(src_cols).astype(np.float32, copy=False),
        np.column_stack(tgt_cols).astype(np.float32, copy=False),
        target_rates,
    )


def empirical_bayes_score(target_rates):
    # Entity quality and user-context affinity form predictions differently
    # from either boosted-tree candidate.
    components = [
        (2.0, target_rates["video"]),
        (1.5, target_rates["author"]),
        (0.8, target_rates["tag"]),
        (2.2, target_rates["user_tag"]),
        (1.2, target_rates["user_tab"]),
        (1.2, target_rates["user_duration"]),
    ]
    total = sum(weight for weight, _ in components)
    result = np.zeros(len(components[0][1]), dtype=np.float64)
    for weight, rate in components:
        result += weight * safe_logit(rate)
    return result / total


def base_matrix(split):
    cols = []
    for name in CAT_FIELDS:
        cols.append(np.asarray(split.X[name], dtype=np.float32))
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        cols.append(x.astype(np.float32))
    return np.column_stack(cols).astype(np.float32, copy=False)


def full_matrix(base, stats):
    return np.ascontiguousarray(
        np.concatenate([base, stats], axis=1), dtype=np.float32
    )


def train_binary(x, y, weights, rounds):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 47,
        "max_depth": -1,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.86,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.5,
        "max_bin": 127,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": min(12, max(1, os.cpu_count() or 1)),
        "verbose": -1,
    }
    ds = lgb.Dataset(
        x,
        label=y,
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    return lgb.train(params, ds, num_boost_round=rounds)


def train_ranker(x, y, user_ids, weights, rounds):
    order = np.argsort(np.asarray(user_ids), kind="stable")
    sorted_users = np.asarray(user_ids)[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    groups = np.diff(boundaries).astype(np.int32)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.045,
        "num_leaves": 39,
        "min_data_in_leaf": 160,
        "feature_fraction": 0.84,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 3.0,
        "max_bin": 127,
        "seed": SEED + 20,
        "feature_fraction_seed": SEED + 21,
        "bagging_seed": SEED + 22,
        "num_threads": min(12, max(1, os.cpu_count() or 1)),
        "verbose": -1,
    }
    ds = lgb.Dataset(
        x[order],
        label=np.asarray(y)[order],
        weight=np.asarray(weights)[order],
        group=groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    return lgb.train(params, ds, num_boost_round=rounds)


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")
if len(inc_valid) != len(y_valid):
    raise RuntimeError("Incumbent validation prediction length mismatch")

base_train = base_matrix(train)
base_valid = base_matrix(valid)

candidate_scores = {"trusted_incumbent": inc_valid}
candidate_primary = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    )
}
candidate_metadata = {
    "trusted_incumbent": {"family": "incumbent", "alpha": 0.0}
}

# Reuse the recent-stat construction for EB and both recent boosted models.
w_recent = age_weights(train.date, 4.0)
stats_train_recent, stats_valid_recent, rates_valid_recent = (
    make_stat_features(train, valid, y_train, w_recent)
)

eb_valid = empirical_bayes_score(rates_valid_recent)
candidate_scores["empirical_bayes_recent"] = eb_valid
candidate_primary["empirical_bayes_recent"] = float(
    evaluate(valid.user_id, y_valid, eb_valid)["primary"]
)
candidate_metadata["empirical_bayes_recent"] = {
    "family": "empirical_bayes_recent", "alpha": 1.0
}

trained_models = {}
feature_cache = {}

for family in ["binary_uniform", "binary_recent", "lambdarank_recent"]:
    half_life = HALF_LIVES[family]
    if half_life == 4.0:
        stats_train = stats_train_recent
        stats_valid = stats_valid_recent
    else:
        weights_uniform = age_weights(train.date, None)
        stats_train, stats_valid, _ = make_stat_features(
            train, valid, y_train, weights_uniform
        )

    x_train = full_matrix(base_train, stats_train)
    x_valid = full_matrix(base_valid, stats_valid)
    weights = age_weights(train.date, half_life)

    if family.startswith("binary"):
        model = train_binary(
            x_train, y_train, weights, NUM_ROUNDS[family]
        )
    else:
        model = train_ranker(
            x_train, y_train, train.user_id, weights,
            NUM_ROUNDS[family]
        )

    pred = model.predict(
        x_valid, num_iteration=model.best_iteration
    ).astype(np.float64, copy=False)
    trained_models[family] = model
    feature_cache[family] = (stats_train, stats_valid)
    candidate_scores[family] = pred
    candidate_primary[family] = float(
        evaluate(valid.user_id, y_valid, pred)["primary"]
    )
    candidate_metadata[family] = {"family": family, "alpha": 1.0}

    print(
        "FINDINGS family=%s primary=%.6f"
        % (family, candidate_primary[family]),
        flush=True,
    )

    del x_train, x_valid
    gc.collect()

# Blend every structurally new ranker with the incumbent in rank space.
inc_rank = within_user_rank(valid.user_id, inc_valid)
for family in [
    "empirical_bayes_recent",
    "binary_uniform",
    "binary_recent",
    "lambdarank_recent",
]:
    family_rank = within_user_rank(
        valid.user_id, candidate_scores[family]
    )
    for alpha in [0.20, 0.35, 0.50, 0.65, 0.80]:
        name = "%s_rankblend_%.2f" % (family, alpha)
        score = (1.0 - alpha) * inc_rank + alpha * family_rank
        metric = evaluate(valid.user_id, y_valid, score)
        candidate_scores[name] = score
        candidate_primary[name] = float(metric["primary"])
        candidate_metadata[name] = {
            "family": family,
            "alpha": float(alpha),
        }

best_name = max(candidate_primary, key=candidate_primary.get)
best_valid_scores = candidate_scores[best_name]
best_metrics = evaluate(valid.user_id, y_valid, best_valid_scores)
selected_family = candidate_metadata[best_name]["family"]
selected_alpha = candidate_metadata[best_name]["alpha"]

print(
    "CANDIDATES " + json.dumps(
        candidate_primary, sort_keys=True, separators=(", ", ": ")
    ),
    flush=True,
)
print(
    "FINDINGS selected=%s family=%s incumbent_rank_weight=%.2f "
    "new_rank_weight=%.2f"
    % (
        best_name,
        selected_family,
        1.0 - selected_alpha,
        selected_alpha,
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Refit the selected recipe on train + validation, then score test without
# reading or otherwise touching test labels.
test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("Incumbent test prediction length mismatch")

if selected_family == "incumbent" or selected_alpha == 0.0:
    test_scores = inc_test.copy()
else:
    combined_y = np.concatenate([
        y_train,
        y_valid,
    ]).astype(np.int8, copy=False)

    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {}
    for name in CAT_FIELDS:
        combined.X[name] = np.concatenate([
            np.asarray(train.X[name]),
            np.asarray(valid.X[name]),
        ])
    # Statistics also require these fields, already included in CAT_FIELDS.
    combined.num = {}
    for name in NUM_FIELDS:
        combined.num[name] = np.concatenate([
            np.asarray(train.num[name]),
            np.asarray(valid.num[name]),
        ])
    combined.date = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])
    combined.user_id = np.concatenate([
        np.asarray(train.user_id),
        np.asarray(valid.user_id),
    ])

    refit_half_life = (
        4.0 if selected_family != "binary_uniform" else None
    )
    combined_weights = age_weights(
        combined.date, refit_half_life
    )
    stats_combined, stats_test, rates_test = make_stat_features(
        combined, test, combined_y, combined_weights
    )

    if selected_family == "empirical_bayes_recent":
        new_test_scores = empirical_bayes_score(rates_test)
    else:
        base_combined = base_matrix(combined)
        base_test = base_matrix(test)
        x_combined = full_matrix(base_combined, stats_combined)
        x_test = full_matrix(base_test, stats_test)

        if selected_family.startswith("binary"):
            final_model = train_binary(
                x_combined,
                combined_y,
                combined_weights,
                NUM_ROUNDS[selected_family],
            )
        elif selected_family == "lambdarank_recent":
            final_model = train_ranker(
                x_combined,
                combined_y,
                combined.user_id,
                combined_weights,
                NUM_ROUNDS[selected_family],
            )
        else:
            raise RuntimeError(
                "Unknown selected family: " + selected_family
            )

        new_test_scores = final_model.predict(
            x_test, num_iteration=final_model.best_iteration
        ).astype(np.float64, copy=False)

    if selected_alpha >= 1.0:
        test_scores = new_test_scores
    else:
        test_scores = (
            (1.0 - selected_alpha)
            * within_user_rank(test.user_id, inc_test)
            + selected_alpha
            * within_user_rank(test.user_id, new_test_scores)
        )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        best_metrics["primary"],
        best_metrics["gauc"],
        best_metrics["ndcg@5"],
        elapsed,
    )
)