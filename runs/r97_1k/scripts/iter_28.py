import os
import gc
import json
import time
import warnings
import numpy as np
import lightgbm as lgb
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

np.random.seed(27183)

TE_FIELDS = [
    "user_id",
    "tag",
    "tab",
    "author_id",
    "video_id",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "music_type",
]

TE_STRENGTH = {
    "user_id": 120.0,
    "tag": 700.0,
    "tab": 900.0,
    "author_id": 80.0,
    "video_id": 45.0,
    "duration_bucket": 800.0,
    "upload_type": 600.0,
    "onehot_feat3": 150.0,
    "onehot_feat8": 150.0,
    "user_active_degree": 600.0,
    "music_type": 800.0,
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

    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = np.arange(n, dtype=np.float64) - start_pos

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    sizes = np.diff(np.r_[-1, np.flatnonzero(ends)]).astype(np.float64)

    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denominator = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = (local_rank / denominator).astype(np.float32)
    return result


def copula(rank):
    p = np.clip(np.asarray(rank, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return ndtri(p).astype(np.float32)


def fit_te_tables(train, labels):
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

        tables[field] = (
            counts,
            sums,
            prior,
            float(TE_STRENGTH[field]),
        )

    return tables


def make_te_features(split, tables, labels=None):
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
        count = np.log1p(c)

        columns.append(rate.astype(np.float32))
        columns.append(count.astype(np.float32))

    return columns


def make_history_features(split_name, expected_names=None):
    columns = []
    names = []
    count_columns = []

    for key in ("video_id", "author_id"):
        history = historical_features(split_name, key=key)

        for name in sorted(history):
            if any(name.endswith(s) for s in HISTORY_SUFFIXES):
                x = np.asarray(history[name], dtype=np.float32)
                x = np.nan_to_num(
                    x, nan=0.0, posinf=0.0, neginf=0.0
                )
                columns.append(x)
                names.append(name)

                if name.endswith("train_count_log1p"):
                    count_columns.append(x)

    if expected_names is not None and names != expected_names:
        raise RuntimeError("Historical feature order mismatch")

    if not count_columns:
        raise RuntimeError("History count columns unavailable")

    confidence_source = np.mean(
        np.column_stack(count_columns), axis=1
    ).astype(np.float32)

    return columns, names, confidence_source


def make_matrix(split, split_name, tables, labels=None,
                expected_history_names=None):
    columns, history_names, confidence_source = make_history_features(
        split_name, expected_names=expected_history_names
    )

    for name in RAW_NUMERIC:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    hour = np.mod(hour, 24.0)
    angle = hour * (2.0 * np.pi / 24.0)
    columns.append(np.sin(angle).astype(np.float32))
    columns.append(np.cos(angle).astype(np.float32))

    columns.extend(make_te_features(split, tables, labels=labels))

    matrix = np.column_stack(columns).astype(np.float32)
    matrix = np.nan_to_num(
        matrix, nan=0.0, posinf=0.0, neginf=0.0
    )

    del columns
    gc.collect()
    return matrix, history_names, confidence_source


def confidence_gate(count_log):
    # The gate is fixed before looking at validation labels. It approaches
    # zero for very sparse entities and one only for well-supported history.
    x = (np.asarray(count_log, dtype=np.float32) - 3.0) / 2.0
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def form_gate(kind, incumbent_z, candidate_z, confidence):
    disagreement = np.abs(candidate_z - incumbent_z).astype(np.float32)

    if kind == "high_disagreement":
        return np.clip(
            (disagreement - 0.35) / 1.50, 0.0, 1.0
        ).astype(np.float32)

    if kind == "confidence":
        return confidence

    if kind == "confidence_high_disagreement":
        return (
            confidence
            * np.clip((disagreement - 0.35) / 1.50, 0.0, 1.0)
        ).astype(np.float32)

    if kind == "top_union_confidence":
        topness = 1.0 / (
            1.0 + np.exp(
                -2.0 * (np.maximum(incumbent_z, candidate_z) - 0.45)
            )
        )
        return (confidence * topness).astype(np.float32)

    if kind == "candidate_promotions":
        promotion = np.clip(
            (candidate_z - incumbent_z - 0.20) / 1.30,
            0.0,
            1.0,
        )
        return (confidence * promotion).astype(np.float32)

    if kind == "candidate_demotions":
        demotion = np.clip(
            (incumbent_z - candidate_z - 0.20) / 1.30,
            0.0,
            1.0,
        )
        return (confidence * demotion).astype(np.float32)

    raise ValueError("Unknown gate: " + kind)


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
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_uid = np.asarray(valid.user_id, dtype=np.int64)

tables = fit_te_tables(train, train_y)

train_x, history_names, train_count_log = make_matrix(
    train,
    "train",
    tables,
    labels=train_y,
)
valid_x, _, valid_count_log = make_matrix(
    valid,
    "valid",
    tables,
    labels=None,
    expected_history_names=history_names,
)

train_dates = np.asarray(train.date, dtype=np.int32)
latest_date = int(train_dates.max())
date_age = (latest_date - train_dates).astype(np.float32)

sample_weight = np.exp(
    -np.log(2.0) * date_age / 4.0
).astype(np.float32)
sample_weight /= np.mean(sample_weight)

dataset = lgb.Dataset(
    train_x,
    label=train_y,
    weight=sample_weight,
    free_raw_data=False,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 1500,
    "feature_fraction": 0.86,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": 27183,
    "feature_fraction_seed": 27183,
    "bagging_seed": 27183,
    "data_random_seed": 27183,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}

model = lgb.train(
    params,
    dataset,
    num_boost_round=180,
)

candidate_valid_raw = model.predict(
    valid_x, num_iteration=model.current_iteration()
).astype(np.float32)
candidate_valid_rank = within_user_rank(
    valid_uid, candidate_valid_raw
)

inc_valid_raw = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_valid_rank = within_user_rank(valid_uid, inc_valid_raw)

inc_z = copula(inc_valid_rank)
cand_z = copula(candidate_valid_rank)
valid_confidence = confidence_gate(valid_count_log)

inc_metrics = evaluate(valid_uid, valid_y, inc_valid_rank)
candidate_metrics = evaluate(
    valid_uid, valid_y, candidate_valid_rank
)

candidate_results = {
    "trusted_incumbent": float(inc_metrics["primary"]),
    "boosted_candidate_standalone": float(
        candidate_metrics["primary"]
    ),
}

best_scores = inc_valid_rank.copy()
best_metrics = inc_metrics
best_name = "trusted_incumbent"
best_kind = "none"
best_alpha = 0.0
best_space = "rank"

alphas = [0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.32]
gate_kinds = [
    "high_disagreement",
    "confidence",
    "confidence_high_disagreement",
    "top_union_confidence",
    "candidate_promotions",
    "candidate_demotions",
]

# No-gate controls in both rank and Gaussian-copula score spaces.
for space, incumbent_base, candidate_base in [
    ("rank", inc_valid_rank, candidate_valid_rank),
    ("copula", inc_z, cand_z),
]:
    for alpha in alphas:
        score = (
            incumbent_base
            + alpha * (candidate_base - incumbent_base)
        ).astype(np.float32)
        metrics = evaluate(valid_uid, valid_y, score)
        name = "no_gate_%s_a%.2f" % (space, alpha)
        candidate_results[name] = float(metrics["primary"])

        if float(metrics["primary"]) > float(best_metrics["primary"]):
            best_scores = score.copy()
            best_metrics = metrics
            best_name = name
            best_kind = "no_gate"
            best_alpha = float(alpha)
            best_space = space

# Gated corrections use fixed unlabeled functions of confidence, score
# position, and model disagreement. Validation only chooses among the same
# permitted finite blend grid used for ordinary incumbent blending.
for kind in gate_kinds:
    gate = form_gate(kind, inc_z, cand_z, valid_confidence)

    print(
        "FINDINGS gate=%s mean=%.6f p90=%.6f active_share=%.6f"
        % (
            kind,
            float(np.mean(gate)),
            float(np.quantile(gate, 0.90)),
            float(np.mean(gate > 0.10)),
        ),
        flush=True,
    )

    for alpha in alphas:
        score = (
            inc_z + alpha * gate * (cand_z - inc_z)
        ).astype(np.float32)
        metrics = evaluate(valid_uid, valid_y, score)
        name = "%s_a%.2f" % (kind, alpha)
        candidate_results[name] = float(metrics["primary"])

        if float(metrics["primary"]) > float(best_metrics["primary"]):
            best_scores = score.copy()
            best_metrics = metrics
            best_name = name
            best_kind = kind
            best_alpha = float(alpha)
            best_space = "copula"

rank_corr = float(
    np.corrcoef(inc_valid_rank, candidate_valid_rank)[0, 1]
)
mean_disagreement = float(
    np.mean(np.abs(inc_z - cand_z))
)

print(
    "FINDINGS candidate_primary=%.6f candidate_gauc=%.6f "
    "candidate_ndcg5=%.6f incumbent_primary=%.6f "
    "rank_corr=%.6f mean_copula_disagreement=%.6f"
    % (
        float(candidate_metrics["primary"]),
        float(candidate_metrics["gauc"]),
        float(candidate_metrics["ndcg@5"]),
        float(inc_metrics["primary"]),
        rank_corr,
        mean_disagreement,
    ),
    flush=True,
)
print(
    "FINDINGS winner=%s kind=%s alpha=%.2f space=%s "
    "primary=%.6f gauc=%.6f ndcg5=%.6f"
    % (
        best_name,
        best_kind,
        best_alpha,
        best_space,
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
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
        np.asarray(candidate_valid_rank, dtype=np.float64),
    )

del dataset
del train_x
del valid_x
del train_y
del sample_weight
del train_dates
del date_age
del candidate_valid_raw
del inc_valid_raw
gc.collect()

test = load("test")
test_uid = np.asarray(test.user_id, dtype=np.int64)

inc_test_raw = np.asarray(
    np.load(inc_test_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_test_rank = within_user_rank(test_uid, inc_test_raw)

if best_kind == "none":
    test_scores = inc_test_rank
else:
    test_x, _, test_count_log = make_matrix(
        test,
        "test",
        tables,
        labels=None,
        expected_history_names=history_names,
    )

    candidate_test_raw = model.predict(
        test_x, num_iteration=model.current_iteration()
    ).astype(np.float32)
    candidate_test_rank = within_user_rank(
        test_uid, candidate_test_raw
    )

    if best_kind == "no_gate":
        if best_space == "copula":
            incumbent_base = copula(inc_test_rank)
            candidate_base = copula(candidate_test_rank)
        else:
            incumbent_base = inc_test_rank
            candidate_base = candidate_test_rank

        test_scores = (
            incumbent_base
            + best_alpha * (candidate_base - incumbent_base)
        ).astype(np.float32)
    else:
        inc_test_z = copula(inc_test_rank)
        candidate_test_z = copula(candidate_test_rank)
        test_confidence = confidence_gate(test_count_log)
        test_gate = form_gate(
            best_kind,
            inc_test_z,
            candidate_test_z,
            test_confidence,
        )
        test_scores = (
            inc_test_z
            + best_alpha
            * test_gate
            * (candidate_test_z - inc_test_z)
        ).astype(np.float32)

    del test_x
    gc.collect()

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)

print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)