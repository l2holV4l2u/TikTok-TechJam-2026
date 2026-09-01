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

SEED = 73129
rng = np.random.default_rng(SEED)

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

RAW_CATEGORICAL = [
    "user_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "music_type",
    "fans_user_num_range",
    "register_days_bucket",
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

TE_STRENGTHS = {
    "user_id": 120.0,
    "tag": 700.0,
    "tab": 700.0,
    "duration_bucket": 700.0,
    "upload_type": 500.0,
    "onehot_feat3": 160.0,
    "onehot_feat8": 160.0,
    "user_active_degree": 500.0,
    "music_type": 700.0,
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
    sizes = np.diff(np.r_[-1, np.flatnonzero(ends)]).astype(np.float64)

    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = (local_rank / denom).astype(np.float32)
    return result


def copula(rank):
    p = np.clip(np.asarray(rank, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return ndtri(p).astype(np.float32)


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
            float(TE_STRENGTHS[field]),
        )

    return tables


def build_features(split, split_name, te_tables, labels=None,
                   expected_history_names=None, row_index=None):
    columns = []
    names = []
    categorical_indices = []

    for key in ("video_id", "author_id"):
        history = historical_features(split_name, key=key)
        for name in sorted(history):
            if any(name.endswith(s) for s in HISTORY_SUFFIXES):
                x = np.asarray(history[name], dtype=np.float32)
                if row_index is not None:
                    x = x[row_index]
                x = np.nan_to_num(
                    x, nan=0.0, posinf=0.0, neginf=0.0
                )
                columns.append(x)
                names.append(name)

    history_names = names.copy()
    if (
        expected_history_names is not None
        and history_names != expected_history_names
    ):
        raise RuntimeError("History feature order mismatch")

    for field in TE_FIELDS:
        count, total, prior, strength = te_tables[field]
        ids = np.asarray(split.X[field], dtype=np.int64)
        ids = np.where((ids >= 0) & (ids < len(count)), ids, 0)

        if row_index is not None:
            ids = ids[row_index]

        c = count[ids]
        s = total[ids]

        if labels is not None:
            local_y = labels if row_index is None else labels[row_index]
            c = np.maximum(c - 1.0, 0.0)
            s = s - local_y

        rate = (s + strength * prior) / (c + strength)
        columns.append(rate.astype(np.float32))
        names.append(field + "_te")
        columns.append(np.log1p(c).astype(np.float32))
        names.append(field + "_count_log")

    for field in RAW_NUMERIC:
        x = np.asarray(split.num[field], dtype=np.float32)
        if row_index is not None:
            x = x[row_index]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        columns.append(x)
        names.append(field + "_log")

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    if row_index is not None:
        hour = hour[row_index]
    hour = np.mod(hour, 24.0)
    angle = 2.0 * np.pi * hour / 24.0
    columns.append(np.sin(angle).astype(np.float32))
    names.append("hour_sin")
    columns.append(np.cos(angle).astype(np.float32))
    names.append("hour_cos")

    for field in RAW_CATEGORICAL:
        x = np.asarray(split.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])
        x = np.where((x >= 0) & (x < card), x, 0)
        if row_index is not None:
            x = x[row_index]

        categorical_indices.append(len(columns))
        columns.append(x.astype(np.float32))
        names.append("cat_" + field)

    matrix = np.column_stack(columns).astype(np.float32)
    matrix = np.nan_to_num(
        matrix, nan=0.0, posinf=0.0, neginf=0.0
    )

    return matrix, names, history_names, categorical_indices


def domain_features(split, row_index=None):
    columns = []

    low_fields = [
        "tag",
        "tab",
        "duration_bucket",
        "upload_type",
        "onehot_feat3",
        "onehot_feat8",
        "user_active_degree",
        "music_type",
        "fans_user_num_range",
        "register_days_bucket",
        "hour",
    ]

    for field in low_fields:
        x = np.asarray(split.X[field], dtype=np.float32)
        if row_index is not None:
            x = x[row_index]
        columns.append(x)

    for field in RAW_NUMERIC:
        x = np.asarray(split.num[field], dtype=np.float32)
        if row_index is not None:
            x = x[row_index]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(
            np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        )

    return np.column_stack(columns).astype(np.float32)


def fit_domain_ratio(train, target_indices):
    dates = np.asarray(train.date, dtype=np.int32)
    unique_dates = np.sort(np.unique(dates))

    early_mask = np.isin(dates, unique_dates[:5])
    late_mask = np.isin(dates, unique_dates[-4:])

    early_idx = np.flatnonzero(early_mask)
    late_idx = np.flatnonzero(late_mask)

    take = min(350000, len(early_idx), len(late_idx))
    early_sample = rng.choice(early_idx, size=take, replace=False)
    late_sample = rng.choice(late_idx, size=take, replace=False)

    domain_idx = np.concatenate([early_sample, late_sample])
    domain_y = np.concatenate([
        np.zeros(take, dtype=np.int8),
        np.ones(take, dtype=np.int8),
    ])

    permutation = rng.permutation(len(domain_idx))
    domain_idx = domain_idx[permutation]
    domain_y = domain_y[permutation]

    dx = domain_features(train, domain_idx)
    dset = lgb.Dataset(dx, label=domain_y, free_raw_data=True)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.06,
        "num_leaves": 31,
        "min_data_in_leaf": 1000,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 8.0,
        "max_bin": 127,
        "num_threads": -1,
        "seed": SEED,
        "verbose": -1,
    }

    model = lgb.train(params, dset, num_boost_round=110)

    del dx, dset, domain_idx, domain_y
    gc.collect()

    tx = domain_features(train, target_indices)
    probability = model.predict(tx).astype(np.float32)
    probability = np.clip(probability, 0.05, 0.95)

    ratio = probability / (1.0 - probability)
    ratio = np.clip(ratio, 0.25, 4.0).astype(np.float32)
    ratio /= np.mean(ratio)

    print(
        "FINDINGS domain_ratio_q05=%.4f q50=%.4f q95=%.4f "
        "early_dates=%s late_dates=%s"
        % (
            float(np.quantile(ratio, 0.05)),
            float(np.quantile(ratio, 0.50)),
            float(np.quantile(ratio, 0.95)),
            str(unique_dates[:5].tolist()),
            str(unique_dates[-4:].tolist()),
        ),
        flush=True,
    )

    del tx, model
    gc.collect()
    return ratio


def recency_weights(selected_dates, latest_date, half_life):
    if half_life is None:
        w = np.ones(len(selected_dates), dtype=np.float32)
    else:
        age = (latest_date - selected_dates).astype(np.float32)
        w = np.exp(
            -np.log(2.0) * age / float(half_life)
        ).astype(np.float32)
    w /= np.mean(w)
    return w


def fit_lgb_model(train_x, y, weights, categorical_indices,
                  family, half_life_name):
    common = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "max_bin": 127,
        "num_threads": -1,
        "seed": SEED + len(half_life_name),
        "feature_fraction_seed": SEED + 17,
        "bagging_seed": SEED + 31,
        "data_random_seed": SEED + 47,
        "verbosity": -1,
    }

    if family == "gbdt":
        params = dict(common)
        params.update({
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "min_data_in_leaf": 700,
            "feature_fraction": 0.82,
            "bagging_fraction": 0.88,
            "bagging_freq": 1,
            "lambda_l1": 0.15,
            "lambda_l2": 8.0,
            "min_gain_to_split": 0.01,
        })
        rounds = 260

    elif family == "linear_tree":
        params = dict(common)
        params.update({
            "boosting_type": "gbdt",
            "linear_tree": True,
            "num_leaves": 23,
            "min_data_in_leaf": 1800,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.82,
            "bagging_freq": 1,
            "lambda_l1": 0.5,
            "lambda_l2": 15.0,
            "linear_lambda": 12.0,
        })
        rounds = 120

    elif family == "random_forest":
        params = dict(common)
        params.update({
            "boosting_type": "rf",
            "learning_rate": 1.0,
            "num_leaves": 63,
            "min_data_in_leaf": 600,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.62,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
        })
        rounds = 180

    else:
        raise ValueError(family)

    dset = lgb.Dataset(
        train_x,
        label=y,
        weight=weights,
        categorical_feature=categorical_indices,
        free_raw_data=True,
    )

    model = lgb.train(params, dset, num_boost_round=rounds)
    del dset
    gc.collect()
    return model


def add_candidate(results, name, uid, y, scores):
    metrics = evaluate(uid, y, scores)
    results[name] = float(metrics["primary"])
    return metrics


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

train_y_full = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y)
valid_uid = np.asarray(valid.user_id, dtype=np.int64)

train_dates_full = np.asarray(train.date, dtype=np.int32)
unique_dates = np.sort(np.unique(train_dates_full))
latest_date = int(unique_dates[-1])

# Preserve every row from the recent half of train and retain a deterministic
# fraction of older rows. This spends capacity on the boundary regime without
# discarding old support entirely.
recent_mask = train_dates_full >= unique_dates[-7]
old_random = rng.random(len(train_dates_full)) < 0.32
selected_mask = recent_mask | old_random
train_idx = np.flatnonzero(selected_mask)

train_y = train_y_full[train_idx]
selected_dates = train_dates_full[train_idx]

print(
    "FINDINGS train_rows_full=%d train_rows_selected=%d "
    "recent_fraction_selected=%.4f"
    % (
        len(train_y_full),
        len(train_y),
        float(np.mean(recent_mask[train_idx])),
    ),
    flush=True,
)

te_tables = fit_te_tables(train, train_y_full)
domain_ratio = fit_domain_ratio(train, train_idx)

train_x, feature_names, history_names, cat_indices = build_features(
    train,
    "train",
    te_tables,
    labels=train_y_full,
    row_index=train_idx,
)

valid_x, valid_names, _, valid_cat_indices = build_features(
    valid,
    "valid",
    te_tables,
    labels=None,
    expected_history_names=history_names,
)

if feature_names != valid_names or cat_indices != valid_cat_indices:
    raise RuntimeError("Feature mismatch")

print(
    "FINDINGS feature_dim=%d history_dim=%d categorical_dim=%d"
    % (train_x.shape[1], len(history_names), len(cat_indices)),
    flush=True,
)

inc_valid_raw = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_valid_rank = within_user_rank(valid_uid, inc_valid_raw)
inc_valid_copula = copula(inc_valid_rank)

candidate_results = {}
inc_metrics = add_candidate(
    candidate_results,
    "trusted_incumbent",
    valid_uid,
    valid_y,
    inc_valid_rank,
)

models = {}
valid_ranks = {}
standalone_metrics = {}

# Main-model recency sweep: these are independent target fits, not side
# components whose contribution is subsequently forced near zero.
for half_life in (2.0, 4.0, 8.0, None):
    label = "uniform" if half_life is None else "hl%.0f" % half_life
    weights = recency_weights(
        selected_dates, latest_date, half_life
    )

    print(
        "FINDINGS fitting=gbdt weighting=%s weight_q05=%.4f "
        "weight_q95=%.4f"
        % (
            label,
            float(np.quantile(weights, 0.05)),
            float(np.quantile(weights, 0.95)),
        ),
        flush=True,
    )

    model = fit_lgb_model(
        train_x,
        train_y,
        weights,
        cat_indices,
        family="gbdt",
        half_life_name=label,
    )
    raw = model.predict(valid_x).astype(np.float32)
    rank = within_user_rank(valid_uid, raw)

    name = "gbdt_" + label
    metrics = add_candidate(
        candidate_results, name, valid_uid, valid_y, rank
    )

    models[name] = model
    valid_ranks[name] = rank
    standalone_metrics[name] = metrics

    print(
        "FINDINGS family=%s primary=%.6f gauc=%.6f ndcg5=%.6f "
        "corr_incumbent=%.6f"
        % (
            name,
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
            float(np.corrcoef(rank, inc_valid_rank)[0, 1]),
        ),
        flush=True,
    )

    del weights, raw
    gc.collect()

# The drift ratio is estimated without labels, entirely inside train. It is
# combined with moderate recency for the structurally different families.
base_weight = recency_weights(selected_dates, latest_date, 4.0)
drift_weight = (base_weight * domain_ratio).astype(np.float32)
drift_weight /= np.mean(drift_weight)

for family in ("linear_tree", "random_forest"):
    name = family + "_drift_ratio"

    print(
        "FINDINGS fitting=%s weighting=recency_x_train_domain_ratio "
        "weight_q05=%.4f weight_q95=%.4f"
        % (
            family,
            float(np.quantile(drift_weight, 0.05)),
            float(np.quantile(drift_weight, 0.95)),
        ),
        flush=True,
    )

    model = fit_lgb_model(
        train_x,
        train_y,
        drift_weight,
        cat_indices,
        family=family,
        half_life_name="domain",
    )
    raw = model.predict(valid_x).astype(np.float32)
    rank = within_user_rank(valid_uid, raw)

    metrics = add_candidate(
        candidate_results, name, valid_uid, valid_y, rank
    )

    models[name] = model
    valid_ranks[name] = rank
    standalone_metrics[name] = metrics

    print(
        "FINDINGS family=%s primary=%.6f gauc=%.6f ndcg5=%.6f "
        "corr_incumbent=%.6f"
        % (
            name,
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
            float(np.corrcoef(rank, inc_valid_rank)[0, 1]),
        ),
        flush=True,
    )

    del raw
    gc.collect()

# Also test a cross-family consensus. It rewards ordering shared by the best
# partitioned, piecewise-linear, and bagged predictors.
best_gbdt_name = max(
    [n for n in valid_ranks if n.startswith("gbdt_")],
    key=lambda n: standalone_metrics[n]["primary"],
)
consensus_members = [
    best_gbdt_name,
    "linear_tree_drift_ratio",
    "random_forest_drift_ratio",
]
consensus_rank = np.mean(
    np.column_stack([valid_ranks[n] for n in consensus_members]),
    axis=1,
).astype(np.float32)

consensus_metrics = add_candidate(
    candidate_results,
    "cross_family_consensus",
    valid_uid,
    valid_y,
    consensus_rank,
)
valid_ranks["cross_family_consensus"] = consensus_rank

best_scores = inc_valid_rank.copy()
best_metrics = inc_metrics
best_source = "trusted_incumbent"
best_alpha = 0.0
best_transform = "rank"
best_gamma = 1.0

alphas = [0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45]
gammas = [1.0, 2.0, 4.0]

for source, rank in valid_ranks.items():
    for gamma in gammas:
        shaped = np.power(
            np.clip(rank, 0.0, 1.0), gamma
        ).astype(np.float32)

        for transform, inc_base, own_base in [
            ("rank", inc_valid_rank, shaped),
            ("copula", inc_valid_copula, copula(shaped)),
        ]:
            for alpha in alphas:
                blended = (
                    (1.0 - alpha) * inc_base
                    + alpha * own_base
                ).astype(np.float32)

                candidate_name = (
                    "%s_blend_%s_g%.0f_a%.2f"
                    % (source, transform, gamma, alpha)
                )
                metrics = add_candidate(
                    candidate_results,
                    candidate_name,
                    valid_uid,
                    valid_y,
                    blended,
                )

                if float(metrics["primary"]) > float(best_metrics["primary"]):
                    best_scores = blended.copy()
                    best_metrics = metrics
                    best_source = source
                    best_alpha = float(alpha)
                    best_transform = transform
                    best_gamma = float(gamma)

final_metrics = evaluate(valid_uid, valid_y, best_scores)

best_standalone_source = max(
    standalone_metrics,
    key=lambda n: standalone_metrics[n]["primary"],
)
raw_to_save = valid_ranks.get(
    best_source, valid_ranks[best_standalone_source]
)

print(
    "FINDINGS winner=%s alpha=%.2f transform=%s gamma=%.1f "
    "incumbent_primary=%.6f final_primary=%.6f "
    "best_standalone=%s standalone_primary=%.6f"
    % (
        best_source,
        best_alpha,
        best_transform,
        best_gamma,
        float(inc_metrics["primary"]),
        float(final_metrics["primary"]),
        best_standalone_source,
        float(standalone_metrics[best_standalone_source]["primary"]),
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
        np.asarray(raw_to_save, dtype=np.float64),
    )

del valid_x, inc_valid_raw
gc.collect()

test = load("test")
test_uid = np.asarray(test.user_id, dtype=np.int64)

inc_test_raw = np.asarray(
    np.load(inc_test_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_test_rank = within_user_rank(test_uid, inc_test_raw)

if best_source == "trusted_incumbent":
    test_scores = inc_test_rank
else:
    test_x, test_names, _, test_cat_indices = build_features(
        test,
        "test",
        te_tables,
        labels=None,
        expected_history_names=history_names,
    )

    if test_names != feature_names or test_cat_indices != cat_indices:
        raise RuntimeError("Test feature mismatch")

    if best_source == "cross_family_consensus":
        test_member_ranks = []
        for name in consensus_members:
            raw = models[name].predict(test_x).astype(np.float32)
            test_member_ranks.append(within_user_rank(test_uid, raw))
            del raw
        selected_test_rank = np.mean(
            np.column_stack(test_member_ranks), axis=1
        ).astype(np.float32)
    else:
        raw = models[best_source].predict(test_x).astype(np.float32)
        selected_test_rank = within_user_rank(test_uid, raw)
        del raw

    shaped_test = np.power(
        np.clip(selected_test_rank, 0.0, 1.0),
        best_gamma,
    ).astype(np.float32)

    if best_transform == "copula":
        incumbent_base = copula(inc_test_rank)
        own_base = copula(shaped_test)
    else:
        incumbent_base = inc_test_rank
        own_base = shaped_test

    test_scores = (
        (1.0 - best_alpha) * incumbent_base
        + best_alpha * own_base
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