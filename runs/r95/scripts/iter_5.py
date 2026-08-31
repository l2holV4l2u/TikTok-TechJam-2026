import os
import time
import json
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
NTHREAD = max(1, min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

# Mostly stationary side fields plus the three central identities. High-missing,
# nearly constant one-hot fields are intentionally excluded.
CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "upload_type",
    "duration_bucket",
    "hour",
    "music_type",
    "video_type",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def get_histories(split_name):
    blocks = []
    names = []
    for entity in ("video_id", "author_id"):
        hist = historical_features(split_name, key=entity)
        for key in sorted(hist):
            a = np.asarray(hist[key], dtype=np.float32)
            if a.ndim != 1:
                continue
            a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
            a = np.clip(a, -1e6, 1e6)
            blocks.append(a)
            names.append(entity + "__" + key)
    return blocks, names


def temporal_context(split):
    dates = np.asarray(split.date, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    end_date = int(dates.max())

    # Calendar distance is valid for this particular April/May interval.
    # Convert YYYYMMDD to a monotonic day index without fitting statistics.
    y = dates // 10000
    m = (dates // 100) % 100
    d = dates % 100
    ordinal = y * 372 + m * 31 + d
    ey, em, ed = end_date // 10000, (end_date // 100) % 100, end_date % 100
    end_ordinal = ey * 372 + em * 31 + ed
    days_to_split_end = (end_ordinal - ordinal).astype(np.float32)

    # Relative time within the observed split is available before outcomes.
    split_start = float(times.min())
    elapsed_days = ((times.astype(np.float64) - split_start) / 86400000.0).astype(
        np.float32
    )
    return [days_to_split_end, elapsed_days]


def make_matrix(split, split_name):
    columns = []
    names = []

    for name in CAT_FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.float32))
        names.append(name)

    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=1e8, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        columns.append(x)
        names.append("log_" + name)

    context = temporal_context(split)
    columns.extend(context)
    names.extend(["days_to_split_end", "elapsed_split_days"])

    histories, hist_names = get_histories(split_name)
    columns.extend(histories)
    names.extend(hist_names)

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32), names


print("FINDINGS building train-only history and context features", flush=True)
Xtr, feature_names = make_matrix(train, "train")
Xva, valid_feature_names = make_matrix(valid, "valid")
Xte, test_feature_names = make_matrix(test, "test")

if feature_names != valid_feature_names or feature_names != test_feature_names:
    raise RuntimeError("Feature definitions differ between splits")

ytr = np.asarray(train.y, dtype=np.int32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)

# Ranking queries must occupy contiguous blocks. Stable ordering preserves the
# logged row order for impressions sharing a timestamp.
train_order = np.argsort(np.asarray(train.user_id, dtype=np.int64), kind="stable")
sorted_users = np.asarray(train.user_id, dtype=np.int64)[train_order]
Xtr_rank = Xtr[train_order]
ytr_rank = ytr[train_order]
_, query_counts = np.unique(sorted_users, return_counts=True)
query_counts = query_counts.astype(np.int32)

cat_indices = list(range(len(CAT_FIELDS)))
dtrain = lgb.Dataset(
    Xtr_rank,
    label=ytr_rank,
    group=query_counts,
    categorical_feature=cat_indices,
    feature_name=feature_names,
    free_raw_data=False,
)

common = {
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 47,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "min_sum_hessian_in_leaf": 1e-3,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.5,
    "max_bin": 127,
    "cat_smooth": 20.0,
    "cat_l2": 12.0,
    "max_cat_threshold": 32,
    "verbosity": -1,
    "verbose": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "num_threads": NTHREAD,
    "force_col_wise": True,
}

models = {}
pred_valid = {}
pred_test = {}

lambda_params = dict(common)
lambda_params.update({
    "objective": "lambdarank",
    "metric": "None",
    "lambdarank_truncation_level": 10,
    "lambdarank_norm": True,
    "label_gain": [0, 1],
})
models["lambdarank"] = lgb.train(
    lambda_params,
    dtrain,
    num_boost_round=280,
)
pred_valid["lambdarank"] = models["lambdarank"].predict(Xva)
pred_test["lambdarank"] = models["lambdarank"].predict(Xte)

xendcg_params = dict(common)
xendcg_params.update({
    "objective": "rank_xendcg",
    "metric": "None",
    "label_gain": [0, 1],
})
models["rank_xendcg"] = lgb.train(
    xendcg_params,
    dtrain,
    num_boost_round=280,
)
pred_valid["rank_xendcg"] = models["rank_xendcg"].predict(Xva)
pred_test["rank_xendcg"] = models["rank_xendcg"].predict(Xte)

# An internal rank-objective ensemble is fixed rather than selected from a
# weight grid. It combines pairwise and listwise errors before incumbent use.
pred_valid["rank_objective_ensemble"] = (
    0.55 * pred_valid["lambdarank"] + 0.45 * pred_valid["rank_xendcg"]
)
pred_test["rank_objective_ensemble"] = (
    0.55 * pred_test["lambdarank"] + 0.45 * pred_test["rank_xendcg"]
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_va_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_te_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_va_path) or not os.path.exists(inc_te_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_va = np.asarray(np.load(inc_va_path), dtype=np.float64)
inc_te = np.asarray(np.load(inc_te_path), dtype=np.float64)


def standardize_pair(valid_scores, test_scores):
    valid_scores = np.asarray(valid_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    mean = float(valid_scores.mean())
    scale = float(valid_scores.std())
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (valid_scores - mean) / scale, (test_scores - mean) / scale


inc_va_z, inc_te_z = standardize_pair(inc_va, inc_te)

candidate_valid = {}
candidate_test = {}
candidate_raw = {}
candidate_metrics = {}

for name in pred_valid:
    own_va = np.asarray(pred_valid[name], dtype=np.float64)
    own_te = np.asarray(pred_test[name], dtype=np.float64)
    candidate_valid[name] = own_va
    candidate_test[name] = own_te
    candidate_raw[name] = name

    met = evaluate(uva, yva, own_va)
    candidate_metrics[name] = float(met["primary"])

    own_va_z, own_te_z = standardize_pair(own_va, own_te)

    # The benchmark explicitly permits selecting an incumbent blend weight on
    # public validation and applying that exact weight to hidden test.
    for own_weight in (0.15, 0.30, 0.45, 0.60):
        cname = "%s_blend_%.2f" % (name, own_weight)
        blend_va = own_weight * own_va_z + (1.0 - own_weight) * inc_va_z
        blend_te = own_weight * own_te_z + (1.0 - own_weight) * inc_te_z
        candidate_valid[cname] = blend_va
        candidate_test[cname] = blend_te
        candidate_raw[cname] = name
        met = evaluate(uva, yva, blend_va)
        candidate_metrics[cname] = float(met["primary"])

winner = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = candidate_valid[winner]
test_scores = candidate_test[winner]
raw_name = candidate_raw[winner]
raw_valid_scores = np.asarray(pred_valid[raw_name], dtype=np.float64)

metrics = evaluate(uva, yva, valid_scores)

print(
    "FINDINGS objectives standalone lambdarank=%.6f rank_xendcg=%.6f ensemble=%.6f winner=%s"
    % (
        candidate_metrics["lambdarank"],
        candidate_metrics["rank_xendcg"],
        candidate_metrics["rank_objective_ensemble"],
        winner,
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(
        {k: round(v, 7) for k, v in candidate_metrics.items()},
        sort_keys=True,
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if winner != raw_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            raw_valid_scores,
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        elapsed,
    )
)