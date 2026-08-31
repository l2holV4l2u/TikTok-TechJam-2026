import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 918273
HALF_LIFE = 4.0
THREADS = min(16, os.cpu_count() or 1)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "hour",
    "upload_type",
    "video_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
    "music_type",
    "register_days_bucket",
    "fans_user_num_range",
]

NUM_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def recency_weights(dates, half_life=HALF_LIFE):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.sort(np.unique(dates))
    mapping = {int(d): i for i, d in enumerate(unique_dates)}
    day_index = np.asarray([mapping[int(d)] for d in dates], dtype=np.float32)
    age = float(len(unique_dates) - 1) - day_index
    weights = np.exp(-np.log(2.0) * age / half_life).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def safe_log_numeric(values):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    clean = np.sign(clean) * np.log1p(np.abs(clean))
    return clean.astype(np.float32), (~finite).astype(np.float32)


def get_histories(split_names):
    all_histories = {}
    for split_name in split_names:
        entity_parts = {}
        for entity in ("video_id", "author_id"):
            hist = historical_features(split_name, key=entity)
            usable = {}
            for key, value in hist.items():
                arr = np.asarray(value)
                if arr.ndim == 1:
                    usable[str(key)] = arr.astype(np.float32, copy=False)
            entity_parts[entity] = usable
        all_histories[split_name] = entity_parts

    common = {}
    for entity in ("video_id", "author_id"):
        key_sets = [
            set(all_histories[name][entity].keys()) for name in split_names
        ]
        keys = sorted(set.intersection(*key_sets)) if key_sets else []
        common[entity] = keys
    return all_histories, common


def build_tree_matrix(split, split_name, histories, history_keys):
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        transformed, missing = safe_log_numeric(split.num[field])
        columns.append(transformed)
        columns.append(missing)

    for entity in ("video_id", "author_id"):
        for key in history_keys[entity]:
            arr = histories[split_name][entity][key]
            arr = np.asarray(arr, dtype=np.float32)
            finite = np.isfinite(arr)
            clean = np.where(finite, arr, 0.0).astype(np.float32)
            columns.append(clean)
            columns.append((~finite).astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    local = row - np.repeat(starts, lengths)
    denominator = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked_sorted = local.astype(np.float64) / denominator.astype(np.float64)

    singleton = np.repeat(lengths, lengths) == 1
    ranked_sorted[singleton] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def fit_binary_tree(X, y, weights, categorical_indices):
    dataset = lgb.Dataset(
        X,
        label=y,
        weight=weights,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_cat_threshold": 64,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "max_bin": 127,
        "num_threads": THREADS,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "data_random_seed": SEED + 3,
        "force_col_wise": True,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=240)


def fit_rank_tree(X, y, users, weights, categorical_indices):
    row = np.arange(users.size, dtype=np.int64)
    order = np.lexsort((row, users))
    sorted_users = users[order]

    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_users.size]
    groups = (ends - starts).astype(np.int32)

    dataset = lgb.Dataset(
        X[order],
        label=y[order],
        weight=weights[order],
        group=groups,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "label_gain": [0, 1],
        "lambdarank_truncation_level": 10,
        "lambdarank_norm": True,
        "learning_rate": 0.045,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 400,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_cat_threshold": 64,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "max_bin": 127,
        "num_threads": THREADS,
        "seed": SEED + 17,
        "feature_fraction_seed": SEED + 18,
        "bagging_seed": SEED + 19,
        "data_random_seed": SEED + 20,
        "force_col_wise": True,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=210)


HASH_SIZE = 1000003
WIDE_BATCH = 32768
WIDE_EPOCHS = 3


class WideCrossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_offsets = {}
        offset = 0
        for field in CAT_FIELDS:
            self.base_offsets[field] = offset
            offset += int(FEATURE_CARDINALITIES[field])

        self.cross_offsets = []
        for _ in range(5):
            self.cross_offsets.append(offset)
            offset += HASH_SIZE

        self.embedding = nn.Embedding(offset, 1)
        self.numeric_weight = nn.Parameter(torch.zeros(len(NUM_FIELDS)))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.embedding.weight)

    def forward(self, indices, numeric):
        score = self.embedding(indices).squeeze(2).sum(dim=1)
        score = score + (numeric * self.numeric_weight).sum(dim=1) + self.bias
        return score


def make_wide_arrays(split):
    base = [
        np.asarray(split.X[field], dtype=np.int64)
        for field in CAT_FIELDS
    ]

    numeric = []
    for field in NUM_FIELDS:
        transformed, _ = safe_log_numeric(split.num[field])
        mean = float(np.mean(transformed))
        std = float(np.std(transformed))
        if std < 1e-5:
            std = 1.0
        numeric.append(((transformed - mean) / std).astype(np.float32))
    numeric = np.column_stack(numeric).astype(np.float32)

    return base, numeric


def wide_indices(model, base, rows):
    pieces = []
    for i, field in enumerate(CAT_FIELDS):
        pieces.append(base[i][rows] + model.base_offsets[field])

    user = base[CAT_FIELDS.index("user_id")][rows]
    video = base[CAT_FIELDS.index("video_id")][rows]
    author = base[CAT_FIELDS.index("author_id")][rows]
    tag = base[CAT_FIELDS.index("tag")][rows]
    tab = base[CAT_FIELDS.index("tab")][rows]
    duration = base[CAT_FIELDS.index("duration_bucket")][rows]
    hour = base[CAT_FIELDS.index("hour")][rows]

    hashes = [
        (user * 1000003 + tag * 9176 + 13) % HASH_SIZE,
        (user * 1000033 + tab * 9283 + 29) % HASH_SIZE,
        (user * 1000037 + duration * 9323 + 43) % HASH_SIZE,
        (user * 1000039 + hour * 9341 + 71) % HASH_SIZE,
        (author * 1000081 + tag * 9377 + video * 17 + 89) % HASH_SIZE,
    ]
    for offset, hashed in zip(model.cross_offsets, hashes):
        pieces.append(hashed + offset)

    return np.column_stack(pieces).astype(np.int64)


def fit_wide(split, sample_weights):
    base, numeric = make_wide_arrays(split)
    labels = np.asarray(split.y, dtype=np.float32)
    n = labels.size

    model = WideCrossModel()
    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=0.08, weight_decay=1e-7
    )
    rng = np.random.default_rng(SEED + 101)

    for _ in range(WIDE_EPOCHS):
        order = rng.permutation(n)
        model.train()
        for start in range(0, n, WIDE_BATCH):
            rows = order[start:start + WIDE_BATCH]
            indices = torch.from_numpy(wide_indices(model, base, rows))
            nums = torch.from_numpy(numeric[rows])
            target = torch.from_numpy(labels[rows])
            weight = torch.from_numpy(sample_weights[rows])

            optimizer.zero_grad(set_to_none=True)
            logits = model(indices, nums)
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                ) * weight
            ).mean()
            loss.backward()
            optimizer.step()

    return model


def predict_wide(model, split):
    base, numeric = make_wide_arrays(split)
    n = numeric.shape[0]
    scores = np.empty(n, dtype=np.float64)
    model.eval()

    with torch.no_grad():
        for start in range(0, n, 65536):
            rows = np.arange(start, min(start + 65536, n), dtype=np.int64)
            indices = torch.from_numpy(wide_indices(model, base, rows))
            nums = torch.from_numpy(numeric[rows])
            scores[rows] = model(indices, nums).cpu().numpy().astype(np.float64)
    return scores


train = load("train")
valid = load("valid")

histories, history_keys = get_histories(["train", "valid"])
X_train = build_tree_matrix(train, "train", histories, history_keys)
X_valid = build_tree_matrix(valid, "valid", histories, history_keys)

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
weights = recency_weights(train.date)
categorical_indices = list(range(len(CAT_FIELDS)))

binary_model = fit_binary_tree(
    X_train, y_train, weights, categorical_indices
)
binary_valid = binary_model.predict(X_valid).astype(np.float64)

rank_model = fit_rank_tree(
    X_train, y_train, train_users, weights, categorical_indices
)
rank_valid = rank_model.predict(X_valid).astype(np.float64)

wide_model = fit_wide(train, weights)
wide_valid = predict_wide(wide_model, valid)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

raw_candidates = {
    "binary_gbdt": binary_valid,
    "lambdarank_gbdt": rank_valid,
    "wide_cross": wide_valid,
}
models = {
    "binary_gbdt": binary_model,
    "lambdarank_gbdt": rank_model,
    "wide_cross": wide_model,
}

candidate_metrics = {}
best_name = None
best_alpha = None
best_score = -np.inf
best_valid = None

blend_alphas = [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]

for name, raw_scores in raw_candidates.items():
    own_rank = within_user_rank(valid_users, raw_scores)

    standalone = evaluate(valid_users, y_valid, raw_scores)
    candidate_metrics[name] = float(standalone["primary"])

    for alpha in blend_alphas:
        blended = (1.0 - alpha) * inc_valid_rank + alpha * own_rank
        result = evaluate(valid_users, y_valid, blended)
        label = "{}_blend_{:.2f}".format(name, alpha)
        candidate_metrics[label] = float(result["primary"])

        if float(result["primary"]) > best_score:
            best_score = float(result["primary"])
            best_name = name
            best_alpha = float(alpha)
            best_valid = blended.copy()

final_metrics = evaluate(valid_users, y_valid, best_valid)

print(
    "FINDINGS selected_family={} blend_alpha={:.2f} history_features={}".format(
        best_name,
        best_alpha,
        sum(len(history_keys[e]) for e in history_keys),
    )
)
print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(raw_candidates[best_name], dtype=np.float64),
    )

del X_train, X_valid, histories
te = load("test")
test_histories, test_history_keys = get_histories(["test"])

# Retain precisely the train-established history column schema.
for entity in ("video_id", "author_id"):
    missing = set(history_keys[entity]) - set(test_history_keys[entity])
    if missing:
        raise KeyError("Test history schema missing keys: {}".format(sorted(missing)))

if best_name in ("binary_gbdt", "lambdarank_gbdt"):
    X_test = build_tree_matrix(te, "test", test_histories, history_keys)
    own_test = models[best_name].predict(X_test).astype(np.float64)
else:
    own_test = predict_wide(models[best_name], te)

test_users = np.asarray(te.user_id, dtype=np.int64)
inc_test = np.load(inc_test_path).astype(np.float64)
inc_test_rank = within_user_rank(test_users, inc_test)
own_test_rank = within_user_rank(test_users, own_test)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * own_test_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
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