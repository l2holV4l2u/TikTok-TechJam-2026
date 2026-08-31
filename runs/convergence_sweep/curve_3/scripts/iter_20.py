import os
import time
import json
import gc

import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260831
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def within_user_rank(scores, users):
    scores = np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=0.0,
        posinf=1e20,
        neginf=-1e20,
    )
    users = np.asarray(users, dtype=np.int64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    su = users[order]
    ss = scores[order]
    positions = np.arange(n, dtype=np.int64)

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = su[1:] != su[:-1]
    starts = np.maximum.accumulate(np.where(user_start, positions, 0))

    user_end = np.empty(n, dtype=bool)
    user_end[-1] = True
    user_end[:-1] = su[:-1] != su[1:]
    ends = np.minimum.accumulate(
        np.where(user_end, positions, n - 1)[::-1]
    )[::-1]

    tie_start = np.empty(n, dtype=bool)
    tie_start[0] = True
    tie_start[1:] = (su[1:] != su[:-1]) | (ss[1:] != ss[:-1])
    tie_starts = np.maximum.accumulate(np.where(tie_start, positions, 0))

    tie_end = np.empty(n, dtype=bool)
    tie_end[-1] = True
    tie_end[:-1] = (su[:-1] != su[1:]) | (ss[:-1] != ss[1:])
    tie_ends = np.minimum.accumulate(
        np.where(tie_end, positions, n - 1)[::-1]
    )[::-1]

    local_rank = 0.5 * (tie_starts + tie_ends) - starts
    denominator = np.maximum(ends - starts, 1)
    ranked = local_rank / denominator

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

train_dates = np.asarray(train.date, dtype=np.int32)
max_train_date = int(np.max(train_dates))
age = max_train_date - train_dates
sample_weights = np.power(0.5, age.astype(np.float64) / 5.0).astype(np.float32)
sample_weights /= np.mean(sample_weights)

global_rate = float(
    np.sum(sample_weights * y_train) / np.sum(sample_weights)
)
global_logit = float(safe_logit(global_rate))

TE_FIELDS = [
    "video_id",
    "author_id",
    "user_id",
    "tab",
    "tag",
    "onehot_feat3",
    "upload_type",
    "onehot_feat8",
    "duration_bucket",
    "onehot_feat1",
    "music_type",
    "onehot_feat7",
    "user_active_degree",
    "register_days_bucket",
    "register_days_range",
    "onehot_feat0",
    "onehot_feat12",
    "fans_user_num_range",
    "onehot_feat11",
    "onehot_feat6",
    "hour",
]

TE_STRENGTH = {
    "video_id": 35.0,
    "author_id": 40.0,
    "user_id": 40.0,
    "tab": 140.0,
    "tag": 110.0,
    "onehot_feat3": 70.0,
    "upload_type": 140.0,
    "onehot_feat8": 80.0,
}
DEFAULT_STRENGTH = 170.0

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def field_values(split, field):
    if field == "video_id":
        return np.asarray(split.video_id, dtype=np.int64)
    if field == "user_id":
        return np.asarray(split.user_id, dtype=np.int64)
    return np.asarray(split.X[field], dtype=np.int64)


target_tables = {}
for field in TE_FIELDS:
    values = field_values(train, field)
    cardinality = int(FEATURE_CARDINALITIES[field])
    counts = np.bincount(
        values,
        weights=sample_weights,
        minlength=cardinality,
    ).astype(np.float64)
    positives = np.bincount(
        values,
        weights=sample_weights * y_train,
        minlength=cardinality,
    ).astype(np.float64)
    target_tables[field] = (counts, positives)


def target_encode(split, field, leave_one_out):
    values = field_values(split, field)
    counts, positives = target_tables[field]
    safe_values = np.clip(values, 0, len(counts) - 1)

    count = counts[safe_values].copy()
    positive = positives[safe_values].copy()

    if leave_one_out:
        count -= sample_weights
        positive -= sample_weights * y_train

    strength = TE_STRENGTH.get(field, DEFAULT_STRENGTH)
    rate = (
        positive + strength * global_rate
    ) / np.maximum(count + strength, 1e-8)
    encoded = safe_logit(rate) - global_logit
    reliability = count / np.maximum(count + strength, 1e-8)

    return encoded.astype(np.float32), reliability.astype(np.float32)


def get_histories(split_name):
    result = {}
    result.update(historical_features(split_name, key="video_id"))
    result.update(historical_features(split_name, key="author_id"))
    return result


hist_train = get_histories("train")
hist_valid = get_histories("valid")
hist_test = get_histories("test")

history_keys = sorted(
    set(hist_train.keys())
    & set(hist_valid.keys())
    & set(hist_test.keys())
)


def build_absolute_matrix(split, histories, leave_one_out):
    columns = []

    for field in TE_FIELDS:
        encoded, reliability = target_encode(split, field, leave_one_out)
        columns.append(encoded)
        if field in ("video_id", "author_id", "user_id"):
            columns.append(reliability)

    for field in NUM_FIELDS:
        values = np.asarray(split.num[field], dtype=np.float32)
        missing = ~np.isfinite(values)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(values, 0.0)).astype(np.float32))
        if np.any(missing):
            columns.append(missing.astype(np.float32))

    for key in history_keys:
        values = np.asarray(histories[key], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(values)

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    columns.append(np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32))
    columns.append(np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False)


A_train = build_absolute_matrix(train, hist_train, True)
A_valid = build_absolute_matrix(valid, hist_valid, False)
A_test = build_absolute_matrix(test, hist_test, False)

del hist_train, hist_valid, hist_test
gc.collect()


def append_slate_relative_features(matrix, users):
    users = np.asarray(users, dtype=np.int64)
    unique_users, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    row_counts = counts[inverse]

    n, d = matrix.shape
    relative = np.empty((n, d), dtype=np.float32)

    for j in range(d):
        sums = np.bincount(
            inverse,
            weights=matrix[:, j].astype(np.float64),
            minlength=len(unique_users),
        )
        means = sums / np.maximum(counts, 1.0)
        relative[:, j] = matrix[:, j] - means[inverse].astype(np.float32)

    count_feature = np.log1p(row_counts).astype(np.float32)[:, None]
    return np.concatenate([matrix, relative, count_feature], axis=1)


X_train = append_slate_relative_features(A_train, train_users)
X_valid = append_slate_relative_features(A_valid, valid_users)
X_test = append_slate_relative_features(A_test, test_users)

del A_train, A_valid, A_test
gc.collect()

feature_mean = np.average(
    X_train.astype(np.float64),
    axis=0,
    weights=sample_weights,
)
feature_var = np.average(
    (X_train.astype(np.float64) - feature_mean) ** 2,
    axis=0,
    weights=sample_weights,
)
feature_scale = np.sqrt(np.maximum(feature_var, 1e-5))

Z_train = np.clip(
    (X_train - feature_mean) / feature_scale,
    -8.0,
    8.0,
).astype(np.float32)
Z_valid = np.clip(
    (X_valid - feature_mean) / feature_scale,
    -8.0,
    8.0,
).astype(np.float32)
Z_test = np.clip(
    (X_test - feature_mean) / feature_scale,
    -8.0,
    8.0,
).astype(np.float32)

del X_train, X_valid, X_test
gc.collect()

candidate_valid = {}
candidate_test = {}

# Family 1: histogram gradient-boosted decision trees.
dtrain_lgb = lgb.Dataset(
    Z_train,
    label=y_train,
    weight=sample_weights,
    free_raw_data=False,
)

gbdt_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.045,
    "num_leaves": 48,
    "max_depth": -1,
    "min_data_in_leaf": 900,
    "feature_fraction": 0.78,
    "bagging_fraction": 0.78,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "verbosity": -1,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
}

gbdt_model = lgb.train(
    gbdt_params,
    dtrain_lgb,
    num_boost_round=260,
)
candidate_valid["slate_relative_gbdt"] = gbdt_model.predict(
    Z_valid
).astype(np.float32)
candidate_test["slate_relative_gbdt"] = gbdt_model.predict(
    Z_test
).astype(np.float32)

# Family 2: randomized tree forest, averaging rather than sequential residual fitting.
rf_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "rf",
    "learning_rate": 1.0,
    "num_leaves": 64,
    "max_depth": 12,
    "min_data_in_leaf": 700,
    "feature_fraction": 0.62,
    "bagging_fraction": 0.68,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 8.0,
    "max_bin": 95,
    "verbosity": -1,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": SEED + 11,
    "feature_fraction_seed": SEED + 12,
    "bagging_seed": SEED + 13,
}

rf_model = lgb.train(
    rf_params,
    dtrain_lgb,
    num_boost_round=180,
)
candidate_valid["slate_relative_random_forest"] = rf_model.predict(
    Z_valid
).astype(np.float32)
candidate_test["slate_relative_random_forest"] = rf_model.predict(
    Z_test
).astype(np.float32)

del rf_model
gc.collect()

# Family 3: conditional pairwise ridge. User-specific intercepts cancel in
# feature differences, directly targeting within-user positive/negative order.
order = np.argsort(train_users, kind="stable")
sorted_users = train_users[order]
sorted_labels = y_train[order]
sorted_weights = sample_weights[order]
sorted_z = Z_train[order]

dimension = Z_train.shape[1]
xtx = np.eye(dimension, dtype=np.float64) * 35.0
xty = np.zeros(dimension, dtype=np.float64)

for shift in (1, 2, 4, 8, 16, 32):
    left = np.arange(shift, len(order), dtype=np.int64)
    right = left - shift
    usable = (
        (sorted_users[left] == sorted_users[right])
        & (sorted_labels[left] != sorted_labels[right])
    )
    left = left[usable]
    right = right[usable]

    max_pairs = 130000
    if len(left) > max_pairs:
        selected = rng.choice(len(left), size=max_pairs, replace=False)
        left = left[selected]
        right = right[selected]

    for start in range(0, len(left), 25000):
        li = left[start:start + 25000]
        ri = right[start:start + 25000]
        difference = (
            sorted_z[li].astype(np.float64)
            - sorted_z[ri].astype(np.float64)
        )
        target = (
            sorted_labels[li].astype(np.float64)
            - sorted_labels[ri].astype(np.float64)
        )
        pair_weight = np.sqrt(
            sorted_weights[li].astype(np.float64)
            * sorted_weights[ri].astype(np.float64)
        )
        weighted_difference = difference * pair_weight[:, None]
        xtx += difference.T @ weighted_difference
        xty += difference.T @ (pair_weight * target)

pair_beta = np.linalg.solve(xtx, xty)
candidate_valid["conditional_pairwise_ridge"] = (
    Z_valid.astype(np.float64) @ pair_beta
).astype(np.float32)
candidate_test["conditional_pairwise_ridge"] = (
    Z_test.astype(np.float64) @ pair_beta
).astype(np.float32)

del sorted_z, xtx, xty
gc.collect()

# Family 4: residual deep pointwise classifier over the same absolute and
# slate-relative inputs.
class ResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, 128)
        self.block1 = nn.Sequential(
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 128),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
        )
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        h = self.input(x)
        h = h + self.block1(h)
        h = h + 0.5 * self.block2(h)
        return self.output(torch.nn.functional.silu(h)).squeeze(1)


network = ResidualMLP(dimension)
optimizer = torch.optim.AdamW(
    network.parameters(),
    lr=1.8e-3,
    weight_decay=2e-4,
)

batch_size = 8192
network.train()
for epoch in range(2):
    permutation = rng.permutation(len(Z_train))
    for start in range(0, len(permutation), batch_size):
        indices = permutation[start:start + batch_size]
        xb = torch.from_numpy(Z_train[indices])
        yb = torch.from_numpy(y_train[indices])
        wb = torch.from_numpy(sample_weights[indices])

        optimizer.zero_grad(set_to_none=True)
        logits = network(xb)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            yb,
            reduction="none",
        )
        loss = torch.sum(losses * wb) / torch.sum(wb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()


def neural_predict(matrix):
    outputs = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(matrix), 16384):
            xb = torch.from_numpy(matrix[start:start + 16384])
            outputs.append(network(xb).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs)


candidate_valid["residual_mlp"] = neural_predict(Z_valid)
candidate_test["residual_mlp"] = neural_predict(Z_test)

del network, optimizer, Z_train, dtrain_lgb, gbdt_model
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

has_incumbent = (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
)

candidate_scores = {}
best_name = None
best_primary = -np.inf
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_is_external_blend = False

# Score all standalone families.
for name in candidate_valid:
    metric = evaluate(valid_users, y_valid, candidate_valid[name])
    primary = float(metric["primary"])
    candidate_scores[name] = primary
    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_valid_scores = candidate_valid[name].astype(np.float64)
        best_test_scores = candidate_test[name].astype(np.float64)
        best_raw_valid = candidate_valid[name].astype(np.float64)
        best_is_external_blend = False

# Compare rank-space blends, avoiding arbitrary score-scale differences.
if has_incumbent:
    incumbent_valid = np.load(inc_valid_path).astype(np.float64)
    incumbent_test = np.load(inc_test_path).astype(np.float64)
    incumbent_valid_rank = within_user_rank(incumbent_valid, valid_users)
    incumbent_test_rank = within_user_rank(incumbent_test, test_users)

    incumbent_metric = evaluate(valid_users, y_valid, incumbent_valid)
    incumbent_primary = float(incumbent_metric["primary"])
    candidate_scores["trusted_incumbent"] = incumbent_primary

    if incumbent_primary > best_primary:
        best_primary = incumbent_primary
        best_name = "trusted_incumbent"
        best_valid_scores = incumbent_valid
        best_test_scores = incumbent_test
        # Preserve the strongest standalone new family as the raw contribution.
        standalone_name = max(
            candidate_valid,
            key=lambda n: candidate_scores[n],
        )
        best_raw_valid = candidate_valid[standalone_name].astype(np.float64)
        best_is_external_blend = True

    blend_alphas = [
        0.04, 0.08, 0.12, 0.16, 0.22, 0.30, 0.40, 0.50, 0.65
    ]

    for name in candidate_valid:
        new_valid_rank = within_user_rank(
            candidate_valid[name],
            valid_users,
        )
        new_test_rank = within_user_rank(
            candidate_test[name],
            test_users,
        )

        family_best = -np.inf
        for alpha in blend_alphas:
            blended_valid = (
                (1.0 - alpha) * incumbent_valid_rank
                + alpha * new_valid_rank
            )
            metric = evaluate(valid_users, y_valid, blended_valid)
            primary = float(metric["primary"])
            family_best = max(family_best, primary)

            if primary > best_primary:
                best_primary = primary
                best_name = "%s_blend_%.2f" % (name, alpha)
                best_valid_scores = blended_valid.astype(np.float64)
                best_test_scores = (
                    (1.0 - alpha) * incumbent_test_rank
                    + alpha * new_test_rank
                ).astype(np.float64)
                best_raw_valid = candidate_valid[name].astype(np.float64)
                best_is_external_blend = True

        candidate_scores[name + "_best_incumbent_blend"] = family_best

final_metrics = evaluate(valid_users, y_valid, best_valid_scores)

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
    if best_is_external_blend:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "n_features": int(dimension),
            "history_features": int(len(history_keys)),
            "used_incumbent": bool(has_incumbent),
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(time.time() - START),
        }
    )
)