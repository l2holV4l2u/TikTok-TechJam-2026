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
np.random.seed(2026)
torch.manual_seed(2026)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

# ---------------------------------------------------------------------------
# Shared, train-only feature construction
# ---------------------------------------------------------------------------

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "music_type",
    "upload_type",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "onehot_feat1",
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


def recency_weights(split):
    day = (np.asarray(split.date, dtype=np.int32) % 100).astype(np.float32)
    age = float(np.max(day)) - day
    weight = np.exp(-np.log(2.0) * age / 5.0).astype(np.float32)
    return weight / np.maximum(np.mean(weight), 1e-6)


train_weight = recency_weights(train)


def transformed_numeric(split, centers=None, scales=None):
    columns = []
    out_centers = []
    out_scales = []

    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        finite = np.isfinite(x)
        clean = np.where(finite, np.maximum(x, 0.0), 0.0)
        z = np.log1p(clean).astype(np.float32)

        if centers is None:
            center = float(np.median(z[finite])) if np.any(finite) else 0.0
            q25, q75 = (
                np.percentile(z[finite], [25.0, 75.0])
                if np.any(finite)
                else (0.0, 1.0)
            )
            scale = max(float(q75 - q25), 0.25)
        else:
            center = float(centers[len(columns) // 2])
            scale = float(scales[len(columns) // 2])

        z = np.clip((z - center) / scale, -8.0, 8.0)
        columns.append(z.astype(np.float32))
        columns.append((~finite).astype(np.float32))
        out_centers.append(center)
        out_scales.append(scale)

    return (
        np.column_stack(columns).astype(np.float32),
        out_centers,
        out_scales,
    )


tr_num, num_centers, num_scales = transformed_numeric(train)
va_num, _, _ = transformed_numeric(valid, num_centers, num_scales)
te_num, _, _ = transformed_numeric(test, num_centers, num_scales)


def get_history(split_name):
    video = historical_features(split_name, key="video_id")
    author = historical_features(split_name, key="author_id")
    names = sorted(video.keys()) + sorted(author.keys())
    arrays = [np.asarray(video[k], dtype=np.float32) for k in sorted(video.keys())]
    arrays += [
        np.asarray(author[k], dtype=np.float32)
        for k in sorted(author.keys())
    ]
    matrix = np.column_stack(arrays).astype(np.float32)
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix, names


tr_hist, hist_names = get_history("train")
va_hist, _ = get_history("valid")
te_hist, _ = get_history("test")

# Random forests receive raw categorical identifiers, robust log-scaled
# numeric features, and train-only entity histories.
tr_cat = np.column_stack(
    [np.asarray(train.X[name], dtype=np.float32) for name in CAT_FIELDS]
).astype(np.float32)
va_cat = np.column_stack(
    [np.asarray(valid.X[name], dtype=np.float32) for name in CAT_FIELDS]
).astype(np.float32)
te_cat = np.column_stack(
    [np.asarray(test.X[name], dtype=np.float32) for name in CAT_FIELDS]
).astype(np.float32)

X_train_rf = np.column_stack([tr_cat, tr_num, tr_hist]).astype(np.float32)
X_valid_rf = np.column_stack([va_cat, va_num, va_hist]).astype(np.float32)
X_test_rf = np.column_stack([te_cat, te_num, te_hist]).astype(np.float32)

categorical_indices = list(range(len(CAT_FIELDS)))

rf_dataset = lgb.Dataset(
    X_train_rf,
    label=y_train,
    weight=train_weight,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

rf_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "rf",
    "num_leaves": 127,
    "max_depth": 12,
    "min_data_in_leaf": 120,
    "learning_rate": 1.0,
    "bagging_fraction": 0.70,
    "bagging_freq": 1,
    "feature_fraction": 0.72,
    "feature_fraction_bynode": 0.72,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": 2026,
    "bagging_seed": 2027,
    "feature_fraction_seed": 2028,
    "verbose": -1,
}

rf_model = lgb.train(
    rf_params,
    rf_dataset,
    num_boost_round=150,
)

rf_valid = rf_model.predict(X_valid_rf).astype(np.float32)
rf_test = rf_model.predict(X_test_rf).astype(np.float32)

del rf_dataset, rf_model
del X_train_rf, X_valid_rf, X_test_rf, tr_cat, va_cat, te_cat
gc.collect()

# ---------------------------------------------------------------------------
# Setwise conditional-logit wide ranker
#
# Unlike pointwise CTR fitting, each mixed user's loss is
#   logsumexp(scores for all impressions) - mean(score of positives).
# Thus user-specific response propensity cancels, and every gradient concerns
# relative utilities within a logged impression set.
# ---------------------------------------------------------------------------

SETWISE_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "music_type",
    "upload_type",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

HASH_SIZE = 262144
CROSS_PAIRS = [
    ("author_id", "tag"),
    ("video_id", "tab"),
    ("tag", "duration_bucket"),
    ("tab", "onehot_feat3"),
]


def make_setwise_inputs(split, numeric):
    cat = np.column_stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in SETWISE_FIELDS]
    )
    crosses = []
    for left, right in CROSS_PAIRS:
        a = np.asarray(split.X[left], dtype=np.uint64)
        b = np.asarray(split.X[right], dtype=np.uint64)
        # Deterministic multiplicative hashing with two distinct odd constants.
        h = (
            a * np.uint64(11995408973635179863)
            + b * np.uint64(10150724397891781847)
        ) % np.uint64(HASH_SIZE)
        crosses.append(h.astype(np.int64))
    cross = np.column_stack(crosses).astype(np.int64)
    return cat.astype(np.int64), cross, numeric.astype(np.float32)


sw_tr_cat, sw_tr_cross, sw_tr_num = make_setwise_inputs(train, tr_num)
sw_va_cat, sw_va_cross, sw_va_num = make_setwise_inputs(valid, va_num)
sw_te_cat, sw_te_cross, sw_te_num = make_setwise_inputs(test, te_num)


class SetwiseWideRanker(nn.Module):
    def __init__(self, n_numeric):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[name]), 1)
            for name in SETWISE_FIELDS
        ])
        self.cross_embeddings = nn.ModuleList([
            nn.Embedding(HASH_SIZE, 1)
            for _ in CROSS_PAIRS
        ])
        self.numeric = nn.Linear(n_numeric, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        for embedding in self.cat_embeddings:
            nn.init.zeros_(embedding.weight)
        for embedding in self.cross_embeddings:
            nn.init.zeros_(embedding.weight)
        nn.init.zeros_(self.numeric.weight)

    def forward(self, cat, cross, numeric):
        score = self.numeric(numeric).squeeze(1) + self.bias
        for j, embedding in enumerate(self.cat_embeddings):
            score = score + embedding(cat[:, j]).squeeze(1)
        for j, embedding in enumerate(self.cross_embeddings):
            score = score + embedding(cross[:, j]).squeeze(1)
        return score


device = torch.device("cpu")
ranker = SetwiseWideRanker(sw_tr_num.shape[1]).to(device)
optimizer = torch.optim.AdamW(
    ranker.parameters(),
    lr=0.025,
    weight_decay=2e-5,
)

tr_users = np.asarray(train.user_id, dtype=np.int64)
order = np.argsort(tr_users, kind="stable")
ordered_users = tr_users[order]

starts = np.r_[0, 1 + np.flatnonzero(ordered_users[1:] != ordered_users[:-1])]
ends = np.r_[starts[1:], len(order)]

users_per_batch = 512
ranker.train()

for epoch in range(6):
    # Alternating traversal direction changes optimization order without
    # changing the user-complete sets used by the loss.
    batch_starts = list(range(0, len(starts), users_per_batch))
    if epoch % 2 == 1:
        batch_starts = batch_starts[::-1]

    epoch_loss = 0.0
    epoch_batches = 0

    for user_begin in batch_starts:
        user_end = min(user_begin + users_per_batch, len(starts))
        row_begin = int(starts[user_begin])
        row_end = int(ends[user_end - 1])
        idx = order[row_begin:row_end]

        local_users_np = ordered_users[row_begin:row_end]
        boundary = np.r_[
            True,
            local_users_np[1:] != local_users_np[:-1]
        ]
        local_group_np = np.cumsum(boundary, dtype=np.int64) - 1
        n_groups = int(local_group_np[-1]) + 1

        cat_t = torch.from_numpy(sw_tr_cat[idx]).long()
        cross_t = torch.from_numpy(sw_tr_cross[idx]).long()
        num_t = torch.from_numpy(sw_tr_num[idx]).float()
        y_t = torch.from_numpy(y_train[idx]).float()
        group_t = torch.from_numpy(local_group_np).long()
        weight_t = torch.from_numpy(train_weight[idx]).float()

        score = ranker(cat_t, cross_t, num_t)

        group_max = torch.full((n_groups,), -1e30, dtype=score.dtype)
        group_max.scatter_reduce_(
            0, group_t, score.detach(), reduce="amax", include_self=True
        )

        exp_score = torch.exp(
            torch.clamp(score - group_max[group_t], min=-30.0, max=30.0)
        )
        denominator = torch.zeros(n_groups, dtype=score.dtype)
        denominator.scatter_add_(0, group_t, exp_score * weight_t)

        positive_weight = y_t * weight_t
        positive_sum = torch.zeros(n_groups, dtype=score.dtype)
        positive_sum.scatter_add_(0, group_t, positive_weight)

        positive_score = torch.zeros(n_groups, dtype=score.dtype)
        positive_score.scatter_add_(0, group_t, positive_weight * score)

        total_weight = torch.zeros(n_groups, dtype=score.dtype)
        total_weight.scatter_add_(0, group_t, weight_t)

        mixed = (
            (positive_sum > 1e-5)
            & (positive_sum < total_weight - 1e-5)
        )
        if not bool(torch.any(mixed)):
            continue

        log_partition = (
            group_max
            + torch.log(torch.clamp(denominator, min=1e-8))
        )
        group_loss = (
            log_partition
            - positive_score / torch.clamp(positive_sum, min=1e-6)
        )

        # Positive-count square-root weighting is a compromise between GAUC's
        # positive weighting and nDCG's equal-user weighting.
        group_importance = torch.sqrt(torch.clamp(positive_sum, min=1.0))
        loss = (
            group_loss[mixed] * group_importance[mixed]
        ).sum() / group_importance[mixed].sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ranker.parameters(), 5.0)
        optimizer.step()

        epoch_loss += float(loss.detach())
        epoch_batches += 1

    print(
        "FINDINGS setwise_epoch=%d mean_loss=%.6f"
        % (epoch + 1, epoch_loss / max(epoch_batches, 1))
    )


def predict_setwise(cat, cross, numeric, batch_size=131072):
    ranker.eval()
    result = np.empty(len(cat), dtype=np.float32)
    with torch.no_grad():
        for begin in range(0, len(cat), batch_size):
            end = min(begin + batch_size, len(cat))
            result[begin:end] = ranker(
                torch.from_numpy(cat[begin:end]).long(),
                torch.from_numpy(cross[begin:end]).long(),
                torch.from_numpy(numeric[begin:end]).float(),
            ).cpu().numpy().astype(np.float32)
    return result


setwise_valid = predict_setwise(sw_va_cat, sw_va_cross, sw_va_num)
setwise_test = predict_setwise(sw_te_cat, sw_te_cross, sw_te_num)

# ---------------------------------------------------------------------------
# Validation comparison and incumbent blending
# ---------------------------------------------------------------------------

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float32,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float32,
)


def standardize_pair(valid_score, test_score):
    center = float(np.mean(valid_score))
    scale = max(float(np.std(valid_score)), 1e-6)
    return (
        ((valid_score - center) / scale).astype(np.float32),
        ((test_score - center) / scale).astype(np.float32),
    )


inc_valid_z, inc_test_z = standardize_pair(inc_valid, inc_test)

families = {
    "bagged_random_forest": (rf_valid, rf_test),
    "setwise_conditional_logit": (setwise_valid, setwise_test),
}

candidate_log = {}
inc_metric = evaluate(valid.user_id, y_valid, inc_valid)
candidate_log["incumbent"] = float(inc_metric["primary"])

best_metric = inc_metric
best_valid = inc_valid.copy()
best_test = inc_test.copy()
best_raw = setwise_valid.copy()
best_name = "incumbent"

blend_alphas = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70]

for family_name, (raw_valid, raw_test) in families.items():
    raw_metric = evaluate(valid.user_id, y_valid, raw_valid)
    candidate_log[family_name] = float(raw_metric["primary"])

    raw_valid_z, raw_test_z = standardize_pair(raw_valid, raw_test)

    for alpha in blend_alphas:
        blend_valid = (
            (1.0 - alpha) * inc_valid_z + alpha * raw_valid_z
        ).astype(np.float32)
        blend_test = (
            (1.0 - alpha) * inc_test_z + alpha * raw_test_z
        ).astype(np.float32)

        metric = evaluate(valid.user_id, y_valid, blend_valid)
        key = "%s_blend_%.2f" % (family_name, alpha)
        candidate_log[key] = float(metric["primary"])

        if float(metric["primary"]) > float(best_metric["primary"]):
            best_metric = metric
            best_valid = blend_valid
            best_test = blend_test
            best_raw = raw_valid
            best_name = key

print("FINDINGS winner=%s" % best_name)
print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
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
        np.asarray(best_raw, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metric["primary"]),
        "gauc": float(best_metric["gauc"]),
        "ndcg@5": float(best_metric["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)