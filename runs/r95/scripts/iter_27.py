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
THREADS = min(8, os.cpu_count() or 8)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)

ntr = len(ytr)
nva = len(yva)
nte = len(test.user_id)

# Temporal weights are fixed from train only. A four-day half-life emphasizes
# the regime nearest the date boundary without discarding all earlier users.
train_dates = np.asarray(train.date, dtype=np.int64)
unique_train_dates = np.unique(train_dates)
train_day = np.searchsorted(unique_train_dates, train_dates)
age = (len(unique_train_dates) - 1 - train_day).astype(np.float32)
recency_weight = np.exp2(-age / 4.0).astype(np.float32)
recency_weight /= recency_weight.mean()


def user_rank(users, scores):
    """Deterministic percentile rank within each evaluation user."""
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    position = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denom = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked = position / denom

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = ranked
    return result


def build_user_day_order(sample):
    users = np.asarray(sample.user_id, dtype=np.int64)
    dates = np.asarray(sample.date, dtype=np.int64)
    times = np.asarray(sample.time_ms, dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)

    # Row position breaks timestamp ties exactly as specified by the API.
    order = np.lexsort((rows, times, dates, users))
    ou = users[order]
    od = dates[order]
    starts = np.flatnonzero(
        np.r_[True, (ou[1:] != ou[:-1]) | (od[1:] != od[:-1])]
    )
    group_sizes = np.diff(np.r_[starts, len(order)]).astype(np.int32)
    return order, group_sizes


# -------------------------------------------------------------------------
# Shared stationary inputs.
# -------------------------------------------------------------------------
CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "music_type",
    "video_type",
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


def numeric_matrix(sample):
    cols = []
    for field in NUM_FIELDS:
        value = np.asarray(sample.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(value, 0.0)))
    return np.column_stack(cols).astype(np.float32, copy=False)


def history_matrix(split_name):
    cols = []
    names = []
    for key in ("video_id", "author_id"):
        history = historical_features(split_name, key=key)
        for name in sorted(history):
            value = np.asarray(history[name], dtype=np.float32)
            value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(value)
            names.append(key + ":" + name)
    if not cols:
        return np.zeros((ntr if split_name == "train" else
                         nva if split_name == "valid" else nte, 0),
                        dtype=np.float32), names
    return np.column_stack(cols).astype(np.float32, copy=False), names


hist_tr, history_names = history_matrix("train")
hist_va, _ = history_matrix("valid")
hist_te, _ = history_matrix("test")

print("FINDINGS history_feature_count=%d" % len(history_names))


def tree_matrix(sample, hist):
    categorical = np.column_stack([
        np.asarray(sample.X[f], dtype=np.float32) for f in CAT_FIELDS
    ])
    numeric = numeric_matrix(sample)
    return np.column_stack([categorical, numeric, hist]).astype(
        np.float32, copy=False
    )


xtr_tree = tree_matrix(train, hist_tr)
xva_tree = tree_matrix(valid, hist_va)
xte_tree = tree_matrix(test, hist_te)

train_order, train_group_sizes = build_user_day_order(train)

# -------------------------------------------------------------------------
# Family 1: LambdaMART over user-day slates.
#
# Unlike a pointwise classifier, every gradient is induced by ordering rows
# shown to the same user on the same day. This prevents prolific historical
# users and cross-day shifts in their base rate from dominating supervision.
# -------------------------------------------------------------------------
rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "lambdarank_truncation_level": 12,
    "label_gain": [0, 1],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 700,
    "lambda_l2": 8.0,
    "feature_fraction": 0.86,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 35.0,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
}

rank_dset = lgb.Dataset(
    xtr_tree[train_order],
    label=ytr[train_order],
    weight=recency_weight[train_order],
    group=train_group_sizes,
    categorical_feature=list(range(len(CAT_FIELDS))),
    free_raw_data=True,
)

rank_model = lgb.train(
    rank_params,
    rank_dset,
    num_boost_round=165,
)

rank_valid = rank_model.predict(xva_tree)
rank_test = rank_model.predict(xte_tree)

rank_metric = evaluate(uva, yva, rank_valid)
print(
    "FINDINGS lambda_user_day primary=%.6f gauc=%.6f ndcg5=%.6f"
    % (
        rank_metric["primary"],
        rank_metric["gauc"],
        rank_metric["ndcg@5"],
    )
)

del rank_dset, rank_model, xtr_tree, xva_tree, xte_tree
gc.collect()

# -------------------------------------------------------------------------
# Family 2: neural conditional-choice model.
#
# It forms predictions through nonlinear embedding interactions. Alongside
# pointwise calibration, BPR pairs are sampled only inside user-day groups.
# Thus the interaction representation is trained explicitly on local choices.
# -------------------------------------------------------------------------
offsets = []
running = 0
for field in CAT_FIELDS:
    offsets.append(running)
    running += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)
total_cardinality = int(running)


def neural_categorical_matrix(sample):
    result = np.column_stack([
        np.asarray(sample.X[f], dtype=np.int64) for f in CAT_FIELDS
    ])
    result += offsets[None, :]
    return result


xtr_cat = neural_categorical_matrix(train)
xva_cat = neural_categorical_matrix(valid)
xte_cat = neural_categorical_matrix(test)

xtr_num = np.column_stack([numeric_matrix(train), hist_tr]).astype(np.float32)
xva_num = np.column_stack([numeric_matrix(valid), hist_va]).astype(np.float32)
xte_num = np.column_stack([numeric_matrix(test), hist_te]).astype(np.float32)

# Robust train-only scaling.
num_center = np.median(xtr_num, axis=0).astype(np.float32)
q25 = np.quantile(xtr_num, 0.25, axis=0).astype(np.float32)
q75 = np.quantile(xtr_num, 0.75, axis=0).astype(np.float32)
num_scale = np.maximum(q75 - q25, 1.0e-3).astype(np.float32)

xtr_num = np.clip((xtr_num - num_center) / num_scale, -8.0, 8.0)
xva_num = np.clip((xva_num - num_center) / num_scale, -8.0, 8.0)
xte_num = np.clip((xte_num - num_center) / num_scale, -8.0, 8.0)


class ConditionalChoiceNet(nn.Module):
    def __init__(self, cardinality, fields, numeric_dim):
        super().__init__()
        dim = 8
        self.embedding = nn.Embedding(cardinality, dim)
        self.linear = nn.Embedding(cardinality, 1)
        self.numeric_linear = nn.Linear(numeric_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(fields * dim + numeric_dim, 112),
            nn.SiLU(),
            nn.LayerNorm(112),
            nn.Dropout(0.08),
            nn.Linear(112, 40),
            nn.SiLU(),
            nn.Linear(40, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.018)
        nn.init.zeros_(self.linear.weight)

    def forward(self, cat, num):
        emb = self.embedding(cat).flatten(1)
        wide = self.linear(cat).sum(dim=1).squeeze(-1)
        nonlinear = self.mlp(torch.cat([emb, num], dim=1)).squeeze(-1)
        return self.bias + wide + self.numeric_linear(num).squeeze(-1) + nonlinear


model = ConditionalChoiceNet(
    total_cardinality, len(CAT_FIELDS), xtr_num.shape[1]
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0018, weight_decay=2.0e-6
)

rng = np.random.default_rng(SEED + 100)

# Construct one local negative per positive, entirely vectorized.
ordered_y = ytr[train_order]
ordered_users = np.asarray(train.user_id, dtype=np.int64)[train_order]
ordered_dates = np.asarray(train.date, dtype=np.int64)[train_order]

new_group = np.r_[
    True,
    (ordered_users[1:] != ordered_users[:-1])
    | (ordered_dates[1:] != ordered_dates[:-1]),
]
ordered_gid = np.cumsum(new_group, dtype=np.int64) - 1
num_groups = int(ordered_gid[-1]) + 1

negative_mask = ordered_y == 0
negative_rows = train_order[negative_mask]
negative_gid = ordered_gid[negative_mask]
negative_counts = np.bincount(
    negative_gid, minlength=num_groups
).astype(np.int64)
negative_starts = np.cumsum(
    np.r_[0, negative_counts[:-1]], dtype=np.int64
)

positive_mask = ordered_y == 1
positive_rows_all = train_order[positive_mask]
positive_gid_all = ordered_gid[positive_mask]
usable = negative_counts[positive_gid_all] > 0
pair_positive = positive_rows_all[usable]
pair_gid = positive_gid_all[usable]

random_fraction = rng.random(len(pair_positive))
random_offset = (
    random_fraction * negative_counts[pair_gid]
).astype(np.int64)
pair_negative = negative_rows[
    negative_starts[pair_gid] + random_offset
]

print(
    "FINDINGS local_pair_count=%d mixed_user_day_groups=%d"
    % (len(pair_positive), int(np.sum(negative_counts > 0)))
)


def tensor_rows(cat_matrix, num_matrix, rows):
    cat = torch.from_numpy(cat_matrix[rows])
    num = torch.from_numpy(
        np.asarray(num_matrix[rows], dtype=np.float32)
    )
    return cat, num


# One recency-weighted pointwise pass establishes probability calibration.
point_rows = np.arange(ntr, dtype=np.int64)
rng.shuffle(point_rows)
model.train()
for begin in range(0, ntr, 16384):
    rows = point_rows[begin:begin + 16384]
    cat, num = tensor_rows(xtr_cat, xtr_num, rows)
    target = torch.from_numpy(ytr[rows])
    weight = torch.from_numpy(recency_weight[rows])

    optimizer.zero_grad(set_to_none=True)
    logits = model(cat, num)
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    loss = (losses * weight).sum() / weight.sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

# Two local conditional-choice passes. Each pair has a positive and negative
# from exactly the same user-day, removing user/day intercepts from the loss.
for epoch in range(2):
    permutation = rng.permutation(len(pair_positive))
    for begin in range(0, len(permutation), 12288):
        choice = permutation[begin:begin + 12288]
        pos_rows = pair_positive[choice]
        neg_rows = pair_negative[choice]

        pos_cat, pos_num = tensor_rows(xtr_cat, xtr_num, pos_rows)
        neg_cat, neg_num = tensor_rows(xtr_cat, xtr_num, neg_rows)
        weight = torch.from_numpy(recency_weight[pos_rows])

        optimizer.zero_grad(set_to_none=True)
        positive_score = model(pos_cat, pos_num)
        negative_score = model(neg_cat, neg_num)
        losses = nn.functional.softplus(
            -(positive_score - negative_score)
        )
        loss = (losses * weight).sum() / weight.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def neural_predict(cat_matrix, num_matrix):
    result = np.empty(len(cat_matrix), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(cat_matrix), 32768):
            end = min(begin + 32768, len(cat_matrix))
            cat = torch.from_numpy(cat_matrix[begin:end])
            num = torch.from_numpy(
                np.asarray(num_matrix[begin:end], dtype=np.float32)
            )
            result[begin:end] = model(cat, num).cpu().numpy()
    return result


neural_valid = neural_predict(xva_cat, xva_num)
neural_test = neural_predict(xte_cat, xte_num)

neural_metric = evaluate(uva, yva, neural_valid)
print(
    "FINDINGS neural_local_choice primary=%.6f gauc=%.6f ndcg5=%.6f"
    % (
        neural_metric["primary"],
        neural_metric["gauc"],
        neural_metric["ndcg@5"],
    )
)

# -------------------------------------------------------------------------
# Rank-space blending with trusted incumbent.
#
# Within-user percentile normalization prevents arbitrary model score scales
# from determining blend weights. Weight selection is explicitly permitted
# for the trusted-incumbent API and is applied unchanged to test.
# -------------------------------------------------------------------------
shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)

valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

inc_valid_rank = user_rank(valid_users, inc_valid)
inc_test_rank = user_rank(test_users, inc_test)

family_predictions = {
    "lambda_user_day": (rank_valid, rank_test),
    "neural_local_choice": (neural_valid, neural_test),
}

candidate_scores = {}
candidate_records = []

inc_metric = evaluate(valid_users, yva, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
candidate_records.append({
    "name": "trusted_incumbent",
    "metric": inc_metric,
    "valid": inc_valid,
    "test": inc_test,
    "raw_valid": rank_valid,
})

for family_name, (own_valid, own_test) in family_predictions.items():
    own_metric = evaluate(valid_users, yva, own_valid)
    candidate_scores[family_name] = float(own_metric["primary"])
    candidate_records.append({
        "name": family_name,
        "metric": own_metric,
        "valid": own_valid,
        "test": own_test,
        "raw_valid": own_valid,
    })

    own_valid_rank = user_rank(valid_users, own_valid)
    own_test_rank = user_rank(test_users, own_test)

    # alpha is the incumbent weight. Include endpoints and moderately fine
    # interior weights; all are mapped unchanged to hidden test.
    for alpha in np.linspace(0.1, 0.9, 9):
        valid_blend = (
            alpha * inc_valid_rank + (1.0 - alpha) * own_valid_rank
        )
        test_blend = (
            alpha * inc_test_rank + (1.0 - alpha) * own_test_rank
        )
        blend_metric = evaluate(valid_users, yva, valid_blend)
        name = "%s_blend_inc_%.1f" % (family_name, alpha)
        candidate_scores[name] = float(blend_metric["primary"])
        candidate_records.append({
            "name": name,
            "metric": blend_metric,
            "valid": valid_blend,
            "test": test_blend,
            "raw_valid": own_valid,
        })

# Also test a genuinely three-family rank aggregation.
rank_valid_rank = user_rank(valid_users, rank_valid)
rank_test_rank = user_rank(test_users, rank_test)
neural_valid_rank = user_rank(valid_users, neural_valid)
neural_test_rank = user_rank(test_users, neural_test)

for incumbent_weight in (0.4, 0.6, 0.8):
    remaining = 1.0 - incumbent_weight
    valid_blend = (
        incumbent_weight * inc_valid_rank
        + 0.5 * remaining * rank_valid_rank
        + 0.5 * remaining * neural_valid_rank
    )
    test_blend = (
        incumbent_weight * inc_test_rank
        + 0.5 * remaining * rank_test_rank
        + 0.5 * remaining * neural_test_rank
    )
    metric = evaluate(valid_users, yva, valid_blend)
    name = "three_family_inc_%.1f" % incumbent_weight
    candidate_scores[name] = float(metric["primary"])
    candidate_records.append({
        "name": name,
        "metric": metric,
        "valid": valid_blend,
        "test": test_blend,
        "raw_valid": 0.5 * rank_valid + 0.5 * neural_valid,
    })

best = max(
    candidate_records,
    key=lambda record: record["metric"]["primary"]
)

print("CANDIDATES " + json.dumps(
    candidate_scores, sort_keys=True
))
print(
    "FINDINGS selected_candidate=%s selected_primary=%.6f"
    % (best["name"], best["metric"]["primary"])
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best["valid"], dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best["test"], dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best["raw_valid"], dtype=np.float64),
    )

elapsed = time.time() - START
final_metric = best["metric"]
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        final_metric["primary"],
        final_metric["gauc"],
        final_metric["ndcg@5"],
        elapsed,
    )
)