import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 731
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour", "onehot_feat3",
    "onehot_feat8", "user_active_degree",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
N_CAT = len(CAT_FIELDS)
MMOE_EPOCHS = 3
MMOE_BATCH = 4096
PRED_BATCH = 32768


def numeric_matrix(split):
    cols = []
    for name in NUM_FIELDS:
        z = np.asarray(split.num[name], dtype=np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.log1p(np.maximum(z, 0.0)).astype(np.float32)
        cols.append(z)
    return np.column_stack(cols).astype(np.float32, copy=False)


def categorical_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int32) for name in CAT_FIELDS
    ]).astype(np.int32, copy=False)


def lgb_matrix(split):
    cat = categorical_matrix(split)
    num = numeric_matrix(split)
    return np.concatenate(
        [cat.astype(np.float32), num], axis=1
    ).astype(np.float32, copy=False)


def recency_weights(dates, half_life=5.0):
    d = np.asarray(dates, dtype=np.int64)
    # Dates are consecutive within each fit window, so ordinal unique-date age
    # avoids doing arithmetic directly on YYYYMMDD values.
    unique = np.unique(d)
    ordinal = np.searchsorted(unique, d)
    age = (len(unique) - 1 - ordinal).astype(np.float32)
    w = np.exp2(-age / float(half_life)).astype(np.float32)
    return w / np.mean(w)


def train_binary(X, y, dates, rounds=240):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.88,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "min_data_per_group": 100,
        "cat_smooth": 20.0,
        "cat_l2": 15.0,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": min(8, os.cpu_count() or 1),
        "verbose": -1,
    }
    ds = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        weight=recency_weights(dates, 5.0),
        categorical_feature=list(range(N_CAT)),
        free_raw_data=False,
    )
    return lgb.train(params, ds, num_boost_round=rounds)


def grouped_order_and_sizes(users):
    users = np.asarray(users)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(sorted_users) == 0:
        return order, np.empty(0, dtype=np.int32)
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    edges = np.concatenate(([0], boundaries, [len(sorted_users)]))
    sizes = np.diff(edges).astype(np.int32)
    return order, sizes


def train_lambdarank(X, y, users, rounds=210):
    order, groups = grouped_order_and_sizes(users)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.045,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 160,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 2.5,
        "max_bin": 127,
        "min_data_per_group": 100,
        "cat_smooth": 20.0,
        "cat_l2": 15.0,
        "label_gain": [0, 1],
        "seed": SEED + 10,
        "feature_fraction_seed": SEED + 11,
        "bagging_seed": SEED + 12,
        "num_threads": min(8, os.cpu_count() or 1),
        "verbose": -1,
    }
    ds = lgb.Dataset(
        X[order],
        label=np.asarray(y, dtype=np.float32)[order],
        group=groups,
        categorical_feature=list(range(N_CAT)),
        free_raw_data=False,
    )
    return lgb.train(params, ds, num_boost_round=rounds)


CAT_OFFSETS = []
_total_cat = 0
for _name in CAT_FIELDS:
    CAT_OFFSETS.append(_total_cat)
    _total_cat += int(FEATURE_CARDINALITIES[_name])
CAT_OFFSETS = np.asarray(CAT_OFFSETS, dtype=np.int64)
TOTAL_CAT = int(_total_cat)


def neural_cat_matrix(split):
    x = categorical_matrix(split).astype(np.int64)
    x += CAT_OFFSETS[None, :]
    return x


class MMoE(nn.Module):
    def __init__(self, num_mean, num_scale, n_tasks=3):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CAT, 8)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.register_buffer(
            "num_mean", torch.as_tensor(num_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "num_scale", torch.as_tensor(num_scale, dtype=torch.float32)
        )

        input_dim = len(CAT_FIELDS) * 8 + len(NUM_FIELDS)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, 4) for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, cat, num):
        emb = self.embedding(cat).flatten(1)
        num = (num - self.num_mean) / self.num_scale
        h = torch.cat([emb, num], dim=1)
        expert = torch.stack([e(h) for e in self.experts], dim=1)

        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(h), dim=1).unsqueeze(-1)
            mixed = torch.sum(expert * weights, dim=1)
            outputs.append(tower(mixed).squeeze(1))
        return torch.stack(outputs, dim=1)


def aux_targets(split):
    click = np.asarray(split.aux["is_click"], dtype=np.float32)
    like = np.asarray(split.aux["is_like"], dtype=np.float32)
    click = np.nan_to_num(click, nan=0.0)
    like = np.nan_to_num(like, nan=0.0)
    return click, like


def train_mmoe(cat, num, targets, dates, epochs=MMOE_EPOCHS):
    torch.manual_seed(SEED + 100)
    np.random.seed(SEED + 100)

    mean = num.mean(axis=0).astype(np.float32)
    scale = num.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, 0.1)

    model = MMoE(mean, scale, n_tasks=3)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.4e-3, weight_decay=2e-6
    )

    cat_t = torch.from_numpy(cat)
    num_t = torch.from_numpy(num)
    target_t = torch.from_numpy(targets.astype(np.float32, copy=False))
    weights_t = torch.from_numpy(recency_weights(dates, 5.0))

    generator = torch.Generator()
    generator.manual_seed(SEED + 101)
    n = len(cat)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        running = 0.0
        for start in range(0, n, MMOE_BATCH):
            idx = order[start:min(start + MMOE_BATCH, n)]
            logits = model(cat_t[idx], num_t[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, target_t[idx], reduction="none"
            )
            # Long-view remains the primary task. Click and like regularize
            # representation learning without becoming inference inputs.
            task_loss = (
                losses[:, 0]
                + 0.25 * losses[:, 1]
                + 0.15 * losses[:, 2]
            )
            loss = (task_loss * weights_t[idx]).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(idx)

        print(
            "mmoe_epoch=%d loss=%.6f" % (epoch + 1, running / n),
            flush=True,
        )
    return model


@torch.no_grad()
def predict_mmoe(model, cat, num):
    model.eval()
    out = np.empty(len(cat), dtype=np.float32)
    for start in range(0, len(cat), PRED_BATCH):
        end = min(start + PRED_BATCH, len(cat))
        logits = model(
            torch.from_numpy(cat[start:end]),
            torch.from_numpy(num[start:end]),
        )
        out[start:end] = logits[:, 0].cpu().numpy()
    return out


def within_user_rank(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, users))
    sorted_users = users[order]

    starts = np.empty(n, dtype=np.int64)
    ends = np.empty(n, dtype=np.int64)
    change = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    group_starts = np.flatnonzero(change)
    group_ends = np.r_[group_starts[1:], n]
    lengths = group_ends - group_starts

    starts[:] = np.repeat(group_starts, lengths)
    ends[:] = np.repeat(group_ends, lengths)
    positions = np.arange(n, dtype=np.float64) - starts
    denom = np.maximum(ends - starts - 1, 1)
    ranked = positions / denom
    singleton = (ends - starts) == 1
    ranked[singleton] = 0.5

    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def score_primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
train_users = np.asarray(train.user_id)

X_train = lgb_matrix(train)
X_valid = lgb_matrix(valid)

print("training_binary", flush=True)
binary_model = train_binary(X_train, train_y, train.date)
pred_binary = binary_model.predict(X_valid).astype(np.float32)

print("training_lambdarank", flush=True)
rank_model = train_lambdarank(X_train, train_y, train_users)
pred_rank = rank_model.predict(X_valid).astype(np.float32)

print("training_mmoe", flush=True)
train_cat = neural_cat_matrix(train)
valid_cat = neural_cat_matrix(valid)
train_num = numeric_matrix(train)
valid_num = numeric_matrix(valid)
tr_click, tr_like = aux_targets(train)
mmoe_targets = np.column_stack([train_y, tr_click, tr_like]).astype(np.float32)
mmoe_model = train_mmoe(
    train_cat, train_num, mmoe_targets, train.date, epochs=MMOE_EPOCHS
)
pred_mmoe = predict_mmoe(mmoe_model, valid_cat, valid_num)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)

family_predictions = {
    "binary_lgb_recency": pred_binary,
    "lambdarank_user_grouped": pred_rank,
    "mmoe_auxiliary": pred_mmoe,
}

candidate_scores = {}
candidate_payloads = []

inc_rank_valid = within_user_rank(valid_users, inc_valid)
candidate_scores["trusted_incumbent"] = score_primary(
    valid_users, valid_y, inc_valid
)

for family, pred in family_predictions.items():
    raw_metric = score_primary(valid_users, valid_y, pred)
    candidate_scores[family + "_raw"] = raw_metric
    candidate_payloads.append((raw_metric, family, 1.0, pred, False))

    own_rank = within_user_rank(valid_users, pred)
    for weight in (0.20, 0.35, 0.50, 0.65, 0.80):
        blended = (1.0 - weight) * inc_rank_valid + weight * own_rank
        metric = score_primary(valid_users, valid_y, blended)
        name = "%s_blend_%.2f" % (family, weight)
        candidate_scores[name] = metric
        candidate_payloads.append(
            (metric, family, weight, blended, True)
        )

best_metric, best_family, best_weight, valid_scores, is_blend = max(
    candidate_payloads, key=lambda z: z[0]
)
valid_scores = np.asarray(valid_scores, dtype=np.float64)
valid_metrics = evaluate(valid_users, valid_y, valid_scores)

print(
    "CANDIDATES " + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
        sort_keys=True,
        separators=(", ", ": "),
    ),
    flush=True,
)
print(
    "FINDINGS selected_family=%s blend=%s weight=%.2f primary=%.6f"
    % (best_family, str(is_blend), best_weight, best_metric),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )
    if is_blend:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(family_predictions[best_family], dtype=np.float64),
        )

# Refit only the selected family on train + validation using the identical
# recipe, then apply the validation-selected blending weight unchanged.
combined_y = np.concatenate([
    train_y,
    valid_y.astype(np.float32, copy=False),
])
combined_dates = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])
combined_users = np.concatenate([
    np.asarray(train.user_id),
    np.asarray(valid.user_id),
])

test = load("test")
test_users = np.asarray(test.user_id)

if best_family in ("binary_lgb_recency", "lambdarank_user_grouped"):
    X_combined = np.concatenate([X_train, X_valid], axis=0)
    X_test = lgb_matrix(test)

    if best_family == "binary_lgb_recency":
        final_model = train_binary(
            X_combined, combined_y, combined_dates
        )
    else:
        final_model = train_lambdarank(
            X_combined, combined_y, combined_users
        )
    own_test = final_model.predict(X_test).astype(np.float32)

else:
    valid_click, valid_like = aux_targets(valid)
    combined_targets = np.concatenate([
        mmoe_targets,
        np.column_stack([
            valid_y.astype(np.float32),
            valid_click,
            valid_like,
        ]).astype(np.float32),
    ], axis=0)
    combined_cat = np.concatenate([train_cat, valid_cat], axis=0)
    combined_num = np.concatenate([train_num, valid_num], axis=0)

    final_model = train_mmoe(
        combined_cat,
        combined_num,
        combined_targets,
        combined_dates,
        epochs=MMOE_EPOCHS,
    )
    test_cat = neural_cat_matrix(test)
    test_num = numeric_matrix(test)
    own_test = predict_mmoe(final_model, test_cat, test_num)

if is_blend:
    inc_test = np.load(inc_test_path).astype(np.float64)
    inc_rank_test = within_user_rank(test_users, inc_test)
    own_rank_test = within_user_rank(test_users, own_test)
    test_scores = (
        (1.0 - best_weight) * inc_rank_test
        + best_weight * own_rank_test
    )
else:
    test_scores = np.asarray(own_test, dtype=np.float64)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
result = {
    "primary": float(valid_metrics["primary"]),
    "gauc": float(valid_metrics["gauc"]),
    "ndcg@5": float(valid_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))