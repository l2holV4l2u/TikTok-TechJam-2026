import os
import gc
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 27183
N_THREADS = min(8, os.cpu_count() or 1)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(N_THREADS)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "hour",
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
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]

BPR_EPOCHS = 6
BPR_BATCH = 8192
PRED_BATCH = 65536


def make_matrix(split):
    cats = np.column_stack(
        [np.asarray(split.X[f], dtype=np.float32) for f in CAT_FIELDS]
    )
    nums = []
    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        value = np.log1p(np.maximum(value, 0.0))
        nums.append(value)
    nums = np.column_stack(nums).astype(np.float32, copy=False)
    return np.concatenate([cats, nums], axis=1).astype(np.float32, copy=False)


def group_order(users):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    if sorted_users.size == 0:
        return order, np.empty(0, dtype=np.int32)
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    groups = np.diff(np.r_[starts, sorted_users.size]).astype(np.int32)
    return order, groups


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]

    starts_mask = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.flatnonzero(starts_mask)
    group_index = np.cumsum(starts_mask) - 1
    group_starts = starts[group_index]
    positions = np.arange(n, dtype=np.int64) - group_starts
    group_sizes = np.diff(np.r_[starts, n])
    denominators = np.maximum(group_sizes[group_index] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denominators
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def fit_lgb_binary(x, y, num_rounds):
    dataset = lgb.Dataset(
        x,
        label=y,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.06,
        "num_leaves": 63,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "max_cat_to_onehot": 16,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "num_threads": N_THREADS,
        "seed": SEED + 11,
        "feature_fraction_seed": SEED + 12,
        "bagging_seed": SEED + 13,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=num_rounds)


def fit_lgb_rank(x, y, users, num_rounds):
    order, groups = group_order(users)
    dataset = lgb.Dataset(
        x[order],
        label=y[order],
        group=groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "label_gain": [0, 1],
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "max_cat_to_onehot": 16,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "num_threads": N_THREADS,
        "seed": SEED + 21,
        "feature_fraction_seed": SEED + 22,
        "bagging_seed": SEED + 23,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=num_rounds)


class BPRModel(nn.Module):
    def __init__(self, rank=32):
        super().__init__()
        self.user = nn.Embedding(int(FEATURE_CARDINALITIES["user_id"]), rank)
        self.video = nn.Embedding(int(FEATURE_CARDINALITIES["video_id"]), rank)
        self.author = nn.Embedding(int(FEATURE_CARDINALITIES["author_id"]), rank)
        self.video_bias = nn.Embedding(
            int(FEATURE_CARDINALITIES["video_id"]), 1
        )
        self.author_bias = nn.Embedding(
            int(FEATURE_CARDINALITIES["author_id"]), 1
        )

        nn.init.normal_(self.user.weight, std=0.03)
        nn.init.normal_(self.video.weight, std=0.03)
        nn.init.normal_(self.author.weight, std=0.03)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.author_bias.weight)

    def forward(self, users, videos, authors):
        u = self.user(users)
        score = torch.sum(u * self.video(videos), dim=1)
        score += 0.55 * torch.sum(u * self.author(authors), dim=1)
        score += self.video_bias(videos).squeeze(1)
        score += 0.55 * self.author_bias(authors).squeeze(1)
        return score


def fit_bpr(users, videos, authors, labels):
    torch.manual_seed(SEED + 31)
    rng = np.random.default_rng(SEED + 32)
    model = BPRModel(rank=32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.006, weight_decay=2e-6
    )

    users = np.asarray(users, dtype=np.int64)
    videos = np.asarray(videos, dtype=np.int64)
    authors = np.asarray(authors, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)

    order = np.argsort(users, kind="mergesort")
    su = users[order]
    sv = videos[order]
    sa = authors[order]
    sy = labels[order]

    cardinality = int(FEATURE_CARDINALITIES["user_id"])
    counts = np.bincount(su, minlength=cardinality).astype(np.int64)
    starts = np.zeros(cardinality, dtype=np.int64)
    if cardinality > 1:
        starts[1:] = np.cumsum(counts[:-1])

    positive_rows = np.flatnonzero(sy == 1)
    positive_users = su[positive_rows]
    eligible = counts[positive_users] > 1
    positive_rows = positive_rows[eligible]
    positive_users = positive_users[eligible]

    for _ in range(BPR_EPOCHS):
        negative_rows = (
            starts[positive_users]
            + (rng.random(positive_rows.size) * counts[positive_users]).astype(
                np.int64
            )
        )

        for _retry in range(12):
            bad = sy[negative_rows] != 0
            if not np.any(bad):
                break
            bad_users = positive_users[bad]
            negative_rows[bad] = (
                starts[bad_users]
                + (rng.random(bad_users.size) * counts[bad_users]).astype(
                    np.int64
                )
            )

        keep = sy[negative_rows] == 0
        pos = positive_rows[keep]
        neg = negative_rows[keep]
        perm = rng.permutation(pos.size)

        model.train()
        for begin in range(0, perm.size, BPR_BATCH):
            idx = perm[begin:begin + BPR_BATCH]
            p = pos[idx]
            n = neg[idx]

            tu = torch.from_numpy(su[p])
            pv = torch.from_numpy(sv[p])
            pa = torch.from_numpy(sa[p])
            nv = torch.from_numpy(sv[n])
            na = torch.from_numpy(sa[n])

            positive_score = model(tu, pv, pa)
            negative_score = model(tu, nv, na)
            loss = -F.logsigmoid(positive_score - negative_score).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_bpr(model, users, videos, authors):
    model.eval()
    users = np.asarray(users, dtype=np.int64)
    videos = np.asarray(videos, dtype=np.int64)
    authors = np.asarray(authors, dtype=np.int64)
    result = np.empty(users.size, dtype=np.float32)

    for begin in range(0, users.size, PRED_BATCH):
        end = min(begin + PRED_BATCH, users.size)
        result[begin:end] = model(
            torch.from_numpy(users[begin:end]),
            torch.from_numpy(videos[begin:end]),
            torch.from_numpy(authors[begin:end]),
        ).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_videos = np.asarray(train.video_id, dtype=np.int64)
valid_videos = np.asarray(valid.video_id, dtype=np.int64)
train_authors = np.asarray(train.X["author_id"], dtype=np.int64)
valid_authors = np.asarray(valid.X["author_id"], dtype=np.int64)

x_train = make_matrix(train)
x_valid = make_matrix(valid)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.size != y_valid.size:
    raise ValueError("Trusted incumbent validation length mismatch")
inc_valid_rank = within_user_rank(valid_users, inc_valid)

raw_predictions = {}
models = {}

binary_rounds = 190
rank_rounds = 210

binary_model = fit_lgb_binary(x_train, y_train, binary_rounds)
raw_predictions["lgb_binary"] = binary_model.predict(
    x_valid, num_iteration=binary_rounds
).astype(np.float64)
models["lgb_binary"] = binary_model

rank_model = fit_lgb_rank(x_train, y_train, train_users, rank_rounds)
raw_predictions["lambdamart"] = rank_model.predict(
    x_valid, num_iteration=rank_rounds
).astype(np.float64)
models["lambdamart"] = rank_model

bpr_model = fit_bpr(
    train_users, train_videos, train_authors, y_train
)
raw_predictions["bpr_latent"] = predict_bpr(
    bpr_model, valid_users, valid_videos, valid_authors
).astype(np.float64)
models["bpr_latent"] = bpr_model

candidate_scores = {}
blend_weights = {}

best_name = None
best_family = None
best_weight = None
best_scores = None
best_raw = None
best_metrics = None

for family, raw in raw_predictions.items():
    raw_metrics = evaluate(valid_users, y_valid, raw)
    candidate_scores[family + "_raw"] = float(raw_metrics["primary"])

    own_rank = within_user_rank(valid_users, raw)
    local_metrics = raw_metrics
    local_scores = raw
    local_weight = 1.0

    for weight in np.linspace(0.0, 1.0, 21):
        blended = weight * own_rank + (1.0 - weight) * inc_valid_rank
        metrics = evaluate(valid_users, y_valid, blended)
        if float(metrics["primary"]) > float(local_metrics["primary"]):
            local_metrics = metrics
            local_scores = blended.copy()
            local_weight = float(weight)

    candidate_scores[family + "_blend"] = float(local_metrics["primary"])
    blend_weights[family] = local_weight

    if best_metrics is None or float(local_metrics["primary"]) > float(
        best_metrics["primary"]
    ):
        best_name = (
            family + "_raw" if local_weight == 1.0 else family + "_blend"
        )
        best_family = family
        best_weight = local_weight
        best_scores = np.asarray(local_scores, dtype=np.float64)
        best_raw = np.asarray(raw, dtype=np.float64)
        best_metrics = local_metrics

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "selected_family": best_family,
            "own_rank_weight": best_weight,
            "blend_weights": blend_weights,
            "binary_rounds": binary_rounds,
            "lambdamart_rounds": rank_rounds,
            "bpr_epochs": BPR_EPOCHS,
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_weight < 1.0 - 1e-12:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

# The selected recipe is now refit on train + validation.
for key in list(models.keys()):
    del models[key]
models.clear()
del binary_model, rank_model, bpr_model
gc.collect()

test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
test_videos = np.asarray(test.video_id, dtype=np.int64)
test_authors = np.asarray(test.X["author_id"], dtype=np.int64)

refit_users = np.concatenate([train_users, valid_users])
refit_videos = np.concatenate([train_videos, valid_videos])
refit_authors = np.concatenate([train_authors, valid_authors])
y_refit = np.concatenate([y_train, y_valid])

if best_family in ("lgb_binary", "lambdamart"):
    x_test = make_matrix(test)
    x_refit = np.concatenate([x_train, x_valid], axis=0)

    if best_family == "lgb_binary":
        refit_model = fit_lgb_binary(x_refit, y_refit, binary_rounds)
        own_test = refit_model.predict(
            x_test, num_iteration=binary_rounds
        ).astype(np.float64)
    else:
        refit_model = fit_lgb_rank(
            x_refit, y_refit, refit_users, rank_rounds
        )
        own_test = refit_model.predict(
            x_test, num_iteration=rank_rounds
        ).astype(np.float64)

    del refit_model, x_refit, x_test
else:
    refit_model = fit_bpr(
        refit_users, refit_videos, refit_authors, y_refit
    )
    own_test = predict_bpr(
        refit_model, test_users, test_videos, test_authors
    ).astype(np.float64)
    del refit_model

if best_weight < 1.0 - 1e-12:
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    if inc_test.size != test_users.size:
        raise ValueError("Trusted incumbent test length mismatch")
    own_test_rank = within_user_rank(test_users, own_test)
    inc_test_rank = within_user_rank(test_users, inc_test)
    test_scores = (
        best_weight * own_test_rank
        + (1.0 - best_weight) * inc_test_rank
    )
else:
    test_scores = own_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
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