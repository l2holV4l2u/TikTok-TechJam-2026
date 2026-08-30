import os
import time
import json
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 19037
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

DEVICE = torch.device("cpu")
BATCH_SIZE = 16384
HALF_LIFE = 7.0

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
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "register_days_bucket",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")


def temporal_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    unique, inverse = np.unique(dates, return_inverse=True)
    age = inverse.max() - inverse
    return np.power(0.5, age.astype(np.float32) / HALF_LIFE).astype(np.float32)


offsets = np.cumsum(
    [0] + [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS[:-1]],
    dtype=np.int64,
)
TOTAL_CARD = int(sum(int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS))


def make_cat_matrix(split):
    x = np.column_stack([split.X[f] for f in CAT_FIELDS]).astype(
        np.int64, copy=False
    )
    x = x + offsets[None, :]
    return np.ascontiguousarray(x)


class PairwiseFM(nn.Module):
    def __init__(self, cardinality, emb_dim=20):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, emb_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        e = self.embedding(x)
        summed = e.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1) - e.square().sum(dim=(1, 2))
        )
        linear = self.linear(x).sum(dim=1).squeeze(1)
        return linear + interaction


def build_pairs(users, labels, repeats, seed):
    users = np.asarray(users, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    rng = np.random.default_rng(seed)

    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    sorted_y = labels[order]

    unique_users, starts, counts = np.unique(
        sorted_users, return_index=True, return_counts=True
    )
    max_uid = int(max(users.max(initial=0), unique_users.max(initial=0))) + 1

    start_map = np.zeros(max_uid, dtype=np.int64)
    count_map = np.zeros(max_uid, dtype=np.int64)
    start_map[unique_users] = starts
    count_map[unique_users] = counts

    positives = np.flatnonzero(labels > 0)
    positive_users = users[positives]

    total_count = count_map[positive_users]
    positive_counts = np.bincount(
        users[labels > 0], minlength=max_uid
    )[positive_users]
    usable = positive_counts < total_count
    positives = positives[usable]

    positives = np.tile(positives, repeats)
    pair_users = users[positives]
    starts_for_pair = start_map[pair_users]
    counts_for_pair = count_map[pair_users]

    offsets_random = (
        rng.random(len(positives)) * counts_for_pair
    ).astype(np.int64)
    neg_sorted_pos = starts_for_pair + offsets_random

    bad = sorted_y[neg_sorted_pos] != 0
    for _ in range(24):
        if not np.any(bad):
            break
        replacement = (
            rng.random(int(bad.sum())) * counts_for_pair[bad]
        ).astype(np.int64)
        neg_sorted_pos[bad] = starts_for_pair[bad] + replacement
        bad = sorted_y[neg_sorted_pos] != 0

    keep = sorted_y[neg_sorted_pos] == 0
    positives = positives[keep]
    negatives = order[neg_sorted_pos[keep]]
    return positives.astype(np.int64), negatives.astype(np.int64)


@torch.no_grad()
def predict_torch(model, x):
    model.eval()
    out = np.empty(len(x), dtype=np.float64)
    for lo in range(0, len(x), BATCH_SIZE * 2):
        hi = min(lo + BATCH_SIZE * 2, len(x))
        xb = torch.from_numpy(x[lo:hi]).to(DEVICE)
        out[lo:hi] = model(xb).cpu().numpy().astype(np.float64)
    return out


def fit_bpr(split_list, label_list, epochs, select_split=None):
    matrices = [make_cat_matrix(s) for s in split_list]
    users = np.concatenate(
        [np.asarray(s.user_id, dtype=np.int64) for s in split_list]
    )
    dates = np.concatenate(
        [np.asarray(s.date, dtype=np.int32) for s in split_list]
    )
    labels = np.concatenate(
        [np.asarray(y, dtype=np.float32) for y in label_list]
    )
    x = np.ascontiguousarray(np.concatenate(matrices, axis=0))
    date_weight = temporal_weights(dates)

    pos_idx, neg_idx = build_pairs(
        users, labels.astype(np.int8), repeats=2, seed=SEED + 71
    )

    model = PairwiseFM(TOTAL_CARD, emb_dim=20).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0017, weight_decay=2e-6
    )
    rng = np.random.default_rng(SEED + 81)

    x_select = make_cat_matrix(select_split) if select_split is not None else None
    best_primary = -np.inf
    best_epoch = epochs
    best_state = None
    best_score = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(pos_idx))
        for lo in range(0, len(perm), BATCH_SIZE):
            take = perm[lo:lo + BATCH_SIZE]
            pi = pos_idx[take]
            ni = neg_idx[take]

            xp = torch.from_numpy(x[pi]).to(DEVICE)
            xn = torch.from_numpy(x[ni]).to(DEVICE)
            wp = torch.from_numpy(date_weight[pi]).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            margin = model(xp) - model(xn)
            losses = nn.functional.softplus(-margin)
            loss = (losses * wp).sum() / wp.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if select_split is not None:
            score = predict_torch(model, x_select)
            met = evaluate(select_split.user_id, y_valid, score)
            if float(met["primary"]) > best_primary:
                best_primary = float(met["primary"])
                best_epoch = epoch
                best_score = score.copy()
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

    if select_split is not None:
        model.load_state_dict(best_state)

    del matrices, x, users, dates, labels, date_weight, pos_idx, neg_idx
    gc.collect()
    return model, best_score, best_epoch


def make_lgb_matrix(split):
    cols = []
    for f in CAT_FIELDS:
        cols.append(np.asarray(split.X[f], dtype=np.float32))
    for f in NUM_FIELDS:
        z = np.asarray(split.num[f], dtype=np.float32)
        z = np.nan_to_num(z, nan=-1.0, posinf=1e8, neginf=-1.0)
        z = np.sign(z) * np.log1p(np.abs(z))
        cols.append(z.astype(np.float32, copy=False))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def rank_order_and_groups(users):
    users = np.asarray(users, dtype=np.int64)
    order = np.argsort(users, kind="stable")
    _, counts = np.unique(users[order], return_counts=True)
    return order, counts.astype(np.int32)


LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.86,
    "bagging_fraction": 0.86,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 2.5,
    "max_bin": 127,
    "cat_smooth": 25.0,
    "cat_l2": 10.0,
    "lambdarank_truncation_level": 12,
    "verbosity": -1,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
}


def fit_lambdarank_valid():
    Xtr = make_lgb_matrix(train)
    Xva = make_lgb_matrix(valid)
    otr, gtr = rank_order_and_groups(train.user_id)
    ova, gva = rank_order_and_groups(valid.user_id)

    dtrain = lgb.Dataset(
        Xtr[otr],
        label=y_train[otr],
        group=gtr,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        Xva[ova],
        label=y_valid[ova],
        group=gva,
        reference=dtrain,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=360,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(45, verbose=False)],
    )
    pred_sorted = model.predict(
        Xva[ova],
        num_iteration=model.best_iteration,
        raw_score=True,
    )
    pred = np.empty(len(valid.user_id), dtype=np.float64)
    pred[ova] = pred_sorted
    rounds = int(model.best_iteration)

    del model, dtrain, dvalid, Xtr, Xva, otr, ova, gtr, gva
    gc.collect()
    return pred, rounds


# Family 1: neural BPR factorization.
bpr_model, bpr_valid, bpr_epoch = fit_bpr(
    [train], [y_train], epochs=5, select_split=valid
)
del bpr_model
gc.collect()

# Family 2: tree-based LambdaRank.
lambda_valid, lambda_rounds = fit_lambdarank_valid()

families = {
    "pairwise_bpr_fm": bpr_valid,
    "tree_lambdarank": lambda_valid,
}

inc_scale = max(float(np.std(inc_valid)), 1e-8)
blend_weights = [0.15, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85]

candidate_scores = {}
best_primary = -np.inf
best_family = None
best_alpha = None
best_valid = None
best_raw_valid = None
best_metrics = None
best_own_scale = None

for family, own in families.items():
    own = np.asarray(own, dtype=np.float64)
    own_scale = max(float(np.std(own)), 1e-8)

    own_met = evaluate(valid.user_id, y_valid, own)
    candidate_scores[family + "_standalone"] = float(own_met["primary"])
    if float(own_met["primary"]) > best_primary:
        best_primary = float(own_met["primary"])
        best_family = family
        best_alpha = 1.0
        best_valid = own.copy()
        best_raw_valid = own.copy()
        best_metrics = own_met
        best_own_scale = own_scale

    family_best = (-np.inf, None, None, None)
    for alpha in blend_weights:
        blend = (
            alpha * own / own_scale
            + (1.0 - alpha) * inc_valid / inc_scale
        )
        met = evaluate(valid.user_id, y_valid, blend)
        p = float(met["primary"])
        if p > family_best[0]:
            family_best = (p, alpha, blend, met)

    p, alpha, blend, met = family_best
    candidate_scores[family + "_incumbent_blend"] = p
    if p > best_primary:
        best_primary = p
        best_family = family
        best_alpha = float(alpha)
        best_valid = blend.copy()
        best_raw_valid = own.copy()
        best_metrics = met
        best_own_scale = own_scale

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s own_weight=%.2f bpr_epoch=%d lambda_rounds=%d"
    % (best_family, best_alpha, bpr_epoch, lambda_rounds)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    if best_alpha < 0.999999:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

# Refit the selected recipe on train + validation, then score test.
test = load("test")

if best_family == "pairwise_bpr_fm":
    torch.manual_seed(SEED)
    final_model, _, _ = fit_bpr(
        [train, valid],
        [y_train, y_valid.astype(np.float32)],
        epochs=bpr_epoch,
        select_split=None,
    )
    xtest = make_cat_matrix(test)
    raw_test = predict_torch(final_model, xtest)
    del final_model, xtest

else:
    Xtr = make_lgb_matrix(train)
    Xva = make_lgb_matrix(valid)
    Xcombined = np.ascontiguousarray(
        np.concatenate([Xtr, Xva], axis=0), dtype=np.float32
    )
    combined_y = np.concatenate(
        [y_train, y_valid.astype(np.float32)]
    )
    combined_users = np.concatenate(
        [
            np.asarray(train.user_id, dtype=np.int64),
            np.asarray(valid.user_id, dtype=np.int64),
        ]
    )
    order, groups = rank_order_and_groups(combined_users)

    dfinal = lgb.Dataset(
        Xcombined[order],
        label=combined_y[order],
        group=groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    final_model = lgb.train(
        LGB_PARAMS,
        dfinal,
        num_boost_round=lambda_rounds,
    )
    Xtest = make_lgb_matrix(test)
    raw_test = final_model.predict(
        Xtest,
        num_iteration=lambda_rounds,
        raw_score=True,
    ).astype(np.float64)
    del Xtr, Xva, Xcombined, combined_y, combined_users
    del order, groups, dfinal, final_model, Xtest

raw_test = np.asarray(raw_test, dtype=np.float64)
if best_alpha < 0.999999:
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_scores = (
        best_alpha * raw_test / best_own_scale
        + (1.0 - best_alpha) * inc_test / inc_scale
    )
else:
    test_scores = raw_test

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
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)