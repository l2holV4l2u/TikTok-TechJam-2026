import os
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket", "tag",
    "upload_type", "music_type", "hour", "user_active_degree",
    "is_video_author", "video_type", "onehot_feat3", "onehot_feat7",
    "onehot_feat8", "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_range",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
BPR_FIELDS = [
    "user_id", "video_id", "author_id", "tag", "tab",
    "duration_bucket", "hour", "upload_type",
]
N_BPR = len(BPR_FIELDS)
BPR_K = 20
BPR_BATCH = 16384
BPR_EPOCHS = 4
PRED_BATCH = 65536


def date_ordinal(dates):
    dates = np.asarray(dates, dtype=np.int32)
    # All dates are in April/May 2022, so this monotone encoding is sufficient.
    month = dates // 100 % 100
    day = dates % 100
    return (month * 31 + day).astype(np.float32)


def recency_weights(dates, half_life, endpoint=None):
    ords = date_ordinal(dates)
    if endpoint is None:
        endpoint = float(ords.max())
    if half_life >= 1e6:
        return np.ones(len(ords), dtype=np.float32)
    age = np.maximum(0.0, endpoint - ords)
    return np.exp2(-age / float(half_life)).astype(np.float32)


def make_lgb_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(CAT_FIELDS) + len(NUM_FIELDS) + 2), dtype=np.float32)
    for j, f in enumerate(CAT_FIELDS):
        x[:, j] = np.asarray(split.X[f], dtype=np.float32)

    base = len(CAT_FIELDS)
    for j, f in enumerate(NUM_FIELDS):
        v = np.asarray(split.num[f], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        x[:, base + j] = np.log1p(np.maximum(v, 0.0))

    x[:, base + len(NUM_FIELDS)] = np.asarray(split.X["hour"], dtype=np.float32)
    x[:, base + len(NUM_FIELDS) + 1] = date_ordinal(split.date)
    return x


def sorted_query_data(x, y, users, weights=None):
    users = np.asarray(users, dtype=np.int64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    _, groups = np.unique(sorted_users, return_counts=True)
    sw = None if weights is None else np.asarray(weights, dtype=np.float32)[order]
    return x[order], np.asarray(y)[order], groups.astype(np.int32), sw, order


def fit_lambdarank(x_train, y_train, train_users, train_dates,
                   x_valid, y_valid, valid_users, half_life):
    weights = recency_weights(train_dates, half_life)
    xt, yt, gt, wt, _ = sorted_query_data(
        x_train, y_train, train_users, weights
    )
    xv, yv, gv, _, valid_order = sorted_query_data(
        x_valid, y_valid, valid_users
    )

    dtrain = lgb.Dataset(
        xt,
        label=yt,
        group=gt,
        weight=wt,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    dvalid = lgb.Dataset(
        xv,
        label=yv,
        group=gv,
        categorical_feature=list(range(len(CAT_FIELDS))),
        reference=dtrain,
        free_raw_data=True,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "verbose": -1,
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=220,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    pred_sorted = model.predict(xv, num_iteration=model.best_iteration)
    pred = np.empty(len(pred_sorted), dtype=np.float32)
    pred[valid_order] = pred_sorted.astype(np.float32)
    return pred, int(model.best_iteration), params


def refit_lambdarank(x, y, users, dates, half_life, params, rounds):
    weights = recency_weights(dates, half_life)
    xs, ys, groups, ws, _ = sorted_query_data(x, y, users, weights)
    dtrain = lgb.Dataset(
        xs,
        label=ys,
        group=groups,
        weight=ws,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    return lgb.train(params, dtrain, num_boost_round=max(1, int(rounds)))


BPR_CARDS = [int(FEATURE_CARDINALITIES[f]) for f in BPR_FIELDS]
BPR_OFFSETS = np.cumsum([0] + BPR_CARDS[:-1], dtype=np.int64)
BPR_TOTAL = int(sum(BPR_CARDS))


def make_bpr_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, N_BPR), dtype=np.int64)
    for j, f in enumerate(BPR_FIELDS):
        x[:, j] = np.asarray(split.X[f], dtype=np.int64) + BPR_OFFSETS[j]
    return x


class PairwiseFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(BPR_TOTAL, 1)
        self.embedding = nn.Embedding(BPR_TOTAL, BPR_K)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.015)

    def forward(self, x):
        lin = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        sv = v.sum(dim=1)
        inter = 0.5 * (sv.square() - v.square().sum(dim=1)).sum(dim=1)
        return lin + inter


def build_negative_sampler(users, labels):
    users = np.asarray(users, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    neg_rows = np.flatnonzero(labels == 0)
    neg_users = users[neg_rows]
    order = np.argsort(neg_users, kind="stable")
    neg_rows = neg_rows[order]
    neg_users = neg_users[order]

    max_uid = int(users.max()) + 1
    counts = np.bincount(neg_users, minlength=max_uid).astype(np.int64)
    starts = np.zeros(max_uid, dtype=np.int64)
    if max_uid > 1:
        starts[1:] = np.cumsum(counts[:-1])
    return neg_rows, starts, counts


def train_bpr(x, y, users, dates, epochs=BPR_EPOCHS):
    model = PairwiseFM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    y = np.asarray(y, dtype=np.int8)
    users = np.asarray(users, dtype=np.int64)

    neg_rows, neg_starts, neg_counts = build_negative_sampler(users, y)
    pos_rows = np.flatnonzero((y == 1) & (neg_counts[users] > 0))
    pair_weights = recency_weights(dates, 7.0)[pos_rows]

    xt = torch.from_numpy(x)
    rng = np.random.default_rng(SEED + 91)

    for epoch in range(epochs):
        order = rng.permutation(len(pos_rows))
        model.train()
        for start in range(0, len(order), BPR_BATCH):
            ii = order[start:start + BPR_BATCH]
            pr = pos_rows[ii]
            pu = users[pr]
            cnt = neg_counts[pu]
            offset = (rng.random(len(ii)) * cnt).astype(np.int64)
            nr = neg_rows[neg_starts[pu] + offset]

            pos_x = xt[torch.from_numpy(pr)]
            neg_x = xt[torch.from_numpy(nr)]
            w = torch.from_numpy(pair_weights[ii])

            optimizer.zero_grad(set_to_none=True)
            margin = model(pos_x) - model(neg_x)
            loss = (-nn.functional.logsigmoid(margin) * w).mean()
            reg = 1e-7 * (
                model.embedding(pos_x).square().mean()
                + model.embedding(neg_x).square().mean()
            )
            (loss + reg).backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_bpr(model, x):
    model.eval()
    xt = torch.from_numpy(x)
    out = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), PRED_BATCH):
        end = min(start + PRED_BATCH, len(x))
        out[start:end] = model(xt[start:end]).cpu().numpy().astype(np.float32)
    return out


def weighted_entity_component(train_ids, labels, weights, query_ids,
                              cardinality, smoothing):
    train_ids = np.asarray(train_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    count = np.bincount(
        train_ids, weights=weights, minlength=cardinality
    ).astype(np.float64)
    pos = np.bincount(
        train_ids, weights=weights * labels, minlength=cardinality
    ).astype(np.float64)
    prior = float(np.sum(weights * labels) / np.maximum(np.sum(weights), 1.0))
    rate = (pos + smoothing * prior) / (count + smoothing)
    rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
    return np.log(rate[query_ids] / (1.0 - rate[query_ids])).astype(np.float32)


def eb_scores(fit_split, fit_y, query_split, half_life, smoothing, entities):
    weights = recency_weights(fit_split.date, half_life)
    parts = []
    for f in entities:
        parts.append(weighted_entity_component(
            fit_split.X[f],
            fit_y,
            weights,
            query_split.X[f],
            int(FEATURE_CARDINALITIES[f]),
            smoothing,
        ))
    return np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return ((x - float(np.mean(x))) / sd).astype(np.float32)


def best_incumbent_blend(candidate, incumbent, users, labels):
    cz = standardized(candidate)
    iz = standardized(incumbent)
    best_score = -np.inf
    best_alpha = 1.0
    best_pred = cz
    for alpha in np.linspace(0.0, 1.0, 21):
        pred = alpha * cz + (1.0 - alpha) * iz
        score = float(evaluate(users, labels, pred)["primary"])
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_pred = pred.copy()
    return best_score, best_alpha, best_pred


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float32)

candidate_log = {}
family_records = []

# Family 1: query-level LambdaRank, with temporal weighting selected internally.
xtr_lgb = make_lgb_matrix(train)
xva_lgb = make_lgb_matrix(valid)
for half_life in (3.0, 7.0, 1e9):
    pred, rounds, params = fit_lambdarank(
        xtr_lgb, y_train, train.user_id, train.date,
        xva_lgb, y_valid, valid.user_id, half_life,
    )
    raw = float(evaluate(valid.user_id, y_valid, pred)["primary"])
    blend_score, alpha, blend_pred = best_incumbent_blend(
        pred, inc_valid, valid.user_id, y_valid
    )
    name = "lambdarank_hl_" + ("inf" if half_life >= 1e6 else str(int(half_life)))
    candidate_log[name] = raw
    candidate_log[name + "_blend"] = blend_score
    family_records.append({
        "name": name,
        "family": "lambdarank",
        "primary": blend_score,
        "alpha": alpha,
        "valid_scores": blend_pred,
        "half_life": half_life,
        "rounds": rounds,
        "params": params,
    })

# Family 2: pairwise BPR-trained factorization.
xtr_bpr = make_bpr_matrix(train)
xva_bpr = make_bpr_matrix(valid)
bpr_model = train_bpr(
    xtr_bpr, y_train, train.user_id, train.date, BPR_EPOCHS
)
bpr_pred = predict_bpr(bpr_model, xva_bpr)
bpr_raw = float(evaluate(valid.user_id, y_valid, bpr_pred)["primary"])
bpr_blend_score, bpr_alpha, bpr_blend = best_incumbent_blend(
    bpr_pred, inc_valid, valid.user_id, y_valid
)
candidate_log["pairwise_bpr"] = bpr_raw
candidate_log["pairwise_bpr_blend"] = bpr_blend_score
family_records.append({
    "name": "pairwise_bpr",
    "family": "bpr",
    "primary": bpr_blend_score,
    "alpha": bpr_alpha,
    "valid_scores": bpr_blend,
    "epochs": BPR_EPOCHS,
})
del bpr_model
gc.collect()

# Family 3: non-parametric recency-weighted empirical Bayes.
entity_sets = [
    ("video_id",),
    ("author_id",),
    ("video_id", "author_id"),
    ("video_id", "author_id", "tag"),
    ("author_id", "tag", "duration_bucket"),
]
for half_life in (3.0, 7.0, 1e9):
    for smoothing in (10.0, 40.0, 120.0):
        for entities in entity_sets:
            pred = eb_scores(
                train, y_train, valid, half_life, smoothing, entities
            )
            raw = float(evaluate(valid.user_id, y_valid, pred)["primary"])
            blend_score, alpha, blend_pred = best_incumbent_blend(
                pred, inc_valid, valid.user_id, y_valid
            )
            suffix = "_".join(entities)
            hname = "inf" if half_life >= 1e6 else str(int(half_life))
            name = f"eb_hl{hname}_s{int(smoothing)}_{suffix}"
            candidate_log[name] = raw
            candidate_log[name + "_blend"] = blend_score
            family_records.append({
                "name": name,
                "family": "eb",
                "primary": blend_score,
                "alpha": alpha,
                "valid_scores": blend_pred,
                "half_life": half_life,
                "smoothing": smoothing,
                "entities": entities,
            })

winner = max(family_records, key=lambda z: z["primary"])
valid_scores = np.asarray(winner["valid_scores"], dtype=np.float32)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected": winner["name"],
    "selected_family": winner["family"],
    "incumbent_weight": float(1.0 - winner["alpha"]),
    "new_model_weight": float(winner["alpha"]),
    "lambdarank_best_rounds": {
        r["name"]: int(r["rounds"])
        for r in family_records if r["family"] == "lambdarank"
    },
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the selected recipe on train + validation, then score test.
test = load("test")
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float32)

alpha = float(winner["alpha"])
combined_y = np.concatenate([
    y_train.astype(np.int8),
    y_valid.astype(np.int8),
])
combined_users = np.concatenate([
    np.asarray(train.user_id, dtype=np.int64),
    np.asarray(valid.user_id, dtype=np.int64),
])
combined_dates = np.concatenate([
    np.asarray(train.date, dtype=np.int32),
    np.asarray(valid.date, dtype=np.int32),
])

if winner["family"] == "lambdarank":
    x_combined = np.concatenate([xtr_lgb, xva_lgb], axis=0)
    x_test = make_lgb_matrix(test)
    final_model = refit_lambdarank(
        x_combined,
        combined_y,
        combined_users,
        combined_dates,
        winner["half_life"],
        winner["params"],
        winner["rounds"],
    )
    new_test = final_model.predict(
        x_test, num_iteration=winner["rounds"]
    ).astype(np.float32)

elif winner["family"] == "bpr":
    x_combined = np.concatenate([xtr_bpr, xva_bpr], axis=0)
    x_test = make_bpr_matrix(test)
    final_model = train_bpr(
        x_combined,
        combined_y,
        combined_users,
        combined_dates,
        winner["epochs"],
    )
    new_test = predict_bpr(final_model, x_test)

else:
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.date = combined_dates
    combined.X = {}
    for f in winner["entities"]:
        combined.X[f] = np.concatenate([
            np.asarray(train.X[f], dtype=np.int64),
            np.asarray(valid.X[f], dtype=np.int64),
        ])
    new_test = eb_scores(
        combined,
        combined_y,
        test,
        winner["half_life"],
        winner["smoothing"],
        winner["entities"],
    )

test_scores = (
    alpha * standardized(new_test)
    + (1.0 - alpha) * standardized(inc_test)
).astype(np.float32)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))