import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
SEED = 27183
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

TREE_FIELDS = [
    "video_id", "author_id", "tab", "duration_bucket", "tag",
    "upload_type", "music_type", "hour", "video_type",
    "onehot_feat2", "onehot_feat3", "onehot_feat7", "onehot_feat8",
]
PNN_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
NUM_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]
NUM_BOOST_ROUND = 160
device = torch.device("cpu")


def recency_weights(dates, half_life=4.0):
    d = np.asarray(dates, dtype=np.float64)
    age = float(np.max(d)) - d
    w = np.exp2(-age / float(half_life))
    w /= max(float(w.mean()), 1e-12)
    return w.astype(np.float32)


def make_tree_features(split_name, split):
    cols = []
    for f in TREE_FIELDS:
        cols.append(np.asarray(split.X[f], dtype=np.float32))

    for f in NUM_FIELDS:
        x = np.asarray(split.num[f], dtype=np.float32)
        finite = np.isfinite(x)
        fill = float(np.nanmedian(x)) if finite.any() else 0.0
        x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
        if f != "duration_ms":
            x = np.sign(x) * np.log1p(np.abs(x))
        else:
            x = np.log1p(np.maximum(x, 0.0))
        cols.append(x.astype(np.float32))

    for key in ("video_id", "author_id"):
        hist = historical_features(split_name, key=key)
        for name in sorted(hist.keys()):
            x = np.asarray(hist[name], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(x)

    return np.column_stack(cols).astype(np.float32, copy=False)


def stable_user_order(user_ids):
    return np.argsort(np.asarray(user_ids), kind="stable")


def group_sizes(sorted_user_ids):
    u = np.asarray(sorted_user_ids)
    if len(u) == 0:
        return np.empty(0, dtype=np.int32)
    cuts = np.flatnonzero(np.r_[True, u[1:] != u[:-1], True])
    return np.diff(cuts).astype(np.int32)


TREE_BASE_PARAMS = {
    "learning_rate": 0.055,
    "num_leaves": 31,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "verbosity": -1,
    "verbose": -1,
    "num_threads": max(1, min(8, os.cpu_count() or 1)),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "force_col_wise": True,
}


def fit_binary_tree(X, y, dates):
    params = dict(TREE_BASE_PARAMS)
    params.update({
        "objective": "binary",
        "metric": "binary_logloss",
    })
    dtrain = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        weight=recency_weights(dates, 4.0),
        categorical_feature=list(range(len(TREE_FIELDS))),
        free_raw_data=True,
    )
    return lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)


def fit_rank_tree(X, y, user_ids):
    order = stable_user_order(user_ids)
    groups = group_sizes(np.asarray(user_ids)[order])
    params = dict(TREE_BASE_PARAMS)
    params.update({
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "lambdarank_truncation_level": 10,
    })
    dtrain = lgb.Dataset(
        X[order],
        label=np.asarray(y, dtype=np.float32)[order],
        group=groups,
        categorical_feature=list(range(len(TREE_FIELDS))),
        free_raw_data=True,
    )
    return lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)


PNN_CARDS = [int(FEATURE_CARDINALITIES[f]) for f in PNN_FIELDS]
PNN_OFFSETS = np.cumsum([0] + PNN_CARDS[:-1], dtype=np.int64)
PNN_TOTAL = int(sum(PNN_CARDS))
PNN_EMBED = 12
PNN_BATCH = 4096
PNN_EPOCHS = 3


def make_pnn_features(split):
    return torch.from_numpy(
        np.stack(
            [
                np.asarray(split.X[f], dtype=np.int64) + PNN_OFFSETS[j]
                for j, f in enumerate(PNN_FIELDS)
            ],
            axis=1,
        )
    )


class PairwisePNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(PNN_TOTAL, PNN_EMBED)
        self.linear = nn.Embedding(PNN_TOTAL, 1)
        pairs = []
        for i in range(len(PNN_FIELDS)):
            for j in range(i + 1, len(PNN_FIELDS)):
                pairs.append((i, j))
        self.register_buffer(
            "pair_i", torch.tensor([p[0] for p in pairs], dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.tensor([p[1] for p in pairs], dtype=torch.long)
        )
        deep_in = len(PNN_FIELDS) * PNN_EMBED + len(pairs)
        self.deep = nn.Sequential(
            nn.Linear(deep_in, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = self.embedding(x)
        products = (e[:, self.pair_i] * e[:, self.pair_j]).sum(dim=2)
        deep_input = torch.cat([e.flatten(1), products], dim=1)
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + wide + self.deep(deep_input).squeeze(-1)


def pair_sampling_state(user_ids, labels):
    uid = np.asarray(user_ids)
    y = np.asarray(labels, dtype=np.int8)
    _, inverse = np.unique(uid, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n_groups)
    positives = np.bincount(inverse, weights=y, minlength=n_groups).astype(np.int64)
    negatives = counts - positives

    row_order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.int64)
    anchors = np.flatnonzero((y == 1) & (negatives[inverse] > 0)).astype(np.int64)
    return inverse, counts, starts, row_order, anchors


def sample_negative_rows(state, labels, rng):
    inverse, counts, starts, row_order, anchors = state
    groups = inverse[anchors]
    offsets = (rng.random(len(anchors)) * counts[groups]).astype(np.int64)
    negatives = row_order[starts[groups] + offsets]
    bad = np.asarray(labels, dtype=np.int8)[negatives] != 0
    while bad.any():
        gb = groups[bad]
        offsets = (rng.random(int(bad.sum())) * counts[gb]).astype(np.int64)
        negatives[bad] = row_order[starts[gb] + offsets]
        bad = np.asarray(labels, dtype=np.int8)[negatives] != 0
    return anchors, negatives


def fit_pairwise_pnn(x, labels, user_ids, dates, seed):
    torch.manual_seed(seed)
    model = PairwisePNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    labels_np = np.asarray(labels, dtype=np.int8)
    state = pair_sampling_state(user_ids, labels_np)
    weights = torch.from_numpy(recency_weights(dates, 4.0))
    rng = np.random.default_rng(seed + 91)
    shuffle_gen = torch.Generator(device="cpu")
    shuffle_gen.manual_seed(seed + 117)

    model.train()
    for _ in range(PNN_EPOCHS):
        anchors, negatives = sample_negative_rows(state, labels_np, rng)
        order = torch.randperm(len(anchors), generator=shuffle_gen)
        anchors_t = torch.from_numpy(anchors)
        negatives_t = torch.from_numpy(negatives)

        for start in range(0, len(order), PNN_BATCH):
            idx = order[start:start + PNN_BATCH]
            pidx = anchors_t[idx]
            nidx = negatives_t[idx]
            xp = x[pidx].to(device)
            xn = x[nidx].to(device)
            wb = weights[pidx].to(device)

            optimizer.zero_grad(set_to_none=True)
            margin = model(xp) - model(xn)
            loss = (nn.functional.softplus(-margin) * wb).mean()
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict_pnn(model, x, batch_size=16384):
    model.eval()
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        logits = model(x[start:end].to(device))
        out[start:end] = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    return out


def within_user_percentile(user_ids, scores):
    uid = np.asarray(user_ids)
    s = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(len(s), dtype=np.int64), s, uid))
    sorted_uid = uid[order]

    change = np.r_[True, sorted_uid[1:] != sorted_uid[:-1]]
    starts = np.maximum.accumulate(
        np.where(change, np.arange(len(s), dtype=np.int64), 0)
    )
    positions = np.arange(len(s), dtype=np.int64) - starts

    end_change = np.r_[sorted_uid[:-1] != sorted_uid[1:], True]
    ends = np.minimum.accumulate(
        np.where(end_change, np.arange(len(s), dtype=np.int64), len(s) - 1)[::-1]
    )[::-1]
    lengths = ends - starts + 1

    ranked = np.where(
        lengths > 1,
        positions.astype(np.float64) / np.maximum(lengths - 1, 1),
        0.5,
    )
    out = np.empty(len(s), dtype=np.float64)
    out[order] = ranked
    return out


train = load("train")
valid = load("valid")

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise RuntimeError("Trusted incumbent validation predictions are unavailable")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

X_train_tree = make_tree_features("train", train)
X_valid_tree = make_tree_features("valid", valid)
y_train = np.asarray(train.y, dtype=np.int8)

raw_predictions = {}
models = {}

binary_model = fit_binary_tree(X_train_tree, y_train, train.date)
raw_predictions["binary_tree_recency4"] = np.asarray(
    binary_model.predict(X_valid_tree), dtype=np.float64
)
models["binary_tree_recency4"] = binary_model

rank_model = fit_rank_tree(X_train_tree, y_train, train.user_id)
raw_predictions["lambdarank_tree"] = np.asarray(
    rank_model.predict(X_valid_tree), dtype=np.float64
)
models["lambdarank_tree"] = rank_model

x_train_pnn = make_pnn_features(train)
x_valid_pnn = make_pnn_features(valid)
pnn_model = fit_pairwise_pnn(
    x_train_pnn, y_train, train.user_id, train.date, SEED + 500
)
raw_predictions["pairwise_pnn"] = predict_pnn(pnn_model, x_valid_pnn)
models["pairwise_pnn"] = pnn_model

candidate_scores = {}
candidate_predictions = {}
candidate_recipes = {}

inc_metric = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_recipes["incumbent"] = {
    "family": "incumbent", "mode": "raw", "alpha": 0.0
}

inc_rank = within_user_percentile(valid.user_id, inc_valid)

for name, pred in raw_predictions.items():
    met = evaluate(valid.user_id, valid.y, pred)
    candidate_scores[name] = float(met["primary"])
    candidate_predictions[name] = pred
    candidate_recipes[name] = {
        "family": name, "mode": "raw", "alpha": 1.0
    }

    pred_rank = within_user_percentile(valid.user_id, pred)
    rank_name = name + "_rank"
    rank_met = evaluate(valid.user_id, valid.y, pred_rank)
    candidate_scores[rank_name] = float(rank_met["primary"])
    candidate_predictions[rank_name] = pred_rank
    candidate_recipes[rank_name] = {
        "family": name, "mode": "rank", "alpha": 1.0
    }

    for alpha in (0.2, 0.4, 0.6, 0.8):
        raw_name = "%s_rawblend_%.1f" % (name, alpha)
        raw_blend = alpha * pred + (1.0 - alpha) * inc_valid
        raw_met = evaluate(valid.user_id, valid.y, raw_blend)
        candidate_scores[raw_name] = float(raw_met["primary"])
        candidate_predictions[raw_name] = raw_blend
        candidate_recipes[raw_name] = {
            "family": name, "mode": "raw", "alpha": alpha
        }

        rb_name = "%s_rankblend_%.1f" % (name, alpha)
        rank_blend = alpha * pred_rank + (1.0 - alpha) * inc_rank
        rank_met = evaluate(valid.user_id, valid.y, rank_blend)
        candidate_scores[rb_name] = float(rank_met["primary"])
        candidate_predictions[rb_name] = rank_blend
        candidate_recipes[rb_name] = {
            "family": name, "mode": "rank", "alpha": alpha
        }

winner = max(candidate_scores, key=candidate_scores.get)
recipe = candidate_recipes[winner]
valid_scores = np.asarray(candidate_predictions[winner], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS standalone binary=%.6f lambdarank=%.6f pairwise_pnn=%.6f winner=%s"
    % (
        candidate_scores["binary_tree_recency4"],
        candidate_scores["lambdarank_tree"],
        candidate_scores["pairwise_pnn"],
        winner,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "scores_valid.npy"), valid_scores)

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
family = recipe["family"]
alpha = float(recipe["alpha"])
mode = recipe["mode"]

if family == "incumbent":
    test_scores = inc_test.copy()
else:
    y_valid = np.asarray(valid.y, dtype=np.int8)
    y_combined = np.concatenate([y_train, y_valid])
    combined_dates = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])
    combined_users = np.concatenate([
        np.asarray(train.user_id),
        np.asarray(valid.user_id),
    ])

    if family in ("binary_tree_recency4", "lambdarank_tree"):
        X_test_tree = make_tree_features("test", test)
        X_combined_tree = np.concatenate([X_train_tree, X_valid_tree], axis=0)

        if family == "binary_tree_recency4":
            final_model = fit_binary_tree(
                X_combined_tree, y_combined, combined_dates
            )
        else:
            final_model = fit_rank_tree(
                X_combined_tree, y_combined, combined_users
            )
        family_test = np.asarray(
            final_model.predict(X_test_tree), dtype=np.float64
        )
    elif family == "pairwise_pnn":
        x_test_pnn = make_pnn_features(test)
        x_combined_pnn = torch.cat([x_train_pnn, x_valid_pnn], dim=0)
        final_model = fit_pairwise_pnn(
            x_combined_pnn,
            y_combined,
            combined_users,
            combined_dates,
            SEED + 500,
        )
        family_test = predict_pnn(final_model, x_test_pnn)
    else:
        raise ValueError(family)

    if mode == "rank":
        family_component = within_user_percentile(test.user_id, family_test)
        incumbent_component = within_user_percentile(test.user_id, inc_test)
    else:
        family_component = family_test
        incumbent_component = inc_test

    test_scores = (
        alpha * family_component + (1.0 - alpha) * incumbent_component
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)