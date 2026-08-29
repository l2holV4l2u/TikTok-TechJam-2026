import os
import time
import json
import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 24681357
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

artifact_dir = os.environ["RUN_ARTIFACTS"]
out_dir = os.environ.get("ITER_OUT")

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type", "hour",
    "onehot_feat1", "onehot_feat2", "onehot_feat3",
    "onehot_feat7", "onehot_feat8", "user_active_degree",
    "register_days_bucket", "fans_user_num_range",
]
STAT_KEYS = [
    ("video", ("video_id",)),
    ("author", ("author_id",)),
    ("tag", ("tag",)),
    ("user_author", ("user_id", "author_id")),
    ("user_tag", ("user_id", "tag")),
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
DIN_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type", "hour",
]
HISTORY_LENGTH = 10
DIN_DIM = 12
DIN_EPOCHS = 2
DIN_BATCH = 8192
PRED_BATCH = 32768


class JoinedSplit:
    pass


def join_splits(a, b):
    z = JoinedSplit()
    z.X = {
        k: np.concatenate([np.asarray(a.X[k]), np.asarray(b.X[k])])
        for k in a.X
    }
    z.num = {
        k: np.concatenate([np.asarray(a.num[k]), np.asarray(b.num[k])])
        for k in a.num
    }
    z.user_id = np.concatenate([
        np.asarray(a.user_id), np.asarray(b.user_id)
    ])
    z.video_id = np.concatenate([
        np.asarray(a.video_id), np.asarray(b.video_id)
    ])
    z.date = np.concatenate([np.asarray(a.date), np.asarray(b.date)])
    z.time_ms = np.concatenate([
        np.asarray(a.time_ms), np.asarray(b.time_ms)
    ])
    return z


def rank_transform(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    su = user_ids[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    pos = np.arange(n) - np.repeat(starts, sizes)
    den = np.maximum(np.repeat(sizes, sizes) - 1, 1)
    ranked = np.where(np.repeat(sizes, sizes) > 1, pos / den, 0.5)
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def packed_key(split, fields):
    key = np.asarray(split.X[fields[0]], dtype=np.int64).copy()
    for f in fields[1:]:
        key = key * int(FEATURE_CARDINALITIES[f]) + np.asarray(
            split.X[f], dtype=np.int64
        )
    return key


def fit_stat_table(split, labels, fields):
    key = packed_key(split, fields)
    uniq, inv, counts = np.unique(
        key, return_inverse=True, return_counts=True
    )
    sums = np.bincount(
        inv, weights=np.asarray(labels, dtype=np.float64),
        minlength=len(uniq)
    )
    return key, uniq, inv, counts.astype(np.float64), sums


def mapped_stats(target, fields, uniq, counts, sums, prior, strength=12.0):
    key = packed_key(target, fields)
    loc = np.searchsorted(uniq, key)
    ok = loc < len(uniq)
    safe = np.minimum(loc, max(len(uniq) - 1, 0))
    ok &= uniq[safe] == key
    c = np.zeros(len(key), dtype=np.float64)
    s = np.zeros(len(key), dtype=np.float64)
    c[ok] = counts[safe[ok]]
    s[ok] = sums[safe[ok]]
    rate = (s + strength * prior) / (c + strength)
    return rate.astype(np.float32), np.log1p(c).astype(np.float32)


def make_stat_features(base, base_y, target, self_training=False):
    prior = float(np.mean(base_y))
    columns = []
    for _, fields in STAT_KEYS:
        base_key, uniq, inv, counts, sums = fit_stat_table(
            base, base_y, fields
        )
        if self_training:
            y = np.asarray(base_y, dtype=np.float64)
            c = counts[inv] - 1.0
            s = sums[inv] - y
            strength = 12.0
            rate = (s + strength * prior) / (c + strength)
            columns.append(rate.astype(np.float32))
            columns.append(np.log1p(np.maximum(c, 0)).astype(np.float32))
        else:
            rate, count = mapped_stats(
                target, fields, uniq, counts, sums, prior
            )
            columns.extend([rate, count])
    return np.column_stack(columns).astype(np.float32)


def make_lgb_matrix(split, stats):
    n = len(split.user_id)
    p = len(CAT_FIELDS) + len(NUM_FIELDS) + stats.shape[1]
    X = np.empty((n, p), dtype=np.float32)
    col = 0
    for f in CAT_FIELDS:
        X[:, col] = np.asarray(split.X[f], dtype=np.float32)
        col += 1
    for f in NUM_FIELDS:
        v = np.asarray(split.num[f], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        X[:, col] = np.log1p(np.maximum(v, 0.0))
        col += 1
    X[:, col:] = stats
    return X


def user_sorted_data(split, X, y=None):
    n = len(split.user_id)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        np.asarray(split.time_ms, dtype=np.int64),
        np.asarray(split.user_id, dtype=np.int64),
    ))
    su = np.asarray(split.user_id)[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    groups = np.diff(np.r_[starts, n]).astype(np.int32)
    ys = None if y is None else np.asarray(y)[order]
    return X[order], ys, groups, order


def fit_lambdarank(train_split, train_y, eval_split=None, eval_y=None,
                   rounds=350):
    tr_stats = make_stat_features(
        train_split, train_y, train_split, self_training=True
    )
    Xtr = make_lgb_matrix(train_split, tr_stats)
    Xtr_s, ytr_s, gtr, _ = user_sorted_data(train_split, Xtr, train_y)

    dtrain = lgb.Dataset(
        Xtr_s, label=ytr_s, group=gtr,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "num_threads": min(16, os.cpu_count() or 1),
        "seed": SEED,
        "verbose": -1,
    }

    valid_sets = None
    callbacks = []
    if eval_split is not None:
        va_stats = make_stat_features(
            train_split, train_y, eval_split, self_training=False
        )
        Xva = make_lgb_matrix(eval_split, va_stats)
        Xva_s, yva_s, gva, _ = user_sorted_data(
            eval_split, Xva, eval_y
        )
        dvalid = lgb.Dataset(
            Xva_s, label=yva_s, group=gva,
            categorical_feature=list(range(len(CAT_FIELDS))),
            reference=dtrain, free_raw_data=True
        )
        valid_sets = [dvalid]
        callbacks = [lgb.early_stopping(35, verbose=False)]

    model = lgb.train(
        params, dtrain, num_boost_round=rounds,
        valid_sets=valid_sets, callbacks=callbacks
    )
    del Xtr, Xtr_s, tr_stats
    gc.collect()
    return model


def predict_lambdarank(model, base, base_y, target):
    stats = make_stat_features(base, base_y, target, self_training=False)
    X = make_lgb_matrix(target, stats)
    pred = model.predict(
        X, num_iteration=model.best_iteration
    ).astype(np.float64)
    del X, stats
    gc.collect()
    return pred


def fit_naive_bayes(base, labels):
    prior = float(np.mean(labels))
    tables = []
    for name, fields in STAT_KEYS:
        _, uniq, _, counts, sums = fit_stat_table(base, labels, fields)
        tables.append((name, fields, uniq, counts, sums))
    return prior, tables


def predict_naive_bayes(model, target):
    prior, tables = model
    prior = np.clip(prior, 1e-5, 1 - 1e-5)
    score = np.full(
        len(target.user_id), math.log(prior / (1 - prior)),
        dtype=np.float64
    )
    weights = {
        "video": 1.0, "author": 0.8, "tag": 0.35,
        "user_author": 1.15, "user_tag": 0.65
    }
    for name, fields, uniq, counts, sums in tables:
        rate, _ = mapped_stats(
            target, fields, uniq, counts, sums, prior, strength=18.0
        )
        p = np.clip(rate.astype(np.float64), 1e-5, 1 - 1e-5)
        evidence = np.log(p / (1 - p)) - math.log(prior / (1 - prior))
        score += weights[name] * evidence
    return score


def causal_positive_history(split, labels):
    n = len(split.user_id)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        np.asarray(split.time_ms, dtype=np.int64),
        np.asarray(split.user_id, dtype=np.int64),
    ))
    y_sorted = np.asarray(labels, dtype=np.int8)[order]
    users_sorted = np.asarray(split.user_id, dtype=np.int64)[order]
    starts = np.r_[0, np.flatnonzero(
        users_sorted[1:] != users_sorted[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])

    cumulative = np.cumsum(y_sorted, dtype=np.int64)
    before_groups = cumulative[starts] - y_sorted[starts]
    base = np.repeat(before_groups, sizes)
    prior_count = cumulative - y_sorted - base

    positive_rows = order[y_sorted == 1]
    videos = np.asarray(split.video_id, dtype=np.int64)
    pad = int(FEATURE_CARDINALITIES["video_id"])
    hs = np.full((n, HISTORY_LENGTH), pad, dtype=np.int64)

    for k in range(1, HISTORY_LENGTH + 1):
        ok = prior_count >= k
        global_pos_index = base[ok] + prior_count[ok] - k
        hs[np.flatnonzero(ok), k - 1] = videos[
            positive_rows[global_pos_index]
        ]

    result = np.empty_like(hs)
    result[order] = hs
    return result


def static_positive_history(base, base_y, target):
    user_card = int(FEATURE_CARDINALITIES["user_id"])
    pad = int(FEATURE_CARDINALITIES["video_id"])
    profile = np.full(
        (user_card, HISTORY_LENGTH), pad, dtype=np.int64
    )

    positive = np.flatnonzero(np.asarray(base_y) == 1)
    if len(positive):
        order = positive[np.lexsort((
            positive,
            np.asarray(base.time_ms)[positive],
            np.asarray(base.user_id)[positive],
        ))]
        users = np.asarray(base.user_id, dtype=np.int64)[order]
        starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
        ends = np.r_[starts[1:], len(order)]
        unique_users = users[starts]
        videos = np.asarray(base.video_id, dtype=np.int64)
        for k in range(1, HISTORY_LENGTH + 1):
            idx = ends - k
            ok = idx >= starts
            profile[unique_users[ok], k - 1] = videos[order[idx[ok]]]

    target_users = np.asarray(target.user_id, dtype=np.int64)
    target_users = np.clip(target_users, 0, user_card - 1)
    return profile[target_users]


DIN_OFFSETS = {}
DIN_TOTAL = 0
for f in DIN_FIELDS:
    DIN_OFFSETS[f] = DIN_TOTAL
    DIN_TOTAL += int(FEATURE_CARDINALITIES[f])


def din_matrix(split):
    X = np.empty((len(split.user_id), len(DIN_FIELDS)), dtype=np.int64)
    for j, f in enumerate(DIN_FIELDS):
        X[:, j] = (
            np.asarray(split.X[f], dtype=np.int64) + DIN_OFFSETS[f]
        )
    return X


class DIN(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.field_emb = nn.Embedding(DIN_TOTAL, DIN_DIM)
        self.linear = nn.Embedding(DIN_TOTAL, 1)
        video_card = int(FEATURE_CARDINALITIES["video_id"])
        self.video_emb = nn.Embedding(
            video_card + 1, DIN_DIM, padding_idx=video_card
        )
        self.attention = nn.Sequential(
            nn.Linear(4 * DIN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        tower_in = len(DIN_FIELDS) * DIN_DIM + 2 * DIN_DIM
        self.tower = nn.Sequential(
            nn.Linear(tower_in, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.tensor(float(intercept)))
        nn.init.normal_(self.field_emb.weight, std=0.025)
        nn.init.normal_(self.video_emb.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history):
        field = self.field_emb(x)
        candidate_video = history.new_tensor(0)
        del candidate_video
        raw_video = x[:, DIN_FIELDS.index("video_id")] - DIN_OFFSETS["video_id"]
        candidate = self.video_emb(raw_video)
        hist = self.video_emb(history)

        cand = candidate[:, None, :].expand_as(hist)
        attention_input = torch.cat([
            cand, hist, cand - hist, cand * hist
        ], dim=2)
        logits = self.attention(attention_input).squeeze(2)
        pad = int(FEATURE_CARDINALITIES["video_id"])
        mask = history != pad
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        interest = (weights[:, :, None] * hist).sum(dim=1)

        tower_input = torch.cat([
            field.flatten(1), candidate, interest
        ], dim=1)
        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(1)
            + self.tower(tower_input).squeeze(1)
        )


def fit_din(split, labels, seed):
    torch.manual_seed(seed)
    X = torch.from_numpy(np.ascontiguousarray(din_matrix(split)))
    H = torch.from_numpy(np.ascontiguousarray(
        causal_positive_history(split, labels)
    ))
    y_np = np.asarray(labels, dtype=np.float32)
    y = torch.from_numpy(y_np)

    p = np.clip(float(y_np.mean()), 1e-5, 1 - 1e-5)
    model = DIN(math.log(p / (1 - p)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0022, weight_decay=1e-6
    )
    generator = torch.Generator().manual_seed(seed + 31)

    model.train()
    for _ in range(DIN_EPOCHS):
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), DIN_BATCH):
            idx = order[start:start + DIN_BATCH]
            logits = model(X[idx], H[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict_din(model, history_base, history_y, target):
    model.eval()
    X = torch.from_numpy(np.ascontiguousarray(din_matrix(target)))
    H = torch.from_numpy(np.ascontiguousarray(
        static_positive_history(history_base, history_y, target)
    ))
    result = np.empty(len(target.user_id), dtype=np.float64)
    for start in range(0, len(result), PRED_BATCH):
        end = min(start + PRED_BATCH, len(result))
        result[start:end] = model(
            X[start:end], H[start:end]
        ).numpy().astype(np.float64)
    return result


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)

inc_valid = np.load(
    os.path.join(artifact_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_rank = rank_transform(valid.user_id, inc_valid)

raw_predictions = {}
candidate_scores = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, valid_y, inc_valid)["primary"]
    )
}

# Family 1: listwise gradient boosting.
ranker = fit_lambdarank(train, train_y, valid, valid_y, rounds=350)
ranker_valid = predict_lambdarank(ranker, train, train_y, valid)
raw_predictions["lambdarank_history"] = ranker_valid
candidate_scores["lambdarank_history_standalone"] = float(
    evaluate(valid.user_id, valid_y, ranker_valid)["primary"]
)
ranker_rounds = int(ranker.best_iteration or 350)
del ranker
gc.collect()

# Family 2: candidate-conditioned positive sequence attention.
din = fit_din(train, train_y, SEED + 100)
din_valid = predict_din(din, train, train_y, valid)
raw_predictions["din_positive_sequence"] = din_valid
candidate_scores["din_positive_sequence_standalone"] = float(
    evaluate(valid.user_id, valid_y, din_valid)["primary"]
)
del din
gc.collect()

# Family 3: non-parametric additive Bayesian evidence.
nb = fit_naive_bayes(train, train_y)
nb_valid = predict_naive_bayes(nb, valid)
raw_predictions["empirical_bayes_additive"] = nb_valid
candidate_scores["empirical_bayes_additive_standalone"] = float(
    evaluate(valid.user_id, valid_y, nb_valid)["primary"]
)

best_name = "trusted_incumbent"
best_alpha = 0.0
best_primary = candidate_scores["trusted_incumbent"]
best_valid = inc_valid.copy()

blend_alphas = [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]
for family, raw in raw_predictions.items():
    rr = rank_transform(valid.user_id, raw)
    for alpha in blend_alphas:
        blended = alpha * rr + (1.0 - alpha) * inc_rank
        metric = evaluate(valid.user_id, valid_y, blended)
        name = family + "_blend_" + str(alpha)
        primary = float(metric["primary"])
        candidate_scores[name] = primary
        if primary > best_primary:
            best_primary = primary
            best_name = family
            best_alpha = alpha
            best_valid = blended.copy()

final_metrics = evaluate(valid.user_id, valid_y, best_valid)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64)
    )

print("FINDINGS " + json.dumps({
    "selected_family": best_name,
    "selected_new_family_weight": best_alpha,
    "lambdarank_best_iteration": ranker_rounds,
}, sort_keys=True))

# Refit only the selected new family on train+validation, then apply the
# validation-selected fixed blend weight to test.
test = load("test")
inc_test = np.load(
    os.path.join(artifact_dir, "incumbent_test_scores.npy")
).astype(np.float64)

if best_name == "trusted_incumbent":
    final_test = inc_test
else:
    joined = join_splits(train, valid)
    joined_y = np.concatenate([
        train_y, valid_y.astype(np.float32)
    ])

    if best_name == "lambdarank_history":
        test_ranker = fit_lambdarank(
            joined, joined_y, eval_split=None, eval_y=None,
            rounds=ranker_rounds
        )
        new_test = predict_lambdarank(
            test_ranker, joined, joined_y, test
        )
    elif best_name == "din_positive_sequence":
        test_din = fit_din(joined, joined_y, SEED + 100)
        new_test = predict_din(
            test_din, joined, joined_y, test
        )
    elif best_name == "empirical_bayes_additive":
        test_nb = fit_naive_bayes(joined, joined_y)
        new_test = predict_naive_bayes(test_nb, test)
    else:
        raise RuntimeError("Unknown selected family")

    if best_alpha >= 1.0:
        final_test = new_test
    else:
        final_test = (
            best_alpha * rank_transform(test.user_id, new_test)
            + (1.0 - best_alpha)
            * rank_transform(test.user_id, inc_test)
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(final_test, dtype=np.float64)
    )

candidate_scores[
    "SELECTED_" + best_name + "_alpha_" + str(best_alpha)
] = float(final_metrics["primary"])
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))