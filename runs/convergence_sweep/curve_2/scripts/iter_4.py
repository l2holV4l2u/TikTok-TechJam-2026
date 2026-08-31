import os
import time
import json
import math
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18431
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "register_days_bucket",
    "music_type",
    "video_type",
]
RAW_NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-10:
        sd = 1.0
    return np.clip((x - float(np.mean(x))) / sd, -8.0, 8.0)


# -------------------------------------------------------------------------
# Shared tabular inputs for the querywise tree model.
# -------------------------------------------------------------------------
def history_dict(split_name):
    result = {}
    for entity in ("video_id", "author_id"):
        d = historical_features(split_name, key=entity)
        for name, value in d.items():
            result[entity + "__" + name] = np.asarray(value, dtype=np.float32)
    return result


htr = history_dict("train")
hva = history_dict("valid")
hte = history_dict("test")
common_hist = sorted(set(htr) & set(hva) & set(hte))
hist_keys = [
    k for k in common_hist
    if (
        "train_count" in k
        or "long_view_rate" in k
        or "is_like_rate" in k
        or "is_follow_rate" in k
        or "is_hate_rate" in k
    )
]
if not hist_keys:
    hist_keys = common_hist[:10]


def make_tree_matrix(split, hd):
    columns = []
    for name in CAT_FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.float32))

    for name in RAW_NUM_FIELDS:
        a = np.asarray(split.num[name], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(a, 0.0)).astype(np.float32))

    for key in hist_keys:
        a = np.asarray(hd[key], dtype=np.float32)
        columns.append(
            np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        )

    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)


Xtr = make_tree_matrix(train, htr)
Xva = make_tree_matrix(valid, hva)
Xte = make_tree_matrix(test, hte)

# Sorting makes each user's impressions one LambdaRank query.
tr_order = np.argsort(np.asarray(train.user_id), kind="stable")
tr_users_sorted = np.asarray(train.user_id)[tr_order]
tr_change = np.r_[True, tr_users_sorted[1:] != tr_users_sorted[:-1]]
tr_starts = np.flatnonzero(tr_change)
tr_groups = np.diff(np.r_[tr_starts, len(tr_order)]).astype(np.int32)

# Recency weighting addresses the date shift without using validation outcomes.
max_train_date = int(np.max(np.asarray(train.date)))
age = (max_train_date - np.asarray(train.date, dtype=np.int32)).astype(np.float32)
rank_weights = np.exp(-math.log(2.0) * age / 5.0).astype(np.float32)
rank_weights /= rank_weights.mean()

rank_set = lgb.Dataset(
    Xtr[tr_order],
    label=y_train[tr_order],
    weight=rank_weights[tr_order],
    group=tr_groups,
    categorical_feature=list(range(len(CAT_FIELDS))),
    free_raw_data=True,
)

rank_params = {
    "objective": "lambdarank",
    "metric": "None",
    "lambdarank_truncation_level": 5,
    "label_gain": [0, 1],
    "learning_rate": 0.045,
    "num_leaves": 63,
    "min_data_in_leaf": 120,
    "max_bin": 127,
    "feature_fraction": 0.86,
    "bagging_fraction": 0.86,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.0,
    "max_depth": -1,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "verbose": -1,
}

rank_model = lgb.train(rank_params, rank_set, num_boost_round=240)
rank_valid = rank_model.predict(Xva).astype(np.float32)
rank_test = rank_model.predict(Xte).astype(np.float32)

del rank_model, rank_set, Xtr, Xva, Xte, htr, hva, hte
gc.collect()


# -------------------------------------------------------------------------
# DIN: attend to each user's most recent train-only positive videos.
# Train rows receive strictly preceding positives in timestamp/row order.
# Validation and test receive the complete train history only.
# -------------------------------------------------------------------------
HISTORY_LEN = 12
user_card = int(FEATURE_CARDINALITIES["user_id"])
video_card = int(FEATURE_CARDINALITIES["video_id"])

tr_uid = np.asarray(train.X["user_id"], dtype=np.int64)
tr_vid = np.asarray(train.X["video_id"], dtype=np.int64)
row_position = np.arange(len(y_train), dtype=np.int64)
chron_order = np.lexsort(
    (row_position, np.asarray(train.time_ms, dtype=np.int64), tr_uid)
)

ord_uid = tr_uid[chron_order]
ord_vid = tr_vid[chron_order]
ord_y = y_train[chron_order].astype(np.int8)

positive_totals = np.bincount(
    ord_uid, weights=ord_y, minlength=user_card
).astype(np.int64)
positive_base = np.zeros(user_card, dtype=np.int64)
positive_base[1:] = np.cumsum(positive_totals[:-1])

packed_positive_videos = ord_vid[ord_y == 1].astype(np.int32)
packed_with_sentinel = np.concatenate(
    [np.zeros(1, dtype=np.int32), packed_positive_videos]
)

global_positive_cumsum = np.cumsum(ord_y, dtype=np.int64)
group_start = np.maximum.accumulate(
    np.where(
        np.r_[True, ord_uid[1:] != ord_uid[:-1]],
        np.arange(len(ord_uid), dtype=np.int64),
        0,
    )
)
positives_before_group = (
    global_positive_cumsum[group_start] - ord_y[group_start]
)
previous_positive_count = (
    global_positive_cumsum - ord_y - positives_before_group
)


def materialize_history(user_ids, counts):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    back = np.arange(1, HISTORY_LEN + 1, dtype=np.int64)[None, :]
    index = (
        1
        + positive_base[user_ids, None]
        + counts[:, None]
        - back
    )
    ok = back <= counts[:, None]
    index = np.where(ok, index, 0)
    return np.ascontiguousarray(
        packed_with_sentinel[index], dtype=np.int32
    )


history_train_ord = materialize_history(ord_uid, previous_positive_count)
history_train = np.empty_like(history_train_ord)
history_train[chron_order] = history_train_ord
del history_train_ord

va_uid = np.asarray(valid.X["user_id"], dtype=np.int64)
te_uid = np.asarray(test.X["user_id"], dtype=np.int64)
history_valid = materialize_history(va_uid, positive_totals[va_uid])
history_test = materialize_history(te_uid, positive_totals[te_uid])

DIN_CONTEXT = [
    "user_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "hour",
]
din_cards = [int(FEATURE_CARDINALITIES[x]) for x in DIN_CONTEXT]
din_offsets = np.cumsum([0] + din_cards[:-1], dtype=np.int64)
din_total = int(sum(din_cards))


def make_din_context(split):
    return np.ascontiguousarray(
        np.stack(
            [np.asarray(split.X[x], dtype=np.int64) for x in DIN_CONTEXT],
            axis=1,
        ),
        dtype=np.int64,
    )


din_cat_train = make_din_context(train)
din_cat_valid = make_din_context(valid)
din_cat_test = make_din_context(test)
din_video_train = np.asarray(train.X["video_id"], dtype=np.int64)
din_video_valid = np.asarray(valid.X["video_id"], dtype=np.int64)
din_video_test = np.asarray(test.X["video_id"], dtype=np.int64)

DIN_DIM = 12


class DIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "context_offsets", torch.from_numpy(din_offsets.copy())
        )
        self.context_embedding = nn.Embedding(din_total, DIN_DIM)
        self.video_embedding = nn.Embedding(
            video_card, DIN_DIM, padding_idx=0
        )
        self.attention = nn.Sequential(
            nn.Linear(4 * DIN_DIM, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        input_dim = len(DIN_CONTEXT) * DIN_DIM + 3 * DIN_DIM
        self.output = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 48),
            nn.PReLU(),
            nn.Linear(48, 1),
        )
        nn.init.normal_(self.context_embedding.weight, std=0.025)
        nn.init.normal_(self.video_embedding.weight, std=0.025)
        with torch.no_grad():
            self.video_embedding.weight[0].zero_()

    def forward(self, context, video, history):
        context_emb = self.context_embedding(
            context + self.context_offsets
        ).flatten(1)
        query = self.video_embedding(video)
        hist = self.video_embedding(history)
        q = query.unsqueeze(1).expand_as(hist)

        attention_input = torch.cat(
            [hist, q, hist * q, hist - q], dim=2
        )
        attention_logits = self.attention(attention_input).squeeze(2)
        mask = history.ne(0)
        attention_logits = attention_logits.masked_fill(mask.logical_not(), -1e4)
        weights = torch.softmax(attention_logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        interest = torch.sum(weights.unsqueeze(2) * hist, dim=1)

        z = torch.cat(
            [context_emb, query, interest, query * interest], dim=1
        )
        return self.output(z).squeeze(1)


@torch.no_grad()
def din_predict(model, context, video, history, batch_size=16384):
    model.eval()
    out = np.empty(len(video), dtype=np.float32)
    for begin in range(0, len(video), batch_size):
        end = min(begin + batch_size, len(video))
        out[begin:end] = model(
            torch.from_numpy(context[begin:end]),
            torch.from_numpy(video[begin:end]),
            torch.from_numpy(history[begin:end].astype(np.int64, copy=False)),
        ).cpu().numpy()
    return out


din_model = DIN()
optimizer = torch.optim.AdamW(
    din_model.parameters(), lr=1.1e-3, weight_decay=2e-6
)

tx_context = torch.from_numpy(din_cat_train)
tx_video = torch.from_numpy(din_video_train)
tx_history = torch.from_numpy(history_train.astype(np.int64))
tx_y = torch.from_numpy(y_train)
tx_weight = torch.from_numpy(rank_weights)

generator = torch.Generator()
generator.manual_seed(SEED + 100)
din_epoch_scores = []
best_din_score = -np.inf
best_din_state = None
best_din_valid = None

for epoch in range(3):
    din_model.train()
    permutation = torch.randperm(len(y_train), generator=generator)

    for begin in range(0, len(y_train), 4096):
        idx = permutation[begin:begin + 4096]
        logits = din_model(
            tx_context[idx], tx_video[idx], tx_history[idx]
        )
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, tx_y[idx], reduction="none"
        )
        loss = (losses * tx_weight[idx]).sum() / tx_weight[idx].sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(din_model.parameters(), 5.0)
        optimizer.step()

    va_scores = din_predict(
        din_model, din_cat_valid, din_video_valid, history_valid
    )
    va_metric = evaluate(valid.user_id, valid.y, va_scores)
    score = float(va_metric["primary"])
    din_epoch_scores.append(score)

    if score > best_din_score:
        best_din_score = score
        best_din_valid = va_scores.copy()
        best_din_state = {
            k: v.detach().cpu().clone()
            for k, v in din_model.state_dict().items()
        }

din_model.load_state_dict(best_din_state)
din_valid = best_din_valid
din_test = din_predict(
    din_model, din_cat_test, din_video_test, history_test
)

del (
    din_model,
    optimizer,
    best_din_state,
    tx_context,
    tx_video,
    tx_history,
    tx_y,
    tx_weight,
    history_train,
    history_valid,
    history_test,
)
gc.collect()


# -------------------------------------------------------------------------
# Latent collaborative family: SVD of train-only positive user-video counts.
# It deliberately ignores side features and estimates a distinct preference
# geometry, making it potentially complementary even if weak standalone.
# -------------------------------------------------------------------------
positive_mask = y_train > 0.5
svd_matrix = sparse.coo_matrix(
    (
        np.ones(int(positive_mask.sum()), dtype=np.float32),
        (tr_uid[positive_mask], tr_vid[positive_mask]),
    ),
    shape=(user_card, video_card),
).tocsr()
svd_matrix.data = np.log1p(svd_matrix.data).astype(np.float32)

try:
    U, S, VT = svds(
        svd_matrix.astype(np.float32),
        k=32,
        which="LM",
        random_state=SEED,
        return_singular_vectors=True,
    )
    order = np.argsort(S)[::-1]
    S = S[order].astype(np.float32)
    U = U[:, order].astype(np.float32)
    VT = VT[order].astype(np.float32)
    user_latent = U * np.sqrt(S)[None, :]
    video_latent = VT.T * np.sqrt(S)[None, :]

    def svd_score(split):
        u = np.asarray(split.X["user_id"], dtype=np.int64)
        v = np.asarray(split.X["video_id"], dtype=np.int64)
        return np.sum(user_latent[u] * video_latent[v], axis=1).astype(np.float32)

    svd_valid = svd_score(valid)
    svd_test = svd_score(test)
except Exception as exc:
    print("FINDINGS svd_failure=" + repr(exc))
    svd_valid = np.zeros(len(va_uid), dtype=np.float32)
    svd_test = np.zeros(len(te_uid), dtype=np.float32)

del svd_matrix
gc.collect()


# -------------------------------------------------------------------------
# Evaluate raw families and every family/incumbent blend.
# -------------------------------------------------------------------------
raw_valid = {
    "lambdarank": rank_valid,
    "din": din_valid,
    "latent_svd": svd_valid,
}
raw_test = {
    "lambdarank": rank_test,
    "din": din_test,
    "latent_svd": svd_test,
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
have_incumbent = (
    os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)
)

candidates = {}
payloads = {}

for name in raw_valid:
    metrics = evaluate(valid.user_id, valid.y, raw_valid[name])
    candidates[name] = float(metrics["primary"])
    payloads[name] = {
        "valid": np.asarray(raw_valid[name], dtype=np.float64),
        "test": np.asarray(raw_test[name], dtype=np.float64),
        "raw": np.asarray(raw_valid[name], dtype=np.float64),
        "metrics": metrics,
    }

if have_incumbent:
    incumbent_valid = np.asarray(
        np.load(inc_valid_path), dtype=np.float64
    )
    incumbent_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
    zinc_valid = standardized(incumbent_valid)
    zinc_test = standardized(incumbent_test)

    for name in raw_valid:
        zraw_valid = standardized(raw_valid[name])
        zraw_test = standardized(raw_test[name])

        for alpha in np.linspace(0.10, 0.90, 17):
            alpha = float(alpha)
            blend_valid = (
                (1.0 - alpha) * zinc_valid + alpha * zraw_valid
            )
            blend_test = (
                (1.0 - alpha) * zinc_test + alpha * zraw_test
            )
            metrics = evaluate(valid.user_id, valid.y, blend_valid)
            blend_name = "%s_inc_a%.2f" % (name, alpha)
            candidates[blend_name] = float(metrics["primary"])
            payloads[blend_name] = {
                "valid": blend_valid,
                "test": blend_test,
                "raw": np.asarray(raw_valid[name], dtype=np.float64),
                "metrics": metrics,
            }

# Also test an equal consensus of all structurally different models before
# blending it with the incumbent.
consensus_valid = np.mean(
    np.stack([standardized(raw_valid[k]) for k in raw_valid]), axis=0
)
consensus_test = np.mean(
    np.stack([standardized(raw_test[k]) for k in raw_test]), axis=0
)
consensus_metrics = evaluate(valid.user_id, valid.y, consensus_valid)
candidates["family_consensus"] = float(consensus_metrics["primary"])
payloads["family_consensus"] = {
    "valid": consensus_valid,
    "test": consensus_test,
    "raw": consensus_valid,
    "metrics": consensus_metrics,
}

if have_incumbent:
    for alpha in np.linspace(0.10, 0.90, 17):
        alpha = float(alpha)
        blend_valid = (
            (1.0 - alpha) * zinc_valid
            + alpha * standardized(consensus_valid)
        )
        blend_test = (
            (1.0 - alpha) * zinc_test
            + alpha * standardized(consensus_test)
        )
        metrics = evaluate(valid.user_id, valid.y, blend_valid)
        name = "consensus_inc_a%.2f" % alpha
        candidates[name] = float(metrics["primary"])
        payloads[name] = {
            "valid": blend_valid,
            "test": blend_test,
            "raw": consensus_valid,
            "metrics": metrics,
        }

winner = max(candidates, key=candidates.get)
chosen = payloads[winner]
chosen_valid = np.asarray(chosen["valid"], dtype=np.float64)
chosen_test = np.asarray(chosen["test"], dtype=np.float64)
chosen_metrics = evaluate(valid.user_id, valid.y, chosen_valid)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        chosen_valid,
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        chosen_test,
    )
    if "_inc_a" in winner:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(chosen["raw"], dtype=np.float64),
        )

print("FINDINGS din_epoch_primary=" + json.dumps(din_epoch_scores))
print(
    "FINDINGS raw_components="
    + json.dumps(
        {
            name: {
                "primary": float(
                    evaluate(valid.user_id, valid.y, raw_valid[name])["primary"]
                ),
                "gauc": float(
                    evaluate(valid.user_id, valid.y, raw_valid[name])["gauc"]
                ),
                "ndcg@5": float(
                    evaluate(valid.user_id, valid.y, raw_valid[name])["ndcg@5"]
                ),
            }
            for name in raw_valid
        },
        sort_keys=True,
    )
)
print("FINDINGS winner=" + winner)
print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(chosen_metrics["primary"]),
            "gauc": float(chosen_metrics["gauc"]),
            "ndcg@5": float(chosen_metrics["ndcg@5"]),
            "gpu_seconds": float(time.time() - START),
        }
    )
)