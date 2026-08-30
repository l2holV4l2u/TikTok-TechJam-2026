import os
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 92741
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour", "user_active_degree",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_range",
    "video_type", "onehot_feat1", "onehot_feat3", "onehot_feat8",
]
EB_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket",
    "upload_type", "music_type", "onehot_feat3", "onehot_feat8",
]
CARDS = {f: int(FEATURE_CARDINALITIES[f]) for f in FIELDS}
USER_CARD = int(FEATURE_CARDINALITIES["user_id"])


def ordinal_day(date):
    d = np.asarray(date, dtype=np.int64)
    month = (d // 100) % 100
    day = d % 100
    return day + 30 * (month == 5)


def recency_weight(date, half_life=8.0):
    od = ordinal_day(date)
    age = od.max() - od
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    return (w / np.maximum(w.mean(), 1e-8)).astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    su = user_ids[order]
    start_mask = np.empty(n, dtype=bool)
    start_mask[0] = True
    start_mask[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(start_mask)
    group = np.cumsum(start_mask) - 1
    pos = np.arange(n) - starts[group]
    size = np.diff(np.r_[starts, n])
    denominator = np.maximum(size[group] - 1, 1)
    ranked_sorted = pos.astype(np.float64) / denominator
    ranked_sorted[size[group] == 1] = 0.5
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def best_incumbent_blend(user_ids, labels, own, incumbent):
    own_rank = within_user_rank(user_ids, own)
    inc_rank = within_user_rank(user_ids, incumbent)
    best = None
    for alpha in np.linspace(0.0, 1.0, 21):
        scores = alpha * own_rank + (1.0 - alpha) * inc_rank
        met = evaluate(user_ids, labels, scores)
        candidate = {
            "primary": float(met["primary"]),
            "alpha": float(alpha),
            "scores": scores,
            "metrics": met,
        }
        if best is None or candidate["primary"] > best["primary"]:
            best = candidate
    return best


class JoinedSplit:
    pass


def join_splits(a, b):
    out = JoinedSplit()
    out.X = {
        f: np.concatenate([
            np.asarray(a.X[f], dtype=np.int64),
            np.asarray(b.X[f], dtype=np.int64),
        ])
        for f in FIELDS
    }
    out.user_id = np.concatenate([
        np.asarray(a.user_id, dtype=np.int64),
        np.asarray(b.user_id, dtype=np.int64),
    ])
    out.video_id = np.concatenate([
        np.asarray(a.video_id, dtype=np.int64),
        np.asarray(b.video_id, dtype=np.int64),
    ])
    out.date = np.concatenate([
        np.asarray(a.date), np.asarray(b.date)
    ])
    out.time_ms = np.concatenate([
        np.asarray(a.time_ms), np.asarray(b.time_ms)
    ])
    return out


def lookup_sorted(keys, unique_keys, values, default):
    keys = np.asarray(keys, dtype=np.int64)
    idx = np.searchsorted(unique_keys, keys)
    valid = idx < len(unique_keys)
    out = np.full(len(keys), default, dtype=np.float32)
    if np.any(valid):
        vi = np.flatnonzero(valid)
        exact = unique_keys[idx[vi]] == keys[vi]
        vi = vi[exact]
        out[vi] = values[idx[vi]]
    return out


class EBModel:
    def __init__(self, base_rate, tables):
        self.base_rate = float(base_rate)
        self.tables = tables


def fit_eb(split, y, weights):
    y = np.asarray(y, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    base = float(np.sum(weights * y) / np.maximum(np.sum(weights), 1e-8))
    tables = {}

    for f in EB_FIELDS:
        x = np.asarray(split.X[f], dtype=np.int64)
        card = CARDS[f]

        entity_count = np.bincount(
            x, weights=weights, minlength=card
        ).astype(np.float32)
        entity_pos = np.bincount(
            x, weights=weights * y, minlength=card
        ).astype(np.float32)
        entity_rate = (
            entity_pos + 25.0 * base
        ) / np.maximum(entity_count + 25.0, 1e-6)

        keys = users * np.int64(card) + x
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        pair_count = np.bincount(
            inverse, weights=weights
        ).astype(np.float32)
        pair_pos = np.bincount(
            inverse, weights=weights * y
        ).astype(np.float32)

        entity_for_pair = (unique_keys % card).astype(np.int64)
        prior = entity_rate[entity_for_pair]

        if f in ("video_id", "author_id"):
            strength = 7.0
        elif f in ("tag", "onehot_feat3", "onehot_feat8"):
            strength = 14.0
        else:
            strength = 22.0

        pair_rate = (
            pair_pos + strength * prior
        ) / np.maximum(pair_count + strength, 1e-6)

        tables[f] = (
            unique_keys.astype(np.int64, copy=False),
            pair_rate.astype(np.float32, copy=False),
            entity_rate.astype(np.float32, copy=False),
        )
        del keys, unique_keys, inverse, pair_count, pair_pos

    return EBModel(base, tables)


def predict_eb_components(model, split):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    personalized = {}
    stationary = {}

    for f in EB_FIELDS:
        x = np.asarray(split.X[f], dtype=np.int64)
        card = CARDS[f]
        unique_keys, pair_rate, entity_rate = model.tables[f]
        keys = users * np.int64(card) + x
        er = entity_rate[np.minimum(x, len(entity_rate) - 1)]
        pr = lookup_sorted(keys, unique_keys, pair_rate, model.base_rate)

        er = np.clip(er.astype(np.float64), 1e-5, 1.0 - 1e-5)
        pr = np.clip(pr.astype(np.float64), 1e-5, 1.0 - 1e-5)
        stationary[f] = np.log(er / (1.0 - er))
        personalized[f] = np.log(pr / (1.0 - pr))

    memorization = (
        0.62 * personalized["video_id"]
        + 0.38 * personalized["author_id"]
    )
    broad_personalization = (
        0.28 * personalized["video_id"]
        + 0.22 * personalized["author_id"]
        + 0.16 * personalized["tag"]
        + 0.10 * personalized["onehot_feat3"]
        + 0.08 * personalized["onehot_feat8"]
        + 0.06 * personalized["duration_bucket"]
        + 0.05 * personalized["upload_type"]
        + 0.05 * personalized["music_type"]
    )
    stationary_content = (
        0.34 * stationary["video_id"]
        + 0.25 * stationary["author_id"]
        + 0.17 * stationary["tag"]
        + 0.08 * stationary["onehot_feat3"]
        + 0.06 * stationary["onehot_feat8"]
        + 0.04 * stationary["duration_bucket"]
        + 0.03 * stationary["upload_type"]
        + 0.03 * stationary["music_type"]
    )
    residual_preference = broad_personalization - 0.55 * stationary_content

    return {
        "eb_memorization": memorization,
        "eb_broad_personalization": broad_personalization,
        "eb_stationary_content": stationary_content,
        "eb_preference_residual": residual_preference,
    }


class AutoInt(nn.Module):
    def __init__(self, cards, embedding_dim=10, heads=2):
        super().__init__()
        self.fields = list(cards)
        offsets = []
        running = 0
        for f in self.fields:
            offsets.append(running)
            running += cards[f]
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )
        self.embedding = nn.Embedding(running, embedding_dim)
        self.linear = nn.Embedding(running, 1)
        self.attention = nn.MultiheadAttention(
            embedding_dim, heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.deep = nn.Sequential(
            nn.Linear(len(self.fields) * embedding_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.attention_head = nn.Linear(
            len(self.fields) * embedding_dim, 1
        )
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        indices = x + self.offsets
        emb = self.embedding(indices)
        attended, _ = self.attention(
            emb, emb, emb, need_weights=False
        )
        attended = self.norm(emb + attended)
        flat = attended.flatten(1)
        first_order = self.linear(indices).sum(dim=1).squeeze(-1)
        return (
            first_order
            + self.attention_head(flat).squeeze(-1)
            + self.deep(flat).squeeze(-1)
            + self.bias
        )


def matrix_for_autoint(split):
    return np.column_stack([
        np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
    ])


def fit_autoint(split, y, sample_weights, epochs=2):
    x = matrix_for_autoint(split)
    y = np.asarray(y, dtype=np.float32)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)

    model = AutoInt({f: CARDS[f] for f in FIELDS})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    batch_size = 8192
    rng = np.random.RandomState(SEED)

    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        epoch_loss = 0.0
        epoch_weight = 0.0
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]
            xb = torch.from_numpy(x[ix])
            yb = torch.from_numpy(y[ix])
            wb = torch.from_numpy(sample_weights[ix])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = loss_fn(logits, yb)
            loss = (losses * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += float((losses.detach() * wb).sum())
            epoch_weight += float(wb.sum())

        print(
            "FINDINGS autoint_epoch=%d weighted_logloss=%.6f"
            % (epoch + 1, epoch_loss / max(epoch_weight, 1.0))
        )

    del x
    return model


def predict_autoint(model, split):
    x = matrix_for_autoint(split)
    batch_size = 16384
    out = np.empty(len(x), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end])
            out[start:end] = model(xb).numpy().astype(np.float64)
    del x
    return out


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.float32)
train_weights = recency_weight(train.date, half_life=8.0)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

own_predictions = {}

# Family 1: empirical-Bayes non-parametric personalization and stationary priors.
eb_model = fit_eb(train, y_train, train_weights)
own_predictions.update(predict_eb_components(eb_model, valid))
del eb_model
gc.collect()

# Family 2: token self-attention over categorical fields.
autoint_model = fit_autoint(
    train, y_train, train_weights, epochs=2
)
own_predictions["autoint"] = predict_autoint(autoint_model, valid)
del autoint_model
gc.collect()

candidate_scores = {}
raw_scores = {}
selection = None

for name, prediction in own_predictions.items():
    raw_met = evaluate(valid.user_id, valid.y, prediction)
    raw_scores[name] = float(raw_met["primary"])

    blend = best_incumbent_blend(
        valid.user_id, valid.y, prediction, inc_valid
    )
    candidate_scores[name + "_blend"] = float(blend["primary"])

    if selection is None or blend["primary"] > selection["primary"]:
        selection = {
            "name": name,
            "primary": blend["primary"],
            "alpha": blend["alpha"],
            "scores": blend["scores"],
            "metrics": blend["metrics"],
            "raw": prediction,
        }

print("FINDINGS raw_primary=" + json.dumps(raw_scores, sort_keys=True))
print(
    "FINDINGS selected=%s own_rank_weight=%.2f"
    % (selection["name"], selection["alpha"])
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

valid_scores = np.asarray(selection["scores"], dtype=np.float64)
metrics = selection["metrics"]

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores,
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(selection["raw"], dtype=np.float64),
    )

# Identical-recipe refit on train + validation for test prediction.
combined = join_splits(train, valid)
combined_y = np.concatenate([y_train, y_valid]).astype(np.float32)
combined_weights = recency_weight(combined.date, half_life=8.0)
test = load("test")

if selection["name"] == "autoint":
    final_model = fit_autoint(
        combined, combined_y, combined_weights, epochs=2
    )
    own_test = predict_autoint(final_model, test)
    del final_model
else:
    final_eb = fit_eb(combined, combined_y, combined_weights)
    test_components = predict_eb_components(final_eb, test)
    own_test = test_components[selection["name"]]
    del final_eb, test_components

inc_test = np.load(inc_test_path).astype(np.float64)
test_scores = (
    selection["alpha"] * within_user_rank(test.user_id, own_test)
    + (1.0 - selection["alpha"])
    * within_user_rank(test.user_id, inc_test)
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}))