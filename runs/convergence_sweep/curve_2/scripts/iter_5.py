import os
import time
import json
import math
import random
import gc

import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

FIELDS = [
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
CARDS = [int(FEATURE_CARDINALITIES[x]) for x in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))
N_FIELDS = len(FIELDS)
EMBED_DIM = 10

USER_COL = FIELDS.index("user_id")
VIDEO_COL = FIELDS.index("video_id")
AUTHOR_COL = FIELDS.index("author_id")
TAG_COL = FIELDS.index("tag")


def make_cat(split):
    return np.ascontiguousarray(
        np.stack(
            [np.asarray(split.X[x], dtype=np.int64) for x in FIELDS],
            axis=1,
        ),
        dtype=np.int64,
    )


x_train_np = make_cat(train)
x_valid_np = make_cat(valid)
x_test_np = make_cat(test)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

max_train_date = int(np.max(np.asarray(train.date)))
train_age = (
    max_train_date - np.asarray(train.date, dtype=np.int32)
).astype(np.float32)

# Moderate recency weighting: old data remains useful for sparse identities,
# but recent behavior has greater influence under the date split.
sample_weight_np = np.exp(
    -math.log(2.0) * train_age / 6.0
).astype(np.float32)
sample_weight_np /= float(sample_weight_np.mean())
sample_weight = torch.from_numpy(sample_weight_np)

offset_tensor = torch.from_numpy(OFFSETS.copy())


class FieldWeightedFM(nn.Module):
    """FM whose interaction strength depends on the ordered field pair."""

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.pair_weight = nn.Parameter(
            torch.ones(N_FIELDS, N_FIELDS) * 0.08
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

        pairs = torch.triu_indices(N_FIELDS, N_FIELDS, offset=1)
        self.register_buffer("pair_i", pairs[0])
        self.register_buffer("pair_j", pairs[1])

    def forward(self, x):
        ids = x + self.offsets
        e = self.embedding(ids)
        linear = self.linear(ids).sum(dim=1).squeeze(1)
        products = (
            e[:, self.pair_i, :] * e[:, self.pair_j, :]
        ).sum(dim=2)
        weights = self.pair_weight[self.pair_i, self.pair_j]
        return self.bias + linear + (products * weights).sum(dim=1)


class MultiTaskMMoE(nn.Module):
    """
    Shared experts are regularized by click/like/follow prediction, while
    separate gates let long_view choose a different expert mixture.
    """

    def __init__(self, n_tasks=4):
        super().__init__()
        self.n_tasks = n_tasks
        self.register_buffer("offsets", offset_tensor.clone())
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        in_dim = N_FIELDS * EMBED_DIM
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 96),
                nn.PReLU(),
                nn.Dropout(0.04),
                nn.Linear(96, 48),
                nn.PReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, len(self.experts))
            for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(48, 24),
                nn.PReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(n_tasks)
        ])
        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, x):
        z = self.embedding(x + self.offsets).flatten(1)
        experts = torch.stack([m(z) for m in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(z), dim=1).unsqueeze(2)
            mixed = (weights * experts).sum(dim=1)
            outputs.append(tower(mixed).squeeze(1))
        return torch.stack(outputs, dim=1)


# ----------------------------------------------------------------------
# Strictly historical positive-video sequences for BST.
# ----------------------------------------------------------------------
HISTORY_LEN = 10
user_card = int(FEATURE_CARDINALITIES["user_id"])
video_card = int(FEATURE_CARDINALITIES["video_id"])

tr_uid = np.asarray(train.X["user_id"], dtype=np.int64)
tr_vid = np.asarray(train.X["video_id"], dtype=np.int64)
row_pos = np.arange(len(y_train_np), dtype=np.int64)
chron_order = np.lexsort(
    (row_pos, np.asarray(train.time_ms, dtype=np.int64), tr_uid)
)

ord_uid = tr_uid[chron_order]
ord_vid = tr_vid[chron_order]
ord_y = y_train_np[chron_order].astype(np.int8)

positive_totals = np.bincount(
    ord_uid, weights=ord_y, minlength=user_card
).astype(np.int64)
positive_base = np.zeros(user_card, dtype=np.int64)
positive_base[1:] = np.cumsum(positive_totals[:-1])

packed_positive = ord_vid[ord_y == 1].astype(np.int32)
packed_positive = np.concatenate(
    [np.zeros(1, dtype=np.int32), packed_positive]
)

global_pos = np.cumsum(ord_y, dtype=np.int64)
new_user = np.r_[True, ord_uid[1:] != ord_uid[:-1]]
group_start = np.maximum.accumulate(
    np.where(new_user, np.arange(len(ord_uid), dtype=np.int64), 0)
)
before_group = global_pos[group_start] - ord_y[group_start]
previous_count = global_pos - ord_y - before_group


def materialize_history(user_ids, counts):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    back = np.arange(HISTORY_LEN, 0, -1, dtype=np.int64)[None, :]
    # Oldest-to-newest among the selected last HISTORY_LEN positives.
    distance_from_end = HISTORY_LEN - back + 1
    indices = (
        1
        + positive_base[user_ids, None]
        + counts[:, None]
        - distance_from_end
    )
    ok = distance_from_end <= counts[:, None]
    indices = np.where(ok, indices, 0)
    return np.ascontiguousarray(packed_positive[indices], dtype=np.int64)


hist_train_ord = materialize_history(ord_uid, previous_count)
hist_train_np = np.empty_like(hist_train_ord)
hist_train_np[chron_order] = hist_train_ord
del hist_train_ord

valid_uid = np.asarray(valid.X["user_id"], dtype=np.int64)
test_uid = np.asarray(test.X["user_id"], dtype=np.int64)
hist_valid_np = materialize_history(
    valid_uid, positive_totals[valid_uid]
)
hist_test_np = materialize_history(
    test_uid, positive_totals[test_uid]
)
hist_train = torch.from_numpy(hist_train_np)


class BST(nn.Module):
    """
    A target video token attends over the ordered train-only positive history.
    Unlike DIN's independent target-to-event weights, history events interact
    with each other before the target representation is formed.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())
        self.context_embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.video_embedding = nn.Embedding(
            video_card, EMBED_DIM, padding_idx=0
        )
        self.position = nn.Parameter(
            torch.zeros(1, HISTORY_LEN + 1, EMBED_DIM)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=2,
            dim_feedforward=40,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        in_dim = N_FIELDS * EMBED_DIM + 2 * EMBED_DIM
        self.output = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.PReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.context_embedding.weight, std=0.025)
        nn.init.normal_(self.video_embedding.weight, std=0.025)
        nn.init.normal_(self.position, std=0.01)
        with torch.no_grad():
            self.video_embedding.weight[0].zero_()

    def forward(self, x, history):
        context = self.context_embedding(
            x + self.offsets
        ).flatten(1)
        target_id = x[:, VIDEO_COL]
        target = self.video_embedding(target_id)
        sequence_ids = torch.cat(
            [history, target_id.unsqueeze(1)], dim=1
        )
        sequence = self.video_embedding(sequence_ids) + self.position
        padding_mask = sequence_ids.eq(0)
        padding_mask[:, -1] = False
        encoded = self.transformer(
            sequence, src_key_padding_mask=padding_mask
        )
        target_state = encoded[:, -1, :]
        z = torch.cat(
            [context, target, target * target_state], dim=1
        )
        return self.output(z).squeeze(1)


class BPRScorer(nn.Module):
    """
    Pairwise collaborative scorer over logged impressions. It combines
    user-video, user-author, and user-tag preference geometries.
    """

    def __init__(self, dim=20):
        super().__init__()
        self.user = nn.Embedding(user_card, dim)
        self.video = nn.Embedding(video_card, dim)
        self.author = nn.Embedding(
            int(FEATURE_CARDINALITIES["author_id"]), dim
        )
        self.tag = nn.Embedding(
            int(FEATURE_CARDINALITIES["tag"]), dim
        )
        self.video_bias = nn.Embedding(video_card, 1)
        self.author_bias = nn.Embedding(
            int(FEATURE_CARDINALITIES["author_id"]), 1
        )
        for emb in (self.user, self.video, self.author, self.tag):
            nn.init.normal_(emb.weight, std=0.025)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.author_bias.weight)

    def forward(self, x):
        u = self.user(x[:, USER_COL])
        v = self.video(x[:, VIDEO_COL])
        a = self.author(x[:, AUTHOR_COL])
        t = self.tag(x[:, TAG_COL])
        score = (
            (u * v).sum(dim=1)
            + 0.55 * (u * a).sum(dim=1)
            + 0.30 * (u * t).sum(dim=1)
            + self.video_bias(x[:, VIDEO_COL]).squeeze(1)
            + 0.5 * self.author_bias(x[:, AUTHOR_COL]).squeeze(1)
        )
        return score


@torch.no_grad()
def predict_pointwise(model, x_np, history_np=None, batch_size=16384):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    for begin in range(0, len(x_np), batch_size):
        end = min(begin + batch_size, len(x_np))
        xb = torch.from_numpy(x_np[begin:end])
        if history_np is None:
            value = model(xb)
            if value.ndim == 2:
                value = value[:, 0]
        else:
            hb = torch.from_numpy(history_np[begin:end])
            value = model(xb, hb)
        result[begin:end] = value.detach().cpu().numpy()
    return result


def train_binary_model(model, name, history=False, epochs=2):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.2e-3, weight_decay=2e-6
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + sum(ord(c) for c in name))

    best_primary = -1.0
    best_state = None
    epoch_scores = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(
            len(y_train_np), generator=generator
        )
        for begin in range(0, len(y_train_np), 8192):
            idx = permutation[begin:begin + 8192]
            xb = x_train[idx]
            if history:
                logits = model(xb, hist_train[idx])
            else:
                logits = model(xb)
                if logits.ndim == 2:
                    logits = logits[:, 0]

            loss_vec = nn.functional.binary_cross_entropy_with_logits(
                logits, y_train[idx], reduction="none"
            )
            w = sample_weight[idx]
            loss = (loss_vec * w).sum() / w.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        va = predict_pointwise(
            model,
            x_valid_np,
            hist_valid_np if history else None,
        )
        metric = evaluate(valid.user_id, valid.y, va)
        primary = float(metric["primary"])
        epoch_scores.append(primary)
        if primary > best_primary:
            best_primary = primary
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    va = predict_pointwise(
        model, x_valid_np, hist_valid_np if history else None
    )
    te = predict_pointwise(
        model, x_test_np, hist_test_np if history else None
    )
    print(
        "FINDINGS %s_epoch_primary=%s"
        % (name, json.dumps(epoch_scores))
    )
    return va, te


def train_mmoe():
    model = MultiTaskMMoE(n_tasks=4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.1e-3, weight_decay=3e-6
    )

    auxiliary_names = ["is_click", "is_like", "is_follow"]
    targets_np = np.stack(
        [y_train_np] + [
            np.asarray(train.aux[x], dtype=np.float32)
            for x in auxiliary_names
        ],
        axis=1,
    )
    targets_np = np.nan_to_num(
        targets_np, nan=0.0, posinf=1.0, neginf=0.0
    ).astype(np.float32)
    targets = torch.from_numpy(targets_np)
    task_weights = torch.tensor(
        [1.0, 0.16, 0.10, 0.06], dtype=torch.float32
    )

    generator = torch.Generator()
    generator.manual_seed(SEED + 700)
    best_primary = -1.0
    best_state = None
    epoch_scores = []

    for epoch in range(2):
        model.train()
        permutation = torch.randperm(
            len(y_train_np), generator=generator
        )
        for begin in range(0, len(y_train_np), 8192):
            idx = permutation[begin:begin + 8192]
            logits = model(x_train[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, targets[idx], reduction="none"
            )
            per_row = (losses * task_weights).sum(dim=1)
            w = sample_weight[idx]
            loss = (per_row * w).sum() / w.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        va = predict_pointwise(model, x_valid_np)
        metric = evaluate(valid.user_id, valid.y, va)
        primary = float(metric["primary"])
        epoch_scores.append(primary)
        if primary > best_primary:
            best_primary = primary
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    va = predict_pointwise(model, x_valid_np)
    te = predict_pointwise(model, x_test_np)
    print(
        "FINDINGS mmoe_epoch_primary=%s"
        % json.dumps(epoch_scores)
    )
    return va, te


def train_bpr():
    model = BPRScorer()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=1e-5
    )

    # For every row, sample another logged impression from the same user.
    # Retain only opposite-label pairs and orient them positive > negative.
    user_order = np.argsort(tr_uid, kind="stable")
    sorted_uid = tr_uid[user_order]
    starts_mask = np.r_[True, sorted_uid[1:] != sorted_uid[:-1]]
    starts = np.flatnonzero(starts_mask)
    sizes = np.diff(np.r_[starts, len(user_order)])
    row_group_start_sorted = np.repeat(starts, sizes)
    row_group_size_sorted = np.repeat(sizes, sizes)

    generator_np = np.random.default_rng(SEED + 900)
    best_primary = -1.0
    best_state = None
    epoch_scores = []

    for epoch in range(2):
        offsets = (
            generator_np.random(len(user_order))
            * row_group_size_sorted
        ).astype(np.int64)
        partner_sorted = (
            row_group_start_sorted + offsets
        )
        first = user_order
        second = user_order[partner_sorted]

        differing = y_train_np[first] != y_train_np[second]
        first = first[differing]
        second = second[differing]

        first_positive = y_train_np[first] > y_train_np[second]
        positive = np.where(first_positive, first, second)
        negative = np.where(first_positive, second, first)

        permutation = generator_np.permutation(len(positive))
        positive = positive[permutation]
        negative = negative[permutation]

        model.train()
        for begin in range(0, len(positive), 8192):
            pidx_np = positive[begin:begin + 8192]
            nidx_np = negative[begin:begin + 8192]
            pidx = torch.from_numpy(pidx_np)
            nidx = torch.from_numpy(nidx_np)

            pos_score = model(x_train[pidx])
            neg_score = model(x_train[nidx])
            pair_loss = nn.functional.softplus(
                -(pos_score - neg_score)
            )
            w = sample_weight[pidx]
            loss = (pair_loss * w).sum() / w.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        va = predict_pointwise(model, x_valid_np)
        metric = evaluate(valid.user_id, valid.y, va)
        primary = float(metric["primary"])
        epoch_scores.append(primary)
        if primary > best_primary:
            best_primary = primary
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    va = predict_pointwise(model, x_valid_np)
    te = predict_pointwise(model, x_test_np)
    print(
        "FINDINGS bpr_epoch_primary=%s"
        % json.dumps(epoch_scores)
    )
    return va, te


predictions = {}

fwfm_valid, fwfm_test = train_binary_model(
    FieldWeightedFM(), "field_weighted_fm", history=False, epochs=2
)
predictions["field_weighted_fm"] = (fwfm_valid, fwfm_test)
gc.collect()

mmoe_valid, mmoe_test = train_mmoe()
predictions["mmoe"] = (mmoe_valid, mmoe_test)
gc.collect()

bst_valid, bst_test = train_binary_model(
    BST(), "bst", history=True, epochs=2
)
predictions["bst"] = (bst_valid, bst_test)
gc.collect()

bpr_valid, bpr_test = train_bpr()
predictions["bpr"] = (bpr_valid, bpr_test)
gc.collect()


# ----------------------------------------------------------------------
# Rank aggregation is invariant to family-specific score calibration and
# directly combines the within-user ordering relevant to both metrics.
# ----------------------------------------------------------------------
shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
have_incumbent = (
    os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)
)

if have_incumbent:
    incumbent_valid = np.asarray(
        np.load(inc_valid_path), dtype=np.float64
    )
    incumbent_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
else:
    incumbent_valid = None
    incumbent_test = None


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    row = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.flatnonzero(starts_mask)
    sizes = np.diff(np.r_[starts, len(order)])
    repeated_starts = np.repeat(starts, sizes)
    repeated_sizes = np.repeat(sizes, sizes)

    position = np.arange(len(order), dtype=np.float64) - repeated_starts
    denom = np.maximum(repeated_sizes - 1, 1)
    normalized = position / denom

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = normalized
    return result


candidate_scores = {}
candidate_payload = {}

if have_incumbent:
    inc_v_rank = within_user_rank(valid.user_id, incumbent_valid)
    inc_t_rank = within_user_rank(test.user_id, incumbent_test)

for name, (va_raw, te_raw) in predictions.items():
    raw_metric = evaluate(valid.user_id, valid.y, va_raw)
    raw_primary = float(raw_metric["primary"])
    candidate_scores[name + "_raw"] = raw_primary
    candidate_payload[name + "_raw"] = (
        np.asarray(va_raw, dtype=np.float64),
        np.asarray(te_raw, dtype=np.float64),
        np.asarray(va_raw, dtype=np.float64),
        False,
    )

    if have_incumbent:
        va_rank = within_user_rank(valid.user_id, va_raw)
        te_rank = within_user_rank(test.user_id, te_raw)

        # Include both conservative and family-dominant combinations.
        for alpha in (0.25, 0.50, 0.75):
            blended_valid = (
                alpha * va_rank + (1.0 - alpha) * inc_v_rank
            )
            blended_test = (
                alpha * te_rank + (1.0 - alpha) * inc_t_rank
            )
            metric = evaluate(
                valid.user_id, valid.y, blended_valid
            )
            key = "%s_rankblend_%.2f" % (name, alpha)
            candidate_scores[key] = float(metric["primary"])
            candidate_payload[key] = (
                blended_valid,
                blended_test,
                np.asarray(va_raw, dtype=np.float64),
                True,
            )

if have_incumbent:
    incumbent_metric = evaluate(
        valid.user_id, valid.y, incumbent_valid
    )
    candidate_scores["trusted_incumbent"] = float(
        incumbent_metric["primary"]
    )
    candidate_payload["trusted_incumbent"] = (
        incumbent_valid,
        incumbent_test,
        incumbent_valid,
        True,
    )

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores, test_scores, raw_scores, is_combination = candidate_payload[winner]
final_metric = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(
    {k: round(v, 7) for k, v in candidate_scores.items()},
    sort_keys=True,
))
print(
    "FINDINGS winner=%s winner_primary=%.7f"
    % (winner, float(final_metric["primary"]))
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if is_combination:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_scores, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        float(final_metric["primary"]),
        float(final_metric["gauc"]),
        float(final_metric["ndcg@5"]),
        elapsed,
    )
)