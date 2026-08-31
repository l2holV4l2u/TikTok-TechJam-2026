import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
HISTORY_LEN = 12
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
BATCH_SIZE = 4096

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
user_card = int(FEATURE_CARDINALITIES["user_id"])
video_card = int(FEATURE_CARDINALITIES["video_id"])


def make_x(split):
    result = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int32)
    for j, (name, card) in enumerate(zip(FIELDS, cards)):
        values = np.asarray(split.X[name], dtype=np.int64)
        if values.size and (values.min() < 0 or values.max() >= card):
            raise ValueError("Out-of-range categorical value in " + name)
        result[:, j] = values.astype(np.int32)
    return result


xtr = make_x(train)
xva = make_x(valid)
xte = make_x(test)
ytr = np.asarray(train.y, dtype=np.float32)

train_users = np.asarray(train.X["user_id"], dtype=np.int64)
train_videos = np.asarray(train.X["video_id"], dtype=np.int64)
train_times = np.asarray(train.time_ms, dtype=np.int64)

# Four-day temporal half-life. This is fixed in advance and only changes the
# contribution of train rows; no validation information enters training.
dates = np.asarray(train.date, dtype=np.int64)
row_weight = np.exp2((dates - dates.max()).astype(np.float32) / 4.0)
row_weight /= row_weight.mean()
row_weight = row_weight.astype(np.float32)

# Stable chronological ordering. Row index breaks feed-batch timestamp ties.
row_index = np.arange(len(ytr), dtype=np.int64)
chronological = np.lexsort((row_index, train_times, train_users))
sorted_users = train_users[chronological]
sorted_videos = train_videos[chronological]
sorted_y = (ytr[chronological] > 0.5).astype(np.int64)

group_start_flag = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
group_starts = np.flatnonzero(group_start_flag)
group_ends = np.r_[group_starts[1:], len(sorted_users)]
group_lengths = group_ends - group_starts

# Construct causal recent-positive histories for every train row without a
# Python loop over rows. Padding has its own ID, video_card.
positive_prefix = np.cumsum(sorted_y, dtype=np.int64)
positive_before = positive_prefix - sorted_y
positive_videos = sorted_videos[sorted_y > 0]

positives_before_group = positive_before[group_starts]
row_group_positive_base = np.repeat(positives_before_group, group_lengths)

history_sorted = np.full(
    (len(ytr), HISTORY_LEN), video_card, dtype=np.int32
)
for lag in range(1, HISTORY_LEN + 1):
    positive_ordinal = positive_before - lag
    available = positive_ordinal >= row_group_positive_base
    history_sorted[available, lag - 1] = positive_videos[
        positive_ordinal[available]
    ].astype(np.int32)

history_train = np.empty_like(history_sorted)
history_train[chronological] = history_sorted
del history_sorted

# Train-only history available for validation and test. Every evaluation row
# for a user receives that user's final train history; evaluation outcomes and
# validation feature statistics are never used.
user_history = np.full(
    (user_card, HISTORY_LEN), video_card, dtype=np.int32
)
group_user = sorted_users[group_starts]
group_positive_end = positive_prefix[group_ends - 1]
group_positive_base = positive_before[group_starts]

for lag in range(1, HISTORY_LEN + 1):
    positive_ordinal = group_positive_end - lag
    available = positive_ordinal >= group_positive_base
    valid_user = (
        available
        & (group_user >= 0)
        & (group_user < user_card)
    )
    user_history[group_user[valid_user], lag - 1] = positive_videos[
        positive_ordinal[valid_user]
    ].astype(np.int32)


class DIN(nn.Module):
    def __init__(self, cardinalities, emb_dim=12):
        super().__init__()
        self.embeddings = nn.ModuleList()
        for j, card in enumerate(cardinalities):
            # Video embedding has one extra train-history padding ID.
            size = card + 1 if j == 1 else card
            emb = nn.Embedding(size, emb_dim)
            nn.init.normal_(emb.weight, std=0.02)
            self.embeddings.append(emb)

        self.padding_video = cardinalities[1]
        input_dim = len(cardinalities) * emb_dim + 2 * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x, history):
        field_vectors = [
            emb(x[:, j]) for j, emb in enumerate(self.embeddings)
        ]
        candidate = field_vectors[1]
        history_vectors = self.embeddings[1](history)

        mask = history != self.padding_video
        attention_logits = (
            history_vectors * candidate.unsqueeze(1)
        ).sum(dim=2) / np.sqrt(candidate.shape[1])
        attention_logits = attention_logits.masked_fill(~mask, -1.0e4)

        attention = torch.softmax(attention_logits, dim=1)
        attention = attention * mask.to(attention.dtype)
        attention = attention / attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)

        interest = (
            attention.unsqueeze(2) * history_vectors
        ).sum(dim=1)

        flat_fields = torch.cat(field_vectors, dim=1)
        features = torch.cat(
            [flat_fields, interest, interest * candidate], dim=1
        )
        return self.mlp(features).squeeze(1)


def train_din():
    torch.manual_seed(SEED)
    model = DIN(cards, emb_dim=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    rng = np.random.default_rng(SEED)

    model.train()
    for _ in range(3):
        order = rng.permutation(len(ytr))
        for lo in range(0, len(order), BATCH_SIZE):
            idx = order[lo:lo + BATCH_SIZE]
            xb = torch.from_numpy(xtr[idx].astype(np.int64, copy=False))
            hb = torch.from_numpy(
                history_train[idx].astype(np.int64, copy=False)
            )
            target = torch.from_numpy(ytr[idx])
            weight = torch.from_numpy(row_weight[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            loss_row = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (loss_row * weight).sum() / weight.sum()
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict_din(model, x):
    model.eval()
    scores = np.empty(len(x), dtype=np.float64)
    local_users = x[:, 0].astype(np.int64, copy=False)

    for lo in range(0, len(x), 16384):
        hi = min(lo + 16384, len(x))
        xb_np = x[lo:hi].astype(np.int64, copy=False)
        uid = local_users[lo:hi]

        hb_np = np.full(
            (hi - lo, HISTORY_LEN), video_card, dtype=np.int64
        )
        known = (uid >= 0) & (uid < user_card)
        if np.any(known):
            hb_np[known] = user_history[uid[known]]

        logits = model(
            torch.from_numpy(xb_np),
            torch.from_numpy(hb_np),
        )
        scores[lo:hi] = logits.cpu().numpy().astype(np.float64)

    return scores


class Item2Vec(nn.Module):
    def __init__(self, n_items, rank=32):
        super().__init__()
        self.center = nn.Embedding(n_items, rank)
        self.context = nn.Embedding(n_items, rank)
        self.context_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.center.weight, std=0.04)
        nn.init.normal_(self.context.weight, std=0.04)
        nn.init.zeros_(self.context_bias.weight)
        self.scale = rank ** -0.5

    def pair_score(self, center, context):
        return (
            (self.center(center) * self.context(context)).sum(dim=1)
            * self.scale
            + self.context_bias(context).squeeze(1)
        )


def train_item2vec():
    # Adjacent positive views define directed next-positive transitions.
    positive_rows_sorted = chronological[sorted_y > 0]
    pos_users = train_users[positive_rows_sorted]
    pos_videos = train_videos[positive_rows_sorted]

    adjacent = pos_users[1:] == pos_users[:-1]
    previous_video = pos_videos[:-1][adjacent]
    next_video = pos_videos[1:][adjacent]
    next_rows = positive_rows_sorted[1:][adjacent]
    pair_weights = row_weight[next_rows]

    # Both the forward transition and reverse neighborhood relation are useful
    # in sparse histories, while prediction remains directed.
    centers = np.concatenate([previous_video, next_video]).astype(np.int64)
    contexts = np.concatenate([next_video, previous_video]).astype(np.int64)
    weights = np.concatenate([pair_weights, pair_weights]).astype(np.float32)

    torch.manual_seed(SEED + 1)
    model = Item2Vec(video_card, rank=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    rng = np.random.default_rng(SEED + 1)

    model.train()
    for _ in range(4):
        order = rng.permutation(len(centers))
        for lo in range(0, len(order), 8192):
            idx = order[lo:lo + 8192]
            center_np = centers[idx]
            context_np = contexts[idx]
            weight_np = weights[idx]

            # Four random negatives per observed transition.
            negative_np = rng.integers(
                0, video_card,
                size=(len(idx), 4),
                dtype=np.int64,
            )

            center = torch.from_numpy(center_np)
            context = torch.from_numpy(context_np)
            negative = torch.from_numpy(negative_np)
            weight = torch.from_numpy(weight_np)

            optimizer.zero_grad(set_to_none=True)
            positive_score = model.pair_score(center, context)

            center_vector = model.center(center).unsqueeze(1)
            negative_vector = model.context(negative)
            negative_score = (
                (center_vector * negative_vector).sum(dim=2)
                * model.scale
                + model.context_bias(negative).squeeze(2)
            )

            loss_row = (
                F.softplus(-positive_score)
                + F.softplus(negative_score).mean(dim=1)
            )
            loss = (loss_row * weight).sum() / weight.sum()
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict_item2vec(model, x):
    model.eval()
    result = np.empty(len(x), dtype=np.float64)
    decay = torch.from_numpy(
        np.exp2(-np.arange(HISTORY_LEN, dtype=np.float32) / 3.0)
    )

    for lo in range(0, len(x), 16384):
        hi = min(lo + 16384, len(x))
        users = x[lo:hi, 0].astype(np.int64, copy=False)
        candidates_np = x[lo:hi, 1].astype(np.int64, copy=False)

        histories = np.full(
            (hi - lo, HISTORY_LEN), video_card, dtype=np.int64
        )
        known = (users >= 0) & (users < user_card)
        if np.any(known):
            histories[known] = user_history[users[known]]

        mask_np = histories != video_card
        safe_history = histories.copy()
        safe_history[~mask_np] = 0

        history = torch.from_numpy(safe_history)
        mask = torch.from_numpy(mask_np)
        candidate = torch.from_numpy(candidates_np)

        history_vectors = model.center(history)
        weights = (
            mask.to(torch.float32) * decay.unsqueeze(0)
        )
        interest = (
            history_vectors * weights.unsqueeze(2)
        ).sum(dim=1)
        interest = interest / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

        candidate_vector = model.context(candidate)
        score = (
            interest * candidate_vector
        ).sum(dim=1) * model.scale
        score = score + model.context_bias(candidate).squeeze(1)

        result[lo:hi] = score.cpu().numpy().astype(np.float64)

    return result


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Ascending percentile ranks; deterministic row index resolves exact ties.
    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ordered_rank = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ordered_rank[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    rank = np.empty(n, dtype=np.float64)
    rank[order] = ordered_rank
    return rank


# Family 1: supervised causal attention over positive user history.
din = train_din()
din_valid = predict_din(din, xva)
din_test = predict_din(din, xte)
del din

# Family 2: self-supervised latent next-positive transition geometry.
item2vec = train_item2vec()
i2v_valid = predict_item2vec(item2vec, xva)
i2v_test = predict_item2vec(item2vec, xte)
del item2vec

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required for incumbent blends")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

if len(inc_valid) != len(valid.user_id) or len(inc_test) != len(test.user_id):
    raise ValueError("Trusted incumbent prediction length mismatch")

families_valid = {
    "din": din_valid,
    "item2vec_transition": i2v_valid,
}
families_test = {
    "din": din_test,
    "item2vec_transition": i2v_test,
}

candidate_valid = {"trusted_incumbent": inc_valid}
candidate_test = {"trusted_incumbent": inc_test}
candidate_raw = {}
candidate_scores = {}

candidate_scores["trusted_incumbent"] = float(
    evaluate(valid.user_id, valid.y, inc_valid)["primary"]
)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

for name in families_valid:
    raw_valid = families_valid[name]
    raw_test = families_test[name]

    candidate_valid[name] = raw_valid
    candidate_test[name] = raw_test
    candidate_raw[name] = raw_valid
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, raw_valid)["primary"]
    )

    own_valid_rank = within_user_rank(valid.user_id, raw_valid)
    own_test_rank = within_user_rank(test.user_id, raw_test)

    # Search only rank-aggregation weights. The selected value is applied
    # unchanged to test, and no test labels or test-derived statistics exist.
    for own_alpha in (0.25, 0.50, 0.75):
        blend_name = "%s_blend_%.2f" % (name, own_alpha)
        blend_valid = (
            own_alpha * own_valid_rank
            + (1.0 - own_alpha) * inc_valid_rank
        )
        blend_test = (
            own_alpha * own_test_rank
            + (1.0 - own_alpha) * inc_test_rank
        )

        candidate_valid[blend_name] = blend_valid
        candidate_test[blend_name] = blend_test
        candidate_raw[blend_name] = raw_valid
        candidate_scores[blend_name] = float(
            evaluate(valid.user_id, valid.y, blend_valid)["primary"]
        )

# A third prediction mechanism: rank aggregation of the two independent
# sequential families before combining with the incumbent.
din_rank_valid = within_user_rank(valid.user_id, din_valid)
din_rank_test = within_user_rank(test.user_id, din_test)
i2v_rank_valid = within_user_rank(valid.user_id, i2v_valid)
i2v_rank_test = within_user_rank(test.user_id, i2v_test)

sequential_ensemble_valid = 0.5 * din_rank_valid + 0.5 * i2v_rank_valid
sequential_ensemble_test = 0.5 * din_rank_test + 0.5 * i2v_rank_test

candidate_valid["sequential_ensemble"] = sequential_ensemble_valid
candidate_test["sequential_ensemble"] = sequential_ensemble_test
candidate_raw["sequential_ensemble"] = sequential_ensemble_valid
candidate_scores["sequential_ensemble"] = float(
    evaluate(
        valid.user_id, valid.y, sequential_ensemble_valid
    )["primary"]
)

for own_alpha in (0.25, 0.50):
    name = "sequential_ensemble_blend_%.2f" % own_alpha
    blend_valid = (
        own_alpha * sequential_ensemble_valid
        + (1.0 - own_alpha) * inc_valid_rank
    )
    blend_test = (
        own_alpha * sequential_ensemble_test
        + (1.0 - own_alpha) * inc_test_rank
    )
    candidate_valid[name] = blend_valid
    candidate_test[name] = blend_test
    candidate_raw[name] = sequential_ensemble_valid
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, blend_valid)["primary"]
    )

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_valid[winner]
test_scores = candidate_test[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if winner != "trusted_incumbent":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
        )

known_history_valid = np.mean(
    np.any(
        user_history[
            np.clip(
                np.asarray(valid.X["user_id"], dtype=np.int64),
                0,
                user_card - 1,
            )
        ] != video_card,
        axis=1,
    )
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "valid_rows_with_train_positive_history": float(
                known_history_valid
            ),
            "din_item2vec_score_correlation": float(
                np.corrcoef(din_valid, i2v_valid)[0, 1]
            ),
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(time.time() - START),
        }
    )
)