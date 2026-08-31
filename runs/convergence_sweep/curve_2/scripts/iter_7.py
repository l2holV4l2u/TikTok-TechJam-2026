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
CARDS = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))
N_FIELDS = len(FIELDS)
EMBED_DIM = 8

offset_tensor = torch.from_numpy(OFFSETS.copy())
pair_index = torch.triu_indices(N_FIELDS, N_FIELDS, offset=1)
N_PAIRS = int(pair_index.shape[1])


def make_cat(split):
    return np.ascontiguousarray(
        np.stack(
            [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS],
            axis=1,
        ),
        dtype=np.int64,
    )


x_train_np = make_cat(train)
x_valid_np = make_cat(valid)
x_test_np = make_cat(test)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

# Emphasize observations nearest the date boundary. This uses train dates only.
max_train_date = int(np.max(np.asarray(train.date, dtype=np.int32)))
train_age = (
    max_train_date - np.asarray(train.date, dtype=np.int32)
).astype(np.float32)
HALF_LIFE_DAYS = 4.5
sample_weight_np = np.exp(
    -math.log(2.0) * train_age / HALF_LIFE_DAYS
).astype(np.float32)
sample_weight_np /= float(sample_weight_np.mean())
sample_weight = torch.from_numpy(sample_weight_np)


def make_sequential_histories(k=5):
    """
    For each train impression, return the previous k train impressions of that
    user. Validation and test receive the final k impressions from train only.
    No validation/test outcomes or rows are incorporated into the history.
    """
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    times = np.asarray(train.time_ms, dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    sv = videos[order]

    new_group = np.r_[True, su[1:] != su[:-1]]
    starts = np.maximum.accumulate(
        np.where(new_group, np.arange(len(order)), 0)
    )
    position = np.arange(len(order), dtype=np.int64) - starts

    hist_sorted = np.zeros((len(order), k), dtype=np.int64)
    for lag in range(1, k + 1):
        valid_lag = position >= lag
        target = np.flatnonzero(valid_lag)
        hist_sorted[target, k - lag] = sv[target - lag]

    hist_train = np.zeros_like(hist_sorted)
    hist_train[order] = hist_sorted

    user_card = int(FEATURE_CARDINALITIES["user_id"])
    final_history = np.zeros((user_card, k), dtype=np.int64)
    group_end = np.r_[
        np.flatnonzero(su[1:] != su[:-1]) + 1,
        len(su),
    ]
    group_users = su[group_end - 1]

    for lag in range(1, k + 1):
        source = group_end - lag
        group_start = np.r_[0, group_end[:-1]]
        ok = source >= group_start
        final_history[group_users[ok], k - lag] = sv[source[ok]]

    def transfer(split):
        split_users = np.asarray(split.user_id, dtype=np.int64)
        result = np.zeros((len(split_users), k), dtype=np.int64)
        known = (split_users >= 0) & (split_users < user_card)
        result[known] = final_history[split_users[known]]
        return result

    return (
        np.ascontiguousarray(hist_train),
        np.ascontiguousarray(transfer(valid)),
        np.ascontiguousarray(transfer(test)),
    )


hist_train_np, hist_valid_np, hist_test_np = make_sequential_histories(k=5)
hist_train = torch.from_numpy(hist_train_np)


class FieldAwareFM(nn.Module):
    """
    Each feature value has a different embedding for every target field.
    Thus user-video, user-author, and content-context pairs need not share
    the same latent geometry.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())
        self.register_buffer("pair_i", pair_index[0].clone())
        self.register_buffer("pair_j", pair_index[1].clone())

        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.ffm = nn.Embedding(TOTAL_CARD * N_FIELDS, EMBED_DIM)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.ffm.weight, std=0.025)

    def forward(self, x, history=None):
        ids = x + self.offsets
        left_ids = ids[:, self.pair_i] * N_FIELDS + self.pair_j
        right_ids = ids[:, self.pair_j] * N_FIELDS + self.pair_i

        left = self.ffm(left_ids)
        right = self.ffm(right_ids)
        interactions = (left * right).sum(dim=(1, 2))
        linear = self.linear(ids).sum(dim=1).squeeze(1)
        return self.bias + linear + interactions


class NeuralFM(nn.Module):
    """
    NFM retains the FM bi-interaction vector instead of immediately summing
    it, allowing a nonlinear tower to distinguish interaction dimensions.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())
        self.embedding = nn.Embedding(TOTAL_CARD, 12)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.tower = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.Linear(12, 48),
            nn.PReLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 20),
            nn.PReLU(),
            nn.Linear(20, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history=None):
        ids = x + self.offsets
        emb = self.embedding(ids)
        summed = emb.sum(dim=1)
        bi = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        )
        linear = self.linear(ids).sum(dim=1).squeeze(1)
        return self.bias + linear + self.tower(bi).squeeze(1)


class WideDeep(nn.Module):
    """
    The wide branch memorizes selected identity/context crosses in hashed
    tables, while the deep branch generalizes across all field embeddings.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)

        self.hash_size = 200003
        self.wide_cross = nn.Embedding(self.hash_size, 1)
        self.deep = nn.Sequential(
            nn.Linear(N_FIELDS * EMBED_DIM, 112),
            nn.PReLU(),
            nn.Dropout(0.06),
            nn.Linear(112, 48),
            nn.PReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.wide_cross.weight)

    def forward(self, x, history=None):
        ids = x + self.offsets

        # user-video, user-author, video-tab, video-tag, author-tag
        cross_pairs = ((0, 1), (0, 2), (1, 3), (1, 4), (2, 4))
        hashes = []
        for number, (a, b) in enumerate(cross_pairs):
            h = (
                x[:, a] * 1000003
                + x[:, b] * 9176
                + number * 7919
            ) % self.hash_size
            hashes.append(h)
        hashes = torch.stack(hashes, dim=1)

        wide = self.linear(ids).sum(dim=1).squeeze(1)
        wide = wide + self.wide_cross(hashes).sum(dim=1).squeeze(1)

        deep_input = self.embedding(ids).flatten(1)
        deep = self.deep(deep_input).squeeze(1)
        return self.bias + wide + deep


class GRUHistory(nn.Module):
    """
    Encodes each user's chronologically preceding train impressions with a
    GRU, then matches this dynamic interest state against the candidate video
    and the current impression context.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())

        self.context_embedding = nn.Embedding(TOTAL_CARD, 7)
        self.video_embedding = nn.Embedding(
            int(FEATURE_CARDINALITIES["video_id"]),
            20,
            padding_idx=0,
        )
        self.gru = nn.GRU(
            input_size=20,
            hidden_size=28,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(N_FIELDS * 7 + 28 + 20 + 28, 96),
            nn.PReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 36),
            nn.PReLU(),
            nn.Linear(36, 1),
        )

        nn.init.normal_(self.context_embedding.weight, std=0.025)
        nn.init.normal_(self.video_embedding.weight, std=0.025)
        with torch.no_grad():
            self.video_embedding.weight[0].zero_()

    def forward(self, x, history):
        context = self.context_embedding(
            x + self.offsets
        ).flatten(1)
        sequence = self.video_embedding(history)
        _, hidden = self.gru(sequence)
        user_state = hidden[-1]

        candidate = self.video_embedding(x[:, 1])
        match = user_state * candidate

        features = torch.cat(
            [context, user_state, candidate, match],
            dim=1,
        )
        return self.output(features).squeeze(1)


@torch.no_grad()
def predict(model, x_np, history_np, batch_size=16384):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    for begin in range(0, len(x_np), batch_size):
        end = min(begin + batch_size, len(x_np))
        xb = torch.from_numpy(x_np[begin:end])
        hb = torch.from_numpy(history_np[begin:end])
        result[begin:end] = (
            model(xb, hb).detach().cpu().numpy()
        )
    return result


def train_model(model, name, epochs=2):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.1e-3,
        weight_decay=3e-6,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + sum(ord(c) for c in name))

    best_primary = -1.0
    best_state = None
    epoch_primaries = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(
            len(y_train_np),
            generator=generator,
        )

        for begin in range(0, len(y_train_np), 4096):
            idx = permutation[begin:begin + 4096]
            xb = x_train[idx]
            hb = hist_train[idx]
            yb = y_train[idx]
            wb = sample_weight[idx]

            logits = model(xb, hb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                yb,
                reduction="none",
            )
            loss = (losses * wb).sum() / wb.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()

        valid_scores = predict(
            model, x_valid_np, hist_valid_np
        )
        metrics = evaluate(
            valid.user_id, valid.y, valid_scores
        )
        primary = float(metrics["primary"])
        epoch_primaries.append(primary)

        if primary > best_primary:
            best_primary = primary
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    valid_scores = predict(model, x_valid_np, hist_valid_np)
    test_scores = predict(model, x_test_np, hist_test_np)

    print(
        "FINDINGS %s_epoch_primary=%s"
        % (name, json.dumps(epoch_primaries))
    )
    return valid_scores, test_scores


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(values), dtype=np.int64)

    order = np.lexsort((rows, values, users))
    sorted_users = users[order]

    new_group = np.r_[
        True, sorted_users[1:] != sorted_users[:-1]
    ]
    starts = np.maximum.accumulate(
        np.where(new_group, np.arange(len(values)), 0)
    )
    position = np.arange(len(values)) - starts

    group_ends = np.r_[
        np.flatnonzero(
            sorted_users[1:] != sorted_users[:-1]
        ) + 1,
        len(values),
    ]
    group_starts = np.r_[0, group_ends[:-1]]
    counts_per_group = group_ends - group_starts
    counts = np.repeat(counts_per_group, counts_per_group)

    ranked_sorted = (
        position.astype(np.float64) + 0.5
    ) / counts.astype(np.float64)
    ranked = np.empty(len(values), dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)

if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path)
inc_test = np.load(inc_test_path)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

families = [
    ("field_aware_fm", FieldAwareFM),
    ("neural_fm", NeuralFM),
    ("wide_deep", WideDeep),
    ("gru_history", GRUHistory),
]

candidate_scores = {}
family_predictions = {}
best_name = None
best_primary = -1.0
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_alpha = None

for family_name, constructor in families:
    torch.manual_seed(
        SEED + sum(ord(c) for c in family_name)
    )
    model = constructor()

    raw_valid, raw_test = train_model(
        model, family_name, epochs=2
    )
    family_predictions[family_name] = (
        raw_valid, raw_test
    )

    standalone_metrics = evaluate(
        valid.user_id, valid.y, raw_valid
    )
    standalone_primary = float(
        standalone_metrics["primary"]
    )
    candidate_scores[
        family_name + "_standalone"
    ] = standalone_primary

    own_valid_rank = within_user_rank(
        valid.user_id, raw_valid
    )
    own_test_rank = within_user_rank(
        test.user_id, raw_test
    )

    for alpha in (0.25, 0.50, 0.75, 0.90):
        blended_valid = (
            alpha * own_valid_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        blended_test = (
            alpha * own_test_rank
            + (1.0 - alpha) * inc_test_rank
        )

        metrics = evaluate(
            valid.user_id, valid.y, blended_valid
        )
        primary = float(metrics["primary"])
        candidate_name = "%s_blend_%.2f" % (
            family_name, alpha
        )
        candidate_scores[candidate_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = candidate_name
            best_valid_scores = blended_valid.copy()
            best_test_scores = blended_test.copy()
            best_raw_valid = raw_valid.copy()
            best_alpha = alpha

    if standalone_primary > best_primary:
        best_primary = standalone_primary
        best_name = family_name + "_standalone"
        best_valid_scores = raw_valid.copy()
        best_test_scores = raw_test.copy()
        best_raw_valid = raw_valid.copy()
        best_alpha = 1.0

    del model
    gc.collect()

print(
    "CANDIDATES "
    + json.dumps(
        candidate_scores,
        sort_keys=True,
    )
)
print(
    "FINDINGS selected=%s alpha=%.2f half_life_days=%.1f"
    % (best_name, best_alpha, HALF_LIFE_DAYS)
)

final_metrics = evaluate(
    valid.user_id,
    valid.y,
    best_valid_scores,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    if best_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)