import os
import time
import json
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 731947
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

ARTIFACTS = os.environ["RUN_ARTIFACTS"]
OUT = os.environ.get("ITER_OUT")

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type", "hour",
    "onehot_feat1", "onehot_feat2", "onehot_feat3",
    "onehot_feat7", "onehot_feat8", "user_active_degree",
    "register_days_bucket", "fans_user_num_range",
]
DIM = 10
HISTORY_LENGTH = 10
BATCH_SIZE = 8192
EPOCHS = 2
PRED_BATCH = 32768
HASH_SIZE = 262144

OFFSETS = {}
TOTAL_CARD = 0
for field in FIELDS:
    OFFSETS[field] = TOTAL_CARD
    TOTAL_CARD += int(FEATURE_CARDINALITIES[field])

FIELD_INDEX = {f: i for i, f in enumerate(FIELDS)}
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])
USER_CARD = int(FEATURE_CARDINALITIES["user_id"])


class JoinedSplit:
    pass


def join_splits(a, b):
    z = JoinedSplit()
    z.X = {
        f: np.concatenate([
            np.asarray(a.X[f], dtype=np.int64),
            np.asarray(b.X[f], dtype=np.int64)
        ])
        for f in FIELDS
    }
    z.user_id = np.concatenate([
        np.asarray(a.user_id, dtype=np.int64),
        np.asarray(b.user_id, dtype=np.int64)
    ])
    z.video_id = np.concatenate([
        np.asarray(a.video_id, dtype=np.int64),
        np.asarray(b.video_id, dtype=np.int64)
    ])
    z.time_ms = np.concatenate([
        np.asarray(a.time_ms, dtype=np.int64),
        np.asarray(b.time_ms, dtype=np.int64)
    ])
    return z


def categorical_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS)), dtype=np.int32)
    for j, field in enumerate(FIELDS):
        x[:, j] = (
            np.asarray(split.X[field], dtype=np.int64) + OFFSETS[field]
        ).astype(np.int32)
    return x


def rank_transform(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((
        np.arange(n, dtype=np.int64), scores, user_ids
    ))
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])
    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    repeated_sizes = np.repeat(sizes, sizes)
    ranks = np.where(
        repeated_sizes > 1,
        positions / np.maximum(repeated_sizes - 1, 1),
        0.5
    )
    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def causal_positive_history(split, labels):
    n = len(split.user_id)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        np.asarray(split.time_ms, dtype=np.int64),
        np.asarray(split.user_id, dtype=np.int64)
    ))
    users_sorted = np.asarray(split.user_id, dtype=np.int64)[order]
    y_sorted = np.asarray(labels, dtype=np.int8)[order]

    starts = np.r_[0, np.flatnonzero(
        users_sorted[1:] != users_sorted[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])

    cumulative = np.cumsum(y_sorted, dtype=np.int64)
    before_group = cumulative[starts] - y_sorted[starts]
    group_base = np.repeat(before_group, sizes)
    prior_positive_count = cumulative - y_sorted - group_base

    positive_rows = order[y_sorted == 1]
    videos = np.asarray(split.video_id, dtype=np.int64)
    history_sorted = np.full(
        (n, HISTORY_LENGTH), VIDEO_CARD, dtype=np.int32
    )

    for lag in range(1, HISTORY_LENGTH + 1):
        ok = prior_positive_count >= lag
        positive_index = (
            group_base[ok] + prior_positive_count[ok] - lag
        )
        history_sorted[np.flatnonzero(ok), lag - 1] = videos[
            positive_rows[positive_index]
        ].astype(np.int32)

    history = np.empty_like(history_sorted)
    history[order] = history_sorted
    return history


def static_positive_history(base, base_y, target):
    profile = np.full(
        (USER_CARD, HISTORY_LENGTH), VIDEO_CARD, dtype=np.int32
    )
    positive = np.flatnonzero(np.asarray(base_y, dtype=np.int8) == 1)

    if len(positive):
        positive_order = positive[np.lexsort((
            positive,
            np.asarray(base.time_ms, dtype=np.int64)[positive],
            np.asarray(base.user_id, dtype=np.int64)[positive]
        ))]
        users = np.asarray(base.user_id, dtype=np.int64)[positive_order]
        starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
        ends = np.r_[starts[1:], len(positive_order)]
        unique_users = users[starts]
        videos = np.asarray(base.video_id, dtype=np.int64)

        for lag in range(1, HISTORY_LENGTH + 1):
            positions = ends - lag
            ok = positions >= starts
            profile[unique_users[ok], lag - 1] = videos[
                positive_order[positions[ok]]
            ].astype(np.int32)

    target_users = np.asarray(target.user_id, dtype=np.int64)
    safe_users = np.clip(target_users, 0, USER_CARD - 1)
    result = profile[safe_users].copy()
    result[(target_users < 0) | (target_users >= USER_CARD)] = VIDEO_CARD
    return result


class WideDeep(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.cross_linear = nn.Embedding(HASH_SIZE, 1)
        self.deep = nn.Sequential(
            nn.Linear(len(FIELDS) * DIM, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.bias = nn.Parameter(torch.tensor(float(intercept)))
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.cross_linear.weight)

    def forward(self, x, history=None):
        emb = self.embedding(x)
        local_user = x[:, FIELD_INDEX["user_id"]] - OFFSETS["user_id"]
        local_video = x[:, FIELD_INDEX["video_id"]] - OFFSETS["video_id"]
        local_author = x[:, FIELD_INDEX["author_id"]] - OFFSETS["author_id"]
        local_tag = x[:, FIELD_INDEX["tag"]] - OFFSETS["tag"]
        local_tab = x[:, FIELD_INDEX["tab"]] - OFFSETS["tab"]

        crosses = torch.stack([
            (local_user * 1000003 + local_author * 9176 + 17) % HASH_SIZE,
            (local_user * 1000033 + local_tag * 6151 + 29) % HASH_SIZE,
            (local_video * 1000037 + local_tab * 4099 + 43) % HASH_SIZE,
            (local_author * 1000039 + local_tag * 8191 + 71) % HASH_SIZE,
        ], dim=1)

        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(1)
            + self.cross_linear(crosses).sum(dim=1).squeeze(1)
            + self.deep(emb.flatten(1)).squeeze(1)
        )


class NFM(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.bi_tower = nn.Sequential(
            nn.Linear(DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )
        self.bias = nn.Parameter(torch.tensor(float(intercept)))
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history=None):
        emb = self.embedding(x)
        summed = emb.sum(dim=1)
        bi_interaction = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        )
        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(1)
            + self.bi_tower(bi_interaction).squeeze(1)
        )


class GRUHistory(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.field_embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.video_embedding = nn.Embedding(
            VIDEO_CARD + 1, DIM, padding_idx=VIDEO_CARD
        )
        self.gru = nn.GRU(DIM, DIM, batch_first=True)
        self.tower = nn.Sequential(
            nn.Linear(len(FIELDS) * DIM + 2 * DIM, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.bias = nn.Parameter(torch.tensor(float(intercept)))
        nn.init.normal_(self.field_embedding.weight, std=0.025)
        nn.init.normal_(self.video_embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history):
        fields = self.field_embedding(x)
        candidate_ids = (
            x[:, FIELD_INDEX["video_id"]] - OFFSETS["video_id"]
        )
        candidate = self.video_embedding(candidate_ids)

        # Histories are stored newest-first. Reverse them so the GRU sees
        # the retained positives in chronological order.
        chronological = torch.flip(history, dims=[1])
        hist_emb = self.video_embedding(chronological)
        gru_out, _ = self.gru(hist_emb)

        valid = chronological != VIDEO_CARD
        lengths = valid.sum(dim=1)
        last_index = torch.clamp(lengths - 1, min=0)
        row_index = torch.arange(len(x), device=x.device)
        interest = gru_out[row_index, last_index]
        interest = interest * (lengths > 0).float().unsqueeze(1)

        tower_input = torch.cat([
            fields.flatten(1), candidate, interest
        ], dim=1)
        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(1)
            + self.tower(tower_input).squeeze(1)
        )


def make_model(family, intercept):
    if family == "wide_deep":
        return WideDeep(intercept)
    if family == "nfm":
        return NFM(intercept)
    if family == "gru_history":
        return GRUHistory(intercept)
    raise ValueError(f"unknown family: {family}")


def fit_model(family, split, labels, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_np = categorical_matrix(split)
    y_np = np.asarray(labels, dtype=np.float32)
    history_np = (
        causal_positive_history(split, labels)
        if family == "gru_history" else None
    )

    p = float(np.clip(y_np.mean(), 1e-5, 1 - 1e-5))
    model = make_model(family, math.log(p / (1.0 - p)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-6
    )

    n = len(y_np)
    model.train()
    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        epoch_loss = 0.0
        epoch_rows = 0

        for start in range(0, n, BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            xb = torch.from_numpy(
                np.ascontiguousarray(x_np[indices])
            ).long()
            yb = torch.from_numpy(
                np.ascontiguousarray(y_np[indices])
            )
            hb = None
            if history_np is not None:
                hb = torch.from_numpy(
                    np.ascontiguousarray(history_np[indices])
                ).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += float(loss.detach()) * len(indices)
            epoch_rows += len(indices)

        print(
            "FINDINGS "
            + json.dumps({
                "family": family,
                "epoch": epoch + 1,
                "train_bce": epoch_loss / max(epoch_rows, 1)
            }, sort_keys=True)
        )

    del x_np, history_np
    gc.collect()
    return model


@torch.no_grad()
def predict_model(model, family, base, base_y, target):
    x_np = categorical_matrix(target)
    history_np = (
        static_positive_history(base, base_y, target)
        if family == "gru_history" else None
    )
    result = np.empty(len(target.user_id), dtype=np.float64)

    model.eval()
    for start in range(0, len(result), PRED_BATCH):
        end = min(start + PRED_BATCH, len(result))
        xb = torch.from_numpy(
            np.ascontiguousarray(x_np[start:end])
        ).long()
        hb = None
        if history_np is not None:
            hb = torch.from_numpy(
                np.ascontiguousarray(history_np[start:end])
            ).long()
        result[start:end] = model(xb, hb).numpy().astype(np.float64)

    del x_np, history_np
    gc.collect()
    return result


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)

inc_valid_path = os.path.join(ARTIFACTS, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(ARTIFACTS, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

if len(inc_valid) != len(valid.user_id):
    raise RuntimeError("trusted incumbent validation prediction length mismatch")

inc_valid_rank = rank_transform(valid.user_id, inc_valid)
families = ["wide_deep", "nfm", "gru_history"]
alphas = [0.10, 0.25, 0.50, 0.75, 1.00]

candidate_scores = {}
raw_predictions = {}
best_primary = float(evaluate(
    valid.user_id, valid_y, inc_valid_rank
)["primary"])
best_family = None
best_alpha = 0.0
best_valid_scores = inc_valid_rank.copy()
candidate_scores["trusted_incumbent"] = best_primary

for family_index, family in enumerate(families):
    model = fit_model(
        family, train, train_y, SEED + 1009 * family_index
    )
    raw = predict_model(model, family, train, train_y, valid)
    raw_rank = rank_transform(valid.user_id, raw)
    raw_predictions[family] = raw_rank

    raw_metrics = evaluate(valid.user_id, valid_y, raw_rank)
    candidate_scores[family + "_standalone"] = float(
        raw_metrics["primary"]
    )

    for alpha in alphas:
        blend = (1.0 - alpha) * inc_valid_rank + alpha * raw_rank
        metrics = evaluate(valid.user_id, valid_y, blend)
        name = family + "_blend_" + str(alpha)
        candidate_scores[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_family = family
            best_alpha = float(alpha)
            best_valid_scores = blend.copy()

    del model, raw
    gc.collect()

chosen_metrics = evaluate(
    valid.user_id, valid_y, best_valid_scores, per_user=True
)
print(
    "FINDINGS "
    + json.dumps({
        "selected_family": (
            "trusted_incumbent" if best_family is None else best_family
        ),
        "selected_new_family_weight": best_alpha,
        "validation_users": int(len(chosen_metrics["per_user"]["user_id"])),
        "validation_primary": float(chosen_metrics["primary"])
    }, sort_keys=True)
)

if OUT:
    os.makedirs(OUT, exist_ok=True)
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64)
    )

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("trusted incumbent test prediction length mismatch")
inc_test_rank = rank_transform(test.user_id, inc_test)

if best_family is None or best_alpha == 0.0:
    test_scores = inc_test_rank
else:
    joined = join_splits(train, valid)
    joined_y = np.concatenate([
        train_y, valid_y.astype(np.float32)
    ])
    refit_seed = SEED + 1009 * families.index(best_family)
    final_model = fit_model(
        best_family, joined, joined_y, refit_seed
    )
    new_test_raw = predict_model(
        final_model, best_family, joined, joined_y, test
    )
    new_test_rank = rank_transform(test.user_id, new_test_raw)
    test_scores = (
        (1.0 - best_alpha) * inc_test_rank
        + best_alpha * new_test_rank
    )
    del final_model, new_test_raw, new_test_rank, joined, joined_y
    gc.collect()

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(chosen_metrics["primary"]),
        "gauc": float(chosen_metrics["gauc"]),
        "ndcg@5": float(chosen_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed)
    })
)