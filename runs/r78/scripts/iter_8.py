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
SEED = 842119
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
FIELD_INDEX = {f: i for i, f in enumerate(FIELDS)}

DIM = 10
HISTORY_LENGTH = 12
BATCH_SIZE = 8192
PRED_BATCH = 32768
EPOCHS = 2

OFFSETS = {}
TOTAL_CARD = 0
for field in FIELDS:
    OFFSETS[field] = TOTAL_CARD
    TOTAL_CARD += int(FEATURE_CARDINALITIES[field])

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
    result = np.empty((n, len(FIELDS)), dtype=np.int32)
    for j, field in enumerate(FIELDS):
        result[:, j] = (
            np.asarray(split.X[field], dtype=np.int64) + OFFSETS[field]
        ).astype(np.int32)
    return result


def rank_transform(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids
    ))
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])

    positions = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    )
    expanded_sizes = np.repeat(sizes, sizes)
    ranked = np.where(
        expanded_sizes > 1,
        positions / np.maximum(expanded_sizes - 1, 1),
        0.5
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def causal_positive_history(split, labels):
    n = len(split.user_id)
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        times,
        users
    ))
    sorted_users = users[order]
    sorted_y = labels[order]

    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])

    cumulative = np.cumsum(sorted_y, dtype=np.int64)
    group_previous_total = cumulative[starts] - sorted_y[starts]
    group_base = np.repeat(group_previous_total, sizes)
    prior_positive_count = cumulative - sorted_y - group_base

    positive_rows = order[sorted_y == 1]
    history_sorted = np.full(
        (n, HISTORY_LENGTH), VIDEO_CARD, dtype=np.int32
    )

    for lag in range(1, HISTORY_LENGTH + 1):
        eligible = prior_positive_count >= lag
        source_index = (
            group_base[eligible] + prior_positive_count[eligible] - lag
        )
        target_rows = np.flatnonzero(eligible)
        history_sorted[target_rows, lag - 1] = videos[
            positive_rows[source_index]
        ].astype(np.int32)

    history = np.empty_like(history_sorted)
    history[order] = history_sorted
    return history


def static_positive_history(base, base_labels, target):
    profile = np.full(
        (USER_CARD, HISTORY_LENGTH), VIDEO_CARD, dtype=np.int32
    )

    base_users = np.asarray(base.user_id, dtype=np.int64)
    base_times = np.asarray(base.time_ms, dtype=np.int64)
    base_videos = np.asarray(base.video_id, dtype=np.int64)
    base_labels = np.asarray(base_labels, dtype=np.int8)
    positive = np.flatnonzero(base_labels == 1)

    if len(positive):
        positive_order = positive[np.lexsort((
            positive,
            base_times[positive],
            base_users[positive]
        ))]
        sorted_users = base_users[positive_order]
        starts = np.r_[0, np.flatnonzero(
            sorted_users[1:] != sorted_users[:-1]
        ) + 1]
        ends = np.r_[starts[1:], len(positive_order)]
        unique_users = sorted_users[starts]

        for lag in range(1, HISTORY_LENGTH + 1):
            source_positions = ends - lag
            eligible = source_positions >= starts
            profile[unique_users[eligible], lag - 1] = base_videos[
                positive_order[source_positions[eligible]]
            ].astype(np.int32)

    target_users = np.asarray(target.user_id, dtype=np.int64)
    safe_users = np.clip(target_users, 0, USER_CARD - 1)
    result = profile[safe_users].copy()
    invalid = (target_users < 0) | (target_users >= USER_CARD)
    result[invalid] = VIDEO_CARD
    return result


class AttentionalFM(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)

        pair_i, pair_j = np.triu_indices(len(FIELDS), k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )

        self.attention = nn.Sequential(
            nn.Linear(DIM, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )
        self.interaction_projection = nn.Linear(DIM, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor(float(intercept)))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history=None):
        embeddings = self.embedding(x)
        products = (
            embeddings[:, self.pair_i, :] *
            embeddings[:, self.pair_j, :]
        )
        weights = torch.softmax(
            self.attention(products).squeeze(-1), dim=1
        )
        pooled = (products * weights.unsqueeze(-1)).sum(dim=1)

        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(1)
            + self.interaction_projection(pooled).squeeze(1)
        )


class DCNv2(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        input_dim = len(FIELDS) * DIM
        self.embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)

        self.cross_w = nn.ModuleList([
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, input_dim),
        ])
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
        )
        self.output = nn.Linear(input_dim + 48, 1)
        self.bias = nn.Parameter(torch.tensor(float(intercept)))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history=None):
        embedded = self.embedding(x).flatten(1)
        x0 = embedded
        crossed = embedded

        for layer in self.cross_w:
            crossed = crossed + x0 * layer(crossed)

        deep = self.deep(embedded)
        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(1)
            + self.output(torch.cat([crossed, deep], dim=1)).squeeze(1)
        )


class SASRecScorer(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.field_embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.video_embedding = nn.Embedding(
            VIDEO_CARD + 1, DIM, padding_idx=VIDEO_CARD
        )
        self.position_embedding = nn.Embedding(HISTORY_LENGTH, DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=2,
            dim_feedforward=40,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.tower = nn.Sequential(
            nn.Linear(len(FIELDS) * DIM + 2 * DIM, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.tensor(float(intercept)))

        nn.init.normal_(self.field_embedding.weight, std=0.025)
        nn.init.normal_(self.video_embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history):
        field_embeddings = self.field_embedding(x)
        candidate_ids = (
            x[:, FIELD_INDEX["video_id"]] - OFFSETS["video_id"]
        )
        candidate = self.video_embedding(candidate_ids)

        chronological = torch.flip(history, dims=[1])
        valid = chronological != VIDEO_CARD
        sequence = self.video_embedding(chronological)

        positions = torch.arange(
            HISTORY_LENGTH, device=x.device
        ).unsqueeze(0)
        sequence = sequence + self.position_embedding(positions)
        sequence = sequence * valid.unsqueeze(-1).float()

        # TransformerEncoder cannot safely receive an entirely masked row.
        safe_mask = ~valid
        empty = ~valid.any(dim=1)
        safe_mask = safe_mask.clone()
        safe_mask[empty, 0] = False

        encoded = self.encoder(
            sequence, src_key_padding_mask=safe_mask
        )
        encoded = encoded * valid.unsqueeze(-1).float()

        attention_logits = (
            encoded * candidate.unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(DIM)
        attention_logits = attention_logits.masked_fill(
            ~valid, -1e4
        )
        attention = torch.softmax(attention_logits, dim=1)
        attention = attention * valid.float()
        attention = attention / attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        interest = (
            encoded * attention.unsqueeze(-1)
        ).sum(dim=1)

        tower_input = torch.cat([
            field_embeddings.flatten(1),
            candidate,
            interest,
        ], dim=1)

        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(1)
            + self.tower(tower_input).squeeze(1)
        )


def make_model(family, intercept):
    if family == "attentional_fm":
        return AttentionalFM(intercept)
    if family == "dcn_v2":
        return DCNv2(intercept)
    if family == "sasrec_history":
        return SASRecScorer(intercept)
    raise ValueError(f"unknown family {family}")


def fit_model(family, split, labels, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_np = categorical_matrix(split)
    y_np = np.asarray(labels, dtype=np.float32)
    history_np = None
    if family == "sasrec_history":
        history_np = causal_positive_history(split, y_np)

    probability = float(np.clip(y_np.mean(), 1e-5, 1 - 1e-5))
    intercept = math.log(probability / (1.0 - probability))
    model = make_model(family, intercept)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-6
    )

    n = len(y_np)
    model.train()
    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        loss_sum = 0.0
        row_sum = 0

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

            loss_sum += float(loss.detach()) * len(indices)
            row_sum += len(indices)

        print("FINDINGS " + json.dumps({
            "family": family,
            "epoch": epoch + 1,
            "train_bce": loss_sum / max(row_sum, 1),
        }, sort_keys=True))

    del x_np, history_np
    gc.collect()
    return model


@torch.no_grad()
def predict_model(model, family, base, base_labels, target):
    x_np = categorical_matrix(target)
    history_np = None
    if family == "sasrec_history":
        history_np = static_positive_history(
            base, base_labels, target
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

        result[start:end] = (
            model(xb, hb).detach().numpy().astype(np.float64)
        )

    del x_np, history_np
    gc.collect()
    return result


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)

inc_valid_path = os.path.join(
    ARTIFACTS, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    ARTIFACTS, "incumbent_test_scores.npy"
)
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

if len(inc_valid) != len(valid.user_id):
    raise RuntimeError("incumbent validation length mismatch")

inc_valid_rank = rank_transform(valid.user_id, inc_valid)
inc_metrics = evaluate(valid.user_id, valid_y, inc_valid_rank)

candidate_scores = {
    "trusted_incumbent": float(inc_metrics["primary"])
}
families = ["attentional_fm", "dcn_v2", "sasrec_history"]
blend_alphas = [0.10, 0.25, 0.50, 0.75, 1.00]

best_primary = float(inc_metrics["primary"])
best_family = None
best_alpha = 0.0
best_valid_scores = inc_valid_rank.copy()

for family_index, family in enumerate(families):
    model = fit_model(
        family, train, train_y, SEED + family_index * 1009
    )
    raw = predict_model(model, family, train, train_y, valid)
    raw_rank = rank_transform(valid.user_id, raw)

    raw_metrics = evaluate(valid.user_id, valid_y, raw_rank)
    candidate_scores[
        family + "_standalone"
    ] = float(raw_metrics["primary"])

    for alpha in blend_alphas:
        blended = (
            (1.0 - alpha) * inc_valid_rank +
            alpha * raw_rank
        )
        metrics = evaluate(valid.user_id, valid_y, blended)
        name = family + "_blend_" + str(alpha)
        candidate_scores[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_family = family
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()

    del model, raw, raw_rank
    gc.collect()

print("CANDIDATES " + json.dumps(
    candidate_scores, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "selected_family": best_family or "trusted_incumbent",
    "selected_new_family_weight": best_alpha,
    "selected_validation_primary": best_primary,
}, sort_keys=True))

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64)
    )

# Produce test predictions from the same selected recipe, refitted on
# train plus validation. If no candidate improves validation, retain the
# trusted incumbent test prediction without fitting on test.
test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("incumbent test length mismatch")
inc_test_rank = rank_transform(test.user_id, inc_test)

if best_family is None:
    test_scores = inc_test_rank
else:
    combined = join_splits(train, valid)
    combined_y = np.concatenate([
        train_y,
        valid_y.astype(np.float32)
    ])
    final_family_index = families.index(best_family)
    final_model = fit_model(
        best_family,
        combined,
        combined_y,
        SEED + final_family_index * 1009
    )
    raw_test = predict_model(
        final_model, best_family, combined, combined_y, test
    )
    raw_test_rank = rank_transform(test.user_id, raw_test)
    test_scores = (
        (1.0 - best_alpha) * inc_test_rank +
        best_alpha * raw_test_rank
    )

    del final_model, raw_test, raw_test_rank, combined, combined_y
    gc.collect()

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

final_metrics = evaluate(
    valid.user_id, valid_y, best_valid_scores
)
elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))