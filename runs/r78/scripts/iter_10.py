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
SEED = 918273
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

ARTIFACTS = os.environ["RUN_ARTIFACTS"]
OUT = os.environ.get("ITER_OUT")

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type", "hour",
    "onehot_feat1", "onehot_feat3", "onehot_feat8",
    "user_active_degree", "register_days_bucket",
    "fans_user_num_range",
]
DIM = 10
HISTORY_LENGTH = 10
BATCH_SIZE = 8192
PAIR_BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 2

OFFSETS = {}
TOTAL_CARD = 0
for field in FIELDS:
    OFFSETS[field] = TOTAL_CARD
    TOTAL_CARD += int(FEATURE_CARDINALITIES[field])

FIELD_INDEX = {f: i for i, f in enumerate(FIELDS)}
USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])


class JoinedSplit:
    pass


def join_splits(a, b, include_aux=False):
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
    if include_aux:
        # These are training targets only. They are never passed to a model
        # while scoring validation or test rows.
        z.aux = {
            name: np.concatenate([
                np.asarray(a.aux[name]),
                np.asarray(b.aux[name])
            ])
            for name in ("is_click", "is_like")
        }
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
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        users
    ))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])
    positions = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    )
    expanded_sizes = np.repeat(sizes, sizes)
    sorted_ranks = np.where(
        expanded_sizes > 1,
        positions / np.maximum(expanded_sizes - 1, 1),
        0.5
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_ranks
    return result


def causal_positive_history(split, labels):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)
    n = len(users)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        times,
        users
    ))
    sorted_users = users[order]
    sorted_y = y[order]

    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])

    cumulative = np.cumsum(sorted_y, dtype=np.int64)
    positives_before_group = cumulative[starts] - sorted_y[starts]
    group_base = np.repeat(positives_before_group, sizes)
    prior_positive_count = cumulative - sorted_y - group_base

    positive_rows = order[sorted_y == 1]
    history_sorted = np.full(
        (n, HISTORY_LENGTH), VIDEO_CARD, dtype=np.int32
    )

    for lag in range(1, HISTORY_LENGTH + 1):
        valid = prior_positive_count >= lag
        positive_position = (
            group_base[valid] + prior_positive_count[valid] - lag
        )
        history_sorted[np.flatnonzero(valid), lag - 1] = videos[
            positive_rows[positive_position]
        ].astype(np.int32)

    result = np.empty_like(history_sorted)
    result[order] = history_sorted
    return result


def static_positive_history(base, base_labels, target):
    profile = np.full(
        (USER_CARD, HISTORY_LENGTH), VIDEO_CARD, dtype=np.int32
    )
    base_y = np.asarray(base_labels, dtype=np.int8)
    positive_rows = np.flatnonzero(base_y == 1)

    if len(positive_rows):
        ordered_positive_rows = positive_rows[np.lexsort((
            positive_rows,
            np.asarray(base.time_ms, dtype=np.int64)[positive_rows],
            np.asarray(base.user_id, dtype=np.int64)[positive_rows]
        ))]
        positive_users = np.asarray(
            base.user_id, dtype=np.int64
        )[ordered_positive_rows]
        starts = np.r_[0, np.flatnonzero(
            positive_users[1:] != positive_users[:-1]
        ) + 1]
        ends = np.r_[starts[1:], len(ordered_positive_rows)]
        unique_users = positive_users[starts]
        base_videos = np.asarray(base.video_id, dtype=np.int64)

        for lag in range(1, HISTORY_LENGTH + 1):
            positions = ends - lag
            valid = positions >= starts
            profile[unique_users[valid], lag - 1] = base_videos[
                ordered_positive_rows[positions[valid]]
            ].astype(np.int32)

    target_users = np.asarray(target.user_id, dtype=np.int64)
    safe_users = np.clip(target_users, 0, USER_CARD - 1)
    result = profile[safe_users].copy()
    unseen = (target_users < 0) | (target_users >= USER_CARD)
    result[unseen] = VIDEO_CARD
    return result


class BPRFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        emb = self.embedding(x)
        summed = emb.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - emb.square().sum(dim=(1, 2))
        )
        return self.linear(x).sum(dim=1).squeeze(1) + interaction


class CandidateAwareDIEN(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.field_embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.video_embedding = nn.Embedding(
            VIDEO_CARD + 1, DIM, padding_idx=VIDEO_CARD
        )
        self.history_gru = nn.GRU(DIM, DIM, batch_first=True)
        self.tower = nn.Sequential(
            nn.Linear(len(FIELDS) * DIM + 3 * DIM, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.bias = nn.Parameter(torch.tensor(float(intercept)))

        nn.init.normal_(self.field_embedding.weight, std=0.025)
        nn.init.normal_(self.video_embedding.weight, std=0.025)
        with torch.no_grad():
            self.video_embedding.weight[VIDEO_CARD].zero_()
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history):
        field_emb = self.field_embedding(x)
        candidate_ids = (
            x[:, FIELD_INDEX["video_id"]] - OFFSETS["video_id"]
        )
        candidate = self.video_embedding(candidate_ids)

        history_emb = self.video_embedding(history)
        evolved, _ = self.history_gru(history_emb)
        mask = history != VIDEO_CARD

        attention_logits = (
            evolved * candidate.unsqueeze(1)
        ).sum(dim=2) / math.sqrt(DIM)
        attention_logits = attention_logits.masked_fill(
            ~mask, -1.0e4
        )
        attention = torch.softmax(attention_logits, dim=1)
        attention = attention * mask.float()
        attention = attention / attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)

        interest = (
            evolved * attention.unsqueeze(2)
        ).sum(dim=1)
        has_history = mask.any(dim=1, keepdim=True).float()
        interest = interest * has_history

        tower_input = torch.cat([
            field_emb.flatten(1),
            candidate,
            interest,
            candidate * interest
        ], dim=1)

        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(1)
            + self.tower(tower_input).squeeze(1)
        )


class PLE(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        input_dim = len(FIELDS) * DIM
        expert_dim = 32
        self.embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)

        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU()
            )
            for _ in range(3)
        ])
        self.private_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.ReLU()
                )
                for _ in range(2)
            ])
            for _ in range(3)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, 5) for _ in range(3)
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            for _ in range(3)
        ])
        self.main_bias = nn.Parameter(torch.tensor(float(intercept)))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        dense = self.embedding(x).flatten(1)
        shared = [expert(dense) for expert in self.shared_experts]
        outputs = []

        for task in range(3):
            private = [
                expert(dense)
                for expert in self.private_experts[task]
            ]
            experts = torch.stack(private + shared, dim=1)
            gate = torch.softmax(self.gates[task](dense), dim=1)
            representation = (
                experts * gate.unsqueeze(2)
            ).sum(dim=1)
            outputs.append(self.heads[task](representation).squeeze(1))

        wide = self.linear(x).sum(dim=1).squeeze(1)
        outputs[0] = outputs[0] + wide + self.main_bias
        return outputs


def make_pointwise_model(family, intercept):
    if family == "dien":
        return CandidateAwareDIEN(intercept)
    if family == "ple":
        return PLE(intercept)
    raise ValueError(family)


def fit_bpr(split, labels, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_np = categorical_matrix(split)
    y = np.asarray(labels, dtype=np.int8)
    users = np.asarray(split.user_id, dtype=np.int64)

    positive_rows = np.flatnonzero(y == 1)
    negative_rows = np.flatnonzero(y == 0)
    negative_order = np.argsort(
        users[negative_rows], kind="stable"
    )
    negative_rows = negative_rows[negative_order]
    negative_users = users[negative_rows]

    negative_counts = np.bincount(
        negative_users, minlength=USER_CARD
    )
    negative_starts = np.cumsum(
        np.r_[0, negative_counts[:-1]], dtype=np.int64
    )

    positive_users = users[positive_rows]
    eligible = negative_counts[
        np.clip(positive_users, 0, USER_CARD - 1)
    ] > 0
    positive_rows = positive_rows[eligible]
    positive_users = users[positive_rows]

    model = BPRFM()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-6
    )

    model.train()
    for epoch in range(EPOCHS):
        count_for_positive = negative_counts[positive_users]
        sampled_offsets = (
            rng.random(len(positive_rows)) * count_for_positive
        ).astype(np.int64)
        sampled_negative_rows = negative_rows[
            negative_starts[positive_users] + sampled_offsets
        ]

        permutation = rng.permutation(len(positive_rows))
        total_loss = 0.0
        seen = 0

        for start in range(0, len(permutation), PAIR_BATCH_SIZE):
            pair_indices = permutation[start:start + PAIR_BATCH_SIZE]
            pos_rows = positive_rows[pair_indices]
            neg_rows = sampled_negative_rows[pair_indices]

            pos_x = torch.from_numpy(
                np.ascontiguousarray(x_np[pos_rows])
            ).long()
            neg_x = torch.from_numpy(
                np.ascontiguousarray(x_np[neg_rows])
            ).long()

            optimizer.zero_grad(set_to_none=True)
            difference = model(pos_x) - model(neg_x)
            loss = F.softplus(-difference).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach()) * len(pair_indices)
            seen += len(pair_indices)

        print("FINDINGS " + json.dumps({
            "family": "bpr_fm",
            "epoch": epoch + 1,
            "pairwise_loss": total_loss / max(seen, 1),
            "eligible_positive_pairs": int(len(positive_rows))
        }, sort_keys=True))

    del x_np
    gc.collect()
    return model


def fit_pointwise(family, split, labels, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_np = categorical_matrix(split)
    y_np = np.asarray(labels, dtype=np.float32)
    p = float(np.clip(y_np.mean(), 1e-5, 1.0 - 1e-5))
    intercept = math.log(p / (1.0 - p))

    history_np = None
    aux_np = None
    if family == "dien":
        history_np = causal_positive_history(split, labels)
    elif family == "ple":
        # Auxiliary outcomes are used only as labels on fitted rows.
        # No validation/test row outcome is consumed during prediction.
        click = np.asarray(
            split.aux["is_click"], dtype=np.float32
        )
        like = np.asarray(
            split.aux["is_like"], dtype=np.float32
        )
        aux_np = np.stack([
            np.clip(np.nan_to_num(click, nan=0.0), 0.0, 1.0),
            np.clip(np.nan_to_num(like, nan=0.0), 0.0, 1.0)
        ], axis=1)

    model = make_pointwise_model(family, intercept)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-6
    )

    model.train()
    n = len(y_np)
    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        total_loss = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            xb = torch.from_numpy(
                np.ascontiguousarray(x_np[indices])
            ).long()
            yb = torch.from_numpy(
                np.ascontiguousarray(y_np[indices])
            )

            optimizer.zero_grad(set_to_none=True)

            if family == "dien":
                hb = torch.from_numpy(
                    np.ascontiguousarray(history_np[indices])
                ).long()
                logits = model(xb, hb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            else:
                targets_aux = torch.from_numpy(
                    np.ascontiguousarray(aux_np[indices])
                )
                outputs = model(xb)
                main_loss = F.binary_cross_entropy_with_logits(
                    outputs[0], yb
                )
                click_loss = F.binary_cross_entropy_with_logits(
                    outputs[1], targets_aux[:, 0]
                )
                like_loss = F.binary_cross_entropy_with_logits(
                    outputs[2], targets_aux[:, 1]
                )
                loss = main_loss + 0.15 * click_loss + 0.15 * like_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach()) * len(indices)
            seen += len(indices)

        print("FINDINGS " + json.dumps({
            "family": family,
            "epoch": epoch + 1,
            "training_loss": total_loss / max(seen, 1),
            "validation_aux_used_as_input": False
        }, sort_keys=True))

    del x_np, history_np, aux_np
    gc.collect()
    return model


@torch.no_grad()
def predict_model(model, family, base, base_labels, target):
    x_np = categorical_matrix(target)
    history_np = None
    if family == "dien":
        history_np = static_positive_history(
            base, base_labels, target
        )

    result = np.empty(len(target.user_id), dtype=np.float64)
    model.eval()

    for start in range(0, len(result), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(result))
        xb = torch.from_numpy(
            np.ascontiguousarray(x_np[start:end])
        ).long()

        if family == "bpr_fm":
            logits = model(xb)
        elif family == "dien":
            hb = torch.from_numpy(
                np.ascontiguousarray(history_np[start:end])
            ).long()
            logits = model(xb, hb)
        elif family == "ple":
            logits = model(xb)[0]
        else:
            raise ValueError(family)

        result[start:end] = logits.numpy().astype(np.float64)

    del x_np, history_np
    gc.collect()
    return result


def fit_family(family, split, labels, seed):
    if family == "bpr_fm":
        return fit_bpr(split, labels, seed)
    return fit_pointwise(family, split, labels, seed)


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
inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
if len(inc_valid) != len(valid.user_id):
    raise RuntimeError("incumbent validation length mismatch")

inc_valid_rank = rank_transform(valid.user_id, inc_valid)
inc_metrics = evaluate(
    valid.user_id, valid_y, inc_valid_rank
)

families = ["bpr_fm", "dien", "ple"]
blend_alphas = [0.10, 0.25, 0.50, 0.75, 1.00]

candidate_scores = {
    "trusted_incumbent": float(inc_metrics["primary"])
}
best_primary = float(inc_metrics["primary"])
best_family = None
best_alpha = 0.0
best_valid_scores = inc_valid_rank.copy()

for family_index, family in enumerate(families):
    model = fit_family(
        family, train, train_y,
        SEED + 1009 * (family_index + 1)
    )
    raw_valid = predict_model(
        model, family, train, train_y, valid
    )
    family_rank = rank_transform(valid.user_id, raw_valid)

    standalone_metrics = evaluate(
        valid.user_id, valid_y, family_rank
    )
    candidate_scores[
        family + "_standalone"
    ] = float(standalone_metrics["primary"])

    for alpha in blend_alphas:
        blended = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * family_rank
        )
        metrics = evaluate(valid.user_id, valid_y, blended)
        name = family + "_blend_" + str(alpha)
        candidate_scores[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_family = family
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()

    del model, raw_valid, family_rank
    gc.collect()

print("CANDIDATES " + json.dumps(
    candidate_scores, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "selected_family": (
        best_family if best_family is not None
        else "trusted_incumbent"
    ),
    "selected_new_family_weight": best_alpha,
    "valid_aux_read_during_scoring": False,
    "test_aux_read_during_scoring": False
}, sort_keys=True))

final_metrics = evaluate(
    valid.user_id, valid_y, best_valid_scores
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64)
    )

# Test is loaded only after all validation selection is complete. Its labels
# and auxiliary outcomes are never touched.
test = load("test")
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("incumbent test length mismatch")
inc_test_rank = rank_transform(test.user_id, inc_test)

if best_family is None or best_alpha == 0.0:
    test_scores = inc_test_rank
else:
    include_aux = best_family == "ple"
    combined = join_splits(
        train, valid, include_aux=include_aux
    )
    combined_y = np.concatenate([
        train_y,
        valid_y.astype(np.float32)
    ])

    final_model = fit_family(
        best_family,
        combined,
        combined_y,
        SEED + 70001
    )
    raw_test = predict_model(
        final_model,
        best_family,
        combined,
        combined_y,
        test
    )
    family_test_rank = rank_transform(test.user_id, raw_test)
    test_scores = (
        (1.0 - best_alpha) * inc_test_rank
        + best_alpha * family_test_rank
    )

    del final_model, raw_test, family_test_rank, combined
    gc.collect()

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))