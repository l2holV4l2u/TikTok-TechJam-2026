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
SEED = 931771
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

OUT = os.environ.get("ITER_OUT")
ARTIFACTS = os.environ["RUN_ARTIFACTS"]

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type", "hour",
    "onehot_feat1", "onehot_feat2", "onehot_feat3",
    "onehot_feat7", "onehot_feat8", "user_active_degree",
    "register_days_bucket", "fans_user_num_range",
]
FIELD_INDEX = {name: j for j, name in enumerate(FIELDS)}

DIM = 12
HISTORY_LENGTH = 10
BATCH_SIZE = 8192
PRED_BATCH = 32768
EPOCHS = 2

OFFSETS = {}
TOTAL_CARD = 0
for field in FIELDS:
    OFFSETS[field] = TOTAL_CARD
    TOTAL_CARD += int(FEATURE_CARDINALITIES[field])

USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])


class JoinedSplit:
    pass


def join_splits(a, b):
    z = JoinedSplit()
    z.X = {
        field: np.concatenate([
            np.asarray(a.X[field], dtype=np.int64),
            np.asarray(b.X[field], dtype=np.int64),
        ])
        for field in FIELDS
    }
    z.user_id = np.concatenate([
        np.asarray(a.user_id, dtype=np.int64),
        np.asarray(b.user_id, dtype=np.int64),
    ])
    z.video_id = np.concatenate([
        np.asarray(a.video_id, dtype=np.int64),
        np.asarray(b.video_id, dtype=np.int64),
    ])
    z.time_ms = np.concatenate([
        np.asarray(a.time_ms, dtype=np.int64),
        np.asarray(b.time_ms, dtype=np.int64),
    ])
    return z


def categorical_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS)), dtype=np.int32)
    for j, field in enumerate(FIELDS):
        values = np.asarray(split.X[field], dtype=np.int64)
        x[:, j] = (values + OFFSETS[field]).astype(np.int32)
    return x


def rank_transform(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])

    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    denominators = np.maximum(np.repeat(sizes, sizes) - 1, 1)
    ranked = positions / denominators
    ranked[np.repeat(sizes, sizes) == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def causal_positive_history(split, labels):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    n = len(users)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        times,
        users,
    ))
    sorted_users = users[order]
    sorted_videos = videos[order]
    sorted_y = labels[order]

    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])

    global_prior = np.cumsum(sorted_y, dtype=np.int64) - sorted_y
    group_positive_base = global_prior[starts]
    local_prior = global_prior - np.repeat(group_positive_base, sizes)

    positive_videos = sorted_videos[sorted_y == 1]
    history_sorted = np.full(
        (n, HISTORY_LENGTH), VIDEO_CARD, dtype=np.int32
    )

    for lag in range(1, HISTORY_LENGTH + 1):
        eligible = local_prior >= lag
        source = global_prior[eligible] - lag
        history_sorted[eligible, lag - 1] = positive_videos[
            source
        ].astype(np.int32)

    history = np.empty_like(history_sorted)
    history[order] = history_sorted
    return history


def static_positive_history(base, base_labels, target):
    profile = np.full(
        (USER_CARD, HISTORY_LENGTH), VIDEO_CARD, dtype=np.int32
    )

    users = np.asarray(base.user_id, dtype=np.int64)
    times = np.asarray(base.time_ms, dtype=np.int64)
    videos = np.asarray(base.video_id, dtype=np.int64)
    labels = np.asarray(base_labels, dtype=np.int8)

    positive_rows = np.flatnonzero(labels == 1)
    if len(positive_rows):
        positive_order = positive_rows[np.lexsort((
            positive_rows,
            times[positive_rows],
            users[positive_rows],
        ))]
        sorted_users = users[positive_order]
        starts = np.r_[0, np.flatnonzero(
            sorted_users[1:] != sorted_users[:-1]
        ) + 1]
        ends = np.r_[starts[1:], len(positive_order)]
        unique_users = sorted_users[starts]

        for lag in range(1, HISTORY_LENGTH + 1):
            positions = ends - lag
            eligible = positions >= starts
            profile[unique_users[eligible], lag - 1] = videos[
                positive_order[positions[eligible]]
            ].astype(np.int32)

    target_users = np.asarray(target.user_id, dtype=np.int64)
    safe = np.clip(target_users, 0, USER_CARD - 1)
    history = profile[safe].copy()
    invalid = (target_users < 0) | (target_users >= USER_CARD)
    history[invalid] = VIDEO_CARD
    return history


def make_pair_indices(user_ids, labels, seed):
    users = np.asarray(user_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    n = len(users)

    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    sizes = np.diff(np.r_[starts, n])

    rng = np.random.default_rng(seed)
    offsets = np.zeros(len(starts), dtype=np.int64)
    multi = sizes > 1
    offsets[multi] = rng.integers(
        1, sizes[multi], size=int(multi.sum())
    )

    local_positions = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    partner_positions = (
        np.repeat(starts, sizes)
        + (
            local_positions
            + np.repeat(offsets, sizes)
        ) % np.repeat(sizes, sizes)
    )
    partner = order[partner_positions]
    anchor = order

    different = labels[anchor] != labels[partner]
    anchor = anchor[different]
    partner = partner[different]

    swap = labels[anchor] == 0
    positives = anchor.copy()
    negatives = partner.copy()
    positives[swap] = partner[swap]
    negatives[swap] = anchor[swap]
    return positives, negatives


class BPRFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history=None):
        embedded = self.embedding(x)
        summed = embedded.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - embedded.square().sum(dim=1)
        ).sum(dim=1)
        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(-1)
            + interaction
        )


class DIENScorer(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.field_embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.video_embedding = nn.Embedding(
            VIDEO_CARD + 1, DIM, padding_idx=VIDEO_CARD
        )
        self.gru = nn.GRU(DIM, DIM, batch_first=True)
        self.evolution = nn.GRUCell(DIM, DIM)
        self.attention = nn.Sequential(
            nn.Linear(4 * DIM, 40),
            nn.PReLU(),
            nn.Linear(40, 1),
        )
        self.tower = nn.Sequential(
            nn.Linear(len(FIELDS) * DIM + 3 * DIM, 96),
            nn.PReLU(),
            nn.Linear(96, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
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

        chronological = torch.flip(history, dims=[1])
        valid = chronological != VIDEO_CARD
        sequence = self.video_embedding(chronological)
        sequence = sequence * valid.unsqueeze(-1).float()

        outputs, _ = self.gru(sequence)

        candidate_expanded = candidate.unsqueeze(1).expand_as(outputs)
        attention_input = torch.cat([
            outputs,
            candidate_expanded,
            outputs - candidate_expanded,
            outputs * candidate_expanded,
        ], dim=-1)
        attention_logits = self.attention(
            attention_input
        ).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~valid, -1e4)
        attention_weights = torch.sigmoid(attention_logits)
        attention_weights = attention_weights * valid.float()

        state = torch.zeros(
            len(x), DIM, device=x.device, dtype=outputs.dtype
        )
        for step in range(HISTORY_LENGTH):
            proposed = self.evolution(outputs[:, step], state)
            gate = attention_weights[:, step].unsqueeze(-1)
            state = gate * proposed + (1.0 - gate) * state

        pooled = (
            outputs * attention_weights.unsqueeze(-1)
        ).sum(dim=1)
        pooled = pooled / attention_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)

        tower_input = torch.cat([
            fields.flatten(1),
            candidate,
            pooled,
            state,
        ], dim=1)

        return (
            self.bias
            + self.linear(x).sum(dim=1).squeeze(-1)
            + self.tower(tower_input).squeeze(-1)
        )


class PLEScorer(nn.Module):
    def __init__(self, intercept, num_tasks):
        super().__init__()
        input_dim = len(FIELDS) * DIM
        hidden = 64

        self.num_tasks = num_tasks
        self.embedding = nn.Embedding(TOTAL_CARD, DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)

        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            for _ in range(2)
        ])
        self.task_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            for _ in range(num_tasks)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, 3) for _ in range(num_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(num_tasks)
        ])
        self.main_bias = nn.Parameter(torch.tensor(float(intercept)))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, history=None, all_tasks=False):
        embedded = self.embedding(x).flatten(1)
        shared = [expert(embedded) for expert in self.shared_experts]

        outputs = []
        for task in range(self.num_tasks):
            task_specific = self.task_experts[task](embedded)
            experts = torch.stack(
                [shared[0], shared[1], task_specific], dim=1
            )
            weights = torch.softmax(
                self.gates[task](embedded), dim=1
            )
            representation = (
                experts * weights.unsqueeze(-1)
            ).sum(dim=1)
            outputs.append(
                self.towers[task](representation).squeeze(-1)
            )

        logits = torch.stack(outputs, dim=1)
        logits[:, 0] = (
            logits[:, 0]
            + self.main_bias
            + self.linear(x).sum(dim=1).squeeze(-1)
        )
        if all_tasks:
            return logits
        return logits[:, 0]


def get_auxiliary_targets(split):
    preferred = [
        "is_click", "is_like", "is_follow",
        "is_comment", "is_forward", "is_profile_enter",
    ]
    targets = []
    names = []
    for key in preferred:
        if key not in split.aux:
            continue
        values = np.asarray(split.aux[key])
        if len(values) != len(split.user_id):
            continue
        finite = np.isfinite(values)
        if not finite.all():
            values = np.where(finite, values, 0)
        values = (values > 0).astype(np.float32)
        rate = float(values.mean())
        if 0.001 < rate < 0.999:
            targets.append(values)
            names.append(key)
        if len(targets) >= 2:
            break
    return names, targets


def fit_bpr(split, labels, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_np = categorical_matrix(split)
    y_np = np.asarray(labels, dtype=np.float32)
    model = BPRFM()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )

    for epoch in range(EPOCHS):
        positive, negative = make_pair_indices(
            split.user_id, y_np, seed + epoch * 101
        )
        permutation = rng.permutation(len(positive))
        loss_total = 0.0
        rows_total = 0

        model.train()
        for start in range(0, len(permutation), BATCH_SIZE):
            ids = permutation[start:start + BATCH_SIZE]
            pidx = positive[ids]
            nidx = negative[ids]

            xp = torch.from_numpy(
                np.ascontiguousarray(x_np[pidx])
            ).long()
            xn = torch.from_numpy(
                np.ascontiguousarray(x_np[nidx])
            ).long()

            optimizer.zero_grad(set_to_none=True)
            positive_score = model(xp)
            negative_score = model(xn)
            pair_loss = F.softplus(
                -(positive_score - negative_score)
            ).mean()

            # A small pointwise anchor prevents unconstrained global biases
            # while leaving the pairwise objective dominant.
            anchor_logits = torch.cat([
                positive_score, negative_score
            ])
            anchor_labels = torch.cat([
                torch.ones_like(positive_score),
                torch.zeros_like(negative_score),
            ])
            point_loss = F.binary_cross_entropy_with_logits(
                anchor_logits, anchor_labels
            )
            loss = pair_loss + 0.10 * point_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_total += float(loss.detach()) * len(ids)
            rows_total += len(ids)

        print("FINDINGS " + json.dumps({
            "family": "bpr_fm",
            "epoch": epoch + 1,
            "pairs": int(len(positive)),
            "train_loss": loss_total / max(rows_total, 1),
        }, sort_keys=True))

    del x_np
    gc.collect()
    return model


def fit_dien(split, labels, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_np = categorical_matrix(split)
    y_np = np.asarray(labels, dtype=np.float32)
    history_np = causal_positive_history(split, y_np)

    p = float(np.clip(y_np.mean(), 1e-5, 1 - 1e-5))
    model = DIENScorer(math.log(p / (1.0 - p)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )

    n = len(y_np)
    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        loss_total = 0.0
        rows_total = 0
        model.train()

        for start in range(0, n, BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            xb = torch.from_numpy(
                np.ascontiguousarray(x_np[indices])
            ).long()
            hb = torch.from_numpy(
                np.ascontiguousarray(history_np[indices])
            ).long()
            yb = torch.from_numpy(
                np.ascontiguousarray(y_np[indices])
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_total += float(loss.detach()) * len(indices)
            rows_total += len(indices)

        print("FINDINGS " + json.dumps({
            "family": "dien",
            "epoch": epoch + 1,
            "train_bce": loss_total / max(rows_total, 1),
        }, sort_keys=True))

    del x_np, history_np
    gc.collect()
    return model


def fit_ple(split, labels, aux_targets, aux_names, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_np = categorical_matrix(split)
    y_main = np.asarray(labels, dtype=np.float32)
    all_targets = [y_main] + [
        np.asarray(x, dtype=np.float32) for x in aux_targets
    ]
    target_np = np.stack(all_targets, axis=1).astype(np.float32)

    p = float(np.clip(y_main.mean(), 1e-5, 1 - 1e-5))
    model = PLEScorer(
        math.log(p / (1.0 - p)),
        num_tasks=target_np.shape[1],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )

    # Main task remains dominant; sparse auxiliary outcomes regularize
    # representation and expert routing.
    task_weights = torch.ones(target_np.shape[1])
    if len(task_weights) > 1:
        task_weights[1:] = 0.25

    n = len(y_main)
    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        loss_total = 0.0
        rows_total = 0
        model.train()

        for start in range(0, n, BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            xb = torch.from_numpy(
                np.ascontiguousarray(x_np[indices])
            ).long()
            yb = torch.from_numpy(
                np.ascontiguousarray(target_np[indices])
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, all_tasks=True)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            ).mean(dim=0)
            loss = (losses * task_weights).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_total += float(loss.detach()) * len(indices)
            rows_total += len(indices)

        print("FINDINGS " + json.dumps({
            "family": "ple_multitask",
            "epoch": epoch + 1,
            "auxiliary_tasks": aux_names,
            "train_weighted_bce": loss_total / max(rows_total, 1),
        }, sort_keys=True))

    del x_np, target_np
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

    for start in range(0, len(result), PRED_BATCH):
        end = min(start + PRED_BATCH, len(result))
        xb = torch.from_numpy(
            np.ascontiguousarray(x_np[start:end])
        ).long()

        if family == "dien":
            hb = torch.from_numpy(
                np.ascontiguousarray(history_np[start:end])
            ).long()
            logits = model(xb, hb)
        else:
            logits = model(xb)

        result[start:end] = logits.detach().cpu().numpy()

    del x_np, history_np
    gc.collect()
    return result


def best_incumbent_blend(user_ids, labels, incumbent, candidate):
    incumbent_rank = rank_transform(user_ids, incumbent)
    candidate_rank = rank_transform(user_ids, candidate)

    alphas = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    best = None

    for alpha in alphas:
        scores = (
            (1.0 - alpha) * incumbent_rank
            + alpha * candidate_rank
        )
        metrics = evaluate(user_ids, labels, scores)
        record = (
            float(metrics["primary"]),
            float(alpha),
            scores.copy(),
            metrics,
        )
        if best is None or record[0] > best[0]:
            best = record

    return best


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)

incumbent_valid = np.load(
    os.path.join(ARTIFACTS, "incumbent_valid_scores.npy")
).astype(np.float64)

aux_names, train_aux = get_auxiliary_targets(train)
print("FINDINGS " + json.dumps({
    "selected_auxiliary_targets": aux_names,
    "rates": {
        name: float(values.mean())
        for name, values in zip(aux_names, train_aux)
    },
}, sort_keys=True))

families = ["bpr_fm", "dien", "ple_multitask"]
models = {}
raw_predictions = {}
candidate_log = {}
selection_records = {}

for family_index, family in enumerate(families):
    family_seed = SEED + 1000 * family_index

    if family == "bpr_fm":
        model = fit_bpr(train, train_y, family_seed)
    elif family == "dien":
        model = fit_dien(train, train_y, family_seed)
    elif family == "ple_multitask":
        model = fit_ple(
            train, train_y, train_aux, aux_names, family_seed
        )
    else:
        raise ValueError(family)

    valid_prediction = predict_model(
        model, family, train, train_y, valid
    )
    raw_metrics = evaluate(
        valid.user_id, valid_y, valid_prediction
    )
    blend = best_incumbent_blend(
        valid.user_id,
        valid_y,
        incumbent_valid,
        valid_prediction,
    )

    models[family] = model
    raw_predictions[family] = valid_prediction
    selection_records[family] = blend
    candidate_log[family + "_raw"] = float(raw_metrics["primary"])
    candidate_log[family + "_blend"] = float(blend[0])

    print("FINDINGS " + json.dumps({
        "family": family,
        "raw_primary": float(raw_metrics["primary"]),
        "raw_gauc": float(raw_metrics["gauc"]),
        "raw_ndcg@5": float(raw_metrics["ndcg@5"]),
        "best_blend_alpha": float(blend[1]),
        "best_blend_primary": float(blend[0]),
    }, sort_keys=True))

best_family = max(
    families, key=lambda name: selection_records[name][0]
)
best_primary, best_alpha, valid_scores, valid_metrics = (
    selection_records[best_family]
)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Release validation-stage models before the permitted train+validation refit.
models.clear()
raw_predictions.clear()
gc.collect()

test = load("test")
incumbent_test = np.load(
    os.path.join(ARTIFACTS, "incumbent_test_scores.npy")
).astype(np.float64)

if best_alpha == 0.0:
    test_scores = rank_transform(test.user_id, incumbent_test)
else:
    combined = join_splits(train, valid)
    combined_y = np.concatenate([
        train_y,
        valid_y.astype(np.float32),
    ])

    refit_seed = SEED + 1000 * families.index(best_family)

    if best_family == "bpr_fm":
        final_model = fit_bpr(combined, combined_y, refit_seed)
    elif best_family == "dien":
        final_model = fit_dien(combined, combined_y, refit_seed)
    else:
        _, valid_aux = get_auxiliary_targets(valid)
        # Match auxiliary task names exactly. If dictionary iteration or
        # availability differs, fetch by the train-selected names directly.
        valid_aux = []
        for name in aux_names:
            values = np.asarray(valid.aux[name])
            values = np.where(np.isfinite(values), values, 0)
            valid_aux.append((values > 0).astype(np.float32))
        combined_aux = [
            np.concatenate([a, b]).astype(np.float32)
            for a, b in zip(train_aux, valid_aux)
        ]
        final_model = fit_ple(
            combined,
            combined_y,
            combined_aux,
            aux_names,
            refit_seed,
        )

    raw_test = predict_model(
        final_model,
        best_family,
        combined,
        combined_y,
        test,
    )
    test_scores = (
        (1.0 - best_alpha)
        * rank_transform(test.user_id, incumbent_test)
        + best_alpha
        * rank_transform(test.user_id, raw_test)
    )

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("FINDINGS " + json.dumps({
    "selected_family": best_family,
    "selected_blend_alpha": float(best_alpha),
    "elapsed_seconds": float(elapsed),
}, sort_keys=True))

print("METRICS " + json.dumps({
    "primary": float(valid_metrics["primary"]),
    "gauc": float(valid_metrics["gauc"]),
    "ndcg@5": float(valid_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))