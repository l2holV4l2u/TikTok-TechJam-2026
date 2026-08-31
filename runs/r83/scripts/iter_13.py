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
SEED = 27183
BATCH_SIZE = 8192
PRED_BATCH = 65536
EPOCHS = 2
EMBED_DIM = 8
HALF_LIFE_DAYS = 7.0
AUX_WEIGHT = 0.18

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "hour",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
]
AUX_NAMES = ["is_click", "is_like", "is_follow"]
FAMILIES = ["hard_shared", "mmoe", "esmm"]
FAMILY_SEEDS = {
    "hard_shared": 101,
    "mmoe": 307,
    "esmm": 509,
}


def make_categorical_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64)
            for name in FIELDS
        ]),
        dtype=np.int64,
    )


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    day = np.searchsorted(unique_dates, dates).astype(np.float32)
    age = float(day.max()) - day
    weights = np.exp2(-age / HALF_LIFE_DAYS).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    denominators = np.repeat(np.maximum(lengths - 1, 1), lengths)
    ranked_sorted = positions.astype(np.float64) / denominators

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def zscore(scores):
    scores = np.asarray(scores, dtype=np.float64)
    std = float(scores.std())
    if std < 1e-12:
        return np.zeros_like(scores)
    return (scores - float(scores.mean())) / std


class CategoricalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList()
        for field in FIELDS:
            card = int(FEATURE_CARDINALITIES[field])
            emb = nn.Embedding(card, EMBED_DIM)
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)
            self.embeddings.append(emb)

        self.output_dim = len(FIELDS) * EMBED_DIM
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        parts = [
            embedding(x[:, j])
            for j, embedding in enumerate(self.embeddings)
        ]
        return self.norm(torch.cat(parts, dim=1))


class HardSharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CategoricalEncoder()
        d = self.encoder.output_dim
        self.shared = nn.Sequential(
            nn.Linear(d, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
        )
        self.skip = nn.Linear(d, 48)
        self.heads = nn.ModuleList([
            nn.Linear(48, 1) for _ in range(1 + len(AUX_NAMES))
        ])

    def forward(self, x):
        encoded = self.encoder(x)
        hidden = self.shared(encoded) + self.skip(encoded)
        return torch.cat([head(hidden) for head in self.heads], dim=1)


class MMoEModel(nn.Module):
    def __init__(self, n_experts=4):
        super().__init__()
        self.encoder = CategoricalEncoder()
        d = self.encoder.output_dim
        self.n_experts = n_experts
        self.n_tasks = 1 + len(AUX_NAMES)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 64),
                nn.SiLU(),
                nn.Linear(64, 40),
                nn.SiLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(d, n_experts) for _ in range(self.n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(40, 24),
                nn.SiLU(),
                nn.Linear(24, 1),
            )
            for _ in range(self.n_tasks)
        ])

    def forward(self, x):
        encoded = self.encoder(x)
        expert_values = torch.stack(
            [expert(encoded) for expert in self.experts], dim=1
        )

        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            gate_weights = torch.softmax(gate(encoded), dim=1).unsqueeze(2)
            task_hidden = (expert_values * gate_weights).sum(dim=1)
            outputs.append(tower(task_hidden))
        return torch.cat(outputs, dim=1)


class ESMMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CategoricalEncoder()
        d = self.encoder.output_dim
        self.shared = nn.Sequential(
            nn.Linear(d, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
        )
        self.click_tower = nn.Sequential(
            nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 1)
        )
        self.conditional_long_tower = nn.Sequential(
            nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 1)
        )
        self.like_tower = nn.Sequential(
            nn.Linear(48, 20), nn.SiLU(), nn.Linear(20, 1)
        )
        self.follow_tower = nn.Sequential(
            nn.Linear(48, 20), nn.SiLU(), nn.Linear(20, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        hidden = self.shared(encoded)

        click_logit = self.click_tower(hidden)
        conditional_long_logit = self.conditional_long_tower(hidden)

        click_probability = torch.sigmoid(click_logit)
        conditional_long_probability = torch.sigmoid(
            conditional_long_logit
        )
        long_probability = (
            click_probability * conditional_long_probability
        ).clamp(1e-6, 1.0 - 1e-6)
        long_logit = torch.log(long_probability) - torch.log1p(
            -long_probability
        )

        like_logit = self.like_tower(hidden)
        follow_logit = self.follow_tower(hidden)
        return torch.cat(
            [long_logit, click_logit, like_logit, follow_logit], dim=1
        )


def construct_model(family):
    if family == "hard_shared":
        return HardSharedModel()
    if family == "mmoe":
        return MMoEModel()
    if family == "esmm":
        return ESMMModel()
    raise ValueError(family)


def fit_model(X, y, aux_targets, aux_mask, weights, family):
    seed = SEED + FAMILY_SEEDS[family]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = construct_model(family)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=3e-5
    )

    xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.int64))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    at = torch.from_numpy(
        np.ascontiguousarray(aux_targets, dtype=np.float32)
    )
    mt = torch.from_numpy(np.asarray(aux_mask, dtype=np.float32))
    wt = torch.from_numpy(np.asarray(weights, dtype=np.float32))

    for _ in range(EPOCHS):
        order = rng.permutation(len(X))
        model.train()

        for start in range(0, len(order), BATCH_SIZE):
            index_np = order[start:start + BATCH_SIZE]
            index = torch.from_numpy(index_np)

            output = model(xt[index])
            batch_weights = wt[index]

            main_losses = F.binary_cross_entropy_with_logits(
                output[:, 0], yt[index], reduction="none"
            )
            main_loss = (
                main_losses * batch_weights
            ).sum() / batch_weights.sum().clamp_min(1.0)

            active = batch_weights * mt[index]
            aux_loss = output.new_tensor(0.0)
            if float(active.sum()) > 0.0:
                for task in range(len(AUX_NAMES)):
                    task_losses = F.binary_cross_entropy_with_logits(
                        output[:, task + 1],
                        at[index, task],
                        reduction="none",
                    )
                    aux_loss = aux_loss + (
                        task_losses * active
                    ).sum() / active.sum().clamp_min(1.0)
                aux_loss = aux_loss / len(AUX_NAMES)

            loss = main_loss + AUX_WEIGHT * aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.inference_mode()
def predict_model(model, X):
    xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.int64))
    result = np.empty(len(X), dtype=np.float32)
    model.eval()

    for start in range(0, len(X), PRED_BATCH):
        end = min(start + PRED_BATCH, len(X))
        result[start:end] = model(xt[start:end])[:, 0].cpu().numpy()

    return result.astype(np.float64)


def form_combination(raw, incumbent, users, mode, alpha):
    raw = np.asarray(raw, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)

    if mode == "raw":
        return raw
    if mode == "zblend":
        return alpha * zscore(incumbent) + (1.0 - alpha) * zscore(raw)
    if mode == "rankblend":
        return (
            alpha * within_user_rank(users, incumbent)
            + (1.0 - alpha) * within_user_rank(users, raw)
        )
    raise ValueError(mode)


train = load("train")
valid = load("valid")

X_train = make_categorical_matrix(train)
X_valid = make_categorical_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

# Only training-split outcome columns are read. Validation auxiliary outcomes
# are never accessed; validation contributes labels only during the permitted
# train+validation refit after model selection.
aux_train = np.ascontiguousarray(
    np.column_stack([
        np.asarray(train.aux[name], dtype=np.float32)
        for name in AUX_NAMES
    ]),
    dtype=np.float32,
)
aux_train = np.clip(aux_train, 0.0, 1.0)
aux_mask_train = np.ones(len(train.y), dtype=np.float32)
weights_train = recency_weights(train.date)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

raw_valid_predictions = {}
candidate_predictions = {}
candidate_specs = {}
candidate_metrics = {}

for family in FAMILIES:
    model = fit_model(
        X_train,
        y_train,
        aux_train,
        aux_mask_train,
        weights_train,
        family,
    )
    raw_scores = predict_model(model, X_valid)
    raw_valid_predictions[family] = raw_scores

    name = family + "_raw"
    candidate_predictions[name] = raw_scores
    candidate_specs[name] = (family, "raw", 0.0)

    for alpha in (0.25, 0.50, 0.75):
        name = family + "_zblend_inc%.2f" % alpha
        candidate_predictions[name] = form_combination(
            raw_scores, inc_valid, valid_users, "zblend", alpha
        )
        candidate_specs[name] = (family, "zblend", alpha)

        name = family + "_rankblend_inc%.2f" % alpha
        candidate_predictions[name] = form_combination(
            raw_scores, inc_valid, valid_users, "rankblend", alpha
        )
        candidate_specs[name] = (family, "rankblend", alpha)

    del model

best_name = None
best_result = None
for name, scores in candidate_predictions.items():
    result = evaluate(valid_users, y_valid, scores)
    candidate_metrics[name] = float(result["primary"])
    if (
        best_result is None
        or result["primary"] > best_result["primary"]
    ):
        best_name = name
        best_result = result

valid_scores = np.asarray(
    candidate_predictions[best_name], dtype=np.float64
)
winning_family, winning_mode, winning_alpha = candidate_specs[best_name]

print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "winner": best_name,
        "family_raw_primary": {
            family: candidate_metrics[family + "_raw"]
            for family in FAMILIES
        },
        "validation_aux_read": False,
        "test_aux_read": False,
    }, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the selected recipe on train+validation long_view labels. Auxiliary
# supervision remains strictly train-only: validation auxiliary columns are
# not read, and their auxiliary-loss mask is zero.
X_combined = np.concatenate([X_train, X_valid], axis=0)
y_combined = np.concatenate([
    y_train,
    np.asarray(valid.y, dtype=np.float32),
])
aux_combined = np.concatenate([
    aux_train,
    np.zeros((len(X_valid), len(AUX_NAMES)), dtype=np.float32),
], axis=0)
aux_mask_combined = np.concatenate([
    np.ones(len(X_train), dtype=np.float32),
    np.zeros(len(X_valid), dtype=np.float32),
])
date_combined = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])
weights_combined = recency_weights(date_combined)

final_model = fit_model(
    X_combined,
    y_combined,
    aux_combined,
    aux_mask_combined,
    weights_combined,
    winning_family,
)

test = load("test")
X_test = make_categorical_matrix(test)
raw_test_scores = predict_model(final_model, X_test)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
test_scores = form_combination(
    raw_test_scores,
    inc_test,
    np.asarray(test.user_id),
    winning_mode,
    winning_alpha,
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        float(best_result["primary"]),
        float(best_result["gauc"]),
        float(best_result["ndcg@5"]),
        float(elapsed),
    )
)