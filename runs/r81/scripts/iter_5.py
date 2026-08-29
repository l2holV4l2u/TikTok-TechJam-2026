import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260829
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, max(1, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
N_FIELDS = len(FIELDS)
EMBED_DIM = 8
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 16384
EPOCHS = 3
DEVICE = torch.device("cpu")


def make_x(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64)
            for name in FIELDS
        ]),
        dtype=np.int64,
    )


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    group_start = np.repeat(starts, sizes)
    group_size = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - group_start

    ranked_sorted = (positions + 0.5) / group_size
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def user_balanced_weights(user_ids):
    u = np.asarray(user_ids, dtype=np.int64)
    counts = np.bincount(
        u, minlength=int(FEATURE_CARDINALITIES["user_id"])
    ).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts[u], 1.0))
    w /= w.mean()
    return w.astype(np.float32)


class FieldEmbedding(nn.Module):
    def __init__(self, cards, dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card in cards
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)

    def forward(self, x):
        return torch.stack(
            [emb(x[:, j]) for j, emb in enumerate(self.embeddings)],
            dim=1,
        )


class LatentFM(nn.Module):
    def __init__(self, cards, dim):
        super().__init__()
        self.latent = FieldEmbedding(cards, dim)
        self.linear = nn.ModuleList([
            nn.Embedding(card, 1) for card in cards
        ])
        self.bias = nn.Parameter(torch.zeros(1))
        for emb in self.linear:
            nn.init.zeros_(emb.weight)

    def forward(self, x):
        e = self.latent(x)
        summed = e.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - e.square().sum(dim=(1, 2))
        )
        linear = torch.stack(
            [emb(x[:, j]).squeeze(1)
             for j, emb in enumerate(self.linear)],
            dim=1,
        ).sum(dim=1)
        return self.bias + linear + interaction


class ProductNN(nn.Module):
    def __init__(self, cards, dim):
        super().__init__()
        self.embedding = FieldEmbedding(cards, dim)
        left = []
        right = []
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                left.append(i)
                right.append(j)
        self.register_buffer(
            "pair_left", torch.tensor(left, dtype=torch.long)
        )
        self.register_buffer(
            "pair_right", torch.tensor(right, dtype=torch.long)
        )
        input_dim = len(cards) * dim + len(left)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        e = self.embedding(x)
        products = (
            e[:, self.pair_left, :] * e[:, self.pair_right, :]
        ).sum(dim=2)
        features = torch.cat([e.flatten(1), products], dim=1)
        return self.mlp(features).squeeze(1)


class AutoIntModel(nn.Module):
    def __init__(self, cards, dim):
        super().__init__()
        self.embedding = FieldEmbedding(cards, dim)
        self.attn1 = nn.MultiheadAttention(
            dim, num_heads=2, dropout=0.05, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            dim, num_heads=2, dropout=0.05, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(len(cards) * dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        e = self.embedding(x)
        a1, _ = self.attn1(e, e, e, need_weights=False)
        e = self.norm1(e + F.relu(a1))
        a2, _ = self.attn2(e, e, e, need_weights=False)
        e = self.norm2(e + F.relu(a2))
        return self.head(e.flatten(1)).squeeze(1)


class MMoE(nn.Module):
    def __init__(self, cards, dim, n_tasks):
        super().__init__()
        self.embedding = FieldEmbedding(cards, dim)
        input_dim = len(cards) * dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 48),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, len(self.experts))
            for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x):
        z = self.embedding(x).flatten(1)
        expert_values = torch.stack(
            [expert(z) for expert in self.experts], dim=1
        )
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            gate_weights = torch.softmax(gate(z), dim=1).unsqueeze(2)
            shared = (expert_values * gate_weights).sum(dim=1)
            outputs.append(tower(shared).squeeze(1))
        return torch.stack(outputs, dim=1)


def predict_primary(model, x, is_mmoe=False):
    model.eval()
    result = np.empty(len(x), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x), PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, len(x))
            xb = torch.from_numpy(x[start:end]).to(DEVICE)
            logits = model(xb)
            if is_mmoe:
                logits = logits[:, 0]
            result[start:end] = logits.detach().cpu().numpy()
    return result


def train_model(name, model, x_train, y_train, sample_weights,
                multitask_targets=None):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0025 if name != "latent_fm" else 0.0035,
        weight_decay=2e-5,
    )
    n = len(y_train)

    for epoch in range(EPOCHS):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(SEED + epoch * 101 + len(name))
        permutation = torch.randperm(n, generator=generator).numpy()
        running_loss = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = torch.from_numpy(x_train[idx]).to(DEVICE)
            wb = torch.from_numpy(sample_weights[idx]).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if multitask_targets is None:
                yb = torch.from_numpy(
                    y_train[idx].astype(np.float32, copy=False)
                ).to(DEVICE)
                logits = model(xb)
                row_loss = F.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )
                loss = (row_loss * wb).sum() / wb.sum()
            else:
                targets = torch.from_numpy(
                    multitask_targets[idx]
                ).to(DEVICE)
                logits = model(xb)
                task_losses = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )

                primary_loss = task_losses[:, 0]
                auxiliary_loss = task_losses[:, 1:].mean(dim=1)
                row_loss = primary_loss + 0.25 * auxiliary_loss
                loss = (row_loss * wb).sum() / wb.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += float(loss.detach()) * len(idx)
            seen += len(idx)

        print(
            "FINDINGS family=%s epoch=%d train_loss=%.6f"
            % (name, epoch + 1, running_loss / max(seen, 1)),
            flush=True,
        )

    return model


train = load("train")
valid = load("valid")

x_train = make_x(train)
x_valid = make_x(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_weights = user_balanced_weights(train.user_id)

# Only train-row outcomes are accessed, and only as auxiliary labels.
# No validation or test auxiliary column is read.
aux_names = ["is_click", "is_like", "is_follow", "is_comment"]
auxiliary_columns = []
for name in aux_names:
    values = np.asarray(train.aux[name], dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    auxiliary_columns.append(np.clip(values, 0.0, 1.0))

multitask_targets = np.column_stack(
    [y_train] + auxiliary_columns
).astype(np.float32, copy=False)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

if len(inc_valid) != len(y_valid):
    raise RuntimeError("Incumbent validation score length mismatch")

models = {
    "latent_fm": LatentFM(CARDS, EMBED_DIM),
    "pnn": ProductNN(CARDS, EMBED_DIM),
    "autoint": AutoIntModel(CARDS, EMBED_DIM),
    "mmoe": MMoE(
        CARDS, EMBED_DIM, n_tasks=multitask_targets.shape[1]
    ),
}

trained_models = {}
raw_valid_predictions = {}
candidate_scores = {"trusted_incumbent": inc_valid}
candidate_primary = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    )
}
candidate_metadata = {
    "trusted_incumbent": {
        "family": "trusted_incumbent",
        "alpha": 0.0,
    }
}

for family, model in models.items():
    targets = multitask_targets if family == "mmoe" else None
    model = train_model(
        family,
        model,
        x_train,
        y_train,
        train_weights,
        multitask_targets=targets,
    )
    pred = predict_primary(
        model, x_valid, is_mmoe=(family == "mmoe")
    )
    metric = evaluate(valid.user_id, y_valid, pred)

    trained_models[family] = model
    raw_valid_predictions[family] = pred
    candidate_scores[family] = pred
    candidate_primary[family] = float(metric["primary"])
    candidate_metadata[family] = {
        "family": family,
        "alpha": 1.0,
    }

    print(
        "FINDINGS family=%s primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            family,
            metric["primary"],
            metric["gauc"],
            metric["ndcg@5"],
        ),
        flush=True,
    )

inc_rank = within_user_rank(valid.user_id, inc_valid)

for family, pred in raw_valid_predictions.items():
    family_rank = within_user_rank(valid.user_id, pred)
    for alpha in [0.20, 0.35, 0.50, 0.65, 0.80]:
        name = "%s_inc_rankblend_%.2f" % (family, alpha)
        scores = (1.0 - alpha) * inc_rank + alpha * family_rank
        metric = evaluate(valid.user_id, y_valid, scores)
        candidate_scores[name] = scores
        candidate_primary[name] = float(metric["primary"])
        candidate_metadata[name] = {
            "family": family,
            "alpha": float(alpha),
        }

best_name = max(candidate_primary, key=candidate_primary.get)
best_valid_scores = candidate_scores[best_name]
best_metadata = candidate_metadata[best_name]
best_family = best_metadata["family"]
best_alpha = float(best_metadata["alpha"])
best_metrics = evaluate(valid.user_id, y_valid, best_valid_scores)

print(
    "CANDIDATES " + json.dumps(
        candidate_primary, sort_keys=True, separators=(", ", ": ")
    ),
    flush=True,
)
print(
    "FINDINGS selected=%s family=%s incumbent_weight=%.2f "
    "family_weight=%.2f aux_access=train_only"
    % (
        best_name,
        best_family,
        1.0 - best_alpha,
        best_alpha,
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Score test without reading test labels or any test auxiliary outcomes.
test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if len(inc_test) != len(test.user_id):
    raise RuntimeError("Incumbent test score length mismatch")

if best_family == "trusted_incumbent" or best_alpha == 0.0:
    test_scores = inc_test.copy()
else:
    x_test = make_x(test)
    selected_model = trained_models[best_family]
    family_test = predict_primary(
        selected_model,
        x_test,
        is_mmoe=(best_family == "mmoe"),
    )

    incumbent_test_rank = within_user_rank(test.user_id, inc_test)
    family_test_rank = within_user_rank(test.user_id, family_test)
    test_scores = (
        (1.0 - best_alpha) * incumbent_test_rank
        + best_alpha * family_test_rank
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }, separators=(", ", ": ")),
    flush=True,
)