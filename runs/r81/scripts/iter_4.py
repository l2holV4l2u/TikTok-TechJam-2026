import os
import gc
import json
import time
import random
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
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
N_FIELDS = len(FIELDS)
EMBED_DIM = 12
BATCH_SIZE = 8192
EPOCHS = 2
HALF_LIFE = 5.0


def matrix_for(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64) for name in FIELDS
        ]),
        dtype=np.int64,
    )


def temporal_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    endpoint = int(dates.max())
    # All fitting dates are in April 2022.
    age = endpoint - dates
    w = np.exp2(-age.astype(np.float32) / HALF_LIFE)
    return (w / np.mean(w)).astype(np.float32)


def within_user_rank(user_ids, scores):
    u = np.asarray(user_ids)
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, s, u))
    sorted_u = u[order]

    start_mask = np.empty(n, dtype=bool)
    start_mask[0] = True
    start_mask[1:] = sorted_u[1:] != sorted_u[:-1]
    starts = np.flatnonzero(start_mask)
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    group_starts = np.repeat(starts, sizes)
    group_sizes = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - group_starts

    ranked_sorted = (positions + 0.5) / group_sizes
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


class FieldEmbedding(nn.Module):
    def __init__(self, cards, dim):
        super().__init__()
        self.tables = nn.ModuleList([
            nn.Embedding(card, dim) for card in cards
        ])
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.025)

    def forward(self, x):
        return torch.stack(
            [table(x[:, j]) for j, table in enumerate(self.tables)],
            dim=1,
        )


class LatentMF(nn.Module):
    """Low-rank user-item and user-author latent preference model."""
    def __init__(self, cards, dim):
        super().__init__()
        self.user = nn.Embedding(cards[0], dim)
        self.video = nn.Embedding(cards[1], dim)
        self.author = nn.Embedding(cards[2], dim)
        self.user_bias = nn.Embedding(cards[0], 1)
        self.video_bias = nn.Embedding(cards[1], 1)
        self.author_bias = nn.Embedding(cards[2], 1)
        for emb in [self.user, self.video, self.author]:
            nn.init.normal_(emb.weight, std=0.03)
        for emb in [self.user_bias, self.video_bias, self.author_bias]:
            nn.init.zeros_(emb.weight)

    def forward(self, x):
        u = self.user(x[:, 0])
        v = self.video(x[:, 1])
        a = self.author(x[:, 2])
        score = (u * v).sum(1) + 0.7 * (u * a).sum(1)
        score = score + self.user_bias(x[:, 0]).squeeze(1)
        score = score + self.video_bias(x[:, 1]).squeeze(1)
        score = score + 0.7 * self.author_bias(x[:, 2]).squeeze(1)
        return score


class PNN(nn.Module):
    """Product neural network using all explicit field-pair products."""
    def __init__(self, cards, dim):
        super().__init__()
        self.embedding = FieldEmbedding(cards, dim)
        pairs = []
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                pairs.append((i, j))
        self.pairs = pairs
        input_dim = len(cards) * dim + len(pairs)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        e = self.embedding(x)
        products = torch.stack(
            [(e[:, i] * e[:, j]).sum(1) for i, j in self.pairs],
            dim=1,
        )
        z = torch.cat([e.flatten(1), products], dim=1)
        return self.net(z).squeeze(1)


class AutoInt(nn.Module):
    """Self-attention forms context-dependent field interactions."""
    def __init__(self, cards, dim):
        super().__init__()
        self.embedding = FieldEmbedding(cards, dim)
        self.field_position = nn.Parameter(
            torch.zeros(1, len(cards), dim)
        )
        nn.init.normal_(self.field_position, std=0.02)
        self.attn1 = nn.MultiheadAttention(
            dim, num_heads=3, dropout=0.05, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            dim, num_heads=3, dropout=0.05, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(len(cards) * dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        e = self.embedding(x) + self.field_position
        a, _ = self.attn1(e, e, e, need_weights=False)
        e = self.norm1(e + a)
        a, _ = self.attn2(e, e, e, need_weights=False)
        e = self.norm2(e + a)
        return self.head(e.flatten(1)).squeeze(1)


class MMoE(nn.Module):
    """
    Multi-gate mixture-of-experts. The main long-view task and two auxiliary
    feedback tasks use separate gates over shared experts.
    """
    def __init__(self, cards, dim, n_tasks=3, n_experts=4):
        super().__init__()
        self.embedding = FieldEmbedding(cards, dim)
        input_dim = len(cards) * dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 72),
                nn.ReLU(),
                nn.Linear(72, 40),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, n_experts) for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(40, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x):
        z = self.embedding(x).flatten(1)
        experts = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(z), dim=1).unsqueeze(2)
            mixed = (experts * weights).sum(1)
            outputs.append(tower(mixed).squeeze(1))
        return outputs


def make_model(family):
    if family == "latent_mf":
        return LatentMF(CARDS, 20)
    if family == "pnn":
        return PNN(CARDS, EMBED_DIM)
    if family == "autoint":
        return AutoInt(CARDS, EMBED_DIM)
    if family == "mmoe":
        return MMoE(CARDS, EMBED_DIM)
    raise ValueError(family)


def fit_model(family, x_np, y_np, weights_np, aux_np=None):
    torch.manual_seed(SEED + {
        "latent_mf": 1,
        "pnn": 2,
        "autoint": 3,
        "mmoe": 4,
    }[family])

    model = make_model(family)
    model.train()

    if family == "latent_mf":
        lr = 0.008
        wd = 1e-6
    elif family == "autoint":
        lr = 0.0018
        wd = 2e-5
    else:
        lr = 0.0022
        wd = 2e-5

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd
    )

    x_tensor = torch.from_numpy(x_np)
    y_tensor = torch.from_numpy(
        np.asarray(y_np, dtype=np.float32)
    )
    w_tensor = torch.from_numpy(
        np.asarray(weights_np, dtype=np.float32)
    )
    aux_tensor = None
    if aux_np is not None:
        aux_tensor = torch.from_numpy(
            np.asarray(aux_np, dtype=np.float32)
        )

    n = len(y_np)
    rng = np.random.default_rng(SEED + 100)

    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        total_loss = 0.0
        total_weight = 0.0

        for left in range(0, n, BATCH_SIZE):
            idx_np = permutation[left:left + BATCH_SIZE]
            idx = torch.from_numpy(idx_np)
            xb = x_tensor[idx]
            yb = y_tensor[idx]
            wb = w_tensor[idx]

            optimizer.zero_grad(set_to_none=True)

            if family == "mmoe":
                outputs = model(xb)
                main_loss = F.binary_cross_entropy_with_logits(
                    outputs[0], yb, reduction="none"
                )
                ab = aux_tensor[idx]
                click_loss = F.binary_cross_entropy_with_logits(
                    outputs[1], ab[:, 0], reduction="none"
                )
                like_loss = F.binary_cross_entropy_with_logits(
                    outputs[2], ab[:, 1], reduction="none"
                )
                row_loss = main_loss + 0.30 * click_loss + 0.18 * like_loss
            else:
                logits = model(xb)
                row_loss = F.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )

            loss = (row_loss * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float((row_loss.detach() * wb).sum())
            total_weight += float(wb.sum())

        print(
            "FINDINGS family=%s epoch=%d weighted_loss=%.6f"
            % (family, epoch + 1, total_loss / max(total_weight, 1.0)),
            flush=True,
        )

    return model


def predict_model(model, family, x_np):
    model.eval()
    x_tensor = torch.from_numpy(x_np)
    out = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for left in range(0, len(x_np), BATCH_SIZE * 2):
            right = min(left + BATCH_SIZE * 2, len(x_np))
            result = model(x_tensor[left:right])
            if family == "mmoe":
                result = result[0]
            out[left:right] = result.cpu().numpy().astype(np.float64)
    return out


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train = matrix_for(train)
x_valid = matrix_for(valid)
w_train = temporal_weights(train.date)

aux_train = np.column_stack([
    np.asarray(train.aux["is_click"], dtype=np.float32),
    np.asarray(train.aux["is_like"], dtype=np.float32),
]).astype(np.float32, copy=False)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

if len(inc_valid) != len(y_valid):
    raise RuntimeError("incumbent validation length mismatch")

candidate_scores = {"trusted_incumbent": inc_valid}
candidate_primary = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    )
}
candidate_meta = {
    "trusted_incumbent": ("incumbent", 0.0)
}

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
families = ["latent_mf", "pnn", "autoint", "mmoe"]

for family in families:
    model = fit_model(
        family,
        x_train,
        y_train,
        w_train,
        aux_train if family == "mmoe" else None,
    )
    pred = predict_model(model, family, x_valid)
    raw_metric = evaluate(valid.user_id, y_valid, pred)

    candidate_scores[family] = pred
    candidate_primary[family] = float(raw_metric["primary"])
    candidate_meta[family] = (family, 1.0)

    print(
        "FINDINGS family=%s raw_primary=%.6f raw_gauc=%.6f raw_ndcg5=%.6f"
        % (
            family,
            raw_metric["primary"],
            raw_metric["gauc"],
            raw_metric["ndcg@5"],
        ),
        flush=True,
    )

    family_rank = within_user_rank(valid.user_id, pred)
    for alpha in [0.20, 0.35, 0.50, 0.65, 0.80]:
        name = "%s_rankblend_%.2f" % (family, alpha)
        blended = (1.0 - alpha) * inc_valid_rank + alpha * family_rank
        metric = evaluate(valid.user_id, y_valid, blended)
        candidate_scores[name] = blended
        candidate_primary[name] = float(metric["primary"])
        candidate_meta[name] = (family, float(alpha))

    del model, pred, family_rank
    gc.collect()

best_name = max(candidate_primary, key=candidate_primary.get)
best_valid_scores = candidate_scores[best_name]
selected_family, selected_alpha = candidate_meta[best_name]
best_metrics = evaluate(valid.user_id, y_valid, best_valid_scores)

print(
    "CANDIDATES " + json.dumps(
        candidate_primary, sort_keys=True, separators=(", ", ": ")
    ),
    flush=True,
)
print(
    "FINDINGS selected=%s family=%s incumbent_weight=%.2f new_weight=%.2f"
    % (
        best_name,
        selected_family,
        1.0 - selected_alpha,
        selected_alpha,
    ),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Refit the selected recipe on train + validation and score test. Test labels
# are never accessed.
test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("incumbent test length mismatch")

if selected_family == "incumbent" or selected_alpha == 0.0:
    test_scores = inc_test.copy()
else:
    x_test = matrix_for(test)
    x_combined = np.ascontiguousarray(
        np.concatenate([x_train, x_valid], axis=0),
        dtype=np.int64,
    )
    y_combined = np.concatenate([
        y_train, y_valid
    ]).astype(np.int8, copy=False)
    dates_combined = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])
    w_combined = temporal_weights(dates_combined)

    aux_combined = None
    if selected_family == "mmoe":
        aux_valid = np.column_stack([
            np.asarray(valid.aux["is_click"], dtype=np.float32),
            np.asarray(valid.aux["is_like"], dtype=np.float32),
        ]).astype(np.float32, copy=False)
        aux_combined = np.concatenate(
            [aux_train, aux_valid], axis=0
        ).astype(np.float32, copy=False)

    refit_model = fit_model(
        selected_family,
        x_combined,
        y_combined,
        w_combined,
        aux_combined,
    )
    new_test_raw = predict_model(
        refit_model, selected_family, x_test
    )

    if selected_alpha >= 1.0:
        test_scores = new_test_raw
    else:
        inc_test_rank = within_user_rank(test.user_id, inc_test)
        new_test_rank = within_user_rank(test.user_id, new_test_raw)
        test_scores = (
            (1.0 - selected_alpha) * inc_test_rank
            + selected_alpha * new_test_rank
        )

    del refit_model, x_combined, x_test
    gc.collect()

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
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