import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2025
BASE_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
FFM_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
    "onehot_feat1",
    "user_active_degree",
]

BASE_RANK = 16
FFM_RANK = 8
BASE_LR = 0.001
FFM_LR = 0.002
BASE_EPOCHS = 12
FFM_EPOCHS = 7
BASE_BATCH_SIZE = 8192
FFM_BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
BLEND_ALPHAS = [0.25, 0.5, 0.75, 1.0]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


def field_layout(fields):
    cards = [int(FEATURE_CARDINALITIES[f]) for f in fields]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    return cards, offsets, int(sum(cards))


base_cards, base_offsets, base_total = field_layout(BASE_FIELDS)
ffm_cards, ffm_offsets, ffm_total = field_layout(FFM_FIELDS)
n_ffm_fields = len(FFM_FIELDS)


def make_matrix(split, fields, offsets):
    x = np.stack([split.X[f] for f in fields], axis=1).astype(
        np.int64, copy=False
    )
    x = x + offsets[None, :]
    return np.ascontiguousarray(x)


class FactorizationMachine(nn.Module):
    def __init__(self, n_values, rank, unknown_offsets):
        super().__init__()
        self.linear = nn.Embedding(n_values, 1, sparse=True)
        self.factors = nn.Embedding(n_values, rank, sparse=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

        with torch.no_grad():
            for off in unknown_offsets:
                self.linear.weight[int(off)].zero_()
                self.factors.weight[int(off)].zero_()

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.factors(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return linear + interaction


class FieldAwareResidual(nn.Module):
    """
    Each categorical value has a distinct embedding for interaction with
    every destination field. For fields i and j, the interaction is
    <v_{x_i, j}, v_{x_j, i}>.
    """

    def __init__(self, n_values, n_fields, rank, unknown_offsets):
        super().__init__()
        self.n_fields = n_fields
        self.rank = rank
        self.linear = nn.Embedding(n_values, 1, sparse=True)
        self.factors = nn.Embedding(
            n_values, n_fields * rank, sparse=True
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.008)

        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )

        with torch.no_grad():
            for off in unknown_offsets:
                self.linear.weight[int(off)].zero_()
                self.factors.weight[int(off)].zero_()

    def forward(self, x):
        batch_size = x.shape[0]
        linear = self.linear(x).sum(dim=1).squeeze(-1)

        # Shape: batch, source field, destination field, latent dimension.
        all_v = self.factors(x).reshape(
            batch_size, self.n_fields, self.n_fields, self.rank
        )

        left = all_v[:, self.pair_i, self.pair_j, :]
        right = all_v[:, self.pair_j, self.pair_i, :]
        interactions = (left * right).sum(dim=-1).sum(dim=-1)
        return linear + interactions


@torch.inference_mode()
def predict_base(model, x_np, batch_size=PRED_BATCH_SIZE):
    model.eval()
    out = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), batch_size):
        end = min(start + batch_size, len(x_np))
        xb = torch.from_numpy(x_np[start:end])
        out[start:end] = (
            model(xb).cpu().numpy().astype(np.float64, copy=False)
        )
    return out


@torch.inference_mode()
def predict_residual(model, x_np, batch_size=16384):
    model.eval()
    out = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), batch_size):
        end = min(start + batch_size, len(x_np))
        xb = torch.from_numpy(x_np[start:end])
        out[start:end] = (
            model(xb).cpu().numpy().astype(np.float64, copy=False)
        )
    return out


# Load only train and validation during fitting and model selection.
train = load("train")
x_train_base = make_matrix(train, BASE_FIELDS, base_offsets)
x_train_ffm = make_matrix(train, FFM_FIELDS, ffm_offsets)
y_train = np.asarray(train.y, dtype=np.float32).copy()
del train

valid = load("valid")
x_valid_base = make_matrix(valid, BASE_FIELDS, base_offsets)
x_valid_ffm = make_matrix(valid, FFM_FIELDS, ffm_offsets)
y_valid = np.asarray(valid.y, dtype=np.int8).copy()
valid_users = np.asarray(valid.user_id, dtype=np.int64).copy()
del valid

x_train_base_t = torch.from_numpy(x_train_base)
x_train_ffm_t = torch.from_numpy(x_train_ffm)
y_train_t = torch.from_numpy(y_train)

# Stage 1: reproduce and select the strong five-field FM.
base_model = FactorizationMachine(
    base_total, BASE_RANK, base_offsets
)
base_optimizer = torch.optim.SparseAdam(
    base_model.parameters(), lr=BASE_LR
)

rng = np.random.default_rng(SEED)
n_train = len(y_train)
best_base_primary = -np.inf
best_base_metrics = None
best_base_state = None
best_base_epoch = -1

for epoch in range(1, BASE_EPOCHS + 1):
    base_model.train()
    permutation = rng.permutation(n_train)

    for start in range(0, n_train, BASE_BATCH_SIZE):
        idx_np = permutation[start:start + BASE_BATCH_SIZE]
        idx = torch.from_numpy(idx_np)
        xb = x_train_base_t.index_select(0, idx)
        yb = y_train_t.index_select(0, idx)

        base_optimizer.zero_grad(set_to_none=True)
        logits = base_model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        base_optimizer.step()

    base_valid_scores_epoch = predict_base(base_model, x_valid_base)
    metrics = evaluate(valid_users, y_valid, base_valid_scores_epoch)
    print(
        f"base_epoch={epoch} primary={metrics['primary']:.6f} "
        f"gauc={metrics['gauc']:.6f} "
        f"ndcg@5={metrics['ndcg@5']:.6f}",
        flush=True,
    )

    if float(metrics["primary"]) > best_base_primary:
        best_base_primary = float(metrics["primary"])
        best_base_metrics = metrics
        best_base_epoch = epoch
        best_base_state = {
            key: value.detach().cpu().clone()
            for key, value in base_model.state_dict().items()
        }

base_model.load_state_dict(best_base_state)
base_model.eval()

# Cache the selected baseline logits. The FFM is trained as a correction to
# this frozen score, rather than replacing the known-good model.
base_train_scores = predict_base(base_model, x_train_base)
base_valid_scores = predict_base(base_model, x_valid_base)
base_train_scores_t = torch.from_numpy(
    base_train_scores.astype(np.float32, copy=False)
)

base_metrics = evaluate(valid_users, y_valid, base_valid_scores)
print(
    f"selected_base_epoch={best_base_epoch} "
    f"primary={base_metrics['primary']:.6f}",
    flush=True,
)

# Stage 2: train the field-aware residual.
residual_model = FieldAwareResidual(
    ffm_total, n_ffm_fields, FFM_RANK, ffm_offsets
)
residual_optimizer = torch.optim.SparseAdam(
    residual_model.parameters(), lr=FFM_LR
)

best_primary = float(base_metrics["primary"])
best_metrics = base_metrics
best_kind = "base"
best_alpha = 0.0
best_ffm_epoch = 0
best_residual_state = None

candidate_scores = {
    "base_fm": float(base_metrics["primary"])
}

ffm_rng = np.random.default_rng(SEED + 991)

for epoch in range(1, FFM_EPOCHS + 1):
    residual_model.train()
    permutation = ffm_rng.permutation(n_train)

    for start in range(0, n_train, FFM_BATCH_SIZE):
        idx_np = permutation[start:start + FFM_BATCH_SIZE]
        idx = torch.from_numpy(idx_np)

        xb = x_train_ffm_t.index_select(0, idx)
        yb = y_train_t.index_select(0, idx)
        baseline_logits = base_train_scores_t.index_select(0, idx)

        residual_optimizer.zero_grad(set_to_none=True)
        residual = residual_model(xb)
        logits = baseline_logits + residual
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        residual_optimizer.step()

    valid_residual = predict_residual(residual_model, x_valid_ffm)

    epoch_parts = []
    for alpha in BLEND_ALPHAS:
        scores = base_valid_scores + alpha * valid_residual
        metrics = evaluate(valid_users, y_valid, scores)
        name = f"ffm_e{epoch}_a{alpha:.2f}"
        candidate_scores[name] = float(metrics["primary"])
        epoch_parts.append(
            f"a={alpha:.2f}:{metrics['primary']:.6f}"
        )

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_kind = "ffm"
            best_alpha = float(alpha)
            best_ffm_epoch = epoch
            best_residual_state = {
                key: value.detach().cpu().clone()
                for key, value in residual_model.state_dict().items()
            }

    print(
        f"ffm_epoch={epoch} " + " ".join(epoch_parts),
        flush=True,
    )

if best_kind == "ffm":
    residual_model.load_state_dict(best_residual_state)
    final_residual = predict_residual(residual_model, x_valid_ffm)
    final_valid_scores = (
        base_valid_scores + best_alpha * final_residual
    )
else:
    final_valid_scores = base_valid_scores

final_metrics = evaluate(valid_users, y_valid, final_valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_model": best_kind,
            "base_epoch": int(best_base_epoch),
            "ffm_epoch": int(best_ffm_epoch),
            "residual_alpha": float(best_alpha),
            "base_primary": float(base_metrics["primary"]),
            "selected_primary": float(final_metrics["primary"]),
        },
        sort_keys=True,
    ),
    flush=True,
)

# Keep the candidate log compact while retaining the baseline and the best
# alpha from every FFM epoch.
compact_candidates = {"base_fm": candidate_scores["base_fm"]}
for epoch in range(1, FFM_EPOCHS + 1):
    names = [
        f"ffm_e{epoch}_a{alpha:.2f}"
        for alpha in BLEND_ALPHAS
    ]
    best_name = max(names, key=lambda n: candidate_scores[n])
    compact_candidates[best_name] = candidate_scores[best_name]

print(
    "CANDIDATES " + json.dumps(compact_candidates, sort_keys=True),
    flush=True,
)

# Score test only after all fitting and validation selection are complete.
out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    test = load("test")
    x_test_base = make_matrix(test, BASE_FIELDS, base_offsets)
    x_test_ffm = make_matrix(test, FFM_FIELDS, ffm_offsets)
    del test

    test_base_scores = predict_base(base_model, x_test_base)
    if best_kind == "ffm":
        test_residual = predict_residual(
            residual_model, x_test_ffm
        )
        test_scores = test_base_scores + best_alpha * test_residual
    else:
        test_scores = test_base_scores

    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

payload = {
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(payload))