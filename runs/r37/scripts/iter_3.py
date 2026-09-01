import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2026
BASE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
]
CONTEXT_FIELDS = [
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
    "register_days_bucket",
    "user_active_degree",
]
FIELDS = BASE_FIELDS + CONTEXT_FIELDS

EMBED_DIM = 12
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
EPOCHS = 7
LR_EMBED = 0.001
LR_DENSE = 0.001
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


def make_offsets():
    offsets = {}
    current = 0
    for name in FIELDS:
        offsets[name] = current
        current += int(FEATURE_CARDINALITIES[name])
    return offsets, current


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack(
            [
                np.asarray(split.X[name], dtype=np.int64) + OFFSETS[name]
                for name in FIELDS
            ]
        ),
        dtype=np.int64,
    )


class FiBiResidualFM(nn.Module):
    def __init__(self, n_categories, n_fields, n_base_fields, embed_dim):
        super().__init__()
        self.n_fields = n_fields
        self.n_base_fields = n_base_fields
        self.embed_dim = embed_dim

        # Column zero is the first-order FM weight; the remainder is shared
        # between the FM and FiBiNET residual.
        self.embedding = nn.Embedding(
            n_categories, embed_dim + 1, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(()))

        se_hidden = max(4, n_fields // 2)
        self.se_reduce = nn.Linear(n_fields, se_hidden)
        self.se_expand = nn.Linear(se_hidden, n_fields)

        # A shared bilinear transform gives each field pair a learned
        # dimension-wise compatibility score without a very large flattened
        # pair-product network.
        self.bilinear = nn.Linear(embed_dim, embed_dim, bias=False)

        left = []
        right = []
        for i in range(n_fields):
            for j in range(i + 1, n_fields):
                left.append(i)
                right.append(j)
        self.register_buffer("pair_left", torch.tensor(left, dtype=torch.long))
        self.register_buffer("pair_right", torch.tensor(right, dtype=torch.long))

        n_pairs = len(left)
        self.deep = nn.Sequential(
            nn.LayerNorm(n_pairs),
            nn.Linear(n_pairs, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(24, 1),
        )

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)
            nn.init.xavier_uniform_(self.bilinear.weight)
            nn.init.xavier_uniform_(self.se_reduce.weight)
            nn.init.zeros_(self.se_reduce.bias)
            nn.init.xavier_uniform_(self.se_expand.weight)
            nn.init.zeros_(self.se_expand.bias)
            for module in self.deep:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            # Begin as a small residual around the reliable FM path.
            self.deep[-1].weight.mul_(0.05)

    def forward(self, x):
        raw = self.embedding(x)
        linear_weights = raw[:, :, 0]
        e = raw[:, :, 1:]

        # Preserve the organizer-style five-field FM as the base path.
        base_e = e[:, :self.n_base_fields, :]
        base_linear = linear_weights[:, :self.n_base_fields].sum(dim=1)
        base_sum = base_e.sum(dim=1)
        base_interaction = 0.5 * (
            base_sum.square() - base_e.square().sum(dim=1)
        ).sum(dim=1)
        fm_logit = self.bias + base_linear + base_interaction

        # FiBiNET squeeze-excitation: infer field importance separately for
        # every impression, then form explicit bilinear field-pair scores.
        squeezed = e.mean(dim=2)
        gates = torch.sigmoid(
            self.se_expand(F.relu(self.se_reduce(squeezed)))
        )
        gated = e * (2.0 * gates.unsqueeze(2))

        transformed = self.bilinear(gated)
        pair_scores = (
            transformed[:, self.pair_left, :]
            * gated[:, self.pair_right, :]
        ).sum(dim=2) / math.sqrt(self.embed_dim)

        deep_residual = self.deep(pair_scores).squeeze(1)
        return fm_logit, deep_residual


@torch.no_grad()
def predict_components(model, x_np):
    model.eval()
    n = len(x_np)
    fm = np.empty(n, dtype=np.float64)
    residual = np.empty(n, dtype=np.float64)

    for start in range(0, n, PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, n)
        xb = torch.from_numpy(x_np[start:end])
        fm_batch, residual_batch = model(xb)
        fm[start:end] = fm_batch.cpu().numpy().astype(np.float64)
        residual[start:end] = residual_batch.cpu().numpy().astype(np.float64)

    return fm, residual


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

model = FiBiResidualFM(
    TOTAL_CARDINALITY,
    len(FIELDS),
    len(BASE_FIELDS),
    EMBED_DIM,
)

embedding_optimizer = torch.optim.SparseAdam(
    [model.embedding.weight], lr=LR_EMBED
)
dense_parameters = [
    parameter
    for name, parameter in model.named_parameters()
    if name != "embedding.weight"
]
dense_optimizer = torch.optim.Adam(
    dense_parameters, lr=LR_DENSE, weight_decay=1e-6
)

generator = torch.Generator()
generator.manual_seed(SEED)

n_train = len(y_train_np)
best_primary = -np.inf
best_metrics = None
best_state = None
best_alpha = None
best_epoch = None
candidate_best = {alpha: -np.inf for alpha in ALPHAS}

for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(n_train, generator=generator)
    total_loss = 0.0

    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x_train[idx]
        yb = y_train[idx]

        embedding_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        fm_logit, residual = model(xb)
        logits = fm_logit + residual
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(dense_parameters, max_norm=5.0)
        embedding_optimizer.step()
        dense_optimizer.step()

        total_loss += float(loss.detach()) * len(idx)

    valid_fm, valid_residual = predict_components(model, x_valid_np)

    epoch_results = []
    for alpha in ALPHAS:
        scores = valid_fm + alpha * valid_residual
        metrics = evaluate(valid.user_id, y_valid, scores)
        primary = float(metrics["primary"])
        candidate_best[alpha] = max(candidate_best[alpha], primary)
        epoch_results.append((primary, alpha, metrics))

        if primary > best_primary:
            best_primary = primary
            best_alpha = alpha
            best_epoch = epoch + 1
            best_metrics = metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    epoch_results.sort(reverse=True, key=lambda z: z[0])
    epoch_primary, epoch_alpha, epoch_metrics = epoch_results[0]
    print(
        "epoch=%d loss=%.6f alpha=%.2f primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            epoch + 1,
            total_loss / n_train,
            epoch_alpha,
            epoch_primary,
            float(epoch_metrics["gauc"]),
            float(epoch_metrics["ndcg@5"]),
        ),
        flush=True,
    )

model.load_state_dict(best_state)

valid_fm, valid_residual = predict_components(model, x_valid_np)
valid_scores = valid_fm + best_alpha * valid_residual
final_metrics = evaluate(valid.user_id, y_valid, valid_scores)

print(
    "FINDINGS selected_epoch=%d selected_residual_alpha=%.2f residual_std=%.6f fm_std=%.6f"
    % (
        best_epoch,
        best_alpha,
        float(np.std(valid_residual)),
        float(np.std(valid_fm)),
    ),
    flush=True,
)

candidate_output = {
    "alpha_%.2f" % alpha: float(candidate_best[alpha])
    for alpha in ALPHAS
}
print(
    "CANDIDATES " + json.dumps(candidate_output, separators=(",", ":")),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    test = load("test")
    x_test_np = make_matrix(test)
    test_fm, test_residual = predict_components(model, x_test_np)
    test_scores = test_fm + best_alpha * test_residual
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

result = {
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(result, separators=(",", ":")))