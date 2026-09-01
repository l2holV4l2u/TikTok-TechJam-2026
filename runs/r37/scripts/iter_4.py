import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2027

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

TASK_NAMES = ["long_view", "is_click", "is_like", "is_follow"]
TASK_WEIGHTS = [1.0, 0.25, 0.10, 0.06]

EMBED_DIM = 12
N_EXPERTS = 2
EXPERT_HIDDEN = 48
EXPERT_OUT = 24
TOWER_HIDDEN = 16

BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 7
LR_EMBED = 0.001
LR_DENSE = 0.001
ALPHAS = [0.0, 0.35, 0.70, 1.0, 1.30]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


def make_offsets():
    offsets = {}
    current = 0
    for field in FIELDS:
        offsets[field] = current
        current += int(FEATURE_CARDINALITIES[field])
    return offsets, current


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack(
            [
                np.asarray(split.X[field], dtype=np.int64) + OFFSETS[field]
                for field in FIELDS
            ]
        ),
        dtype=np.int64,
    )


class MMoEResidualFM(nn.Module):
    def __init__(
        self,
        n_categories,
        n_fields,
        n_base_fields,
        embed_dim,
        n_experts,
        n_tasks,
    ):
        super().__init__()
        self.n_fields = n_fields
        self.n_base_fields = n_base_fields
        self.embed_dim = embed_dim
        self.n_experts = n_experts
        self.n_tasks = n_tasks

        # First column supplies the first-order FM terms. The remaining
        # columns are shared by the target FM and all MMoE tasks.
        self.embedding = nn.Embedding(
            n_categories,
            embed_dim + 1,
            sparse=True,
        )
        self.target_bias = nn.Parameter(torch.zeros(()))

        flat_dim = n_fields * embed_dim

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(flat_dim, EXPERT_HIDDEN),
                    nn.ReLU(),
                    nn.Dropout(0.08),
                    nn.Linear(EXPERT_HIDDEN, EXPERT_OUT),
                    nn.ReLU(),
                )
                for _ in range(n_experts)
            ]
        )

        self.gates = nn.ModuleList(
            [nn.Linear(flat_dim, n_experts) for _ in range(n_tasks)]
        )

        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(EXPERT_OUT, TOWER_HIDDEN),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(TOWER_HIDDEN, 1),
                )
                for _ in range(n_tasks)
            ]
        )

        # Auxiliary task biases let sparse actions establish their base rates
        # without forcing the shared experts to represent those rates.
        self.aux_biases = nn.Parameter(torch.zeros(n_tasks - 1))

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

            for expert in self.experts:
                for module in expert:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)

            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

            for tower in self.towers:
                for module in tower:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)
                # Start each task as a modest residual so the dependable FM
                # path controls early target predictions.
                tower[-1].weight.mul_(0.10)

    def forward(self, x):
        raw = self.embedding(x)
        linear_weights = raw[:, :, 0]
        embeddings = raw[:, :, 1:]

        base_embeddings = embeddings[:, :self.n_base_fields, :]
        base_sum = base_embeddings.sum(dim=1)
        base_interaction = 0.5 * (
            base_sum.square()
            - base_embeddings.square().sum(dim=1)
        ).sum(dim=1)
        fm_logit = (
            self.target_bias
            + linear_weights[:, :self.n_base_fields].sum(dim=1)
            + base_interaction
        )

        flat = embeddings.reshape(embeddings.shape[0], -1)
        expert_outputs = torch.stack(
            [expert(flat) for expert in self.experts],
            dim=1,
        )

        residuals = []
        for task_index in range(self.n_tasks):
            gate_weights = torch.softmax(
                self.gates[task_index](flat),
                dim=1,
            )
            task_representation = (
                expert_outputs * gate_weights.unsqueeze(2)
            ).sum(dim=1)
            residual = self.towers[task_index](
                task_representation
            ).squeeze(1)
            residuals.append(residual)

        target_residual = residuals[0]
        logits = [fm_logit + target_residual]
        for task_index in range(1, self.n_tasks):
            logits.append(
                residuals[task_index] + self.aux_biases[task_index - 1]
            )

        return fm_logit, target_residual, logits


@torch.no_grad()
def predict_components(model, x_np):
    model.eval()
    n_rows = len(x_np)
    fm_scores = np.empty(n_rows, dtype=np.float64)
    residual_scores = np.empty(n_rows, dtype=np.float64)

    for start in range(0, n_rows, PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, n_rows)
        xb = torch.from_numpy(x_np[start:end])
        fm, residual, _ = model(xb)
        fm_scores[start:end] = fm.numpy().astype(np.float64)
        residual_scores[start:end] = residual.numpy().astype(np.float64)

    return fm_scores, residual_scores


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)

target_arrays = [
    np.asarray(train.y, dtype=np.float32),
    np.asarray(train.aux["is_click"], dtype=np.float32),
    np.asarray(train.aux["is_like"], dtype=np.float32),
    np.asarray(train.aux["is_follow"], dtype=np.float32),
]
y_train_np = np.ascontiguousarray(
    np.column_stack(target_arrays),
    dtype=np.float32,
)
y_valid = np.asarray(valid.y)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

model = MMoEResidualFM(
    n_categories=TOTAL_CARDINALITY,
    n_fields=len(FIELDS),
    n_base_fields=len(BASE_FIELDS),
    embed_dim=EMBED_DIM,
    n_experts=N_EXPERTS,
    n_tasks=len(TASK_NAMES),
)

embedding_optimizer = torch.optim.SparseAdam(
    [model.embedding.weight],
    lr=LR_EMBED,
)

dense_parameters = [
    parameter
    for name, parameter in model.named_parameters()
    if name != "embedding.weight"
]
dense_optimizer = torch.optim.Adam(
    dense_parameters,
    lr=LR_DENSE,
    weight_decay=1e-6,
)

generator = torch.Generator()
generator.manual_seed(SEED)

n_train = len(x_train_np)
best_primary = -np.inf
best_metrics = None
best_state = None
best_alpha = None
best_epoch = None
candidate_best = {alpha: -np.inf for alpha in ALPHAS}

for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(n_train, generator=generator)
    epoch_losses = np.zeros(len(TASK_NAMES), dtype=np.float64)

    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x_train[idx]
        yb = y_train[idx]

        embedding_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        _, _, task_logits = model(xb)

        task_losses = [
            F.binary_cross_entropy_with_logits(
                task_logits[task_index],
                yb[:, task_index],
            )
            for task_index in range(len(TASK_NAMES))
        ]

        loss = sum(
            TASK_WEIGHTS[task_index] * task_losses[task_index]
            for task_index in range(len(TASK_NAMES))
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            dense_parameters,
            max_norm=5.0,
        )
        embedding_optimizer.step()
        dense_optimizer.step()

        batch_n = len(idx)
        for task_index, task_loss in enumerate(task_losses):
            epoch_losses[task_index] += (
                float(task_loss.detach()) * batch_n
            )

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

    epoch_results.sort(reverse=True, key=lambda item: item[0])
    epoch_primary, epoch_alpha, epoch_metrics = epoch_results[0]

    print(
        "epoch=%d losses=%s alpha=%.2f primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            epoch + 1,
            ",".join(
                "%.5f" % (value / n_train)
                for value in epoch_losses
            ),
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

with torch.no_grad():
    gate_means = []
    sample_n = min(50000, len(x_valid_np))
    sample_x = torch.from_numpy(x_valid_np[:sample_n])
    raw = model.embedding(sample_x)[:, :, 1:]
    flat = raw.reshape(raw.shape[0], -1)
    for gate in model.gates:
        weights = torch.softmax(gate(flat), dim=1)
        gate_means.append(
            weights.mean(dim=0).numpy().astype(float).tolist()
        )

print(
    "FINDINGS selected_epoch=%d selected_alpha=%.2f fm_std=%.6f residual_std=%.6f "
    "train_rates=%s gate_means=%s"
    % (
        best_epoch,
        best_alpha,
        float(np.std(valid_fm)),
        float(np.std(valid_residual)),
        ",".join(
            "%s:%.5f" % (TASK_NAMES[i], float(target_arrays[i].mean()))
            for i in range(len(TASK_NAMES))
        ),
        json.dumps(gate_means, separators=(",", ":")),
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