import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "onehot_feat3",
    "upload_type",
    "onehot_feat8",
    "duration_bucket",
    "onehot_feat1",
    "music_type",
    "onehot_feat7",
    "user_active_degree",
    "register_days_bucket",
    "register_days_range",
]

MODEL_SPECS = [
    ("deepfm_731_uniform", "deepfm", 731, "uniform"),
    ("dcnv2_731_uniform", "dcnv2", 731, "uniform"),
    ("deepfm_313_metric", "deepfm", 313, "metric_aligned"),
    ("mmoe_2027_multitask", "mmoe", 2027, "uniform"),
]

AUX_TASKS = ["is_click", "is_like", "is_follow"]

EMBED_DIM = 16
DCN_RANK = 32
DCN_LAYERS = 3
MMOE_EXPERTS = 4
MMOE_EXPERT_DIM = 64

LEARNING_RATE = 0.001
BATCH_SIZE = 4096
PREDICT_BATCH_SIZE = 32768
MAX_EPOCHS = 12
MIN_EPOCHS = 5
PATIENCE = 2

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets(fields):
    offsets = []
    running = 0
    for name in fields:
        offsets.append(running)
        running += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), running


OFFSETS, TOTAL_CARDINALITY = make_offsets(FIELDS)


def encode_split(split):
    columns = []
    for j, name in enumerate(FIELDS):
        values = np.asarray(split.X[name], dtype=np.int64)
        columns.append(values + OFFSETS[j])
    return torch.from_numpy(np.stack(columns, axis=1))


def make_training_weights(user_ids, labels, mode):
    labels = np.asarray(labels, dtype=np.float64)

    if mode == "uniform":
        row_weights = np.ones(labels.shape[0], dtype=np.float64)
    elif mode == "metric_aligned":
        _, inverse = np.unique(np.asarray(user_ids), return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        positives = np.bincount(inverse, weights=labels).astype(np.float64)

        positive_mean = max(float(positives.mean()), 1e-12)
        user_mass = 0.5 + 0.5 * positives / positive_mean
        row_weights = user_mass[inverse] / counts[inverse]

        lower, upper = np.quantile(row_weights, [0.005, 0.995])
        row_weights = np.clip(row_weights, lower, upper)
    else:
        raise ValueError("Unknown weighting mode: " + mode)

    row_weights /= row_weights.mean()
    return torch.from_numpy(row_weights.astype(np.float32))


class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim, rank):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x0, x):
        transformed = self.up(self.down(x)) + self.bias
        return x + x0 * transformed


class InteractionModel(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        self.bias = nn.Parameter(torch.zeros(1))

        dense_dim = len(FIELDS) * EMBED_DIM

        if architecture == "deepfm":
            self.deep = nn.Sequential(
                nn.Linear(dense_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, 1),
            )
            self.cross_layers = None
            self.cross_output = None

            for module in self.deep:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            nn.init.normal_(self.deep[-1].weight, mean=0.0, std=0.01)

        elif architecture == "dcnv2":
            self.deep = None
            self.cross_layers = nn.ModuleList(
                [
                    LowRankCrossLayer(dense_dim, DCN_RANK)
                    for _ in range(DCN_LAYERS)
                ]
            )
            self.cross_output = nn.Linear(dense_dim, 1)
            nn.init.normal_(self.cross_output.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.cross_output.bias)

        else:
            raise ValueError("Unknown architecture: " + architecture)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        fm_term = 0.5 * (
            summed.square().sum(dim=1)
            - latent.square().sum(dim=(1, 2))
        )

        dense_input = latent.flatten(start_dim=1)

        if self.architecture == "deepfm":
            nonlinear = self.deep(dense_input).squeeze(-1)
        else:
            crossed = dense_input
            for layer in self.cross_layers:
                crossed = layer(dense_input, crossed)
            nonlinear = self.cross_output(crossed).squeeze(-1)

        return self.bias + linear_term + fm_term + nonlinear


class MMoEModel(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks

        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        self.task_bias = nn.Parameter(torch.zeros(num_tasks))

        dense_dim = len(FIELDS) * EMBED_DIM

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dense_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.08),
                    nn.Linear(128, MMOE_EXPERT_DIM),
                    nn.ReLU(),
                )
                for _ in range(MMOE_EXPERTS)
            ]
        )

        self.gates = nn.ModuleList(
            [nn.Linear(dense_dim, MMOE_EXPERTS) for _ in range(num_tasks)]
        )

        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(MMOE_EXPERT_DIM, 48),
                    nn.ReLU(),
                    nn.Dropout(0.08),
                    nn.Linear(48, 1),
                )
                for _ in range(num_tasks)
            ]
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        for expert in self.experts:
            for module in expert:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        for gate in self.gates:
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)

        for tower in self.towers:
            for module in tower:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            nn.init.normal_(tower[-1].weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        fm_term = 0.5 * (
            summed.square().sum(dim=1)
            - latent.square().sum(dim=(1, 2))
        )
        common_term = linear_term + fm_term

        dense_input = latent.flatten(start_dim=1)
        expert_outputs = torch.stack(
            [expert(dense_input) for expert in self.experts],
            dim=1,
        )

        task_logits = []
        for task_index in range(self.num_tasks):
            gate_weights = torch.softmax(
                self.gates[task_index](dense_input),
                dim=1,
            )
            mixed = torch.sum(
                expert_outputs * gate_weights.unsqueeze(-1),
                dim=1,
            )
            tower_output = self.towers[task_index](mixed).squeeze(-1)
            task_logits.append(
                common_term + tower_output + self.task_bias[task_index]
            )

        return torch.stack(task_logits, dim=1)


def predict_long_view(model, architecture, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, x.shape[0], PREDICT_BATCH_SIZE):
            stop = min(start + PREDICT_BATCH_SIZE, x.shape[0])
            output = model(x[start:stop])
            if architecture == "mmoe":
                output = output[:, 0]
            result[start:stop] = output.cpu().numpy()

    return result


def train_one(
    architecture,
    seed,
    weighting_mode,
    x_train,
    y_train,
    multitask_targets,
    train_weights,
    x_valid,
    valid,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if architecture == "mmoe":
        model = MMoEModel(num_tasks=multitask_targets.shape[1])
    else:
        model = InteractionModel(architecture)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    generator = torch.Generator()
    generator.manual_seed(seed)

    best_primary = -np.inf
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    # Long view remains the dominant objective. Auxiliary tasks regularize
    # shared experts and embeddings without determining model selection.
    task_coefficients = torch.tensor(
        [1.0, 0.30, 0.20, 0.10],
        dtype=torch.float32,
    )

    for epoch in range(MAX_EPOCHS):
        model.train()
        permutation = torch.randperm(
            x_train.shape[0],
            generator=generator,
        )

        for start in range(0, x_train.shape[0], BATCH_SIZE):
            batch_idx = permutation[start:start + BATCH_SIZE]
            batch_weights = train_weights[batch_idx]

            if architecture == "mmoe":
                logits = model(x_train[batch_idx])
                losses = criterion(
                    logits,
                    multitask_targets[batch_idx],
                )
                per_row_loss = (
                    losses * task_coefficients.unsqueeze(0)
                ).sum(dim=1) / task_coefficients.sum()
            else:
                logits = model(x_train[batch_idx])
                per_row_loss = criterion(
                    logits,
                    y_train[batch_idx],
                )

            loss = (
                per_row_loss * batch_weights
            ).sum() / batch_weights.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        valid_scores = predict_long_view(model, architecture, x_valid)
        epoch_metrics = evaluate(
            valid.user_id,
            valid.y,
            valid_scores,
        )
        epoch_primary = float(epoch_metrics["primary"])

        if epoch_primary > best_primary:
            best_primary = epoch_primary
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch + 1 >= MIN_EPOCHS
            and epochs_without_improvement >= PATIENCE
        ):
            break

    model.load_state_dict(best_state)
    final_scores = predict_long_view(model, architecture, x_valid)
    return model, final_scores, best_epoch + 1


def combine_predictions(matrix, coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    coefficients = coefficients / coefficients.sum()
    return np.tensordot(coefficients, matrix, axes=(0, 0))


train = load("train")
valid = load("valid")

x_train = encode_split(train)
x_valid = encode_split(valid)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

multitask_columns = [
    np.asarray(train.y, dtype=np.float32)
]
for task_name in AUX_TASKS:
    multitask_columns.append(
        np.asarray(train.aux[task_name], dtype=np.float32)
    )
multitask_targets = torch.from_numpy(
    np.stack(multitask_columns, axis=1)
)

weight_tensors = {
    "uniform": make_training_weights(
        train.user_id, train.y, "uniform"
    ),
    "metric_aligned": make_training_weights(
        train.user_id, train.y, "metric_aligned"
    ),
}

models = []
valid_predictions = []
best_epochs = []

for model_name, architecture, seed, weighting_mode in MODEL_SPECS:
    model, scores, best_epoch = train_one(
        architecture=architecture,
        seed=seed,
        weighting_mode=weighting_mode,
        x_train=x_train,
        y_train=y_train,
        multitask_targets=multitask_targets,
        train_weights=weight_tensors[weighting_mode],
        x_valid=x_valid,
        valid=valid,
    )
    models.append(model)
    valid_predictions.append(scores)
    best_epochs.append(best_epoch)

valid_matrix = np.stack(valid_predictions, axis=0)

# The first three coefficients reconstruct the strongest previously selected
# heterogeneous blend. The new candidates only vary the contribution of MMoE.
candidate_coefficients = {
    "incumbent_metric_blend": [0.375, 0.375, 0.250, 0.000],
    "mmoe_only": [0.000, 0.000, 0.000, 1.000],
    "incumbent_plus_mmoe_10": [0.3375, 0.3375, 0.2250, 0.1000],
    "incumbent_plus_mmoe_20": [0.3000, 0.3000, 0.2000, 0.2000],
    "incumbent_plus_mmoe_30": [0.2625, 0.2625, 0.1750, 0.3000],
    "incumbent_plus_mmoe_40": [0.2250, 0.2250, 0.1500, 0.4000],
    "incumbent_plus_mmoe_50": [0.1875, 0.1875, 0.1250, 0.5000],
}

candidate_values = {}
candidate_scores = {}

for name, coefficients in candidate_coefficients.items():
    scores = combine_predictions(valid_matrix, coefficients)
    candidate_values[name] = scores
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, scores)["primary"]
    )

selected_name = max(candidate_scores, key=candidate_scores.get)
selected_coefficients = candidate_coefficients[selected_name]
valid_scores = candidate_values[selected_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

correlations = np.corrcoef(valid_matrix)
aux_rates = {
    "long_view": float(np.mean(train.y))
}
for task_name in AUX_TASKS:
    aux_rates[task_name] = float(
        np.mean(np.asarray(train.aux[task_name]))
    )

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": selected_name,
            "best_epochs": {
                MODEL_SPECS[i][0]: best_epochs[i]
                for i in range(len(MODEL_SPECS))
            },
            "train_task_rates": aux_rates,
            "mmoe_vs_deepfm_correlation": float(correlations[3, 0]),
            "mmoe_vs_dcnv2_correlation": float(correlations[3, 1]),
            "mmoe_vs_metric_deepfm_correlation": float(
                correlations[3, 2]
            ),
        },
        separators=(",", ":"),
    )
)

print(
    "CANDIDATES "
    + json.dumps(candidate_scores, separators=(",", ":"))
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = encode_split(test)

    selected_coefficients_array = np.asarray(
        selected_coefficients,
        dtype=np.float64,
    )
    needed_indices = np.flatnonzero(
        selected_coefficients_array != 0.0
    )
    coefficient_sum = selected_coefficients_array[
        needed_indices
    ].sum()

    test_scores = np.zeros(x_test.shape[0], dtype=np.float64)
    for index in needed_indices:
        model_scores = predict_long_view(
            models[int(index)],
            MODEL_SPECS[int(index)][1],
            x_test,
        )
        test_scores += (
            selected_coefficients_array[index] / coefficient_sum
        ) * model_scores

    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final_metrics = {
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final_metrics, separators=(",", ":")))