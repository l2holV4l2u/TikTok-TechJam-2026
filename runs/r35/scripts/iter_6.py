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
    ("deepfm_731_unweighted", "deepfm", 731, "uniform"),
    ("dcnv2_731_unweighted", "dcnv2", 731, "uniform"),
    ("deepfm_991_user_sqrt", "deepfm", 991, "user_sqrt"),
    ("deepfm_313_metric_aligned", "deepfm", 313, "metric_aligned"),
]

EMBED_DIM = 16
DCN_RANK = 32
DCN_LAYERS = 3
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
    _, inverse = np.unique(np.asarray(user_ids), return_inverse=True)

    counts = np.bincount(inverse).astype(np.float64)
    positives = np.bincount(inverse, weights=labels).astype(np.float64)

    if mode == "uniform":
        row_weights = np.ones(labels.shape[0], dtype=np.float64)

    elif mode == "user_sqrt":
        # Each user's total training mass is proportional to sqrt(history size),
        # rather than history size as in ordinary row-wise BCE.
        row_weights = 1.0 / np.sqrt(counts[inverse])

    elif mode == "metric_aligned":
        # nDCG is averaged equally over users, while GAUC weights users by
        # positive count. Give each user a 50/50 mixture of these masses and
        # distribute that mass uniformly over their rows.
        positive_mean = max(float(positives.mean()), 1e-12)
        user_mass = 0.5 + 0.5 * positives / positive_mean
        row_weights = user_mass[inverse] / counts[inverse]

        # Avoid a few tiny histories producing extreme stochastic gradients.
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
    def __init__(self, cardinality, num_fields, embedding_dim, architecture):
        super().__init__()
        self.architecture = architecture
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        dense_dim = num_fields * embedding_dim

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
            nn.init.zeros_(self.deep[-1].bias)

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
        fm_interaction = 0.5 * (
            summed.square().sum(dim=1)
            - latent.square().sum(dim=(1, 2))
        )

        dense_input = latent.flatten(start_dim=1)

        if self.architecture == "deepfm":
            nonlinear_term = self.deep(dense_input).squeeze(-1)
        else:
            crossed = dense_input
            for layer in self.cross_layers:
                crossed = layer(dense_input, crossed)
            nonlinear_term = self.cross_output(crossed).squeeze(-1)

        return self.bias + linear_term + fm_interaction + nonlinear_term


def predict(model, x, batch_size=PREDICT_BATCH_SIZE):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            result[start:stop] = model(x[start:stop]).cpu().numpy()

    return result


def train_one(
    architecture,
    seed,
    x_train,
    y_train,
    train_weights,
    x_valid,
    valid,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = InteractionModel(
        cardinality=TOTAL_CARDINALITY,
        num_fields=len(FIELDS),
        embedding_dim=EMBED_DIM,
        architecture=architecture,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    generator = torch.Generator()
    generator.manual_seed(seed)

    best_primary = -np.inf
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        permutation = torch.randperm(
            x_train.shape[0],
            generator=generator,
        )

        for start in range(0, x_train.shape[0], BATCH_SIZE):
            batch_idx = permutation[start:start + BATCH_SIZE]

            logits = model(x_train[batch_idx])
            losses = criterion(logits, y_train[batch_idx])
            batch_weights = train_weights[batch_idx]
            loss = (losses * batch_weights).sum() / batch_weights.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        valid_scores = predict(model, x_valid)
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
    final_scores = predict(model, x_valid)
    return model, final_scores, best_epoch + 1


def combine_predictions(matrix, coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    coefficients = coefficients / coefficients.sum()
    return np.tensordot(coefficients, matrix, axes=(0, 0))


train = load("train")
valid = load("valid")

x_train = encode_split(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = encode_split(valid)

weight_tensors = {
    mode: make_training_weights(train.user_id, train.y, mode)
    for mode in ("uniform", "user_sqrt", "metric_aligned")
}

models = []
valid_predictions = []
best_epochs = []

for model_name, architecture, seed, weighting_mode in MODEL_SPECS:
    model, model_valid_scores, best_epoch = train_one(
        architecture=architecture,
        seed=seed,
        x_train=x_train,
        y_train=y_train,
        train_weights=weight_tensors[weighting_mode],
        x_valid=x_valid,
        valid=valid,
    )
    models.append(model)
    valid_predictions.append(model_valid_scores)
    best_epochs.append(best_epoch)

valid_matrix = np.stack(valid_predictions, axis=0)

# Models 0 and 1 exactly reconstruct the best previous heterogeneous
# seed-731 logit ensemble. The remaining candidates test whether differently
# weighted objectives add complementary ranking information.
candidate_coefficients = {
    "incumbent_heterogeneous_731": [0.5, 0.5, 0.0, 0.0],
    "user_sqrt_only": [0.0, 0.0, 1.0, 0.0],
    "metric_aligned_only": [0.0, 0.0, 0.0, 1.0],
    "weighted_pair": [0.0, 0.0, 0.5, 0.5],
    "incumbent_plus_sqrt_25": [0.375, 0.375, 0.25, 0.0],
    "incumbent_plus_sqrt_50": [0.25, 0.25, 0.50, 0.0],
    "incumbent_plus_metric_25": [0.375, 0.375, 0.0, 0.25],
    "incumbent_plus_metric_50": [0.25, 0.25, 0.0, 0.50],
    "incumbent_plus_weighted_25": [0.375, 0.375, 0.125, 0.125],
    "incumbent_plus_weighted_50": [0.25, 0.25, 0.25, 0.25],
    "incumbent_plus_weighted_75": [0.125, 0.125, 0.375, 0.375],
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

prediction_correlations = np.corrcoef(valid_matrix)
weight_findings = {}
for mode, weights in weight_tensors.items():
    values = weights.numpy()
    weight_findings[mode] = {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }

print(
    "FINDINGS "
    + json.dumps(
        {
            "best_epochs": {
                MODEL_SPECS[i][0]: best_epochs[i]
                for i in range(len(MODEL_SPECS))
            },
            "selected": selected_name,
            "training_weight_distributions": weight_findings,
            "sqrt_vs_incumbent_deepfm_correlation": float(
                prediction_correlations[0, 2]
            ),
            "metric_vs_incumbent_deepfm_correlation": float(
                prediction_correlations[0, 3]
            ),
            "sqrt_vs_metric_correlation": float(
                prediction_correlations[2, 3]
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
    needed_indices = np.flatnonzero(selected_coefficients_array != 0.0)

    test_scores = np.zeros(x_test.shape[0], dtype=np.float64)
    coefficient_sum = selected_coefficients_array[needed_indices].sum()

    for index in needed_indices:
        model_scores = predict(models[int(index)], x_test)
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