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
    ("deepfm_2024", "deepfm", 2024),
    ("deepfm_731", "deepfm", 731),
    ("dcnv2_2024", "dcnv2", 2024),
    ("dcnv2_731", "dcnv2", 731),
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
    def __init__(
        self,
        cardinality,
        num_fields,
        embedding_dim,
        architecture,
    ):
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


def within_user_percentile_ranks(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]

    order = np.lexsort((scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)

    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    positions = np.arange(n, dtype=np.int64)
    repeated_starts = np.repeat(starts, sizes)
    repeated_sizes = np.repeat(sizes, sizes)

    sorted_ranks = (
        positions - repeated_starts + 0.5
    ).astype(np.float64) / repeated_sizes

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def train_one(
    architecture,
    seed,
    x_train,
    y_train,
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
    criterion = nn.BCEWithLogitsLoss()

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
            loss = criterion(logits, y_train[batch_idx])

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


def build_candidate(matrix, rank_matrix, indices, method):
    if method == "single":
        return matrix[indices[0]]
    if method == "logit":
        return matrix[np.asarray(indices)].mean(axis=0)
    if method == "borda":
        return rank_matrix[np.asarray(indices)].mean(axis=0)
    raise ValueError("Unknown aggregation method: " + method)


train = load("train")
valid = load("valid")

x_train = encode_split(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = encode_split(valid)

models = []
valid_predictions = []
best_epochs = []

for model_name, architecture, seed in MODEL_SPECS:
    model, valid_scores_one, best_epoch = train_one(
        architecture,
        seed,
        x_train,
        y_train,
        x_valid,
        valid,
    )
    models.append(model)
    valid_predictions.append(valid_scores_one)
    best_epochs.append(best_epoch)

valid_matrix = np.stack(valid_predictions, axis=0)
valid_rank_matrix = np.stack(
    [
        within_user_percentile_ranks(
            valid.user_id,
            valid_matrix[i],
        )
        for i in range(valid_matrix.shape[0])
    ],
    axis=0,
)

candidate_specs = {}

for i, (model_name, _, _) in enumerate(MODEL_SPECS):
    candidate_specs[model_name] = ((i,), "single")

candidate_specs.update(
    {
        "deepfm_logit_mean": ((0, 1), "logit"),
        "deepfm_borda": ((0, 1), "borda"),
        "dcnv2_logit_mean": ((2, 3), "logit"),
        "dcnv2_borda": ((2, 3), "borda"),
        "heterogeneous_seed2024_logit": ((0, 2), "logit"),
        "heterogeneous_seed2024_borda": ((0, 2), "borda"),
        "heterogeneous_seed731_logit": ((1, 3), "logit"),
        "heterogeneous_seed731_borda": ((1, 3), "borda"),
        "heterogeneous_all_logit": ((0, 1, 2, 3), "logit"),
        "heterogeneous_all_borda": ((0, 1, 2, 3), "borda"),
    }
)

candidate_values = {}
candidate_scores = {}

for name, (indices, method) in candidate_specs.items():
    scores = build_candidate(
        valid_matrix,
        valid_rank_matrix,
        indices,
        method,
    )
    candidate_values[name] = scores
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, scores)["primary"]
    )

selected_name = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_values[selected_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

within_arch_correlations = {
    "deepfm": float(
        np.corrcoef(valid_rank_matrix[0], valid_rank_matrix[1])[0, 1]
    ),
    "dcnv2": float(
        np.corrcoef(valid_rank_matrix[2], valid_rank_matrix[3])[0, 1]
    ),
}

cross_arch_correlations = []
for deep_index in (0, 1):
    for dcn_index in (2, 3):
        cross_arch_correlations.append(
            float(
                np.corrcoef(
                    valid_rank_matrix[deep_index],
                    valid_rank_matrix[dcn_index],
                )[0, 1]
            )
        )

print(
    "FINDINGS "
    + json.dumps(
        {
            "best_epochs": {
                MODEL_SPECS[i][0]: best_epochs[i]
                for i in range(len(MODEL_SPECS))
            },
            "selected": selected_name,
            "within_arch_rank_correlation": within_arch_correlations,
            "mean_cross_arch_rank_correlation": float(
                np.mean(cross_arch_correlations)
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

    selected_indices, selected_method = candidate_specs[selected_name]
    needed_indices = sorted(set(selected_indices))

    test_predictions = {}
    for index in needed_indices:
        test_predictions[index] = predict(models[index], x_test)

    if selected_method == "single":
        test_scores = test_predictions[selected_indices[0]]
    else:
        selected_test_matrix = np.stack(
            [test_predictions[index] for index in selected_indices],
            axis=0,
        )

        if selected_method == "logit":
            test_scores = selected_test_matrix.mean(axis=0)
        else:
            selected_test_rank_matrix = np.stack(
                [
                    within_user_percentile_ranks(
                        test.user_id,
                        selected_test_matrix[i],
                    )
                    for i in range(selected_test_matrix.shape[0])
                ],
                axis=0,
            )
            test_scores = selected_test_rank_matrix.mean(axis=0)

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