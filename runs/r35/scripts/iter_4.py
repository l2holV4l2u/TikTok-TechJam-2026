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

SEEDS = [2024, 731, 9917, 17041]
EMBED_DIM = 16
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


class DeepFM(nn.Module):
    def __init__(self, cardinality, num_fields, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        deep_input_dim = num_fields * embedding_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(self.deep[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.deep[-1].bias)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        fm_interaction = 0.5 * (
            summed.square().sum(dim=1)
            - latent.square().sum(dim=(1, 2))
        )

        deep_term = self.deep(latent.flatten(start_dim=1)).squeeze(-1)
        return self.bias + linear_term + fm_interaction + deep_term


def predict(model, x, batch_size=PREDICT_BATCH_SIZE):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            result[start:stop] = model(x[start:stop]).cpu().numpy()
    return result


def within_user_percentile_ranks(user_ids, scores):
    """
    Convert scores to ascending ordinal percentiles independently per user.
    This preserves each model's ordering while removing seed-specific scale.
    """
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

    row_positions = np.arange(n, dtype=np.int64)
    repeated_starts = np.repeat(starts, sizes)
    repeated_sizes = np.repeat(sizes, sizes)

    # Mid-rank style scaling gives singleton groups a neutral score.
    sorted_ranks = (
        (row_positions - repeated_starts).astype(np.float64) + 0.5
    ) / repeated_sizes

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def train_one(seed, x_train, y_train, x_valid, valid):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = DeepFM(
        cardinality=TOTAL_CARDINALITY,
        num_fields=len(FIELDS),
        embedding_dim=EMBED_DIM,
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
        permutation = torch.randperm(x_train.shape[0], generator=generator)

        for start in range(0, x_train.shape[0], BATCH_SIZE):
            batch_idx = permutation[start:start + BATCH_SIZE]
            logits = model(x_train[batch_idx])
            loss = criterion(logits, y_train[batch_idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        valid_scores = predict(model, x_valid)
        epoch_metrics = evaluate(valid.user_id, valid.y, valid_scores)
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

        if epoch + 1 >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    final_scores = predict(model, x_valid)
    return model, final_scores, best_epoch + 1


train = load("train")
valid = load("valid")

x_train = encode_split(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = encode_split(valid)

models = []
valid_predictions = []
best_epochs = []

for seed in SEEDS:
    model, seed_scores, best_epoch = train_one(
        seed, x_train, y_train, x_valid, valid
    )
    models.append(model)
    valid_predictions.append(seed_scores)
    best_epochs.append(best_epoch)

valid_matrix = np.stack(valid_predictions, axis=0)

candidate_scores = {}
candidate_values = {}

for i, seed in enumerate(SEEDS):
    name = "seed_" + str(seed)
    candidate_values[name] = valid_matrix[i]
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, valid_matrix[i])["primary"]
    )

for count in range(2, len(SEEDS) + 1):
    name = "logit_mean_" + str(count)
    scores = valid_matrix[:count].mean(axis=0)
    candidate_values[name] = scores
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, scores)["primary"]
    )

valid_rank_matrix = np.stack(
    [
        within_user_percentile_ranks(valid.user_id, valid_matrix[i])
        for i in range(len(SEEDS))
    ],
    axis=0,
)

for count in range(2, len(SEEDS) + 1):
    name = "borda_" + str(count)
    scores = valid_rank_matrix[:count].mean(axis=0)
    candidate_values[name] = scores
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, scores)["primary"]
    )

selected_name = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_values[selected_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

correlations = []
for i in range(len(SEEDS)):
    for j in range(i + 1, len(SEEDS)):
        correlations.append(
            float(np.corrcoef(valid_rank_matrix[i], valid_rank_matrix[j])[0, 1])
        )

print(
    "FINDINGS "
    + json.dumps(
        {
            "best_epochs": best_epochs,
            "selected": selected_name,
            "mean_pairwise_rank_correlation": float(np.mean(correlations)),
        },
        separators=(",", ":"),
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, separators=(",", ":")))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = encode_split(test)

    if selected_name.startswith("seed_"):
        selected_seed = int(selected_name.split("_")[1])
        model_index = SEEDS.index(selected_seed)
        test_scores = predict(models[model_index], x_test)
    else:
        selected_count = int(selected_name.rsplit("_", 1)[1])
        test_matrix = np.stack(
            [predict(models[i], x_test) for i in range(selected_count)],
            axis=0,
        )

        if selected_name.startswith("logit_mean_"):
            test_scores = test_matrix.mean(axis=0)
        else:
            test_rank_matrix = np.stack(
                [
                    within_user_percentile_ranks(test.user_id, test_matrix[i])
                    for i in range(selected_count)
                ],
                axis=0,
            )
            test_scores = test_rank_matrix.mean(axis=0)

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