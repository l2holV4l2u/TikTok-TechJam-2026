import copy
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 2048
PRED_BATCH_SIZE = 32768
EPOCHS = 10


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_features(split):
    x = np.stack([np.asarray(split.X[f], dtype=np.int64) for f in FIELDS], axis=1)
    x += offsets[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class FactorizationMachine(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # Last coordinate is the first-order coefficient; the others are
        # the standard FM interaction factors.
        self.embedding = nn.Embedding(num_embeddings, embedding_dim + 1)
        self.bias = nn.Parameter(torch.zeros(()))

        with torch.no_grad():
            self.embedding.weight[:, :embedding_dim].normal_(mean=0.0, std=0.01)
            self.embedding.weight[:, embedding_dim].zero_()

    def forward(self, x):
        parameters = self.embedding(x)
        factors = parameters[:, :, :EMBED_DIM]
        linear = parameters[:, :, EMBED_DIM].sum(dim=1)

        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear + interaction


def predict(model, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, x.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, x.shape[0])
            result[start:end] = model(x[start:end]).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

x_train = make_features(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = make_features(valid)
valid_y = np.asarray(valid.y)
valid_users = np.asarray(valid.user_id)

model = FactorizationMachine(total_cardinality, EMBED_DIM)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    foreach=True,
)

generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -math.inf
best_metrics = None
best_state = None
best_valid_scores = None

n_train = x_train.shape[0]

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(n_train, generator=generator)

    loss_sum = 0.0
    seen = 0

    for start in range(0, n_train, BATCH_SIZE):
        indices = permutation[start:start + BATCH_SIZE]
        xb = x_train[indices]
        yb = y_train[indices]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()

        batch_n = indices.numel()
        loss_sum += float(loss.detach()) * batch_n
        seen += batch_n

    valid_scores = predict(model, x_valid)
    metrics = evaluate(valid_users, valid_y, valid_scores)

    print(
        f"epoch={epoch + 1} "
        f"loss={loss_sum / seen:.6f} "
        f"primary={float(metrics['primary']):.6f} "
        f"gauc={float(metrics['gauc']):.6f} "
        f"ndcg@5={float(metrics['ndcg@5']):.6f}",
        flush=True,
    )

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_metrics = {k: float(v) for k, v in metrics.items()}
        best_valid_scores = valid_scores.copy()
        best_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

model.load_state_dict(best_state)

# Re-evaluate the retained validation predictions so the reported model and
# selected checkpoint are exactly aligned.
best_metrics = evaluate(valid_users, valid_y, best_valid_scores)
best_metrics = {k: float(v) for k, v in best_metrics.items()}

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = make_features(test)
    test_scores = predict(model, x_test)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final_metrics = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final_metrics))