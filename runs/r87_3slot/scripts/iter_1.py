import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
EPOCHS = 5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
seed_everything(SEED)

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(parts):
    columns = []
    for field, offset in zip(FIELDS, offsets):
        if len(parts) == 1:
            col = np.asarray(parts[0].X[field], dtype=np.int64)
        else:
            col = np.concatenate(
                [np.asarray(part.X[field], dtype=np.int64) for part in parts]
            )
        columns.append(col + offset)
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_categories, rank):
        super().__init__()
        # The final coordinate is the first-order weight; the others are
        # the shared FM latent factors.
        self.embedding = nn.Embedding(
            n_categories, rank + 1, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.embedding.weight[:, :rank].normal_(0.0, 0.01)
            self.embedding.weight[:, rank].zero_()
        self.rank = rank

    def forward(self, x):
        parameters = self.embedding(x)
        factors = parameters[:, :, :self.rank]
        linear = parameters[:, :, self.rank].sum(dim=1)

        summed = factors.sum(dim=1)
        interactions = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interactions


def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x_np), BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, len(x_np))
            result[start:end] = (
                model(x[start:end]).detach().cpu().numpy().astype(np.float64)
            )
    return result


def fit_model(x_np, y_np, epochs, valid_data=None, seed=SEED):
    seed_everything(seed)
    model = FactorizationMachine(total_cardinality, K)
    embedding_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(
        np.ascontiguousarray(y_np, dtype=np.float32)
    )

    best_state = None
    best_epoch = epochs
    best_primary = -np.inf
    best_scores = None
    best_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(x), generator=generator)

        for start in range(0, len(x), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            embedding_optimizer.zero_grad(set_to_none=True)
            bias_optimizer.zero_grad(set_to_none=True)

            logits = model(x[idx])
            loss = criterion(logits, y[idx])
            loss.backward()

            embedding_optimizer.step()
            bias_optimizer.step()

        if valid_data is not None:
            valid_x, valid_users, valid_y = valid_data
            valid_scores = predict(model, valid_x)
            metrics = evaluate(valid_users, valid_y, valid_scores)
            print(
                "epoch=%d primary=%.6f gauc=%.6f ndcg@5=%.6f"
                % (
                    epoch,
                    metrics["primary"],
                    metrics["gauc"],
                    metrics["ndcg@5"],
                ),
                flush=True,
            )
            if metrics["primary"] > best_primary:
                best_primary = float(metrics["primary"])
                best_epoch = epoch
                best_scores = valid_scores.copy()
                best_metrics = metrics
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

    if valid_data is not None:
        model.load_state_dict(best_state)
        return model, best_epoch, best_scores, best_metrics
    return model


train = load("train")
valid = load("valid")

x_train = make_matrix([train])
x_valid = make_matrix([valid])
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

model, selected_epochs, valid_scores, metrics = fit_model(
    x_train,
    y_train,
    EPOCHS,
    valid_data=(x_valid, valid.user_id, y_valid),
    seed=SEED,
)

print("selected_epoch=%d" % selected_epochs, flush=True)

# Refit the identical recipe on all labels available before the test period.
x_train_valid = np.concatenate([x_train, x_valid], axis=0)
y_train_valid = np.concatenate(
    [y_train, y_valid.astype(np.float32)], axis=0
)

refit_model = fit_model(
    x_train_valid,
    y_train_valid,
    selected_epochs,
    valid_data=None,
    seed=SEED,
)

test = load("test")
x_test = make_matrix([test])
test_scores = predict(refit_model, x_test)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.3f}'
    % (
        metrics["primary"],
        metrics["gauc"],
        metrics["ndcg@5"],
        elapsed,
    )
)