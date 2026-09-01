import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.manual_seed(2024)
np.random.seed(2024)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "onehot_feat3",
    "hour",
]
CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]

offsets = np.zeros(len(FIELDS), dtype=np.int64)
offsets[1:] = np.cumsum(CARDINALITIES[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDINALITIES))


def make_features(split):
    x = np.column_stack([split.X[name] for name in FIELDS]).astype(
        np.int64, copy=False
    )
    x = x + offsets[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class MultiTaskDeepFM(nn.Module):
    def __init__(self, num_categories, num_fields, num_factors=16):
        super().__init__()
        self.num_fields = num_fields
        self.num_factors = num_factors

        self.linear = nn.Embedding(num_categories, 1, sparse=True)
        self.embedding = nn.Embedding(
            num_categories, num_factors, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(()))

        self.deep_trunk = nn.Sequential(
            nn.Linear(num_fields * num_factors, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.long_view_head = nn.Linear(48, 1)
        self.play_time_head = nn.Linear(48, 1)

        nn.init.zeros_(self.linear.weight)
        nn.init.xavier_uniform_(self.embedding.weight)

        for module in self.deep_trunk:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Keep the scored branch initially close to the reliable FM.
        nn.init.normal_(
            self.long_view_head.weight, mean=0.0, std=0.01
        )
        nn.init.zeros_(self.long_view_head.bias)

        nn.init.xavier_uniform_(self.play_time_head.weight)
        nn.init.zeros_(self.play_time_head.bias)

    def forward(self, x, return_aux=False):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        v = self.embedding(x)
        summed = v.sum(dim=1)
        fm_interaction = 0.5 * (
            summed.square().sum(dim=1)
            - v.square().sum(dim=(1, 2))
        )

        hidden = self.deep_trunk(v.flatten(start_dim=1))
        deep_term = self.long_view_head(hidden).squeeze(-1)

        logits = (
            self.bias + linear_term + fm_interaction + deep_term
        )

        if return_aux:
            play_prediction = self.play_time_head(hidden).squeeze(-1)
            return logits, play_prediction
        return logits


@torch.inference_mode()
def predict(model, x, batch_size=65536):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        result[start:end] = (
            model(x[start:end]).detach().double().cpu().numpy()
        )
    return result


train = load("train")
valid = load("valid")

x_train = make_features(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = make_features(valid)
y_valid_np = np.asarray(valid.y)

# Use the outcome only as a training target. It is never supplied as an
# input and no validation/test outcome is read.
raw_play = np.asarray(
    train.aux["play_time_ms"], dtype=np.float64
)
raw_play = np.nan_to_num(
    raw_play, nan=0.0, posinf=0.0, neginf=0.0
)
raw_play = np.maximum(raw_play, 0.0)

play_clip = float(np.quantile(raw_play, 0.995))
log_play = np.log1p(np.minimum(raw_play, play_clip))
play_mean = float(log_play.mean())
play_std = float(log_play.std())
if play_std < 1e-6:
    play_std = 1.0

play_target_np = (
    (log_play - play_mean) / play_std
).astype(np.float32)
play_target = torch.from_numpy(play_target_np)

model = MultiTaskDeepFM(
    num_categories=TOTAL_CARDINALITY,
    num_fields=len(FIELDS),
    num_factors=16,
)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.embedding.weight],
    lr=0.001,
)
dense_parameters = (
    [model.bias]
    + list(model.deep_trunk.parameters())
    + list(model.long_view_head.parameters())
    + list(model.play_time_head.parameters())
)
dense_optimizer = torch.optim.Adam(
    dense_parameters,
    lr=0.001,
)

batch_size = 8192
num_epochs = 10
auxiliary_weight = 0.20

generator = torch.Generator()
generator.manual_seed(2024)

best_primary = -np.inf
best_metrics = None
best_state = None

n = x_train.shape[0]

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(n, generator=generator)

    for start in range(0, n, batch_size):
        idx = permutation[start:start + batch_size]
        xb = x_train[idx]
        yb = y_train[idx]
        pb = play_target[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits, play_prediction = model(xb, return_aux=True)

        long_view_loss = F.binary_cross_entropy_with_logits(
            logits, yb
        )
        play_time_loss = F.smooth_l1_loss(
            play_prediction,
            pb,
            beta=1.0,
        )
        loss = (
            long_view_loss
            + auxiliary_weight * play_time_loss
        )
        loss.backward()

        sparse_optimizer.step()
        dense_optimizer.step()

    valid_scores = predict(model, x_valid)
    metrics = evaluate(
        valid.user_id,
        y_valid_np,
        valid_scores,
    )
    primary = float(metrics["primary"])

    if primary > best_primary:
        best_primary = primary
        best_metrics = {
            key: float(value)
            for key, value in metrics.items()
        }
        best_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

model.load_state_dict(best_state)

valid_scores = predict(model, x_valid)
best_metrics = {
    key: float(value)
    for key, value in evaluate(
        valid.user_id,
        y_valid_np,
        valid_scores,
    ).items()
}

test = load("test")
x_test = make_features(test)
test_scores = predict(model, x_test)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        },
        separators=(",", ":"),
    )
)