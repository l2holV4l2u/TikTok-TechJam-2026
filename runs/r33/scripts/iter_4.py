import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "hour",
    "user_active_degree",
    "is_live_streamer",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat8",
    "upload_type",
    "music_type",
    "tag",
    "duration_bucket",
    "register_days_bucket",
]
EMBED_DIM = 12
HIDDEN_DIM = 64
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
EPOCHS = 10
AUX_CLICK_WEIGHT = 0.25
AUX_LIKE_WEIGHT = 0.10

# The auxiliary logits are used only as model outputs at inference time.
BLENDS = [
    ("long_only", 0.00, 0.00),
    ("click_005", 0.05, 0.00),
    ("click_010", 0.10, 0.00),
    ("click_020", 0.20, 0.00),
    ("click_010_like_005", 0.10, 0.05),
    ("click_020_like_005", 0.20, 0.05),
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_features(split):
    x = np.column_stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS]
    )
    x += offsets_np[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


def make_targets(split):
    y_long = np.asarray(split.y, dtype=np.float32)
    y_click = np.asarray(split.aux["is_click"], dtype=np.float32)
    y_like = np.asarray(split.aux["is_like"], dtype=np.float32)
    y = np.column_stack([y_long, y_click, y_like]).astype(
        np.float32, copy=False
    )
    return torch.from_numpy(np.ascontiguousarray(y))


class MultiTaskNFM(nn.Module):
    def __init__(self, n_categories, dim, hidden_dim, n_tasks=3):
        super().__init__()
        self.linear = nn.Embedding(
            n_categories, n_tasks, sparse=True
        )
        self.factors = nn.Embedding(
            n_categories, dim, sparse=True
        )

        self.direct_interaction = nn.Linear(dim, n_tasks, bias=False)
        self.deep = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, n_tasks),
        )
        self.bias = nn.Parameter(torch.zeros(n_tasks))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.02)
        nn.init.normal_(
            self.direct_interaction.weight, mean=0.0, std=0.05
        )

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        wide = self.linear(x).sum(dim=1)

        embeddings = self.factors(x)
        summed = embeddings.sum(dim=1)
        bi_interaction = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        )

        return (
            self.bias
            + wide
            + self.direct_interaction(bi_interaction)
            + self.deep(bi_interaction)
        )


@torch.inference_mode()
def predict_logits(model, x, batch_size=16384):
    model.eval()
    result = np.empty((x.shape[0], 3), dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        logits = model(x[start:end])
        result[start:end] = logits.cpu().numpy().astype(
            np.float64, copy=False
        )
    return result


def blended_score(logits, click_coef, like_coef):
    return (
        logits[:, 0]
        + click_coef * logits[:, 1]
        + like_coef * logits[:, 2]
    )


train = load("train")
valid = load("valid")

x_train = make_features(train)
targets_train = make_targets(train)
x_valid = make_features(valid)

valid_labels = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

model = MultiTaskNFM(
    total_cardinality,
    EMBED_DIM,
    HIDDEN_DIM,
    n_tasks=3,
)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.factors.weight],
    lr=LEARNING_RATE,
)

dense_parameters = [
    p
    for name, p in model.named_parameters()
    if name not in {"linear.weight", "factors.weight"}
]
dense_optimizer = torch.optim.Adam(
    dense_parameters,
    lr=LEARNING_RATE,
    weight_decay=1e-6,
)

n_train = x_train.shape[0]
best_primary = -np.inf
best_metrics = None
best_state = None
best_blend = None
candidate_best = {name: -np.inf for name, _, _ in BLENDS}

for epoch in range(EPOCHS):
    model.train()

    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)
    order = torch.randperm(n_train, generator=generator)

    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x_train[idx]
        yb = targets_train[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)

        loss_long = F.binary_cross_entropy_with_logits(
            logits[:, 0], yb[:, 0]
        )
        loss_click = F.binary_cross_entropy_with_logits(
            logits[:, 1], yb[:, 1]
        )
        loss_like = F.binary_cross_entropy_with_logits(
            logits[:, 2], yb[:, 2]
        )

        loss = (
            loss_long
            + AUX_CLICK_WEIGHT * loss_click
            + AUX_LIKE_WEIGHT * loss_like
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(dense_parameters, max_norm=5.0)
        sparse_optimizer.step()
        dense_optimizer.step()

    # Skip only the very earliest, underfit checkpoint.
    if epoch < 1:
        continue

    valid_logits = predict_logits(model, x_valid)

    for name, click_coef, like_coef in BLENDS:
        scores = blended_score(valid_logits, click_coef, like_coef)
        metrics = evaluate(valid_users, valid_labels, scores)
        primary = float(metrics["primary"])
        candidate_best[name] = max(candidate_best[name], primary)

        if primary > best_primary:
            best_primary = primary
            best_metrics = {
                key: float(value) for key, value in metrics.items()
            }
            best_state = copy.deepcopy(model.state_dict())
            best_blend = (name, click_coef, like_coef, epoch + 1)

model.load_state_dict(best_state)

selected_name, selected_click, selected_like, selected_epoch = best_blend
final_valid_logits = predict_logits(model, x_valid)
final_valid_scores = blended_score(
    final_valid_logits, selected_click, selected_like
)
best_metrics = {
    key: float(value)
    for key, value in evaluate(
        valid_users, valid_labels, final_valid_scores
    ).items()
}

print(
    "FINDINGS "
    + json.dumps(
        {
            "train_long_rate": float(np.mean(train.y)),
            "train_click_rate": float(
                np.mean(np.asarray(train.aux["is_click"]))
            ),
            "train_like_rate": float(
                np.mean(np.asarray(train.aux["is_like"]))
            ),
            "selected_blend": selected_name,
            "selected_epoch": int(selected_epoch),
            "click_logit_coef": float(selected_click),
            "like_logit_coef": float(selected_like),
        },
        sort_keys=True,
    )
)

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: float(score)
            for name, score in candidate_best.items()
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = make_features(test)
    test_logits = predict_logits(model, x_test)
    test_scores = blended_score(
        test_logits, selected_click, selected_like
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "METRICS "
    + json.dumps(
        {
            "primary": best_metrics["primary"],
            "gauc": best_metrics["gauc"],
            "ndcg@5": best_metrics["ndcg@5"],
            "gpu_seconds": 0.0,
        }
    )
)