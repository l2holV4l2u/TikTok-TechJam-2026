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
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
EMBED_DIM = 8
HIDDEN_DIMS = (64, 32)
DROPOUT = 0.10
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 16384
EPOCHS = 9
BLEND_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))
num_fields = len(FIELDS)


def make_features(split):
    x = np.column_stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS]
    )
    x += offsets_np[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class DeepFM(nn.Module):
    def __init__(self, n_categories, n_fields, embed_dim, hidden_dims, dropout):
        super().__init__()
        self.linear = nn.Embedding(n_categories, 1, sparse=True)
        self.factors = nn.Embedding(n_categories, embed_dim, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))

        layers = []
        input_dim = n_fields * embed_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.deep_body = nn.Sequential(*layers)
        self.deep_output = nn.Linear(input_dim, 1)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

        for module in self.deep_body:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        # Begin as a conventional FM, then let the deep residual enter gradually.
        nn.init.zeros_(self.deep_output.weight)
        nn.init.zeros_(self.deep_output.bias)

    def components(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        embeddings = self.factors(x)

        summed = embeddings.sum(dim=1)
        fm_interaction = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        ).sum(dim=1)
        shallow_logit = self.bias + linear_term + fm_interaction

        deep_input = embeddings.reshape(embeddings.shape[0], -1)
        deep_logit = self.deep_output(
            self.deep_body(deep_input)
        ).squeeze(-1)

        return shallow_logit, deep_logit

    def forward(self, x):
        shallow_logit, deep_logit = self.components(x)
        return shallow_logit + deep_logit


@torch.inference_mode()
def predict_components(model, x, batch_size=PRED_BATCH_SIZE):
    model.eval()
    shallow = np.empty(x.shape[0], dtype=np.float64)
    deep = np.empty(x.shape[0], dtype=np.float64)

    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        shallow_batch, deep_batch = model.components(x[start:end])
        shallow[start:end] = shallow_batch.cpu().numpy().astype(
            np.float64, copy=False
        )
        deep[start:end] = deep_batch.cpu().numpy().astype(
            np.float64, copy=False
        )

    return shallow, deep


train = load("train")
valid = load("valid")

x_train = make_features(train)
y_train = torch.from_numpy(
    np.ascontiguousarray(np.asarray(train.y, dtype=np.float32))
)
x_valid = make_features(valid)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

model = DeepFM(
    total_cardinality,
    num_fields,
    EMBED_DIM,
    HIDDEN_DIMS,
    DROPOUT,
)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.factors.weight],
    lr=LEARNING_RATE,
)
dense_parameters = (
    [model.bias]
    + list(model.deep_body.parameters())
    + list(model.deep_output.parameters())
)
dense_optimizer = torch.optim.Adam(
    dense_parameters,
    lr=LEARNING_RATE,
)

n_train = x_train.shape[0]
best_primary = -np.inf
best_metrics = None
best_state = None
best_alpha = None
candidate_best = {
    "deepfm_alpha_%.2f" % alpha: -np.inf for alpha in BLEND_ALPHAS
}

for epoch in range(EPOCHS):
    model.train()
    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)
    order = torch.randperm(n_train, generator=generator)

    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x_train[idx]
        yb = y_train[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        dense_optimizer.step()

    shallow_valid, deep_valid = predict_components(model, x_valid)

    for alpha in BLEND_ALPHAS:
        scores = shallow_valid + alpha * deep_valid
        metrics = evaluate(valid_users, y_valid, scores)
        primary = float(metrics["primary"])
        name = "deepfm_alpha_%.2f" % alpha
        candidate_best[name] = max(candidate_best[name], primary)

        if primary > best_primary:
            best_primary = primary
            best_metrics = {k: float(v) for k, v in metrics.items()}
            best_alpha = float(alpha)
            best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
shallow_valid, deep_valid = predict_components(model, x_valid)
final_valid_scores = shallow_valid + best_alpha * deep_valid
best_metrics = {
    k: float(v)
    for k, v in evaluate(valid_users, y_valid, final_valid_scores).items()
}

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_epoch_model_alpha": best_alpha,
            "fields": len(FIELDS),
            "embedding_dim": EMBED_DIM,
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_best.items()},
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = make_features(test)
    shallow_test, deep_test = predict_components(model, x_test)
    test_scores = shallow_test + best_alpha * deep_test
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