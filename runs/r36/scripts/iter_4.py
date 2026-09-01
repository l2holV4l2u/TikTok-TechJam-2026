import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2025

BASE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
]

DEEP_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
    "hour",
]

RANK = 16
DEEP_EMBED_DIM = 12
HIDDEN_DIMS = (128, 64)
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 10
SPARSE_LR = 0.001
DENSE_LR = 0.001
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

base_cards = [int(FEATURE_CARDINALITIES[f]) for f in BASE_FIELDS]
base_offsets = np.cumsum([0] + base_cards[:-1], dtype=np.int64)
base_total = int(sum(base_cards))

deep_cards = [int(FEATURE_CARDINALITIES[f]) for f in DEEP_FIELDS]
deep_offsets = np.cumsum([0] + deep_cards[:-1], dtype=np.int64)
deep_total = int(sum(deep_cards))


def make_base_matrix(split):
    x = np.stack([split.X[f] for f in BASE_FIELDS], axis=1)
    x = x.astype(np.int64, copy=False)
    x = x + base_offsets[None, :]
    return np.ascontiguousarray(x)


def make_deep_matrix(split):
    x = np.stack([split.X[f] for f in DEEP_FIELDS], axis=1)
    x = x.astype(np.int64, copy=False)
    x = x + deep_offsets[None, :]
    return np.ascontiguousarray(x)


class DeepFMResidual(nn.Module):
    def __init__(self):
        super().__init__()

        # Exact five-field FM backbone.
        self.linear = nn.Embedding(base_total, 1, sparse=True)
        self.factors = nn.Embedding(base_total, RANK, sparse=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

        # Separate embeddings let the nonlinear branch specialize without
        # forcing the FM latent geometry to serve two objectives.
        self.deep_embedding = nn.Embedding(
            deep_total, DEEP_EMBED_DIM, sparse=True
        )
        nn.init.normal_(
            self.deep_embedding.weight, mean=0.0, std=0.02
        )

        input_dim = len(DEEP_FIELDS) * DEEP_EMBED_DIM
        layers = []
        previous = input_dim
        for hidden in HIDDEN_DIMS:
            layers.append(nn.Linear(previous, hidden))
            layers.append(nn.ReLU())
            previous = hidden
        self.deep_tower = nn.Sequential(*layers)
        self.deep_output = nn.Linear(previous, 1)

        for module in self.deep_tower:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, a=np.sqrt(5)
                )
                nn.init.zeros_(module.bias)

        # Start as a small residual around the reliable FM.
        nn.init.normal_(self.deep_output.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.deep_output.bias)

        with torch.no_grad():
            for offset in base_offsets:
                self.linear.weight[int(offset)].zero_()
                self.factors.weight[int(offset)].zero_()
            for offset in deep_offsets:
                self.deep_embedding.weight[int(offset)].zero_()

    def forward_components(self, x_base, x_deep):
        linear_term = self.linear(x_base).sum(dim=1).squeeze(-1)

        v = self.factors(x_base)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        fm_score = linear_term + interaction

        deep_vector = self.deep_embedding(x_deep).flatten(start_dim=1)
        deep_hidden = self.deep_tower(deep_vector)
        deep_score = self.deep_output(deep_hidden).squeeze(-1)
        return fm_score, deep_score

    def forward(self, x_base, x_deep):
        fm_score, deep_score = self.forward_components(x_base, x_deep)
        return fm_score + deep_score


@torch.inference_mode()
def predict_components(model, x_base, x_deep):
    model.eval()
    n = len(x_base)
    fm_result = np.empty(n, dtype=np.float64)
    deep_result = np.empty(n, dtype=np.float64)

    for start in range(0, n, PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, n)
        xb = torch.from_numpy(x_base[start:end])
        xd = torch.from_numpy(x_deep[start:end])
        fm, deep = model.forward_components(xb, xd)
        fm_result[start:end] = fm.cpu().numpy().astype(
            np.float64, copy=False
        )
        deep_result[start:end] = deep.cpu().numpy().astype(
            np.float64, copy=False
        )

    return fm_result, deep_result


train = load("train")
x_train_base = make_base_matrix(train)
x_train_deep = make_deep_matrix(train)
y_train = np.asarray(train.y, dtype=np.float32).copy()
del train

valid = load("valid")
x_valid_base = make_base_matrix(valid)
x_valid_deep = make_deep_matrix(valid)
y_valid = np.asarray(valid.y, dtype=np.int8).copy()
valid_users = np.asarray(valid.user_id, dtype=np.int64).copy()
del valid

x_train_base_t = torch.from_numpy(x_train_base)
x_train_deep_t = torch.from_numpy(x_train_deep)
y_train_t = torch.from_numpy(y_train)

model = DeepFMResidual()

sparse_parameters = [
    model.linear.weight,
    model.factors.weight,
    model.deep_embedding.weight,
]
dense_parameters = list(model.deep_tower.parameters()) + list(
    model.deep_output.parameters()
)

sparse_optimizer = torch.optim.SparseAdam(
    sparse_parameters, lr=SPARSE_LR
)
dense_optimizer = torch.optim.AdamW(
    dense_parameters, lr=DENSE_LR, weight_decay=1e-5
)

rng = np.random.default_rng(SEED)
n_train = len(y_train)

best_primary = -np.inf
best_metrics = None
best_state = None
best_epoch = -1
best_alpha = None
candidate_best = {
    f"alpha_{alpha:g}": -np.inf for alpha in ALPHAS
}

for epoch in range(1, EPOCHS + 1):
    model.train()
    permutation = rng.permutation(n_train)

    running_loss = 0.0
    seen = 0

    for start in range(0, n_train, BATCH_SIZE):
        idx_np = permutation[start:start + BATCH_SIZE]
        idx = torch.from_numpy(idx_np)

        xb = x_train_base_t.index_select(0, idx)
        xd = x_train_deep_t.index_select(0, idx)
        yb = y_train_t.index_select(0, idx)

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb, xd)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(dense_parameters, max_norm=5.0)
        sparse_optimizer.step()
        dense_optimizer.step()

        batch_n = len(idx_np)
        running_loss += float(loss.detach()) * batch_n
        seen += batch_n

    valid_fm, valid_deep = predict_components(
        model, x_valid_base, x_valid_deep
    )

    epoch_best_primary = -np.inf
    epoch_best_alpha = None
    epoch_best_metrics = None

    for alpha in ALPHAS:
        scores = valid_fm + alpha * valid_deep
        metrics = evaluate(valid_users, y_valid, scores)
        primary = float(metrics["primary"])
        name = f"alpha_{alpha:g}"
        candidate_best[name] = max(candidate_best[name], primary)

        if primary > epoch_best_primary:
            epoch_best_primary = primary
            epoch_best_alpha = float(alpha)
            epoch_best_metrics = metrics

    print(
        f"epoch={epoch} loss={running_loss / max(seen, 1):.6f} "
        f"alpha={epoch_best_alpha:g} "
        f"primary={epoch_best_primary:.6f} "
        f"gauc={epoch_best_metrics['gauc']:.6f} "
        f"ndcg@5={epoch_best_metrics['ndcg@5']:.6f}",
        flush=True,
    )

    if epoch_best_primary > best_primary:
        best_primary = epoch_best_primary
        best_metrics = epoch_best_metrics
        best_epoch = epoch
        best_alpha = epoch_best_alpha
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

model.load_state_dict(best_state)
valid_fm, valid_deep = predict_components(
    model, x_valid_base, x_valid_deep
)
valid_scores = valid_fm + best_alpha * valid_deep
final_metrics = evaluate(valid_users, y_valid, valid_scores)

deep_std = float(np.std(valid_deep))
fm_std = float(np.std(valid_fm))
correlation = float(np.corrcoef(valid_fm, valid_deep)[0, 1])

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_epoch": int(best_epoch),
            "selected_alpha": float(best_alpha),
            "fm_score_std": fm_std,
            "deep_residual_std": deep_std,
            "fm_deep_correlation": correlation,
        },
        sort_keys=True,
    ),
    flush=True,
)
print(
    "CANDIDATES "
    + json.dumps(
        {key: float(value) for key, value in candidate_best.items()},
        sort_keys=True,
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test_base = make_base_matrix(test)
    x_test_deep = make_deep_matrix(test)
    del test

    test_fm, test_deep = predict_components(
        model, x_test_base, x_test_deep
    )
    test_scores = test_fm + best_alpha * test_deep
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

payload = {
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(payload))