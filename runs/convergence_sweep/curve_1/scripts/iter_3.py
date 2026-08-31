import os
import time
import json
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_THREADS = min(16, os.cpu_count() or 8)
torch.set_num_threads(N_THREADS)
try:
    torch.set_num_interop_threads(min(4, N_THREADS))
except RuntimeError:
    pass


# All three new model families receive the same categorical inputs. These cover
# identity, content, exposure context, and relatively stable user state without
# expanding to every potentially non-stationary one-hot field.
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "music_type",
    "upload_type",
    "user_active_degree",
    "fans_user_num_range",
    "onehot_feat3",
    "onehot_feat8",
]

CARDS = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDS))
N_FIELDS = len(FIELDS)


def categorical_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64)
        for name in FIELDS
    ])


def recency_weights(dates, half_life=4.0):
    _, inverse = np.unique(
        np.asarray(dates, dtype=np.int32),
        return_inverse=True,
    )
    age = inverse.max() - inverse
    weights = np.exp2(
        -age.astype(np.float32) / np.float32(half_life)
    ).astype(np.float32)
    weights /= weights.mean()
    return weights


def score_scale(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return scale


class DCNv2(nn.Module):
    """Low-rank mixture-free DCNv2-style explicit cross network plus MLP."""

    def __init__(
        self,
        total_cardinality,
        offsets,
        n_fields,
        embedding_dim=8,
        cross_rank=24,
        n_cross_layers=3,
    ):
        super().__init__()
        self.register_buffer(
            "offsets",
            torch.as_tensor(offsets, dtype=torch.long),
        )
        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
        self.linear = nn.Embedding(total_cardinality, 1)
        input_dim = n_fields * embedding_dim

        self.cross_u = nn.ModuleList([
            nn.Linear(input_dim, cross_rank, bias=False)
            for _ in range(n_cross_layers)
        ])
        self.cross_v = nn.ModuleList([
            nn.Linear(cross_rank, input_dim, bias=True)
            for _ in range(n_cross_layers)
        ])

        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
        )
        self.output = nn.Linear(input_dim + 64, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.linear.weight, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        indices = x + self.offsets
        emb = self.embedding(indices)
        x0 = emb.flatten(start_dim=1)
        crossed = x0
        for u_layer, v_layer in zip(self.cross_u, self.cross_v):
            crossed = crossed + x0 * v_layer(torch.relu(u_layer(crossed)))
        deep = self.deep(x0)
        nonlinear = self.output(torch.cat([crossed, deep], dim=1)).squeeze(1)
        wide = self.linear(indices).sum(dim=1).squeeze(1)
        return self.bias + wide + nonlinear


class ProductNN(nn.Module):
    """PNN using all pairwise field inner products as explicit MLP inputs."""

    def __init__(
        self,
        total_cardinality,
        offsets,
        n_fields,
        embedding_dim=8,
    ):
        super().__init__()
        self.register_buffer(
            "offsets",
            torch.as_tensor(offsets, dtype=torch.long),
        )
        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i",
            torch.as_tensor(pair_i, dtype=torch.long),
        )
        self.register_buffer(
            "pair_j",
            torch.as_tensor(pair_j, dtype=torch.long),
        )

        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
        self.linear = nn.Embedding(total_cardinality, 1)
        product_dim = len(pair_i)
        input_dim = n_fields * embedding_dim + product_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.linear.weight, std=0.01)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        indices = x + self.offsets
        emb = self.embedding(indices)
        products = (
            emb[:, self.pair_i, :] * emb[:, self.pair_j, :]
        ).sum(dim=2)
        network_input = torch.cat(
            [emb.flatten(start_dim=1), products],
            dim=1,
        )
        nonlinear = self.network(network_input).squeeze(1)
        wide = self.linear(indices).sum(dim=1).squeeze(1)
        return self.bias + wide + nonlinear


def fit_torch_model(
    model,
    x_train_tensor,
    labels_tensor,
    weights_tensor,
    epochs=3,
    batch_size=8192,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.1e-3,
        weight_decay=2e-6,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    n_rows = labels_tensor.shape[0]

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_rows, generator=generator)
        loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, n_rows, batch_size):
            idx = permutation[start:start + batch_size]
            xb = x_train_tensor[idx]
            yb = labels_tensor[idx]
            wb = weights_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            row_losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                yb,
                reduction="none",
            )
            loss = (row_losses * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float((row_losses.detach() * wb).sum())
            weight_sum += float(wb.sum())

        print(
            "FINDINGS model={} epoch={} weighted_loss={:.6f}".format(
                model.__class__.__name__,
                epoch + 1,
                loss_sum / max(weight_sum, 1.0),
            ),
            flush=True,
        )

    return model


@torch.inference_mode()
def torch_predict(model, matrix, batch_size=32768):
    model.eval()
    result = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        xb = torch.from_numpy(matrix[start:end])
        result[start:end] = model(xb).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_weights = recency_weights(train.date, half_life=4.0)

x_train = categorical_matrix(train)
x_valid = categorical_matrix(valid)

print(
    "FINDINGS recency_weight_range={:.4f}/{:.4f} effective_rows={:.0f}".format(
        float(train_weights.min()),
        float(train_weights.max()),
        float(train_weights.sum() ** 2 / np.square(train_weights).sum()),
    ),
    flush=True,
)


# ---------------------------------------------------------------------------
# Family 1: LambdaRank.
#
# Rows are grouped by user and the objective directly optimizes ordering within
# each logged impression set, emphasizing the top five positions.
# ---------------------------------------------------------------------------
train_user_ids = np.asarray(train.user_id, dtype=np.int64)
rank_order = np.argsort(train_user_ids, kind="stable")
rank_users = train_user_ids[rank_order]
_, rank_group = np.unique(rank_users, return_counts=True)

rank_dataset = lgb.Dataset(
    x_train[rank_order].astype(np.int32, copy=False),
    label=y_train[rank_order],
    weight=train_weights[rank_order],
    group=rank_group,
    categorical_feature=list(range(N_FIELDS)),
    free_raw_data=True,
)

rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.055,
    "num_leaves": 63,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "cat_smooth": 25.0,
    "cat_l2": 12.0,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "num_threads": N_THREADS,
    "verbose": -1,
}

rank_model = lgb.train(
    rank_params,
    rank_dataset,
    num_boost_round=220,
)
rank_valid = rank_model.predict(
    x_valid,
    raw_score=True,
).astype(np.float64)

del rank_dataset, rank_order, rank_users, rank_group
gc.collect()


# ---------------------------------------------------------------------------
# Families 2 and 3: DCNv2 and PNN.
# ---------------------------------------------------------------------------
x_train_tensor = torch.from_numpy(x_train)
labels_tensor = torch.from_numpy(y_train)
weights_tensor = torch.from_numpy(train_weights)

dcn_model = DCNv2(
    TOTAL_CARDINALITY,
    OFFSETS,
    N_FIELDS,
    embedding_dim=8,
    cross_rank=24,
    n_cross_layers=3,
)
fit_torch_model(
    dcn_model,
    x_train_tensor,
    labels_tensor,
    weights_tensor,
    epochs=3,
)
dcn_valid = torch_predict(dcn_model, x_valid)

pnn_model = ProductNN(
    TOTAL_CARDINALITY,
    OFFSETS,
    N_FIELDS,
    embedding_dim=8,
)
fit_torch_model(
    pnn_model,
    x_train_tensor,
    labels_tensor,
    weights_tensor,
    epochs=3,
)
pnn_valid = torch_predict(pnn_model, x_valid)


# Test features are used only after every model and hyperparameter is fixed.
# The test labels are never accessed.
test = load("test")
x_test = categorical_matrix(test)

rank_test = rank_model.predict(
    x_test,
    raw_score=True,
).astype(np.float64)
dcn_test = torch_predict(dcn_model, x_test)
pnn_test = torch_predict(pnn_model, x_test)


# Trusted incumbent blending. Candidate and incumbent scores are divided by
# validation-set standard deviations so that blend weights compare ranking
# signals on similar scales. The exact validation-derived scales and selected
# weight are then applied unchanged to test.
shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise FileNotFoundError("Trusted incumbent predictions are required")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)

if len(inc_valid) != len(y_valid) or len(inc_test) != len(x_test):
    raise ValueError("Trusted incumbent prediction length mismatch")

inc_scale = score_scale(inc_valid)

families = {
    "lambdarank": (rank_valid, rank_test),
    "dcnv2": (dcn_valid, dcn_test),
    "pnn": (pnn_valid, pnn_test),
}

candidate_results = {}
best_primary = -np.inf
best_metrics = None
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_name = None
best_is_blend = False

for family_name, (valid_raw, test_raw) in families.items():
    raw_metrics = evaluate(valid.user_id, y_valid, valid_raw)
    candidate_results[family_name] = float(raw_metrics["primary"])

    if float(raw_metrics["primary"]) > best_primary:
        best_primary = float(raw_metrics["primary"])
        best_metrics = raw_metrics
        best_valid_scores = valid_raw.copy()
        best_test_scores = test_raw.copy()
        best_raw_valid = None
        best_name = family_name
        best_is_blend = False

    family_scale = score_scale(valid_raw)
    normalized_valid = valid_raw / family_scale
    normalized_test = test_raw / family_scale
    normalized_inc_valid = inc_valid / inc_scale
    normalized_inc_test = inc_test / inc_scale

    for own_weight in (0.10, 0.20, 0.35, 0.50, 0.65, 0.80):
        blend_valid = (
            own_weight * normalized_valid
            + (1.0 - own_weight) * normalized_inc_valid
        )
        blend_test = (
            own_weight * normalized_test
            + (1.0 - own_weight) * normalized_inc_test
        )
        blend_metrics = evaluate(
            valid.user_id,
            y_valid,
            blend_valid,
        )
        blend_name = "{}_blend_{:.2f}".format(
            family_name,
            own_weight,
        )
        candidate_results[blend_name] = float(
            blend_metrics["primary"]
        )

        if float(blend_metrics["primary"]) > best_primary:
            best_primary = float(blend_metrics["primary"])
            best_metrics = blend_metrics
            best_valid_scores = blend_valid.copy()
            best_test_scores = blend_test.copy()
            best_raw_valid = valid_raw.copy()
            best_name = blend_name
            best_is_blend = True

print(
    "FINDINGS winner={} blended={} train_positive_rate={:.6f}".format(
        best_name,
        best_is_blend,
        float(y_train.mean()),
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(
        candidate_results,
        sort_keys=True,
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    if best_is_blend:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)