import os
import time
import math
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
RANK = 12
BATCH_SIZE = 8192
PRED_BATCH = 32768
EPOCHS = 3

torch.set_num_threads(min(16, os.cpu_count() or 1))
torch.manual_seed(SEED)
np.random.seed(SEED)

OFFSETS = {}
TOTAL_CARDINALITY = 0
for field in FIELDS:
    OFFSETS[field] = TOTAL_CARDINALITY
    TOTAL_CARDINALITY += int(FEATURE_CARDINALITIES[field])


def make_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = (
            np.asarray(split.X[field], dtype=np.int64) + OFFSETS[field]
        )
    return x


def initial_logit(y):
    p = float(np.mean(y))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def user_frequency_weights(user_ids):
    ids = np.asarray(user_ids, dtype=np.int64)
    counts = np.bincount(
        ids, minlength=int(FEATURE_CARDINALITIES["user_id"])
    ).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts[ids], 1.0))
    weights /= float(np.mean(weights))
    return weights.astype(np.float32)


class SparseCTRBase(nn.Module):
    def __init__(self, intercept):
        super().__init__()
        self.embedding = nn.Embedding(
            TOTAL_CARDINALITY, RANK + 1, sparse=True
        )
        self.register_buffer(
            "intercept",
            torch.tensor(float(intercept), dtype=torch.float32),
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.02)

    def embedded(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        return linear, e[:, :, 1:]


class DCNModel(SparseCTRBase):
    def __init__(self, intercept):
        super().__init__(intercept)
        dim = len(FIELDS) * RANK
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(dim)) for _ in range(3)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(3)
        ])
        for w in self.cross_w:
            nn.init.normal_(w, mean=0.0, std=0.02)

        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 64, 1)

    def forward(self, x):
        linear, v = self.embedded(x)
        x0 = v.reshape(v.shape[0], -1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(xl * w, dim=1, keepdim=True)
            xl = x0 * scalar + b + xl
        deep = self.deep(x0)
        nonlinear = self.output(torch.cat([xl, deep], dim=1)).squeeze(1)
        return self.intercept + linear + nonlinear


class AutoIntModel(SparseCTRBase):
    def __init__(self, intercept):
        super().__init__(intercept)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=RANK, num_heads=3, dropout=0.05, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=RANK, num_heads=3, dropout=0.05, batch_first=True
        )
        self.norm1 = nn.LayerNorm(RANK)
        self.norm2 = nn.LayerNorm(RANK)
        self.output = nn.Sequential(
            nn.Linear(len(FIELDS) * RANK, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        linear, v = self.embedded(x)
        a, _ = self.attn1(v, v, v, need_weights=False)
        h = self.norm1(v + a)
        a, _ = self.attn2(h, h, h, need_weights=False)
        h = self.norm2(h + a)
        nonlinear = self.output(h.reshape(h.shape[0], -1)).squeeze(1)
        return self.intercept + linear + nonlinear


class PNNModel(SparseCTRBase):
    def __init__(self, intercept):
        super().__init__(intercept)
        n_fields = len(FIELDS)
        pair_i = []
        pair_j = []
        for i in range(n_fields):
            for j in range(i + 1, n_fields):
                pair_i.append(i)
                pair_j.append(j)
        self.register_buffer(
            "pair_i", torch.tensor(pair_i, dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.tensor(pair_j, dtype=torch.long)
        )
        input_dim = n_fields * RANK + len(pair_i)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        linear, v = self.embedded(x)
        pair_products = (
            v[:, self.pair_i, :] * v[:, self.pair_j, :]
        ).sum(dim=2)
        features = torch.cat(
            [v.reshape(v.shape[0], -1), pair_products], dim=1
        )
        nonlinear = self.mlp(features).squeeze(1)
        return self.intercept + linear + nonlinear


def construct_model(family, intercept):
    if family == "dcn":
        return DCNModel(intercept)
    if family == "autoint":
        return AutoIntModel(intercept)
    if family == "pnn":
        return PNNModel(intercept)
    raise ValueError(family)


def fit_model(family, x_np, y_np, user_ids, seed):
    torch.manual_seed(seed)
    model = construct_model(family, initial_logit(y_np))

    sparse_optimizer = torch.optim.SparseAdam(
        model.embedding.parameters(), lr=0.0015
    )
    dense_parameters = [
        p for name, p in model.named_parameters()
        if not name.startswith("embedding.")
    ]
    dense_optimizer = torch.optim.AdamW(
        dense_parameters, lr=0.0012, weight_decay=1e-5
    )

    x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.int64))
    y = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.float32))
    weights = torch.from_numpy(user_frequency_weights(user_ids))

    generator = torch.Generator()
    generator.manual_seed(seed + 1009)
    n = x.shape[0]
    model.train()

    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(x[idx])
            row_loss = F.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            loss = torch.mean(row_loss * weights[idx])

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_parameters, 5.0)
            sparse_optimizer.step()
            dense_optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, x_np):
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.int64))
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], PRED_BATCH):
        end = min(start + PRED_BATCH, x.shape[0])
        result[start:end] = (
            model(x[start:end]).cpu().numpy().astype(np.float64)
        )
    return result


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
x_train = make_matrix(train)
x_valid = make_matrix(valid)

artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

families = ["dcn", "autoint", "pnn"]
blend_weights = [1.0, 0.75, 0.50, 0.25]

candidate_scores = {}
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metrics["primary"])

best_primary = float(inc_metrics["primary"])
best_metrics = inc_metrics
best_scores = inc_valid.copy()
best_family = "incumbent"
best_alpha = 0.0

for family_index, family in enumerate(families):
    model = fit_model(
        family,
        x_train,
        y_train,
        train.user_id,
        SEED + 101 * family_index,
    )
    family_valid = predict_model(model, x_valid)

    for alpha in blend_weights:
        if alpha == 1.0:
            name = family
            scores = family_valid
        else:
            name = "%s_blend_%.2f" % (family, alpha)
            scores = alpha * family_valid + (1.0 - alpha) * inc_valid

        metrics = evaluate(valid.user_id, y_valid, scores)
        primary = float(metrics["primary"])
        candidate_scores[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_scores = np.asarray(scores, dtype=np.float64).copy()
            best_family = family
            best_alpha = float(alpha)

    del model, family_valid
    gc.collect()

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

test = load("test")

if best_family == "incumbent":
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
else:
    y_joint = np.concatenate([
        y_train,
        np.asarray(valid.y, dtype=np.float32),
    ])
    x_joint = np.concatenate([x_train, x_valid], axis=0)
    joint_user_ids = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])

    selected_index = families.index(best_family)
    joint_model = fit_model(
        best_family,
        x_joint,
        y_joint,
        joint_user_ids,
        SEED + 101 * selected_index,
    )

    x_test = make_matrix(test)
    family_test = predict_model(joint_model, x_test)

    if best_alpha < 1.0:
        inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
        test_scores = (
            best_alpha * family_test
            + (1.0 - best_alpha) * inc_test
        )
    else:
        test_scores = family_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected_family=%s blend_new_weight=%.2f "
    "inverse_sqrt_user_weighting=true"
    % (best_family, best_alpha)
)

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)