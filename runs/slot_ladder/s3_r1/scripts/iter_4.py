import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
]
EMBED_DIM = 16
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 16384
EPOCHS = 4
LR = 0.002

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
NUM_FEATURES = int(sum(cardinalities))
NUM_FIELDS = len(FIELDS)
FLAT_DIM = NUM_FIELDS * EMBED_DIM


def make_matrix(split):
    cols = [
        np.asarray(split.X[field], dtype=np.int64) + offset
        for field, offset in zip(FIELDS, offsets)
    ]
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


def init_embedding(embedding):
    nn.init.normal_(embedding.weight, mean=0.0, std=0.015)


class DCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_FEATURES, EMBED_DIM)
        init_embedding(self.embedding)

        self.cross_w = nn.ParameterList(
            [nn.Parameter(torch.empty(FLAT_DIM)) for _ in range(3)]
        )
        self.cross_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(FLAT_DIM)) for _ in range(3)]
        )
        for w in self.cross_w:
            nn.init.normal_(w, mean=0.0, std=0.02)

        self.deep = nn.Sequential(
            nn.Linear(FLAT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(FLAT_DIM + 64, 1)

    def forward(self, x):
        x0 = self.embedding(x).reshape(x.shape[0], -1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(xl * w, dim=1, keepdim=True)
            xl = xl + x0 * scalar + b
        deep = self.deep(x0)
        return self.output(torch.cat([xl, deep], dim=1)).squeeze(1)


class AutoIntModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_FEATURES, EMBED_DIM)
        init_embedding(self.embedding)
        self.position = nn.Parameter(torch.zeros(1, NUM_FIELDS, EMBED_DIM))

        self.attn1 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=4, dropout=0.0, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=4, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.output = nn.Sequential(
            nn.Linear(FLAT_DIM, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        z = self.embedding(x) + self.position
        a, _ = self.attn1(z, z, z, need_weights=False)
        z = self.norm1(z + a)
        a, _ = self.attn2(z, z, z, need_weights=False)
        z = self.norm2(z + a)
        return self.output(z.reshape(z.shape[0], -1)).squeeze(1)


class FiBiNETModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_FEATURES, EMBED_DIM)
        init_embedding(self.embedding)

        reduction = max(2, NUM_FIELDS // 2)
        self.senet = nn.Sequential(
            nn.Linear(NUM_FIELDS, reduction),
            nn.ReLU(),
            nn.Linear(reduction, NUM_FIELDS),
            nn.Sigmoid(),
        )
        self.bilinear = nn.Parameter(
            torch.empty(NUM_FIELDS, EMBED_DIM, EMBED_DIM)
        )
        nn.init.xavier_uniform_(self.bilinear)

        self.pairs = [
            (i, j)
            for i in range(NUM_FIELDS)
            for j in range(i + 1, NUM_FIELDS)
        ]
        pair_dim = len(self.pairs) * EMBED_DIM
        self.output = nn.Sequential(
            nn.Linear(FLAT_DIM + pair_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        e = self.embedding(x)
        summary = e.mean(dim=2)
        gates = self.senet(summary).unsqueeze(2)
        se = e * gates

        transformed = torch.einsum("bfd,fde->bfe", se, self.bilinear)
        interactions = [
            transformed[:, i, :] * se[:, j, :]
            for i, j in self.pairs
        ]
        pair_features = torch.cat(interactions, dim=1)
        features = torch.cat(
            [se.reshape(se.shape[0], -1), pair_features], dim=1
        )
        return self.output(features).squeeze(1)


class MMoEModel(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.embedding = nn.Embedding(NUM_FEATURES, EMBED_DIM)
        init_embedding(self.embedding)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(FLAT_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(FLAT_DIM, len(self.experts))
            for _ in range(num_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        flat = self.embedding(x).reshape(x.shape[0], -1)
        experts = torch.stack([expert(flat) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(flat), dim=1).unsqueeze(2)
            mixed = torch.sum(experts * weights, dim=1)
            outputs.append(tower(mixed).squeeze(1))
        return outputs


def train_binary(model, x_tensor, y_tensor, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = x_tensor.shape[0]

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = x_tensor.index_select(0, idx)
            yb = y_tensor.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()


def train_multitask(model, x_tensor, targets, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = x_tensor.shape[0]

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = x_tensor.index_select(0, idx)
            yb = targets.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)
            main_loss = F.binary_cross_entropy_with_logits(outputs[0], yb[:, 0])
            if len(outputs) > 1:
                aux_losses = [
                    F.binary_cross_entropy_with_logits(outputs[k], yb[:, k])
                    for k in range(1, len(outputs))
                ]
                loss = main_loss + 0.20 * torch.stack(aux_losses).mean()
            else:
                loss = main_loss
            loss.backward()
            optimizer.step()


def predict_logits(model, matrix, multitask=False):
    result = np.empty(matrix.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, matrix.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, matrix.shape[0])
            xb = torch.from_numpy(matrix[start:end])
            output = model(xb)
            if multitask:
                output = output[0]
            result[start:end] = output.cpu().numpy()
    return result


def sigmoid_np(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


train = load("train")
x_train_np = make_matrix(train)
x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

aux_names = [name for name in ("is_click", "is_like", "is_follow")
             if name in train.aux]
aux_arrays = []
for name in aux_names[:2]:
    values = np.asarray(train.aux[name], dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    aux_arrays.append(np.clip(values, 0.0, 1.0))
if aux_arrays:
    multitask_targets_np = np.column_stack(
        [np.asarray(train.y, dtype=np.float32)] + aux_arrays
    ).astype(np.float32)
else:
    multitask_targets_np = np.asarray(train.y, dtype=np.float32)[:, None]
multitask_targets = torch.from_numpy(multitask_targets_np)

models = {}
models["dcn"] = (DCNModel(), False)
models["autoint"] = (AutoIntModel(), False)
models["fibinet"] = (FiBiNETModel(), False)
models["mmoe"] = (MMoEModel(multitask_targets.shape[1]), True)

for model_index, (name, (model, is_multitask)) in enumerate(models.items()):
    if is_multitask:
        train_multitask(
            model, x_train, multitask_targets, SEED + 100 * model_index
        )
    else:
        train_binary(model, x_train, y_train, SEED + 100 * model_index)

del x_train, x_train_np, y_train, multitask_targets, multitask_targets_np, train

valid = load("valid")
x_valid = make_matrix(valid)
valid_predictions = {}
candidate_scores = {}

for name, (model, is_multitask) in models.items():
    logits = predict_logits(model, x_valid, multitask=is_multitask)
    probabilities = sigmoid_np(logits)
    valid_predictions[name] = probabilities
    score = evaluate(valid.user_id, valid.y, probabilities)
    candidate_scores[name] = float(score["primary"])

shared_dir = os.environ.get("SHARED_ARTIFACTS")
inc_valid = None
inc_test_path = None
if shared_dir:
    p_valid = os.path.join(shared_dir, "incumbent_valid_scores.npy")
    p_test = os.path.join(shared_dir, "incumbent_test_scores.npy")
    if os.path.exists(p_valid) and os.path.exists(p_test):
        inc_valid = sigmoid_np(np.load(p_valid))
        inc_test_path = p_test

blend_alphas = [0.10, 0.20, 0.35, 0.50]
all_candidates = {}
for name, pred in valid_predictions.items():
    all_candidates[name] = {
        "scores": pred,
        "model": name,
        "alpha": 1.0,
        "blended": False,
    }
    if inc_valid is not None:
        for alpha in blend_alphas:
            blended = alpha * pred + (1.0 - alpha) * inc_valid
            blend_name = f"{name}_blend_{alpha:.2f}"
            metric = evaluate(valid.user_id, valid.y, blended)
            candidate_scores[blend_name] = float(metric["primary"])
            all_candidates[blend_name] = {
                "scores": blended,
                "model": name,
                "alpha": alpha,
                "blended": True,
            }

winner_name = max(candidate_scores, key=candidate_scores.get)
winner = all_candidates[winner_name]
valid_scores = winner["scores"]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("FINDINGS multitask_aux=" + json.dumps(aux_names[:2]))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner["blended"]:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(valid_predictions[winner["model"]], dtype=np.float64),
        )

del x_valid, valid_predictions, valid

test = load("test")
x_test = make_matrix(test)
winner_model, winner_is_multitask = models[winner["model"]]
test_raw = sigmoid_np(
    predict_logits(winner_model, x_test, multitask=winner_is_multitask)
)

if winner["blended"]:
    incumbent_test = sigmoid_np(np.load(inc_test_path))
    alpha = winner["alpha"]
    test_scores = alpha * test_raw + (1.0 - alpha) * incumbent_test
else:
    test_scores = test_raw

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))