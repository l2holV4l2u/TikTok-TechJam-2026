import os
import time
import json
import copy
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()

# Reproducible CPU training.
SEED = 20260829
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

device = torch.device("cpu")

train = load("train")
valid = load("valid")
test = load("test")

y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

if not (os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)):
    raise RuntimeError("Trusted incumbent predictions are required")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if inc_valid.shape != (len(valid.user_id),):
    raise RuntimeError("Validation incumbent shape mismatch")
if inc_test.shape != (len(test.user_id),):
    raise RuntimeError("Test incumbent shape mismatch")


categorical_fields = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "video_type",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

numeric_fields = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def make_categorical(split):
    result = []
    for name in categorical_fields:
        x = np.asarray(split.X[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        # Defensive clipping keeps unseen or malformed IDs at the reserved ID.
        x = np.where((x >= 0) & (x < card), x, 0).astype(np.int64)
        result.append(torch.from_numpy(x))
    return result


# Fit all numeric transformations on train only. Counts and durations are
# heavy-tailed, so log1p is substantially easier for the cross tower.
numeric_center = []
numeric_scale = []

train_numeric_columns = []
for name in numeric_fields:
    x = np.asarray(train.num[name], dtype=np.float64)
    finite = np.isfinite(x)
    median = float(np.median(x[finite])) if np.any(finite) else 0.0
    x = np.where(finite, x, median)
    x = np.log1p(np.maximum(x, 0.0))
    center = float(np.mean(x))
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    numeric_center.append((median, center))
    numeric_scale.append(scale)
    train_numeric_columns.append(
        np.clip((x - center) / scale, -6.0, 6.0).astype(np.float32)
    )

train_num = torch.from_numpy(
    np.column_stack(train_numeric_columns).astype(np.float32)
)


def make_numeric(split):
    columns = []
    for j, name in enumerate(numeric_fields):
        median, center = numeric_center[j]
        scale = numeric_scale[j]
        x = np.asarray(split.num[name], dtype=np.float64)
        x = np.where(np.isfinite(x), x, median)
        x = np.log1p(np.maximum(x, 0.0))
        x = np.clip((x - center) / scale, -6.0, 6.0)
        columns.append(x.astype(np.float32))
    return torch.from_numpy(np.column_stack(columns).astype(np.float32))


train_cat = make_categorical(train)
valid_cat = make_categorical(valid)
test_cat = make_categorical(test)

valid_num = make_numeric(valid)
test_num = make_numeric(test)

y_train = torch.from_numpy(y_train_np)


class LowRankCrossLayer(nn.Module):
    """
    DCN-V2 low-rank matrix cross:
        x_{l+1} = x_l + x_0 * (U activation(V x_l) + b)
    """

    def __init__(self, dim, rank):
        super().__init__()
        self.v = nn.Linear(dim, rank, bias=False)
        self.u = nn.Linear(rank, dim, bias=True)
        nn.init.xavier_uniform_(self.v.weight, gain=0.35)
        nn.init.xavier_uniform_(self.u.weight, gain=0.35)
        nn.init.zeros_(self.u.bias)

    def forward(self, x0, xl):
        crossed = self.u(F.relu(self.v(xl)))
        return xl + x0 * crossed


class DCNV2(nn.Module):
    def __init__(self, cardinalities, num_numeric):
        super().__init__()
        self.embedding_dim = 8
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(card, self.embedding_dim)
                for card in cardinalities
            ]
        )
        self.linear_embeddings = nn.ModuleList(
            [nn.Embedding(card, 1) for card in cardinalities]
        )

        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)
            with torch.no_grad():
                emb.weight[0].zero_()

        for emb in self.linear_embeddings:
            nn.init.zeros_(emb.weight)

        dim = len(cardinalities) * self.embedding_dim + num_numeric
        self.input_norm = nn.LayerNorm(dim)

        self.cross_layers = nn.ModuleList(
            [
                LowRankCrossLayer(dim, rank=32),
                LowRankCrossLayer(dim, rank=32),
                LowRankCrossLayer(dim, rank=24),
            ]
        )

        self.deep = nn.Sequential(
            nn.Linear(dim, 192),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.ReLU(),
        )

        self.cross_head = nn.Linear(dim, 1)
        self.deep_head = nn.Linear(48, 1)
        self.numeric_linear = nn.Linear(num_numeric, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.deep[0].weight)
        nn.init.zeros_(self.deep[0].bias)
        nn.init.xavier_uniform_(self.deep[3].weight)
        nn.init.zeros_(self.deep[3].bias)
        nn.init.xavier_uniform_(self.deep[6].weight)
        nn.init.zeros_(self.deep[6].bias)
        nn.init.xavier_uniform_(self.cross_head.weight, gain=0.15)
        nn.init.zeros_(self.cross_head.bias)
        nn.init.xavier_uniform_(self.deep_head.weight, gain=0.15)
        nn.init.zeros_(self.deep_head.bias)
        nn.init.zeros_(self.numeric_linear.weight)

    def forward(self, cats, nums):
        dense_embeddings = [
            emb(x) for emb, x in zip(self.embeddings, cats)
        ]
        x0 = torch.cat(dense_embeddings + [nums], dim=1)
        x0 = self.input_norm(x0)

        cross = x0
        for layer in self.cross_layers:
            cross = layer(x0, cross)

        deep = self.deep(x0)

        first_order = torch.zeros(
            len(nums), 1, dtype=nums.dtype, device=nums.device
        )
        for emb, x in zip(self.linear_embeddings, cats):
            first_order = first_order + emb(x)

        logits = (
            first_order
            + self.numeric_linear(nums)
            + self.cross_head(cross)
            + self.deep_head(deep)
            + self.bias
        )
        return logits.squeeze(1)


cardinalities = [
    int(FEATURE_CARDINALITIES[name]) for name in categorical_fields
]

model = DCNV2(cardinalities, len(numeric_fields)).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1.8e-3,
    weight_decay=1.0e-6,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=4,
    eta_min=4.0e-4,
)


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if not np.isfinite(std) or std < 1e-10:
        std = 1.0
    return (x - mean) / std


@torch.no_grad()
def predict(model, cat_arrays, num_array, batch_size=16384):
    model.eval()
    n = len(num_array)
    result = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        cats = [
            x[start:stop].to(device, non_blocking=False)
            for x in cat_arrays
        ]
        nums = num_array[start:stop].to(device, non_blocking=False)
        logits = model(cats, nums)
        result[start:stop] = logits.cpu().numpy().astype(np.float64)

    return result


inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
inc_valid_z = standardize(inc_valid)
inc_test_z = standardize(inc_test)

best_metrics = inc_metrics
best_valid_scores = inc_valid.copy()
best_test_scores = None
best_state = None
best_epoch = 0
best_alpha = 0.0

candidate_report = {
    "incumbent": float(inc_metrics["primary"])
}
epoch_findings = []

n_train = len(y_train)
batch_size = 4096

# A moderate set of blend weights tests complementarity without relying on
# the raw calibration of either neural network.
blend_alphas = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.65,
    0.80,
    1.00,
]

for epoch in range(1, 5):
    model.train()
    permutation = torch.randperm(
        n_train,
        generator=torch.Generator().manual_seed(SEED + epoch),
    )

    running_loss = 0.0
    seen = 0

    for start in range(0, n_train, batch_size):
        idx = permutation[start:min(start + batch_size, n_train)]

        cats = [x[idx].to(device) for x in train_cat]
        nums = train_num[idx].to(device)
        target = y_train[idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(cats, nums)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        count = len(idx)
        running_loss += float(loss.detach().cpu()) * count
        seen += count

    scheduler.step()

    valid_logits = predict(model, valid_cat, valid_num)
    valid_z = standardize(valid_logits)

    raw_metrics = evaluate(valid.user_id, y_valid, valid_logits)
    candidate_report[
        "epoch_%d_raw" % epoch
    ] = float(raw_metrics["primary"])

    epoch_best_primary = float(inc_metrics["primary"])
    epoch_best_alpha = 0.0
    epoch_best_metrics = inc_metrics
    epoch_best_scores = inc_valid.copy()

    for alpha in blend_alphas:
        # Convex interpolation of standardized scores. alpha=1 is raw DCN
        # ranking and alpha=0 is the trusted incumbent.
        scores = (
            (1.0 - float(alpha)) * inc_valid_z
            + float(alpha) * valid_z
        )
        metrics = evaluate(valid.user_id, y_valid, scores)
        primary = float(metrics["primary"])

        candidate_report[
            "epoch_%d_blend_%.2f" % (epoch, alpha)
        ] = primary

        if primary > epoch_best_primary:
            epoch_best_primary = primary
            epoch_best_alpha = float(alpha)
            epoch_best_metrics = metrics
            epoch_best_scores = scores.copy()

    epoch_findings.append(
        {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "raw_primary": float(raw_metrics["primary"]),
            "best_blend_primary": epoch_best_primary,
            "best_alpha": epoch_best_alpha,
        }
    )

    if epoch_best_primary > float(best_metrics["primary"]):
        best_metrics = epoch_best_metrics
        best_valid_scores = epoch_best_scores
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        best_alpha = epoch_best_alpha

    del valid_logits, valid_z
    gc.collect()


# If no DCN blend improves validation, preserve the trusted incumbent exactly.
if best_state is None or best_alpha <= 0.0:
    best_valid_scores = inc_valid.copy()
    best_test_scores = inc_test.copy()
    selected_name = "incumbent"
else:
    model.load_state_dict(best_state)
    test_logits = predict(model, test_cat, test_num)
    test_z = standardize(test_logits)
    best_test_scores = (
        (1.0 - best_alpha) * inc_test_z
        + best_alpha * test_z
    )
    selected_name = "dcnv2_epoch_%d_alpha_%.2f" % (
        best_epoch,
        best_alpha,
    )

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": selected_name,
            "selected_epoch": best_epoch,
            "selected_alpha": best_alpha,
            "incumbent_primary": float(inc_metrics["primary"]),
            "selected_primary": float(best_metrics["primary"]),
            "selected_gain": float(
                best_metrics["primary"] - inc_metrics["primary"]
            ),
            "epochs": epoch_findings,
            "categorical_fields": len(categorical_fields),
            "numeric_fields": len(numeric_fields),
        },
        sort_keys=True,
    )
)

print("CANDIDATES " + json.dumps(candidate_report, sort_keys=True))

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

elapsed = time.time() - START

print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)