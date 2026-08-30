import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]
K = 16
FM_MAX_EPOCHS = 9
DEEP_MAX_EPOCHS = 6
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
LR = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")
if OUT:
    os.makedirs(OUT, exist_ok=True)


offsets = []
total_cardinality = 0
for field in FIELDS:
    offsets.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[field])
OFFSETS = np.asarray(offsets, dtype=np.int64)


def make_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + OFFSETS[j]
    return x


class ExpandedFM(nn.Module):
    def __init__(self, positive_rate):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, K)
        p = float(np.clip(positive_rate, 1e-5, 1.0 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, 0.0, 0.01)

    def forward(self, x):
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        pairwise = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + wide + pairwise


class DeepFM(nn.Module):
    def __init__(self, positive_rate):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, K)
        self.deep = nn.Sequential(
            nn.Linear(len(FIELDS) * K, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        p = float(np.clip(positive_rate, 1e-5, 1.0 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, 0.0, 0.01)
        for layer in self.deep:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(v.reshape(v.shape[0], -1)).squeeze(-1)
        return self.bias + wide + fm + deep


@torch.no_grad()
def predict_neural(model, x_np):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float32)
    for start in range(0, x_np.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x_np.shape[0])
        xb = torch.from_numpy(x_np[start:end])
        result[start:end] = model(xb).cpu().numpy()
    return result


def fit_select(model_class, train_x, train_y, valid_x, valid_y,
               valid_users, max_epochs, name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_class(float(np.mean(train_y)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    x_tensor = torch.from_numpy(train_x)
    y_tensor = torch.from_numpy(
        train_y.astype(np.float32, copy=False)
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    stale = 0
    n = train_x.shape[0]

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=generator)

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:min(start + BATCH_SIZE, n)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[idx])
            loss = criterion(logits, y_tensor[idx])
            loss.backward()
            optimizer.step()

        scores = predict_neural(model, valid_x)
        metrics = evaluate(valid_users, valid_y, scores)
        primary = float(metrics["primary"])
        print(
            "FINDINGS %s_epoch=%d primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (
                name,
                epoch,
                primary,
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
            ),
            flush=True,
        )

        if primary > best_primary:
            material = primary > best_primary + 0.00015
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()
            stale = 0 if material else stale + 1
        else:
            stale += 1

        if epoch >= 4 and stale >= 2:
            break

    del model, optimizer, x_tensor, y_tensor
    gc.collect()
    return best_scores, best_epoch


def fit_fixed(model_class, x_np, y_np, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_class(float(np.mean(y_np)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(x_np)
    y_tensor = torch.from_numpy(y_np.astype(np.float32, copy=False))
    generator = torch.Generator()
    generator.manual_seed(seed)

    n = x_np.shape[0]
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:min(start + BATCH_SIZE, n)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[idx])
            loss = criterion(logits, y_tensor[idx])
            loss.backward()
            optimizer.step()

    del optimizer, x_tensor, y_tensor
    return model


STAT_FIELDS = [
    ("video_id", 30.0, 1.00),
    ("author_id", 60.0, 0.70),
    ("tag", 250.0, 0.35),
    ("duration_bucket", 400.0, 0.25),
    ("tab", 500.0, 0.20),
]


def fit_eb_stats(split, y):
    global_rate = float(np.mean(y))
    stats = {}
    for field, smoothing, weight in STAT_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, minlength=card).astype(np.float64)
        positives = np.bincount(
            ids, weights=y.astype(np.float64, copy=False), minlength=card
        )
        rates = (positives + smoothing * global_rate) / (
            counts + smoothing
        )
        stats[field] = (
            np.log(np.clip(rates, 1e-5, 1.0 - 1e-5))
            / np.clip(1.0 - rates, 1e-5, None),
            weight,
        )
    global_logit = np.log(global_rate / (1.0 - global_rate))
    return global_logit, stats


def predict_eb(split, fitted):
    global_logit, stats = fitted
    scores = np.full(len(split.user_id), global_logit, dtype=np.float64)
    for field, (entity_logits, weight) in stats.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        scores += weight * (entity_logits[ids] - global_logit)
    return scores.astype(np.float32)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    scale = float(np.std(x))
    if scale < 1e-10:
        scale = 1.0
    return (x - float(np.mean(x))) / scale


def candidate_metric(users, labels, scores):
    return evaluate(users, labels, np.asarray(scores, dtype=np.float64))


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_x = make_matrix(train)
valid_x = make_matrix(valid)

fm_valid, fm_epoch = fit_select(
    ExpandedFM,
    train_x,
    train_y,
    valid_x,
    valid_y,
    valid_users,
    FM_MAX_EPOCHS,
    "expanded_fm",
    SEED,
)

deep_valid, deep_epoch = fit_select(
    DeepFM,
    train_x,
    train_y,
    valid_x,
    valid_y,
    valid_users,
    DEEP_MAX_EPOCHS,
    "deepfm",
    SEED + 17,
)

eb_fit_train = fit_eb_stats(train, train_y)
eb_valid = predict_eb(valid, eb_fit_train)

inc_valid_path = os.path.join(
    SHARED or "", "incumbent_valid_scores.npy"
)
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("trusted incumbent validation scores missing")
inc_valid = np.load(inc_valid_path).astype(np.float64, copy=False)

family_valid = {
    "expanded_fm": np.asarray(fm_valid, dtype=np.float64),
    "deepfm": np.asarray(deep_valid, dtype=np.float64),
    "empirical_bayes": np.asarray(eb_valid, dtype=np.float64),
}

candidate_scores = {}
candidate_summary = {}
candidate_info = {}

for family, own_scores in family_valid.items():
    metrics = candidate_metric(valid_users, valid_y, own_scores)
    key = family + "_standalone"
    candidate_scores[key] = own_scores
    candidate_summary[key] = float(metrics["primary"])
    candidate_info[key] = (family, 1.0)

    own_z = zscore(own_scores)
    inc_z = zscore(inc_valid)
    for alpha in (0.25, 0.40, 0.55, 0.70, 0.85):
        blended = alpha * own_z + (1.0 - alpha) * inc_z
        metrics = candidate_metric(valid_users, valid_y, blended)
        key = "%s_blend_%.2f" % (family, alpha)
        candidate_scores[key] = blended
        candidate_summary[key] = float(metrics["primary"])
        candidate_info[key] = (family, float(alpha))

winner = max(candidate_summary, key=candidate_summary.get)
winner_family, winner_alpha = candidate_info[winner]
valid_scores = candidate_scores[winner]
valid_metrics = candidate_metric(valid_users, valid_y, valid_scores)

print(
    "FINDINGS selected=%s fm_epoch=%d deepfm_epoch=%d"
    % (winner, fm_epoch, deep_epoch),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(
        candidate_summary, sort_keys=True, separators=(", ", ": ")
    ),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_alpha < 1.0:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(family_valid[winner_family], dtype=np.float64),
        )

# Refit the selected family on train + validation, using exactly its selected
# epoch count or empirical-Bayes recipe, then apply the fixed blend weight.
test = load("test")

if winner_family in ("expanded_fm", "deepfm"):
    combined_x = np.concatenate([train_x, valid_x], axis=0)
    combined_y = np.concatenate(
        [train_y, valid_y.astype(np.float32, copy=False)], axis=0
    )
    test_x = make_matrix(test)

    if winner_family == "expanded_fm":
        selected_model = fit_fixed(
            ExpandedFM, combined_x, combined_y, fm_epoch, SEED
        )
    else:
        selected_model = fit_fixed(
            DeepFM, combined_x, combined_y, deep_epoch, SEED + 17
        )

    own_test = predict_neural(selected_model, test_x).astype(
        np.float64, copy=False
    )
    del selected_model, combined_x, combined_y, test_x
else:
    class CombinedSplit:
        pass

    combined_split = CombinedSplit()
    combined_split.user_id = np.concatenate(
        [np.asarray(train.user_id), np.asarray(valid.user_id)]
    )
    combined_split.X = {}
    for field, _, _ in STAT_FIELDS:
        combined_split.X[field] = np.concatenate(
            [
                np.asarray(train.X[field], dtype=np.int64),
                np.asarray(valid.X[field], dtype=np.int64),
            ]
        )
    combined_y = np.concatenate(
        [train_y, valid_y.astype(np.float32, copy=False)]
    )
    eb_fit_combined = fit_eb_stats(combined_split, combined_y)
    own_test = predict_eb(test, eb_fit_combined).astype(
        np.float64, copy=False
    )

if winner_alpha < 1.0:
    inc_test_path = os.path.join(
        SHARED or "", "incumbent_test_scores.npy"
    )
    if not os.path.exists(inc_test_path):
        raise FileNotFoundError("trusted incumbent test scores missing")
    inc_test = np.load(inc_test_path).astype(np.float64, copy=False)
    test_scores = (
        winner_alpha * zscore(own_test)
        + (1.0 - winner_alpha) * zscore(inc_test)
    )
else:
    test_scores = own_test

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
result = {
    "primary": float(valid_metrics["primary"]),
    "gauc": float(valid_metrics["gauc"]),
    "ndcg@5": float(valid_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))