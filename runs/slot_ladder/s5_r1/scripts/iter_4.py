import os
import time
import json
import random
import numpy as np
import torch
from torch import nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

offsets = []
total_cardinality = 0
for field in CAT_FIELDS:
    offsets.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)


def categorical_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64) + offsets[j]
            for j, name in enumerate(CAT_FIELDS)
        ]),
        dtype=np.int64,
    )


def raw_numeric_matrix(split, hist_keys=None):
    columns = []
    names = []

    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.log1p(np.maximum(np.nan_to_num(x, nan=0.0), 0.0))
        columns.append(x)
        names.append("num_" + name)

    history_dicts = {}
    for entity in ("video_id", "author_id"):
        h = historical_features(split, key=entity)
        for key, value in h.items():
            history_dicts[entity + "::" + key] = np.asarray(value, dtype=np.float32)

    if hist_keys is None:
        hist_keys = sorted(history_dicts.keys())

    for key in hist_keys:
        x = np.nan_to_num(
            history_dicts[key],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
        columns.append(x)
        names.append(key)

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32), hist_keys, names


def normalize_numeric(x, mean=None, std=None):
    if mean is None:
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = x.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, 1e-4)
    z = (x - mean) / std
    z = np.clip(z, -8.0, 8.0)
    return np.ascontiguousarray(z, dtype=np.float32), mean, std


class DCNCrossModel(nn.Module):
    def __init__(self, cardinality, n_fields, n_numeric, emb_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, emb_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        input_dim = n_fields * emb_dim + n_numeric
        self.cross1 = nn.Linear(input_dim, 1)
        self.cross2 = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.cross_out = nn.Linear(input_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, cat, num):
        emb = self.embedding(cat).flatten(1)
        x0 = torch.cat([emb, num], dim=1)
        x1 = x0 * self.cross1(x0) + x0
        x2 = x0 * self.cross2(x1) + x1
        return (
            self.cross_out(x2).squeeze(1)
            + self.deep(x0).squeeze(1)
            + self.bias
        )


class LoggedMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_videos, rank=24):
        super().__init__()
        self.user_factor = nn.Embedding(n_users, rank)
        self.video_factor = nn.Embedding(n_videos, rank)
        self.user_bias = nn.Embedding(n_users, 1)
        self.video_bias = nn.Embedding(n_videos, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_factor.weight, std=0.03)
        nn.init.normal_(self.video_factor.weight, std=0.03)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.video_bias.weight)

    def forward(self, users, videos):
        interaction = (
            self.user_factor(users) * self.video_factor(videos)
        ).sum(dim=1)
        return (
            interaction
            + self.user_bias(users).squeeze(1)
            + self.video_bias(videos).squeeze(1)
            + self.global_bias
        )


def train_dcn(cat_np, num_np, y_np):
    model = DCNCrossModel(
        total_cardinality,
        cat_np.shape[1],
        num_np.shape[1],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    cat = torch.from_numpy(cat_np)
    num = torch.from_numpy(num_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    n = len(y)
    generator = torch.Generator().manual_seed(SEED + 1)

    model.train()
    for _ in range(3):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(cat[idx], num[idx])
            loss = criterion(logits, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def train_mf(users_np, videos_np, y_np):
    model = LoggedMatrixFactorization(
        int(FEATURE_CARDINALITIES["user_id"]),
        int(FEATURE_CARDINALITIES["video_id"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=2e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    users = torch.from_numpy(np.asarray(users_np, dtype=np.int64))
    videos = torch.from_numpy(np.asarray(videos_np, dtype=np.int64))
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    n = len(y)
    generator = torch.Generator().manual_seed(SEED + 2)

    model.train()
    for _ in range(3):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(users[idx], videos[idx])
            loss = criterion(logits, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def predict_dcn(model, cat_np, num_np):
    model.eval()
    cat = torch.from_numpy(cat_np)
    num = torch.from_numpy(num_np)
    out = np.empty(len(cat_np), dtype=np.float64)
    for start in range(0, len(cat_np), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(cat_np))
        out[start:end] = model(cat[start:end], num[start:end]).numpy()
    return out


@torch.inference_mode()
def predict_mf(model, users_np, videos_np):
    model.eval()
    users = torch.from_numpy(np.asarray(users_np, dtype=np.int64))
    videos = torch.from_numpy(np.asarray(videos_np, dtype=np.int64))
    out = np.empty(len(users_np), dtype=np.float64)
    for start in range(0, len(users_np), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(users_np))
        out[start:end] = model(users[start:end], videos[start:end]).numpy()
    return out


def fit_eb_tables(train):
    y = np.asarray(train.y, dtype=np.float64)
    global_rate = float(y.mean())
    tables = {}

    specifications = {
        "video_id": 25.0,
        "author_id": 35.0,
        "tag": 80.0,
        "tab": 150.0,
        "duration_bucket": 120.0,
        "upload_type": 120.0,
    }

    for field, alpha in specifications.items():
        ids = np.asarray(train.X[field], dtype=np.int64)
        size = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, minlength=size).astype(np.float64)
        positives = np.bincount(ids, weights=y, minlength=size)
        rates = (positives + alpha * global_rate) / (counts + alpha)
        rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
        tables[field] = np.log(rates / (1.0 - rates))

    return tables


def predict_eb(split, tables):
    weights = {
        "video_id": 0.32,
        "author_id": 0.33,
        "tag": 0.16,
        "tab": 0.08,
        "duration_bucket": 0.07,
        "upload_type": 0.04,
    }
    result = np.zeros(len(split.user_id), dtype=np.float64)
    for field, weight in weights.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += weight * tables[field][ids]
    return result


def scale_for_blend(valid_score, test_score):
    valid_score = np.asarray(valid_score, dtype=np.float64)
    test_score = np.asarray(test_score, dtype=np.float64)
    mean = float(valid_score.mean())
    std = max(float(valid_score.std()), 1e-8)
    return (valid_score - mean) / std, (test_score - mean) / std


train = load("train")
valid = load("valid")

train_cat = categorical_matrix(train)
valid_cat = categorical_matrix(valid)

train_num_raw, hist_keys, numeric_names = raw_numeric_matrix("train")
valid_num_raw, _, _ = raw_numeric_matrix("valid", hist_keys=hist_keys)
train_num, num_mean, num_std = normalize_numeric(train_num_raw)
valid_num, _, _ = normalize_numeric(valid_num_raw, num_mean, num_std)

dcn_model = train_dcn(train_cat, train_num, train.y)
mf_model = train_mf(
    train.X["user_id"],
    train.X["video_id"],
    train.y,
)
eb_tables = fit_eb_tables(train)

valid_dcn = predict_dcn(dcn_model, valid_cat, valid_num)
valid_mf = predict_mf(
    mf_model,
    valid.X["user_id"],
    valid.X["video_id"],
)
valid_eb = predict_eb(valid, eb_tables)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_z = (inc_valid - inc_valid.mean()) / max(inc_valid.std(), 1e-8)

family_valid = {
    "dcn_cross": valid_dcn,
    "matrix_factorization": valid_mf,
    "empirical_bayes": valid_eb,
}

candidate_scores = {}
candidate_arrays = {}
candidate_raw = {}

for name, score in family_valid.items():
    standalone_metrics = evaluate(valid.user_id, valid.y, score)
    candidate_scores[name] = float(standalone_metrics["primary"])
    candidate_arrays[name] = score
    candidate_raw[name] = score

    own_z = (score - score.mean()) / max(score.std(), 1e-8)
    for weight in (0.15, 0.30, 0.50):
        blend = (1.0 - weight) * inc_valid_z + weight * own_z
        blend_name = name + "_inc_w" + str(weight)
        blend_metrics = evaluate(valid.user_id, valid.y, blend)
        candidate_scores[blend_name] = float(blend_metrics["primary"])
        candidate_arrays[blend_name] = blend
        candidate_raw[blend_name] = score

all_new_valid_z = np.mean(
    [
        (score - score.mean()) / max(score.std(), 1e-8)
        for score in family_valid.values()
    ],
    axis=0,
)
for weight in (0.15, 0.30, 0.50):
    blend = (1.0 - weight) * inc_valid_z + weight * all_new_valid_z
    name = "all_new_inc_w" + str(weight)
    blend_metrics = evaluate(valid.user_id, valid.y, blend)
    candidate_scores[name] = float(blend_metrics["primary"])
    candidate_arrays[name] = blend
    candidate_raw[name] = all_new_valid_z

winner_name = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(candidate_arrays[winner_name], dtype=np.float64)
valid_raw = np.asarray(candidate_raw[winner_name], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

test = load("test")
test_cat = categorical_matrix(test)
test_num_raw, _, _ = raw_numeric_matrix("test", hist_keys=hist_keys)
test_num, _, _ = normalize_numeric(test_num_raw, num_mean, num_std)

test_dcn = predict_dcn(dcn_model, test_cat, test_num)
test_mf = predict_mf(
    mf_model,
    test.X["user_id"],
    test.X["video_id"],
)
test_eb = predict_eb(test, eb_tables)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
inc_scale_mean = float(inc_valid.mean())
inc_scale_std = max(float(inc_valid.std()), 1e-8)
inc_test_z = (inc_test - inc_scale_mean) / inc_scale_std

family_test = {
    "dcn_cross": test_dcn,
    "matrix_factorization": test_mf,
    "empirical_bayes": test_eb,
}

valid_scalers = {
    name: (
        float(family_valid[name].mean()),
        max(float(family_valid[name].std()), 1e-8),
    )
    for name in family_valid
}
family_test_z = {
    name: (family_test[name] - valid_scalers[name][0]) / valid_scalers[name][1]
    for name in family_test
}

if winner_name in family_test:
    test_scores = family_test[winner_name]
elif winner_name.startswith("all_new_inc_w"):
    weight = float(winner_name.rsplit("w", 1)[1])
    all_new_test_z = np.mean(
        [family_test_z[name] for name in family_test_z],
        axis=0,
    )
    test_scores = (1.0 - weight) * inc_test_z + weight * all_new_test_z
else:
    family_name, weight_text = winner_name.rsplit("_inc_w", 1)
    weight = float(weight_text)
    test_scores = (
        (1.0 - weight) * inc_test_z
        + weight * family_test_z[family_name]
    )

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(valid_raw, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": winner_name,
    "numeric_features": len(numeric_names),
    "dcn_standalone": candidate_scores["dcn_cross"],
    "mf_standalone": candidate_scores["matrix_factorization"],
    "eb_standalone": candidate_scores["empirical_bayes"],
}, sort_keys=True))

elapsed = time.time() - START
print(
    "METRICS " + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)