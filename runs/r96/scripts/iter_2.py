import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

OUT_DIR = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

FIELDS = [
    "author_id", "duration_bucket", "fans_user_num_range",
    "follow_user_num_range", "friend_user_num_range", "hour",
    "is_live_streamer", "is_video_author", "music_type",
    "onehot_feat0", "onehot_feat1", "onehot_feat10", "onehot_feat11",
    "onehot_feat12", "onehot_feat13", "onehot_feat14", "onehot_feat15",
    "onehot_feat16", "onehot_feat17", "onehot_feat2", "onehot_feat3",
    "onehot_feat4", "onehot_feat5", "onehot_feat6", "onehot_feat7",
    "onehot_feat8", "onehot_feat9", "register_days_bucket",
    "register_days_range", "tab", "tag", "upload_type",
    "user_active_degree", "user_id", "video_id", "video_type"
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days"
]

train = load("train")
valid = load("valid")
y_train_np = np.asarray(train.y, dtype=np.float32)

# Four-day half-life emphasizes the portion of train most representative of
# the future date split while retaining all legal train observations.
max_train_date = int(np.max(train.date))
day_index = (
    np.asarray(train.date, dtype=np.int64) % 100
    + 31 * ((np.asarray(train.date, dtype=np.int64) // 100) % 100)
)
max_day_index = int(np.max(day_index))
age_days = max_day_index - day_index
sample_weight = np.exp2(-age_days.astype(np.float32) / 4.0)
sample_weight /= sample_weight.mean()


# --------------------------------------------------------------------------
# Family 1: recency-weighted DeepFM
# --------------------------------------------------------------------------
offsets = np.cumsum(
    np.asarray([0] + [FEATURE_CARDINALITIES[f] for f in FIELDS[:-1]],
               dtype=np.int64)
)
total_categories = int(sum(FEATURE_CARDINALITIES[f] for f in FIELDS))


def make_deep_matrix(split):
    cols = [
        np.asarray(split.X[f], dtype=np.int64) + offsets[j]
        for j, f in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int64)


class DeepFM(nn.Module):
    def __init__(self, num_categories, n_fields, emb_dim, initial_bias):
        super().__init__()
        self.linear = nn.Embedding(num_categories, 1)
        self.embedding = nn.Embedding(num_categories, emb_dim)
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))
        self.deep = nn.Sequential(
            nn.Linear(n_fields * emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.012)
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        emb = self.embedding(x)
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        summed = emb.sum(dim=1)
        fm = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(emb.flatten(start_dim=1)).squeeze(-1)
        return self.bias + linear + fm + deep


def torch_predict(model, matrix, batch_size=16384):
    model.eval()
    out = np.empty(len(matrix), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            xb = torch.from_numpy(matrix[start:end])
            out[start:end] = model(xb).cpu().numpy()
    return out


x_train_deep_np = make_deep_matrix(train)
x_valid_deep_np = make_deep_matrix(valid)

x_train_deep = torch.from_numpy(x_train_deep_np)
y_train_t = torch.from_numpy(y_train_np)
weight_t = torch.from_numpy(sample_weight.astype(np.float32))

p = float(y_train_np.mean())
initial_bias = float(np.log(p / (1.0 - p)))
deep_model = DeepFM(total_categories, len(FIELDS), 8, initial_bias)
deep_optimizer = torch.optim.AdamW(
    deep_model.parameters(), lr=9e-4, weight_decay=2e-6
)

n_train = len(train)
batch_size = 4096
generator = torch.Generator()
generator.manual_seed(SEED)

for epoch in range(4):
    deep_model.train()
    permutation = torch.randperm(n_train, generator=generator)
    for start in range(0, n_train, batch_size):
        idx = permutation[start:start + batch_size]
        xb = x_train_deep[idx]
        yb = y_train_t[idx]
        wb = weight_t[idx]

        deep_optimizer.zero_grad(set_to_none=True)
        logits = deep_model(xb)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, yb, reduction="none"
        )
        loss = (losses * wb).sum() / wb.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(deep_model.parameters(), 5.0)
        deep_optimizer.step()

deep_valid = torch_predict(deep_model, x_valid_deep_np)


# --------------------------------------------------------------------------
# Family 2: recency-weighted LightGBM with categorical, numeric, and strictly
# train-derived entity-history features.
# --------------------------------------------------------------------------
hist_train_video = historical_features("train", key="video_id")
hist_valid_video = historical_features("valid", key="video_id")
hist_train_author = historical_features("train", key="author_id")
hist_valid_author = historical_features("valid", key="author_id")

video_hist_names = sorted(hist_train_video.keys())
author_hist_names = sorted(hist_train_author.keys())

numeric_medians = {}
for name in NUM_FIELDS:
    raw = np.asarray(train.num[name], dtype=np.float32)
    transformed = np.log1p(np.maximum(raw, 0.0))
    numeric_medians[name] = float(np.nanmedian(transformed))


def make_lgb_matrix(split, video_hist, author_hist):
    columns = []

    # Categorical ids remain integer-valued even though the consolidated
    # NumPy matrix is float32; their column indices are declared categorical.
    for f in FIELDS:
        columns.append(np.asarray(split.X[f], dtype=np.float32))

    for name in NUM_FIELDS:
        raw = np.asarray(split.num[name], dtype=np.float32)
        value = np.log1p(np.maximum(raw, 0.0))
        value = np.where(np.isfinite(value), value, numeric_medians[name])
        columns.append(value.astype(np.float32))

    for name in video_hist_names:
        columns.append(np.asarray(video_hist[name], dtype=np.float32))
    for name in author_hist_names:
        columns.append(np.asarray(author_hist[name], dtype=np.float32))

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


x_train_lgb = make_lgb_matrix(
    train, hist_train_video, hist_train_author
)
x_valid_lgb = make_lgb_matrix(
    valid, hist_valid_video, hist_valid_author
)

categorical_indices = list(range(len(FIELDS)))
lgb_train = lgb.Dataset(
    x_train_lgb,
    label=y_train_np,
    weight=sample_weight,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 300,
    "min_data_per_group": 200,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "num_threads": min(16, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=420,
)
lgb_valid_prob = lgb_model.predict(x_valid_lgb)
lgb_valid = np.log(
    np.clip(lgb_valid_prob, 1e-6, 1.0 - 1e-6)
    / np.clip(1.0 - lgb_valid_prob, 1e-6, 1.0)
).astype(np.float32)


# --------------------------------------------------------------------------
# Family 3: non-parametric empirical-Bayes long-view histories.
# --------------------------------------------------------------------------
def find_rate_key(names):
    exact = [n for n in names if n.endswith("long_view_rate")]
    if exact:
        return exact[0]
    contains = [n for n in names if "long_view" in n and "rate" in n]
    if contains:
        return contains[0]
    raise RuntimeError("No long_view rate history found")


video_rate_key = find_rate_key(video_hist_names)
author_rate_key = find_rate_key(author_hist_names)


def empirical_score(video_hist, author_hist):
    vr = np.asarray(video_hist[video_rate_key], dtype=np.float64)
    ar = np.asarray(author_hist[author_rate_key], dtype=np.float64)
    rate = np.clip(0.65 * vr + 0.35 * ar, 1e-5, 1.0 - 1e-5)
    return np.log(rate / (1.0 - rate)).astype(np.float32)


emp_valid = empirical_score(hist_valid_video, hist_valid_author)


# --------------------------------------------------------------------------
# Compare standalone families and each legal incumbent blend. The explicit
# reusable-incumbent contract permits selecting a fixed blend weight on valid.
# --------------------------------------------------------------------------
inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy") if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy") if SHARED else ""
)
if not inc_valid_path or not os.path.exists(inc_valid_path):
    raise RuntimeError("Trusted incumbent validation scores are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

families_valid = {
    "deepfm_recency": np.asarray(deep_valid, dtype=np.float64),
    "lightgbm_recency_history": np.asarray(lgb_valid, dtype=np.float64),
    "empirical_bayes_history": np.asarray(emp_valid, dtype=np.float64),
}

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_family = None
best_weight = None
best_valid_scores = None
best_metrics = None

blend_weights = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]

for family_name, raw_scores in families_valid.items():
    raw_metrics = evaluate(valid.user_id, valid.y, raw_scores)
    candidate_scores[family_name] = float(raw_metrics["primary"])

    family_best_primary = -np.inf
    family_best_weight = None
    family_best_metrics = None
    family_best_scores = None

    # w is the new-family weight; 0 is the trusted incumbent.
    for w in blend_weights:
        blended = (1.0 - w) * inc_valid + w * raw_scores
        m = evaluate(valid.user_id, valid.y, blended)
        if float(m["primary"]) > family_best_primary:
            family_best_primary = float(m["primary"])
            family_best_weight = float(w)
            family_best_metrics = m
            family_best_scores = blended.copy()

    blend_name = family_name + "_incumbent_blend"
    candidate_scores[blend_name] = family_best_primary

    if family_best_primary > best_primary:
        best_primary = family_best_primary
        best_name = blend_name
        best_family = family_name
        best_weight = family_best_weight
        best_valid_scores = family_best_scores
        best_metrics = family_best_metrics

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS " + json.dumps({
        "selected": best_name,
        "new_family_weight": best_weight,
        "recency_half_life_days": 4.0
    }, sort_keys=True)
)


# --------------------------------------------------------------------------
# Test scoring with exactly the selected model and fixed validation-selected
# incumbent blend. No test labels are accessed.
# --------------------------------------------------------------------------
test = load("test")
hist_test_video = historical_features("test", key="video_id")
hist_test_author = historical_features("test", key="author_id")

if best_family == "deepfm_recency":
    x_test_deep_np = make_deep_matrix(test)
    raw_test_scores = torch_predict(deep_model, x_test_deep_np).astype(np.float64)
elif best_family == "lightgbm_recency_history":
    x_test_lgb = make_lgb_matrix(test, hist_test_video, hist_test_author)
    test_prob = lgb_model.predict(x_test_lgb)
    raw_test_scores = np.log(
        np.clip(test_prob, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - test_prob, 1e-6, 1.0)
    ).astype(np.float64)
elif best_family == "empirical_bayes_history":
    raw_test_scores = empirical_score(
        hist_test_video, hist_test_author
    ).astype(np.float64)
else:
    raise RuntimeError("Unknown selected family")

if not inc_test_path or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent test scores are unavailable")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
test_scores = (1.0 - best_weight) * inc_test + best_weight * raw_test_scores

if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(
        os.path.join(OUT_DIR, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(OUT_DIR, "scores_valid_raw.npy"),
        np.asarray(families_valid[best_family], dtype=np.float64),
    )
    np.save(
        os.path.join(OUT_DIR, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print(
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)