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
SEED = 2025

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_THREADS = min(16, os.cpu_count() or 8)
torch.set_num_threads(N_THREADS)
try:
    torch.set_num_interop_threads(min(4, N_THREADS))
except RuntimeError:
    pass


# These fields retain item/content and user-state information while avoiding a
# very wide expansion over all 37 fields.
CAT_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "music_type",
    "upload_type",
    "video_type",
    "is_video_author",
    "is_live_streamer",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

DEEP_FIELDS = [
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

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def training_recency_weights(dates, half_life=5.0):
    unique_dates = np.sort(np.unique(dates))
    date_to_index = {int(d): i for i, d in enumerate(unique_dates)}
    day_index = np.fromiter(
        (date_to_index[int(d)] for d in dates),
        dtype=np.int16,
        count=len(dates),
    )
    age = (len(unique_dates) - 1 - day_index).astype(np.float32)
    weights = np.exp2(-age / np.float32(half_life)).astype(np.float32)
    weights /= np.mean(weights)
    return weights


def fit_entity_stats(ids, labels, weights, cardinality, prior, alpha):
    weighted_count = np.bincount(
        ids,
        weights=weights,
        minlength=cardinality,
    ).astype(np.float64)
    weighted_pos = np.bincount(
        ids,
        weights=weights * labels,
        minlength=cardinality,
    ).astype(np.float64)
    rates = (weighted_pos + alpha * prior) / (weighted_count + alpha)
    return weighted_count, weighted_pos, rates


def loo_entity_features(ids, labels, weights, count, pos, prior, alpha):
    own_count = weights.astype(np.float64)
    own_pos = own_count * labels
    denominator = count[ids] - own_count
    numerator = pos[ids] - own_pos
    rate = (numerator + alpha * prior) / (denominator + alpha)
    log_count = np.log1p(np.maximum(denominator, 0.0))
    return rate.astype(np.float32), log_count.astype(np.float32)


def eval_entity_features(ids, count, rates):
    return (
        rates[ids].astype(np.float32),
        np.log1p(count[ids]).astype(np.float32),
    )


def safe_log_numeric(values):
    values = np.asarray(values, dtype=np.float32)
    missing = ~np.isfinite(values)
    clean = np.where(missing, 0.0, np.maximum(values, 0.0))
    logged = np.log1p(clean).astype(np.float32)
    logged[missing] = -1.0
    return logged


def make_lgb_matrix(split, video_features, author_features):
    columns = [
        np.asarray(split.X[name], dtype=np.float32)
        for name in CAT_FIELDS
    ]
    columns.extend(safe_log_numeric(split.num[name]) for name in NUM_FIELDS)
    columns.extend([
        video_features[0],
        video_features[1],
        author_features[0],
        author_features[1],
    ])
    return np.column_stack(columns).astype(np.float32, copy=False)


def make_deep_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64)
        for name in DEEP_FIELDS
    ])


def clipped_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
recency_weights = training_recency_weights(train.date, half_life=5.0)

weighted_prior = float(
    np.sum(recency_weights * y_train) / np.sum(recency_weights)
)

video_card = int(FEATURE_CARDINALITIES["video_id"])
author_card = int(FEATURE_CARDINALITIES["author_id"])

tr_video_ids = np.asarray(train.X["video_id"], dtype=np.int64)
va_video_ids = np.asarray(valid.X["video_id"], dtype=np.int64)
tr_author_ids = np.asarray(train.X["author_id"], dtype=np.int64)
va_author_ids = np.asarray(valid.X["author_id"], dtype=np.int64)

video_count, video_pos, video_rate = fit_entity_stats(
    tr_video_ids,
    y_train,
    recency_weights,
    video_card,
    weighted_prior,
    alpha=20.0,
)
author_count, author_pos, author_rate = fit_entity_stats(
    tr_author_ids,
    y_train,
    recency_weights,
    author_card,
    weighted_prior,
    alpha=25.0,
)

tr_video_features = loo_entity_features(
    tr_video_ids,
    y_train,
    recency_weights,
    video_count,
    video_pos,
    weighted_prior,
    alpha=20.0,
)
va_video_features = eval_entity_features(
    va_video_ids,
    video_count,
    video_rate,
)

tr_author_features = loo_entity_features(
    tr_author_ids,
    y_train,
    recency_weights,
    author_count,
    author_pos,
    weighted_prior,
    alpha=25.0,
)
va_author_features = eval_entity_features(
    va_author_ids,
    author_count,
    author_rate,
)

# Family 1: non-parametric recency-weighted empirical Bayes.
eb_valid_probability = (
    0.60 * va_video_features[0]
    + 0.40 * va_author_features[0]
)
eb_valid_scores = clipped_logit(eb_valid_probability)


# Family 2: nonlinear gradient-boosted trees over categorical, numeric, and
# leakage-safe leave-one-out entity statistics.
x_train_lgb = make_lgb_matrix(
    train,
    tr_video_features,
    tr_author_features,
)
x_valid_lgb = make_lgb_matrix(
    valid,
    va_video_features,
    va_author_features,
)

categorical_indices = list(range(len(CAT_FIELDS)))

lgb_train = lgb.Dataset(
    x_train_lgb,
    label=y_train,
    weight=recency_weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "num_threads": N_THREADS,
    "verbose": -1,
}

gbdt = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=230,
)
gbdt_valid_scores = gbdt.predict(
    x_valid_lgb,
    raw_score=True,
).astype(np.float64)

del lgb_train, x_train_lgb
gc.collect()


# Family 3: DeepFM, where the MLP forms higher-order interactions separately
# from the linear and pairwise FM pathways.
deep_cards = [int(FEATURE_CARDINALITIES[f]) for f in DEEP_FIELDS]
deep_offsets = np.cumsum([0] + deep_cards[:-1], dtype=np.int64)
deep_total_cardinality = int(sum(deep_cards))


class DeepFM(nn.Module):
    def __init__(self, cardinality, offsets, n_fields, embedding_dim=10):
        super().__init__()
        self.register_buffer(
            "offsets",
            torch.as_tensor(offsets, dtype=torch.long),
        )
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(n_fields * embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        indices = x + self.offsets
        linear = self.linear(indices).sum(dim=1).squeeze(-1)
        emb = self.embedding(indices)
        summed = emb.sum(dim=1)
        fm = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.mlp(emb.flatten(start_dim=1)).squeeze(-1)
        return self.bias + linear + fm + deep


@torch.inference_mode()
def deep_predict(model, matrix, batch_size=32768):
    model.eval()
    result = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        xb = torch.from_numpy(matrix[start:end])
        result[start:end] = model(xb).cpu().numpy()
    return result


x_train_deep_np = make_deep_matrix(train)
x_valid_deep_np = make_deep_matrix(valid)

x_train_deep = torch.from_numpy(x_train_deep_np)
deep_y = torch.from_numpy(y_train)
deep_w = torch.from_numpy(recency_weights)

deep_model = DeepFM(
    deep_total_cardinality,
    deep_offsets,
    len(DEEP_FIELDS),
    embedding_dim=10,
)
deep_optimizer = torch.optim.AdamW(
    deep_model.parameters(),
    lr=0.0012,
    weight_decay=1e-6,
)

generator = torch.Generator()
generator.manual_seed(SEED)
batch_size = 8192
n_train = len(y_train)

for epoch in range(4):
    deep_model.train()
    permutation = torch.randperm(n_train, generator=generator)
    epoch_loss = 0.0
    epoch_weight = 0.0

    for start in range(0, n_train, batch_size):
        idx = permutation[start:start + batch_size]
        xb = x_train_deep[idx]
        yb = deep_y[idx]
        wb = deep_w[idx]

        deep_optimizer.zero_grad(set_to_none=True)
        logits = deep_model(xb)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits,
            yb,
            reduction="none",
        )
        loss = (losses * wb).sum() / wb.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(deep_model.parameters(), 5.0)
        deep_optimizer.step()

        epoch_loss += float((losses.detach() * wb).sum())
        epoch_weight += float(wb.sum())

    print(
        "FINDINGS deepfm_epoch={} weighted_loss={:.6f}".format(
            epoch + 1,
            epoch_loss / max(epoch_weight, 1.0),
        ),
        flush=True,
    )

deep_valid_scores = deep_predict(deep_model, x_valid_deep_np)


# Load test features only, then create predictions from every fitted family.
test = load("test")
te_video_ids = np.asarray(test.X["video_id"], dtype=np.int64)
te_author_ids = np.asarray(test.X["author_id"], dtype=np.int64)

te_video_features = eval_entity_features(
    te_video_ids,
    video_count,
    video_rate,
)
te_author_features = eval_entity_features(
    te_author_ids,
    author_count,
    author_rate,
)

eb_test_probability = (
    0.60 * te_video_features[0]
    + 0.40 * te_author_features[0]
)
eb_test_scores = clipped_logit(eb_test_probability)

x_test_lgb = make_lgb_matrix(
    test,
    te_video_features,
    te_author_features,
)
gbdt_test_scores = gbdt.predict(
    x_test_lgb,
    raw_score=True,
).astype(np.float64)

x_test_deep_np = make_deep_matrix(test)
deep_test_scores = deep_predict(deep_model, x_test_deep_np)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not (os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)):
    raise FileNotFoundError("Trusted incumbent predictions are required")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)

families = {
    "empirical_bayes": (eb_valid_scores, eb_test_scores),
    "gbdt_binary": (gbdt_valid_scores, gbdt_test_scores),
    "deepfm": (deep_valid_scores, deep_test_scores),
}

candidate_results = {}
best_primary = -np.inf
best_metrics = None
best_valid_scores = None
best_test_scores = None
best_raw_valid_scores = None
best_name = None
best_blended = False

# Include each standalone model and several fixed convex blends. Since all
# scores are logits, direct blending preserves a meaningful common scale.
for family_name, (valid_scores_raw, test_scores_raw) in families.items():
    raw_metrics = evaluate(valid.user_id, y_valid, valid_scores_raw)
    candidate_results[family_name] = float(raw_metrics["primary"])

    if float(raw_metrics["primary"]) > best_primary:
        best_primary = float(raw_metrics["primary"])
        best_metrics = raw_metrics
        best_valid_scores = valid_scores_raw.copy()
        best_test_scores = test_scores_raw.copy()
        best_raw_valid_scores = None
        best_name = family_name
        best_blended = False

    for own_weight in (0.20, 0.35, 0.50, 0.65, 0.80):
        blend_valid = (
            own_weight * valid_scores_raw
            + (1.0 - own_weight) * inc_valid
        )
        blend_metrics = evaluate(valid.user_id, y_valid, blend_valid)
        blend_name = "{}_blend_{:.2f}".format(family_name, own_weight)
        candidate_results[blend_name] = float(blend_metrics["primary"])

        if float(blend_metrics["primary"]) > best_primary:
            best_primary = float(blend_metrics["primary"])
            best_metrics = blend_metrics
            best_valid_scores = blend_valid.copy()
            best_test_scores = (
                own_weight * test_scores_raw
                + (1.0 - own_weight) * inc_test
            )
            best_raw_valid_scores = valid_scores_raw.copy()
            best_name = blend_name
            best_blended = True

print(
    "FINDINGS weighted_train_prior={:.6f} unweighted_train_prior={:.6f} winner={}".format(
        weighted_prior,
        float(np.mean(y_train)),
        best_name,
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(
        candidate_results,
        sort_keys=True,
        separators=(",", ":"),
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
    if best_blended:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid_scores, dtype=np.float64),
        )

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(",", ":")))