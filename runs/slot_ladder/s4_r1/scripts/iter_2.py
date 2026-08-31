import os
import time
import json
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
THREADS = min(8, os.cpu_count() or 1)
HALF_LIFE = 4.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int32)
date_age = train_dates.max() - train_dates
sample_weight = np.power(0.5, date_age.astype(np.float32) / HALF_LIFE)
sample_weight /= sample_weight.mean()

candidate_pairs = {}
candidate_primary = {}
candidate_details = {}


def register_candidate(name, valid_scores, test_scores):
    valid_scores = np.asarray(valid_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    met = evaluate(valid.user_id, valid.y, valid_scores)
    candidate_pairs[name] = (valid_scores, test_scores)
    candidate_primary[name] = float(met["primary"])
    candidate_details[name] = met
    return met


# ----------------------------------------------------------------------
# Family 1: empirical-Bayes target statistics with recent-day weighting.
# Only training labels are used. Within-user signal comes from video,
# author, tag, tab, and duration preferences; user-only offsets are omitted
# because they are constant inside a user's candidate set.
# ----------------------------------------------------------------------
def safe_logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


global_rate = float(np.sum(sample_weight * y_train) / np.sum(sample_weight))
global_logit = float(safe_logit(global_rate))

eb_specs = [
    ("video_id", 10.0, 1.00),
    ("author_id", 20.0, 0.55),
    ("tag", 80.0, 0.32),
    ("tab", 120.0, 0.32),
    ("duration_bucket", 100.0, 0.20),
]

eb_tables = {}
for field, prior, coefficient in eb_specs:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])
    count = np.bincount(ids, weights=sample_weight, minlength=card).astype(np.float64)
    positive = np.bincount(
        ids, weights=sample_weight * y_train, minlength=card
    ).astype(np.float64)
    rate = (positive + prior * global_rate) / (count + prior)
    reliability = count / (count + prior)
    delta = reliability * (safe_logit(rate) - global_logit)
    delta[0] = 0.0
    eb_tables[field] = coefficient * delta


def predict_eb(split):
    score = np.full(len(split.user_id), global_logit, dtype=np.float64)
    for field, _, _ in eb_specs:
        ids = np.asarray(split.X[field], dtype=np.int64)
        score += eb_tables[field][ids]
    return score


eb_valid = predict_eb(valid)
eb_test = predict_eb(test)
register_candidate("empirical_bayes", eb_valid, eb_test)


# ----------------------------------------------------------------------
# Family 2: history-augmented LightGBM.
# Entity histories convert unstable identity IDs into train-only behavioral
# summaries, while trees form conditional interactions with stable context
# and numeric quantities. Recent rows receive greater training weight.
# ----------------------------------------------------------------------
lgb_cat_fields = [
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "register_days_range",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "is_live_streamer",
    "is_video_author",
    "onehot_feat1",
    "onehot_feat2",
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

hist_train_video = historical_features("train", key="video_id")
hist_valid_video = historical_features("valid", key="video_id")
hist_test_video = historical_features("test", key="video_id")
hist_train_author = historical_features("train", key="author_id")
hist_valid_author = historical_features("valid", key="author_id")
hist_test_author = historical_features("test", key="author_id")

video_hist_names = sorted(hist_train_video.keys())
author_hist_names = sorted(hist_train_author.keys())


def make_lgb_matrix(split, video_hist, author_hist):
    columns = []

    for field in lgb_cat_fields:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in numeric_fields:
        raw = np.asarray(split.num[field], dtype=np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(raw, 0.0)).astype(np.float32))

    for name in video_hist_names:
        columns.append(np.asarray(video_hist[name], dtype=np.float32))

    for name in author_hist_names:
        columns.append(np.asarray(author_hist[name], dtype=np.float32))

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


x_lgb_train = make_lgb_matrix(train, hist_train_video, hist_train_author)
x_lgb_valid = make_lgb_matrix(valid, hist_valid_video, hist_valid_author)
x_lgb_test = make_lgb_matrix(test, hist_test_video, hist_test_author)

del hist_train_video, hist_valid_video, hist_test_video
del hist_train_author, hist_valid_author, hist_test_author
gc.collect()

categorical_indices = list(range(len(lgb_cat_fields)))
lgb_data = lgb.Dataset(
    x_lgb_train,
    label=y_train,
    weight=sample_weight,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 700,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "cat_smooth": 25.0,
    "cat_l2": 12.0,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "num_threads": THREADS,
    "verbose": -1,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_data,
    num_boost_round=190,
)

lgb_valid_prob = lgb_model.predict(x_lgb_valid)
lgb_test_prob = lgb_model.predict(x_lgb_test)
lgb_valid_score = safe_logit(lgb_valid_prob)
lgb_test_score = safe_logit(lgb_test_prob)
register_candidate("lightgbm_history", lgb_valid_score, lgb_test_score)

del x_lgb_train, x_lgb_valid, x_lgb_test, lgb_data, lgb_model
gc.collect()


# ----------------------------------------------------------------------
# Family 3: DeepFM.
# The FM branch retains robust low-rank pairwise interactions; the deep
# branch forms higher-order user/item/context interactions. Its main-model
# BCE is explicitly weighted toward the end of the training period.
# ----------------------------------------------------------------------
deep_fields = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "user_active_degree",
]

cards = [int(FEATURE_CARDINALITIES[f]) for f in deep_fields]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_deep_matrix(split):
    cols = []
    for field, offset, card in zip(deep_fields, offsets, cards):
        ids = np.asarray(split.X[field], dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= card):
            raise ValueError("categorical ID outside declared cardinality")
        cols.append(ids + offset)
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int64)


class DeepFM(nn.Module):
    def __init__(self, n_features, n_fields, embedding_dim=12):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.deep = nn.Sequential(
            nn.Linear(n_fields * embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.015)
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        emb = self.embedding(x)
        linear = self.linear(x).squeeze(-1).sum(dim=1)

        summed = emb.sum(dim=1)
        fm = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)

        deep = self.deep(emb.flatten(start_dim=1)).squeeze(1)
        return self.bias + linear + fm + deep


x_deep_train = make_deep_matrix(train)
x_deep_valid = make_deep_matrix(valid)
x_deep_test = make_deep_matrix(test)

deep_model = DeepFM(total_cardinality, len(deep_fields))
deep_optimizer = torch.optim.AdamW(
    deep_model.parameters(), lr=0.0018, weight_decay=2e-6
)

rng = np.random.default_rng(SEED + 10)
batch_size = 4096
n_train = len(y_train)
deep_model.train()

for epoch in range(4):
    order = rng.permutation(n_train)
    for start in range(0, n_train, batch_size):
        idx = order[start:start + batch_size]
        xb = torch.from_numpy(x_deep_train[idx])
        yb = torch.from_numpy(y_train[idx])
        wb = torch.from_numpy(sample_weight[idx])

        deep_optimizer.zero_grad(set_to_none=True)
        logits = deep_model(xb)
        losses = F.binary_cross_entropy_with_logits(
            logits, yb, reduction="none"
        )
        loss = torch.sum(losses * wb) / torch.sum(wb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(deep_model.parameters(), 5.0)
        deep_optimizer.step()


@torch.inference_mode()
def predict_deep(x):
    deep_model.eval()
    out = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], 32768):
        end = min(start + 32768, x.shape[0])
        xb = torch.from_numpy(x[start:end])
        out[start:end] = deep_model(xb).cpu().numpy().astype(np.float64)
    return out


deep_valid = predict_deep(x_deep_valid)
deep_test = predict_deep(x_deep_test)
register_candidate("deepfm_recent", deep_valid, deep_test)

del x_deep_train, x_deep_valid, x_deep_test, deep_model, deep_optimizer
gc.collect()


# ----------------------------------------------------------------------
# Blend each structurally different family with the trusted incumbent.
# Both components are logits or log-odds-like scores, so no validation-
# fitted normalization is used. Alpha is selected using the explicitly
# reusable public-validation blending protocol.
# ----------------------------------------------------------------------
shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")

best_name = max(candidate_primary, key=candidate_primary.get)
best_valid, best_test = candidate_pairs[best_name]
best_metrics = candidate_details[best_name]
best_raw_valid = None

if os.path.exists(inc_valid_path) and os.path.exists(inc_test_path):
    incumbent_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
    incumbent_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

    inc_metrics = register_candidate(
        "trusted_incumbent", incumbent_valid, incumbent_test
    )

    blend_alphas = [0.05, 0.10, 0.16, 0.23, 0.32, 0.43, 0.56, 0.70, 0.85]

    own_family_names = [
        "empirical_bayes",
        "lightgbm_history",
        "deepfm_recent",
    ]

    for family_name in own_family_names:
        own_valid, own_test = candidate_pairs[family_name]
        for alpha in blend_alphas:
            blended_valid = (
                (1.0 - alpha) * incumbent_valid + alpha * own_valid
            )
            blended_test = (
                (1.0 - alpha) * incumbent_test + alpha * own_test
            )
            blend_name = "%s_blend_%.2f" % (family_name, alpha)
            met = register_candidate(
                blend_name, blended_valid, blended_test
            )

    best_name = max(candidate_primary, key=candidate_primary.get)
    best_valid, best_test = candidate_pairs[best_name]
    best_metrics = candidate_details[best_name]

    if "_blend_" in best_name:
        raw_family = best_name.split("_blend_")[0]
        best_raw_valid = candidate_pairs[raw_family][0]

print(
    "FINDINGS temporal_weight_half_life=4_days "
    "global_weighted_rate=%.6f winner=%s"
    % (global_rate, best_name)
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: round(v, 10) for k, v in candidate_primary.items()},
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_raw_valid is not None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
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