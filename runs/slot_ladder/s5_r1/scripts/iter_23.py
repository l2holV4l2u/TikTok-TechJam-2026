import os
import time
import json
import warnings

import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
warnings.filterwarnings("ignore")
torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.manual_seed(2026)
np.random.seed(2026)

FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
    "tab",
    "music_type",
    "video_type",
    "hour",
    "is_live_streamer",
    "is_video_author",
]

HALF_LIFE_DAYS = 4.0
BLEND_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.50)


def date_to_day(date_array):
    dates = np.asarray(date_array, dtype=np.int64)
    unique = np.unique(dates)
    mapping = {int(d): i for i, d in enumerate(unique)}
    return np.asarray([mapping[int(d)] for d in dates], dtype=np.int16)


def recency_weights(date_array):
    day = date_to_day(date_array).astype(np.float64)
    age = day.max() - day
    weight = np.power(0.5, age / HALF_LIFE_DAYS)
    return weight / weight.mean()


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    sizes = np.diff(np.r_[starts, n])

    position = np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ranked = position / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def categorical_matrix(split):
    return np.column_stack(
        [np.asarray(split.X[field], dtype=np.int32) for field in FIELDS]
    )


class RecencyNaiveBayes:
    def fit(self, train, sample_weight):
        y = np.asarray(train.y, dtype=np.float64)
        w = np.asarray(sample_weight, dtype=np.float64)
        self.global_rate = float(np.sum(w * y) / np.sum(w))
        self.global_logit = float(
            np.log(np.clip(self.global_rate, 1e-5, 1 - 1e-5))
            - np.log1p(-np.clip(self.global_rate, 1e-5, 1 - 1e-5))
        )
        self.tables = {}

        for field in FIELDS:
            ids = np.asarray(train.X[field], dtype=np.int64)
            card = int(FEATURE_CARDINALITIES[field])

            count = np.bincount(
                ids, weights=w, minlength=card
            ).astype(np.float64)
            positive = np.bincount(
                ids, weights=w * y, minlength=card
            ).astype(np.float64)

            smoothing = 35.0 if card > 100 else 75.0
            rate = (
                positive + smoothing * self.global_rate
            ) / (count + smoothing)
            rate = np.clip(rate, 0.015, 0.985)
            residual = np.log(rate) - np.log1p(-rate) - self.global_logit

            reliability = count / (count + smoothing)
            residual *= reliability

            self.tables[field] = residual.astype(np.float32)

        return self

    def predict(self, split):
        score = np.zeros(len(split.user_id), dtype=np.float64)
        for field in FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            table = self.tables[field]
            safe = np.clip(ids, 0, len(table) - 1)
            component = table[safe]
            component = np.where(
                (ids >= 0) & (ids < len(table)), component, 0.0
            )
            score += component
        return score


class AdditiveWide(nn.Module):
    def __init__(self):
        super().__init__()
        self.offsets = []
        total = 0
        for field in FIELDS:
            self.offsets.append(total)
            total += int(FEATURE_CARDINALITIES[field])
        self.register_buffer(
            "offset_tensor",
            torch.tensor(self.offsets, dtype=torch.long),
        )
        self.embedding = nn.Embedding(total, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.embedding.weight)

    def forward(self, x):
        shifted = x.long() + self.offset_tensor[None, :]
        return self.embedding(shifted).squeeze(-1).sum(dim=1) + self.bias


def fit_additive_wide(train_matrix, labels, weights):
    model = AdditiveWide()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.035, weight_decay=2e-5
    )

    n = len(labels)
    batch_size = 65536
    x_all = torch.from_numpy(train_matrix.astype(np.int64, copy=False))
    y_all = torch.from_numpy(labels.astype(np.float32, copy=False))
    w_all = torch.from_numpy(weights.astype(np.float32, copy=False))

    generator = torch.Generator()
    generator.manual_seed(2026)

    model.train()
    for _ in range(3):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            logits = model(x_all[idx])
            loss_vector = F.binary_cross_entropy_with_logits(
                logits, y_all[idx], reduction="none"
            )
            loss = torch.sum(loss_vector * w_all[idx]) / torch.sum(w_all[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def predict_additive_wide(model, matrix):
    model.eval()
    result = np.empty(len(matrix), dtype=np.float64)
    batch_size = 131072
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            x = torch.from_numpy(
                matrix[start:end].astype(np.int64, copy=False)
            )
            result[start:end] = model(x).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
sample_weight = recency_weights(train.date).astype(np.float32)

x_train = categorical_matrix(train)
x_valid = categorical_matrix(valid)

# Family 1: additive wide logistic model, implemented directly in torch to
# avoid the unavailable sklearn dependency from the failed attempt.
wide_model = fit_additive_wide(x_train, train_y, sample_weight)
valid_wide = predict_additive_wide(wide_model, x_valid)

# Family 2: nonlinear gradient-boosted trees over the same categorical inputs.
lgb_train = lgb.Dataset(
    x_train,
    label=train_y,
    weight=sample_weight,
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=True,
)

gbdt_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 4.0,
    "max_bin": 127,
    "cat_smooth": 30.0,
    "cat_l2": 15.0,
    "min_data_per_group": 100,
    "verbosity": -1,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": 2026,
    "feature_fraction_seed": 2026,
    "bagging_seed": 2026,
}

gbdt_model = lgb.train(
    gbdt_params,
    lgb_train,
    num_boost_round=220,
)
valid_gbdt = gbdt_model.predict(x_valid)

# Family 3: recency-weighted empirical-Bayes evidence aggregation.
bayes_model = RecencyNaiveBayes().fit(train, sample_weight)
valid_bayes = bayes_model.predict(valid)

valid_raw = {
    "recency_wide": within_user_rank(valid.user_id, valid_wide),
    "recency_gbdt": within_user_rank(valid.user_id, valid_gbdt),
    "recency_bayes": within_user_rank(valid.user_id, valid_bayes),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores are missing")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test scores are missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {"incumbent": inc_valid_rank}
candidate_primary = {
    "incumbent": float(
        evaluate(valid.user_id, valid.y, inc_valid_rank)["primary"]
    )
}
recipes = {"incumbent": ("incumbent", "", 0.0)}

for family, own_rank in valid_raw.items():
    standalone = family + "_standalone"
    candidate_scores[standalone] = own_rank
    candidate_primary[standalone] = float(
        evaluate(valid.user_id, valid.y, own_rank)["primary"]
    )
    recipes[standalone] = ("standalone", family, 1.0)

    for alpha in BLEND_WEIGHTS:
        name = f"{family}_blend_{alpha:.2f}"
        blended = (1.0 - alpha) * inc_valid_rank + alpha * own_rank
        candidate_scores[name] = blended
        candidate_primary[name] = float(
            evaluate(valid.user_id, valid.y, blended)["primary"]
        )
        recipes[name] = ("blend", family, alpha)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
winner_metrics = evaluate(valid.user_id, valid.y, valid_scores)
recipe_type, winner_family, winner_alpha = recipes[winner]

best_own_family = max(
    valid_raw,
    key=lambda name: candidate_primary[name + "_standalone"],
)
raw_for_audit = valid_raw[
    winner_family if winner_family in valid_raw else best_own_family
]

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "best_own_family": best_own_family,
            "half_life_days": HALF_LIFE_DAYS,
            "oldest_to_newest_weight_ratio": float(
                sample_weight.min() / sample_weight.max()
            ),
            "standalone": {
                family: candidate_primary[family + "_standalone"]
                for family in valid_raw
            },
            "incumbent": candidate_primary["incumbent"],
        },
        separators=(",", ":"),
    )
)

print(
    "CANDIDATES "
    + json.dumps(
        {name: float(value) for name, value in candidate_primary.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

test = load("test")
x_test = categorical_matrix(test)

test_raw = {
    "recency_wide": within_user_rank(
        test.user_id, predict_additive_wide(wide_model, x_test)
    ),
    "recency_gbdt": within_user_rank(
        test.user_id, gbdt_model.predict(x_test)
    ),
    "recency_bayes": within_user_rank(
        test.user_id, bayes_model.predict(test)
    ),
}

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if recipe_type == "incumbent":
    test_scores = inc_test_rank
elif recipe_type == "standalone":
    test_scores = test_raw[winner_family]
else:
    test_scores = (
        (1.0 - winner_alpha) * inc_test_rank
        + winner_alpha * test_raw[winner_family]
    )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(raw_for_audit, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(winner_metrics["primary"]),
            "gauc": float(winner_metrics["gauc"]),
            "ndcg@5": float(winner_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)