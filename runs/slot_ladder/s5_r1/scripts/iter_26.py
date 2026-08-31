import os
import time
import json

import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260831
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "onehot_feat3",
    "upload_type",
    "music_type",
]


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

    ranked = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def make_matrix(split):
    return np.column_stack(
        [np.asarray(split.X[f], dtype=np.int32) for f in FIELDS]
    )


def equal_user_weights(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    _, inverse, counts = np.unique(
        user_ids, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


class BalancedEmpiricalBayes:
    def __init__(self, train, sample_weight):
        y = np.asarray(train.y, dtype=np.float64)
        w = np.asarray(sample_weight, dtype=np.float64)

        self.global_rate = float(np.sum(w * y) / np.sum(w))
        self.global_logit = float(
            np.log(np.clip(self.global_rate, 1e-5, 1 - 1e-5))
            - np.log1p(-np.clip(self.global_rate, 1e-5, 1 - 1e-5))
        )

        self.tables = {}
        self.coefficients = {
            "user_id": 0.08,
            "video_id": 0.52,
            "author_id": 0.48,
            "tab": 0.18,
            "duration_bucket": 0.18,
            "tag": 0.30,
            "onehot_feat3": 0.24,
            "upload_type": 0.18,
            "music_type": 0.10,
        }

        for field in FIELDS:
            ids = np.asarray(train.X[field], dtype=np.int64)
            card = int(FEATURE_CARDINALITIES[field])

            weighted_count = np.bincount(
                ids, weights=w, minlength=card
            ).astype(np.float64)
            weighted_positive = np.bincount(
                ids, weights=w * y, minlength=card
            ).astype(np.float64)

            if field == "user_id":
                prior = 45.0
            elif card > 1000:
                prior = 30.0
            elif card > 100:
                prior = 50.0
            else:
                prior = 80.0

            rate = (
                weighted_positive + prior * self.global_rate
            ) / (weighted_count + prior)
            rate = np.clip(rate, 0.01, 0.99)
            residual = np.log(rate) - np.log1p(-rate) - self.global_logit
            residual[weighted_count == 0] = 0.0

            self.tables[field] = (
                residual.astype(np.float32),
                weighted_count > 0,
            )

    def predict(self, split):
        score = np.zeros(len(split.user_id), dtype=np.float64)

        for field in FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            values, seen_table = self.tables[field]
            valid = (ids >= 0) & (ids < len(values))
            safe_ids = np.where(valid, ids, 0)
            seen = valid & seen_table[safe_ids]
            component = np.where(seen, values[safe_ids], 0.0)
            score += self.coefficients[field] * component

        return score


class BalancedFM(nn.Module):
    def __init__(self, cardinalities, dimension=16):
        super().__init__()
        self.cardinalities = list(cardinalities)
        offsets = np.cumsum([0] + self.cardinalities[:-1])
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )
        total = int(sum(self.cardinalities))

        self.linear = nn.Embedding(total, 1)
        self.embedding = nn.Embedding(total, dimension)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        z = x + self.offsets
        linear_term = self.linear(z).sum(dim=1).squeeze(1)

        embeddings = self.embedding(z)
        summed = embeddings.sum(dim=1)
        interaction = 0.5 * (
            summed.square()
            - embeddings.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear_term + interaction


def train_fm(train_matrix, labels, sample_weight):
    cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
    model = BalancedFM(cardinalities, dimension=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)

    labels = np.asarray(labels, dtype=np.float32)
    sample_weight = np.asarray(sample_weight, dtype=np.float32)
    rng = np.random.default_rng(SEED)
    batch_size = 65536
    n = len(labels)

    model.train()
    for _ in range(4):
        permutation = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.from_numpy(
                np.asarray(train_matrix[idx], dtype=np.int64)
            )
            yb = torch.from_numpy(labels[idx])
            wb = torch.from_numpy(sample_weight[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            optimizer.step()

    return model


def predict_fm(model, matrix):
    model.eval()
    output = np.empty(len(matrix), dtype=np.float32)
    batch_size = 131072

    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            xb = torch.from_numpy(
                np.asarray(matrix[start:end], dtype=np.int64)
            )
            output[start:end] = model(xb).cpu().numpy()

    return output.astype(np.float64)


train = load("train")
valid = load("valid")

train_matrix = make_matrix(train)
valid_matrix = make_matrix(valid)
train_y = np.asarray(train.y, dtype=np.float32)
user_weight = equal_user_weights(train.user_id)

# Family 1: user-balanced empirical-Bayes additive evidence.
eb_model = BalancedEmpiricalBayes(train, user_weight)
eb_valid = eb_model.predict(valid)

# Family 2: user-balanced binary gradient-boosted decision trees.
lgb_train = lgb.Dataset(
    train_matrix,
    label=train_y,
    weight=user_weight,
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=True,
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 8.0,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 30.0,
    "cat_l2": 12.0,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "verbose": -1,
}

tree_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=220,
)
tree_valid = tree_model.predict(valid_matrix)

# Family 3: user-balanced second-order factorisation.
fm_model = train_fm(train_matrix, train_y, user_weight)
fm_valid = predict_fm(fm_model, valid_matrix)

own_valid_raw = {
    "balanced_empirical_bayes": eb_valid,
    "balanced_binary_gbdt": tree_valid,
    "balanced_fm": fm_valid,
}
own_valid_rank = {
    name: within_user_rank(valid.user_id, score)
    for name, score in own_valid_raw.items()
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

blend_weights = (0.10, 0.20, 0.30, 0.40, 0.50)

for family, family_rank in own_valid_rank.items():
    standalone = family + "_standalone"
    candidate_scores[standalone] = family_rank
    candidate_primary[standalone] = float(
        evaluate(valid.user_id, valid.y, family_rank)["primary"]
    )
    recipes[standalone] = ("standalone", family, 1.0)

    for alpha in blend_weights:
        name = f"{family}_blend_{alpha:.2f}"
        scores = (1.0 - alpha) * inc_valid_rank + alpha * family_rank
        candidate_scores[name] = scores
        candidate_primary[name] = float(
            evaluate(valid.user_id, valid.y, scores)["primary"]
        )
        recipes[name] = ("blend", family, alpha)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)
recipe_type, winner_family, winner_alpha = recipes[winner]

best_own_family = max(
    own_valid_rank,
    key=lambda family: candidate_primary[family + "_standalone"],
)
raw_for_audit = own_valid_rank[
    winner_family if winner_family in own_valid_rank else best_own_family
]

unique_users = np.unique(train.user_id).size
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "best_own_family": best_own_family,
            "incumbent_primary": candidate_primary["incumbent"],
            "standalone_primary": {
                family: candidate_primary[family + "_standalone"]
                for family in own_valid_rank
            },
            "training_users": int(unique_users),
            "user_weight_min": float(user_weight.min()),
            "user_weight_median": float(np.median(user_weight)),
            "user_weight_max": float(user_weight.max()),
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
test_matrix = make_matrix(test)

eb_test = eb_model.predict(test)
tree_test = tree_model.predict(test_matrix)
fm_test = predict_fm(fm_model, test_matrix)

own_test_rank = {
    "balanced_empirical_bayes": within_user_rank(test.user_id, eb_test),
    "balanced_binary_gbdt": within_user_rank(test.user_id, tree_test),
    "balanced_fm": within_user_rank(test.user_id, fm_test),
}

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if recipe_type == "incumbent":
    test_scores = inc_test_rank
elif recipe_type == "standalone":
    test_scores = own_test_rank[winner_family]
else:
    test_scores = (
        (1.0 - winner_alpha) * inc_test_rank
        + winner_alpha * own_test_rank[winner_family]
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
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)