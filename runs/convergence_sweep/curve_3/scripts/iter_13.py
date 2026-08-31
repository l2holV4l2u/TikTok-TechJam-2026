import os
import time
import json
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
SEED = 73129
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag", "upload_type",
    "onehot_feat3", "onehot_feat8", "duration_bucket", "onehot_feat1",
    "music_type", "onehot_feat7", "user_active_degree",
    "register_days_bucket", "register_days_range", "onehot_feat0",
    "onehot_feat12", "fans_user_num_range",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HISTORY_SUFFIXES = [
    "train_count_log1p", "long_view_rate", "is_click_rate",
    "play_time_ms_logmean",
]
HALF_LIFE = 5.0
BATCH_SIZE = 16384
PRED_BATCH_SIZE = 32768
LINEAR_EPOCHS = 3


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    weights = np.power(
        0.5, age.astype(np.float32) / HALF_LIFE
    ).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return np.ascontiguousarray(weights)


def make_cat_matrix(split):
    result = np.empty(
        (len(split.user_id), len(CAT_FIELDS)), dtype=np.int32
    )
    for j, name in enumerate(CAT_FIELDS):
        result[:, j] = np.asarray(split.X[name], dtype=np.int32)
    return np.ascontiguousarray(result)


def make_offset_cats(cat_matrix):
    cards = np.asarray(
        [int(FEATURE_CARDINALITIES[x]) for x in CAT_FIELDS],
        dtype=np.int64,
    )
    offsets = np.cumsum(
        np.concatenate(([0], cards[:-1])), dtype=np.int64
    )
    result = cat_matrix.astype(np.int64, copy=True)
    result += offsets[None, :]
    return np.ascontiguousarray(result), cards


def make_numeric_raw(split, split_name):
    columns = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        hist = historical_features(split_name, key=entity)
        for suffix in HISTORY_SUFFIXES:
            key = entity + "_" + suffix
            x = np.asarray(hist[key], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            columns.append(x.astype(np.float32))
        del hist

    return np.ascontiguousarray(
        np.column_stack(columns), dtype=np.float32
    )


def fit_numeric_transform(x):
    lo = np.quantile(x, 0.002, axis=0).astype(np.float32)
    hi = np.quantile(x, 0.998, axis=0).astype(np.float32)
    clipped = np.clip(x, lo, hi)
    mean = clipped.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = clipped.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-4] = 1.0
    return lo, hi, mean, std


def transform_numeric(x, transform):
    lo, hi, mean, std = transform
    return np.ascontiguousarray(
        (np.clip(x, lo, hi) - mean) / std,
        dtype=np.float32,
    )


class AdditiveLinear(nn.Module):
    def __init__(self, total_categories, n_num):
        super().__init__()
        self.cat_weight = nn.Embedding(total_categories, 1)
        self.num_weight = nn.Linear(n_num, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.cat_weight.weight)
        nn.init.zeros_(self.num_weight.weight)
        nn.init.zeros_(self.num_weight.bias)

    def forward(self, cats, nums):
        return (
            self.cat_weight(cats).sum(dim=1).squeeze(1)
            + self.num_weight(nums).squeeze(1)
            + self.bias
        )


def train_linear(model, cats, nums, labels, weights):
    rng = np.random.default_rng(SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.012, weight_decay=2e-5
    )
    labels = np.asarray(labels, dtype=np.float32)

    model.train()
    for _ in range(LINEAR_EPOCHS):
        order = rng.permutation(len(labels))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            c = torch.from_numpy(cats[idx])
            x = torch.from_numpy(nums[idx])
            y = torch.from_numpy(labels[idx])
            w = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(c, x)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = (row_loss * w).sum() / w.sum()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_linear(model, cats, nums):
    model.eval()
    result = np.empty(len(cats), dtype=np.float32)
    for start in range(0, len(cats), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(cats))
        result[start:end] = model(
            torch.from_numpy(cats[start:end]),
            torch.from_numpy(nums[start:end]),
        ).cpu().numpy()
    return result


class NaiveBayesModel:
    def __init__(self, cat_log_ratio, num_coef, num_const, prior):
        self.cat_log_ratio = cat_log_ratio
        self.num_coef = num_coef
        self.num_const = num_const
        self.prior = float(prior)

    def predict(self, cats, nums):
        score = np.full(len(cats), self.prior, dtype=np.float64)
        for j, values in enumerate(self.cat_log_ratio):
            score += values[cats[:, j]]
        score += nums.astype(np.float64) @ self.num_coef
        score += self.num_const
        return score.astype(np.float32)


def fit_naive_bayes(cats, nums, labels, weights, cards):
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    w1 = weights * labels
    w0 = weights * (1.0 - labels)
    total1 = float(w1.sum())
    total0 = float(w0.sum())
    smoothing = 4.0

    cat_ratios = []
    for j, card in enumerate(cards):
        card = int(card)
        c1 = np.bincount(
            cats[:, j], weights=w1, minlength=card
        ).astype(np.float64)
        c0 = np.bincount(
            cats[:, j], weights=w0, minlength=card
        ).astype(np.float64)
        log_p1 = np.log(c1 + smoothing) - np.log(
            total1 + smoothing * card
        )
        log_p0 = np.log(c0 + smoothing) - np.log(
            total0 + smoothing * card
        )
        ratio = np.clip(log_p1 - log_p0, -3.0, 3.0)
        cat_ratios.append(ratio.astype(np.float32))

    sum1 = np.maximum(total1, 1e-8)
    sum0 = np.maximum(total0, 1e-8)
    mean1 = (nums * w1[:, None]).sum(axis=0) / sum1
    mean0 = (nums * w0[:, None]).sum(axis=0) / sum0
    var1 = (
        ((nums - mean1) ** 2) * w1[:, None]
    ).sum(axis=0) / sum1
    var0 = (
        ((nums - mean0) ** 2) * w0[:, None]
    ).sum(axis=0) / sum0
    var1 = np.maximum(var1, 0.08)
    var0 = np.maximum(var0, 0.08)

    # Difference of diagonal Gaussian log likelihoods. The quadratic
    # component is evaluated explicitly through augmented numeric columns.
    coef_linear = mean1 / var1 - mean0 / var0
    coef_square = -0.5 / var1 + 0.5 / var0
    const = -0.5 * np.sum(
        np.log(var1 / var0)
        + mean1 * mean1 / var1
        - mean0 * mean0 / var0
    )
    prior = np.log((total1 + 1.0) / (total0 + 1.0))

    # Store coefficients for [x, x^2]. Numeric evidence is shrunk because
    # categorical conditional independence otherwise produces overconfident
    # scores.
    num_coef = 0.35 * np.concatenate(
        [coef_linear, coef_square]
    ).astype(np.float64)
    return NaiveBayesModel(
        cat_ratios, num_coef, 0.35 * const, prior
    )


def augment_nb_numeric(nums):
    return np.ascontiguousarray(
        np.concatenate([nums, nums * nums], axis=1),
        dtype=np.float32,
    )


def make_tree_matrix(cats, nums):
    return np.ascontiguousarray(
        np.concatenate(
            [cats.astype(np.float32), nums.astype(np.float32)], axis=1
        ),
        dtype=np.float32,
    )


def within_user_rank(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    row = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]

    changed = np.empty(len(order), dtype=bool)
    changed[0] = True
    changed[1:] = sorted_users[1:] != sorted_users[:-1]

    positions = np.arange(len(order), dtype=np.int64)
    starts = np.maximum.accumulate(np.where(changed, positions, 0))
    local_rank = positions - starts

    _, inverse, counts = np.unique(
        sorted_users, return_inverse=True, return_counts=True
    )
    denom = np.maximum(counts[inverse] - 1, 1)
    normalized = local_rank.astype(np.float64) / denom

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = normalized
    return result


def centered_scaled(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    _, inv = np.unique(users, return_inverse=True)
    counts = np.bincount(inv)
    means = np.bincount(inv, weights=scores) / np.maximum(counts, 1)
    centered = scores - means[inv]
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return centered / scale


def metric_primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


train = load("train")
valid = load("valid")
test = load("test")

train_cat = make_cat_matrix(train)
valid_cat = make_cat_matrix(valid)
test_cat = make_cat_matrix(test)

train_offset, cards = make_offset_cats(train_cat)
valid_offset, _ = make_offset_cats(valid_cat)
test_offset, _ = make_offset_cats(test_cat)

train_raw = make_numeric_raw(train, "train")
valid_raw = make_numeric_raw(valid, "valid")
test_raw = make_numeric_raw(test, "test")
num_transform = fit_numeric_transform(train_raw)
train_num = transform_numeric(train_raw, num_transform)
valid_num = transform_numeric(valid_raw, num_transform)
test_num = transform_numeric(test_raw, num_transform)
del train_raw, valid_raw, test_raw
gc.collect()

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
weights = recency_weights(train.date)

# Family 1: recency-weighted additive linear discrimination.
linear_model = AdditiveLinear(
    int(np.sum(cards)), train_num.shape[1]
)
train_linear(
    linear_model, train_offset, train_num, train_y, weights
)
linear_valid = predict_linear(
    linear_model, valid_offset, valid_num
)
linear_test = predict_linear(
    linear_model, test_offset, test_num
)

# Family 2: categorical multinomial plus diagonal Gaussian Naive Bayes.
train_nb_num = augment_nb_numeric(train_num)
valid_nb_num = augment_nb_numeric(valid_num)
test_nb_num = augment_nb_numeric(test_num)
nb_model = fit_naive_bayes(
    train_cat, train_num, train_y, weights, cards
)
nb_valid = nb_model.predict(valid_cat, valid_nb_num)
nb_test = nb_model.predict(test_cat, test_nb_num)
del train_nb_num, valid_nb_num, test_nb_num
gc.collect()

# Family 3: random-subspace bagged decision trees using LightGBM's RF mode.
tree_train = make_tree_matrix(train_cat, train_num)
tree_valid = make_tree_matrix(valid_cat, valid_num)
tree_test = make_tree_matrix(test_cat, test_num)
feature_names = CAT_FIELDS + [
    "num_" + str(i) for i in range(train_num.shape[1])
]
categorical_indices = list(range(len(CAT_FIELDS)))

tree_dataset = lgb.Dataset(
    tree_train,
    label=train_y,
    weight=weights,
    feature_name=feature_names,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
tree_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "rf",
    "learning_rate": 1.0,
    "num_leaves": 31,
    "max_depth": 10,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.62,
    "bagging_fraction": 0.70,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "min_data_per_group": 100,
    "cat_smooth": 20.0,
    "cat_l2": 15.0,
    "verbosity": -1,
    "verbose": -1,
    "num_threads": min(8, max(1, os.cpu_count() or 1)),
    "seed": SEED,
    "bagging_seed": SEED + 1,
    "feature_fraction_seed": SEED + 2,
    "force_col_wise": True,
}
tree_model = lgb.train(
    tree_params, tree_dataset, num_boost_round=140
)
tree_valid_scores = tree_model.predict(
    tree_valid, num_iteration=tree_model.current_iteration()
).astype(np.float32)
tree_test_scores = tree_model.predict(
    tree_test, num_iteration=tree_model.current_iteration()
).astype(np.float32)
del tree_train, tree_valid, tree_test, tree_dataset
gc.collect()

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise FileNotFoundError(
        "Trusted incumbent predictions are unavailable"
    )

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)

family_valid = {
    "linear": np.asarray(linear_valid, dtype=np.float64),
    "naive_bayes": np.asarray(nb_valid, dtype=np.float64),
    "bagged_trees": np.asarray(tree_valid_scores, dtype=np.float64),
}
family_test = {
    "linear": np.asarray(linear_test, dtype=np.float64),
    "naive_bayes": np.asarray(nb_test, dtype=np.float64),
    "bagged_trees": np.asarray(tree_test_scores, dtype=np.float64),
}

# Family 4: Borda aggregation of structurally different family rankings.
new_borda_valid = np.mean(
    [
        within_user_rank(x, valid.user_id)
        for x in family_valid.values()
    ],
    axis=0,
)
new_borda_test = np.mean(
    [
        within_user_rank(x, test.user_id)
        for x in family_test.values()
    ],
    axis=0,
)
all_borda_valid = (
    3.0 * new_borda_valid
    + within_user_rank(inc_valid, valid.user_id)
) / 4.0
all_borda_test = (
    3.0 * new_borda_test
    + within_user_rank(inc_test, test.user_id)
) / 4.0

family_valid["borda_new"] = new_borda_valid
family_test["borda_new"] = new_borda_test

candidate_scores = {}
candidate_arrays = {}
candidate_raw = {}
candidate_test = {}
candidate_alpha = {}

inc_primary = metric_primary(
    valid.user_id, valid_y, inc_valid
)
candidate_scores["incumbent"] = inc_primary
candidate_arrays["incumbent"] = inc_valid
candidate_test["incumbent"] = inc_test
candidate_raw["incumbent"] = linear_valid
candidate_alpha["incumbent"] = 0.0

for name in family_valid:
    va = family_valid[name]
    te = family_test[name]
    standalone = metric_primary(valid.user_id, valid_y, va)
    candidate_scores[name] = standalone
    candidate_arrays[name] = va
    candidate_test[name] = te
    candidate_raw[name] = va
    candidate_alpha[name] = 1.0

    inc_v = centered_scaled(inc_valid, valid.user_id)
    own_v = centered_scaled(va, valid.user_id)
    inc_t = centered_scaled(inc_test, test.user_id)
    own_t = centered_scaled(te, test.user_id)

    best_score = -np.inf
    best_alpha = None
    best_valid = None
    best_test = None
    for alpha in (0.20, 0.35, 0.50, 0.65, 0.80):
        blended_valid = (
            (1.0 - alpha) * inc_v + alpha * own_v
        )
        score = metric_primary(
            valid.user_id, valid_y, blended_valid
        )
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_valid = blended_valid
            best_test = (
                (1.0 - alpha) * inc_t + alpha * own_t
            )

    blend_name = name + "_inc_blend"
    candidate_scores[blend_name] = float(best_score)
    candidate_arrays[blend_name] = best_valid
    candidate_test[blend_name] = best_test
    candidate_raw[blend_name] = va
    candidate_alpha[blend_name] = float(best_alpha)

all_borda_primary = metric_primary(
    valid.user_id, valid_y, all_borda_valid
)
candidate_scores["borda_with_incumbent"] = all_borda_primary
candidate_arrays["borda_with_incumbent"] = all_borda_valid
candidate_test["borda_with_incumbent"] = all_borda_test
candidate_raw["borda_with_incumbent"] = new_borda_valid
candidate_alpha["borda_with_incumbent"] = 0.75

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(
    candidate_arrays[winner], dtype=np.float64
)
test_scores = np.asarray(
    candidate_test[winner], dtype=np.float64
)
raw_valid_scores = np.asarray(
    candidate_raw[winner], dtype=np.float64
)

final_metrics = evaluate(valid.user_id, valid_y, valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "winner_alpha": candidate_alpha[winner],
            "linear_primary": candidate_scores["linear"],
            "naive_bayes_primary": candidate_scores["naive_bayes"],
            "bagged_trees_primary": candidate_scores["bagged_trees"],
            "borda_new_primary": candidate_scores["borda_new"],
            "incumbent_primary": inc_primary,
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        test_scores,
    )
    if (
        "_blend" in winner
        or winner == "borda_with_incumbent"
        or winner == "incumbent"
    ):
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            raw_valid_scores,
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)