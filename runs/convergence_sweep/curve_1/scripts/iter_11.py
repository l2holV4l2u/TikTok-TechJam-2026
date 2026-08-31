import os
import time
import json
import random
import gc

import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 161803
random.seed(SEED)
np.random.seed(SEED)

N_THREADS = min(8, os.cpu_count() or 8)

CAT_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "music_type",
    "upload_type",
    "video_type",
    "hour",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat16",
]

# The empirical-Bayes model intentionally emphasizes fields that can vary
# within user and remain available across the date boundary.
EB_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "music_type",
    "upload_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def date_ordinals(dates):
    dates = np.asarray(dates, dtype=np.int32)
    result = np.empty(len(dates), dtype=np.int32)
    for value in np.unique(dates):
        text = str(int(value))
        result[dates == value] = int(
            np.datetime64(
                "{}-{}-{}".format(text[:4], text[4:6], text[6:8]), "D"
            ).astype(np.int64)
        )
    return result


def recency_weights(dates, half_life=3.5):
    days = date_ordinals(dates)
    age = days.max() - days
    weights = np.exp2(-age.astype(np.float64) / float(half_life))
    weights /= weights.mean()
    return weights.astype(np.float64)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)
    if n == 0:
        return values.copy()

    # Stable row index resolves exact ties deterministically.
    order = np.lexsort((np.arange(n, dtype=np.int64), values, users))
    ordered_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = ordered_users[1:] != ordered_users[:-1]
    starts = np.flatnonzero(starts_flag)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    expanded_starts = np.repeat(starts, lengths)
    expanded_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - expanded_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = expanded_lengths > 1
    ranked[multi] = positions[multi] / (expanded_lengths[multi] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def load_histories(split_name):
    result = {}
    for entity in ("video_id", "author_id"):
        values = historical_features(split_name, key=entity)
        for name, array in values.items():
            result[entity + "__" + name] = np.asarray(array, dtype=np.float32)
    return result


def choose_history_keys(train_history):
    all_keys = sorted(train_history)
    preferred = []
    for key in all_keys:
        lower = key.lower()
        if any(token in lower for token in (
            "long", "rate", "mean", "smooth", "count", "impression"
        )):
            preferred.append(key)

    selected = preferred[:14]
    if len(selected) < min(10, len(all_keys)):
        for key in all_keys:
            if key not in selected:
                selected.append(key)
            if len(selected) >= min(14, len(all_keys)):
                break
    return selected


class NumericPreprocessor:
    def fit(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        self.medians = np.zeros(raw.shape[1], dtype=np.float32)

        for j in range(raw.shape[1]):
            finite = np.isfinite(raw[:, j])
            if finite.any():
                self.medians[j] = np.median(raw[finite, j])
            else:
                self.medians[j] = 0.0

        filled = np.where(np.isfinite(raw), raw, self.medians[None, :])
        signed_log = np.sign(filled) * np.log1p(np.abs(filled))
        self.means = signed_log.mean(axis=0).astype(np.float32)
        self.stds = signed_log.std(axis=0).astype(np.float32)
        self.stds[self.stds < 1e-5] = 1.0
        return self

    def transform(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        missing = (~np.isfinite(raw)).astype(np.float32)
        filled = np.where(np.isfinite(raw), raw, self.medians[None, :])
        signed_log = np.sign(filled) * np.log1p(np.abs(filled))
        standardized = (signed_log - self.means[None, :]) / self.stds[None, :]
        standardized = np.clip(standardized, -8.0, 8.0)
        return np.column_stack((standardized, missing)).astype(
            np.float32, copy=False
        )


def raw_numeric_matrix(split, histories, history_keys):
    columns = [
        np.asarray(split.num[name], dtype=np.float32)
        for name in NUM_FIELDS
    ]
    n = len(split.user_id)

    for key in history_keys:
        if key in histories:
            columns.append(np.asarray(histories[key], dtype=np.float32))
        else:
            columns.append(np.full(n, np.nan, dtype=np.float32))

    # Day-of-week and distance from the final training date are low-dimensional
    # temporal contexts. Trees cannot extrapolate ordinal date usefully, so the
    # cyclic calendar representation is used instead.
    ordinal = date_ordinals(split.date)
    weekday = np.mod(ordinal + 3, 7).astype(np.float32)
    columns.append(np.sin(2.0 * np.pi * weekday / 7.0).astype(np.float32))
    columns.append(np.cos(2.0 * np.pi * weekday / 7.0).astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False)


def categorical_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int32)
        for name in CAT_FIELDS
    ])


def design_matrix(categorical, numeric):
    return np.column_stack((
        categorical.astype(np.float32, copy=False),
        numeric.astype(np.float32, copy=False),
    ))


def group_sorted_indices(user_ids):
    users = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(sorted_users) == 0:
        return order, np.empty(0, dtype=np.int32)

    starts = np.r_[0, 1 + np.flatnonzero(sorted_users[1:] != sorted_users[:-1])]
    ends = np.r_[starts[1:], len(sorted_users)]
    groups = (ends - starts).astype(np.int32)
    return order, groups


def fit_xendcg(x, labels, user_ids):
    order, groups = group_sorted_indices(user_ids)
    dataset = lgb.Dataset(
        x[order],
        label=np.asarray(labels, dtype=np.float32)[order],
        group=groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )

    params = {
        "objective": "rank_xendcg",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": 0.045,
        "num_leaves": 55,
        "max_depth": 10,
        "min_data_in_leaf": 300,
        "min_sum_hessian_in_leaf": 15.0,
        "lambda_l1": 0.1,
        "lambda_l2": 3.0,
        "feature_fraction": 0.82,
        "feature_fraction_seed": SEED + 11,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "bagging_seed": SEED + 12,
        "max_bin": 127,
        "cat_smooth": 35.0,
        "cat_l2": 20.0,
        "max_cat_threshold": 64,
        "num_threads": N_THREADS,
        "seed": SEED + 13,
        "verbose": -1,
    }
    model = lgb.train(params, dataset, num_boost_round=210)
    print(
        "FINDINGS family=xendcg trees={}".format(model.current_iteration()),
        flush=True,
    )
    return model


def fit_random_forest(x, labels, weights):
    dataset = lgb.Dataset(
        x,
        label=np.asarray(labels, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )

    params = {
        "boosting_type": "rf",
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 1.0,
        "num_leaves": 63,
        "max_depth": 11,
        "min_data_in_leaf": 250,
        "min_sum_hessian_in_leaf": 12.0,
        "lambda_l1": 0.05,
        "lambda_l2": 3.0,
        "feature_fraction": 0.68,
        "feature_fraction_seed": SEED + 21,
        "bagging_fraction": 0.70,
        "bagging_freq": 1,
        "bagging_seed": SEED + 22,
        "max_bin": 127,
        "cat_smooth": 40.0,
        "cat_l2": 20.0,
        "max_cat_threshold": 64,
        "num_threads": N_THREADS,
        "seed": SEED + 23,
        "verbose": -1,
    }
    model = lgb.train(params, dataset, num_boost_round=180)
    print(
        "FINDINGS family=random_forest trees={}".format(
            model.current_iteration()
        ),
        flush=True,
    )
    return model


class AdditiveEmpiricalBayes:
    def __init__(self, fields, prior_strength=45.0):
        self.fields = list(fields)
        self.prior_strength = float(prior_strength)

    def fit(self, split, labels, weights):
        y = np.asarray(labels, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        self.global_rate = float(np.sum(w * y) / np.sum(w))
        self.tables = {}
        self.field_scales = {}

        global_logit = np.log(
            np.clip(self.global_rate, 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - self.global_rate, 1e-6, 1.0)
        )

        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            size = int(ids.max()) + 1
            counts = np.bincount(ids, weights=w, minlength=size)
            positives = np.bincount(ids, weights=w * y, minlength=size)

            smoothed = (
                positives + self.prior_strength * self.global_rate
            ) / (counts + self.prior_strength)
            smoothed = np.clip(smoothed, 1e-5, 1.0 - 1e-5)
            effects = np.log(smoothed / (1.0 - smoothed)) - global_logit

            # Reliability scaling prevents large-cardinality identity fields
            # from dominating solely through noisier extreme estimates.
            reliability = counts / (counts + self.prior_strength)
            weighted_variance = np.sum(
                counts * np.square(effects)
            ) / max(np.sum(counts), 1.0)
            scale = 1.0 / max(np.sqrt(weighted_variance), 0.08)

            self.tables[field] = effects.astype(np.float32)
            self.field_scales[field] = float(min(scale, 5.0))

        return self

    def predict(self, split):
        result = np.zeros(len(split.user_id), dtype=np.float64)
        total_scale = 0.0

        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            table = self.tables[field]
            values = np.zeros(len(ids), dtype=np.float64)
            known = ids < len(table)
            values[known] = table[ids[known]]

            scale = self.field_scales[field]
            # Video and author estimates receive modest extra weight because
            # they vary for most validation users and have strongest signal.
            if field == "video_id":
                scale *= 1.35
            elif field == "author_id":
                scale *= 1.20
            elif field == "tab":
                scale *= 1.10

            result += scale * values
            total_scale += scale

        return result / max(total_scale, 1e-12)


def primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)

train_history = load_histories("train")
valid_history = load_histories("valid")
history_keys = choose_history_keys(train_history)
print(
    "FINDINGS history_features={}".format(json.dumps(history_keys)),
    flush=True,
)

train_cat = categorical_matrix(train)
valid_cat = categorical_matrix(valid)

train_raw_num = raw_numeric_matrix(train, train_history, history_keys)
valid_raw_num = raw_numeric_matrix(valid, valid_history, history_keys)

preprocessor = NumericPreprocessor().fit(train_raw_num)
train_num = preprocessor.transform(train_raw_num)
valid_num = preprocessor.transform(valid_raw_num)

del train_raw_num, valid_raw_num, train_history, valid_history
gc.collect()

train_design = design_matrix(train_cat, train_num)
valid_design = design_matrix(valid_cat, valid_num)

weights = recency_weights(train.date, half_life=3.5)

# Family 1: listwise stochastic cross-entropy NDCG boosting.
xendcg_model = fit_xendcg(train_design, train_y, train.user_id)
xendcg_valid_raw = xendcg_model.predict(train_design[:1])  # initialize library
del xendcg_valid_raw
xendcg_valid = xendcg_model.predict(valid_design).astype(np.float64)

# Family 2: independently bagged trees rather than sequential boosting.
rf_model = fit_random_forest(train_design, train_y, weights)
rf_valid = rf_model.predict(valid_design).astype(np.float64)

# Family 3: closed-form recency-weighted empirical Bayes.
eb_model = AdditiveEmpiricalBayes(
    EB_FIELDS, prior_strength=45.0
).fit(train, train_y, weights)
eb_valid = eb_model.predict(valid)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid_raw = np.load(inc_valid_path).astype(np.float64)
if len(inc_valid_raw) != len(valid.user_id):
    raise ValueError("Incumbent validation prediction length mismatch")

valid_ranks = {
    "incumbent": within_user_rank(valid.user_id, inc_valid_raw),
    "xendcg": within_user_rank(valid.user_id, xendcg_valid),
    "random_forest": within_user_rank(valid.user_id, rf_valid),
    "empirical_bayes": within_user_rank(valid.user_id, eb_valid),
}

candidate_scores = {}
candidate_arrays = {}
candidate_recipes = {}

for name, score in valid_ranks.items():
    candidate_scores[name] = primary(valid.user_id, valid.y, score)
    candidate_arrays[name] = score
    candidate_recipes[name] = ("single", name, 1.0)

# Every new family is blended with the trusted incumbent.
for family in ("xendcg", "random_forest", "empirical_bayes"):
    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        name = "{}_inc_borda_{:.2f}".format(family, alpha)
        score = (
            (1.0 - alpha) * valid_ranks["incumbent"]
            + alpha * valid_ranks[family]
        )
        candidate_arrays[name] = score
        candidate_scores[name] = primary(valid.user_id, valid.y, score)
        candidate_recipes[name] = ("inc_blend", family, alpha)

# A consensus of the two learned tree mechanisms can reduce their distinct
# boosting/bagging variance before incumbent aggregation.
tree_consensus = 0.5 * (
    valid_ranks["xendcg"] + valid_ranks["random_forest"]
)
candidate_arrays["tree_consensus"] = tree_consensus
candidate_scores["tree_consensus"] = primary(
    valid.user_id, valid.y, tree_consensus
)
candidate_recipes["tree_consensus"] = ("tree_consensus", "", 1.0)

for alpha in (0.10, 0.20, 0.30, 0.40):
    name = "tree_consensus_inc_borda_{:.2f}".format(alpha)
    score = (
        (1.0 - alpha) * valid_ranks["incumbent"]
        + alpha * tree_consensus
    )
    candidate_arrays[name] = score
    candidate_scores[name] = primary(valid.user_id, valid.y, score)
    candidate_recipes[name] = ("tree_inc_blend", "", alpha)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_arrays[winner]
winner_recipe = candidate_recipes[winner]

print(
    "CANDIDATES " + json.dumps(
        {k: round(float(v), 7) for k, v in candidate_scores.items()},
        sort_keys=True,
    ),
    flush=True,
)
print(
    "FINDINGS selected={} recipe={} validation_primary={:.7f}".format(
        winner, winner_recipe, candidate_scores[winner]
    ),
    flush=True,
)

metrics = evaluate(valid.user_id, valid.y, valid_scores)

# Produce test scores from the same train-fitted models and fixed validation
# recipe. Test labels are never accessed.
test = load("test")
test_history = load_histories("test")
test_cat = categorical_matrix(test)
test_raw_num = raw_numeric_matrix(test, test_history, history_keys)
test_num = preprocessor.transform(test_raw_num)
test_design = design_matrix(test_cat, test_num)

xendcg_test = xendcg_model.predict(test_design).astype(np.float64)
rf_test = rf_model.predict(test_design).astype(np.float64)
eb_test = eb_model.predict(test)
inc_test_raw = np.load(inc_test_path).astype(np.float64)

if len(inc_test_raw) != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

test_ranks = {
    "incumbent": within_user_rank(test.user_id, inc_test_raw),
    "xendcg": within_user_rank(test.user_id, xendcg_test),
    "random_forest": within_user_rank(test.user_id, rf_test),
    "empirical_bayes": within_user_rank(test.user_id, eb_test),
}
test_tree_consensus = 0.5 * (
    test_ranks["xendcg"] + test_ranks["random_forest"]
)

kind, family, alpha = winner_recipe
if kind == "single":
    test_scores = test_ranks[family]
    raw_valid_scores = valid_ranks.get(family, valid_ranks["xendcg"])
elif kind == "inc_blend":
    test_scores = (
        (1.0 - alpha) * test_ranks["incumbent"]
        + alpha * test_ranks[family]
    )
    raw_valid_scores = valid_ranks[family]
elif kind == "tree_consensus":
    test_scores = test_tree_consensus
    raw_valid_scores = tree_consensus
elif kind == "tree_inc_blend":
    test_scores = (
        (1.0 - alpha) * test_ranks["incumbent"]
        + alpha * test_tree_consensus
    )
    raw_valid_scores = tree_consensus
else:
    raise RuntimeError("Unknown winner recipe")

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

    # Always preserve the strongest relevant newly fitted component when the
    # reported result includes trusted incumbent predictions.
    if kind in ("inc_blend", "tree_inc_blend") or (
        kind == "single" and family == "incumbent"
    ):
        if kind == "single" and family == "incumbent":
            best_new = max(
                ("xendcg", "random_forest", "empirical_bayes"),
                key=lambda name: candidate_scores[name],
            )
            raw_valid_scores = valid_ranks[best_new]
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid_scores, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {{"primary": {:.10f}, "gauc": {:.10f}, '
    '"ndcg@5": {:.10f}, "gpu_seconds": {:.6f}}}'.format(
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)