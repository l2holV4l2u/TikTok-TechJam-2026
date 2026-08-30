import os
import time
import json
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
np.random.seed(SEED)

NB_FIELDS = [
    "video_id", "author_id", "tab", "tag", "duration_bucket",
    "upload_type", "music_type", "video_type", "onehot_feat1",
    "onehot_feat3", "onehot_feat7", "onehot_feat8",
]

LGB_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type", "video_type",
    "hour", "user_active_degree", "fans_user_num_range",
    "follow_user_num_range", "friend_user_num_range",
    "register_days_bucket", "onehot_feat1", "onehot_feat3",
    "onehot_feat7", "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]

BLEND_WEIGHTS = (0.05, 0.10, 0.16, 0.23, 0.31, 0.40, 0.50)


def concat_y(splits):
    return np.concatenate([
        np.asarray(s.y, dtype=np.float32) for s in splits
    ])


def concat_cat(splits, field):
    return np.concatenate([
        np.asarray(s.X[field], dtype=np.int64) for s in splits
    ])


def concat_users(splits):
    return np.concatenate([
        np.asarray(s.user_id, dtype=np.int64) for s in splits
    ])


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd < 1e-12:
        sd = 1.0
    return (x - float(np.mean(x))) / sd


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    su = users[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(starts_mask)
    counts = np.diff(np.r_[starts, n])

    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, counts)
    denominators = np.repeat(np.maximum(counts - 1, 1), counts)
    ranks = positions / denominators
    ranks[np.repeat(counts == 1, counts)] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


# ----------------------------------------------------------------------
# Family 1: categorical Naive Bayes.
# Each field supplies a smoothed log likelihood ratio P(x|y=1)/P(x|y=0).
# The average avoids excessive confidence from correlated descriptors.
# ----------------------------------------------------------------------
def fit_naive_bayes(splits):
    y = concat_y(splits).astype(np.float64)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    prior_logit = np.log((n_pos + 20.0) / (n_neg + 20.0))

    tables = {}
    for field in NB_FIELDS:
        x = concat_cat(splits, field)
        card = int(FEATURE_CARDINALITIES[field])
        total = np.bincount(x, minlength=card).astype(np.float64)
        pos = np.bincount(x, weights=y, minlength=card).astype(np.float64)
        neg = total - pos

        # Symmetric Dirichlet smoothing gives finite evidence for rare IDs.
        alpha = 3.0 if card < 100 else 8.0
        p1 = (pos + alpha) / (n_pos + alpha * card)
        p0 = (neg + alpha) / (n_neg + alpha * card)
        llr = np.log(np.maximum(p1, 1e-15)) - np.log(
            np.maximum(p0, 1e-15)
        )

        # Rare categories otherwise produce unstable likelihood ratios.
        reliability = total / (total + 12.0)
        tables[field] = (llr * reliability).astype(np.float32)

    return {"prior": prior_logit, "tables": tables}


def predict_naive_bayes(model, split):
    score = np.full(len(split.user_id), model["prior"], dtype=np.float64)
    for field, table in model["tables"].items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        score += table[ids].astype(np.float64) / len(model["tables"])
    return score


# ----------------------------------------------------------------------
# Family 2: empirical-Bayes association rules.
# Global video quality is augmented by user-specific author/tag/duration/
# upload preferences. These are explicit local rules, not low-rank factors.
# ----------------------------------------------------------------------
def smoothed_rates(ids, y, card, alpha):
    count = np.bincount(ids, minlength=card).astype(np.float64)
    pos = np.bincount(ids, weights=y, minlength=card).astype(np.float64)
    base = float(np.mean(y))
    rate = (pos + alpha * base) / (count + alpha)
    return rate.astype(np.float32)


def fit_pair_residual(splits, field, alpha):
    users = concat_users(splits)
    entities = concat_cat(splits, field)
    y = concat_y(splits).astype(np.float64)
    card = int(FEATURE_CARDINALITIES[field])

    entity_rate = smoothed_rates(entities, y, card, alpha=35.0)
    keys = users * np.int64(card) + entities
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    positives = np.bincount(
        inverse, weights=y, minlength=len(unique_keys)
    ).astype(np.float64)
    key_entities = (unique_keys % card).astype(np.int64)
    prior = entity_rate[key_entities].astype(np.float64)
    posterior = (
        positives + alpha * prior
    ) / (counts.astype(np.float64) + alpha)

    return {
        "field": field,
        "card": card,
        "keys": unique_keys.astype(np.int64),
        "residual": (posterior - prior).astype(np.float32),
    }


def lookup_pair(table, split):
    users = np.asarray(split.user_id, dtype=np.int64)
    entities = np.asarray(split.X[table["field"]], dtype=np.int64)
    query = users * np.int64(table["card"]) + entities

    positions = np.searchsorted(table["keys"], query)
    valid = positions < len(table["keys"])
    safe = np.minimum(positions, len(table["keys"]) - 1)
    valid &= table["keys"][safe] == query

    result = np.zeros(len(query), dtype=np.float64)
    result[valid] = table["residual"][safe[valid]]
    return result


def fit_association_rules(splits):
    y = concat_y(splits).astype(np.float64)
    video = concat_cat(splits, "video_id")
    video_rate = smoothed_rates(
        video, y, int(FEATURE_CARDINALITIES["video_id"]), alpha=45.0
    )
    tables = [
        fit_pair_residual(splits, "author_id", 8.0),
        fit_pair_residual(splits, "tag", 11.0),
        fit_pair_residual(splits, "duration_bucket", 13.0),
        fit_pair_residual(splits, "upload_type", 15.0),
    ]
    return {"video_rate": video_rate, "tables": tables}


def predict_association_rules(model, split):
    video = np.asarray(split.video_id, dtype=np.int64)
    score = model["video_rate"][video].astype(np.float64)
    weights = (1.00, 0.55, 0.38, 0.27)
    for weight, table in zip(weights, model["tables"]):
        score += weight * lookup_pair(table, split)
    return score


# ----------------------------------------------------------------------
# Family 3: nonlinear boosted decision trees over categorical IDs and
# transformed continuous quantities. Unlike the first two families, trees
# form conditional partitions and threshold interactions.
# ----------------------------------------------------------------------
def make_lgb_matrix(splits):
    columns = []
    for field in LGB_FIELDS:
        columns.append(
            concat_cat(splits, field).astype(np.float32, copy=False)
        )

    for field in NUM_FIELDS:
        values = np.concatenate([
            np.asarray(s.num[field], dtype=np.float32) for s in splits
        ])
        values = np.nan_to_num(values, nan=-1.0, posinf=1e8, neginf=-1.0)
        transformed = np.where(
            values >= 0.0, np.log1p(values), -1.0
        ).astype(np.float32)
        columns.append(transformed)

    return np.column_stack(columns).astype(np.float32, copy=False)


def make_lgb_split_matrix(split):
    columns = []
    for field in LGB_FIELDS:
        columns.append(
            np.asarray(split.X[field], dtype=np.float32)
        )
    for field in NUM_FIELDS:
        values = np.asarray(split.num[field], dtype=np.float32)
        values = np.nan_to_num(values, nan=-1.0, posinf=1e8, neginf=-1.0)
        columns.append(
            np.where(values >= 0.0, np.log1p(values), -1.0).astype(
                np.float32
            )
        )
    return np.column_stack(columns).astype(np.float32, copy=False)


def fit_lgb_binary(splits):
    x = make_lgb_matrix(splits)
    y = concat_y(splits)
    dataset = lgb.Dataset(
        x,
        label=y,
        categorical_feature=list(range(len(LGB_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 47,
        "max_depth": 9,
        "min_data_in_leaf": 800,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 4.0,
        "max_bin": 127,
        "cat_smooth": 25.0,
        "cat_l2": 12.0,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": 8,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=220)


def predict_lgb_binary(model, split):
    x = make_lgb_split_matrix(split)
    return model.predict(x).astype(np.float64)


def fit_family(name, splits):
    if name == "naive_bayes":
        return fit_naive_bayes(splits)
    if name == "association_rules":
        return fit_association_rules(splits)
    if name == "lightgbm_binary":
        return fit_lgb_binary(splits)
    raise ValueError(name)


def predict_family(name, model, split):
    if name == "naive_bayes":
        return predict_naive_bayes(model, split)
    if name == "association_rules":
        return predict_association_rules(model, split)
    if name == "lightgbm_binary":
        return predict_lgb_binary(model, split)
    raise ValueError(name)


train = load("train")
valid = load("valid")
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are missing")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_metrics = evaluate(valid_users, valid_y, inc_valid)
inc_z = zscore(inc_valid)
inc_rank = within_user_rank(valid_users, inc_valid)

family_names = ["naive_bayes", "association_rules", "lightgbm_binary"]
valid_models = {}
valid_raw = {}
candidate_scores = {"incumbent": float(inc_metrics["primary"])}

best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_primary = float(inc_metrics["primary"])
best_spec = ("incumbent", 0.0, "raw")
best_own_raw = None

best_standalone_name = None
best_standalone_primary = -np.inf

for name in family_names:
    model = fit_family(name, [train])
    pred = predict_family(name, model, valid)
    valid_models[name] = model
    valid_raw[name] = pred

    raw_metrics = evaluate(valid_users, valid_y, pred)
    raw_primary = float(raw_metrics["primary"])
    candidate_scores[name] = raw_primary

    if raw_primary > best_standalone_primary:
        best_standalone_primary = raw_primary
        best_standalone_name = name

    if raw_primary > best_primary:
        best_primary = raw_primary
        best_scores = pred.copy()
        best_metrics = raw_metrics
        best_spec = (name, 1.0, "raw")
        best_own_raw = pred

    correlation = float(np.corrcoef(inc_valid, pred)[0, 1])
    print(
        "FINDINGS family=%s standalone=%.6f incumbent_corr=%.6f"
        % (name, raw_primary, correlation)
    )

    pred_z = zscore(pred)
    pred_rank = within_user_rank(valid_users, pred)

    for weight in BLEND_WEIGHTS:
        z_blend = (1.0 - weight) * inc_z + weight * pred_z
        z_key = "%s_zblend_%.2f" % (name, weight)
        z_metrics = evaluate(valid_users, valid_y, z_blend)
        z_primary = float(z_metrics["primary"])
        candidate_scores[z_key] = z_primary
        if z_primary > best_primary:
            best_primary = z_primary
            best_scores = z_blend.copy()
            best_metrics = z_metrics
            best_spec = (name, float(weight), "z")
            best_own_raw = pred

        rank_blend = (1.0 - weight) * inc_rank + weight * pred_rank
        rank_key = "%s_rankblend_%.2f" % (name, weight)
        rank_metrics = evaluate(valid_users, valid_y, rank_blend)
        rank_primary = float(rank_metrics["primary"])
        candidate_scores[rank_key] = rank_primary
        if rank_primary > best_primary:
            best_primary = rank_primary
            best_scores = rank_blend.copy()
            best_metrics = rank_metrics
            best_spec = (name, float(weight), "rank")
            best_own_raw = pred

if best_own_raw is None:
    best_own_raw = valid_raw[best_standalone_name]

print(
    "FINDINGS winner_family=%s weight=%.3f fusion=%s"
    % (best_spec[0], best_spec[1], best_spec[2])
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_spec[0] == "incumbent" or best_spec[2] != "raw":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_own_raw, dtype=np.float64),
        )

# Refit the selected new family on train + validation and apply the same
# split-wise normalization/ranking fusion to test. The incumbent test scores
# are trusted predictions and no test labels are loaded or inspected.
test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)
selected_name, selected_weight, selected_fusion = best_spec

if selected_name == "incumbent":
    test_scores = inc_test
else:
    selected_model = fit_family(selected_name, [train, valid])
    test_raw = predict_family(selected_name, selected_model, test)

    if selected_fusion == "raw":
        test_scores = test_raw
    elif selected_fusion == "z":
        test_scores = (
            (1.0 - selected_weight) * zscore(inc_test)
            + selected_weight * zscore(test_raw)
        )
    elif selected_fusion == "rank":
        test_users = np.asarray(test.user_id, dtype=np.int64)
        test_scores = (
            (1.0 - selected_weight)
            * within_user_rank(test_users, inc_test)
            + selected_weight
            * within_user_rank(test_users, test_raw)
        )
    else:
        raise ValueError(selected_fusion)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.6f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)