import os
import time
import json
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
EPS = 1.0e-6

CAT_NAMES = [
    "user_id", "video_id", "author_id", "tab", "hour",
    "duration_bucket", "tag", "upload_type", "music_type", "video_type",
    "user_active_degree", "is_live_streamer", "is_video_author",
    "onehot_feat3", "onehot_feat8",
]
NUM_NAMES = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]


def group_positions(order, changes):
    n = order.size
    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = changes
    starts = np.flatnonzero(starts_mask)
    groups = np.cumsum(starts_mask) - 1
    sorted_pos = np.arange(n, dtype=np.int64) - starts[groups]
    result = np.empty(n, dtype=np.int64)
    result[order] = sorted_pos
    return result


def make_context(split):
    uid = np.asarray(split.user_id, dtype=np.int64)
    tm = np.asarray(split.time_ms, dtype=np.int64)
    date = np.asarray(split.date, dtype=np.int64)
    rows = np.arange(uid.size, dtype=np.int64)

    order = np.lexsort((rows, tm, uid))
    su = uid[order]
    st = tm[order]
    sd = date[order]

    day_changes = (su[1:] != su[:-1]) | (sd[1:] != sd[:-1])
    batch_changes = (su[1:] != su[:-1]) | (st[1:] != st[:-1])
    day_pos = group_positions(order, day_changes)
    batch_pos = group_positions(order, batch_changes)

    gap_sorted = np.empty(uid.size, dtype=np.float64)
    gap_sorted[0] = 1.0e15
    gap_sorted[1:] = st[1:].astype(np.float64) - st[:-1].astype(np.float64)
    same_user = np.empty(uid.size, dtype=bool)
    same_user[0] = False
    same_user[1:] = su[1:] == su[:-1]
    gap_sorted[~same_user] = 1.0e15
    gap = np.empty(uid.size, dtype=np.float64)
    gap[order] = gap_sorted
    gap_bin = np.digitize(
        gap,
        np.asarray(
            [0, 1000, 10000, 60000, 600000, 3600000, 86400000],
            dtype=np.float64,
        ),
        right=True,
    )

    def repeat_position(entity):
        entity = np.asarray(entity, dtype=np.int64)
        eo = np.lexsort((rows, tm, entity, date, uid))
        eu = uid[eo]
        ed = date[eo]
        ee = entity[eo]
        changes = (
            (eu[1:] != eu[:-1])
            | (ed[1:] != ed[:-1])
            | (ee[1:] != ee[:-1])
        )
        return group_positions(eo, changes)

    repeat_video = repeat_position(split.video_id)
    repeat_author = repeat_position(split.X["author_id"])
    weekday = ((date - 20220404) % 7).astype(np.int64)

    return {
        "weekday": weekday,
        "day_pos": np.minimum(day_pos, 20).astype(np.int64),
        "batch_pos": np.minimum(batch_pos, 10).astype(np.int64),
        "gap_bin": gap_bin.astype(np.int64),
        "repeat_video": np.minimum(repeat_video, 10).astype(np.int64),
        "repeat_author": np.minimum(repeat_author, 15).astype(np.int64),
    }


CONTEXT_CARDS = {
    "weekday": 7,
    "day_pos": 21,
    "batch_pos": 11,
    "gap_bin": 8,
    "repeat_video": 11,
    "repeat_author": 16,
}


def get_cat(split, name):
    if name == "user_id":
        return np.asarray(split.user_id, dtype=np.int64)
    if name == "video_id":
        return np.asarray(split.video_id, dtype=np.int64)
    return np.asarray(split.X[name], dtype=np.int64)


def build_base_matrix(split, context):
    columns = []
    for name in CAT_NAMES:
        columns.append(get_cat(split, name).astype(np.float32))

    for name in CONTEXT_CARDS:
        columns.append(context[name].astype(np.float32))

    for name in NUM_NAMES:
        x = np.asarray(split.num[name], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0))
        columns.append(x.astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False)


def history_arrays(split_name):
    video = historical_features(split_name, key="video_id")
    author = historical_features(split_name, key="author_id")
    result = []
    names = []
    for prefix, values in (("video", video), ("author", author)):
        for key in sorted(values):
            x = np.asarray(values[key], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            result.append(x)
            names.append(prefix + ":" + key)
    return result, names


def build_full_matrix(split, context, split_name):
    base = build_base_matrix(split, context)
    histories, names = history_arrays(split_name)
    if histories:
        hist = np.column_stack(histories).astype(np.float32, copy=False)
        matrix = np.column_stack((base, hist)).astype(np.float32, copy=False)
    else:
        matrix = base
    return matrix, names


def recency_weights(dates, half_life, latest=None):
    dates = np.asarray(dates, dtype=np.int64)
    if latest is None:
        latest = int(dates.max())
    age = latest - dates
    weights = np.power(2.0, -age.astype(np.float64) / float(half_life))
    weights /= np.mean(weights)
    return weights.astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    su = user_ids[order]
    ss = scores[order]

    user_start_mask = np.empty(n, dtype=bool)
    user_start_mask[0] = True
    user_start_mask[1:] = su[1:] != su[:-1]
    user_starts = np.flatnonzero(user_start_mask)
    user_groups = np.cumsum(user_start_mask) - 1
    user_sizes = np.diff(np.append(user_starts, n))
    positions = np.arange(n, dtype=np.int64) - user_starts[user_groups]

    tie_start_mask = np.empty(n, dtype=bool)
    tie_start_mask[0] = True
    tie_start_mask[1:] = (su[1:] != su[:-1]) | (ss[1:] != ss[:-1])
    tie_starts = np.flatnonzero(tie_start_mask)
    tie_sizes = np.diff(np.append(tie_starts, n))
    tie_groups = np.cumsum(tie_start_mask) - 1
    tie_first_pos = positions[tie_starts]
    tie_average = tie_first_pos + 0.5 * (tie_sizes - 1)
    average_positions = tie_average[tie_groups]

    denominators = np.maximum(user_sizes[user_groups] - 1, 1)
    ranked_sorted = average_positions / denominators
    ranked_sorted[user_sizes[user_groups] == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def lgb_params(boosting="gbdt", seed=2026):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055 if boosting == "gbdt" else 0.045,
        "num_leaves": 96,
        "min_data_in_leaf": 180,
        "max_bin": 127,
        "lambda_l1": 0.1,
        "lambda_l2": 2.0,
        "feature_fraction": 0.78,
        "num_threads": 8,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "force_col_wise": True,
    }
    if boosting == "rf":
        params.update({
            "boosting_type": "rf",
            "bagging_fraction": 0.72,
            "bagging_freq": 1,
            "feature_fraction": 0.68,
            "num_leaves": 128,
            "min_data_in_leaf": 120,
        })
    else:
        params.update({
            "boosting_type": "gbdt",
            "bagging_fraction": 0.86,
            "bagging_freq": 1,
        })
    return params


def fit_lgb(X, y, weights, boosting, rounds, seed):
    categorical = list(range(len(CAT_NAMES) + len(CONTEXT_CARDS)))
    dataset = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        categorical_feature=categorical,
        free_raw_data=False,
    )
    return lgb.train(
        lgb_params(boosting, seed),
        dataset,
        num_boost_round=rounds,
    )


def logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def fit_rate(code, cardinality, y, w, base, prior):
    count = np.bincount(
        code, weights=w, minlength=cardinality
    ).astype(np.float64)
    positive = np.bincount(
        code, weights=w * y, minlength=cardinality
    ).astype(np.float64)
    rate = (positive + prior * base) / (count + prior)
    return rate, count


def eb_codes(split, context):
    return {
        "video": get_cat(split, "video_id"),
        "author": get_cat(split, "author_id"),
        "tab": get_cat(split, "tab"),
        "tag": get_cat(split, "tag"),
        "duration": get_cat(split, "duration_bucket"),
        "upload": get_cat(split, "upload_type"),
        "hour": get_cat(split, "hour"),
        "day_pos": context["day_pos"],
        "repeat_video": context["repeat_video"],
        "repeat_author": context["repeat_author"],
    }


EB_CARDS = {
    "video": int(FEATURE_CARDINALITIES["video_id"]),
    "author": int(FEATURE_CARDINALITIES["author_id"]),
    "tab": int(FEATURE_CARDINALITIES["tab"]),
    "tag": int(FEATURE_CARDINALITIES["tag"]),
    "duration": int(FEATURE_CARDINALITIES["duration_bucket"]),
    "upload": int(FEATURE_CARDINALITIES["upload_type"]),
    "hour": int(FEATURE_CARDINALITIES["hour"]),
    "day_pos": CONTEXT_CARDS["day_pos"],
    "repeat_video": CONTEXT_CARDS["repeat_video"],
    "repeat_author": CONTEXT_CARDS["repeat_author"],
}


def fit_empirical_bayes(split, context, y, weights):
    codes = eb_codes(split, context)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    base = float(np.sum(weights * y) / np.sum(weights))

    priors = {
        "video": 80.0,
        "author": 110.0,
        "tab": 300.0,
        "tag": 220.0,
        "duration": 260.0,
        "upload": 260.0,
        "hour": 300.0,
        "day_pos": 240.0,
        "repeat_video": 220.0,
        "repeat_author": 220.0,
    }
    tables = {}
    counts = {}
    for name in codes:
        tables[name], counts[name] = fit_rate(
            codes[name], EB_CARDS[name], y, weights, base, priors[name]
        )

    crosses = [
        ("video", "tab", 180.0),
        ("author", "tab", 220.0),
        ("tag", "tab", 260.0),
        ("duration", "tab", 300.0),
        ("repeat_video", "day_pos", 260.0),
        ("repeat_author", "day_pos", 260.0),
    ]
    cross_tables = {}
    cross_counts = {}
    for left, right, prior in crosses:
        code = codes[left] * EB_CARDS[right] + codes[right]
        card = EB_CARDS[left] * EB_CARDS[right]
        key = (left, right)
        cross_tables[key], cross_counts[key] = fit_rate(
            code, card, y, weights, base, prior
        )

    return {
        "base": base,
        "tables": tables,
        "counts": counts,
        "crosses": crosses,
        "cross_tables": cross_tables,
        "cross_counts": cross_counts,
    }


def predict_empirical_bayes(model, split, context):
    codes = eb_codes(split, context)
    base_logit = logit(model["base"])
    n = split.user_id.size
    numerator = np.full(n, 0.55 * base_logit, dtype=np.float64)
    denominator = np.full(n, 0.55, dtype=np.float64)

    strengths = {
        "video": 1.45,
        "author": 1.15,
        "tab": 0.85,
        "tag": 0.65,
        "duration": 0.50,
        "upload": 0.35,
        "hour": 0.25,
        "day_pos": 0.45,
        "repeat_video": 0.55,
        "repeat_author": 0.45,
    }

    for name, strength in strengths.items():
        code = codes[name]
        rate = model["tables"][name][code]
        count = model["counts"][name][code]
        reliability = count / (count + 180.0)
        contribution = strength * reliability
        numerator += contribution * logit(rate)
        denominator += contribution

    for left, right, prior in model["crosses"]:
        code = codes[left] * EB_CARDS[right] + codes[right]
        key = (left, right)
        rate = model["cross_tables"][key][code]
        count = model["cross_counts"][key][code]
        reliability = count / (count + prior)
        contribution = 0.70 * reliability
        numerator += contribution * logit(rate)
        denominator += contribution

    score = numerator / np.maximum(denominator, EPS)

    score += (
        -0.11 * np.log1p(context["repeat_video"].astype(np.float64))
        -0.045 * np.log1p(context["repeat_author"].astype(np.float64))
        -0.012 * context["batch_pos"].astype(np.float64)
    )
    return score


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
train_context = make_context(train)
valid_context = make_context(valid)

# Select temporal weighting exclusively on the last three days of train.
base_train = build_base_matrix(train, train_context)
dates = np.asarray(train.date, dtype=np.int64)
inner_fit = dates <= 20220418
inner_holdout = dates >= 20220419

inner_scores = {}
for half_life in (2.0, 4.0, 8.0):
    weights = recency_weights(
        dates[inner_fit], half_life, latest=int(dates[inner_fit].max())
    )
    model = fit_lgb(
        base_train[inner_fit],
        y_train[inner_fit],
        weights,
        boosting="gbdt",
        rounds=105,
        seed=1000 + int(half_life),
    )
    pred = model.predict(base_train[inner_holdout])
    result = evaluate(
        np.asarray(train.user_id)[inner_holdout],
        y_train[inner_holdout],
        pred,
    )
    inner_scores[str(half_life)] = float(result["primary"])
    del model

selected_half_life = float(max(inner_scores, key=inner_scores.get))
print(
    "FINDINGS train_only_half_life_selection "
    + json.dumps({
        "scores": inner_scores,
        "selected": selected_half_life,
    }, sort_keys=True)
)

del base_train

X_train, history_names = build_full_matrix(train, train_context, "train")
X_valid, valid_history_names = build_full_matrix(valid, valid_context, "valid")
if history_names != valid_history_names:
    raise RuntimeError("Historical feature mismatch")
if X_train.shape[1] != X_valid.shape[1]:
    raise RuntimeError("Feature dimension mismatch")

full_weights = recency_weights(dates, selected_half_life)

gbdt = fit_lgb(
    X_train, y_train, full_weights,
    boosting="gbdt", rounds=260, seed=2026
)
rf = fit_lgb(
    X_train, y_train, full_weights,
    boosting="rf", rounds=150, seed=3031
)
eb = fit_empirical_bayes(
    train, train_context, y_train.astype(np.float64),
    full_weights.astype(np.float64)
)

valid_raw = {
    "recency_gbdt": gbdt.predict(X_valid),
    "recency_random_forest": rf.predict(X_valid),
    "recency_empirical_bayes": predict_empirical_bayes(
        eb, valid, valid_context
    ),
}
valid_raw["gbdt_eb_rank_ensemble"] = (
    0.70 * within_user_rank(valid.user_id, valid_raw["recency_gbdt"])
    + 0.30 * within_user_rank(
        valid.user_id, valid_raw["recency_empirical_bayes"]
    )
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.size != valid.user_id.size:
    raise RuntimeError("Incumbent validation length mismatch")
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {"trusted_incumbent": inc_valid_rank}
candidate_metrics = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, valid.y, inc_valid_rank)["primary"]
    )
}
candidate_family = {"trusted_incumbent": "recency_gbdt"}
candidate_alpha = {"trusted_incumbent": 0.0}

for family, raw in valid_raw.items():
    own_rank = within_user_rank(valid.user_id, raw)
    standalone_name = family + "_standalone"
    candidate_scores[standalone_name] = own_rank
    candidate_metrics[standalone_name] = float(
        evaluate(valid.user_id, valid.y, own_rank)["primary"]
    )
    candidate_family[standalone_name] = family
    candidate_alpha[standalone_name] = 1.0

    for alpha in (0.05, 0.10, 0.20, 0.35, 0.50, 0.70):
        score = (1.0 - alpha) * inc_valid_rank + alpha * own_rank
        name = family + "_blend_" + format(alpha, ".2f")
        candidate_scores[name] = score
        candidate_metrics[name] = float(
            evaluate(valid.user_id, valid.y, score)["primary"]
        )
        candidate_family[name] = family
        candidate_alpha[name] = alpha

winner = max(candidate_metrics, key=candidate_metrics.get)
winner_family = candidate_family[winner]
winner_alpha = candidate_alpha[winner]
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))
print(
    "FINDINGS winner "
    + json.dumps({
        "name": winner,
        "family": winner_family,
        "incumbent_weight": 1.0 - winner_alpha,
        "own_weight": winner_alpha,
        "history_features": len(history_names),
    }, sort_keys=True)
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
        np.asarray(valid_raw[winner_family], dtype=np.float64),
    )

test = load("test")
test_context = make_context(test)
X_test, test_history_names = build_full_matrix(test, test_context, "test")
if history_names != test_history_names:
    raise RuntimeError("Test historical feature mismatch")
if X_test.shape[1] != X_train.shape[1]:
    raise RuntimeError("Test feature dimension mismatch")

test_raw = {
    "recency_gbdt": gbdt.predict(X_test),
    "recency_random_forest": rf.predict(X_test),
    "recency_empirical_bayes": predict_empirical_bayes(
        eb, test, test_context
    ),
}
test_raw["gbdt_eb_rank_ensemble"] = (
    0.70 * within_user_rank(test.user_id, test_raw["recency_gbdt"])
    + 0.30 * within_user_rank(
        test.user_id, test_raw["recency_empirical_bayes"]
    )
)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if inc_test.size != test.user_id.size:
    raise RuntimeError("Incumbent test length mismatch")
inc_test_rank = within_user_rank(test.user_id, inc_test)
own_test_rank = within_user_rank(test.user_id, test_raw[winner_family])
test_scores = (
    (1.0 - winner_alpha) * inc_test_rank
    + winner_alpha * own_test_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)