import os
import time
import json
import gc
import random

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)

FIELDS = [
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

# Fields used for personalized user-content affinities. Very rare identity
# tokens are retained, while essentially constant fields are excluded.
ROCC_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
]

HASH_DIM = 1 << 19
HASH_MASK = HASH_DIM - 1


def date_ordinals(dates):
    dates = np.asarray(dates, dtype=np.int32)
    result = np.empty(len(dates), dtype=np.int32)
    for value in np.unique(dates):
        text = str(int(value))
        day = np.datetime64(
            "{}-{}-{}".format(text[:4], text[4:6], text[6:8]), "D"
        )
        result[dates == value] = int(day.astype(np.int64))
    return result


def recency_weights(dates, half_life=4.5):
    days = date_ordinals(dates)
    age = days.max() - days
    result = np.exp2(-age.astype(np.float64) / float(half_life))
    result /= result.mean()
    return result.astype(np.float32)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)
    if n == 0:
        return values.copy()

    order = np.lexsort((np.arange(n, dtype=np.int64), values, users))
    sorted_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_flag)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    expanded_starts = np.repeat(starts, lengths)
    expanded_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - expanded_starts

    ranks = np.full(n, 0.5, dtype=np.float64)
    multi = expanded_lengths > 1
    ranks[multi] = positions[multi] / (expanded_lengths[multi] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def metric_primary(split, scores):
    return float(evaluate(split.user_id, split.y, scores)["primary"])


class GenerativeCategoricalRanker:
    """
    Shrunk categorical log-likelihood ratios.

    Unlike discriminative CTR models, this estimates how much more frequently
    each token occurs among positive than negative impressions. Recency weights
    make the likelihoods represent the end of the training window.
    """

    def __init__(self, fields, prior=35.0):
        self.fields = list(fields)
        self.prior = float(prior)
        self.tables = {}

    def fit(self, split, labels, weights):
        y = np.asarray(labels, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        pos_mass = float(np.sum(w * y))
        neg_mass = float(np.sum(w * (1.0 - y)))
        global_rate = pos_mass / max(pos_mass + neg_mass, 1.0)

        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            size = int(FEATURE_CARDINALITIES[field])

            pos = np.bincount(ids, weights=w * y, minlength=size)
            neg = np.bincount(ids, weights=w * (1.0 - y), minlength=size)
            total = pos + neg

            # Beta shrinkage produces a stable token posterior. Converting the
            # posterior to an odds ratio gives an additive generative score.
            rate = (
                pos + self.prior * global_rate
            ) / (total + self.prior)
            rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
            effect = np.log(rate / (1.0 - rate))
            effect -= np.log(global_rate / (1.0 - global_rate))

            reliability = np.sqrt(total / (total + self.prior))
            effect *= reliability

            # Normalize fields so high-cardinality identities do not dominate
            # only because they have noisier extreme estimates.
            variance = np.sum(total * effect * effect) / max(total.sum(), 1.0)
            effect /= max(np.sqrt(variance), 0.15)
            self.tables[field] = effect.astype(np.float32)

        return self

    def predict(self, split):
        result = np.zeros(len(split.user_id), dtype=np.float64)
        total_weight = 0.0

        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            table = self.tables[field]
            field_weight = 1.0
            if field == "video_id":
                field_weight = 1.30
            elif field == "author_id":
                field_weight = 1.20
            elif field == "tab":
                field_weight = 1.10

            known = ids < len(table)
            values = np.zeros(len(ids), dtype=np.float64)
            values[known] = table[ids[known]]
            result += field_weight * values
            total_weight += field_weight

        return result / max(total_weight, 1e-12)


def rocchio_scores(train, valid, test, labels, weights, prior=12.0):
    """
    For every user and content token, aggregate the user's recency-weighted
    residual preference relative to their own long-view rate. This is a sparse
    content-profile/Rocchio predictor rather than a globally fitted CTR model.
    """
    train_users = np.asarray(train.user_id, dtype=np.int64)
    valid_users = np.asarray(valid.user_id, dtype=np.int64)
    test_users = np.asarray(test.user_id, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    user_mass = np.bincount(train_users, weights=w, minlength=n_users)
    user_pos = np.bincount(train_users, weights=w * y, minlength=n_users)
    global_rate = float(np.sum(w * y) / np.sum(w))
    user_rate = (user_pos + 25.0 * global_rate) / (user_mass + 25.0)

    residual = y - user_rate[train_users]
    valid_result = np.zeros(len(valid_users), dtype=np.float64)
    test_result = np.zeros(len(test_users), dtype=np.float64)

    for field in ROCC_FIELDS:
        n_tokens = int(FEATURE_CARDINALITIES[field])
        tr_token = np.asarray(train.X[field], dtype=np.int64)
        va_token = np.asarray(valid.X[field], dtype=np.int64)
        te_token = np.asarray(test.X[field], dtype=np.int64)

        counts = sparse.coo_matrix(
            (w, (train_users, tr_token)),
            shape=(n_users, n_tokens),
            dtype=np.float32,
        ).tocsr()
        preferences = sparse.coo_matrix(
            ((w * residual).astype(np.float32), (train_users, tr_token)),
            shape=(n_users, n_tokens),
            dtype=np.float32,
        ).tocsr()

        va_num = preferences[valid_users, va_token].A1.astype(np.float64)
        va_den = counts[valid_users, va_token].A1.astype(np.float64)
        te_num = preferences[test_users, te_token].A1.astype(np.float64)
        te_den = counts[test_users, te_token].A1.astype(np.float64)

        field_weight = 1.0
        if field == "video_id":
            field_weight = 1.25
        elif field == "author_id":
            field_weight = 1.15

        valid_result += field_weight * va_num / (va_den + prior)
        test_result += field_weight * te_num / (te_den + prior)

        del counts, preferences
        gc.collect()

    return valid_result, test_result


def hashed_design(split, row_indices):
    """
    Hash global content tokens and user-by-content crosses. The latter let a
    linear ranker express personalized preferences without a dense user-item
    parameter matrix.
    """
    idx = np.asarray(row_indices, dtype=np.int64)
    n = len(idx)
    user = np.asarray(split.user_id, dtype=np.uint64)[idx]

    n_features_per_row = 2 * len(FIELDS)
    rows = np.repeat(np.arange(n, dtype=np.int32), n_features_per_row)
    cols = np.empty(n * n_features_per_row, dtype=np.int32)
    data = np.ones(n * n_features_per_row, dtype=np.float32)

    primes = [
        2654435761, 2246822519, 3266489917, 668265263,
        374761393, 1274126177, 1431374977, 42595009,
        1367130551, 1103515245,
    ]

    position = 0
    for j, field in enumerate(FIELDS):
        token = np.asarray(split.X[field], dtype=np.uint64)[idx]
        prime = np.uint64(primes[j])

        global_hash = (
            token * prime + np.uint64((j + 1) * 104729)
        ) & np.uint64(HASH_MASK)
        cross_hash = (
            token * prime
            + user * np.uint64(1000003 + 2 * j)
            + np.uint64((j + 17) * 15485863)
        ) & np.uint64(HASH_MASK)

        cols[position::n_features_per_row] = global_hash.astype(np.int32)
        cols[position + 1::n_features_per_row] = cross_hash.astype(np.int32)
        position += 2

    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n, HASH_DIM),
        dtype=np.float32,
    )
    return matrix


def make_rank_pairs(split, labels, max_pairs=360000):
    users = np.asarray(split.user_id, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)
    order = np.lexsort((
        np.arange(len(users), dtype=np.int64),
        np.asarray(split.time_ms, dtype=np.int64),
        users,
    ))

    pair_a = []
    pair_b = []
    for shift in (1, 3, 11, 37):
        left = order[:-shift]
        right = order[shift:]
        keep = (users[left] == users[right]) & (y[left] != y[right])
        left = left[keep]
        right = right[keep]

        positive = np.where(y[left] == 1, left, right)
        negative = np.where(y[left] == 0, left, right)
        pair_a.append(positive)
        pair_b.append(negative)

    positive = np.concatenate(pair_a)
    negative = np.concatenate(pair_b)

    rng = np.random.default_rng(SEED + 91)
    if len(positive) > max_pairs:
        chosen = rng.choice(len(positive), size=max_pairs, replace=False)
        positive = positive[chosen]
        negative = negative[chosen]

    return positive.astype(np.int64), negative.astype(np.int64)


def fit_rank_svm(train, labels, weights):
    positive, negative = make_rank_pairs(train, labels)
    all_indices = np.concatenate((positive, negative))
    all_matrix = hashed_design(train, all_indices)

    n_pairs = len(positive)
    x_pos = all_matrix[:n_pairs]
    x_neg = all_matrix[n_pairs:]
    differences = x_pos - x_neg
    del all_matrix, x_pos, x_neg
    gc.collect()

    rng = np.random.default_rng(SEED + 92)
    signs = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32), size=n_pairs
    )
    pair_labels = (signs > 0).astype(np.int8)
    differences = differences.multiply(signs[:, None]).tocsr()

    pair_weights = np.sqrt(
        np.asarray(weights, dtype=np.float64)[positive]
        * np.asarray(weights, dtype=np.float64)[negative]
    )

    model = SGDClassifier(
        loss="hinge",
        penalty="elasticnet",
        alpha=2.5e-6,
        l1_ratio=0.03,
        fit_intercept=False,
        max_iter=9,
        tol=5e-4,
        shuffle=True,
        random_state=SEED + 93,
        average=True,
        n_jobs=1,
    )
    model.fit(differences, pair_labels, sample_weight=pair_weights)

    print(
        "FINDINGS ranksvm_pairs={} hashed_dimensions={} iterations={}".format(
            n_pairs, HASH_DIM, int(model.n_iter_)
        ),
        flush=True,
    )
    del differences
    gc.collect()
    return model


def predict_rank_svm(model, split, batch_size=160000):
    n = len(split.user_id)
    result = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        matrix = hashed_design(split, np.arange(start, end, dtype=np.int64))
        result[start:end] = model.decision_function(matrix)
        del matrix
    return result


train = load("train")
valid = load("valid")
test = load("test")

train_y = np.asarray(train.y, dtype=np.float32)
weights = recency_weights(train.date, half_life=4.5)

print(
    "FINDINGS recency_weight_ratio_oldest_to_newest={:.4f}".format(
        float(weights[np.argmin(train.date)] / weights[np.argmax(train.date)])
    ),
    flush=True,
)

# Family 1: generative categorical likelihood-ratio model.
generative = GenerativeCategoricalRanker(
    FIELDS, prior=35.0
).fit(train, train_y, weights)
gen_valid = generative.predict(valid)
gen_test = generative.predict(test)

# Family 2: sparse personalized Rocchio/content-affinity model.
roc_valid, roc_test = rocchio_scores(
    train, valid, test, train_y, weights, prior=12.0
)

# Family 3: hashed personalized large-margin RankSVM.
rank_svm = fit_rank_svm(train, train_y, weights)
svm_valid = predict_rank_svm(rank_svm, valid)
svm_test = predict_rank_svm(rank_svm, test)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")

valid_raw = {
    "generative": gen_valid,
    "rocchio": roc_valid,
    "ranksvm": svm_valid,
}
test_raw = {
    "generative": gen_test,
    "rocchio": roc_test,
    "ranksvm": svm_test,
}

valid_rank = {
    name: within_user_rank(valid.user_id, score)
    for name, score in valid_raw.items()
}
test_rank = {
    name: within_user_rank(test.user_id, score)
    for name, score in test_raw.items()
}
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_scores = {}
candidate_arrays = {}
candidate_test_arrays = {}
candidate_own_raw = {}

# Record all standalone families.
candidate_scores["incumbent"] = metric_primary(valid, inc_valid)
candidate_arrays["incumbent"] = inc_valid
candidate_test_arrays["incumbent"] = inc_test
candidate_own_raw["incumbent"] = None

for name in valid_raw:
    score = metric_primary(valid, valid_raw[name])
    candidate_scores[name] = score
    candidate_arrays[name] = valid_raw[name]
    candidate_test_arrays[name] = test_raw[name]
    candidate_own_raw[name] = valid_raw[name]

# Blend every structurally different family with the incumbent.
for name in valid_raw:
    for alpha in (0.10, 0.20, 0.35, 0.50):
        key = "{}_blend_{:.2f}".format(name, alpha)
        va = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * valid_rank[name]
        )
        te = (
            (1.0 - alpha) * inc_test_rank
            + alpha * test_rank[name]
        )
        candidate_scores[key] = metric_primary(valid, va)
        candidate_arrays[key] = va
        candidate_test_arrays[key] = te
        candidate_own_raw[key] = valid_raw[name]

# Also test whether the three mutually distinct mechanisms are more stable as
# an ensemble before blending with the incumbent.
ensemble_valid = (
    valid_rank["generative"]
    + valid_rank["rocchio"]
    + valid_rank["ranksvm"]
) / 3.0
ensemble_test = (
    test_rank["generative"]
    + test_rank["rocchio"]
    + test_rank["ranksvm"]
) / 3.0

candidate_scores["new_family_ensemble"] = metric_primary(
    valid, ensemble_valid
)
candidate_arrays["new_family_ensemble"] = ensemble_valid
candidate_test_arrays["new_family_ensemble"] = ensemble_test
candidate_own_raw["new_family_ensemble"] = ensemble_valid

for alpha in (0.10, 0.20, 0.35, 0.50):
    key = "ensemble_blend_{:.2f}".format(alpha)
    va = (1.0 - alpha) * inc_valid_rank + alpha * ensemble_valid
    te = (1.0 - alpha) * inc_test_rank + alpha * ensemble_test
    candidate_scores[key] = metric_primary(valid, va)
    candidate_arrays[key] = va
    candidate_test_arrays[key] = te
    candidate_own_raw[key] = ensemble_valid

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(candidate_arrays[winner], dtype=np.float64)
test_scores = np.asarray(candidate_test_arrays[winner], dtype=np.float64)

print(
    "CANDIDATES " + json.dumps(
        {k: float(v) for k, v in sorted(candidate_scores.items())},
        sort_keys=True,
    ),
    flush=True,
)
print(
    "FINDINGS selected_candidate={} selected_primary={:.8f}".format(
        winner, candidate_scores[winner]
    ),
    flush=True,
)

metrics = evaluate(valid.user_id, valid.y, valid_scores)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        test_scores.astype(np.float64),
    )
    own = candidate_own_raw[winner]
    if own is not None and (
        "blend" in winner or winner == "new_family_ensemble"
    ):
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {{"primary": {:.10f}, "gauc": {:.10f}, '
    '"ndcg@5": {:.10f}, "gpu_seconds": {:.4f}}}'.format(
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)