import os
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 9473
LATENT_DIM = 32
SMOOTHING = 80.0

np.random.seed(SEED)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    sorted_users = user_ids[order]
    starts = np.r_[
        0,
        np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
    ]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    mask = repeated_counts > 1
    ranked[mask] = positions[mask] / (repeated_counts[mask] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def build_matrix(user_ids, video_ids, labels, mode):
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    labels = np.asarray(labels, dtype=np.float32)
    if mode == "positive":
        mask = labels > 0.5
        rows = np.asarray(user_ids[mask], dtype=np.int32)
        cols = np.asarray(video_ids[mask], dtype=np.int32)
        values = np.ones(mask.sum(), dtype=np.float32)
    elif mode == "signed":
        rows = np.asarray(user_ids, dtype=np.int32)
        cols = np.asarray(video_ids, dtype=np.int32)
        # A negative observed impression is weak negative evidence rather than
        # equivalent in magnitude to a positive long view.
        values = np.where(labels > 0.5, 1.0, -0.30).astype(np.float32)
    else:
        raise ValueError(mode)

    matrix = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    if mode == "positive":
        # Reduce domination by extremely active users and globally popular videos.
        row_norm = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
        col_freq = np.asarray((matrix != 0).sum(axis=0)).ravel().astype(np.float32)
        row_scale = 1.0 / np.maximum(row_norm, 1.0)
        idf = np.log1p(float(n_users) / np.maximum(col_freq, 1.0)).astype(
            np.float32
        )
        matrix = sparse.diags(row_scale).dot(matrix).dot(sparse.diags(idf))
        matrix = matrix.tocsr().astype(np.float32)

    return matrix


def fit_svd(user_ids, video_ids, labels, mode):
    matrix = build_matrix(user_ids, video_ids, labels, mode)
    k = min(LATENT_DIM, min(matrix.shape) - 1)

    u, singular, vt = svds(
        matrix,
        k=k,
        which="LM",
        tol=3e-3,
        maxiter=300,
        random_state=SEED,
        return_singular_vectors=True,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    user_factors = u * singular[None, :]
    item_factors = vt.T.copy()
    return user_factors, item_factors


def svd_predict(model, user_ids, video_ids):
    user_factors, item_factors = model
    u = user_factors[np.asarray(user_ids, dtype=np.int64)]
    v = item_factors[np.asarray(video_ids, dtype=np.int64)]
    return np.einsum("ij,ij->i", u, v, optimize=True).astype(np.float32)


def temporal_features(user_ids, time_ms, hour, dates):
    user_ids = np.asarray(user_ids)
    time_ms = np.asarray(time_ms, dtype=np.int64)
    hour = np.asarray(hour, dtype=np.int64)
    dates = np.asarray(dates, dtype=np.int64)
    n = len(user_ids)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, time_ms, user_ids))
    su = user_ids[order]
    st = time_ms[order]

    new_user = np.r_[True, su[1:] != su[:-1]]
    user_starts = np.flatnonzero(new_user)
    user_ends = np.r_[user_starts[1:], n]
    user_counts = user_ends - user_starts
    repeated_user_starts = np.repeat(user_starts, user_counts)
    repeated_user_counts = np.repeat(user_counts, user_counts)
    user_pos_sorted = np.arange(n, dtype=np.int64) - repeated_user_starts

    new_batch = np.r_[
        True,
        (su[1:] != su[:-1]) | (st[1:] != st[:-1]),
    ]
    batch_starts = np.flatnonzero(new_batch)
    batch_ends = np.r_[batch_starts[1:], n]
    batch_counts = batch_ends - batch_starts
    repeated_batch_starts = np.repeat(batch_starts, batch_counts)
    repeated_batch_counts = np.repeat(batch_counts, batch_counts)
    batch_pos_sorted = np.arange(n, dtype=np.int64) - repeated_batch_starts

    gap_seconds_sorted = np.zeros(n, dtype=np.float64)
    if n > 1:
        same_user = su[1:] == su[:-1]
        gaps = np.maximum(st[1:] - st[:-1], 0) / 1000.0
        gap_seconds_sorted[1:] = np.where(same_user, gaps, 0.0)

    new_session = new_user | (gap_seconds_sorted > 1800.0)
    session_starts = np.flatnonzero(new_session)
    session_ends = np.r_[session_starts[1:], n]
    session_counts = session_ends - session_starts
    repeated_session_starts = np.repeat(session_starts, session_counts)
    session_pos_sorted = np.arange(n, dtype=np.int64) - repeated_session_starts

    user_decile_sorted = np.minimum(
        9,
        (
            10.0
            * user_pos_sorted
            / np.maximum(repeated_user_counts, 1)
        ).astype(np.int64),
    )

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)

    user_pos = user_pos_sorted[inverse]
    user_count = repeated_user_counts[inverse]
    user_decile = user_decile_sorted[inverse]
    batch_pos = batch_pos_sorted[inverse]
    batch_size = repeated_batch_counts[inverse]
    session_pos = session_pos_sorted[inverse]
    gap_seconds = gap_seconds_sorted[inverse]

    gap_bucket = np.minimum(
        15,
        np.floor(np.log2(1.0 + gap_seconds)).astype(np.int64),
    )
    weekday = dates % 100
    weekday = np.asarray(
        [
            # Dates cover only one month here; this deterministic transform is
            # merely a stable seven-day phase feature.
            int(x) % 7 for x in weekday
        ],
        dtype=np.int64,
    )

    features = [
        np.minimum(user_pos, 31).astype(np.int64),
        np.minimum(user_count, 31).astype(np.int64),
        user_decile.astype(np.int64),
        np.minimum(batch_pos, 15).astype(np.int64),
        np.minimum(batch_size, 16).astype(np.int64),
        np.minimum(session_pos, 31).astype(np.int64),
        gap_bucket,
        np.minimum(hour, 31).astype(np.int64),
        weekday,
        (
            np.minimum(batch_pos, 7) * 17
            + np.minimum(batch_size, 16)
        ).astype(np.int64),
        (
            np.minimum(session_pos, 15) * 10
            + user_decile
        ).astype(np.int64),
        (
            np.minimum(hour, 31) * 10
            + user_decile
        ).astype(np.int64),
    ]
    return features


class AdditiveRateModel:
    def __init__(self, smoothing=SMOOTHING):
        self.smoothing = float(smoothing)
        self.global_rate = None
        self.tables = []

    def fit(self, features, labels):
        labels = np.asarray(labels, dtype=np.float64)
        self.global_rate = float(labels.mean())
        global_logit = np.log(
            np.clip(self.global_rate, 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - self.global_rate, 1e-6, 1.0)
        )

        self.tables = []
        for feature in features:
            feature = np.asarray(feature, dtype=np.int64)
            size = int(feature.max()) + 1
            counts = np.bincount(feature, minlength=size).astype(np.float64)
            positives = np.bincount(
                feature, weights=labels, minlength=size
            ).astype(np.float64)
            rates = (
                positives + self.smoothing * self.global_rate
            ) / (counts + self.smoothing)
            logits = np.log(
                np.clip(rates, 1e-6, 1.0 - 1e-6)
                / np.clip(1.0 - rates, 1e-6, 1.0)
            )
            self.tables.append(logits - global_logit)
        return self

    def predict(self, features):
        global_logit = np.log(
            np.clip(self.global_rate, 1e-6, 1.0 - 1e-6)
            / np.clip(1.0 - self.global_rate, 1e-6, 1.0)
        )
        score = np.full(len(features[0]), global_logit, dtype=np.float64)

        # Averaging deviations is deliberately conservative under temporal drift.
        scale = 1.0 / np.sqrt(max(len(features), 1))
        for feature, table in zip(features, self.tables):
            feature = np.asarray(feature, dtype=np.int64)
            known = feature < len(table)
            contribution = np.zeros(len(feature), dtype=np.float64)
            contribution[known] = table[feature[known]]
            score += scale * contribution
        return score.astype(np.float32)


def concatenate_features(a, b):
    return [
        np.concatenate([x, y]).astype(np.int64, copy=False)
        for x, y in zip(a, b)
    ]


train = load("train")
valid = load("valid")

train_uid = np.asarray(train.X["user_id"], dtype=np.int64)
train_vid = np.asarray(train.X["video_id"], dtype=np.int64)
valid_uid = np.asarray(valid.X["user_id"], dtype=np.int64)
valid_vid = np.asarray(valid.X["video_id"], dtype=np.int64)
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_eval_users = np.asarray(valid.user_id)

train_temporal = temporal_features(
    train.user_id,
    train.time_ms,
    train.X["hour"],
    train.date,
)
valid_temporal = temporal_features(
    valid.user_id,
    valid.time_ms,
    valid.X["hour"],
    valid.date,
)

raw_valid = {}

positive_svd = fit_svd(train_uid, train_vid, train_y, "positive")
raw_valid["positive_svd"] = svd_predict(
    positive_svd, valid_uid, valid_vid
)
del positive_svd

signed_svd = fit_svd(train_uid, train_vid, train_y, "signed")
raw_valid["signed_svd"] = svd_predict(
    signed_svd, valid_uid, valid_vid
)
del signed_svd

temporal_model = AdditiveRateModel().fit(train_temporal, train_y)
raw_valid["temporal_additive"] = temporal_model.predict(valid_temporal)

# A parameter-free fatigue/exposure family tests the direction independently
# of target-rate estimation.
raw_valid["chronology_fatigue"] = (
    -valid_temporal[0].astype(np.float32)
    - 0.35 * valid_temporal[5].astype(np.float32)
    - 0.15 * valid_temporal[3].astype(np.float32)
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path)
inc_valid_rank = within_user_rank(valid_eval_users, inc_valid)

alphas = np.asarray(
    [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0],
    dtype=np.float64,
)

candidate_log = {}
best_primary = -np.inf
best_family = None
best_alpha = None
best_valid_scores = None
best_raw_valid = None
best_metrics = None

for name, scores in raw_valid.items():
    standalone = evaluate(valid_eval_users, valid_y, scores)
    candidate_log[name + "_standalone"] = float(standalone["primary"])

    model_rank = within_user_rank(valid_eval_users, scores)
    family_best = -np.inf
    family_best_alpha = None

    for alpha in alphas:
        blended = (
            (1.0 - float(alpha)) * inc_valid_rank
            + float(alpha) * model_rank
        )
        metrics = evaluate(valid_eval_users, valid_y, blended)
        primary = float(metrics["primary"])

        if primary > family_best:
            family_best = primary
            family_best_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_family = name
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()
            best_raw_valid = scores.copy()
            best_metrics = metrics

    candidate_log[name + "_blend"] = family_best
    candidate_log[name + "_alpha"] = family_best_alpha

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_family,
            "alpha": best_alpha,
            "incumbent_valid_primary": float(
                evaluate(valid_eval_users, valid_y, inc_valid_rank)["primary"]
            ),
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

# Refit the selected recipe on train + validation, then score test.
test = load("test")
test_uid = np.asarray(test.X["user_id"], dtype=np.int64)
test_vid = np.asarray(test.X["video_id"], dtype=np.int64)
test_eval_users = np.asarray(test.user_id)

fit_uid = np.concatenate([train_uid, valid_uid])
fit_vid = np.concatenate([train_vid, valid_vid])
fit_y = np.concatenate(
    [train_y, valid_y.astype(np.float32)]
)

if best_family == "positive_svd":
    final_model = fit_svd(fit_uid, fit_vid, fit_y, "positive")
    raw_test = svd_predict(final_model, test_uid, test_vid)
elif best_family == "signed_svd":
    final_model = fit_svd(fit_uid, fit_vid, fit_y, "signed")
    raw_test = svd_predict(final_model, test_uid, test_vid)
elif best_family == "temporal_additive":
    fit_temporal = concatenate_features(train_temporal, valid_temporal)
    test_temporal = temporal_features(
        test.user_id,
        test.time_ms,
        test.X["hour"],
        test.date,
    )
    final_model = AdditiveRateModel().fit(fit_temporal, fit_y)
    raw_test = final_model.predict(test_temporal)
elif best_family == "chronology_fatigue":
    test_temporal = temporal_features(
        test.user_id,
        test.time_ms,
        test.X["hour"],
        test.date,
    )
    raw_test = (
        -test_temporal[0].astype(np.float32)
        - 0.35 * test_temporal[5].astype(np.float32)
        - 0.15 * test_temporal[3].astype(np.float32)
    )
else:
    raise RuntimeError("Unknown winner")

inc_test = np.load(inc_test_path)
inc_test_rank = within_user_rank(test_eval_users, inc_test)
raw_test_rank = within_user_rank(test_eval_users, raw_test)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * raw_test_rank
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)