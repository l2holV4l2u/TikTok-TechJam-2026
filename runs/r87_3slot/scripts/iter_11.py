import os
import time
import json
import gc
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 74129
np.random.seed(SEED)

CONTENT_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
    "tab",
]
NB_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "tab",
]
PAIR_FIELDS = [
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
]

SMOOTHING = {
    "video_id": 45.0,
    "author_id": 55.0,
    "tag": 90.0,
    "duration_bucket": 120.0,
    "upload_type": 120.0,
    "music_type": 120.0,
    "onehot_feat3": 55.0,
    "onehot_feat8": 65.0,
    "tab": 100.0,
    "hour": 100.0,
}


def concat_array(parts, getter, dtype=None):
    if len(parts) == 1:
        x = np.asarray(getter(parts[0]))
    else:
        x = np.concatenate([np.asarray(getter(p)) for p in parts])
    if dtype is not None:
        x = x.astype(dtype, copy=False)
    return x


def source_labels(parts):
    return concat_array(parts, lambda p: p.y, np.float64)


def source_dates(parts):
    return concat_array(parts, lambda p: p.date, np.int64)


def recency_weights(parts, half_life):
    dates = source_dates(parts)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = day_index.max() - day_index
    return np.exp2(-age.astype(np.float64) / float(half_life))


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1.0e-5, 1.0 - 1.0e-5)
    return np.log(p) - np.log1p(-p)


def fit_temporal_eb(parts):
    y = source_labels(parts)
    w_slow = recency_weights(parts, 6.0)
    w_fast = recency_weights(parts, 2.0)

    prior_slow = float(np.sum(w_slow * y) / np.sum(w_slow))
    prior_fast = float(np.sum(w_fast * y) / np.sum(w_fast))

    model = {
        "prior_slow": prior_slow,
        "prior_fast": prior_fast,
        "fields": {},
    }

    for field in CONTENT_FIELDS:
        x = concat_array(parts, lambda p, f=field: p.X[f], np.int64)
        k = int(FEATURE_CARDINALITIES[field])
        smooth = SMOOTHING[field]

        den_slow = np.bincount(x, weights=w_slow, minlength=k)
        pos_slow = np.bincount(x, weights=w_slow * y, minlength=k)
        rate_slow = (
            pos_slow + smooth * prior_slow
        ) / (den_slow + smooth)

        den_fast = np.bincount(x, weights=w_fast, minlength=k)
        pos_fast = np.bincount(x, weights=w_fast * y, minlength=k)
        rate_fast = (
            pos_fast + smooth * prior_fast
        ) / (den_fast + smooth)

        slow_logit = safe_logit(rate_slow)
        fast_logit = safe_logit(rate_fast)

        # Conservative one-step extrapolation of the recent-vs-slow movement.
        forecast = fast_logit + 0.65 * np.clip(
            fast_logit - slow_logit, -1.25, 1.25
        )
        support = den_fast / (den_fast + smooth)
        forecast = (
            support * forecast
            + (1.0 - support) * safe_logit(prior_fast)
        )

        model["fields"][field] = forecast.astype(np.float32)

    return model


def predict_temporal_eb(model, split):
    field_weights = {
        "video_id": 1.35,
        "author_id": 1.20,
        "tag": 0.70,
        "duration_bucket": 0.55,
        "upload_type": 0.45,
        "music_type": 0.30,
        "onehot_feat3": 0.75,
        "onehot_feat8": 0.55,
        "tab": 0.65,
    }
    result = np.zeros(len(split.user_id), dtype=np.float64)
    total = 0.0
    for field in CONTENT_FIELDS:
        weight = field_weights[field]
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += weight * model["fields"][field][ids]
        total += weight
    return result / total


def fit_naive_bayes(parts):
    y = source_labels(parts)
    w = recency_weights(parts, 3.5)
    pos_total = float(np.sum(w * y))
    neg_total = float(np.sum(w * (1.0 - y)))
    prior = (pos_total + 1.0) / (pos_total + neg_total + 2.0)

    model = {
        "prior_logit": float(safe_logit(prior)),
        "fields": {},
    }

    for field in NB_FIELDS:
        x = concat_array(parts, lambda p, f=field: p.X[f], np.int64)
        k = int(FEATURE_CARDINALITIES[field])
        alpha = 2.0

        pos = np.bincount(x, weights=w * y, minlength=k)
        neg = np.bincount(x, weights=w * (1.0 - y), minlength=k)

        log_pos_prob = np.log(pos + alpha) - np.log(
            pos_total + alpha * k
        )
        log_neg_prob = np.log(neg + alpha) - np.log(
            neg_total + alpha * k
        )
        evidence = np.clip(log_pos_prob - log_neg_prob, -2.5, 2.5)

        support = (pos + neg) / (pos + neg + 30.0)
        model["fields"][field] = (
            support * evidence
        ).astype(np.float32)

    return model


def predict_naive_bayes(model, split):
    result = np.full(
        len(split.user_id), model["prior_logit"], dtype=np.float64
    )
    # Tempering avoids excessive confidence from conditionally dependent fields.
    for field in NB_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += 0.42 * model["fields"][field][ids]
    return result


def fit_pair_table(users, values, labels, weights, value_cardinality,
                   prior, smoothing):
    keys = (
        users.astype(np.int64) * np.int64(value_cardinality)
        + values.astype(np.int64)
    )
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    den = np.bincount(inverse, weights=weights)
    pos = np.bincount(inverse, weights=weights * labels)
    rate = (pos + smoothing * prior) / (den + smoothing)

    reliability = den / (den + smoothing)
    residual_logit = reliability * (
        safe_logit(rate) - float(safe_logit(prior))
    )
    return (
        unique_keys.astype(np.int64),
        residual_logit.astype(np.float32),
    )


def lookup_pair(table, users, values, value_cardinality):
    keys_table, values_table = table
    query = (
        users.astype(np.int64) * np.int64(value_cardinality)
        + values.astype(np.int64)
    )
    idx = np.searchsorted(keys_table, query)
    valid = idx < len(keys_table)
    out = np.zeros(len(query), dtype=np.float64)
    if np.any(valid):
        valid_rows = np.flatnonzero(valid)
        matched = keys_table[idx[valid]] == query[valid]
        matched_rows = valid_rows[matched]
        out[matched_rows] = values_table[idx[matched_rows]]
    return out


def fit_hierarchical_cohort(parts):
    y = source_labels(parts)
    w = recency_weights(parts, 4.0)
    users = concat_array(parts, lambda p: p.user_id, np.int64)
    prior = float(np.sum(w * y) / np.sum(w))

    base = fit_temporal_eb(parts)
    tables = {}

    pair_smoothing = {
        "author_id": 10.0,
        "tag": 7.0,
        "duration_bucket": 8.0,
        "tab": 9.0,
    }

    for field in PAIR_FIELDS:
        values = concat_array(parts, lambda p, f=field: p.X[f], np.int64)
        k = int(FEATURE_CARDINALITIES[field])
        tables[field] = fit_pair_table(
            users,
            values,
            y,
            w,
            k,
            prior,
            pair_smoothing[field],
        )

    return {
        "base": base,
        "prior": prior,
        "tables": tables,
    }


def predict_hierarchical_cohort(model, split):
    result = predict_temporal_eb(model["base"], split)
    users = np.asarray(split.user_id, dtype=np.int64)

    weights = {
        "author_id": 0.75,
        "tag": 0.55,
        "duration_bucket": 0.45,
        "tab": 0.35,
    }
    for field in PAIR_FIELDS:
        values = np.asarray(split.X[field], dtype=np.int64)
        result += weights[field] * lookup_pair(
            model["tables"][field],
            users,
            values,
            int(FEATURE_CARDINALITIES[field]),
        )
    return result


def fit_implicit_svd(parts, rank=24):
    users = concat_array(parts, lambda p: p.user_id, np.int64)
    videos = concat_array(parts, lambda p: p.video_id, np.int64)
    y = source_labels(parts)
    w = recency_weights(parts, 5.0)

    positive = y > 0.5
    matrix = sparse.coo_matrix(
        (
            w[positive].astype(np.float32),
            (users[positive], videos[positive]),
        ),
        shape=(
            int(FEATURE_CARDINALITIES["user_id"]),
            int(FEATURE_CARDINALITIES["video_id"]),
        ),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data = np.log1p(matrix.data)

    row_norm = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
    inv_norm = np.zeros_like(row_norm, dtype=np.float32)
    nonzero = row_norm > 0
    inv_norm[nonzero] = 1.0 / row_norm[nonzero]
    normalized = sparse.diags(inv_norm).dot(matrix)

    u, singular, vt = svds(
        normalized,
        k=rank,
        which="LM",
        random_state=SEED,
        tol=2.0e-3,
        maxiter=350,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    root_s = np.sqrt(np.maximum(singular, 1.0e-8))
    user_factors = u * root_s[None, :]
    video_factors = vt.T * root_s[None, :]

    popularity = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    popularity = np.log1p(popularity)

    return {
        "user_factors": user_factors,
        "video_factors": video_factors,
        "popularity": popularity,
    }


def predict_implicit_svd(model, split):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    result = np.einsum(
        "ij,ij->i",
        model["user_factors"][users],
        model["video_factors"][videos],
        optimize=True,
    ).astype(np.float64)
    result += 0.08 * model["popularity"][videos]
    return result


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    positions = np.arange(n, dtype=np.int64)
    start_pos = np.where(starts, positions, 0)
    start_pos = np.maximum.accumulate(start_pos)
    local = positions - start_pos

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_pos = np.where(ends, positions + 1, n)
    end_pos = np.minimum.accumulate(end_pos[::-1])[::-1]
    sizes = end_pos - start_pos

    ranked = local / np.maximum(sizes - 1, 1)
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def fit_family(name, parts):
    if name == "temporal_eb":
        return fit_temporal_eb(parts)
    if name == "naive_bayes":
        return fit_naive_bayes(parts)
    if name == "hierarchical_cohort":
        return fit_hierarchical_cohort(parts)
    if name == "implicit_svd":
        return fit_implicit_svd(parts)
    raise ValueError(name)


def predict_family(name, model, split):
    if name == "temporal_eb":
        return predict_temporal_eb(model, split)
    if name == "naive_bayes":
        return predict_naive_bayes(model, split)
    if name == "hierarchical_cohort":
        return predict_hierarchical_cohort(model, split)
    if name == "implicit_svd":
        return predict_implicit_svd(model, split)
    raise ValueError(name)


train = load("train")
valid = load("valid")
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

families = [
    "temporal_eb",
    "naive_bayes",
    "hierarchical_cohort",
    "implicit_svd",
]

valid_predictions = {}
models = {}
for family in families:
    model = fit_family(family, [train])
    models[family] = model
    valid_predictions[family] = predict_family(family, model, valid)

candidate_scores = {}
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])

best_name = "incumbent"
best_metrics = inc_metrics
best_scores = inc_valid.copy()
best_family = None
best_alpha = 0.0
best_blend_type = "incumbent"
best_raw = None

inc_rank = within_user_rank(valid.user_id, inc_valid)
alphas = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70]

for family in families:
    raw = valid_predictions[family]
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_scores[family] = float(raw_metrics["primary"])

    if raw_metrics["primary"] > best_metrics["primary"]:
        best_name = family
        best_metrics = raw_metrics
        best_scores = raw.copy()
        best_family = family
        best_alpha = 1.0
        best_blend_type = "raw"
        best_raw = raw.copy()

    family_rank = within_user_rank(valid.user_id, raw)
    for alpha in alphas:
        blended = (1.0 - alpha) * inc_rank + alpha * family_rank
        metrics = evaluate(valid.user_id, y_valid, blended)
        name = "%s_rankblend_%.2f" % (family, alpha)
        candidate_scores[name] = float(metrics["primary"])

        if metrics["primary"] > best_metrics["primary"]:
            best_name = name
            best_metrics = metrics
            best_scores = blended.copy()
            best_family = family
            best_alpha = float(alpha)
            best_blend_type = "rank"
            best_raw = raw.copy()

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)
print(
    "FINDINGS selected=%s family=%s blend=%s alpha=%.2f "
    "temporal_eb=%.6f naive_bayes=%.6f cohort=%.6f svd=%.6f"
    % (
        best_name,
        str(best_family),
        best_blend_type,
        best_alpha,
        candidate_scores["temporal_eb"],
        candidate_scores["naive_bayes"],
        candidate_scores["hierarchical_cohort"],
        candidate_scores["implicit_svd"],
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_raw is not None and best_blend_type == "rank":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

# Refit the selected recipe on train + validation, then score test.
test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)

if best_family is None:
    test_scores = inc_test
else:
    # Release validation-only models before the refit.
    models.clear()
    gc.collect()

    refit_model = fit_family(best_family, [train, valid])
    raw_test = predict_family(best_family, refit_model, test)

    if best_blend_type == "raw":
        test_scores = raw_test
    else:
        incumbent_test_rank = within_user_rank(test.user_id, inc_test)
        raw_test_rank = within_user_rank(test.user_id, raw_test)
        test_scores = (
            (1.0 - best_alpha) * incumbent_test_rank
            + best_alpha * raw_test_rank
        )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)