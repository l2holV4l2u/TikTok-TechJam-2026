import os
import time
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()

SINGLE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "tab",
]
PAIR_FIELDS = [
    "author_id",
    "tag",
    "duration_bucket",
    "onehot_feat3",
]
HALF_LIVES = [None, 8.0, 4.0, 2.0]
RIDGE = 0.04


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def probability_scale(x):
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
        return np.clip(x, 1e-6, 1.0 - 1e-6)
    return expit(np.clip(x, -30.0, 30.0))


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.sort(np.unique(dates))
    day_index = {int(d): i for i, d in enumerate(unique_dates)}
    index = np.fromiter(
        (day_index[int(d)] for d in dates),
        dtype=np.float64,
        count=dates.size,
    )
    age = index.max() - index
    if half_life is None:
        w = np.ones(dates.size, dtype=np.float64)
    else:
        w = np.exp2(-age / float(half_life))
        w /= np.mean(w)
    return w


class DenseRateMap:
    def __init__(self, field, rate, count, global_rate):
        self.field = field
        self.rate = rate
        self.count = count
        self.global_rate = float(global_rate)

    def transform(self, split):
        ids = np.asarray(split.X[self.field], dtype=np.int64)
        valid = (ids >= 0) & (ids < self.rate.size)
        rate = np.full(ids.size, self.global_rate, dtype=np.float64)
        count = np.zeros(ids.size, dtype=np.float64)
        rate[valid] = self.rate[ids[valid]]
        count[valid] = self.count[ids[valid]]
        return rate, count


class PairRateMap:
    def __init__(self, field, cardinality, keys, rate, count, global_rate):
        self.field = field
        self.cardinality = int(cardinality)
        self.keys = keys
        self.rate = rate
        self.count = count
        self.global_rate = float(global_rate)

    def transform(self, split):
        users = np.asarray(split.user_id, dtype=np.int64)
        values = np.asarray(split.X[self.field], dtype=np.int64)
        query = users * self.cardinality + values

        pos = np.searchsorted(self.keys, query)
        bounded = np.minimum(pos, max(self.keys.size - 1, 0))
        found = (
            (self.keys.size > 0)
            & (pos < self.keys.size)
            & (self.keys[bounded] == query)
        )

        rate = np.full(query.size, self.global_rate, dtype=np.float64)
        count = np.zeros(query.size, dtype=np.float64)
        if self.keys.size:
            rate[found] = self.rate[pos[found]]
            count[found] = self.count[pos[found]]
        return rate, count


def fit_maps(split, row_mask, weights):
    y = np.asarray(split.y, dtype=np.float64)[row_mask]
    w = np.asarray(weights, dtype=np.float64)[row_mask]
    global_rate = float(np.sum(w * y) / np.sum(w))

    maps = []
    for field in SINGLE_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)[row_mask]
        card = int(FEATURE_CARDINALITIES[field])
        count = np.bincount(ids, weights=w, minlength=card).astype(np.float64)
        positive = np.bincount(
            ids, weights=w * y, minlength=card
        ).astype(np.float64)

        if card >= 1000:
            alpha = 18.0
        elif card >= 100:
            alpha = 35.0
        else:
            alpha = 90.0

        rate = (positive + alpha * global_rate) / (count + alpha)
        maps.append(DenseRateMap(field, rate, count, global_rate))

    users = np.asarray(split.user_id, dtype=np.int64)[row_mask]
    for field in PAIR_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        values = np.asarray(split.X[field], dtype=np.int64)[row_mask]
        keys = users * card + values
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        count = np.bincount(inverse, weights=w).astype(np.float64)
        positive = np.bincount(inverse, weights=w * y).astype(np.float64)

        alpha = 8.0 if field in ("tag", "duration_bucket") else 12.0
        rate = (positive + alpha * global_rate) / (count + alpha)
        maps.append(
            PairRateMap(
                field, card, unique_keys, rate, count, global_rate
            )
        )

    return maps, global_rate


def make_features(split, maps, global_rate):
    columns = []
    centered_logits = []
    log_counts = []

    base_logit = safe_logit(global_rate)
    for mapping in maps:
        rate, count = mapping.transform(split)
        evidence = safe_logit(rate) - base_logit
        log_count = np.log1p(count)

        centered_logits.append(evidence)
        log_counts.append(log_count)
        columns.append(evidence)
        columns.append(log_count)

    X = np.column_stack(columns).astype(np.float64, copy=False)
    evidence = np.column_stack(centered_logits)
    counts = np.column_stack(log_counts)
    return X, evidence, counts


def fit_logistic(X, y):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-5] = 1.0
    Z = (X - mean) / scale
    y = np.asarray(y, dtype=np.float64)

    initial = np.zeros(Z.shape[1] + 1, dtype=np.float64)
    initial[0] = safe_logit(y.mean())

    def objective(theta):
        logits = theta[0] + Z @ theta[1:]
        loss = np.logaddexp(0.0, logits) - y * logits
        penalty = 0.5 * RIDGE * np.dot(theta[1:], theta[1:])
        value = float(loss.mean() + penalty)

        residual = expit(logits) - y
        gradient = np.empty_like(theta)
        gradient[0] = residual.mean()
        gradient[1:] = Z.T @ residual / y.size + RIDGE * theta[1:]
        return value, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 80, "ftol": 1e-9, "maxls": 25},
    )
    return result.x, mean, scale, bool(result.success), float(result.fun)


def logistic_predict(X, fitted):
    theta, mean, scale = fitted[:3]
    Z = (X - mean) / scale
    return theta[0] + Z @ theta[1:]


def family_scores(X, evidence, counts, fitted):
    logistic = logistic_predict(X, fitted)

    n_single = len(SINGLE_FIELDS)
    single_evidence = evidence[:, :n_single]
    pair_evidence = evidence[:, n_single:]
    single_counts = counts[:, :n_single]
    pair_counts = counts[:, n_single:]

    # Naive-Bayes-style sum of smoothed log likelihood evidence. Reliability
    # factors prevent rare categories from contributing as strongly as videos
    # and authors with substantial support.
    single_reliability = 1.0 - np.exp(-single_counts / 3.0)
    nb = np.sum(single_evidence * single_reliability, axis=1)
    nb /= np.sqrt(float(n_single))

    # User preference score: stable item/entity evidence plus deviations for
    # the user's historically observed author and content attributes.
    item_part = (
        1.25 * single_evidence[:, 0]
        + 0.85 * single_evidence[:, 1]
        + 0.45 * single_evidence[:, 2]
        + 0.30 * single_evidence[:, 3]
        + 0.20 * single_evidence[:, 8]
    )
    pair_reliability = 1.0 - np.exp(-pair_counts / 1.5)
    preference = item_part + np.sum(
        pair_evidence * pair_reliability, axis=1
    ) / np.sqrt(float(len(PAIR_FIELDS)))

    return {
        "empirical_bayes_stack": logistic,
        "naive_bayes_evidence": nb,
        "user_attribute_preference": preference,
    }


train = load("train")
dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.sort(np.unique(dates))
inner_valid_dates = unique_dates[-3:]
fit_mask = dates < inner_valid_dates[0]
inner_mask = ~fit_mask
inner_y = np.asarray(train.y, dtype=np.int8)[inner_mask]
inner_users = np.asarray(train.user_id, dtype=np.int64)[inner_mask]

inner_results = {}
selected = None

for half_life in HALF_LIVES:
    all_weights = recency_weights(dates, half_life)
    maps, global_rate = fit_maps(train, fit_mask, all_weights)
    X_all, evidence_all, counts_all = make_features(
        train, maps, global_rate
    )
    X_inner = X_all[inner_mask]
    evidence_inner = evidence_all[inner_mask]
    counts_inner = counts_all[inner_mask]

    fitted_full = fit_logistic(X_inner, inner_y)
    fitted = fitted_full[:3]
    family = family_scores(
        X_inner, evidence_inner, counts_inner, fitted
    )
    metrics = evaluate(
        inner_users, inner_y, family["empirical_bayes_stack"]
    )
    label = "uniform" if half_life is None else f"hl_{half_life:g}"
    inner_results[label] = float(metrics["primary"])

    if selected is None or metrics["primary"] > selected["primary"]:
        selected = {
            "half_life": half_life,
            "primary": float(metrics["primary"]),
            "theta": fitted[0],
            "mean": fitted[1],
            "scale": fitted[2],
            "optimizer_success": fitted_full[3],
            "objective": fitted_full[4],
        }

    del X_all, evidence_all, counts_all, X_inner

print("FINDINGS " + json.dumps({
    "inner_temporal_primary": inner_results,
    "selected_half_life": selected["half_life"],
    "inner_rows": int(inner_mask.sum()),
    "fit_rows": int(fit_mask.sum()),
    "optimizer_success": selected["optimizer_success"],
    "optimizer_objective": selected["objective"],
}))

# Re-estimate all empirical rates from the full training split. The stacker's
# combination coefficients and selected half-life remain entirely determined
# by the train-only temporal experiment above.
full_weights = recency_weights(dates, selected["half_life"])
full_mask = np.ones(dates.size, dtype=bool)
full_maps, full_global_rate = fit_maps(train, full_mask, full_weights)
fitted = (selected["theta"], selected["mean"], selected["scale"])

valid = load("valid")
X_valid, evidence_valid, counts_valid = make_features(
    valid, full_maps, full_global_rate
)
valid_families = family_scores(
    X_valid, evidence_valid, counts_valid, fitted
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_valid_prob = probability_scale(inc_valid)

candidate_log = {}
best = None
blend_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

for family_name, raw_scores in valid_families.items():
    own_prob = probability_scale(raw_scores)
    raw_metrics = evaluate(valid.user_id, valid.y, raw_scores)
    candidate_log[family_name] = float(raw_metrics["primary"])

    for alpha in blend_alphas:
        blended = alpha * own_prob + (1.0 - alpha) * inc_valid_prob
        metrics = evaluate(valid.user_id, valid.y, blended)
        name = f"{family_name}_blend_{alpha:.2f}"
        candidate_log[name] = float(metrics["primary"])

        if best is None or metrics["primary"] > best["metrics"]["primary"]:
            best = {
                "name": name,
                "family": family_name,
                "alpha": alpha,
                "scores": blended,
                "raw": np.asarray(raw_scores, dtype=np.float64),
                "metrics": metrics,
            }

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": best["name"],
    "winner_family": best["family"],
    "winner_blend_alpha": best["alpha"],
    "global_train_rate_weighted": full_global_rate,
}))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best["scores"], dtype=np.float64),
    )
    if best["alpha"] < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best["raw"], dtype=np.float64),
        )

test = load("test")
X_test, evidence_test, counts_test = make_features(
    test, full_maps, full_global_rate
)
test_families = family_scores(
    X_test, evidence_test, counts_test, fitted
)
own_test_raw = np.asarray(
    test_families[best["family"]], dtype=np.float64
)
own_test_prob = probability_scale(own_test_raw)
inc_test = np.load(inc_test_path).astype(np.float64)
inc_test_prob = probability_scale(inc_test)
test_scores = (
    best["alpha"] * own_test_prob
    + (1.0 - best["alpha"]) * inc_test_prob
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best["metrics"]["primary"]),
    "gauc": float(best["metrics"]["gauc"]),
    "ndcg@5": float(best["metrics"]["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))