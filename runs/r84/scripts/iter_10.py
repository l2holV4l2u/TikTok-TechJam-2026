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
SEED = 27183
np.random.seed(SEED)

USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])
HIST_LEN = 30

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)


def concatenate_fit(fit_splits, fit_labels):
    users = np.concatenate([
        np.asarray(s.user_id, dtype=np.int64) for s in fit_splits
    ])
    videos = np.concatenate([
        np.asarray(s.video_id, dtype=np.int64) for s in fit_splits
    ])
    times = np.concatenate([
        np.asarray(s.time_ms, dtype=np.int64) for s in fit_splits
    ])
    dates = np.concatenate([
        np.asarray(s.date, dtype=np.int32) for s in fit_splits
    ])
    labels = np.concatenate([
        np.asarray(y, dtype=np.int8) for y in fit_labels
    ])
    return users, videos, times, dates, labels


def static_positive_histories(fit_splits, fit_labels, target,
                              hist_len=HIST_LEN):
    users, videos, times, _, labels = concatenate_fit(
        fit_splits, fit_labels
    )
    positions = np.arange(len(labels), dtype=np.int64)
    positive = labels > 0

    pu = users[positive]
    pv = videos[positive]
    pt = times[positive]
    pp = positions[positive]

    order = np.lexsort((pp, pt, pu))
    pu = pu[order]
    pv = pv[order]

    counts = np.bincount(pu, minlength=USER_CARD).astype(np.int64)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = ends - counts

    target_users = np.asarray(target.user_id, dtype=np.int64)
    row_starts = starts[target_users]
    row_counts = counts[target_users]

    histories = np.full(
        (len(target_users), hist_len), -1, dtype=np.int32
    )
    for col in range(hist_len):
        lag = hist_len - col
        ok = row_counts >= lag
        indices = row_starts + row_counts - lag
        histories[ok, col] = pv[indices[ok]].astype(np.int32)

    return histories


def item_prior(users, videos, labels):
    n = np.bincount(videos, minlength=VIDEO_CARD).astype(np.float64)
    p = np.bincount(
        videos,
        weights=labels.astype(np.float64),
        minlength=VIDEO_CARD
    )
    global_rate = float(labels.mean())
    rate = (p + 30.0 * global_rate) / (n + 30.0)
    return np.log(
        np.clip(rate, 1e-5, 1.0 - 1e-5) /
        np.clip(1.0 - rate, 1e-5, 1.0)
    )


def fit_signed_svd(fit_splits, fit_labels, rank=40, half_life=8.0):
    """
    Matrix-completion family. Each logged impression contributes a signed
    residual. Recent rows receive more weight so the latent factors emphasize
    preferences nearer the target date.
    """
    users, videos, _, dates, labels = concatenate_fit(
        fit_splits, fit_labels
    )
    global_rate = float(labels.mean())

    ordinal = np.asarray([
        np.datetime64(
            f"{int(d) // 10000:04d}-{(int(d) // 100) % 100:02d}-{int(d) % 100:02d}"
        ).astype("datetime64[D]").astype(np.int64)
        for d in np.unique(dates)
    ])
    unique_dates = np.unique(dates)
    date_map = dict(zip(unique_dates.tolist(), ordinal.tolist()))
    day_values = np.fromiter(
        (date_map[int(d)] for d in dates),
        dtype=np.int64,
        count=len(dates)
    )
    age = day_values.max() - day_values
    recency = np.exp2(-age.astype(np.float64) / half_life)

    # Signed residuals let non-long-view exposures repel candidates rather
    # than treating every unobserved entry as an identical negative.
    values = (
        labels.astype(np.float64) - global_rate
    ) * np.sqrt(recency)

    matrix = sparse.coo_matrix(
        (values, (users, videos)),
        shape=(USER_CARD, VIDEO_CARD),
        dtype=np.float64
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    k = min(rank, min(matrix.shape) - 1)
    u, singular, vt = svds(
        matrix,
        k=k,
        which="LM",
        return_singular_vectors=True,
        random_state=SEED
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order]

    user_factors = u * singular[None, :]
    item_factors = vt.T

    prior = item_prior(users, videos, labels)
    return user_factors.astype(np.float32), \
        item_factors.astype(np.float32), prior


def predict_signed_svd(model, target):
    user_factors, item_factors, prior = model
    users = np.asarray(target.user_id, dtype=np.int64)
    videos = np.asarray(target.video_id, dtype=np.int64)
    latent = np.einsum(
        "ij,ij->i",
        user_factors[users],
        item_factors[videos],
        optimize=True
    ).astype(np.float64)
    # A small stationary item prior stabilizes sparse users.
    latent_sd = max(float(np.std(latent)), 1e-8)
    prior_values = prior[videos]
    prior_sd = max(float(np.std(prior_values)), 1e-8)
    return latent / latent_sd + 0.20 * prior_values / prior_sd


def fit_cowatch_graph(fit_splits, fit_labels):
    """
    Graph family. Build a binary positive user-item incidence graph and derive
    cosine-normalized item-item co-watch edges.
    """
    users, videos, _, _, labels = concatenate_fit(
        fit_splits, fit_labels
    )
    positive = labels > 0
    pu = users[positive]
    pv = videos[positive]

    incidence = sparse.coo_matrix(
        (
            np.ones(len(pu), dtype=np.float32),
            (pu, pv)
        ),
        shape=(USER_CARD, VIDEO_CARD),
        dtype=np.float32
    ).tocsr()
    incidence.sum_duplicates()
    incidence.data[:] = 1.0

    item_degree = np.asarray(
        incidence.sum(axis=0)
    ).ravel().astype(np.float64)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(item_degree, 1.0))

    cowatch = (incidence.T @ incidence).tocsr()
    cowatch.setdiag(0.0)
    cowatch.eliminate_zeros()
    cowatch = sparse.diags(inv_sqrt.astype(np.float32)) @ cowatch
    cowatch = cowatch @ sparse.diags(inv_sqrt.astype(np.float32))
    cowatch = cowatch.tocsr()

    prior = item_prior(users, videos, labels)
    del incidence
    gc.collect()
    return cowatch, prior


def predict_cowatch(model, fit_splits, fit_labels, target):
    cowatch, prior = model
    histories = static_positive_histories(
        fit_splits, fit_labels, target
    )
    candidates = np.asarray(target.video_id, dtype=np.int64)

    valid_history = histories >= 0
    safe_history = np.maximum(histories, 0).astype(np.int64)

    rows = np.repeat(candidates, histories.shape[1])
    cols = safe_history.ravel()
    similarities = np.asarray(
        cowatch[rows, cols]
    ).reshape(-1)
    similarities = similarities.reshape(histories.shape)
    similarities[~valid_history] = 0.0

    # Top-neighbor pooling is less diluted than a mean when a long profile
    # contains many unrelated historical positives.
    similarities.sort(axis=1)
    graph_score = similarities[:, -5:].sum(axis=1).astype(np.float64)

    prior_values = prior[candidates].astype(np.float64)
    graph_sd = max(float(np.std(graph_score)), 1e-8)
    prior_sd = max(float(np.std(prior_values)), 1e-8)
    return graph_score / graph_sd + 0.30 * prior_values / prior_sd


def standardized_blend(own, incumbent, own_weight):
    own = np.asarray(own, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    own_z = (own - own.mean()) / max(float(own.std()), 1e-8)
    inc_z = (
        incumbent - incumbent.mean()
    ) / max(float(incumbent.std()), 1e-8)
    return own_weight * own_z + (1.0 - own_weight) * inc_z


shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

# Fit two structurally different collaborative families on train only.
svd_model = fit_signed_svd([train], [y_train])
svd_valid = predict_signed_svd(svd_model, valid)

graph_model = fit_cowatch_graph([train], [y_train])
graph_valid = predict_cowatch(
    graph_model, [train], [y_train], valid
)

raw_families = {
    "signed_recency_svd": svd_valid,
    "positive_cowatch_graph": graph_valid,
}

weights = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]
candidate_results = {}
best_primary = -np.inf
best_family = None
best_weight = None
best_scores = None
best_raw = None
best_metrics = None

for family_name, raw_scores in raw_families.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_results[f"{family_name}_raw"] = float(
        raw_metrics["primary"]
    )

    for weight in weights:
        scores = standardized_blend(
            raw_scores, inc_valid, weight
        )
        metrics = evaluate(valid.user_id, y_valid, scores)
        name = f"{family_name}_blend_{weight:.2f}"
        candidate_results[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_family = family_name
            best_weight = float(weight)
            best_scores = scores.copy()
            best_raw = raw_scores.copy()
            best_metrics = metrics

print("CANDIDATES " + json.dumps(
    candidate_results, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "winner_family": best_family,
    "winner_own_weight": best_weight,
    "svd_graph_correlation": float(
        np.corrcoef(svd_valid, graph_valid)[0, 1]
    ),
    "svd_incumbent_correlation": float(
        np.corrcoef(svd_valid, inc_valid)[0, 1]
    ),
    "graph_incumbent_correlation": float(
        np.corrcoef(graph_valid, inc_valid)[0, 1]
    )
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64)
        )

# Refit the selected recipe on train + validation, then score test.
te = load("test")
if best_family == "signed_recency_svd":
    del graph_model
    del svd_model
    gc.collect()
    final_model = fit_signed_svd(
        [train, valid], [y_train, y_valid]
    )
    raw_test = predict_signed_svd(final_model, te)
else:
    del svd_model
    del graph_model
    gc.collect()
    final_model = fit_cowatch_graph(
        [train, valid], [y_train, y_valid]
    )
    raw_test = predict_cowatch(
        final_model,
        [train, valid],
        [y_train, y_valid],
        te
    )

if best_weight < 1.0:
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_scores = standardized_blend(
        raw_test, inc_test, best_weight
    )
else:
    test_scores = raw_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))