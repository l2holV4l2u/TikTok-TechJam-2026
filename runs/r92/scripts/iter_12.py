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
SEED = 18473
RANK = 32


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = (len(unique_dates) - 1 - day_index).astype(np.float64)
    w = np.exp2(-age / float(half_life))
    return (w / np.mean(w)).astype(np.float64)


def within_user_ranks(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    sorted_users = user_ids[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first)
    group_id = np.cumsum(first) - 1
    local = np.arange(n, dtype=np.int64) - starts[group_id]
    sizes = np.diff(np.r_[starts, n])

    rank = (local.astype(np.float64) + 0.5) / sizes[group_id]
    result = np.empty(n, dtype=np.float64)
    result[order] = rank
    return result


def build_spectral_model(
    user_ids,
    entity_ids,
    labels,
    dates,
    n_users,
    n_entities,
    objective,
    rank=RANK,
):
    users = np.asarray(user_ids, dtype=np.int32)
    entities = np.asarray(entity_ids, dtype=np.int32)
    y = np.asarray(labels, dtype=np.float64)
    w = recency_weights(dates, half_life=4.0)

    if objective == "signed":
        prior = float(np.sum(w * y) / np.sum(w))
        values = w * (y - prior)
    elif objective == "positive":
        prior = 0.0
        values = w * y
    else:
        raise ValueError(objective)

    signal = sparse.coo_matrix(
        (values, (users, entities)),
        shape=(n_users, n_entities),
        dtype=np.float64,
    ).tocsr()

    support = sparse.coo_matrix(
        (w, (users, entities)),
        shape=(n_users, n_entities),
        dtype=np.float64,
    ).tocsr()

    # Normalize repeated pairs, active users, and globally frequent entities.
    # This prevents the leading singular vectors from merely reproducing
    # exposure volume or popularity.
    pair_support = support.copy()
    pair_support.data = np.sqrt(np.maximum(pair_support.data, 1e-12))
    inv_pair = pair_support.copy()
    inv_pair.data = 1.0 / inv_pair.data
    signal = signal.multiply(inv_pair).tocsr()

    row_degree = np.asarray(support.sum(axis=1)).ravel()
    col_degree = np.asarray(support.sum(axis=0)).ravel()

    row_scale = np.zeros_like(row_degree)
    col_scale = np.zeros_like(col_degree)
    good_rows = row_degree > 0
    good_cols = col_degree > 0
    row_scale[good_rows] = np.power(row_degree[good_rows], -0.25)
    col_scale[good_cols] = np.power(col_degree[good_cols], -0.25)

    normalized = sparse.diags(row_scale).dot(signal)
    normalized = normalized.dot(sparse.diags(col_scale)).tocsr()

    k = min(rank, min(normalized.shape) - 1)
    v0 = np.full(
        min(normalized.shape),
        1.0 / np.sqrt(float(min(normalized.shape))),
        dtype=np.float64,
    )

    u, singular, vt = svds(
        normalized,
        k=k,
        which="LM",
        v0=v0,
        tol=2e-3,
        maxiter=350,
        return_singular_vectors=True,
    )

    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order, :]

    # Store the singular value symmetrically for stable row-wise scoring.
    root_s = np.sqrt(np.maximum(singular, 0.0))
    user_factors = (u * root_s[None, :]).astype(np.float32)
    entity_factors = (vt.T * root_s[None, :]).astype(np.float32)

    return {
        "user_factors": user_factors,
        "entity_factors": entity_factors,
        "prior": prior,
        "objective": objective,
    }


def spectral_predict(model, user_ids, entity_ids):
    users = np.asarray(user_ids, dtype=np.int64)
    entities = np.asarray(entity_ids, dtype=np.int64)
    uf = model["user_factors"]
    ef = model["entity_factors"]

    valid = (
        (users >= 0) & (users < len(uf)) &
        (entities >= 0) & (entities < len(ef))
    )
    result = np.zeros(len(users), dtype=np.float32)

    idx = np.flatnonzero(valid)
    batch = 200000
    for begin in range(0, len(idx), batch):
        take = idx[begin:begin + batch]
        result[take] = np.einsum(
            "ij,ij->i",
            uf[users[take]],
            ef[entities[take]],
            optimize=True,
        )
    return result


def fit_and_predict_single(
    fit_parts,
    target,
    entity_name,
    objective,
):
    fit_users = np.concatenate([
        np.asarray(p.user_id, dtype=np.int32) for p in fit_parts
    ])
    fit_entities = np.concatenate([
        np.asarray(
            p.video_id if entity_name == "video_id" else p.X[entity_name],
            dtype=np.int32,
        )
        for p in fit_parts
    ])
    fit_labels = np.concatenate([
        np.asarray(p.y, dtype=np.float64) for p in fit_parts
    ])
    fit_dates = np.concatenate([
        np.asarray(p.date) for p in fit_parts
    ])

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_entities = int(FEATURE_CARDINALITIES[entity_name])

    model = build_spectral_model(
        fit_users,
        fit_entities,
        fit_labels,
        fit_dates,
        n_users,
        n_entities,
        objective,
    )

    target_entities = (
        np.asarray(target.video_id, dtype=np.int32)
        if entity_name == "video_id"
        else np.asarray(target.X[entity_name], dtype=np.int32)
    )
    pred = spectral_predict(
        model,
        np.asarray(target.user_id, dtype=np.int32),
        target_entities,
    )
    return pred, model


train = load("train")
valid = load("valid")
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_rank = within_user_ranks(valid.user_id, inc_valid)

raw_predictions = {}

# Family 1: signed collaborative residual factorization. Every exposure,
# including negatives, contributes evidence about relative user-video taste.
signed_video, model = fit_and_predict_single(
    [train], valid, "video_id", "signed"
)
raw_predictions["signed_video_spectral"] = signed_video
del model
gc.collect()

# Family 2: positive-only implicit PureSVD. This models affinity among consumed
# entities without assuming that every logged negative is a true dislike.
positive_video, model = fit_and_predict_single(
    [train], valid, "video_id", "positive"
)
raw_predictions["positive_video_puresvd"] = positive_video
del model
gc.collect()

# Family 3: content-graph spectral profile. Author and tag are substantially
# more reusable than individual videos across the date boundary.
signed_author, model = fit_and_predict_single(
    [train], valid, "author_id", "signed"
)
del model
gc.collect()

signed_tag, model = fit_and_predict_single(
    [train], valid, "tag", "signed"
)
del model
gc.collect()

author_rank = within_user_ranks(valid.user_id, signed_author)
tag_rank = within_user_ranks(valid.user_id, signed_tag)
raw_predictions["content_graph_spectral"] = (
    0.65 * author_rank + 0.35 * tag_rank
)

# Family 4: rank aggregation of identity-level collaborative affinity and the
# more stationary content graph.
signed_video_rank = within_user_ranks(valid.user_id, signed_video)
content_rank = within_user_ranks(
    valid.user_id,
    raw_predictions["content_graph_spectral"],
)
raw_predictions["spectral_rank_ensemble"] = (
    0.55 * signed_video_rank + 0.45 * content_rank
)

candidates = {}
records = {}

for family, raw in raw_predictions.items():
    raw = np.asarray(raw, dtype=np.float64)
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidates[family + "_raw"] = float(raw_metrics["primary"])

    raw_rank = within_user_ranks(valid.user_id, raw)
    correlation = float(np.corrcoef(inc_rank, raw_rank)[0, 1])

    for alpha in (0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0):
        scores = (1.0 - alpha) * inc_rank + alpha * raw_rank
        metrics = evaluate(valid.user_id, y_valid, scores)
        name = family + "_blend_" + str(alpha)
        candidates[name] = float(metrics["primary"])
        records[name] = {
            "family": family,
            "alpha": float(alpha),
            "scores": scores,
            "raw": raw,
            "metrics": metrics,
            "correlation": correlation,
        }

winner_name = max(
    records,
    key=lambda name: records[name]["metrics"]["primary"],
)
winner = records[winner_name]

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": winner_name,
    "winner_family": winner["family"],
    "winner_alpha": winner["alpha"],
    "winner_incumbent_rank_correlation": winner["correlation"],
    "incumbent_primary": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    ),
    "family_correlations": {
        family: float(np.corrcoef(
            inc_rank,
            within_user_ranks(valid.user_id, raw),
        )[0, 1])
        for family, raw in raw_predictions.items()
    },
}, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner["scores"], dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(winner["raw"], dtype=np.float64),
    )

# Apply the validation-selected family and fixed blend weight to test. The
# spectral component is refit on train+validation; test labels are never read.
test = load("test")
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
inc_test_rank = within_user_ranks(test.user_id, inc_test)

selected = winner["family"]

if selected == "signed_video_spectral":
    raw_test, model = fit_and_predict_single(
        [train, valid], test, "video_id", "signed"
    )
    del model

elif selected == "positive_video_puresvd":
    raw_test, model = fit_and_predict_single(
        [train, valid], test, "video_id", "positive"
    )
    del model

elif selected in ("content_graph_spectral", "spectral_rank_ensemble"):
    test_author, model = fit_and_predict_single(
        [train, valid], test, "author_id", "signed"
    )
    del model
    gc.collect()

    test_tag, model = fit_and_predict_single(
        [train, valid], test, "tag", "signed"
    )
    del model
    gc.collect()

    test_content = (
        0.65 * within_user_ranks(test.user_id, test_author) +
        0.35 * within_user_ranks(test.user_id, test_tag)
    )

    if selected == "content_graph_spectral":
        raw_test = test_content
    else:
        test_video, model = fit_and_predict_single(
            [train, valid], test, "video_id", "signed"
        )
        del model
        gc.collect()
        raw_test = (
            0.55 * within_user_ranks(test.user_id, test_video) +
            0.45 * within_user_ranks(test.user_id, test_content)
        )
else:
    raise RuntimeError("Unknown selected family: " + selected)

raw_test_rank = within_user_ranks(test.user_id, raw_test)
test_scores = (
    (1.0 - winner["alpha"]) * inc_test_rank +
    winner["alpha"] * raw_test_rank
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

metrics = winner["metrics"]
elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))