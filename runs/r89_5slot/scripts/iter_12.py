import os
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 28411
RANK = 32
BLEND_WEIGHTS = (0.04, 0.07, 0.10, 0.14, 0.19, 0.25, 0.32, 0.40)

np.random.seed(SEED)


def arrays_from_splits(splits, field, need_y=True):
    users = np.concatenate([
        np.asarray(s.user_id, dtype=np.int64) for s in splits
    ])
    entities = np.concatenate([
        np.asarray(s.X[field], dtype=np.int64) for s in splits
    ])
    if need_y:
        labels = np.concatenate([
            np.asarray(s.y, dtype=np.float32) for s in splits
        ])
        return users, entities, labels
    return users, entities


def smoothed_entity_rates(entity, y, cardinality, alpha=30.0):
    counts = np.bincount(entity, minlength=cardinality).astype(np.float64)
    positives = np.bincount(
        entity, weights=y, minlength=cardinality
    ).astype(np.float64)
    global_rate = float(y.mean())
    rates = (positives + alpha * global_rate) / (counts + alpha)
    return rates.astype(np.float32), global_rate


def aggregate_sparse(users, entities, values, shape):
    sums = sparse.coo_matrix(
        (values.astype(np.float32, copy=False), (users, entities)),
        shape=shape,
        dtype=np.float32,
    ).tocsr()
    counts = sparse.coo_matrix(
        (
            np.ones(len(users), dtype=np.float32),
            (users, entities),
        ),
        shape=shape,
        dtype=np.float32,
    ).tocsr()
    inv = counts.copy()
    inv.data = 1.0 / np.maximum(inv.data, 1.0)
    return sums.multiply(inv).tocsr()


def fit_residual_svd(splits, field, rank=RANK):
    users, entities, y = arrays_from_splits(splits, field, need_y=True)
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_entities = int(FEATURE_CARDINALITIES[field])

    rates, global_rate = smoothed_entity_rates(
        entities, y, n_entities, alpha=35.0
    )
    residual = y - rates[entities]

    mat = aggregate_sparse(
        users, entities, residual, (n_users, n_entities)
    )

    # Equalize highly active users so the factors describe preference
    # direction rather than merely training activity.
    row_nnz = np.diff(mat.indptr).astype(np.float32)
    row_scale = 1.0 / np.sqrt(np.maximum(row_nnz, 1.0))
    weighted = sparse.diags(row_scale).dot(mat).tocsr()

    k = min(rank, min(weighted.shape) - 1)
    u, singular, vt = svds(
        weighted,
        k=k,
        which="LM",
        random_state=SEED + (17 if field == "video_id" else 31),
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    # Undo the row normalization at prediction time.
    user_factor = (
        u * singular[None, :] *
        np.sqrt(np.maximum(row_nnz, 1.0))[:, None]
    ).astype(np.float32)
    entity_factor = vt.T.astype(np.float32)

    return {
        "field": field,
        "rates": rates,
        "global": global_rate,
        "user_factor": user_factor,
        "entity_factor": entity_factor,
    }


def predict_residual_svd(model, split):
    users = np.asarray(split.user_id, dtype=np.int64)
    entities = np.asarray(split.X[model["field"]], dtype=np.int64)
    interaction = np.einsum(
        "ij,ij->i",
        model["user_factor"][users],
        model["entity_factor"][entities],
        optimize=True,
    )
    return (
        model["rates"][entities].astype(np.float64)
        + interaction.astype(np.float64)
    )


def fit_positive_spectral(splits, rank=RANK):
    users, videos, y = arrays_from_splits(
        splits, "video_id", need_y=True
    )
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    rates, global_rate = smoothed_entity_rates(
        videos, y, n_videos, alpha=40.0
    )

    positive = sparse.coo_matrix(
        (y.astype(np.float32), (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    positive.data[:] = 1.0

    user_degree = np.asarray(positive.sum(axis=1)).ravel().astype(np.float32)
    item_degree = np.asarray(positive.sum(axis=0)).ravel().astype(np.float32)

    # Symmetric degree normalization makes this a spectral collaborative
    # model rather than a popularity reconstruction.
    left = 1.0 / np.sqrt(np.maximum(user_degree, 1.0))
    right = 1.0 / np.sqrt(np.maximum(item_degree, 1.0))
    normalized = (
        sparse.diags(left).dot(positive).dot(sparse.diags(right)).tocsr()
    )

    k = min(rank, min(normalized.shape) - 1)
    u, singular, vt = svds(
        normalized,
        k=k,
        which="LM",
        random_state=SEED + 53,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    user_factor = (
        u * singular[None, :] *
        np.sqrt(np.maximum(user_degree, 1.0))[:, None]
    ).astype(np.float32)
    item_factor = (
        vt.T * np.sqrt(np.maximum(item_degree, 1.0))[:, None]
    ).astype(np.float32)

    return {
        "rates": rates,
        "global": global_rate,
        "user_factor": user_factor,
        "item_factor": item_factor,
    }


def predict_positive_spectral(model, split):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    latent = np.einsum(
        "ij,ij->i",
        model["user_factor"][users],
        model["item_factor"][videos],
        optimize=True,
    )
    return (
        model["rates"][videos].astype(np.float64)
        + 0.12 * latent.astype(np.float64)
    )


def fit_pair_table(splits, field, alpha):
    users, entities, y = arrays_from_splits(splits, field, need_y=True)
    card = int(FEATURE_CARDINALITIES[field])
    entity_rate, global_rate = smoothed_entity_rates(
        entities, y, card, alpha=35.0
    )

    pair_key = users * np.int64(card) + entities
    unique_key, inverse, counts = np.unique(
        pair_key, return_inverse=True, return_counts=True
    )
    positives = np.bincount(
        inverse, weights=y, minlength=len(unique_key)
    ).astype(np.float64)
    entity_for_key = (unique_key % card).astype(np.int64)
    prior = entity_rate[entity_for_key].astype(np.float64)

    posterior = (
        positives + alpha * prior
    ) / (counts.astype(np.float64) + alpha)

    # Store a centered preference residual, leaving global entity quality
    # to the separate item prior.
    residual = (posterior - prior).astype(np.float32)
    return {
        "field": field,
        "card": card,
        "keys": unique_key.astype(np.int64),
        "residual": residual,
        "entity_rate": entity_rate,
        "global": global_rate,
    }


def lookup_pair_residual(table, split):
    users = np.asarray(split.user_id, dtype=np.int64)
    entities = np.asarray(split.X[table["field"]], dtype=np.int64)
    query = users * np.int64(table["card"]) + entities

    pos = np.searchsorted(table["keys"], query)
    valid = pos < len(table["keys"])
    safe = np.minimum(pos, len(table["keys"]) - 1)
    valid &= table["keys"][safe] == query

    out = np.zeros(len(query), dtype=np.float64)
    out[valid] = table["residual"][safe[valid]]
    return out


def fit_hierarchical_eb(splits):
    video_users, video_ids, y = arrays_from_splits(
        splits, "video_id", need_y=True
    )
    video_rates, global_rate = smoothed_entity_rates(
        video_ids,
        y,
        int(FEATURE_CARDINALITIES["video_id"]),
        alpha=45.0,
    )
    tables = [
        fit_pair_table(splits, "author_id", alpha=8.0),
        fit_pair_table(splits, "tag", alpha=11.0),
        fit_pair_table(splits, "duration_bucket", alpha=13.0),
        fit_pair_table(splits, "upload_type", alpha=14.0),
    ]
    return {
        "video_rates": video_rates,
        "global": global_rate,
        "tables": tables,
    }


def predict_hierarchical_eb(model, split):
    videos = np.asarray(split.video_id, dtype=np.int64)
    score = model["video_rates"][videos].astype(np.float64)
    weights = (1.00, 0.52, 0.36, 0.25)
    for weight, table in zip(weights, model["tables"]):
        score += weight * lookup_pair_residual(table, split)
    return score


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        sd = 1.0
    return (x - float(x.mean())) / sd


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, users))
    sorted_users = users[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(new_group)
    counts = np.diff(np.r_[starts, n])

    positions = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, counts)
    )
    repeated_counts = np.repeat(counts, counts)
    ranks = positions / np.maximum(repeated_counts - 1, 1)
    ranks[repeated_counts == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


train = load("train")
valid = load("valid")
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_metrics = evaluate(valid_users, valid_y, inc_valid)

models = {}
valid_raw = {}

models["video_residual_svd"] = fit_residual_svd(
    [train], "video_id", rank=RANK
)
valid_raw["video_residual_svd"] = predict_residual_svd(
    models["video_residual_svd"], valid
)

models["author_residual_svd"] = fit_residual_svd(
    [train], "author_id", rank=RANK
)
valid_raw["author_residual_svd"] = predict_residual_svd(
    models["author_residual_svd"], valid
)

models["positive_spectral"] = fit_positive_spectral(
    [train], rank=RANK
)
valid_raw["positive_spectral"] = predict_positive_spectral(
    models["positive_spectral"], valid
)

models["hierarchical_eb"] = fit_hierarchical_eb([train])
valid_raw["hierarchical_eb"] = predict_hierarchical_eb(
    models["hierarchical_eb"], valid
)

candidate_scores = {
    "incumbent": float(inc_metrics["primary"])
}
candidate_specs = {}

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_primary = float(inc_metrics["primary"])
best_spec = ("incumbent", 0.0, "raw")
best_own_raw = valid_raw["video_residual_svd"]

inc_z = zscore(inc_valid)
inc_rank = within_user_rank(valid_users, inc_valid)

for name, pred in valid_raw.items():
    met = evaluate(valid_users, valid_y, pred)
    candidate_scores[name] = float(met["primary"])
    candidate_specs[name] = (name, 1.0, "raw")

    if float(met["primary"]) > best_primary:
        best_primary = float(met["primary"])
        best_name = name
        best_scores = pred.copy()
        best_metrics = met
        best_spec = (name, 1.0, "raw")
        best_own_raw = pred

    pred_z = zscore(pred)
    pred_rank = within_user_rank(valid_users, pred)

    corr = float(np.corrcoef(inc_valid, pred)[0, 1])
    print(
        "FINDINGS %s incumbent_pearson=%.6f standalone=%.6f"
        % (name, corr, float(met["primary"]))
    )

    for weight in BLEND_WEIGHTS:
        z_blend = (1.0 - weight) * inc_z + weight * pred_z
        z_name = "%s_zblend_%.2f" % (name, weight)
        z_met = evaluate(valid_users, valid_y, z_blend)
        candidate_scores[z_name] = float(z_met["primary"])
        candidate_specs[z_name] = (name, float(weight), "z")
        if float(z_met["primary"]) > best_primary:
            best_primary = float(z_met["primary"])
            best_name = z_name
            best_scores = z_blend.copy()
            best_metrics = z_met
            best_spec = (name, float(weight), "z")
            best_own_raw = pred

        rank_blend = (
            (1.0 - weight) * inc_rank + weight * pred_rank
        )
        rank_name = "%s_rankblend_%.2f" % (name, weight)
        rank_met = evaluate(valid_users, valid_y, rank_blend)
        candidate_scores[rank_name] = float(rank_met["primary"])
        candidate_specs[rank_name] = (name, float(weight), "rank")
        if float(rank_met["primary"]) > best_primary:
            best_primary = float(rank_met["primary"])
            best_name = rank_name
            best_scores = rank_blend.copy()
            best_metrics = rank_met
            best_spec = (name, float(weight), "rank")
            best_own_raw = pred

print(
    "FINDINGS winner=%s model=%s weight=%.3f fusion=%s"
    % (best_name, best_spec[0], best_spec[1], best_spec[2])
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_spec[0] != "incumbent" or best_spec[2] != "raw":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_own_raw, dtype=np.float64),
        )

# Refit the selected recipe on train + validation, without inspecting test
# labels, and apply exactly the selected fusion type and weight.
test = load("test")
selected_model_name, selected_weight, selected_fusion = best_spec

if selected_model_name == "incumbent":
    if not os.path.exists(inc_test_path):
        raise RuntimeError("Trusted incumbent test predictions are missing")
    test_scores = np.load(inc_test_path).astype(np.float64)
else:
    fit_splits = [train, valid]

    if selected_model_name == "video_residual_svd":
        final_model = fit_residual_svd(
            fit_splits, "video_id", rank=RANK
        )
        own_test = predict_residual_svd(final_model, test)
    elif selected_model_name == "author_residual_svd":
        final_model = fit_residual_svd(
            fit_splits, "author_id", rank=RANK
        )
        own_test = predict_residual_svd(final_model, test)
    elif selected_model_name == "positive_spectral":
        final_model = fit_positive_spectral(
            fit_splits, rank=RANK
        )
        own_test = predict_positive_spectral(final_model, test)
    elif selected_model_name == "hierarchical_eb":
        final_model = fit_hierarchical_eb(fit_splits)
        own_test = predict_hierarchical_eb(final_model, test)
    else:
        raise RuntimeError("Unknown selected model: " + selected_model_name)

    if selected_fusion == "raw":
        test_scores = own_test
    else:
        if not os.path.exists(inc_test_path):
            raise RuntimeError("Trusted incumbent test predictions are missing")
        inc_test = np.load(inc_test_path).astype(np.float64)
        test_users = np.asarray(test.user_id, dtype=np.int64)

        if selected_fusion == "z":
            test_scores = (
                (1.0 - selected_weight) * zscore(inc_test)
                + selected_weight * zscore(own_test)
            )
        elif selected_fusion == "rank":
            test_scores = (
                (1.0 - selected_weight)
                * within_user_rank(test_users, inc_test)
                + selected_weight
                * within_user_rank(test_users, own_test)
            )
        else:
            raise RuntimeError("Unknown fusion: " + selected_fusion)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
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