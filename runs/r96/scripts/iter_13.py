import os
import time
import json
import gc
import numpy as np
from scipy import sparse

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 27183
np.random.seed(SEED)


def rank_percentile(user_ids, scores):
    """Tie-aware percentile ranks independently within each user."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    order = np.lexsort((scores, user_ids))
    su = user_ids[order]
    ss = scores[order]

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = su[1:] != su[:-1]
    user_start_idx = np.maximum.accumulate(
        np.where(user_start, np.arange(n, dtype=np.int64), 0)
    )

    user_end = np.empty(n, dtype=bool)
    user_end[-1] = True
    user_end[:-1] = su[:-1] != su[1:]
    user_end_idx = np.flatnonzero(user_end)
    user_sizes = np.diff(
        np.concatenate((np.array([-1], dtype=np.int64), user_end_idx))
    )
    row_user_sizes = np.repeat(user_sizes, user_sizes)

    tie_start = np.empty(n, dtype=bool)
    tie_start[0] = True
    tie_start[1:] = (
        (su[1:] != su[:-1])
        | (ss[1:] != ss[:-1])
    )
    tie_starts = np.flatnonzero(tie_start)

    tie_end = np.empty(n, dtype=bool)
    tie_end[-1] = True
    tie_end[:-1] = (
        (su[:-1] != su[1:])
        | (ss[:-1] != ss[1:])
    )
    tie_ends = np.flatnonzero(tie_end)
    tie_lengths = tie_ends - tie_starts + 1
    tie_midpoints = 0.5 * (tie_starts + tie_ends)
    row_midpoints = np.repeat(tie_midpoints, tie_lengths)

    within_user_midpoint = row_midpoints - user_start_idx
    ranked = (within_user_midpoint + 0.5) / row_user_sizes

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = np.maximum(int(dates.max()) - dates, 0).astype(np.float64)
    return np.power(0.5, age / float(half_life))


def aggregate_sparse(rows, cols, values, shape):
    matrix = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
            ),
        ),
        shape=shape,
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def normalize_sparse(matrix, symmetric_columns=True):
    matrix = matrix.tocsr().astype(np.float32)
    row_sq = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    row_scale = 1.0 / np.sqrt(np.maximum(row_sq, 1e-8))
    result = sparse.diags(row_scale.astype(np.float32)) @ matrix

    if symmetric_columns:
        col_sq = np.asarray(result.multiply(result).sum(axis=0)).ravel()
        col_scale = 1.0 / np.sqrt(np.maximum(col_sq, 1e-8))
        result = result @ sparse.diags(col_scale.astype(np.float32))

    return result.tocsr()


def prune_rows(matrix, topk=120, by_absolute=False, positive_only=False):
    """Keep only the strongest top-k entries per item row."""
    matrix = matrix.tocsr()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()

    out_indices = []
    out_values = []
    out_indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int64)

    for row in range(matrix.shape[0]):
        lo, hi = matrix.indptr[row], matrix.indptr[row + 1]
        idx = matrix.indices[lo:hi]
        val = matrix.data[lo:hi]

        if positive_only and len(val):
            keep_positive = val > 0
            idx = idx[keep_positive]
            val = val[keep_positive]

        if len(val) > topk:
            criterion = np.abs(val) if by_absolute else val
            selected = np.argpartition(criterion, -topk)[-topk:]
            idx = idx[selected]
            val = val[selected]

        if len(val):
            order = np.argsort(idx)
            idx = idx[order]
            val = val[order]
            out_indices.append(idx.astype(np.int32, copy=False))
            out_values.append(val.astype(np.float32, copy=False))

        out_indptr[row + 1] = out_indptr[row] + len(val)

    if out_indices:
        indices = np.concatenate(out_indices)
        values = np.concatenate(out_values)
    else:
        indices = np.empty(0, dtype=np.int32)
        values = np.empty(0, dtype=np.float32)

    return sparse.csr_matrix(
        (values, indices, out_indptr),
        shape=matrix.shape,
        dtype=np.float32,
    )


def fit_positive_graph(train, half_life=4.0, topk=140):
    n_users = FEATURE_CARDINALITIES["user_id"]
    n_videos = FEATURE_CARDINALITIES["video_id"]

    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    weights = recency_weights(train.date, half_life)

    positive = labels == 1
    interaction = aggregate_sparse(
        users[positive],
        videos[positive],
        weights[positive],
        (n_users, n_videos),
    )
    profile = normalize_sparse(interaction, symmetric_columns=True)

    graph = (profile.T @ profile).tocsr()
    graph = prune_rows(
        graph, topk=topk, by_absolute=False, positive_only=True
    )
    return profile, graph


def fit_signed_graph(train, half_life=4.0, topk=120):
    n_users = FEATURE_CARDINALITIES["user_id"]
    n_videos = FEATURE_CARDINALITIES["video_id"]

    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float64)
    weights = recency_weights(train.date, half_life)

    # Remove each video's broad attractiveness so the graph connects videos
    # through users who liked or disliked them unusually strongly.
    denominator = np.bincount(
        videos, weights=weights, minlength=n_videos
    ).astype(np.float64)
    numerator = np.bincount(
        videos, weights=weights * labels, minlength=n_videos
    ).astype(np.float64)
    global_prior = float(np.sum(weights * labels) / np.sum(weights))
    video_rate = (
        numerator + 25.0 * global_prior
    ) / (denominator + 25.0)

    residual = weights * (labels - video_rate[videos])
    interaction = aggregate_sparse(
        users,
        videos,
        residual,
        (n_users, n_videos),
    )
    profile = normalize_sparse(interaction, symmetric_columns=False)

    graph = (profile.T @ profile).tocsr()
    graph = prune_rows(
        graph, topk=topk, by_absolute=True, positive_only=False
    )
    return profile, graph


def graph_predict(split, profile, graph):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    n = len(users)
    scores = np.zeros(n, dtype=np.float64)

    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    n_users = profile.shape[0]
    block_size = 512

    for start in range(0, n_users, block_size):
        stop = min(start + block_size, n_users)
        lo = np.searchsorted(sorted_users, start, side="left")
        hi = np.searchsorted(sorted_users, stop, side="left")
        if hi <= lo:
            continue

        rows = order[lo:hi]
        propagated = profile[start:stop] @ graph
        local_users = users[rows] - start
        selected = propagated[local_users, videos[rows]]
        scores[rows] = selected.A1.astype(np.float64)

    return scores


def previous_context_for_train(train, field):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    values = np.asarray(train.X[field], dtype=np.int64)
    times = np.asarray(train.time_ms, dtype=np.int64)
    row_id = np.arange(len(train), dtype=np.int64)

    order = np.lexsort((row_id, times, users))
    su = users[order]
    sv = values[order]

    previous_sorted = np.zeros(len(train), dtype=np.int64)
    same = su[1:] == su[:-1]
    previous_sorted[1:][same] = sv[:-1][same]

    previous = np.empty(len(train), dtype=np.int64)
    previous[order] = previous_sorted
    return previous, order


def last_train_context(train, order, field):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    values = np.asarray(train.X[field], dtype=np.int64)
    n_users = FEATURE_CARDINALITIES["user_id"]

    su = users[order]
    user_end = np.empty(len(order), dtype=bool)
    user_end[-1] = True
    user_end[:-1] = su[:-1] != su[1:]
    end_rows = order[user_end]

    last = np.zeros(n_users, dtype=np.int64)
    last[users[end_rows]] = values[end_rows]
    return last


def previous_context_for_split(split, field, last_train):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    values = np.asarray(split.X[field], dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    row_id = np.arange(len(split), dtype=np.int64)

    order = np.lexsort((row_id, times, users))
    su = users[order]
    sv = values[order]

    previous_sorted = np.empty(len(split), dtype=np.int64)
    first = np.empty(len(split), dtype=bool)
    first[0] = True
    first[1:] = su[1:] != su[:-1]

    previous_sorted[first] = last_train[su[first]]
    not_first = ~first
    previous_sorted[not_first] = sv[:-1][not_first[1:]]

    previous = np.empty(len(split), dtype=np.int64)
    previous[order] = previous_sorted
    return previous


def fit_rate_map(keys, labels, weights, prior_values, smoothing):
    keys = np.asarray(keys, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    prior_values = np.asarray(prior_values, dtype=np.float64)

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    denominator = np.bincount(inverse, weights=weights)
    numerator = np.bincount(inverse, weights=weights * labels)
    prior_sum = np.bincount(
        inverse, weights=prior_values
    )
    prior_mean = prior_sum / np.maximum(
        np.bincount(inverse), 1
    )
    rates = (
        numerator + smoothing * prior_mean
    ) / (denominator + smoothing)
    return unique_keys, rates


def lookup(keys, unique_keys, values, default_values):
    keys = np.asarray(keys, dtype=np.int64)
    output = np.asarray(default_values, dtype=np.float64).copy()
    positions = np.searchsorted(unique_keys, keys)
    valid = positions < len(unique_keys)
    valid_rows = np.flatnonzero(valid)
    if len(valid_rows):
        matched = (
            unique_keys[positions[valid_rows]]
            == keys[valid_rows]
        )
        rows = valid_rows[matched]
        output[rows] = values[positions[rows]]
    return output


def fit_markov_model(train, half_life=4.0):
    fields = ["video_id", "author_id", "tag"]
    field_weights = {
        "video_id": 0.45,
        "author_id": 0.35,
        "tag": 0.20,
    }
    smoothings = {
        "video_id": 7.0,
        "author_id": 14.0,
        "tag": 35.0,
    }

    labels = np.asarray(train.y, dtype=np.float64)
    weights = recency_weights(train.date, half_life)
    global_prior = float(np.sum(weights * labels) / np.sum(weights))

    model = {
        "fields": {},
        "field_weights": field_weights,
        "global_prior": global_prior,
    }

    common_order = None
    for field in fields:
        current = np.asarray(train.X[field], dtype=np.int64)
        card = FEATURE_CARDINALITIES[field]

        denominator = np.bincount(
            current, weights=weights, minlength=card
        ).astype(np.float64)
        numerator = np.bincount(
            current, weights=weights * labels, minlength=card
        ).astype(np.float64)
        entity_rate = (
            numerator + 25.0 * global_prior
        ) / (denominator + 25.0)

        previous, order = previous_context_for_train(train, field)
        if common_order is None:
            common_order = order

        # Zero is the unknown/start context and is retained as a backoff state.
        pair_key = previous * np.int64(card) + current
        pair_prior = entity_rate[current]
        keys, rates = fit_rate_map(
            pair_key,
            labels,
            weights,
            pair_prior,
            smoothings[field],
        )

        last_train = last_train_context(train, order, field)
        model["fields"][field] = {
            "card": card,
            "entity_rate": entity_rate,
            "pair_keys": keys,
            "pair_rates": rates,
            "last_train": last_train,
        }

    return model


def predict_markov(model, split):
    total = np.zeros(len(split), dtype=np.float64)
    total_weight = 0.0

    for field, field_model in model["fields"].items():
        weight = model["field_weights"][field]
        current = np.asarray(split.X[field], dtype=np.int64)
        previous = previous_context_for_split(
            split, field, field_model["last_train"]
        )
        base = field_model["entity_rate"][current]
        pair_key = (
            previous * np.int64(field_model["card"]) + current
        )
        pair_rate = lookup(
            pair_key,
            field_model["pair_keys"],
            field_model["pair_rates"],
            base,
        )
        total += weight * logit(pair_rate)
        total_weight += weight

    return total / total_weight


train = load("train")
valid = load("valid")
test = load("test")

# Family 1: positive-only collaborative graph diffusion.
positive_profile, positive_graph = fit_positive_graph(
    train, half_life=4.0, topk=140
)
positive_valid = graph_predict(valid, positive_profile, positive_graph)
positive_test = graph_predict(test, positive_profile, positive_graph)

del positive_graph
gc.collect()

# Family 2: signed residual graph, where dislikes create negative edges.
signed_profile, signed_graph = fit_signed_graph(
    train, half_life=4.0, topk=120
)
signed_valid = graph_predict(valid, signed_profile, signed_graph)
signed_test = graph_predict(test, signed_profile, signed_graph)

del signed_graph
gc.collect()

# Family 3: chronological transition model using preceding logged impressions.
markov_model = fit_markov_model(train, half_life=4.0)
markov_valid = predict_markov(markov_model, valid)
markov_test = predict_markov(markov_model, test)

raw_valid = {
    "positive_graph_diffusion": positive_valid,
    "signed_residual_graph": signed_valid,
    "chronological_markov": markov_valid,
}
raw_test = {
    "positive_graph_diffusion": positive_test,
    "signed_residual_graph": signed_test,
    "chronological_markov": markov_test,
}

valid_ranks = {
    name: rank_percentile(valid.user_id, scores)
    for name, scores in raw_valid.items()
}
test_ranks = {
    name: rank_percentile(test.user_id, scores)
    for name, scores in raw_test.items()
}

# Cross-family graph/sequence ensembles.
ensemble_specs = {
    "positive_graph_plus_markov": {
        "positive_graph_diffusion": 0.60,
        "chronological_markov": 0.40,
    },
    "signed_graph_plus_markov": {
        "signed_residual_graph": 0.55,
        "chronological_markov": 0.45,
    },
    "all_graph_sequence": {
        "positive_graph_diffusion": 0.45,
        "signed_residual_graph": 0.25,
        "chronological_markov": 0.30,
    },
}

ensemble_valid = {}
ensemble_test = {}
for ensemble_name, specification in ensemble_specs.items():
    va = np.zeros(len(valid), dtype=np.float64)
    te = np.zeros(len(test), dtype=np.float64)
    for component, weight in specification.items():
        va += weight * valid_ranks[component]
        te += weight * test_ranks[component]
    ensemble_valid[ensemble_name] = va
    ensemble_test[ensemble_name] = te

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

all_own_valid = dict(raw_valid)
all_own_valid.update(ensemble_valid)
all_own_test = dict(raw_test)
all_own_test.update(ensemble_test)

candidate_valid = {
    "incumbent": inc_valid,
}
candidate_test = {
    "incumbent": inc_test,
}
candidate_raw = {
    "incumbent": inc_valid,
}

for name in all_own_valid:
    candidate_valid[name + "_standalone"] = all_own_valid[name]
    candidate_test[name + "_standalone"] = all_own_test[name]
    candidate_raw[name + "_standalone"] = all_own_valid[name]

    own_va_rank = rank_percentile(valid.user_id, all_own_valid[name])
    own_te_rank = rank_percentile(test.user_id, all_own_test[name])

    for alpha in (0.10, 0.20, 0.35, 0.50, 0.70):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_va_rank
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * own_te_rank
        )
        candidate_raw[key] = all_own_valid[name]

candidate_metrics = {}
for name, scores in candidate_valid.items():
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, scores
    )

best_key = max(
    candidate_metrics,
    key=lambda key: float(candidate_metrics[key]["primary"]),
)
best_metrics = candidate_metrics[best_key]
best_valid = candidate_valid[best_key]
best_test = candidate_test[best_key]

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_key,
    "raw_rank_correlations_with_incumbent": {
        name: float(np.corrcoef(
            inc_valid_rank,
            rank_percentile(valid.user_id, values)
        )[0, 1])
        for name, values in raw_valid.items()
    },
    "cross_family_rank_correlations": {
        "positive_vs_signed": float(np.corrcoef(
            valid_ranks["positive_graph_diffusion"],
            valid_ranks["signed_residual_graph"],
        )[0, 1]),
        "positive_vs_markov": float(np.corrcoef(
            valid_ranks["positive_graph_diffusion"],
            valid_ranks["chronological_markov"],
        )[0, 1]),
        "signed_vs_markov": float(np.corrcoef(
            valid_ranks["signed_residual_graph"],
            valid_ranks["chronological_markov"],
        )[0, 1]),
    },
    "positive_graph_nnz": int(positive_profile.nnz),
    "signed_graph_profile_nnz": int(signed_profile.nnz),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_key != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[best_key], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))