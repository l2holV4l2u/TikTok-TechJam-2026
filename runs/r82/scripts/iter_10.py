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
SEED = 73129
SVD_DIM = 48
MARKOV_ALPHA = 20.0
SEQ_FIELDS = ["video_id", "author_id", "tag"]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_users[1:] != sorted_users[:-1]

    starts = np.maximum.accumulate(
        np.where(boundary, np.arange(n, dtype=np.int64), 0)
    )
    positions = np.arange(n, dtype=np.int64) - starts

    _, counts = np.unique(sorted_users, return_counts=True)
    denom = np.repeat(np.maximum(counts - 1, 1), counts)

    result = np.empty(n, dtype=np.float32)
    result[order] = (positions / denom).astype(np.float32)
    return result


def fit_positive_svd(user_ids, video_ids, labels, recency_weights=None):
    users = np.asarray(user_ids, dtype=np.int64)
    videos = np.asarray(video_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)

    positive = labels == 1
    pu = users[positive]
    pv = videos[positive]

    if recency_weights is None:
        values = np.ones(len(pu), dtype=np.float32)
    else:
        values = np.asarray(recency_weights, dtype=np.float32)[positive]

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    matrix = sparse.coo_matrix(
        (values, (pu, pv)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    # Damp repeated positive exposures before factorization.
    matrix.data = np.log1p(matrix.data).astype(np.float32)

    k = min(SVD_DIM, min(matrix.shape) - 1)
    u, singular, vt = svds(
        matrix,
        k=k,
        which="LM",
        return_singular_vectors=True,
        random_state=SEED,
        tol=1e-3,
        maxiter=500,
    )
    descending = np.argsort(singular)[::-1]
    singular = singular[descending].astype(np.float32)
    u = u[:, descending].astype(np.float32)
    vt = vt[descending].astype(np.float32)

    user_factors = u * singular[None, :]
    item_factors = vt.T.copy()
    return user_factors, item_factors


def svd_predict(user_factors, item_factors, user_ids, video_ids):
    users = np.asarray(user_ids, dtype=np.int64)
    videos = np.asarray(video_ids, dtype=np.int64)
    return np.einsum(
        "ij,ij->i",
        user_factors[users],
        item_factors[videos],
        optimize=True,
    ).astype(np.float32)


class MarkovLift:
    def __init__(
        self,
        field,
        user_ids,
        time_ms,
        labels,
        values,
        row_weights=None,
        alpha=20.0,
    ):
        self.field = field
        self.cardinality = int(FEATURE_CARDINALITIES[field])
        self.alpha = float(alpha)

        users = np.asarray(user_ids, dtype=np.int64)
        times = np.asarray(time_ms, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int8)
        values = np.asarray(values, dtype=np.int64)
        n_users = int(FEATURE_CARDINALITIES["user_id"])

        if row_weights is None:
            weights = np.ones(len(users), dtype=np.float64)
        else:
            weights = np.asarray(row_weights, dtype=np.float64)

        positive_rows = np.flatnonzero(labels == 1)
        row_tie = positive_rows.astype(np.int64)
        order_local = np.lexsort(
            (
                row_tie,
                times[positive_rows],
                users[positive_rows],
            )
        )
        ordered_rows = positive_rows[order_local]
        ordered_users = users[ordered_rows]
        ordered_values = values[ordered_rows]
        ordered_weights = weights[ordered_rows]

        self.last_state = np.zeros(n_users, dtype=np.int64)
        if len(ordered_rows):
            ends = np.empty(len(ordered_rows), dtype=bool)
            ends[:-1] = ordered_users[:-1] != ordered_users[1:]
            ends[-1] = True
            self.last_state[ordered_users[ends]] = ordered_values[ends]

        destination_weight = np.bincount(
            values[positive_rows],
            weights=weights[positive_rows],
            minlength=self.cardinality,
        ).astype(np.float64)
        destination_weight += 1e-3
        self.destination_prob = (
            destination_weight / destination_weight.sum()
        )

        if len(ordered_rows) < 2:
            self.pair_keys = np.empty(0, dtype=np.int64)
            self.pair_counts = np.empty(0, dtype=np.float64)
            self.source_totals = np.zeros(
                self.cardinality, dtype=np.float64
            )
            return

        adjacent = ordered_users[:-1] == ordered_users[1:]
        source = ordered_values[:-1][adjacent]
        destination = ordered_values[1:][adjacent]
        transition_weights = ordered_weights[1:][adjacent]

        pair_key = (
            source.astype(np.int64) * self.cardinality
            + destination.astype(np.int64)
        )
        unique_keys, inverse = np.unique(
            pair_key, return_inverse=True
        )
        pair_counts = np.bincount(
            inverse,
            weights=transition_weights,
            minlength=len(unique_keys),
        ).astype(np.float64)

        self.pair_keys = unique_keys.astype(np.int64)
        self.pair_counts = pair_counts
        self.source_totals = np.bincount(
            source,
            weights=transition_weights,
            minlength=self.cardinality,
        ).astype(np.float64)

    def predict(self, user_ids, candidate_values):
        users = np.asarray(user_ids, dtype=np.int64)
        candidates = np.asarray(candidate_values, dtype=np.int64)
        source = self.last_state[users]

        query = source * self.cardinality + candidates
        positions = np.searchsorted(self.pair_keys, query)
        present = positions < len(self.pair_keys)
        safe_positions = np.minimum(
            positions, max(len(self.pair_keys) - 1, 0)
        )

        counts = np.zeros(len(query), dtype=np.float64)
        if len(self.pair_keys):
            exact = present & (
                self.pair_keys[safe_positions] == query
            )
            counts[exact] = self.pair_counts[
                safe_positions[exact]
            ]

        base = np.maximum(
            self.destination_prob[candidates], 1e-12
        )
        numerator = counts + self.alpha * base
        denominator = self.source_totals[source] + self.alpha

        # Conditional transition log-probability minus global destination
        # log-probability. Unobserved transitions therefore have neutral lift.
        score = (
            np.log(np.maximum(numerator, 1e-12))
            - np.log(np.maximum(denominator, 1e-12))
            - np.log(base)
        )
        return score.astype(np.float32)


def fit_markov_models(
    user_ids, time_ms, labels, field_arrays, row_weights=None
):
    models = {}
    for field in SEQ_FIELDS:
        models[field] = MarkovLift(
            field=field,
            user_ids=user_ids,
            time_ms=time_ms,
            labels=labels,
            values=field_arrays[field],
            row_weights=row_weights,
            alpha=MARKOV_ALPHA,
        )
    return models


def markov_predictions(models, split):
    result = {}
    for field, model in models.items():
        result[field] = model.predict(
            split.user_id, split.X[field]
        )
    return result


def score_candidate(valid, y_valid, scores):
    return float(
        evaluate(valid.user_id, y_valid, scores)["primary"]
    )


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float32, copy=False)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

# Family 1: implicit collaborative latent reconstruction.
svd_user, svd_item = fit_positive_svd(
    train.user_id, train.video_id, y_train
)
svd_valid_raw = svd_predict(
    svd_user,
    svd_item,
    valid.user_id,
    valid.video_id,
)
svd_valid_rank = within_user_rank(
    valid.user_id, svd_valid_raw
)

# Family 2: chronological transition lift from the last positive state.
train_fields = {
    f: np.asarray(train.X[f], dtype=np.int64)
    for f in SEQ_FIELDS
}
markov_models = fit_markov_models(
    train.user_id,
    train.time_ms,
    y_train,
    train_fields,
)
markov_raw = markov_predictions(markov_models, valid)
markov_ranks = {
    f: within_user_rank(valid.user_id, p)
    for f, p in markov_raw.items()
}
sequence_valid_rank = np.mean(
    np.stack(
        [
            markov_ranks["video_id"],
            markov_ranks["author_id"],
            markov_ranks["tag"],
        ],
        axis=0,
    ),
    axis=0,
).astype(np.float32)

# Cross-family aggregation is itself materially different from either model:
# collaborative affinity plus immediate sequential intent.
latent_sequence_rank = (
    0.5 * svd_valid_rank + 0.5 * sequence_valid_rank
).astype(np.float32)

base_candidates = {
    "svd_latent": svd_valid_rank,
    "markov_video": markov_ranks["video_id"],
    "markov_author": markov_ranks["author_id"],
    "markov_tag": markov_ranks["tag"],
    "markov_multistate": sequence_valid_rank,
    "svd_plus_markov": latent_sequence_rank,
}

candidate_log = {
    "trusted_incumbent": score_candidate(
        valid, y_valid, inc_valid_rank
    )
}

winner_primary = candidate_log["trusted_incumbent"]
winner_name = "trusted_incumbent"
winner_alpha = 0.0
winner_scores = inc_valid_rank.copy()

blend_grid = np.linspace(0.0, 1.0, 11)

for name, candidate_rank in base_candidates.items():
    standalone = score_candidate(
        valid, y_valid, candidate_rank
    )
    candidate_log[name] = standalone

    best_blend = -np.inf
    best_alpha = 0.0
    best_scores = None
    for alpha in blend_grid:
        blended = (
            (1.0 - float(alpha)) * inc_valid_rank
            + float(alpha) * candidate_rank
        )
        primary = score_candidate(valid, y_valid, blended)
        if primary > best_blend:
            best_blend = primary
            best_alpha = float(alpha)
            best_scores = blended.copy()

    candidate_log[name + "_inc_blend"] = float(best_blend)

    if best_blend > winner_primary:
        winner_primary = float(best_blend)
        winner_name = name
        winner_alpha = best_alpha
        winner_scores = best_scores

valid_scores = np.asarray(winner_scores, dtype=np.float32)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": winner_name,
            "selected_incumbent_weight": 1.0 - winner_alpha,
            "selected_new_family_weight": winner_alpha,
            "svd_dimension": SVD_DIM,
            "markov_alpha": MARKOV_ALPHA,
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical selected recipe on train + validation, then score test.
test = load("test")
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float32, copy=False)
inc_test_rank = within_user_rank(test.user_id, inc_test)

if winner_name == "trusted_incumbent" or winner_alpha <= 0.0:
    test_scores = inc_test
else:
    combined_users = np.concatenate(
        [
            np.asarray(train.user_id, dtype=np.int64),
            np.asarray(valid.user_id, dtype=np.int64),
        ]
    )
    combined_videos = np.concatenate(
        [
            np.asarray(train.video_id, dtype=np.int64),
            np.asarray(valid.video_id, dtype=np.int64),
        ]
    )
    combined_times = np.concatenate(
        [
            np.asarray(train.time_ms, dtype=np.int64),
            np.asarray(valid.time_ms, dtype=np.int64),
        ]
    )
    combined_labels = np.concatenate(
        [y_train, y_valid]
    )

    need_svd = winner_name in {
        "svd_latent",
        "svd_plus_markov",
    }
    need_markov = winner_name in {
        "markov_video",
        "markov_author",
        "markov_tag",
        "markov_multistate",
        "svd_plus_markov",
    }

    svd_test_rank = None
    sequence_test_rank = None
    markov_test_ranks = {}

    if need_svd:
        del svd_user, svd_item
        gc.collect()
        full_user, full_item = fit_positive_svd(
            combined_users,
            combined_videos,
            combined_labels,
        )
        svd_test_raw = svd_predict(
            full_user,
            full_item,
            test.user_id,
            test.video_id,
        )
        svd_test_rank = within_user_rank(
            test.user_id, svd_test_raw
        )

    if need_markov:
        combined_fields = {}
        for field in SEQ_FIELDS:
            combined_fields[field] = np.concatenate(
                [
                    np.asarray(train.X[field], dtype=np.int64),
                    np.asarray(valid.X[field], dtype=np.int64),
                ]
            )

        full_markov = fit_markov_models(
            combined_users,
            combined_times,
            combined_labels,
            combined_fields,
        )
        full_markov_raw = markov_predictions(
            full_markov, test
        )
        markov_test_ranks = {
            f: within_user_rank(test.user_id, p)
            for f, p in full_markov_raw.items()
        }
        sequence_test_rank = np.mean(
            np.stack(
                [
                    markov_test_ranks["video_id"],
                    markov_test_ranks["author_id"],
                    markov_test_ranks["tag"],
                ],
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)

    if winner_name == "svd_latent":
        selected_test_rank = svd_test_rank
    elif winner_name == "markov_video":
        selected_test_rank = markov_test_ranks["video_id"]
    elif winner_name == "markov_author":
        selected_test_rank = markov_test_ranks["author_id"]
    elif winner_name == "markov_tag":
        selected_test_rank = markov_test_ranks["tag"]
    elif winner_name == "markov_multistate":
        selected_test_rank = sequence_test_rank
    elif winner_name == "svd_plus_markov":
        selected_test_rank = (
            0.5 * svd_test_rank
            + 0.5 * sequence_test_rank
        ).astype(np.float32)
    else:
        raise ValueError("Unknown selected family: " + winner_name)

    test_scores = (
        (1.0 - winner_alpha) * inc_test_rank
        + winner_alpha * selected_test_rank
    ).astype(np.float32)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)