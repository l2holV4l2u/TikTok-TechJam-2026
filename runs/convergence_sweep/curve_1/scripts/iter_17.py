import os
import time
import json
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
warnings.filterwarnings("ignore")

SEED = 271828
RANK = 28
HASH_SIZE = 1 << 20
SMOOTHING = 30.0
BLEND_WEIGHTS = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]

np.random.seed(SEED)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = positions[multi] / (
        repeated_lengths[multi].astype(np.float64) - 1.0
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def recency_weights(dates, half_life=3.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    positions = np.searchsorted(unique_dates, dates)
    ages = len(unique_dates) - 1 - positions
    weights = np.exp2(
        -ages.astype(np.float64) / float(half_life)
    )
    weights /= np.mean(weights)
    return weights.astype(np.float64)


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def ordered_training_arrays(train):
    n = len(train.user_id)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort(
        (rows, np.asarray(train.time_ms), np.asarray(train.user_id))
    )
    users = np.asarray(train.user_id, dtype=np.int64)[order]
    videos = np.asarray(train.video_id, dtype=np.int64)[order]
    labels = np.asarray(train.y, dtype=np.float64)[order]
    dates = np.asarray(train.date, dtype=np.int32)[order]

    user_card = FEATURE_CARDINALITIES["user_id"]
    last_video = np.full(user_card, -1, dtype=np.int64)
    last_label = np.zeros(user_card, dtype=np.float64)

    group_end = np.r_[users[1:] != users[:-1], True]
    ending_positions = np.flatnonzero(group_end)
    ending_users = users[ending_positions]
    valid = (
        (ending_users >= 0) &
        (ending_users < user_card)
    )
    last_video[ending_users[valid]] = videos[ending_positions[valid]]
    last_label[ending_users[valid]] = labels[ending_positions[valid]]

    return order, users, videos, labels, dates, last_video, last_label


def build_transition_graph(users, videos, labels, dates, n_video):
    edge_rows = []
    edge_cols = []
    edge_values = []
    date_weights = recency_weights(dates, half_life=3.0)

    for lag, lag_decay in [(1, 1.0), (2, 0.60), (3, 0.35), (4, 0.20)]:
        same = users[lag:] == users[:-lag]
        src = videos[:-lag][same]
        dst = videos[lag:][same]
        source_labels = labels[:-lag][same]
        target_labels = labels[lag:][same]
        rw = date_weights[lag:][same]

        # Positive endpoints define preference-bearing transitions, while
        # the small floor keeps the exposure graph connected.
        value = lag_decay * rw * (
            0.08 + 0.62 * source_labels + 0.30 * target_labels
        )

        edge_rows.append(src)
        edge_cols.append(dst)
        edge_values.append(value)

        edge_rows.append(dst)
        edge_cols.append(src)
        edge_values.append(0.65 * value)

    row = np.concatenate(edge_rows).astype(np.int32, copy=False)
    col = np.concatenate(edge_cols).astype(np.int32, copy=False)
    data = np.concatenate(edge_values).astype(np.float32, copy=False)

    graph = sparse.coo_matrix(
        (data, (row, col)),
        shape=(n_video, n_video),
        dtype=np.float32,
    ).tocsr()
    graph.sum_duplicates()
    graph.setdiag(0.0)
    graph.eliminate_zeros()

    # Positive-PMI normalization suppresses globally popular transitions.
    row_mass = np.asarray(graph.sum(axis=1)).ravel().astype(np.float64)
    col_mass = np.asarray(graph.sum(axis=0)).ravel().astype(np.float64)
    total = max(float(graph.data.sum()), 1.0)

    coo = graph.tocoo()
    denom = row_mass[coo.row] * col_mass[coo.col]
    pmi = np.log(
        np.maximum(
            coo.data.astype(np.float64) * total /
            np.maximum(denom, 1e-12),
            1e-12,
        )
    )
    keep = pmi > 0.0

    ppmi = sparse.coo_matrix(
        (
            pmi[keep].astype(np.float32),
            (coo.row[keep], coo.col[keep]),
        ),
        shape=graph.shape,
        dtype=np.float32,
    ).tocsr()
    ppmi.eliminate_zeros()
    return graph, ppmi


def spectral_item_embeddings(ppmi, rank):
    rank = min(rank, min(ppmi.shape) - 2)
    try:
        u, s, vt = svds(
            ppmi.astype(np.float64),
            k=rank,
            which="LM",
            random_state=SEED,
            maxiter=500,
            tol=1e-4,
        )
        order = np.argsort(s)[::-1]
        u = u[:, order]
        s = s[order]
        vt = vt[order]

        scale = np.sqrt(np.maximum(s, 1e-8))
        left = u * scale[None, :]
        right = vt.T * scale[None, :]
        emb = 0.5 * (left + right)
    except Exception as exc:
        print(
            "FINDINGS spectral_svd_fallback={}".format(type(exc).__name__),
            flush=True,
        )
        rng = np.random.RandomState(SEED)
        projection = rng.normal(
            0.0, 1.0 / np.sqrt(rank), size=(ppmi.shape[1], rank)
        )
        emb = ppmi.dot(projection)

    emb = np.asarray(emb, dtype=np.float64)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb /= np.maximum(norm, 1e-8)
    return emb.astype(np.float32)


def build_user_spectral_profiles(train, item_embeddings):
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float64)
    weights = recency_weights(train.date, half_life=3.0)

    n_user = FEATURE_CARDINALITIES["user_id"]
    dim = item_embeddings.shape[1]
    profiles = np.zeros((n_user, dim), dtype=np.float64)
    normalizer = np.zeros(n_user, dtype=np.float64)

    prior = np.sum(labels * weights) / np.sum(weights)
    signed = weights * (labels - prior)

    np.add.at(
        profiles,
        users,
        item_embeddings[videos].astype(np.float64) * signed[:, None],
    )
    np.add.at(normalizer, users, np.abs(signed))

    profiles /= np.maximum(normalizer[:, None], 1.0)
    profile_norm = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles /= np.maximum(profile_norm, 1e-8)
    return profiles.astype(np.float32)


def spectral_scores(split, item_embeddings, user_profiles):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    safe_users = np.clip(users, 0, len(user_profiles) - 1)
    scores = np.sum(
        user_profiles[safe_users].astype(np.float64) *
        item_embeddings[videos].astype(np.float64),
        axis=1,
    )
    invalid = (users < 0) | (users >= len(user_profiles))
    scores[invalid] = 0.0
    return scores


def transition_hash(previous_video, current_video, tab, n_hash=HASH_SIZE):
    previous_video = np.asarray(previous_video, dtype=np.uint64) + np.uint64(1)
    current_video = np.asarray(current_video, dtype=np.uint64) + np.uint64(1)
    tab = np.asarray(tab, dtype=np.uint64)

    value = (
        previous_video * np.uint64(11400714819323198485) ^
        current_video * np.uint64(14029467366897019727) ^
        tab * np.uint64(1609587929392839161)
    )
    return np.asarray(
        value & np.uint64(n_hash - 1),
        dtype=np.int64,
    )


def fit_markov(train, ordered):
    order, users, videos, labels, dates, _, _ = ordered
    tabs = np.asarray(train.X["tab"], dtype=np.int64)[order]
    weights = recency_weights(dates, half_life=2.5)

    same = users[1:] == users[:-1]
    previous = videos[:-1][same]
    current = videos[1:][same]
    current_tab = tabs[1:][same]
    target = labels[1:][same]
    w = weights[1:][same]

    keys = transition_hash(previous, current, current_tab)
    count = np.bincount(
        keys, weights=w, minlength=HASH_SIZE
    ).astype(np.float64)
    positive = np.bincount(
        keys, weights=w * target, minlength=HASH_SIZE
    ).astype(np.float64)
    prior = float(np.sum(w * target) / np.sum(w))
    return count, positive, prior


def markov_scores(split, last_video, markov):
    count, positive, prior = markov
    users = np.asarray(split.user_id, dtype=np.int64)
    current = np.asarray(split.video_id, dtype=np.int64)
    tabs = np.asarray(split.X["tab"], dtype=np.int64)

    safe_users = np.clip(users, 0, len(last_video) - 1)
    previous = last_video[safe_users]
    known = (
        (users >= 0) &
        (users < len(last_video)) &
        (previous >= 0)
    )
    previous_safe = np.maximum(previous, 0)
    keys = transition_hash(previous_safe, current, tabs)

    rate = (
        positive[keys] + 24.0 * prior
    ) / (
        count[keys] + 24.0
    )
    rate[~known] = prior
    return logit(rate)


def mix_hash(parts, n_hash=HASH_SIZE):
    key = np.full(
        len(parts[0]),
        np.uint64(1469598103934665603),
        dtype=np.uint64,
    )
    constants = [
        np.uint64(1099511628211),
        np.uint64(11400714819323198485),
        np.uint64(14029467366897019727),
        np.uint64(1609587929392839161),
    ]
    for i, part in enumerate(parts):
        x = np.asarray(part, dtype=np.uint64) + np.uint64(1)
        key ^= x * constants[i % len(constants)]
        key *= np.uint64(1099511628211)
    return np.asarray(
        key & np.uint64(n_hash - 1),
        dtype=np.int64,
    )


HAZARD_CONFIGS = [
    ("duration_bucket", "tab", "video_type"),
    ("duration_bucket", "tag", "upload_type"),
    ("duration_bucket", "user_active_degree", "tab"),
    ("video_id", "duration_bucket"),
    ("author_id", "duration_bucket", "tab"),
]


def split_key(split, fields):
    return mix_hash([split.X[field] for field in fields])


def fit_hazard_tables(train):
    labels = np.asarray(train.y, dtype=np.float64)
    weights = recency_weights(train.date, half_life=3.0)
    prior = float(np.sum(labels * weights) / np.sum(weights))
    tables = []

    for fields in HAZARD_CONFIGS:
        keys = split_key(train, fields)
        count = np.bincount(
            keys, weights=weights, minlength=HASH_SIZE
        ).astype(np.float32)
        positive = np.bincount(
            keys, weights=weights * labels, minlength=HASH_SIZE
        ).astype(np.float32)
        tables.append((fields, count, positive))

    return prior, tables


def hazard_scores(split, fitted):
    prior, tables = fitted
    base = float(logit(prior))
    accumulated = np.zeros(len(split.user_id), dtype=np.float64)
    evidence = np.zeros(len(split.user_id), dtype=np.float64)

    for fields, count, positive in tables:
        keys = split_key(split, fields)
        c = count[keys].astype(np.float64)
        p = positive[keys].astype(np.float64)
        rate = (p + SMOOTHING * prior) / (c + SMOOTHING)
        reliability = c / (c + SMOOTHING)
        accumulated += reliability * (logit(rate) - base)
        evidence += reliability

    return base + accumulated / np.maximum(evidence, 1.0)


def combine_raw_families(user_ids, arrays):
    ranks = [within_user_rank(user_ids, x) for x in arrays]
    return np.mean(np.vstack(ranks), axis=0)


def evaluate_score(valid, score):
    return evaluate(valid.user_id, valid.y, score)


train = load("train")
valid = load("valid")

n_video = FEATURE_CARDINALITIES["video_id"]
ordered = ordered_training_arrays(train)
_, ordered_users, ordered_videos, ordered_labels, ordered_dates, last_video, _ = ordered

graph, ppmi = build_transition_graph(
    ordered_users,
    ordered_videos,
    ordered_labels,
    ordered_dates,
    n_video,
)
item_embeddings = spectral_item_embeddings(ppmi, RANK)
user_profiles = build_user_spectral_profiles(train, item_embeddings)

markov_model = fit_markov(train, ordered)
hazard_model = fit_hazard_tables(train)

valid_spectral = spectral_scores(valid, item_embeddings, user_profiles)
valid_markov = markov_scores(valid, last_video, markov_model)
valid_hazard = hazard_scores(valid, hazard_model)

valid_spectral_markov = combine_raw_families(
    valid.user_id, [valid_spectral, valid_markov]
)
valid_all = combine_raw_families(
    valid.user_id, [valid_spectral, valid_markov, valid_hazard]
)

raw_valid_candidates = {
    "spectral_diffusion": valid_spectral,
    "markov_transition": valid_markov,
    "discrete_hazard": valid_hazard,
    "spectral_markov": valid_spectral_markov,
    "all_family_rank_ensemble": valid_all,
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_log = {}
best_primary = -np.inf
best_name = None
best_alpha = None
best_raw_name = None
best_valid_scores = None
best_metrics = None

inc_metrics = evaluate_score(valid, inc_valid)
candidate_log["incumbent"] = float(inc_metrics["primary"])

for raw_name, raw_score in raw_valid_candidates.items():
    raw_rank = within_user_rank(valid.user_id, raw_score)
    raw_metrics = evaluate_score(valid, raw_rank)
    candidate_log[raw_name] = float(raw_metrics["primary"])

    for alpha in BLEND_WEIGHTS:
        blended = (
            (1.0 - alpha) * inc_valid_rank +
            alpha * raw_rank
        )
        metrics = evaluate_score(valid, blended)
        name = "{}_blend_{:.2f}".format(raw_name, alpha)
        candidate_log[name] = float(metrics["primary"])

        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_name = name
            best_alpha = float(alpha)
            best_raw_name = raw_name
            best_valid_scores = blended.copy()
            best_metrics = metrics

print(
    "FINDINGS transition_graph_nnz={} ppmi_nnz={} spectral_rank={}".format(
        int(graph.nnz), int(ppmi.nnz), int(item_embeddings.shape[1])
    ),
    flush=True,
)
print(
    "FINDINGS selected={} raw_family={} alpha={:.2f}".format(
        best_name, best_raw_name, best_alpha
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_log, sort_keys=True),
    flush=True,
)

test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)
inc_test_rank = within_user_rank(test.user_id, inc_test)

test_spectral = spectral_scores(test, item_embeddings, user_profiles)
test_markov = markov_scores(test, last_video, markov_model)
test_hazard = hazard_scores(test, hazard_model)
test_spectral_markov = combine_raw_families(
    test.user_id, [test_spectral, test_markov]
)
test_all = combine_raw_families(
    test.user_id, [test_spectral, test_markov, test_hazard]
)

raw_test_candidates = {
    "spectral_diffusion": test_spectral,
    "markov_transition": test_markov,
    "discrete_hazard": test_hazard,
    "spectral_markov": test_spectral_markov,
    "all_family_rank_ensemble": test_all,
}

selected_raw_valid = raw_valid_candidates[best_raw_name]
selected_raw_test = raw_test_candidates[best_raw_name]
selected_test_rank = within_user_rank(test.user_id, selected_raw_test)
best_test_scores = (
    (1.0 - best_alpha) * inc_test_rank +
    best_alpha * selected_test_rank
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(selected_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)