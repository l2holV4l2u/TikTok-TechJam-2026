import os
import time
import json
import gc
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
np.random.seed(20260831)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float64)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

# Item/content nodes used in each session hyperedge. User identity is excluded:
# its evaluation distribution is unusually sparse and temporally unstable.
FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "upload_type",
    "duration_bucket",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

tr_values = [
    np.asarray(train.X[f], dtype=np.int64) for f in FIELDS
]
va_values = [
    np.asarray(valid.X[f], dtype=np.int64) for f in FIELDS
]
te_values = [
    np.asarray(test.X[f], dtype=np.int64) for f in FIELDS
]
cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]

# A hyperedge is an actual feed batch: impressions sharing both user and exact
# timestamp. Row position only breaks ordering ties and is not part of the key.
tr_user = np.asarray(train.user_id, dtype=np.int64)
tr_time = np.asarray(train.time_ms, dtype=np.int64)
rows = np.arange(len(ytr), dtype=np.int64)
order = np.lexsort((rows, tr_time, tr_user))

sorted_user = tr_user[order]
sorted_time = tr_time[order]
new_session = np.r_[
    True,
    (sorted_user[1:] != sorted_user[:-1])
    | (sorted_time[1:] != sorted_time[:-1]),
]
sid_sorted = np.cumsum(new_session, dtype=np.int64) - 1
session_id = np.empty(len(ytr), dtype=np.int64)
session_id[order] = sid_sorted
n_sessions = int(sid_sorted[-1]) + 1

session_size = np.bincount(
    session_id, minlength=n_sessions
).astype(np.float64)
singleton_share = float(np.mean(session_size == 1))
multirow_share = float(
    session_size[session_size > 1].sum() / len(ytr)
)
print(
    "FINDINGS sessions=%d singleton_session_share=%.6f "
    "rows_in_multirow_sessions=%.6f max_session=%d"
    % (
        n_sessions,
        singleton_share,
        multirow_share,
        int(session_size.max()),
    )
)

dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
day_index = np.searchsorted(unique_dates, dates)
age = (len(unique_dates) - 1 - day_index).astype(np.float64)


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    row_id = np.arange(len(scores), dtype=np.int64)

    idx = np.lexsort((row_id, scores, users))
    su = users[idx]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], len(idx)]
    lengths = ends - starts

    pos = (
        np.arange(len(idx), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    den = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked = pos / den

    result = np.empty(len(scores), dtype=np.float64)
    result[idx] = ranked
    return result


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1.0e-5, 1.0 - 1.0e-5)
    return np.log(p) - np.log1p(-p)


def row_mean_from_tables(values, tables):
    out = np.zeros(len(values[0]), dtype=np.float64)
    for x, table in zip(values, tables):
        out += table[x]
    out /= float(len(values))
    return out


def aggregate_nodes(signal, row_weight, prior, smoothing):
    """Aggregate a row signal into each categorical node with shrinkage."""
    signal = np.asarray(signal, dtype=np.float64)
    row_weight = np.asarray(row_weight, dtype=np.float64)
    tables = []

    for x, card in zip(tr_values, cards):
        denominator = np.bincount(
            x, weights=row_weight, minlength=card
        ).astype(np.float64)
        numerator = np.bincount(
            x, weights=row_weight * signal, minlength=card
        ).astype(np.float64)
        table = (
            numerator + smoothing * prior
        ) / np.maximum(denominator + smoothing, 1.0e-12)

        # ID zero represents unknown. Giving it the prior makes validation and
        # test behavior explicit even when it occurred rarely in train.
        table[0] = prior
        tables.append(table)
    return tables


def aggregate_residual_nodes(signal, row_weight, smoothing):
    signal = np.asarray(signal, dtype=np.float64)
    row_weight = np.asarray(row_weight, dtype=np.float64)
    tables = []

    for x, card in zip(tr_values, cards):
        denominator = np.bincount(
            x, weights=row_weight, minlength=card
        ).astype(np.float64)
        numerator = np.bincount(
            x, weights=row_weight * signal, minlength=card
        ).astype(np.float64)
        table = numerator / np.maximum(
            denominator + smoothing, 1.0e-12
        )
        table[0] = 0.0
        tables.append(table)
    return tables


def session_average(row_signal, row_weight):
    numerator = np.bincount(
        session_id,
        weights=row_weight * row_signal,
        minlength=n_sessions,
    ).astype(np.float64)
    denominator = np.bincount(
        session_id,
        weights=row_weight,
        minlength=n_sessions,
    ).astype(np.float64)
    return numerator / np.maximum(denominator, 1.0e-12)


def predict_tables(values, tables, use_logit=True):
    if use_logit:
        transformed = [logit(t) for t in tables]
    else:
        transformed = tables
    return row_mean_from_tables(values, transformed)


families_valid = {}
families_test = {}

# This is a train-only temporal sweep. It changes the training distribution,
# not merely model capacity. Uniform, moderate, and aggressive decay test
# whether session relations themselves remain stable across the date gap.
weight_specs = {
    "uniform": np.ones(len(ytr), dtype=np.float64),
    "half8": np.exp2(-age / 8.0),
    "half4": np.exp2(-age / 4.0),
    "half2": np.exp2(-age / 2.0),
}

for weight_name, row_weight in weight_specs.items():
    row_weight = row_weight / max(float(row_weight.mean()), 1.0e-12)
    prior = float(np.sum(row_weight * ytr) / np.sum(row_weight))

    # ------------------------------------------------------------------
    # Family 1: independent empirical-Bayes entity potentials. This is the
    # no-hyperedge control needed to measure whether sessions add anything.
    # ------------------------------------------------------------------
    direct_tables = aggregate_nodes(
        ytr, row_weight, prior=prior, smoothing=35.0
    )
    direct_train_logit = row_mean_from_tables(
        tr_values, [logit(t) for t in direct_tables]
    )
    direct_train_prob = sigmoid(direct_train_logit)

    direct_valid = predict_tables(
        va_values, direct_tables, use_logit=True
    )
    direct_test = predict_tables(
        te_values, direct_tables, use_logit=True
    )

    base_name = "direct_entity_" + weight_name
    families_valid[base_name] = within_user_rank(uva, direct_valid)
    families_test[base_name] = within_user_rank(ute, direct_test)

    # Canonical hypergraph normalization: each feed batch emits one message,
    # so a large batch does not dominate merely because it has many rows.
    edge_equal_weight = row_weight / np.maximum(
        session_size[session_id], 1.0
    )

    # ------------------------------------------------------------------
    # Family 2: supervised hyperedge pooling.
    #
    # First estimate each session's long-view rate, then transfer that common
    # session signal to every entity node incident on the hyperedge.
    # ------------------------------------------------------------------
    session_label = session_average(ytr, row_weight)
    row_session_label = session_label[session_id]
    supervised_tables = aggregate_nodes(
        row_session_label,
        edge_equal_weight,
        prior=prior,
        smoothing=18.0,
    )
    supervised_valid = predict_tables(
        va_values, supervised_tables, use_logit=True
    )
    supervised_test = predict_tables(
        te_values, supervised_tables, use_logit=True
    )

    name = "supervised_session_hypergraph_" + weight_name
    families_valid[name] = within_user_rank(uva, supervised_valid)
    families_test[name] = within_user_rank(ute, supervised_test)

    # ------------------------------------------------------------------
    # Family 3: unsupervised random-walk diffusion of supervised node seeds.
    #
    # A node sends its direct target estimate to sessions, sessions pool all
    # incident nodes, and the pooled message returns to nodes. Restart keeps
    # identity-specific evidence while one and two hops capture recurrent
    # session-level co-exposure structure.
    # ------------------------------------------------------------------
    walk_tables = [t.copy() for t in direct_tables]
    restart = 0.62

    walk_predictions = []
    for walk_step in range(2):
        node_row_prob = row_mean_from_tables(tr_values, walk_tables)
        edge_message = session_average(node_row_prob, row_weight)
        propagated_tables = aggregate_nodes(
            edge_message[session_id],
            edge_equal_weight,
            prior=prior,
            smoothing=22.0,
        )
        walk_tables = [
            restart * seed + (1.0 - restart) * propagated
            for seed, propagated in zip(
                direct_tables, propagated_tables
            )
        ]
        walk_predictions.append(
            (
                predict_tables(va_values, walk_tables, use_logit=True),
                predict_tables(te_values, walk_tables, use_logit=True),
            )
        )

    for step, (walk_valid, walk_test) in enumerate(
        walk_predictions, start=1
    ):
        name = "session_random_walk%d_%s" % (step, weight_name)
        families_valid[name] = within_user_rank(uva, walk_valid)
        families_test[name] = within_user_rank(ute, walk_test)

    # ------------------------------------------------------------------
    # Family 4: residual hypergraph correction.
    #
    # Unlike pooling or diffusion, this estimates the error of independent
    # entity potentials at the session level. Nodes repeatedly occurring in
    # unexpectedly positive/negative sessions receive a signed correction.
    # ------------------------------------------------------------------
    session_observed = session_average(ytr, row_weight)
    session_expected = session_average(
        direct_train_prob, row_weight
    )
    session_residual = session_observed - session_expected

    residual_tables = aggregate_residual_nodes(
        session_residual[session_id],
        edge_equal_weight,
        smoothing=12.0,
    )

    residual_train = row_mean_from_tables(
        tr_values, residual_tables
    )
    residual_scale = float(
        np.std(direct_train_logit)
        / max(np.std(residual_train), 1.0e-6)
    )
    residual_scale = min(residual_scale, 3.0)

    va_residual = row_mean_from_tables(va_values, residual_tables)
    te_residual = row_mean_from_tables(te_values, residual_tables)

    for strength in (0.35, 0.70):
        residual_valid = (
            direct_valid
            + strength * residual_scale * va_residual
        )
        residual_test = (
            direct_test
            + strength * residual_scale * te_residual
        )
        name = "session_residual%.2f_%s" % (
            strength,
            weight_name,
        )
        families_valid[name] = within_user_rank(
            uva, residual_valid
        )
        families_test[name] = within_user_rank(
            ute, residual_test
        )

    del (
        direct_tables,
        supervised_tables,
        walk_tables,
        propagated_tables,
        residual_tables,
    )
    gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
inc_rank_valid = within_user_rank(uva, inc_valid)
inc_rank_test = within_user_rank(ute, inc_test)

# The trusted-incumbent blend is explicitly allowed. Rank normalization makes
# alpha comparable across families while preserving all relevant order.
alphas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw = None
best_name = None

for name, own_valid in families_valid.items():
    own_test = families_test[name]
    standalone = evaluate(uva, yva, own_valid)
    candidate_scores[name + "_standalone"] = float(
        standalone["primary"]
    )

    corr = float(np.corrcoef(inc_rank_valid, own_valid)[0, 1])
    print(
        "FINDINGS family=%s standalone=%.6f incumbent_corr=%.6f"
        % (name, float(standalone["primary"]), corr)
    )

    for alpha in alphas:
        blended_valid = (
            (1.0 - alpha) * inc_rank_valid
            + alpha * own_valid
        )
        metrics = evaluate(uva, yva, blended_valid)
        primary = float(metrics["primary"])

        candidate_name = "%s_blend_%.2f" % (name, alpha)
        candidate_scores[candidate_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = blended_valid.copy()
            best_test = (
                (1.0 - alpha) * inc_rank_test
                + alpha * own_test
            )
            best_raw = own_valid.copy()
            best_name = candidate_name

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s primary=%.6f"
    % (best_name, best_primary)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)