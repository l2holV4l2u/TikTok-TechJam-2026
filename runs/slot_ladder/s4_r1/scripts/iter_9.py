import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
BATCH = 8192

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
dates = np.asarray(train.date, dtype=np.int64)

# A fixed four-day half-life was motivated before this iteration by the
# measured date drift. It is applied to every supervised/statistical family.
recency_weight = np.exp2(
    (dates - dates.max()).astype(np.float32) / 4.0
)
recency_weight /= recency_weight.mean()
recency_weight = recency_weight.astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Deterministic row-position tie breaking.
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ranked_ordered = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked_ordered[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_ordered
    return result


# ----------------------------------------------------------------------
# Family 1: spectral propagation on recency-weighted positive graphs.
#
# This is not an FM trained on individual rows. It forms a sparse
# user-entity incidence graph, degree-normalizes it, and obtains a low-rank
# spectral reconstruction. A logged entity scores highly when it lies near
# the user's historical positive neighborhood in the graph.
# ----------------------------------------------------------------------
def fit_spectral_graph(entity_field, rank=48):
    user_card = int(FEATURE_CARDINALITIES["user_id"])
    entity_card = int(FEATURE_CARDINALITIES[entity_field])

    users = np.asarray(train.X["user_id"], dtype=np.int64)
    entities = np.asarray(train.X[entity_field], dtype=np.int64)
    positive = ytr > 0.5

    graph = sp.coo_matrix(
        (
            recency_weight[positive].astype(np.float64),
            (users[positive], entities[positive]),
        ),
        shape=(user_card, entity_card),
        dtype=np.float64,
    ).tocsr()
    graph.sum_duplicates()

    user_degree = np.asarray(graph.sum(axis=1)).ravel()
    entity_degree = np.asarray(graph.sum(axis=0)).ravel()

    user_scale = np.zeros_like(user_degree)
    entity_scale = np.zeros_like(entity_degree)
    np.power(
        np.maximum(user_degree, 1e-12),
        -0.5,
        out=user_scale,
    )
    np.power(
        np.maximum(entity_degree, 1e-12),
        -0.5,
        out=entity_scale,
    )
    user_scale[user_degree == 0] = 0.0
    entity_scale[entity_degree == 0] = 0.0

    normalized = (
        sp.diags(user_scale, format="csr")
        @ graph
        @ sp.diags(entity_scale, format="csr")
    )

    k = min(rank, min(normalized.shape) - 1)
    u, singular, vt = svds(
        normalized,
        k=k,
        which="LM",
        random_state=SEED,
    )

    # Sort from strongest to weakest component.
    ordering = np.argsort(singular)[::-1]
    singular = singular[ordering]
    u = u[:, ordering]
    vt = vt[ordering]

    root_s = np.sqrt(np.maximum(singular, 0.0))
    user_embedding = (
        u * root_s[None, :]
    ).astype(np.float32)
    entity_embedding = (
        vt.T * root_s[None, :]
    ).astype(np.float32)

    return user_embedding, entity_embedding


def spectral_predict(split, entity_field, user_embedding, entity_embedding):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    entities = np.asarray(split.X[entity_field], dtype=np.int64)

    valid_ids = (
        (users >= 0)
        & (users < len(user_embedding))
        & (entities >= 0)
        & (entities < len(entity_embedding))
    )
    result = np.zeros(len(users), dtype=np.float64)

    for lo in range(0, len(users), 65536):
        hi = min(lo + 65536, len(users))
        mask = valid_ids[lo:hi]
        if not np.any(mask):
            continue

        local_users = users[lo:hi][mask]
        local_entities = entities[lo:hi][mask]
        local_scores = np.einsum(
            "ij,ij->i",
            user_embedding[local_users],
            entity_embedding[local_entities],
            optimize=True,
        )
        chunk = result[lo:hi]
        chunk[mask] = local_scores.astype(np.float64)

    return result


video_user_emb, video_entity_emb = fit_spectral_graph(
    "video_id", rank=56
)
author_user_emb, author_entity_emb = fit_spectral_graph(
    "author_id", rank=40
)

graph_video_valid = spectral_predict(
    valid, "video_id", video_user_emb, video_entity_emb
)
graph_video_test = spectral_predict(
    test, "video_id", video_user_emb, video_entity_emb
)
graph_author_valid = spectral_predict(
    valid, "author_id", author_user_emb, author_entity_emb
)
graph_author_test = spectral_predict(
    test, "author_id", author_user_emb, author_entity_emb
)

graph_hetero_valid = (
    0.65 * within_user_rank(valid.user_id, graph_video_valid)
    + 0.35 * within_user_rank(valid.user_id, graph_author_valid)
)
graph_hetero_test = (
    0.65 * within_user_rank(test.user_id, graph_video_test)
    + 0.35 * within_user_rank(test.user_id, graph_author_test)
)

del video_user_emb, video_entity_emb
del author_user_emb, author_entity_emb


# ----------------------------------------------------------------------
# Family 2: wide logistic regression with explicit categorical crosses.
#
# Unlike an FM, prediction is a sum of independently learned marginal and
# explicit cross coefficients. The selected crosses directly memorize
# user-context and entity-context deviations without learning a universal
# low-rank interaction geometry.
# ----------------------------------------------------------------------
WIDE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "register_days_bucket",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
]

CROSSES = [
    ("user_id", "tab"),
    ("user_id", "duration_bucket"),
    ("video_id", "tab"),
    ("video_id", "tag"),
    ("author_id", "tab"),
    ("author_id", "tag"),
]


wide_cards = [
    int(FEATURE_CARDINALITIES[field]) for field in WIDE_FIELDS
]
cross_cards = [
    int(FEATURE_CARDINALITIES[a])
    * int(FEATURE_CARDINALITIES[b])
    for a, b in CROSSES
]
all_wide_cards = wide_cards + cross_cards
wide_offsets = np.cumsum(
    np.r_[np.int64(0), np.asarray(all_wide_cards[:-1], dtype=np.int64)]
)
wide_total_cardinality = int(sum(all_wide_cards))


def make_wide_matrix(split):
    columns = []

    for field, card in zip(WIDE_FIELDS, wide_cards):
        x = np.asarray(split.X[field], dtype=np.int64)
        if x.size and (x.min() < 0 or x.max() >= card):
            raise ValueError("Out-of-range ID in " + field)
        columns.append(x)

    for (field_a, field_b), cross_card in zip(CROSSES, cross_cards):
        a = np.asarray(split.X[field_a], dtype=np.int64)
        b = np.asarray(split.X[field_b], dtype=np.int64)
        card_b = int(FEATURE_CARDINALITIES[field_b])
        crossed = a * card_b + b
        if crossed.size and (
            crossed.min() < 0 or crossed.max() >= cross_card
        ):
            raise ValueError("Out-of-range cross")
        columns.append(crossed)

    matrix = np.stack(columns, axis=1).astype(np.int64, copy=False)
    matrix += wide_offsets[None, :]
    return np.ascontiguousarray(matrix)


xtr_wide = make_wide_matrix(train)
xva_wide = make_wide_matrix(valid)
xte_wide = make_wide_matrix(test)


class WideCrossLogistic(nn.Module):
    def __init__(self, cardinality):
        super().__init__()
        self.coefficient = nn.Embedding(cardinality, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.coefficient.weight)

    def forward(self, x):
        return (
            self.bias
            + self.coefficient(x).squeeze(-1).sum(dim=1)
        )


wide_model = WideCrossLogistic(wide_total_cardinality)
wide_optimizer = torch.optim.AdamW(
    wide_model.parameters(),
    lr=0.012,
    weight_decay=1e-6,
)
rng = np.random.default_rng(SEED + 1)

for _ in range(4):
    order = rng.permutation(len(ytr))
    wide_model.train()

    for lo in range(0, len(order), BATCH):
        idx = order[lo:lo + BATCH]
        xb = torch.from_numpy(xtr_wide[idx])
        target = torch.from_numpy(ytr[idx])
        weight = torch.from_numpy(recency_weight[idx])

        wide_optimizer.zero_grad(set_to_none=True)
        logits = wide_model(xb)
        row_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        loss = (row_loss * weight).sum() / weight.sum()
        loss.backward()
        wide_optimizer.step()


@torch.inference_mode()
def predict_wide(matrix):
    wide_model.eval()
    result = np.empty(len(matrix), dtype=np.float64)
    for lo in range(0, len(matrix), 32768):
        hi = min(lo + 32768, len(matrix))
        logits = wide_model(torch.from_numpy(matrix[lo:hi]))
        result[lo:hi] = logits.cpu().numpy().astype(np.float64)
    return result


wide_valid = predict_wide(xva_wide)
wide_test = predict_wide(xte_wide)

del xtr_wide, xva_wide, xte_wide
del wide_model, wide_optimizer


# ----------------------------------------------------------------------
# Family 3: categorical Naive Bayes.
#
# Prediction is a generative log-likelihood ratio rather than a learned
# discriminative interaction. Dirichlet smoothing makes sparse identities
# back off toward stable field-level evidence.
# ----------------------------------------------------------------------
NB_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "register_days_bucket",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "music_type",
]

nb_log_ratios = {}
positive_weight = recency_weight * ytr
negative_weight = recency_weight * (1.0 - ytr)
total_positive = float(positive_weight.sum())
total_negative = float(negative_weight.sum())
alpha = 1.5

for field in NB_FIELDS:
    card = int(FEATURE_CARDINALITIES[field])
    ids = np.asarray(train.X[field], dtype=np.int64)

    positive_count = np.bincount(
        ids,
        weights=positive_weight,
        minlength=card,
    ).astype(np.float64)
    negative_count = np.bincount(
        ids,
        weights=negative_weight,
        minlength=card,
    ).astype(np.float64)

    positive_log_prob = np.log(
        (positive_count + alpha)
        / (total_positive + alpha * card)
    )
    negative_log_prob = np.log(
        (negative_count + alpha)
        / (total_negative + alpha * card)
    )
    nb_log_ratios[field] = positive_log_prob - negative_log_prob


def predict_naive_bayes(split):
    prior = np.log(
        (total_positive + alpha)
        / (total_negative + alpha)
    )
    result = np.full(len(split.user_id), prior, dtype=np.float64)

    # Averaging correlated identity fields reduces the extreme confidence
    # caused by Naive Bayes' conditional-independence approximation.
    scale = 1.0 / np.sqrt(len(NB_FIELDS))
    for field in NB_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += scale * nb_log_ratios[field][ids]
    return result


nb_valid = predict_naive_bayes(valid)
nb_test = predict_naive_bayes(test)


# ----------------------------------------------------------------------
# Compare each raw family and fixed rank aggregations with the trusted
# incumbent. Blend weights are permitted to be selected on public
# validation and are transferred unchanged to test.
# ----------------------------------------------------------------------
shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

families = {
    "spectral_video": (graph_video_valid, graph_video_test),
    "spectral_heterogeneous": (
        graph_hetero_valid, graph_hetero_test
    ),
    "wide_cross_logistic": (wide_valid, wide_test),
    "categorical_naive_bayes": (nb_valid, nb_test),
}

candidate_scores = {}
candidate_valid = {}
candidate_test = {}
candidate_raw = {}

blend_alphas = [0.15, 0.30, 0.50, 0.70, 1.00]

for family_name, (raw_valid, raw_test) in families.items():
    raw_valid = np.asarray(raw_valid, dtype=np.float64)
    raw_test = np.asarray(raw_test, dtype=np.float64)

    own_valid_rank = within_user_rank(valid.user_id, raw_valid)
    own_test_rank = within_user_rank(test.user_id, raw_test)

    raw_metrics = evaluate(valid.user_id, valid.y, raw_valid)
    candidate_scores[family_name + "_raw"] = float(
        raw_metrics["primary"]
    )
    candidate_valid[family_name + "_raw"] = raw_valid
    candidate_test[family_name + "_raw"] = raw_test
    candidate_raw[family_name + "_raw"] = raw_valid

    for blend_alpha in blend_alphas:
        if blend_alpha == 1.0:
            blended_valid = own_valid_rank
            blended_test = own_test_rank
        else:
            blended_valid = (
                (1.0 - blend_alpha) * inc_valid_rank
                + blend_alpha * own_valid_rank
            )
            blended_test = (
                (1.0 - blend_alpha) * inc_test_rank
                + blend_alpha * own_test_rank
            )

        name = "%s_blend_%.2f" % (family_name, blend_alpha)
        result = evaluate(
            valid.user_id, valid.y, blended_valid
        )
        candidate_scores[name] = float(result["primary"])
        candidate_valid[name] = blended_valid
        candidate_test[name] = blended_test
        candidate_raw[name] = raw_valid


# Also test a three-family consensus before blending with the incumbent.
consensus_valid = (
    within_user_rank(valid.user_id, graph_hetero_valid)
    + within_user_rank(valid.user_id, wide_valid)
    + within_user_rank(valid.user_id, nb_valid)
) / 3.0
consensus_test = (
    within_user_rank(test.user_id, graph_hetero_test)
    + within_user_rank(test.user_id, wide_test)
    + within_user_rank(test.user_id, nb_test)
) / 3.0

for blend_alpha in blend_alphas:
    if blend_alpha == 1.0:
        blended_valid = consensus_valid
        blended_test = consensus_test
    else:
        blended_valid = (
            (1.0 - blend_alpha) * inc_valid_rank
            + blend_alpha * consensus_valid
        )
        blended_test = (
            (1.0 - blend_alpha) * inc_test_rank
            + blend_alpha * consensus_test
        )

    name = "three_family_consensus_blend_%.2f" % blend_alpha
    result = evaluate(valid.user_id, valid.y, blended_valid)
    candidate_scores[name] = float(result["primary"])
    candidate_valid[name] = blended_valid
    candidate_test[name] = blended_test
    candidate_raw[name] = consensus_valid


winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(candidate_valid[winner], dtype=np.float64)
test_scores = np.asarray(candidate_test[winner], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores,
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        test_scores,
    )

    # Every candidate involving the incumbent receives the corresponding
    # new-family score here; harmlessly save it for raw candidates as well.
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(candidate_raw[winner], dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS winner=%s raw_graph_video=%.6f raw_graph_hetero=%.6f "
    "raw_wide=%.6f raw_nb=%.6f"
    % (
        winner,
        candidate_scores["spectral_video_raw"],
        candidate_scores["spectral_heterogeneous_raw"],
        candidate_scores["wide_cross_logistic_raw"],
        candidate_scores["categorical_naive_bayes_raw"],
    )
)
print(
    "METRICS "
    + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(time.time() - START),
    })
)