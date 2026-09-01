import os
import gc
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from scipy.special import ndtri

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

np.random.seed(7319)
torch.manual_seed(7319)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
]

CARDS = np.asarray(
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
    dtype=np.int64,
)

FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}

PAIR_SPECS = [
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("user_id", "tab"),
    ("author_id", "tag"),
    ("video_id", "duration_bucket"),
    ("tag", "duration_bucket"),
    ("tab", "tag"),
    ("upload_type", "tag"),
]

HALF_LIFE_DAYS = 4.0
NB_SINGLE_SMOOTH = 120.0
NB_PAIR_SMOOTH = 80.0
BATCH_SIZE = 32768
WIDE_EPOCHS = 2


def categorical_matrix(split):
    columns = []
    for field, card in zip(FIELDS, CARDS):
        x = np.asarray(split.X[field], dtype=np.int64)
        x = np.where((x >= 0) & (x < card), x, 0)
        columns.append(x.astype(np.int32))
    return np.column_stack(columns).astype(np.int32, copy=False)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_position = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = (
        np.arange(n, dtype=np.float32)
        - start_position.astype(np.float32)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    group_sizes = np.diff(
        np.r_[-1, np.flatnonzero(ends)]
    ).astype(np.float32)

    group_number = np.cumsum(starts, dtype=np.int64) - 1
    denominator = np.maximum(
        group_sizes[group_number] - 1.0,
        1.0,
    )

    result = np.empty(n, dtype=np.float32)
    result[order] = local_rank / denominator
    return result


def copula(rank):
    p = np.clip(np.asarray(rank, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return ndtri(p).astype(np.float32)


def safe_logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def pair_ids(matrix, pair):
    left_name, right_name = pair
    left_index = FIELD_INDEX[left_name]
    right_index = FIELD_INDEX[right_name]
    right_card = int(CARDS[right_index])
    return (
        matrix[:, left_index].astype(np.int64) * right_card
        + matrix[:, right_index].astype(np.int64)
    )


def fit_table(index, cardinality, y, weights, global_rate, smoothing):
    count = np.bincount(
        index,
        weights=weights,
        minlength=cardinality,
    ).astype(np.float32)

    positive = np.bincount(
        index,
        weights=weights * y,
        minlength=cardinality,
    ).astype(np.float32)

    rate = (
        positive + np.float32(smoothing * global_rate)
    ) / (
        count + np.float32(smoothing)
    )
    rate = np.clip(rate, 1e-5, 1.0 - 1e-5).astype(np.float32)

    deviation = (
        safe_logit(rate) - safe_logit(global_rate)
    ).astype(np.float32)

    confidence = (
        count / (count + np.float32(smoothing))
    ).astype(np.float32)

    return {
        "rate": rate,
        "deviation": deviation,
        "confidence": confidence,
        "count": count,
    }


def fit_nonparametric_models(matrix, y, weights):
    global_rate = float(
        np.sum(weights * y, dtype=np.float64)
        / np.sum(weights, dtype=np.float64)
    )

    singles = []
    for column, card, name in zip(
        range(len(FIELDS)), CARDS, FIELDS
    ):
        table = fit_table(
            matrix[:, column].astype(np.int64),
            int(card),
            y,
            weights,
            global_rate,
            NB_SINGLE_SMOOTH,
        )
        singles.append(table)
        print(
            "FINDINGS fitted_single=%s occupied=%d cardinality=%d"
            % (
                name,
                int(np.count_nonzero(table["count"])),
                int(card),
            ),
            flush=True,
        )

    pairs = []
    for pair in PAIR_SPECS:
        right_card = int(CARDS[FIELD_INDEX[pair[1]]])
        cardinality = (
            int(CARDS[FIELD_INDEX[pair[0]]]) * right_card
        )
        ids = pair_ids(matrix, pair)
        table = fit_table(
            ids,
            cardinality,
            y,
            weights,
            global_rate,
            NB_PAIR_SMOOTH,
        )
        pairs.append(table)
        print(
            "FINDINGS fitted_pair=%s_x_%s occupied=%d cardinality=%d"
            % (
                pair[0],
                pair[1],
                int(np.count_nonzero(table["count"])),
                cardinality,
            ),
            flush=True,
        )
        del ids
        gc.collect()

    return {
        "global_rate": global_rate,
        "singles": singles,
        "pairs": pairs,
    }


def predict_likelihood_ratio(model, matrix):
    n = len(matrix)
    score = np.full(
        n,
        safe_logit(model["global_rate"]),
        dtype=np.float32,
    )

    # Generative evidence accumulation. Correlated identity fields are
    # tempered, while user/context fields retain full evidence.
    single_strength = {
        "user_id": 0.75,
        "video_id": 0.45,
        "author_id": 0.55,
        "tag": 0.70,
        "tab": 0.80,
        "duration_bucket": 0.65,
        "upload_type": 0.45,
    }

    for j, name in enumerate(FIELDS):
        ids = matrix[:, j]
        score += (
            single_strength[name]
            * model["singles"][j]["deviation"][ids]
        ).astype(np.float32)

    pair_strength = {
        ("user_id", "tag"): 0.65,
        ("user_id", "duration_bucket"): 0.75,
        ("user_id", "tab"): 0.55,
        ("author_id", "tag"): 0.45,
        ("video_id", "duration_bucket"): 0.50,
        ("tag", "duration_bucket"): 0.55,
        ("tab", "tag"): 0.55,
        ("upload_type", "tag"): 0.40,
    }

    for spec, table in zip(PAIR_SPECS, model["pairs"]):
        ids = pair_ids(matrix, spec)
        score += (
            pair_strength[spec]
            * table["deviation"][ids]
            * np.sqrt(table["confidence"][ids])
        ).astype(np.float32)

    return score


def predict_decision_mixture(model, matrix):
    n = len(matrix)
    global_rate = np.float32(model["global_rate"])

    numerator = np.full(n, 2.0 * global_rate, dtype=np.float32)
    denominator = np.full(n, 2.0, dtype=np.float32)

    # This model forms predictions as a reliability-weighted mixture of
    # posterior rates rather than adding likelihood evidence.
    for j, table in enumerate(model["singles"]):
        ids = matrix[:, j]
        confidence = table["confidence"][ids]
        weight = (0.25 + 1.75 * confidence).astype(np.float32)
        numerator += weight * table["rate"][ids]
        denominator += weight

    for spec, table in zip(PAIR_SPECS, model["pairs"]):
        ids = pair_ids(matrix, spec)
        confidence = table["confidence"][ids]
        weight = (2.5 * confidence).astype(np.float32)
        numerator += weight * table["rate"][ids]
        denominator += weight

    probability = np.clip(
        numerator / denominator,
        1e-5,
        1.0 - 1e-5,
    )
    return safe_logit(probability).astype(np.float32)


SINGLE_OFFSETS = np.r_[
    0,
    np.cumsum(CARDS[:-1]),
].astype(np.int64)
SINGLE_TOTAL = int(np.sum(CARDS))

PAIR_CARDINALITIES = []
for left, right in PAIR_SPECS:
    PAIR_CARDINALITIES.append(
        int(CARDS[FIELD_INDEX[left]])
        * int(CARDS[FIELD_INDEX[right]])
    )
PAIR_CARDINALITIES = np.asarray(PAIR_CARDINALITIES, dtype=np.int64)

PAIR_OFFSETS = (
    SINGLE_TOTAL
    + np.r_[0, np.cumsum(PAIR_CARDINALITIES[:-1])]
).astype(np.int64)

WIDE_TOTAL = int(
    SINGLE_TOTAL + np.sum(PAIR_CARDINALITIES)
)


def wide_feature_ids(matrix_slice):
    batch = len(matrix_slice)
    result = np.empty(
        (batch, len(FIELDS) + len(PAIR_SPECS)),
        dtype=np.int64,
    )

    for j in range(len(FIELDS)):
        result[:, j] = (
            matrix_slice[:, j].astype(np.int64)
            + SINGLE_OFFSETS[j]
        )

    for k, spec in enumerate(PAIR_SPECS):
        result[:, len(FIELDS) + k] = (
            pair_ids(matrix_slice, spec)
            + PAIR_OFFSETS[k]
        )

    return result


class MemorizedWideCross(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Embedding(
            WIDE_TOTAL,
            1,
            sparse=True,
        )
        nn.init.zeros_(self.weights.weight)

    def forward(self, feature_ids):
        return self.weights(feature_ids).sum(dim=1).squeeze(1)


def train_wide_cross(matrix, y, weights):
    model = MemorizedWideCross()
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=0.18,
        initial_accumulator_value=0.1,
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(9137)
    n = len(y)

    for epoch in range(WIDE_EPOCHS):
        permutation = rng.permutation(n)
        weighted_loss = 0.0
        total_weight = 0.0
        model.train()

        for start in range(0, n, BATCH_SIZE):
            index = permutation[start:start + BATCH_SIZE]
            feature_ids = wide_feature_ids(matrix[index])

            x_tensor = torch.from_numpy(feature_ids)
            y_tensor = torch.from_numpy(
                np.asarray(y[index], dtype=np.float32)
            )
            w_tensor = torch.from_numpy(
                np.asarray(weights[index], dtype=np.float32)
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor)
            element_loss = loss_function(logits, y_tensor)
            loss = (
                torch.sum(element_loss * w_tensor)
                / torch.sum(w_tensor).clamp_min(1.0)
            )
            loss.backward()
            optimizer.step()

            weighted_loss += float(
                torch.sum(element_loss * w_tensor).detach()
            )
            total_weight += float(torch.sum(w_tensor))

        print(
            "FINDINGS wide_cross_epoch=%d weighted_logloss=%.6f"
            % (
                epoch + 1,
                weighted_loss / max(total_weight, 1.0),
            ),
            flush=True,
        )

    return model


def predict_wide_cross(model, matrix):
    result = np.empty(len(matrix), dtype=np.float32)
    model.eval()

    with torch.no_grad():
        for start in range(0, len(matrix), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(matrix))
            feature_ids = wide_feature_ids(matrix[start:end])
            result[start:end] = (
                model(torch.from_numpy(feature_ids))
                .numpy()
                .astype(np.float32)
            )

    return result


inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)

if not (
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent artifacts are required")

train = load("train")
valid = load("valid")

train_matrix = categorical_matrix(train)
valid_matrix = categorical_matrix(valid)

train_y = np.asarray(train.y, dtype=np.float32)
train_date = np.asarray(train.date, dtype=np.int32)
latest_train_date = int(train_date.max())

age_days = (latest_train_date - train_date).astype(np.float32)
train_weights = np.exp(
    -np.log(2.0) * age_days / HALF_LIFE_DAYS
).astype(np.float32)
train_weights /= np.mean(train_weights, dtype=np.float64)

valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)

nonparametric_model = fit_nonparametric_models(
    train_matrix,
    train_y,
    train_weights,
)

wide_model = train_wide_cross(
    train_matrix,
    train_y,
    train_weights,
)

raw_valid = {
    "categorical_likelihood_ratio": predict_likelihood_ratio(
        nonparametric_model,
        valid_matrix,
    ),
    "hierarchical_decision_mixture": predict_decision_mixture(
        nonparametric_model,
        valid_matrix,
    ),
    "memorized_wide_cross": predict_wide_cross(
        wide_model,
        valid_matrix,
    ),
}

inc_valid_raw = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_valid_rank = within_user_rank(valid_uid, inc_valid_raw)
inc_valid_copula = copula(inc_valid_rank)

control_metrics = evaluate(
    valid_uid,
    valid_y,
    inc_valid_rank,
)

candidate_results = {
    "trusted_incumbent": float(control_metrics["primary"])
}

best_scores = inc_valid_rank.copy()
best_primary = float(control_metrics["primary"])
best_family = None
best_alpha = 0.0
best_transform = "rank"
best_own_rank = None

blend_alphas = [0.02, 0.04, 0.07, 0.10, 0.15, 0.22]

for family, raw in raw_valid.items():
    standalone_metrics = evaluate(valid_uid, valid_y, raw)
    candidate_results[
        family + "_standalone"
    ] = float(standalone_metrics["primary"])

    family_rank = within_user_rank(valid_uid, raw)
    family_copula = copula(family_rank)

    correlation = float(
        np.corrcoef(
            inc_valid_rank.astype(np.float64),
            family_rank.astype(np.float64),
        )[0, 1]
    )

    print(
        "FINDINGS family=%s standalone_primary=%.6f "
        "gauc=%.6f ndcg5=%.6f incumbent_rank_corr=%.6f"
        % (
            family,
            float(standalone_metrics["primary"]),
            float(standalone_metrics["gauc"]),
            float(standalone_metrics["ndcg@5"]),
            correlation,
        ),
        flush=True,
    )

    for transform_name, incumbent_base, family_base in [
        ("rank", inc_valid_rank, family_rank),
        ("copula", inc_valid_copula, family_copula),
    ]:
        for alpha in blend_alphas:
            blended = (
                (1.0 - alpha) * incumbent_base
                + alpha * family_base
            ).astype(np.float32)

            metrics = evaluate(valid_uid, valid_y, blended)
            candidate_name = "%s_%s_blend_%.2f" % (
                family,
                transform_name,
                alpha,
            )
            candidate_results[candidate_name] = float(
                metrics["primary"]
            )

            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_scores = blended.copy()
                best_family = family
                best_alpha = float(alpha)
                best_transform = transform_name
                best_own_rank = family_rank.copy()

final_metrics = evaluate(valid_uid, valid_y, best_scores)

print(
    "FINDINGS winner=%s transform=%s alpha=%.2f "
    "control=%.6f final=%.6f wide_parameters=%d"
    % (
        best_family if best_family is not None else "trusted_incumbent",
        best_transform,
        best_alpha,
        float(control_metrics["primary"]),
        float(final_metrics["primary"]),
        WIDE_TOTAL,
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_results, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_own_rank is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_own_rank, dtype=np.float64),
        )

del train
del valid
del train_matrix
del valid_matrix
del train_y
del train_date
del train_weights
del raw_valid
del inc_valid_raw
gc.collect()

test = load("test")
test_uid = np.asarray(test.user_id, dtype=np.int64)

inc_test_raw = np.asarray(
    np.load(inc_test_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_test_rank = within_user_rank(test_uid, inc_test_raw)

if best_family is None:
    test_scores = inc_test_rank
else:
    test_matrix = categorical_matrix(test)

    if best_family == "categorical_likelihood_ratio":
        own_test_raw = predict_likelihood_ratio(
            nonparametric_model,
            test_matrix,
        )
    elif best_family == "hierarchical_decision_mixture":
        own_test_raw = predict_decision_mixture(
            nonparametric_model,
            test_matrix,
        )
    elif best_family == "memorized_wide_cross":
        own_test_raw = predict_wide_cross(
            wide_model,
            test_matrix,
        )
    else:
        raise RuntimeError("Unknown selected family")

    own_test_rank = within_user_rank(test_uid, own_test_raw)

    if best_transform == "copula":
        incumbent_base = copula(inc_test_rank)
        own_base = copula(own_test_rank)
    else:
        incumbent_base = inc_test_rank
        own_base = own_test_rank

    test_scores = (
        (1.0 - best_alpha) * incumbent_base
        + best_alpha * own_base
    ).astype(np.float32)

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)