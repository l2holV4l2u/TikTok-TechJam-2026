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
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

THREADS = max(1, min(12, os.cpu_count() or 1))
torch.set_num_threads(THREADS)
torch.manual_seed(7319)
np.random.seed(7319)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
]

CARDS = np.asarray(
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
    dtype=np.int64,
)
OFFSETS = np.r_[0, np.cumsum(CARDS[:-1])].astype(np.int64)
TOTAL_CARD = int(np.sum(CARDS))

FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}

CROSS_PAIRS = [
    ("user_id", "tag"),
    ("user_id", "tab"),
    ("user_id", "duration_bucket"),
    ("user_id", "upload_type"),
    ("user_id", "onehot_feat3"),
    ("user_id", "onehot_feat8"),
    ("author_id", "tag"),
    ("author_id", "tab"),
    ("video_id", "tab"),
    ("tag", "duration_bucket"),
]

HASH_BITS = 21
HASH_SIZE = 1 << HASH_BITS
HASH_MASK = HASH_SIZE - 1

BATCH_SIZE = 32768
EPOCHS = 2
HALF_LIFE_DAYS = 4.0
FM_DIM = 10


def categorical_matrix(split):
    cols = []
    for field, card in zip(FIELDS, CARDS):
        x = np.asarray(split.X[field], dtype=np.int64)
        x = np.where((x >= 0) & (x < card), x, 0)
        cols.append(x.astype(np.int32))
    return np.column_stack(cols).astype(np.int32, copy=False)


def offset_categorical(raw_cat):
    return (
        raw_cat.astype(np.int64) + OFFSETS.reshape(1, -1)
    ).astype(np.int32)


def hashed_cross_matrix(raw_cat):
    result = np.empty(
        (len(raw_cat), len(CROSS_PAIRS)),
        dtype=np.int32,
    )

    for k, (left_name, right_name) in enumerate(CROSS_PAIRS):
        left = raw_cat[:, FIELD_INDEX[left_name]].astype(np.uint64)
        right = raw_cat[:, FIELD_INDEX[right_name]].astype(np.uint64)

        salt = np.uint64(
            (k + 1) * 2654435761
        )
        value = (
            left * np.uint64(1000003)
            + right * np.uint64(9176)
            + salt
        )
        value ^= value >> np.uint64(17)
        value *= np.uint64(2246822519)
        value ^= value >> np.uint64(13)

        result[:, k] = (
            TOTAL_CARD
            + (value & np.uint64(HASH_MASK)).astype(np.int64)
        ).astype(np.int32)

    return result


def selected_history(split_name):
    arrays = []
    names = []
    suffixes = (
        "train_count_log1p",
        "long_view_rate",
        "is_click_rate",
        "play_time_ms_logmean",
        "comment_stay_time_logmean",
    )

    for entity in ("video_id", "author_id"):
        history = historical_features(split_name, key=entity)
        for name in sorted(history):
            if any(name.endswith(s) for s in suffixes):
                arrays.append(
                    np.asarray(history[name], dtype=np.float32)
                )
                names.append(name)

    if not arrays:
        raise RuntimeError("Historical features unavailable")

    return np.column_stack(arrays).astype(np.float32), names


def raw_numeric(split):
    arrays = []
    for name in (
        "duration_ms",
        "user_fans_user_num",
        "user_follow_user_num",
        "user_friend_user_num",
        "user_register_days",
    ):
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        arrays.append(x)
    return np.column_stack(arrays).astype(np.float32)


def fit_numeric_transform(history, raw):
    x = np.column_stack([history, raw]).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)
    return mean, std


def transform_numeric(history, raw, mean, std):
    x = np.column_stack([history, raw]).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - mean) / std
    return np.clip(x, -8.0, 8.0).astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = (
        np.arange(n, dtype=np.float32)
        - start_positions.astype(np.float32)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    group_sizes = np.diff(
        np.r_[-1, np.flatnonzero(ends)]
    ).astype(np.float32)

    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denominator = np.maximum(
        group_sizes[group_index] - 1.0,
        1.0,
    )

    result = np.empty(n, dtype=np.float32)
    result[order] = local_rank / denominator
    return result


def copula_score(rank):
    rank = np.asarray(rank, dtype=np.float64)
    probability = np.clip(rank, 1e-4, 1.0 - 1e-4)
    return ndtri(probability).astype(np.float32)


class HashedConjunctionWide(nn.Module):
    """
    Linear model over singleton categorical identities and explicit hashed
    pair identities. Unlike an FM, each selected conjunction has an
    independently estimated coefficient and is not constrained to low rank.
    """
    def __init__(self, numeric_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            TOTAL_CARD + HASH_SIZE,
            1,
            sparse=True,
        )
        self.numeric = nn.Linear(numeric_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.embedding.weight)
        nn.init.zeros_(self.numeric.weight)
        nn.init.zeros_(self.numeric.bias)

    def forward(self, singleton_ids, cross_ids, numeric):
        all_ids = torch.cat([singleton_ids, cross_ids], dim=1)
        wide = self.embedding(all_ids).sum(dim=1).squeeze(1)
        numeric_term = self.numeric(numeric).squeeze(1)
        return wide + numeric_term + self.bias


class FactorizationMachine(nn.Module):
    """
    Standard low-rank FM over the same singleton fields, plus a linear
    historical/numeric tower. It smooths pair effects through shared latent
    factors rather than assigning an independent coefficient to each cross.
    """
    def __init__(self, numeric_dim):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.embedding = nn.Embedding(TOTAL_CARD, FM_DIM)
        self.numeric = nn.Linear(numeric_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.numeric.weight)
        nn.init.zeros_(self.numeric.bias)

    def forward(self, singleton_ids, numeric):
        linear_term = self.linear(
            singleton_ids
        ).sum(dim=1).squeeze(1)

        vectors = self.embedding(singleton_ids)
        summed = vectors.sum(dim=1)
        interaction = 0.5 * (
            summed.square()
            - vectors.square().sum(dim=1)
        ).sum(dim=1)

        numeric_term = self.numeric(numeric).squeeze(1)
        return linear_term + interaction + numeric_term + self.bias


def train_hash_model(
    model,
    singleton,
    crosses,
    numeric,
    labels,
    weights,
    seed,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    sparse_parameters = [model.embedding.weight]
    dense_parameters = [
        model.numeric.weight,
        model.numeric.bias,
        model.bias,
    ]

    sparse_optimizer = torch.optim.SparseAdam(
        sparse_parameters,
        lr=0.025,
    )
    dense_optimizer = torch.optim.AdamW(
        dense_parameters,
        lr=0.004,
        weight_decay=2e-6,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    n = len(labels)
    model.train()

    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]

            single_t = torch.from_numpy(
                np.asarray(singleton[idx], dtype=np.int64)
            )
            cross_t = torch.from_numpy(
                np.asarray(crosses[idx], dtype=np.int64)
            )
            numeric_t = torch.from_numpy(
                np.asarray(numeric[idx], dtype=np.float32)
            )
            label_t = torch.from_numpy(
                np.asarray(labels[idx], dtype=np.float32)
            )
            weight_t = torch.from_numpy(
                np.asarray(weights[idx], dtype=np.float32)
            )

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            logits = model(single_t, cross_t, numeric_t)
            losses = criterion(logits, label_t)
            loss = torch.sum(losses * weight_t) / torch.sum(
                weight_t
            ).clamp_min(1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dense_parameters,
                5.0,
            )
            sparse_optimizer.step()
            dense_optimizer.step()

            loss_sum += float(
                torch.sum(losses * weight_t).detach()
            )
            weight_sum += float(torch.sum(weight_t))

        print(
            "FINDINGS family=hashed_conjunction epoch=%d "
            "weighted_logloss=%.6f"
            % (
                epoch + 1,
                loss_sum / max(weight_sum, 1.0),
            ),
            flush=True,
        )

    return model


def train_fm_model(
    model,
    singleton,
    numeric,
    labels,
    weights,
    seed,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=3e-6,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    n = len(labels)
    model.train()

    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]

            single_t = torch.from_numpy(
                np.asarray(singleton[idx], dtype=np.int64)
            )
            numeric_t = torch.from_numpy(
                np.asarray(numeric[idx], dtype=np.float32)
            )
            label_t = torch.from_numpy(
                np.asarray(labels[idx], dtype=np.float32)
            )
            weight_t = torch.from_numpy(
                np.asarray(weights[idx], dtype=np.float32)
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(single_t, numeric_t)
            losses = criterion(logits, label_t)
            loss = torch.sum(losses * weight_t) / torch.sum(
                weight_t
            ).clamp_min(1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )
            optimizer.step()

            loss_sum += float(
                torch.sum(losses * weight_t).detach()
            )
            weight_sum += float(torch.sum(weight_t))

        print(
            "FINDINGS family=factorization_machine epoch=%d "
            "weighted_logloss=%.6f"
            % (
                epoch + 1,
                loss_sum / max(weight_sum, 1.0),
            ),
            flush=True,
        )

    return model


def predict_hash(model, singleton, crosses, numeric):
    model.eval()
    result = np.empty(len(singleton), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(singleton), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(singleton))
            single_t = torch.from_numpy(
                np.asarray(singleton[start:end], dtype=np.int64)
            )
            cross_t = torch.from_numpy(
                np.asarray(crosses[start:end], dtype=np.int64)
            )
            numeric_t = torch.from_numpy(
                np.asarray(numeric[start:end], dtype=np.float32)
            )
            result[start:end] = model(
                single_t,
                cross_t,
                numeric_t,
            ).numpy().astype(np.float32)

    return result


def predict_fm(model, singleton, numeric):
    model.eval()
    result = np.empty(len(singleton), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(singleton), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(singleton))
            single_t = torch.from_numpy(
                np.asarray(singleton[start:end], dtype=np.int64)
            )
            numeric_t = torch.from_numpy(
                np.asarray(numeric[start:end], dtype=np.float32)
            )
            result[start:end] = model(
                single_t,
                numeric_t,
            ).numpy().astype(np.float32)

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
    raise RuntimeError("Trusted incumbent predictions are required")

train = load("train")
valid = load("valid")

train_raw_cat = categorical_matrix(train)
valid_raw_cat = categorical_matrix(valid)

train_singleton = offset_categorical(train_raw_cat)
valid_singleton = offset_categorical(valid_raw_cat)

train_crosses = hashed_cross_matrix(train_raw_cat)
valid_crosses = hashed_cross_matrix(valid_raw_cat)

del train_raw_cat, valid_raw_cat
gc.collect()

train_history, history_names = selected_history("train")
valid_history, valid_history_names = selected_history("valid")

if history_names != valid_history_names:
    raise RuntimeError("History feature order differs")

train_raw_num = raw_numeric(train)
valid_raw_num = raw_numeric(valid)

numeric_mean, numeric_std = fit_numeric_transform(
    train_history,
    train_raw_num,
)
train_numeric = transform_numeric(
    train_history,
    train_raw_num,
    numeric_mean,
    numeric_std,
)
valid_numeric = transform_numeric(
    valid_history,
    valid_raw_num,
    numeric_mean,
    numeric_std,
)

del train_history, valid_history
del train_raw_num, valid_raw_num
gc.collect()

train_y = np.asarray(train.y, dtype=np.float32)
train_date = np.asarray(train.date, dtype=np.int32)
max_train_date = int(train_date.max())
age_days = (max_train_date - train_date).astype(np.float32)

train_weights = np.exp(
    -np.log(2.0) * age_days / HALF_LIFE_DAYS
).astype(np.float32)
train_weights /= np.mean(train_weights)

valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)

hash_model = HashedConjunctionWide(train_numeric.shape[1])
fm_model = FactorizationMachine(train_numeric.shape[1])

train_hash_model(
    hash_model,
    train_singleton,
    train_crosses,
    train_numeric,
    train_y,
    train_weights,
    seed=7319,
)
train_fm_model(
    fm_model,
    train_singleton,
    train_numeric,
    train_y,
    train_weights,
    seed=9473,
)

hash_valid_raw = predict_hash(
    hash_model,
    valid_singleton,
    valid_crosses,
    valid_numeric,
)
fm_valid_raw = predict_fm(
    fm_model,
    valid_singleton,
    valid_numeric,
)

hash_valid_rank = within_user_rank(valid_uid, hash_valid_raw)
fm_valid_rank = within_user_rank(valid_uid, fm_valid_raw)

inc_valid_raw = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_valid_rank = within_user_rank(valid_uid, inc_valid_raw)

family_ranks = {
    "hashed_conjunction": hash_valid_rank,
    "factorization_machine": fm_valid_rank,
}

candidate_results = {}
inc_metrics = evaluate(valid_uid, valid_y, inc_valid_rank)
candidate_results["trusted_incumbent"] = float(
    inc_metrics["primary"]
)

for name, rank in family_ranks.items():
    metrics = evaluate(valid_uid, valid_y, rank)
    candidate_results[name + "_standalone"] = float(
        metrics["primary"]
    )
    print(
        "FINDINGS family=%s standalone_primary=%.6f "
        "gauc=%.6f ndcg5=%.6f"
        % (
            name,
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        ),
        flush=True,
    )

pair_mean_rank = (
    0.5 * hash_valid_rank + 0.5 * fm_valid_rank
).astype(np.float32)
pair_metrics = evaluate(valid_uid, valid_y, pair_mean_rank)
candidate_results["new_family_mean_standalone"] = float(
    pair_metrics["primary"]
)

best_scores = inc_valid_rank.copy()
best_primary = float(inc_metrics["primary"])
best_family = "trusted_incumbent"
best_transform = "rank"
best_alpha = 0.0
best_raw_rank = None

blend_sources = {
    "hashed_conjunction": hash_valid_rank,
    "factorization_machine": fm_valid_rank,
    "new_family_mean": pair_mean_rank,
}

alphas = [0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.32]
inc_copula = copula_score(inc_valid_rank)

for name, source_rank in blend_sources.items():
    source_copula = copula_score(source_rank)

    for transform, inc_base, source_base in (
        ("rank", inc_valid_rank, source_rank),
        ("copula", inc_copula, source_copula),
    ):
        for alpha in alphas:
            blended = (
                (1.0 - alpha) * inc_base
                + alpha * source_base
            ).astype(np.float32)

            metrics = evaluate(valid_uid, valid_y, blended)
            candidate_name = "%s_%s_blend_%.2f" % (
                name,
                transform,
                alpha,
            )
            candidate_results[candidate_name] = float(
                metrics["primary"]
            )

            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_scores = blended.copy()
                best_family = name
                best_transform = transform
                best_alpha = float(alpha)
                best_raw_rank = source_rank.copy()

final_metrics = evaluate(valid_uid, valid_y, best_scores)

rank_corr = float(
    np.corrcoef(
        hash_valid_rank.astype(np.float64),
        fm_valid_rank.astype(np.float64),
    )[0, 1]
)

print(
    "FINDINGS cross_count=%d hash_size=%d numeric_dim=%d "
    "half_life_days=%.1f family_rank_corr=%.6f"
    % (
        len(CROSS_PAIRS),
        HASH_SIZE,
        train_numeric.shape[1],
        HALF_LIFE_DAYS,
        rank_corr,
    ),
    flush=True,
)
print(
    "FINDINGS winner=%s transform=%s alpha=%.2f "
    "incumbent_primary=%.6f winner_primary=%.6f"
    % (
        best_family,
        best_transform,
        best_alpha,
        float(inc_metrics["primary"]),
        float(final_metrics["primary"]),
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(
        candidate_results,
        sort_keys=True,
    ),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_raw_rank is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_rank, dtype=np.float64),
        )

del train
del valid
del train_singleton
del train_crosses
del train_numeric
del train_y
del train_date
del train_weights
del valid_singleton
del valid_crosses
del valid_numeric
del inc_valid_raw
del hash_valid_raw
del fm_valid_raw
gc.collect()

test = load("test")
test_raw_cat = categorical_matrix(test)
test_singleton = offset_categorical(test_raw_cat)

inc_test_raw = np.asarray(
    np.load(inc_test_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_test_rank = within_user_rank(test.user_id, inc_test_raw)

if best_family == "trusted_incumbent":
    test_scores = inc_test_rank
else:
    test_history, test_history_names = selected_history("test")
    if history_names != test_history_names:
        raise RuntimeError("Test history feature order differs")

    test_raw_num = raw_numeric(test)
    test_numeric = transform_numeric(
        test_history,
        test_raw_num,
        numeric_mean,
        numeric_std,
    )
    del test_history, test_raw_num
    gc.collect()

    if best_family in (
        "hashed_conjunction",
        "new_family_mean",
    ):
        test_crosses = hashed_cross_matrix(test_raw_cat)
        hash_test_raw = predict_hash(
            hash_model,
            test_singleton,
            test_crosses,
            test_numeric,
        )
        hash_test_rank = within_user_rank(
            test.user_id,
            hash_test_raw,
        )
        del test_crosses, hash_test_raw
    else:
        hash_test_rank = None

    if best_family in (
        "factorization_machine",
        "new_family_mean",
    ):
        fm_test_raw = predict_fm(
            fm_model,
            test_singleton,
            test_numeric,
        )
        fm_test_rank = within_user_rank(
            test.user_id,
            fm_test_raw,
        )
        del fm_test_raw
    else:
        fm_test_rank = None

    if best_family == "hashed_conjunction":
        source_test_rank = hash_test_rank
    elif best_family == "factorization_machine":
        source_test_rank = fm_test_rank
    elif best_family == "new_family_mean":
        source_test_rank = (
            0.5 * hash_test_rank + 0.5 * fm_test_rank
        ).astype(np.float32)
    else:
        raise RuntimeError("Unknown selected family")

    if best_transform == "copula":
        incumbent_base = copula_score(inc_test_rank)
        source_base = copula_score(source_test_rank)
    else:
        incumbent_base = inc_test_rank
        source_base = source_test_rank

    test_scores = (
        (1.0 - best_alpha) * incumbent_base
        + best_alpha * source_base
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