import os
import gc
import json
import time
import warnings
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.data import load
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

np.random.seed(2026)
torch.manual_seed(2026)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = (
        np.arange(n, dtype=np.float32) -
        start_positions.astype(np.float32)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_positions]).astype(np.float32)

    groups = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[groups] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = local_rank / denom
    return result


def sequence_base_features(split):
    uid = np.asarray(split.user_id, dtype=np.int64)
    tm = np.asarray(split.time_ms, dtype=np.int64)
    n = len(uid)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, tm, uid))
    sorted_uid = uid[order]
    sorted_tm = tm[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = sorted_uid[1:] != sorted_uid[:-1]

    gap_sorted = np.zeros(n, dtype=np.int64)
    valid_previous = ~new_user
    previous_positions = np.flatnonzero(valid_previous) - 1
    gap_sorted[valid_previous] = np.maximum(
        sorted_tm[valid_previous] - sorted_tm[previous_positions], 0
    )

    session_start = new_user | (gap_sorted > 30 * 60 * 1000)
    session_start_position = np.maximum.accumulate(
        np.where(session_start, np.arange(n, dtype=np.int64), 0)
    )
    user_start_position = np.maximum.accumulate(
        np.where(new_user, np.arange(n, dtype=np.int64), 0)
    )

    session_position_sorted = (
        np.arange(n, dtype=np.int64) - session_start_position
    )
    user_position_sorted = (
        np.arange(n, dtype=np.int64) - user_start_position
    )

    previous_gap = np.empty(n, dtype=np.int64)
    session_position = np.empty(n, dtype=np.int32)
    user_position = np.empty(n, dtype=np.int32)

    previous_gap[order] = gap_sorted
    session_position[order] = session_position_sorted.astype(np.int32)
    user_position[order] = user_position_sorted.astype(np.int32)

    previous = {}
    for name in ("tag", "tab", "duration_bucket"):
        values = np.asarray(split.X[name], dtype=np.int32)
        sorted_values = values[order]
        previous_sorted = np.full(n, -1, dtype=np.int32)
        previous_sorted[1:] = sorted_values[:-1]
        previous_sorted[new_user] = -1

        restored = np.empty(n, dtype=np.int32)
        restored[order] = previous_sorted
        previous[name] = restored

    return {
        "previous_gap": previous_gap,
        "session_position": session_position,
        "user_position": user_position,
        "previous_tag": previous["tag"],
        "previous_tab": previous["tab"],
        "previous_duration": previous["duration_bucket"],
    }


def repeated_entity_features(split, field):
    uid = np.asarray(split.user_id, dtype=np.int64)
    tm = np.asarray(split.time_ms, dtype=np.int64)
    entity = np.asarray(split.X[field], dtype=np.int64)
    n = len(uid)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, tm, entity, uid))
    sorted_uid = uid[order]
    sorted_entity = entity[order]
    sorted_tm = tm[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = (
        (sorted_uid[1:] != sorted_uid[:-1]) |
        (sorted_entity[1:] != sorted_entity[:-1])
    )

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    count_sorted = (
        np.arange(n, dtype=np.int64) - start_positions
    )

    gap_sorted = np.zeros(n, dtype=np.int64)
    has_previous = ~starts
    previous_positions = np.flatnonzero(has_previous) - 1
    gap_sorted[has_previous] = np.maximum(
        sorted_tm[has_previous] - sorted_tm[previous_positions], 0
    )

    unknown = sorted_entity == 0
    count_sorted[unknown] = 0
    gap_sorted[unknown] = 0

    count = np.empty(n, dtype=np.int32)
    gap = np.empty(n, dtype=np.int64)
    count[order] = count_sorted.astype(np.int32)
    gap[order] = gap_sorted
    return count, gap


def log_bucket(values, cap):
    values = np.asarray(values)
    return np.minimum(
        np.floor(np.log2(np.maximum(values, 0) + 1.0)).astype(np.int32),
        int(cap),
    )


def gap_bucket(gap_ms):
    gap_minutes = np.asarray(gap_ms, dtype=np.float64) / 60000.0
    edges = np.asarray(
        [0.001, 0.25, 1.0, 5.0, 30.0, 120.0, 720.0, 2880.0],
        dtype=np.float64,
    )
    return np.searchsorted(
        edges, gap_minutes, side="right"
    ).astype(np.int32)


def infer_transition_cards(split):
    return {
        "tag": int(np.max(np.asarray(split.X["tag"]))) + 2,
        "tab": int(np.max(np.asarray(split.X["tab"]))) + 2,
        "duration": (
            int(np.max(np.asarray(split.X["duration_bucket"]))) + 2
        ),
    }


FEATURE_NAMES = [
    "hour",
    "user_position",
    "session_position",
    "previous_gap",
    "video_repeat",
    "author_repeat",
    "video_count",
    "author_count",
    "tag_transition",
    "tab_transition",
    "duration_transition",
]


def construct_matrix(split, transition_cards):
    base = sequence_base_features(split)
    video_count, video_gap = repeated_entity_features(split, "video_id")
    author_count, author_gap = repeated_entity_features(split, "author_id")

    tag = np.asarray(split.X["tag"], dtype=np.int32)
    tab = np.asarray(split.X["tab"], dtype=np.int32)
    duration = np.asarray(
        split.X["duration_bucket"], dtype=np.int32
    )
    hour = np.maximum(
        np.asarray(split.X["hour"], dtype=np.int32), 0
    )

    tag_transition = (
        (base["previous_tag"] + 1) * transition_cards["tag"] +
        (tag + 1)
    ).astype(np.int32)
    tab_transition = (
        (base["previous_tab"] + 1) * transition_cards["tab"] +
        (tab + 1)
    ).astype(np.int32)
    duration_transition = (
        (base["previous_duration"] + 1) *
        transition_cards["duration"] +
        (duration + 1)
    ).astype(np.int32)

    video_count_bucket = np.minimum(video_count, 7).astype(np.int32)
    author_count_bucket = np.minimum(author_count, 15).astype(np.int32)

    matrix = np.column_stack([
        hour,
        log_bucket(base["user_position"], 13),
        log_bucket(base["session_position"], 10),
        gap_bucket(base["previous_gap"]),
        video_count_bucket * 10 + gap_bucket(video_gap),
        author_count_bucket * 10 + gap_bucket(author_gap),
        video_count_bucket,
        author_count_bucket,
        tag_transition,
        tab_transition,
        duration_transition,
    ]).astype(np.int32, copy=False)

    del base, video_count, video_gap, author_count, author_gap
    del tag, tab, duration, hour
    gc.collect()
    return matrix


def fit_rate_tables(X, y, weights, prior):
    alphas = np.asarray(
        [500, 800, 500, 500, 250, 300, 500, 500, 180, 250, 250],
        dtype=np.float32,
    )
    tables = []
    fallback = float(safe_logit(np.asarray([prior]))[0])

    for j in range(X.shape[1]):
        code = X[:, j].astype(np.int64, copy=False)
        size = int(np.max(code)) + 1
        sw = np.bincount(
            code, weights=weights, minlength=size
        ).astype(np.float32)
        sy = np.bincount(
            code, weights=weights * y, minlength=size
        ).astype(np.float32)
        rate = (
            sy + alphas[j] * prior
        ) / np.maximum(sw + alphas[j], 1e-6)
        tables.append(safe_logit(rate))

    return tables, fallback


ADDITIVE_WEIGHTS = np.asarray(
    [0.15, 0.0, 0.40, 0.55, 0.80, 1.00, 0.0, 0.0, 0.65, 0.50, 0.35],
    dtype=np.float32,
)


def predict_additive(X, tables, fallback):
    result = np.zeros(len(X), dtype=np.float32)
    scale = float(np.sum(np.abs(ADDITIVE_WEIGHTS)))

    for j, weight in enumerate(ADDITIVE_WEIGHTS):
        if weight == 0:
            continue
        code = X[:, j].astype(np.int64, copy=False)
        table = tables[j]
        contribution = np.full(len(X), fallback, dtype=np.float32)
        ok = (code >= 0) & (code < len(table))
        contribution[ok] = table[code[ok]]
        result += float(weight) * (contribution - fallback)

    return result / max(scale, 1e-6)


class SequenceMLP(nn.Module):
    def __init__(self, cards):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_width = 0

        for card in cards:
            dim = int(min(16, max(4, round(card ** 0.25 * 3))))
            self.embeddings.append(nn.Embedding(card, dim))
            total_width += dim

        self.network = nn.Sequential(
            nn.Linear(total_width, 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        parts = [
            embedding(x[:, j])
            for j, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat(parts, dim=1)).squeeze(1)


def train_mlp(X, y, weights, cards):
    model = SequenceMLP(cards)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=1e-5
    )

    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.int64, copy=False)),
        torch.from_numpy(y.astype(np.float32, copy=False)),
        torch.from_numpy(weights.astype(np.float32, copy=False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=32768,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model.train()
    for _ in range(2):
        for xb, yb, wb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.clamp(
                torch.sum(wb), min=1.0
            )
            loss.backward()
            optimizer.step()

    return model


def predict_mlp(model, X):
    model.eval()
    output = np.empty(len(X), dtype=np.float32)
    batch_size = 65536

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            xb = torch.from_numpy(
                X[start:end].astype(np.int64, copy=False)
            )
            output[start:end] = model(xb).numpy().astype(np.float32)

    return output


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
train_date = np.asarray(train.date, dtype=np.int32)

max_train_date = int(np.max(train_date))
age = (max_train_date - train_date).astype(np.float32)
train_weight = np.power(0.5, age / 4.0).astype(np.float32)
train_weight /= np.mean(train_weight)

prior = float(
    np.sum(train_weight * train_y) /
    np.maximum(np.sum(train_weight), 1e-6)
)

transition_cards = infer_transition_cards(train)
train_X = construct_matrix(train, transition_cards)
feature_cards = [
    int(np.max(train_X[:, j])) + 1
    for j in range(train_X.shape[1])
]

print(
    "FINDINGS sequence_cards=%s temporal_prior=%.6f "
    "weight_q10_q50_q90=%.4f,%.4f,%.4f"
    % (
        ",".join(map(str, feature_cards)),
        prior,
        float(np.quantile(train_weight, 0.10)),
        float(np.quantile(train_weight, 0.50)),
        float(np.quantile(train_weight, 0.90)),
    ),
    flush=True,
)

rate_tables, rate_fallback = fit_rate_tables(
    train_X, train_y, train_weight, prior
)

lgb_train = lgb.Dataset(
    train_X,
    label=train_y,
    weight=train_weight,
    categorical_feature=list(range(train_X.shape[1])),
    free_raw_data=False,
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 3000,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "min_gain_to_split": 1e-4,
    "num_threads": max(1, min(16, os.cpu_count() or 1)),
    "seed": 2026,
    "feature_fraction_seed": 2026,
    "bagging_seed": 2026,
    "verbose": -1,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=240,
)

mlp_model = train_mlp(
    train_X, train_y, train_weight, feature_cards
)

del lgb_train, train_y, train_weight, train_date, train
gc.collect()

valid = load("valid")
valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)
valid_X = construct_matrix(valid, transition_cards)

valid_predictions = {
    "additive_rate_replication": predict_additive(
        valid_X, rate_tables, rate_fallback
    ),
    "sequence_lightgbm": lgb_model.predict(
        valid_X, num_iteration=lgb_model.best_iteration
    ).astype(np.float32),
    "sequence_embedding_mlp": predict_mlp(mlp_model, valid_X),
}

inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)

if not (
    inc_valid_path and inc_test_path and
    os.path.exists(inc_valid_path) and
    os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent artifacts are required")

inc_valid = np.load(inc_valid_path, mmap_mode="r")
inc_valid_rank = within_user_rank(valid_uid, inc_valid)
inc_metrics = evaluate(valid_uid, valid_y, inc_valid_rank)

candidate_scores = {
    "trusted_incumbent": float(inc_metrics["primary"])
}

best_name = "trusted_incumbent"
best_family = None
best_alpha = 0.0
best_primary = float(inc_metrics["primary"])
best_valid_scores = inc_valid_rank.copy()
best_raw_valid = None

ALPHAS = (
    -0.20, -0.12, -0.08, -0.05, -0.03,
     0.02,  0.03,  0.05,  0.08,  0.12,
     0.18,  0.25,  0.35,  0.50,
)

for family, raw_scores in valid_predictions.items():
    raw_metrics = evaluate(valid_uid, valid_y, raw_scores)
    candidate_scores[family + "_standalone"] = float(
        raw_metrics["primary"]
    )

    family_rank = within_user_rank(valid_uid, raw_scores)
    local_best = -np.inf
    local_alpha = 0.0

    for alpha in ALPHAS:
        blended = (
            inc_valid_rank +
            float(alpha) * (family_rank - 0.5)
        ).astype(np.float32)

        metrics = evaluate(valid_uid, valid_y, blended)
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = "%s_blend_%+.2f" % (family, alpha)
            best_family = family
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()
            best_raw_valid = raw_scores.copy()

    candidate_scores[family + "_best_blend"] = float(local_best)

    print(
        "FINDINGS family=%s standalone=%.6f best_alpha=%+.2f "
        "best_blend=%.6f rank_corr_incumbent=%.6f"
        % (
            family,
            float(raw_metrics["primary"]),
            local_alpha,
            local_best,
            float(np.corrcoef(inc_valid_rank, family_rank)[0, 1]),
        ),
        flush=True,
    )

final_metrics = evaluate(
    valid_uid, valid_y, best_valid_scores
)

print(
    "FINDINGS winner=%s alpha=%+.2f" %
    (best_name, best_alpha),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    if best_family is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

del valid_X, valid_predictions, valid_y
del best_valid_scores, best_raw_valid, inc_valid, inc_valid_rank
gc.collect()

test = load("test")
test_X = construct_matrix(test, transition_cards)
inc_test = np.load(inc_test_path, mmap_mode="r")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if best_family is None:
    test_scores = inc_test_rank
elif best_family == "additive_rate_replication":
    raw_test = predict_additive(
        test_X, rate_tables, rate_fallback
    )
    test_scores = (
        inc_test_rank +
        best_alpha * (
            within_user_rank(test.user_id, raw_test) - 0.5
        )
    ).astype(np.float32)
elif best_family == "sequence_lightgbm":
    raw_test = lgb_model.predict(
        test_X, num_iteration=lgb_model.best_iteration
    ).astype(np.float32)
    test_scores = (
        inc_test_rank +
        best_alpha * (
            within_user_rank(test.user_id, raw_test) - 0.5
        )
    ).astype(np.float32)
elif best_family == "sequence_embedding_mlp":
    raw_test = predict_mlp(mlp_model, test_X)
    test_scores = (
        inc_test_rank +
        best_alpha * (
            within_user_rank(test.user_id, raw_test) - 0.5
        )
    ).astype(np.float32)
else:
    raise RuntimeError("Unknown selected family")

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS " + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)