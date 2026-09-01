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

torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
torch.manual_seed(4103)
np.random.seed(4103)

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

BATCH_SIZE = 16384
EPOCHS = 2
HALF_LIFE_DAYS = 4.0
EMBED_DIM = 8


def clipped_categorical(split):
    columns = []
    for field, card, offset in zip(FIELDS, CARDS, OFFSETS):
        x = np.asarray(split.X[field], dtype=np.int64)
        x = np.where((x >= 0) & (x < card), x, 0)
        columns.append((x + offset).astype(np.int32))
    return np.column_stack(columns).astype(np.int32, copy=False)


def selected_history(split_name):
    arrays = []
    names = []

    wanted_suffixes = (
        "train_count_log1p",
        "long_view_rate",
        "is_click_rate",
        "play_time_ms_logmean",
        "comment_stay_time_logmean",
    )

    for key in ("video_id", "author_id"):
        history = historical_features(split_name, key=key)
        for name in sorted(history):
            if any(name.endswith(suffix) for suffix in wanted_suffixes):
                arrays.append(np.asarray(history[name], dtype=np.float32))
                names.append(name)

    if not arrays:
        raise RuntimeError("No requested historical features were available")

    return np.column_stack(arrays).astype(np.float32), names


def raw_numeric(split):
    columns = []
    for name in [
        "duration_ms",
        "user_fans_user_num",
        "user_follow_user_num",
        "user_friend_user_num",
        "user_register_days",
    ]:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        columns.append(x)
    return np.column_stack(columns).astype(np.float32)


def fit_numeric_transform(train_history, train_raw):
    matrix = np.column_stack([train_history, train_raw]).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)
    return mean, std


def transform_numeric(history, raw, mean, std):
    matrix = np.column_stack([history, raw]).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix - mean) / std
    return np.clip(matrix, -8.0, 8.0).astype(np.float32)


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

    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = (
        np.arange(n, dtype=np.float32)
        - start_pos.astype(np.float32)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    sizes = np.diff(np.r_[-1, np.flatnonzero(ends)]).astype(np.float32)

    group = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = local / denom
    return result


def copula_score(rank):
    rank = np.asarray(rank, dtype=np.float64)
    p = np.clip(rank, 1e-4, 1.0 - 1e-4)
    return ndtri(p).astype(np.float32)


class AdditiveWide(nn.Module):
    """
    A regularized generalized additive model over categorical identities plus
    a linear history/numeric component. No learned feature interactions enter
    the prediction.
    """
    def __init__(self, numeric_dim):
        super().__init__()
        self.wide = nn.Embedding(TOTAL_CARD, 1)
        self.numeric = nn.Linear(numeric_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.numeric.weight)
        nn.init.zeros_(self.numeric.bias)

    def forward(self, categorical, numeric):
        categorical_term = self.wide(categorical).sum(dim=1).squeeze(1)
        numeric_term = self.numeric(numeric).squeeze(1)
        return categorical_term + numeric_term + self.bias


class ExplicitCIN(nn.Module):
    """
    xDeepFM-style compressed interaction network. Each layer explicitly forms
    vector-wise outer products between original fields and the previous
    interaction layer, yielding bounded third-order interactions without an
    unrestricted MLP.
    """
    def __init__(self, numeric_dim):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.wide = nn.Embedding(TOTAL_CARD, 1)
        self.numeric = nn.Linear(numeric_dim, 1)

        field_count = len(FIELDS)
        self.cin_weights = nn.ParameterList()
        previous_channels = field_count

        for output_channels in [16, 12]:
            weight = nn.Parameter(
                torch.empty(
                    output_channels,
                    field_count * previous_channels,
                )
            )
            nn.init.xavier_uniform_(weight)
            self.cin_weights.append(weight)
            previous_channels = output_channels

        self.cin_output = nn.Linear(16 + 12, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.numeric.weight)
        nn.init.zeros_(self.numeric.bias)
        nn.init.xavier_uniform_(self.cin_output.weight)
        nn.init.zeros_(self.cin_output.bias)

    def forward(self, categorical, numeric):
        x0 = self.embedding(categorical)  # B, fields, embedding_dim
        layer = x0
        pooled = []

        for weight in self.cin_weights:
            product = (
                x0.unsqueeze(2) * layer.unsqueeze(1)
            ).reshape(
                categorical.shape[0],
                x0.shape[1] * layer.shape[1],
                EMBED_DIM,
            )
            layer = torch.einsum("bce,oc->boe", product, weight)
            layer = torch.tanh(layer)
            pooled.append(layer.sum(dim=2))

        cin_term = self.cin_output(torch.cat(pooled, dim=1)).squeeze(1)
        wide_term = self.wide(categorical).sum(dim=1).squeeze(1)
        numeric_term = self.numeric(numeric).squeeze(1)
        return cin_term + wide_term + numeric_term + self.bias


def train_model(model, categorical, numeric, labels, weights, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0025,
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

            cat = torch.from_numpy(
                np.asarray(categorical[idx], dtype=np.int64)
            )
            num = torch.from_numpy(
                np.asarray(numeric[idx], dtype=np.float32)
            )
            y = torch.from_numpy(
                np.asarray(labels[idx], dtype=np.float32)
            )
            w = torch.from_numpy(
                np.asarray(weights[idx], dtype=np.float32)
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat, num)
            losses = criterion(logits, y)
            loss = torch.sum(losses * w) / torch.sum(w).clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float(torch.sum(losses * w).detach())
            weight_sum += float(torch.sum(w))

        print(
            "FINDINGS family=%s epoch=%d weighted_logloss=%.6f"
            % (
                model.__class__.__name__,
                epoch + 1,
                loss_sum / max(weight_sum, 1.0),
            ),
            flush=True,
        )

    return model


def predict_model(model, categorical, numeric):
    model.eval()
    result = np.empty(len(categorical), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(categorical), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(categorical))
            cat = torch.from_numpy(
                np.asarray(categorical[start:end], dtype=np.int64)
            )
            num = torch.from_numpy(
                np.asarray(numeric[start:end], dtype=np.float32)
            )
            result[start:end] = model(cat, num).numpy().astype(np.float32)

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

train_cat = clipped_categorical(train)
valid_cat = clipped_categorical(valid)

train_history, history_names = selected_history("train")
valid_history, valid_history_names = selected_history("valid")

if history_names != valid_history_names:
    raise RuntimeError("History feature order differs across splits")

train_raw = raw_numeric(train)
valid_raw = raw_numeric(valid)

num_mean, num_std = fit_numeric_transform(train_history, train_raw)
train_num = transform_numeric(
    train_history, train_raw, num_mean, num_std
)
valid_num = transform_numeric(
    valid_history, valid_raw, num_mean, num_std
)

del train_history, valid_history, train_raw, valid_raw
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

inc_valid_raw = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_rank = within_user_rank(valid_uid, inc_valid_raw)
inc_copula = copula_score(inc_rank)

control_metrics = evaluate(valid_uid, valid_y, inc_rank)

models = {
    "additive_wide": AdditiveWide(train_num.shape[1]),
    "explicit_cin": ExplicitCIN(train_num.shape[1]),
}
seeds = {
    "additive_wide": 5101,
    "explicit_cin": 6203,
}

candidate_results = {
    "trusted_incumbent": float(control_metrics["primary"])
}
raw_valid_by_family = {}
rank_valid_by_family = {}

for family, model in models.items():
    train_model(
        model,
        train_cat,
        train_num,
        train_y,
        train_weights,
        seeds[family],
    )

    raw = predict_model(model, valid_cat, valid_num)
    rank = within_user_rank(valid_uid, raw)
    metrics = evaluate(valid_uid, valid_y, rank)

    raw_valid_by_family[family] = raw
    rank_valid_by_family[family] = rank
    candidate_results[family + "_standalone"] = float(
        metrics["primary"]
    )

    print(
        "FINDINGS family=%s standalone_primary=%.6f "
        "gauc=%.6f ndcg5=%.6f"
        % (
            family,
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        ),
        flush=True,
    )

best_scores = inc_rank.copy()
best_primary = float(control_metrics["primary"])
best_family = None
best_transform = "rank"
best_alpha = 0.0
best_raw_rank = None

alphas = [0.03, 0.06, 0.10, 0.16, 0.25]

for family, rank in rank_valid_by_family.items():
    family_copula = copula_score(rank)

    for transform_name, incumbent_base, family_base in [
        ("rank", inc_rank, rank),
        ("copula", inc_copula, family_copula),
    ]:
        for alpha in alphas:
            blended = (
                (1.0 - alpha) * incumbent_base
                + alpha * family_base
            ).astype(np.float32)

            metrics = evaluate(valid_uid, valid_y, blended)
            name = "%s_%s_blend_%.2f" % (
                family,
                transform_name,
                alpha,
            )
            candidate_results[name] = float(metrics["primary"])

            print(
                "FINDINGS candidate=%s primary=%.6f "
                "gauc=%.6f ndcg5=%.6f delta=%+.6f"
                % (
                    name,
                    float(metrics["primary"]),
                    float(metrics["gauc"]),
                    float(metrics["ndcg@5"]),
                    float(metrics["primary"])
                    - float(control_metrics["primary"]),
                ),
                flush=True,
            )

            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_scores = blended.copy()
                best_family = family
                best_transform = transform_name
                best_alpha = float(alpha)
                best_raw_rank = rank.copy()

final_metrics = evaluate(valid_uid, valid_y, best_scores)

print(
    "FINDINGS history_features=%s numeric_dim=%d half_life_days=%.1f"
    % (
        ",".join(history_names),
        train_num.shape[1],
        HALF_LIFE_DAYS,
    ),
    flush=True,
)
print(
    "FINDINGS winner=%s transform=%s alpha=%.2f "
    "control_primary=%.6f winner_primary=%.6f"
    % (
        best_family if best_family is not None else "trusted_incumbent",
        best_transform,
        best_alpha,
        float(control_metrics["primary"]),
        float(final_metrics["primary"]),
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
    if best_raw_rank is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_rank, dtype=np.float64),
        )

del train
del valid
del train_cat
del valid_cat
del train_num
del valid_num
del train_y
del train_date
del train_weights
del inc_valid_raw
del raw_valid_by_family
del rank_valid_by_family
gc.collect()

test = load("test")
inc_test_raw = np.asarray(
    np.load(inc_test_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_test_rank = within_user_rank(test.user_id, inc_test_raw)

if best_family is None:
    test_scores = inc_test_rank
else:
    test_cat = clipped_categorical(test)
    test_history, test_history_names = selected_history("test")
    if history_names != test_history_names:
        raise RuntimeError("Test history feature order differs")

    test_raw_numeric = raw_numeric(test)
    test_num = transform_numeric(
        test_history,
        test_raw_numeric,
        num_mean,
        num_std,
    )
    del test_history, test_raw_numeric
    gc.collect()

    selected_raw = predict_model(
        models[best_family],
        test_cat,
        test_num,
    )
    selected_rank = within_user_rank(test.user_id, selected_raw)

    if best_transform == "copula":
        incumbent_base = copula_score(inc_test_rank)
        selected_base = copula_score(selected_rank)
    else:
        incumbent_base = inc_test_rank
        selected_base = selected_rank

    test_scores = (
        (1.0 - best_alpha) * incumbent_base
        + best_alpha * selected_base
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