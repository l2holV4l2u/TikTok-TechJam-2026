import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
THREADS = min(8, os.cpu_count() or 1)
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
HISTORY_LENGTH = 12

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
n_train = y_train.shape[0]

cards = [int(FEATURE_CARDINALITIES[name]) for name in CAT_FIELDS]
offsets = np.zeros(len(cards), dtype=np.int64)
offsets[1:] = np.cumsum(cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_categorical(split):
    result = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, name in enumerate(CAT_FIELDS):
        result[:, j] = np.asarray(split.X[name], dtype=np.int64) + offsets[j]
    return result


xcat_train = make_categorical(train)
xcat_valid = make_categorical(valid)
xcat_test = make_categorical(test)


def make_raw_numeric(split_name, split):
    columns = []
    for name in NUM_FIELDS:
        values = np.asarray(split.num[name], dtype=np.float32)
        if name != "user_register_days":
            values = np.log1p(np.maximum(values, 0.0))
        columns.append(values)

    for entity in ("video_id", "author_id"):
        histories = historical_features(split_name, key=entity)
        for name in sorted(histories):
            columns.append(np.asarray(histories[name], dtype=np.float32))

    return np.column_stack(columns).astype(np.float32)


raw_num_train = make_raw_numeric("train", train)
raw_num_valid = make_raw_numeric("valid", valid)
raw_num_test = make_raw_numeric("test", test)

finite_train = np.where(np.isfinite(raw_num_train), raw_num_train, np.nan)
num_mean = np.nanmean(finite_train, axis=0).astype(np.float32)
num_mean = np.where(np.isfinite(num_mean), num_mean, 0.0).astype(np.float32)
num_std = np.nanstd(finite_train, axis=0).astype(np.float32)
num_std = np.where(
    np.isfinite(num_std) & (num_std > 1e-5), num_std, 1.0
).astype(np.float32)


def normalize_numeric(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.where(np.isfinite(values), values, num_mean[None, :])
    values = (values - num_mean[None, :]) / num_std[None, :]
    return np.clip(values, -8.0, 8.0).astype(np.float32)


xnum_train = normalize_numeric(raw_num_train)
xnum_valid = normalize_numeric(raw_num_valid)
xnum_test = normalize_numeric(raw_num_test)
n_num = xnum_train.shape[1]

del raw_num_train, raw_num_valid, raw_num_test, finite_train

base_rate = float(np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6))
base_logit = float(np.log(base_rate / (1.0 - base_rate)))

train_dates = np.asarray(train.date, dtype=np.int64)
age_days = (train_dates.max() - train_dates).astype(np.float32)


def recency_weights(half_life):
    weights = np.power(0.5, age_days / float(half_life)).astype(np.float32)
    weights /= weights.mean()
    return weights


def build_sequential_histories():
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    times = np.asarray(train.time_ms, dtype=np.int64)
    rows = np.arange(n_train, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    sorted_users = users[order]
    sorted_videos = videos[order]
    sorted_y = y_train[order].astype(np.int64)

    new_group = np.empty(n_train, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(new_group)
    group = np.cumsum(new_group, dtype=np.int64) - 1

    global_cum = np.cumsum(sorted_y, dtype=np.int64)
    group_base = global_cum[starts] - sorted_y[starts]
    positives_before = global_cum - sorted_y - group_base[group]

    max_positive = int((positives_before + sorted_y).max())
    user_card = int(FEATURE_CARDINALITIES["user_id"])
    positive_table = np.zeros(
        (user_card, max_positive + 1), dtype=np.int32
    )

    positive_mask = sorted_y == 1
    positive_ordinals = positives_before[positive_mask] + 1
    positive_table[
        sorted_users[positive_mask], positive_ordinals
    ] = sorted_videos[positive_mask].astype(np.int32)

    lags = np.arange(HISTORY_LENGTH, dtype=np.int64)
    ordinals = positives_before[:, None] - lags[None, :]
    valid_ordinal = ordinals > 0
    safe_ordinals = np.maximum(ordinals, 0)
    sorted_history = positive_table[
        sorted_users[:, None], safe_ordinals
    ]
    sorted_history[~valid_ordinal] = 0

    train_history = np.empty_like(sorted_history, dtype=np.int32)
    train_history[order] = sorted_history.astype(np.int32)

    total_positive = np.zeros(user_card, dtype=np.int64)
    group_ends = np.r_[starts[1:] - 1, n_train - 1]
    total_positive[sorted_users[group_ends]] = (
        positives_before[group_ends] + sorted_y[group_ends]
    )

    def evaluation_history(split):
        split_users = np.asarray(split.user_id, dtype=np.int64)
        ords = total_positive[split_users, None] - lags[None, :]
        ok = ords > 0
        safe = np.maximum(ords, 0)
        result = positive_table[split_users[:, None], safe]
        result[~ok] = 0
        return result.astype(np.int32)

    return (
        train_history,
        evaluation_history(valid),
        evaluation_history(test),
        total_positive,
    )


xhist_train, xhist_valid, xhist_test, total_user_positives = (
    build_sequential_histories()
)

print(
    "FINDINGS sequential_history users_with_positive=%d mean_train_prior=%.3f"
    % (
        int(np.count_nonzero(total_user_positives)),
        float(np.count_nonzero(xhist_train, axis=1).mean()),
    ),
    flush=True,
)


class AutoInt(nn.Module):
    def __init__(self, rank=12, heads=3):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.025)

        self.numeric_token = nn.Linear(n_num, rank)
        self.attention1 = nn.MultiheadAttention(
            rank, heads, dropout=0.04, batch_first=True
        )
        self.attention2 = nn.MultiheadAttention(
            rank, heads, dropout=0.04, batch_first=True
        )
        self.norm1 = nn.LayerNorm(rank)
        self.norm2 = nn.LayerNorm(rank)

        token_count = len(CAT_FIELDS) + 1
        self.output = nn.Sequential(
            nn.Linear(token_count * rank + n_num, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 1),
        )
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum, xhist=None):
        categorical = self.embedding(xcat)
        numeric = self.numeric_token(xnum).unsqueeze(1)
        tokens = torch.cat([categorical, numeric], dim=1)

        attended, _ = self.attention1(
            tokens, tokens, tokens, need_weights=False
        )
        tokens = self.norm1(tokens + attended)
        attended, _ = self.attention2(
            tokens, tokens, tokens, need_weights=False
        )
        tokens = self.norm2(tokens + attended)

        features = torch.cat([tokens.flatten(1), xnum], dim=1)
        return self.bias + self.output(features).squeeze(1)


class DIN(nn.Module):
    def __init__(self, rank=16):
        super().__init__()
        self.rank = rank
        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.025)

        video_card = int(FEATURE_CARDINALITIES["video_id"])
        self.history_embedding = nn.Embedding(
            video_card, rank, padding_idx=0
        )
        nn.init.normal_(self.history_embedding.weight, std=0.025)
        with torch.no_grad():
            self.history_embedding.weight[0].zero_()

        self.attention = nn.Sequential(
            nn.Linear(4 * rank, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

        input_dim = len(CAT_FIELDS) * rank + rank + n_num
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum, xhist):
        fields = self.embedding(xcat)

        candidate_raw = xcat[:, 1] - int(offsets[1])
        candidate_raw = torch.clamp(
            candidate_raw,
            min=0,
            max=self.history_embedding.num_embeddings - 1,
        )
        query = self.history_embedding(candidate_raw)
        history = self.history_embedding(xhist.long())

        expanded_query = query.unsqueeze(1).expand_as(history)
        attention_input = torch.cat(
            [
                expanded_query,
                history,
                expanded_query - history,
                expanded_query * history,
            ],
            dim=2,
        )
        attention_logits = self.attention(attention_input).squeeze(2)
        mask = xhist != 0

        shifted = attention_logits - attention_logits.max(
            dim=1, keepdim=True
        ).values
        weights = torch.exp(shifted) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        interest = (history * weights.unsqueeze(2)).sum(dim=1)

        features = torch.cat(
            [fields.flatten(1), interest, xnum], dim=1
        )
        return self.bias + self.network(features).squeeze(1)


def train_neural(model, weights, seed, use_history):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.3e-3, weight_decay=2e-6
    )

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(n_train)
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            cat = torch.from_numpy(xcat_train[idx])
            num = torch.from_numpy(xnum_train[idx])
            target = torch.from_numpy(y_train[idx])
            weight = torch.from_numpy(weights[idx])
            hist = (
                torch.from_numpy(xhist_train[idx])
                if use_history else None
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat, num, hist)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (row_loss * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float((row_loss.detach() * weight).sum())
            total_weight += float(weight.sum())

        print(
            "TRAIN model=%s epoch=%d loss=%.6f"
            % (
                model.__class__.__name__,
                epoch + 1,
                total_loss / total_weight,
            ),
            flush=True,
        )

    return model


def predict_neural(model, xcat, xnum, xhist, use_history):
    result = np.empty(xcat.shape[0], dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, xcat.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, xcat.shape[0])
            hist = (
                torch.from_numpy(xhist[start:end])
                if use_history else None
            )
            logits = model(
                torch.from_numpy(xcat[start:end]),
                torch.from_numpy(xnum[start:end]),
                hist,
            )
            result[start:end] = logits.numpy().astype(np.float64)
    return result


family_predictions = {}

for half_life in (2.0, 4.0, 8.0):
    model = AutoInt()
    model = train_neural(
        model,
        recency_weights(half_life),
        SEED + int(half_life * 10),
        use_history=False,
    )
    valid_scores = predict_neural(
        model, xcat_valid, xnum_valid, xhist_valid, False
    )
    test_scores = predict_neural(
        model, xcat_test, xnum_test, xhist_test, False
    )
    family_predictions["autoint_hl%d" % int(half_life)] = (
        valid_scores,
        test_scores,
    )
    del model

din_model = DIN()
din_model = train_neural(
    din_model,
    recency_weights(4.0),
    SEED + 401,
    use_history=True,
)
family_predictions["din_sequential"] = (
    predict_neural(
        din_model, xcat_valid, xnum_valid, xhist_valid, True
    ),
    predict_neural(
        din_model, xcat_test, xnum_test, xhist_test, True
    ),
)
del din_model

# A positive-interaction latent SVD is structurally distinct from the
# supervised CTR models. Recency weighting makes recent positive co-occurrence
# dominate factors that must extrapolate across the date boundary.
svd_users = np.asarray(train.user_id, dtype=np.int64)
svd_videos = np.asarray(train.video_id, dtype=np.int64)
positive = y_train > 0.5
svd_data = recency_weights(4.0)[positive].astype(np.float64)

interaction_matrix = sparse.coo_matrix(
    (
        svd_data,
        (svd_users[positive], svd_videos[positive]),
    ),
    shape=(
        int(FEATURE_CARDINALITIES["user_id"]),
        int(FEATURE_CARDINALITIES["video_id"]),
    ),
).tocsr()

u_factor, singular, vt_factor = svds(
    interaction_matrix,
    k=24,
    which="LM",
    random_state=SEED,
)
order = np.argsort(singular)[::-1]
u_factor = u_factor[:, order]
singular = singular[order]
vt_factor = vt_factor[order]


def predict_svd(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    left = u_factor[users] * singular[None, :]
    right = vt_factor[:, videos].T
    return np.sum(left * right, axis=1, dtype=np.float64)


family_predictions["latent_svd"] = (
    predict_svd(valid),
    predict_svd(test),
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    group = np.cumsum(new_group, dtype=np.int64) - 1
    positions = np.arange(n, dtype=np.int64) - starts[group]
    denominators = np.maximum(lengths[group] - 1, 1)
    ranked_sorted = positions.astype(np.float64) / denominators
    ranked_sorted[lengths[group] == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)
inc_valid_rank = within_user_rank(valid_users, inc_valid)
inc_test_rank = within_user_rank(test_users, inc_test)

inc_metrics = evaluate(valid_users, y_valid, inc_valid_rank)
candidate_log = {
    "trusted_incumbent": float(inc_metrics["primary"])
}

best_name = "trusted_incumbent"
best_metrics = inc_metrics
best_valid = inc_valid_rank
best_test = inc_test_rank
best_raw_valid = None

blend_alphas = (0.25, 0.50, 0.75, 1.00)

for family_name, (raw_valid, raw_test) in family_predictions.items():
    family_valid_rank = within_user_rank(valid_users, raw_valid)
    family_test_rank = within_user_rank(test_users, raw_test)

    standalone_metrics = evaluate(
        valid_users, y_valid, family_valid_rank
    )
    candidate_log[family_name] = float(
        standalone_metrics["primary"]
    )

    local_best = -np.inf
    local_best_alpha = None

    for alpha in blend_alphas:
        blended_valid = (
            alpha * family_valid_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        metrics = evaluate(valid_users, y_valid, blended_valid)
        blend_name = "%s_blend_a%.2f" % (family_name, alpha)
        candidate_log[blend_name] = float(metrics["primary"])

        if metrics["primary"] > local_best:
            local_best = float(metrics["primary"])
            local_best_alpha = alpha

        if metrics["primary"] > best_metrics["primary"]:
            best_name = blend_name
            best_metrics = metrics
            best_valid = blended_valid
            best_test = (
                alpha * family_test_rank
                + (1.0 - alpha) * inc_test_rank
            )
            best_raw_valid = family_valid_rank

    print(
        "FINDINGS family=%s standalone=%.6f best_blend=%.6f alpha=%.2f"
        % (
            family_name,
            standalone_metrics["primary"],
            local_best,
            local_best_alpha,
        ),
        flush=True,
    )

print(
    "CANDIDATES " + json.dumps(candidate_log, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS selected=%s" % best_name,
    flush=True,
)

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
    if best_raw_valid is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        best_metrics["primary"],
        best_metrics["gauc"],
        best_metrics["ndcg@5"],
        elapsed,
    )
)