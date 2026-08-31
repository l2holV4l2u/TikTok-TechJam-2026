import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
n_train = len(y_train)

# Recency weighting is fit entirely from train dates.
train_dates = np.asarray(train.date, dtype=np.int32)
last_train_date = int(np.max(train_dates))
date_age = np.array(
    [
        (
            np.datetime64(str(last_train_date), "D")
            - np.datetime64(str(int(d)), "D")
        ).astype(int)
        for d in np.unique(train_dates)
    ],
    dtype=np.float32,
)
unique_dates = np.unique(train_dates)
age_lookup = {int(d): float(a) for d, a in zip(unique_dates, date_age)}
ages = np.fromiter(
    (age_lookup[int(d)] for d in train_dates),
    dtype=np.float32,
    count=n_train,
)
train_weight = np.exp(-np.log(2.0) * ages / 5.0).astype(np.float32)
train_weight /= max(float(np.mean(train_weight)), 1e-6)

# ----------------------------------------------------------------------
# Family 1: hashed explicit-cross wide logistic regression.
#
# Unlike an FM, every chosen conjunction has an independent scalar
# coefficient. This can memorize stable user/content and content/context
# effects without forcing them through a low-rank interaction geometry.
# ----------------------------------------------------------------------

WIDE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "music_type",
    "upload_type",
    "video_type",
    "user_active_degree",
    "register_days_bucket",
    "hour",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

FIELD_INDEX = {name: i for i, name in enumerate(WIDE_FIELDS)}

CROSS_PAIRS = [
    ("user_id", "video_id"),
    ("user_id", "author_id"),
    ("user_id", "tab"),
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("user_id", "onehot_feat3"),
    ("video_id", "tab"),
    ("video_id", "tag"),
    ("video_id", "duration_bucket"),
    ("author_id", "tab"),
    ("author_id", "tag"),
    ("tab", "duration_bucket"),
    ("tag", "duration_bucket"),
]

HASH_SIZE = 1 << 20
CONSTANT_TOKEN = HASH_SIZE - 1


def hashed_tokens(split, start, end):
    size = end - start
    columns = []

    # Disjoint salts are mixed before hashing, so the table remains one
    # sparse parameter vector while feature identities remain distinguishable.
    for field_number, name in enumerate(WIDE_FIELDS):
        x = np.asarray(split.X[name][start:end], dtype=np.int64)
        token = (
            x * np.int64(1000003)
            + np.int64(7919 * (field_number + 1))
            + np.int64(104729)
        ) % np.int64(HASH_SIZE - 1)
        columns.append(token)

    for cross_number, (left, right) in enumerate(CROSS_PAIRS):
        a = np.asarray(split.X[left][start:end], dtype=np.int64)
        b = np.asarray(split.X[right][start:end], dtype=np.int64)
        token = (
            a * np.int64(1000003)
            + b * np.int64(9176)
            + np.int64(15485863 + 32452843 * cross_number)
        ) % np.int64(HASH_SIZE - 1)
        columns.append(token)

    columns.append(np.full(size, CONSTANT_TOKEN, dtype=np.int64))
    return np.column_stack(columns)


class HashedWideModel(nn.Module):
    def __init__(self, hash_size):
        super().__init__()
        self.weight = nn.Embedding(hash_size, 1, sparse=True)
        nn.init.zeros_(self.weight.weight)

    def forward(self, tokens):
        return self.weight(tokens).squeeze(-1).sum(dim=1)


wide_model = HashedWideModel(HASH_SIZE)
wide_optimizer = torch.optim.SparseAdam(
    wide_model.parameters(),
    lr=0.075,
    betas=(0.9, 0.99),
)

batch_size = 65536
rng = np.random.default_rng(SEED)

wide_model.train()
for epoch in range(3):
    permutation = rng.permutation(n_train)
    epoch_loss_sum = 0.0
    epoch_weight_sum = 0.0

    for batch_start in range(0, n_train, batch_size):
        row_ids = permutation[batch_start:batch_start + batch_size]

        # Construct hashed tokens for arbitrary shuffled rows directly.
        token_columns = []
        for field_number, name in enumerate(WIDE_FIELDS):
            x = np.asarray(train.X[name][row_ids], dtype=np.int64)
            token = (
                x * np.int64(1000003)
                + np.int64(7919 * (field_number + 1))
                + np.int64(104729)
            ) % np.int64(HASH_SIZE - 1)
            token_columns.append(token)

        for cross_number, (left, right) in enumerate(CROSS_PAIRS):
            a = np.asarray(train.X[left][row_ids], dtype=np.int64)
            b = np.asarray(train.X[right][row_ids], dtype=np.int64)
            token = (
                a * np.int64(1000003)
                + b * np.int64(9176)
                + np.int64(15485863 + 32452843 * cross_number)
            ) % np.int64(HASH_SIZE - 1)
            token_columns.append(token)

        token_columns.append(
            np.full(len(row_ids), CONSTANT_TOKEN, dtype=np.int64)
        )
        tokens = torch.from_numpy(np.column_stack(token_columns))
        targets = torch.from_numpy(y_train[row_ids])
        weights = torch.from_numpy(train_weight[row_ids])

        wide_optimizer.zero_grad(set_to_none=True)
        logits = wide_model(tokens)
        losses = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        loss = torch.sum(losses * weights) / torch.sum(weights)
        loss.backward()
        wide_optimizer.step()

        epoch_loss_sum += float(torch.sum(losses * weights).detach())
        epoch_weight_sum += float(torch.sum(weights))

    print(
        "FINDINGS wide_epoch_%d_weighted_logloss=%.6f"
        % (epoch + 1, epoch_loss_sum / max(epoch_weight_sum, 1e-12))
    )


def predict_wide(split):
    result = np.empty(len(split.user_id), dtype=np.float32)
    wide_model.eval()
    with torch.no_grad():
        for start in range(0, len(result), batch_size):
            end = min(start + batch_size, len(result))
            tokens = torch.from_numpy(hashed_tokens(split, start, end))
            result[start:end] = (
                wide_model(tokens).cpu().numpy().astype(np.float32)
            )
    return result


wide_valid = predict_wide(valid)
wide_test = predict_wide(test)

del wide_optimizer, wide_model
gc.collect()

# ----------------------------------------------------------------------
# Family 2: hierarchical varying-coefficient duration-response GLM.
#
# Train-only item and author histories form an offset. Each user then gets
# a ridge-shrunk logistic intercept and nonlinear duration preference.
# This separates "the item is generally long-viewed" from "this user tends
# to long-view videos of this length."
# ----------------------------------------------------------------------


def find_history_rate(histories, entity):
    preferred = entity + "_long_view_rate"
    if preferred in histories:
        return np.asarray(histories[preferred], dtype=np.float32)
    matches = [
        name for name in histories
        if "long_view_rate" in name
    ]
    if not matches:
        raise RuntimeError("No long_view_rate history for " + entity)
    return np.asarray(histories[sorted(matches)[0]], dtype=np.float32)


tr_video_hist = historical_features("train", key="video_id")
va_video_hist = historical_features("valid", key="video_id")
te_video_hist = historical_features("test", key="video_id")

tr_author_hist = historical_features("train", key="author_id")
va_author_hist = historical_features("valid", key="author_id")
te_author_hist = historical_features("test", key="author_id")

tr_video_rate = find_history_rate(tr_video_hist, "video_id")
va_video_rate = find_history_rate(va_video_hist, "video_id")
te_video_rate = find_history_rate(te_video_hist, "video_id")

tr_author_rate = find_history_rate(tr_author_hist, "author_id")
va_author_rate = find_history_rate(va_author_hist, "author_id")
te_author_rate = find_history_rate(te_author_hist, "author_id")

global_rate = float(
    np.sum(train_weight * y_train) / np.sum(train_weight)
)
global_rate = float(np.clip(global_rate, 1e-4, 1.0 - 1e-4))


def safe_logit(rate):
    rate = np.asarray(rate, dtype=np.float32)
    rate = np.where(np.isfinite(rate), rate, global_rate)
    rate = np.clip(rate, 1e-4, 1.0 - 1e-4)
    return np.log(rate / (1.0 - rate)).astype(np.float32)


def popularity_offset(video_rate, author_rate):
    return (
        0.72 * safe_logit(video_rate)
        + 0.28 * safe_logit(author_rate)
    ).astype(np.float32)


offset_train = popularity_offset(tr_video_rate, tr_author_rate)
offset_valid = popularity_offset(va_video_rate, va_author_rate)
offset_test = popularity_offset(te_video_rate, te_author_rate)

log_duration_train = np.log1p(
    np.maximum(
        np.asarray(train.num["duration_ms"], dtype=np.float32), 0.0
    )
)
duration_center = float(np.mean(log_duration_train))
duration_scale = max(float(np.std(log_duration_train)), 1e-4)


def duration_basis(split):
    x = np.log1p(
        np.maximum(
            np.asarray(split.num["duration_ms"], dtype=np.float32), 0.0
        )
    )
    z = np.clip(
        (x - duration_center) / duration_scale, -4.0, 4.0
    ).astype(np.float32)
    return np.column_stack(
        [
            np.ones(len(z), dtype=np.float32),
            z,
            (z * z - 1.0).astype(np.float32),
        ]
    ).astype(np.float32)


Dtr = duration_basis(train)
Dva = duration_basis(valid)
Dte = duration_basis(test)

train_users = np.asarray(train.user_id, dtype=np.int64)
n_users = max(
    int(FEATURE_CARDINALITIES["user_id"]),
    int(np.max(train_users)) + 1,
)
user_coef = np.zeros((n_users, 3), dtype=np.float64)

ridge = np.array([7.0, 18.0, 24.0], dtype=np.float64)
weighted_count = np.bincount(
    train_users,
    weights=train_weight,
    minlength=n_users,
).astype(np.float64)
reliability = weighted_count / (weighted_count + 12.0)

# Batched Newton updates for all users; no row-wise or user-wise Python loop.
for iteration in range(6):
    eta = offset_train.astype(np.float64) + np.sum(
        Dtr.astype(np.float64) * user_coef[train_users], axis=1
    )
    eta = np.clip(eta, -18.0, 18.0)
    probability = 1.0 / (1.0 + np.exp(-eta))

    residual_weight = train_weight.astype(np.float64) * (
        probability - y_train
    )
    curvature = train_weight.astype(np.float64) * (
        probability * (1.0 - probability)
    )

    gradient = np.empty((n_users, 3), dtype=np.float64)
    for j in range(3):
        gradient[:, j] = np.bincount(
            train_users,
            weights=residual_weight * Dtr[:, j],
            minlength=n_users,
        )
    gradient += ridge[None, :] * user_coef

    hessian = np.zeros((n_users, 3, 3), dtype=np.float64)
    for j in range(3):
        for k in range(j, 3):
            values = np.bincount(
                train_users,
                weights=curvature * Dtr[:, j] * Dtr[:, k],
                minlength=n_users,
            )
            hessian[:, j, k] = values
            hessian[:, k, j] = values

    hessian[:, 0, 0] += ridge[0]
    hessian[:, 1, 1] += ridge[1]
    hessian[:, 2, 2] += ridge[2]
    hessian[:, 0, 0] += 1e-4
    hessian[:, 1, 1] += 1e-4
    hessian[:, 2, 2] += 1e-4

    step = np.linalg.solve(hessian, gradient[..., None])[..., 0]
    step = np.clip(step, -1.5, 1.5)
    user_coef -= step
    user_coef *= reliability[:, None]

    print(
        "FINDINGS duration_glm_iter_%d_max_step=%.6f"
        % (iteration + 1, float(np.max(np.abs(step))))
    )


def duration_glm_predict(split, basis, offset):
    users = np.asarray(split.user_id, dtype=np.int64)
    known = users < n_users
    adjustment = np.zeros(len(users), dtype=np.float64)
    adjustment[known] = np.sum(
        basis[known].astype(np.float64) * user_coef[users[known]],
        axis=1,
    )
    return (offset.astype(np.float64) + adjustment).astype(np.float32)


duration_valid = duration_glm_predict(valid, Dva, offset_valid)
duration_test = duration_glm_predict(test, Dte, offset_test)

del Dtr, Dva, Dte
del tr_video_hist, va_video_hist, te_video_hist
del tr_author_hist, va_author_hist, te_author_hist
gc.collect()

# ----------------------------------------------------------------------
# Family 3: Rocchio collaborative content prototype.
#
# Fixed random embeddings turn several categorical item descriptors into
# a shared geometry. A user vector points from their negative centroid to
# their positive centroid. Prediction is cosine alignment with the current
# item's descriptor vector.
# ----------------------------------------------------------------------

PROFILE_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "music_type",
    "upload_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
PROFILE_DIM = 28
profile_rng = np.random.default_rng(SEED + 91)

embedding_tables = {}
for name in PROFILE_FIELDS:
    cardinality = int(FEATURE_CARDINALITIES[name])
    table = profile_rng.normal(
        0.0,
        1.0 / np.sqrt(PROFILE_DIM),
        size=(cardinality, PROFILE_DIM),
    ).astype(np.float32)
    table[0] = 0.0
    embedding_tables[name] = table


def content_vectors(split):
    n = len(split.user_id)
    vectors = np.zeros((n, PROFILE_DIM), dtype=np.float32)
    for name in PROFILE_FIELDS:
        ids = np.asarray(split.X[name], dtype=np.int64)
        vectors += embedding_tables[name][ids]
    norms = np.sqrt(np.sum(vectors * vectors, axis=1, keepdims=True))
    vectors /= np.maximum(norms, 1e-5)
    return vectors


Vtr = content_vectors(train)

positive_weight = train_weight * y_train
negative_weight = train_weight * (1.0 - y_train)

positive_count = np.bincount(
    train_users, weights=positive_weight, minlength=n_users
).astype(np.float32)
negative_count = np.bincount(
    train_users, weights=negative_weight, minlength=n_users
).astype(np.float32)

positive_sum = np.zeros((n_users, PROFILE_DIM), dtype=np.float32)
negative_sum = np.zeros((n_users, PROFILE_DIM), dtype=np.float32)

for j in range(PROFILE_DIM):
    positive_sum[:, j] = np.bincount(
        train_users,
        weights=positive_weight * Vtr[:, j],
        minlength=n_users,
    ).astype(np.float32)
    negative_sum[:, j] = np.bincount(
        train_users,
        weights=negative_weight * Vtr[:, j],
        minlength=n_users,
    ).astype(np.float32)

positive_centroid = positive_sum / (positive_count[:, None] + 8.0)
negative_centroid = negative_sum / (negative_count[:, None] + 8.0)
user_profile = positive_centroid - negative_centroid

profile_reliability = (
    np.minimum(positive_count, negative_count)
    / (np.minimum(positive_count, negative_count) + 6.0)
)
user_profile *= profile_reliability[:, None]

profile_norm = np.sqrt(
    np.sum(user_profile * user_profile, axis=1, keepdims=True)
)
user_profile /= np.maximum(profile_norm, 1e-5)


def rocchio_predict(split):
    vectors = content_vectors(split)
    users = np.asarray(split.user_id, dtype=np.int64)
    result = np.zeros(len(users), dtype=np.float32)
    known = users < n_users
    result[known] = np.sum(
        vectors[known] * user_profile[users[known]], axis=1
    )
    return result


rocchio_valid = rocchio_predict(valid)
rocchio_test = rocchio_predict(test)

del Vtr, positive_sum, negative_sum, positive_centroid, negative_centroid
gc.collect()

# ----------------------------------------------------------------------
# Validation comparison and incumbent blends.
# ----------------------------------------------------------------------

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float32,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float32,
)


def standardize_from_valid(valid_scores, test_scores):
    center = float(np.mean(valid_scores))
    scale = max(float(np.std(valid_scores)), 1e-6)
    return (
        ((valid_scores - center) / scale).astype(np.float32),
        ((test_scores - center) / scale).astype(np.float32),
    )


inc_valid_z, inc_test_z = standardize_from_valid(inc_valid, inc_test)

families = {
    "hashed_explicit_cross_wide": (wide_valid, wide_test),
    "hierarchical_duration_glm": (duration_valid, duration_test),
    "rocchio_content_prototype": (rocchio_valid, rocchio_test),
}

candidate_scores = {}

inc_metric = evaluate(valid.user_id, y_valid, inc_valid_z)
candidate_scores["incumbent"] = float(inc_metric["primary"])

best_name = "incumbent"
best_metric = inc_metric
best_valid = inc_valid_z
best_test = inc_test_z
best_raw_valid = wide_valid
best_alpha = 0.0

blend_alphas = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.65, 0.80,
]

for family_name, (family_valid, family_test) in families.items():
    family_valid_z, family_test_z = standardize_from_valid(
        family_valid, family_test
    )

    standalone_metric = evaluate(
        valid.user_id, y_valid, family_valid_z
    )
    candidate_scores[family_name + "_standalone"] = float(
        standalone_metric["primary"]
    )

    family_best_metric = standalone_metric
    family_best_valid = family_valid_z
    family_best_test = family_test_z
    family_best_alpha = 1.0

    for alpha in blend_alphas:
        blend_valid = (
            (1.0 - alpha) * inc_valid_z + alpha * family_valid_z
        ).astype(np.float32)
        blend_metric = evaluate(valid.user_id, y_valid, blend_valid)

        if blend_metric["primary"] > family_best_metric["primary"]:
            family_best_metric = blend_metric
            family_best_valid = blend_valid
            family_best_test = (
                (1.0 - alpha) * inc_test_z + alpha * family_test_z
            ).astype(np.float32)
            family_best_alpha = alpha

    blend_key = (
        family_name + "_best_blend_a%.2f" % family_best_alpha
    )
    candidate_scores[blend_key] = float(
        family_best_metric["primary"]
    )

    if family_best_metric["primary"] > best_metric["primary"]:
        best_name = blend_key
        best_metric = family_best_metric
        best_valid = family_best_valid
        best_test = family_best_test
        best_raw_valid = family_valid_z
        best_alpha = family_best_alpha

print(
    "FINDINGS selected=%s alpha=%.2f incumbent_primary=%.6f"
    % (best_name, best_alpha, inc_metric["primary"])
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

    # Every possible winner is either an incumbent blend or the incumbent.
    # Save the associated new-family score separately for auditability.
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        best_metric["primary"],
        best_metric["gauc"],
        best_metric["ndcg@5"],
        elapsed,
    )
)