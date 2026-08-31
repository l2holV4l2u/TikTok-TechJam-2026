import os
import time
import json
import gc

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 84173
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag", "upload_type",
    "onehot_feat3", "onehot_feat8", "duration_bucket", "music_type",
    "user_active_degree", "register_days_bucket", "register_days_range",
    "onehot_feat1", "onehot_feat7", "onehot_feat0",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HISTORY_SUFFIXES = [
    "train_count_log1p", "long_view_rate", "is_click_rate",
    "play_time_ms_logmean",
]
TREND_FIELDS = [
    "video_id", "author_id", "tag", "tab", "duration_bucket",
    "upload_type", "onehot_feat3",
]


def safe_logit(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(values) - np.log1p(-values)


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    result = np.power(0.5, age.astype(np.float64) / half_life)
    result /= max(float(np.mean(result)), 1e-12)
    return result.astype(np.float32)


def build_encoding_tables(train, weights):
    labels = np.asarray(train.y, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    prior = float(np.sum(weights64 * labels) / np.sum(weights64))
    tables = {}

    for field in CAT_FIELDS:
        ids = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(
            ids, weights=weights64, minlength=cardinality
        ).astype(np.float64)
        positives = np.bincount(
            ids, weights=weights64 * labels, minlength=cardinality
        ).astype(np.float64)
        tables[field] = (counts, positives)

    return prior, tables


def build_dense_matrix(split, split_name, train, weights, prior, tables):
    is_train = split_name == "train"
    columns = []
    labels = np.asarray(train.y, dtype=np.float64) if is_train else None
    row_weights = np.asarray(weights, dtype=np.float64) if is_train else None
    prior_logit = safe_logit(prior)

    for field in CAT_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        counts, positives = tables[field]
        row_counts = counts[ids].copy()
        row_positives = positives[ids].copy()

        if is_train:
            row_counts -= row_weights
            row_positives -= row_weights * labels
            np.maximum(row_counts, 0.0, out=row_counts)
            np.maximum(row_positives, 0.0, out=row_positives)

        posterior = (
            row_positives + 20.0 * prior
        ) / (row_counts + 20.0)

        columns.append(
            (safe_logit(posterior) - prior_logit).astype(np.float32)
        )
        columns.append(np.log1p(row_counts).astype(np.float32))

    for field in NUM_FIELDS:
        values = np.asarray(split.num[field], dtype=np.float32)
        values = np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        )
        columns.append(
            np.log1p(np.maximum(values, 0.0)).astype(np.float32)
        )

    for entity in ("video_id", "author_id"):
        histories = historical_features(split_name, key=entity)
        for suffix in HISTORY_SUFFIXES:
            key = entity + "_" + suffix
            values = np.asarray(histories[key], dtype=np.float32)
            values = np.nan_to_num(
                values, nan=0.0, posinf=0.0, neginf=0.0
            )
            if suffix.endswith("_rate"):
                values = (
                    safe_logit(values) - prior_logit
                ).astype(np.float32)
            columns.append(values.astype(np.float32))
        del histories

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    angle = 2.0 * np.pi * hour / 24.0
    columns.append(np.sin(angle).astype(np.float32))
    columns.append(np.cos(angle).astype(np.float32))

    date_offset = (
        np.asarray(split.date, dtype=np.int64) -
        int(np.min(np.asarray(train.date, dtype=np.int64)))
    ).astype(np.float32)
    columns.append((date_offset / 10.0).astype(np.float32))

    return np.ascontiguousarray(
        np.column_stack(columns), dtype=np.float32
    )


def standardize(train_matrix, valid_matrix, test_matrix):
    mean = np.mean(train_matrix, axis=0, dtype=np.float64)
    std = np.std(train_matrix, axis=0, dtype=np.float64)
    std[~np.isfinite(std) | (std < 1e-5)] = 1.0
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    matrices = []
    for matrix in (train_matrix, valid_matrix, test_matrix):
        transformed = np.asarray(
            (matrix - mean) / std, dtype=np.float32
        )
        np.clip(transformed, -8.0, 8.0, out=transformed)
        matrices.append(transformed)
    return tuple(matrices)


def predict_linear(matrix, coefficients, intercept):
    return (
        np.asarray(matrix @ coefficients, dtype=np.float32) +
        np.float32(intercept)
    )


def train_listwise_linear(
    x_train, labels, users, dates, x_valid, x_test
):
    labels_np = np.asarray(labels, dtype=np.float32)
    users_np = np.asarray(users, dtype=np.int64)
    dates_np = np.asarray(dates, dtype=np.int64)

    _, group_ids = np.unique(users_np, return_inverse=True)
    group_ids = group_ids.astype(np.int64)
    n_groups = int(group_ids.max()) + 1

    group_count = np.bincount(
        group_ids, minlength=n_groups
    ).astype(np.int64)
    group_positive = np.bincount(
        group_ids, weights=labels_np, minlength=n_groups
    ).astype(np.float32)

    mixed = (
        (group_positive > 0.0) &
        (group_positive < group_count)
    )

    group_latest = np.full(n_groups, dates_np.min(), dtype=np.int64)
    np.maximum.at(group_latest, group_ids, dates_np)
    group_weight = np.power(
        0.5,
        (dates_np.max() - group_latest).astype(np.float32) / 4.0
    )
    group_weight *= (
        group_positive /
        np.maximum(float(np.mean(group_positive[mixed])), 1e-6)
    )
    group_weight[~mixed] = 0.0
    group_weight /= max(
        float(np.mean(group_weight[mixed])), 1e-8
    )

    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(labels_np)
    group_tensor = torch.from_numpy(group_ids)
    positive_count_tensor = torch.from_numpy(
        np.maximum(group_positive, 1.0).astype(np.float32)
    )
    group_weight_tensor = torch.from_numpy(group_weight.astype(np.float32))
    mixed_tensor = torch.from_numpy(mixed)

    coefficients = torch.zeros(
        x_train.shape[1], dtype=torch.float32, requires_grad=True
    )
    intercept = torch.zeros(
        (), dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.AdamW(
        [coefficients, intercept], lr=0.035, weight_decay=2e-3
    )

    for _ in range(55):
        optimizer.zero_grad(set_to_none=True)
        row_scores = x_tensor.mv(coefficients) + intercept

        group_max = torch.full(
            (n_groups,), -torch.inf, dtype=torch.float32
        )
        group_max.scatter_reduce_(
            0, group_tensor, row_scores.detach(),
            reduce="amax", include_self=True
        )

        shifted_exp = torch.exp(
            torch.clamp(
                row_scores - group_max[group_tensor],
                min=-40.0, max=20.0
            )
        )
        group_exp_sum = torch.zeros(n_groups, dtype=torch.float32)
        group_exp_sum.scatter_add_(0, group_tensor, shifted_exp)
        group_logsumexp = (
            group_max + torch.log(group_exp_sum.clamp_min(1e-12))
        )

        positive_score_sum = torch.zeros(n_groups, dtype=torch.float32)
        positive_score_sum.scatter_add_(
            0, group_tensor, row_scores * y_tensor
        )
        mean_positive_score = (
            positive_score_sum / positive_count_tensor
        )

        group_losses = group_logsumexp - mean_positive_score
        selected_loss = (
            group_losses[mixed_tensor] *
            group_weight_tensor[mixed_tensor]
        ).mean()
        regularizer = 5e-4 * torch.mean(coefficients.square())
        loss = selected_loss + regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [coefficients, intercept], 10.0
        )
        optimizer.step()

    coef_np = coefficients.detach().cpu().numpy().astype(np.float32)
    intercept_np = float(intercept.detach().cpu())
    valid_scores = predict_linear(x_valid, coef_np, intercept_np)
    test_scores = predict_linear(x_test, coef_np, intercept_np)
    return valid_scores, test_scores


def train_diagonal_qda(
    x_train, labels, weights, x_valid, x_test
):
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    x64 = np.asarray(x_train, dtype=np.float64)

    class_stats = []
    for target in (0.0, 1.0):
        class_weight = weights * (labels == target)
        total = max(float(class_weight.sum()), 1e-12)
        mean = np.sum(
            x64 * class_weight[:, None], axis=0
        ) / total
        variance = np.sum(
            ((x64 - mean) ** 2) * class_weight[:, None],
            axis=0
        ) / total
        variance = np.maximum(variance, 0.08)
        class_stats.append((mean, variance))

    prior = float(np.sum(weights * labels) / np.sum(weights))
    mean0, var0 = class_stats[0]
    mean1, var1 = class_stats[1]

    def score(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        result = np.empty(len(matrix), dtype=np.float32)
        batch_size = 65536
        prior_odds = float(safe_logit(prior))
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            xb = matrix[start:end]
            logp0 = -0.5 * np.sum(
                np.log(var0) + ((xb - mean0) ** 2) / var0,
                axis=1
            )
            logp1 = -0.5 * np.sum(
                np.log(var1) + ((xb - mean1) ** 2) / var1,
                axis=1
            )
            result[start:end] = np.asarray(
                logp1 - logp0 + prior_odds, dtype=np.float32
            )
        return result

    return score(x_valid), score(x_test)


def build_temporal_trend_tables(train, weights, prior):
    labels = np.asarray(train.y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    train_dates = np.asarray(train.date, dtype=np.int64)
    day = (train_dates - train_dates.min()).astype(np.float64)
    tables = {}

    for field in TREND_FIELDS:
        ids = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])

        s0 = np.bincount(
            ids, weights=weights, minlength=cardinality
        ).astype(np.float64)
        s1 = np.bincount(
            ids, weights=weights * day, minlength=cardinality
        ).astype(np.float64)
        s2 = np.bincount(
            ids, weights=weights * day * day, minlength=cardinality
        ).astype(np.float64)
        sy = np.bincount(
            ids, weights=weights * labels, minlength=cardinality
        ).astype(np.float64)
        syt = np.bincount(
            ids, weights=weights * labels * day,
            minlength=cardinality
        ).astype(np.float64)

        safe_count = np.maximum(s0, 1e-8)
        mean_t = s1 / safe_count
        mean_y = (sy + 25.0 * prior) / (s0 + 25.0)
        centered_denominator = (
            s2 - (s1 * s1) / safe_count
        )
        centered_numerator = (
            syt - (s1 * sy) / safe_count
        )
        slope = centered_numerator / (
            centered_denominator + 35.0
        )
        slope *= s0 / (s0 + 50.0)
        slope = np.clip(slope, -0.012, 0.012)

        tables[field] = (
            mean_t.astype(np.float32),
            mean_y.astype(np.float32),
            slope.astype(np.float32),
            np.log1p(s0).astype(np.float32),
        )

    return int(train_dates.min()), tables


def temporal_trend_scores(split, base_date, tables, prior):
    row_day = (
        np.asarray(split.date, dtype=np.int64) - base_date
    ).astype(np.float32)
    effects = []

    for field in TREND_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        mean_t, mean_y, slope, log_count = tables[field]
        predicted_rate = (
            mean_y[ids] +
            slope[ids] * (row_day - mean_t[ids])
        )
        confidence = 1.0 - np.exp(-log_count[ids] / 3.0)
        predicted_rate = (
            confidence * predicted_rate +
            (1.0 - confidence) * prior
        )
        effects.append(
            (safe_logit(predicted_rate) - safe_logit(prior))
            .astype(np.float32)
        )

    return np.mean(np.column_stack(effects), axis=1).astype(np.float32)


def user_center_scale(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    means = (
        np.bincount(inverse, weights=scores) /
        np.maximum(counts, 1)
    )
    centered = scores - means[inverse]
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return centered / scale


train = load("train")
valid = load("valid")
test = load("test")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

weights = recency_weights(train.date, half_life=4.0)
prior, encoding_tables = build_encoding_tables(train, weights)

x_train = build_dense_matrix(
    train, "train", train, weights, prior, encoding_tables
)
x_valid = build_dense_matrix(
    valid, "valid", train, weights, prior, encoding_tables
)
x_test = build_dense_matrix(
    test, "test", train, weights, prior, encoding_tables
)
x_train, x_valid, x_test = standardize(
    x_train, x_valid, x_test
)

families = {}

listwise_valid, listwise_test = train_listwise_linear(
    x_train=x_train,
    labels=train_y,
    users=train.user_id,
    dates=train.date,
    x_valid=x_valid,
    x_test=x_test,
)
families["listwise_softmax_linear"] = (
    listwise_valid, listwise_test
)

qda_valid, qda_test = train_diagonal_qda(
    x_train=x_train,
    labels=train_y,
    weights=weights,
    x_valid=x_valid,
    x_test=x_test,
)
families["diagonal_generative_qda"] = (
    qda_valid, qda_test
)

base_date, trend_tables = build_temporal_trend_tables(
    train, weights, prior
)
trend_valid = temporal_trend_scores(
    valid, base_date, trend_tables, prior
)
trend_test = temporal_trend_scores(
    test, base_date, trend_tables, prior
)
families["entity_rate_temporal_forecast"] = (
    trend_valid, trend_test
)

del x_train, x_valid, x_test
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not (
    os.path.exists(inc_valid_path) and
    os.path.exists(inc_test_path)
):
    raise FileNotFoundError(
        "Trusted incumbent predictions are unavailable"
    )

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)
if len(inc_valid) != len(valid_y) or len(inc_test) != len(test_users):
    raise ValueError("Trusted incumbent prediction length mismatch")

inc_valid_norm = user_center_scale(
    inc_valid, valid_users
)
inc_test_norm = user_center_scale(
    inc_test, test_users
)

candidate_metrics = {}
candidate_payloads = {}

inc_metric = evaluate(
    valid_users, valid_y, inc_valid
)
candidate_metrics["trusted_incumbent"] = float(
    inc_metric["primary"]
)
candidate_payloads["trusted_incumbent"] = (
    inc_valid, inc_test, None, False
)

for family_name, payload in families.items():
    own_valid, own_test = payload
    own_metric = evaluate(
        valid_users, valid_y, own_valid
    )
    candidate_metrics[family_name] = float(
        own_metric["primary"]
    )
    candidate_payloads[family_name] = (
        own_valid, own_test, own_valid, False
    )

    own_valid_norm = user_center_scale(
        own_valid, valid_users
    )
    own_test_norm = user_center_scale(
        own_test, test_users
    )

    for alpha in (0.10, 0.20, 0.35, 0.50, 0.70):
        blend_valid = (
            (1.0 - alpha) * inc_valid_norm +
            alpha * own_valid_norm
        )
        blend_test = (
            (1.0 - alpha) * inc_test_norm +
            alpha * own_test_norm
        )
        name = family_name + "_blend_" + str(alpha)
        metric = evaluate(
            valid_users, valid_y, blend_valid
        )
        candidate_metrics[name] = float(
            metric["primary"]
        )
        candidate_payloads[name] = (
            blend_valid, blend_test, own_valid, True
        )

best_name = max(
    candidate_metrics, key=candidate_metrics.get
)
valid_scores, test_scores, raw_valid_scores, is_blend = (
    candidate_payloads[best_name]
)
final_metric = evaluate(
    valid_users, valid_y, valid_scores
)

standalone_report = {
    name: candidate_metrics[name]
    for name in ["trusted_incumbent"] + list(families.keys())
}
print(
    "FINDINGS standalone=" +
    json.dumps(standalone_report, sort_keys=True)
)
print(
    "FINDINGS selected=" + best_name
)
print(
    "CANDIDATES " +
    json.dumps(candidate_metrics, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )
    if is_blend and raw_valid_scores is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid_scores, dtype=np.float64)
        )

elapsed = time.time() - START
print(
    "METRICS " +
    json.dumps({
        "primary": float(final_metric["primary"]),
        "gauc": float(final_metric["gauc"]),
        "ndcg@5": float(final_metric["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)