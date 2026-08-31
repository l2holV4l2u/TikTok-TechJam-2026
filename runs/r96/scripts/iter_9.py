import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18437
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "hour", "tag", "upload_type", "music_type", "user_active_degree",
    "is_live_streamer", "is_video_author", "onehot_feat3",
    "onehot_feat8", "video_type",
]
PNN_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "hour", "tag", "upload_type", "music_type", "user_active_degree",
    "onehot_feat3", "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
N_FOLDS = 4
SMOOTH = 12.0
HALF_LIFE = 4.0
EMBED_DIM = 8
BATCH_SIZE = 8192
PRED_BATCH = 32768
PNN_EPOCHS = 2


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    w = np.power(0.5, age.astype(np.float32) / half_life)
    return (w / max(float(w.mean()), 1e-8)).astype(np.float32)


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_indices = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((np.array([-1]), end_indices)))
    row_sizes = np.repeat(sizes, sizes)

    positions = np.arange(n, dtype=np.int64) - start_pos
    ranks = (positions.astype(np.float64) + 0.5) / row_sizes
    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def packed_pair(a, b, b_cardinality):
    return (
        np.asarray(a, dtype=np.int64) * np.int64(b_cardinality)
        + np.asarray(b, dtype=np.int64)
    )


def map_unique_values(unique_keys, values, query, default):
    query = np.asarray(query, dtype=np.int64)
    positions = np.searchsorted(unique_keys, query)
    safe_positions = np.minimum(positions, len(unique_keys) - 1)
    found = (
        (positions < len(unique_keys))
        & (unique_keys[safe_positions] == query)
    )
    result = np.full(len(query), default, dtype=np.float32)
    result[found] = values[safe_positions[found]]
    return result


def cross_fitted_encoding(train_key, valid_key, test_key, y, weights, folds,
                          prior, smooth):
    train_key = np.asarray(train_key, dtype=np.int64)
    unique_keys, inverse = np.unique(train_key, return_inverse=True)
    n_keys = len(unique_keys)

    total_weight = np.bincount(
        inverse, weights=weights, minlength=n_keys
    ).astype(np.float64)
    total_positive = np.bincount(
        inverse, weights=weights * y, minlength=n_keys
    ).astype(np.float64)

    fold_index = inverse * N_FOLDS + folds
    fold_weight = np.bincount(
        fold_index, weights=weights, minlength=n_keys * N_FOLDS
    ).reshape(n_keys, N_FOLDS)
    fold_positive = np.bincount(
        fold_index, weights=weights * y, minlength=n_keys * N_FOLDS
    ).reshape(n_keys, N_FOLDS)

    excluded_weight = total_weight[inverse] - fold_weight[inverse, folds]
    excluded_positive = (
        total_positive[inverse] - fold_positive[inverse, folds]
    )
    train_rate = (
        excluded_positive + smooth * prior
    ) / (excluded_weight + smooth)
    train_count = np.log1p(excluded_weight)

    full_rate = (
        total_positive + smooth * prior
    ) / (total_weight + smooth)
    full_count = np.log1p(total_weight)

    valid_rate = map_unique_values(
        unique_keys, full_rate.astype(np.float32), valid_key, prior
    )
    test_rate = map_unique_values(
        unique_keys, full_rate.astype(np.float32), test_key, prior
    )
    valid_count = map_unique_values(
        unique_keys, full_count.astype(np.float32), valid_key, 0.0
    )
    test_count = map_unique_values(
        unique_keys, full_count.astype(np.float32), test_key, 0.0
    )

    return (
        train_rate.astype(np.float32),
        train_count.astype(np.float32),
        valid_rate,
        valid_count,
        test_rate,
        test_count,
    )


def build_keys(split):
    x = split.X
    return {
        "video": np.asarray(x["video_id"], dtype=np.int64),
        "author": np.asarray(x["author_id"], dtype=np.int64),
        "tag": np.asarray(x["tag"], dtype=np.int64),
        "duration": np.asarray(x["duration_bucket"], dtype=np.int64),
        "author_tag": packed_pair(
            x["author_id"], x["tag"], FEATURE_CARDINALITIES["tag"]
        ),
        "user_tag": packed_pair(
            x["user_id"], x["tag"], FEATURE_CARDINALITIES["tag"]
        ),
        "user_author": packed_pair(
            x["user_id"], x["author_id"],
            FEATURE_CARDINALITIES["author_id"]
        ),
        "user_duration": packed_pair(
            x["user_id"], x["duration_bucket"],
            FEATURE_CARDINALITIES["duration_bucket"]
        ),
        "video_tab": packed_pair(
            x["video_id"], x["tab"], FEATURE_CARDINALITIES["tab"]
        ),
        "author_tab": packed_pair(
            x["author_id"], x["tab"], FEATURE_CARDINALITIES["tab"]
        ),
        "user_video": packed_pair(
            x["user_id"], x["video_id"], FEATURE_CARDINALITIES["video_id"]
        ),
    }


def make_lgb_matrix(split, encoded):
    columns = []
    categorical_indices = []

    for field in CAT_FIELDS:
        categorical_indices.append(len(columns))
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(value, 0.0)).astype(np.float32))

    for j in range(encoded.shape[1]):
        columns.append(encoded[:, j])

    return (
        np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32),
        categorical_indices,
    )


def categorical_matrix(split, fields):
    cards = [FEATURE_CARDINALITIES[f] for f in fields]
    offsets = np.cumsum(
        np.asarray([0] + cards[:-1], dtype=np.int64)
    )
    matrix = np.stack(
        [
            np.asarray(split.X[field], dtype=np.int64) + offsets[j]
            for j, field in enumerate(fields)
        ],
        axis=1,
    )
    return np.ascontiguousarray(matrix, dtype=np.int64)


class ProductNetwork(nn.Module):
    def __init__(self, total_categories, n_fields, n_numeric, bias):
        super().__init__()
        self.n_fields = n_fields
        self.embedding = nn.Embedding(total_categories, EMBED_DIM)
        self.linear = nn.Embedding(total_categories, 1)

        pair_count = n_fields * (n_fields - 1) // 2
        input_dim = n_fields * EMBED_DIM + pair_count + n_numeric

        self.register_buffer(
            "pair_i",
            torch.tensor(
                [i for i in range(n_fields) for j in range(i + 1, n_fields)],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "pair_j",
            torch.tensor(
                [j for i in range(n_fields) for j in range(i + 1, n_fields)],
                dtype=torch.long,
            ),
        )

        self.numeric_norm = nn.LayerNorm(n_numeric)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, numeric):
        emb = self.embedding(x)
        products = (
            emb[:, self.pair_i, :] * emb[:, self.pair_j, :]
        ).sum(dim=2)
        numeric = self.numeric_norm(numeric)
        deep_input = torch.cat(
            [emb.flatten(1), products, numeric], dim=1
        )
        deep = self.network(deep_input).squeeze(1)
        wide = self.linear(x).sum(dim=1).squeeze(1)
        return self.bias + wide + deep


def train_pnn(model, x, numeric, y, sample_weights):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator().manual_seed(SEED + 11)
    n = len(y)

    model.train()
    for _ in range(PNN_EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x[idx], numeric[idx])
            loss = (
                criterion(logits, y[idx]) * sample_weights[idx]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_pnn(model, x, numeric):
    result = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), PRED_BATCH):
            end = min(start + PRED_BATCH, len(x))
            xb = torch.from_numpy(x[start:end])
            nb = torch.from_numpy(numeric[start:end])
            result[start:end] = model(xb, nb).cpu().numpy()
    return result


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
train_weights = recency_weights(train.date, HALF_LIFE)
prior = float(np.sum(train_weights * y_train) / np.sum(train_weights))

date_values = np.asarray(train.date, dtype=np.int32)
unique_dates = np.unique(date_values)
date_to_index = {
    int(date): i for i, date in enumerate(unique_dates.tolist())
}
folds = np.asarray(
    [date_to_index[int(date)] % N_FOLDS for date in date_values],
    dtype=np.int64,
)

train_keys = build_keys(train)
valid_keys = build_keys(valid)
test_keys = build_keys(test)

encoded_train_columns = []
encoded_valid_columns = []
encoded_test_columns = []
rate_train_columns = []
rate_valid_columns = []
rate_test_columns = []
encoding_names = []

for name in train_keys:
    (
        train_rate, train_count,
        valid_rate, valid_count,
        test_rate, test_count,
    ) = cross_fitted_encoding(
        train_keys[name],
        valid_keys[name],
        test_keys[name],
        y_train,
        train_weights,
        folds,
        prior,
        SMOOTH,
    )

    encoded_train_columns.extend([train_rate, train_count])
    encoded_valid_columns.extend([valid_rate, valid_count])
    encoded_test_columns.extend([test_rate, test_count])
    rate_train_columns.append(train_rate)
    rate_valid_columns.append(valid_rate)
    rate_test_columns.append(test_rate)
    encoding_names.append(name)

encoded_train = np.ascontiguousarray(
    np.stack(encoded_train_columns, axis=1), dtype=np.float32
)
encoded_valid = np.ascontiguousarray(
    np.stack(encoded_valid_columns, axis=1), dtype=np.float32
)
encoded_test = np.ascontiguousarray(
    np.stack(encoded_test_columns, axis=1), dtype=np.float32
)
rate_train = np.ascontiguousarray(
    np.stack(rate_train_columns, axis=1), dtype=np.float32
)
rate_valid = np.ascontiguousarray(
    np.stack(rate_valid_columns, axis=1), dtype=np.float32
)
rate_test = np.ascontiguousarray(
    np.stack(rate_test_columns, axis=1), dtype=np.float32
)

# Family 1: non-parametric robust log-odds aggregation.
def encoder_aggregate(rate_matrix):
    clipped = np.clip(rate_matrix.astype(np.float64), 1e-4, 1.0 - 1e-4)
    logits = np.log(clipped / (1.0 - clipped))
    # Personalized crosses receive more mass than global entity rates.
    weights = np.asarray(
        [0.08, 0.08, 0.04, 0.04, 0.10, 0.15, 0.15, 0.12, 0.07, 0.07, 0.10],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return logits @ weights


aggregate_valid = encoder_aggregate(rate_valid)
aggregate_test = encoder_aggregate(rate_test)

# Families 2 and 3: pointwise binary boosting and grouped LambdaMART.
lgb_train_x, categorical_indices = make_lgb_matrix(train, encoded_train)
lgb_valid_x, _ = make_lgb_matrix(valid, encoded_valid)
lgb_test_x, _ = make_lgb_matrix(test, encoded_test)

common_params = {
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.84,
    "bagging_fraction": 0.84,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.2,
    "max_bin": 127,
    "num_threads": min(16, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "verbose": -1,
}

binary_dataset = lgb.Dataset(
    lgb_train_x,
    label=y_train,
    weight=train_weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
binary_params = dict(common_params)
binary_params.update({
    "objective": "binary",
    "metric": "None",
})
binary_model = lgb.train(
    binary_params, binary_dataset, num_boost_round=300
)
binary_valid = binary_model.predict(lgb_valid_x)
binary_test = binary_model.predict(lgb_test_x)

train_users = np.asarray(train.user_id, dtype=np.int64)
train_order = np.argsort(train_users, kind="stable")
sorted_users = train_users[train_order]
_, group_counts = np.unique(sorted_users, return_counts=True)

rank_dataset = lgb.Dataset(
    lgb_train_x[train_order],
    label=y_train[train_order],
    weight=train_weights[train_order],
    group=group_counts,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
rank_params = dict(common_params)
rank_params.update({
    "objective": "lambdarank",
    "metric": "None",
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 5,
})
rank_model = lgb.train(
    rank_params, rank_dataset, num_boost_round=300
)
rank_valid = rank_model.predict(lgb_valid_x)
rank_test = rank_model.predict(lgb_test_x)

del binary_dataset, rank_dataset, lgb_train_x
gc.collect()

# Family 4: explicit pairwise-product neural network over the same encodings.
pnn_train_x = categorical_matrix(train, PNN_FIELDS)
pnn_valid_x = categorical_matrix(valid, PNN_FIELDS)
pnn_test_x = categorical_matrix(test, PNN_FIELDS)

pnn_numeric_train = encoded_train
pnn_numeric_valid = encoded_valid
pnn_numeric_test = encoded_test

positive_rate = float(y_train.mean())
initial_bias = float(
    np.log(positive_rate / max(1.0 - positive_rate, 1e-8))
)
total_categories = int(
    sum(FEATURE_CARDINALITIES[f] for f in PNN_FIELDS)
)
pnn = ProductNetwork(
    total_categories,
    len(PNN_FIELDS),
    pnn_numeric_train.shape[1],
    initial_bias,
)
pnn = train_pnn(
    pnn,
    torch.from_numpy(pnn_train_x),
    torch.from_numpy(pnn_numeric_train),
    torch.from_numpy(y_train),
    torch.from_numpy(train_weights),
)
pnn_valid = predict_pnn(pnn, pnn_valid_x, pnn_numeric_valid)
pnn_test = predict_pnn(pnn, pnn_test_x, pnn_numeric_test)

own_valid = {
    "crossfit_nonparametric": np.asarray(aggregate_valid, dtype=np.float64),
    "crossfit_binary_lgb": np.asarray(binary_valid, dtype=np.float64),
    "crossfit_lambdamart": np.asarray(rank_valid, dtype=np.float64),
    "crossfit_pnn": np.asarray(pnn_valid, dtype=np.float64),
}
own_test = {
    "crossfit_nonparametric": np.asarray(aggregate_test, dtype=np.float64),
    "crossfit_binary_lgb": np.asarray(binary_test, dtype=np.float64),
    "crossfit_lambdamart": np.asarray(rank_test, dtype=np.float64),
    "crossfit_pnn": np.asarray(pnn_test, dtype=np.float64),
}

# A fifth prediction family: rank aggregation of structurally distinct models.
valid_family_ranks = [
    rank_percentile(valid.user_id, own_valid[name])
    for name in own_valid
]
test_family_ranks = [
    rank_percentile(test.user_id, own_test[name])
    for name in own_test
]
own_valid["crossfit_family_ensemble"] = np.mean(
    np.stack(valid_family_ranks, axis=1), axis=1
)
own_test["crossfit_family_ensemble"] = np.mean(
    np.stack(test_family_ranks, axis=1), axis=1
)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

candidate_scores = {}
candidate_metrics = {}
candidate_specs = {}

for family in own_valid:
    standalone = family + "_standalone"
    candidate_scores[standalone] = own_valid[family]
    candidate_metrics[standalone] = evaluate(
        valid.user_id, valid.y, own_valid[family]
    )
    candidate_specs[standalone] = (family, None)

    family_valid_rank = rank_percentile(valid.user_id, own_valid[family])
    for alpha in (0.20, 0.40, 0.60, 0.80):
        name = f"{family}_blend_{alpha:.2f}"
        score = (
            alpha * family_valid_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        candidate_scores[name] = score
        candidate_metrics[name] = evaluate(
            valid.user_id, valid.y, score
        )
        candidate_specs[name] = (family, alpha)

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_scores[best_name]
best_family, best_alpha = candidate_specs[best_name]

if best_alpha is None:
    best_test = own_test[best_family]
else:
    family_test_rank = rank_percentile(
        test.user_id, own_test[best_family]
    )
    best_test = (
        best_alpha * family_test_rank
        + (1.0 - best_alpha) * inc_test_rank
    )

print("CANDIDATES " + json.dumps(
    {
        name: float(metrics["primary"])
        for name, metrics in candidate_metrics.items()
    },
    sort_keys=True,
))
print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "encoding_keys": encoding_names,
    "date_crossfit_folds": N_FOLDS,
    "target_smoothing": SMOOTH,
    "main_sample_weight_half_life_days": HALF_LIFE,
    "weighted_train_prior": prior,
}, sort_keys=True))

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
    if best_alpha is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_valid[best_family], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))