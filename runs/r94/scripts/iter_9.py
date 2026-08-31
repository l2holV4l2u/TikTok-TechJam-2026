import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18427
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
HALF_LIFE = 4.0
EMBED_DIM = 16
SETWISE_EPOCHS = 4
USERS_PER_BATCH = 512
LR = 1.0e-3

offsets = []
total_cardinality = 0
for field in FIELDS:
    offsets.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)


def make_categorical_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int32)
            for field in FIELDS
        ]),
        dtype=np.int32,
    )


def make_offset_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + offsets[j]
            for j, field in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def recency_weights(dates, half_life=HALF_LIFE):
    dates = np.asarray(dates, dtype=np.int32)
    latest = int(dates.max())
    weights = np.power(
        2.0,
        (dates.astype(np.float64) - latest) / float(half_life),
    )
    weights /= weights.mean()
    return weights.astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    group_index = np.cumsum(starts_mask) - 1

    positions = np.arange(n, dtype=np.int64) - starts[group_index]
    sizes = np.diff(np.append(starts, n))
    denominators = np.maximum(sizes[group_index] - 1, 1)

    sorted_ranks = positions.astype(np.float64) / denominators
    sorted_ranks[sizes[group_index] == 1] = 0.5

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


class FieldWeightedFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.linear = nn.Embedding(total_cardinality, 1)
        self.pair_weight = nn.Parameter(
            torch.ones(len(FIELDS), len(FIELDS))
        )
        self.bias = nn.Parameter(torch.zeros(()))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = self.embedding(x)
        wide = self.linear(x).squeeze(-1).sum(dim=1)

        interaction = torch.zeros_like(wide)
        for i in range(len(FIELDS)):
            for j in range(i + 1, len(FIELDS)):
                dot = torch.sum(e[:, i, :] * e[:, j, :], dim=1)
                interaction = interaction + self.pair_weight[i, j] * dot

        return self.bias + wide + interaction


class ProductNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.linear = nn.Embedding(total_cardinality, 1)
        num_pairs = len(FIELDS) * (len(FIELDS) - 1) // 2
        input_dim = len(FIELDS) * EMBED_DIM + num_pairs

        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = self.embedding(x)
        products = []
        for i in range(len(FIELDS)):
            for j in range(i + 1, len(FIELDS)):
                products.append(
                    torch.sum(e[:, i, :] * e[:, j, :], dim=1, keepdim=True)
                )

        product_features = torch.cat(products, dim=1)
        deep_input = torch.cat(
            [e.flatten(start_dim=1), product_features], dim=1
        )
        wide = self.linear(x).squeeze(-1).sum(dim=1)
        return self.bias + wide + self.network(deep_input).squeeze(1)


def prepare_grouped_training(train):
    user_ids = np.asarray(train.user_id, dtype=np.int64)
    row_ids = np.arange(user_ids.size, dtype=np.int64)
    order = np.lexsort((row_ids, user_ids))
    sorted_users = user_ids[order]

    start_mask = np.empty(sorted_users.size, dtype=bool)
    start_mask[0] = True
    start_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(start_mask)
    ends = np.append(starts[1:], sorted_users.size)

    return order, starts, ends


def setwise_loss(scores, labels, row_weights, group_sizes):
    num_groups = int(group_sizes.numel())
    group_ids = torch.repeat_interleave(
        torch.arange(num_groups, dtype=torch.long),
        group_sizes,
    )

    log_weights = torch.log(torch.clamp(row_weights, min=1.0e-8))
    weighted_scores = scores + log_weights

    maxima = torch.full(
        (num_groups,),
        -torch.inf,
        dtype=scores.dtype,
    )
    maxima.scatter_reduce_(
        0, group_ids, weighted_scores, reduce="amax", include_self=True
    )

    exp_values = torch.exp(weighted_scores - maxima[group_ids])
    exp_sums = torch.zeros(num_groups, dtype=scores.dtype)
    exp_sums.scatter_add_(0, group_ids, exp_values)
    log_partition = maxima + torch.log(torch.clamp(exp_sums, min=1.0e-12))

    positive_weights = row_weights * labels
    positive_weight_sum = torch.zeros(num_groups, dtype=scores.dtype)
    positive_weight_sum.scatter_add_(0, group_ids, positive_weights)

    positive_score_sum = torch.zeros(num_groups, dtype=scores.dtype)
    positive_score_sum.scatter_add_(
        0, group_ids, positive_weights * scores
    )

    positive_counts = torch.zeros(num_groups, dtype=scores.dtype)
    positive_counts.scatter_add_(0, group_ids, labels)

    mixed = (
        (positive_counts > 0.0)
        & (positive_counts < group_sizes.to(scores.dtype))
    )
    if not bool(mixed.any()):
        return scores.sum() * 0.0

    target_score = positive_score_sum / torch.clamp(
        positive_weight_sum, min=1.0e-8
    )
    return (log_partition[mixed] - target_score[mixed]).mean()


def train_setwise(model, x_np, labels_np, weights_np, order, starts, ends, seed):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1.0e-6
    )
    generator = np.random.default_rng(seed)
    num_users = starts.size

    model.train()
    for _ in range(SETWISE_EPOCHS):
        user_order = generator.permutation(num_users)

        for batch_start in range(0, num_users, USERS_PER_BATCH):
            selected_users = user_order[
                batch_start:batch_start + USERS_PER_BATCH
            ]

            selected_users = selected_users[
                np.argsort(starts[selected_users])
            ]

            row_parts = [
                order[starts[u]:ends[u]] for u in selected_users
            ]
            rows = np.concatenate(row_parts)
            sizes = np.asarray(
                [ends[u] - starts[u] for u in selected_users],
                dtype=np.int64,
            )

            xb = torch.from_numpy(x_np[rows])
            yb = torch.from_numpy(labels_np[rows])
            wb = torch.from_numpy(weights_np[rows])
            group_sizes = torch.from_numpy(sizes)

            optimizer.zero_grad(set_to_none=True)
            scores = model(xb)
            loss = setwise_loss(scores, yb, wb, group_sizes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_torch(model, x_np):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, x_np.shape[0], 32768):
            end = min(start + 32768, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            result[start:end] = (
                model(xb).cpu().numpy().astype(np.float64)
            )
    return result


train = load("train")
valid = load("valid")

x_train_cat = make_categorical_matrix(train)
x_valid_cat = make_categorical_matrix(valid)
x_train_offset = make_offset_matrix(train)
x_valid_offset = make_offset_matrix(valid)

train_labels = np.asarray(train.y, dtype=np.float32)
train_weights = recency_weights(train.date)
group_order, group_starts, group_ends = prepare_grouped_training(train)

print(
    "FINDINGS "
    + json.dumps({
        "objective": "recency_weighted_exposure_setwise",
        "half_life_days": HALF_LIFE,
        "mixed_train_users": int(np.sum([
            0 < train_labels[group_order[s:e]].sum() < (e - s)
            for s, e in zip(group_starts, group_ends)
        ])),
    }, sort_keys=True)
)

# Family 1: recency-weighted binary gradient boosting.
lgb_train = lgb.Dataset(
    x_train_cat,
    label=train_labels,
    weight=train_weights,
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=False,
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 150,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "num_threads": max(1, min(16, os.cpu_count() or 1)),
    "verbose": -1,
}

gbdt_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=260,
)
gbdt_valid = gbdt_model.predict(
    x_valid_cat, num_iteration=gbdt_model.current_iteration()
).astype(np.float64)

# Family 2: field-weighted factorization trained with an exposure-setwise loss.
torch.manual_seed(SEED + 100)
fwfm_model = train_setwise(
    FieldWeightedFM(),
    x_train_offset,
    train_labels,
    train_weights,
    group_order,
    group_starts,
    group_ends,
    SEED + 101,
)
fwfm_valid = predict_torch(fwfm_model, x_valid_offset)

# Family 3: product-network prediction trained with the same setwise comparisons.
torch.manual_seed(SEED + 200)
pnn_model = train_setwise(
    ProductNetwork(),
    x_train_offset,
    train_labels,
    train_weights,
    group_order,
    group_starts,
    group_ends,
    SEED + 201,
)
pnn_valid = predict_torch(pnn_model, x_valid_offset)

family_valid = {
    "binary_gbdt": gbdt_valid,
    "setwise_fwfm": fwfm_valid,
    "setwise_product_network": pnn_valid,
}

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores unavailable")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test scores unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {}
candidate_arrays = {}
candidate_family = {}
candidate_alpha = {}

for family_name, raw_scores in family_valid.items():
    standalone_result = evaluate(valid.user_id, valid.y, raw_scores)
    candidate_scores[family_name] = float(standalone_result["primary"])
    candidate_arrays[family_name] = raw_scores
    candidate_family[family_name] = family_name
    candidate_alpha[family_name] = 1.0

    own_rank = within_user_rank(valid.user_id, raw_scores)
    for alpha in (0.10, 0.20, 0.35, 0.50, 0.70):
        name = f"{family_name}_blend_{alpha:.2f}"
        blended = alpha * own_rank + (1.0 - alpha) * inc_valid_rank
        result = evaluate(valid.user_id, valid.y, blended)

        candidate_scores[name] = float(result["primary"])
        candidate_arrays[name] = blended
        candidate_family[name] = family_name
        candidate_alpha[name] = alpha

winner_name = max(candidate_scores, key=candidate_scores.get)
winner_family = candidate_family[winner_name]
winner_alpha = candidate_alpha[winner_name]
valid_scores = candidate_arrays[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "winner": winner_name,
        "winner_family": winner_family,
        "own_weight": float(winner_alpha),
        "rank_agreement_with_incumbent": float(np.corrcoef(
            within_user_rank(valid.user_id, family_valid[winner_family]),
            inc_valid_rank,
        )[0, 1]),
    }, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(
                family_valid[winner_family], dtype=np.float64
            ),
        )

test = load("test")
x_test_cat = make_categorical_matrix(test)
x_test_offset = make_offset_matrix(test)

if winner_family == "binary_gbdt":
    test_raw = gbdt_model.predict(
        x_test_cat, num_iteration=gbdt_model.current_iteration()
    ).astype(np.float64)
elif winner_family == "setwise_fwfm":
    test_raw = predict_torch(fwfm_model, x_test_offset)
elif winner_family == "setwise_product_network":
    test_raw = predict_torch(pnn_model, x_test_offset)
else:
    raise RuntimeError("Unknown winning family")

if winner_alpha < 1.0:
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    test_scores = (
        winner_alpha * within_user_rank(test.user_id, test_raw)
        + (1.0 - winner_alpha)
        * within_user_rank(test.user_id, inc_test)
    )
else:
    test_scores = test_raw

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)