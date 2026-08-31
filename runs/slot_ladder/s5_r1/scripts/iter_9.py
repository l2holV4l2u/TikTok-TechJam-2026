import os
import time
import json
import random
import numpy as np
import torch
from torch import nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EMBED_DIM = 8

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets(fields):
    offsets = []
    total = 0
    for field in fields:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[field])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets(FIELDS)


def categorical_matrix(split):
    columns = [
        np.asarray(split.X[field], dtype=np.int64) + OFFSETS[j]
        for j, field in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int32)


def unscaled_numeric_matrix(split_name):
    # The failed attempt passed a split name and then accessed `.num` on the
    # string. Resolve the split object here before reading numeric columns.
    split = load(split_name)
    blocks = []
    names = []

    for name in sorted(split.num):
        values = np.asarray(split.num[name], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.log1p(np.maximum(values, 0.0))
        blocks.append(values)
        names.append("raw_" + name)

    for entity in ("video_id", "author_id"):
        histories = historical_features(split_name, key=entity)
        for key in sorted(histories):
            values = np.asarray(histories[key], dtype=np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            blocks.append(values)
            names.append(entity + "__" + key)

    matrix = np.column_stack(blocks).astype(np.float32, copy=False)
    return np.ascontiguousarray(matrix), names


def standardize_numeric(train_raw, other_raw):
    mean = train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_raw.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1.0e-5, std, 1.0).astype(np.float32)

    train_scaled = (train_raw - mean) / std
    other_scaled = (other_raw - mean) / std

    train_scaled = np.clip(train_scaled, -8.0, 8.0)
    other_scaled = np.clip(other_scaled, -8.0, 8.0)

    return (
        np.ascontiguousarray(train_scaled, dtype=np.float32),
        np.ascontiguousarray(other_scaled, dtype=np.float32),
        mean,
        std,
    )


class DCNv2Model(nn.Module):
    """Parallel deep and explicit cross towers."""

    def __init__(self, cardinality, n_fields, numeric_dim, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, embed_dim)
        self.linear = nn.Embedding(cardinality, 1)

        input_dim = n_fields * embed_dim + numeric_dim
        self.cross_layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(3)]
        )
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.output = nn.Linear(input_dim + 48, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.weight)
        for layer in self.cross_layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.15)
            nn.init.zeros_(layer.bias)

    def forward(self, x_cat, x_num):
        x_cat = x_cat.long()
        embeddings = self.embedding(x_cat).flatten(start_dim=1)
        x0 = torch.cat([embeddings, x_num], dim=1)

        crossed = x0
        for layer in self.cross_layers:
            crossed = x0 * layer(crossed) + crossed

        deep = self.deep(x0)
        wide = self.linear(x_cat).sum(dim=1).squeeze(-1)
        return (
            self.bias
            + wide
            + self.output(torch.cat([crossed, deep], dim=1)).squeeze(-1)
        )


class PNNModel(nn.Module):
    """Explicit inner-product layer followed by a nonlinear prediction tower."""

    def __init__(self, cardinality, n_fields, numeric_dim, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, embed_dim)
        self.linear = nn.Embedding(cardinality, 1)

        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )

        input_dim = n_fields * embed_dim + len(pair_i) + numeric_dim
        self.tower = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x_cat, x_num):
        x_cat = x_cat.long()
        embeddings = self.embedding(x_cat)
        products = (
            embeddings[:, self.pair_i, :]
            * embeddings[:, self.pair_j, :]
        ).sum(dim=-1)

        features = torch.cat(
            [embeddings.flatten(start_dim=1), products, x_num], dim=1
        )
        wide = self.linear(x_cat).sum(dim=1).squeeze(-1)
        return self.bias + wide + self.tower(features).squeeze(-1)


class FiBiNETModel(nn.Module):
    """Squeeze-excitation field weighting plus field-specific bilinear products."""

    def __init__(self, cardinality, n_fields, numeric_dim, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, embed_dim)
        self.linear = nn.Embedding(cardinality, 1)
        self.n_fields = n_fields

        squeeze_dim = max(4, n_fields // 2)
        self.se = nn.Sequential(
            nn.Linear(n_fields, squeeze_dim),
            nn.ReLU(),
            nn.Linear(squeeze_dim, n_fields),
            nn.Sigmoid(),
        )

        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )
        n_pairs = len(pair_i)

        self.bilinear = nn.Parameter(
            torch.empty(n_pairs, embed_dim, embed_dim)
        )
        nn.init.xavier_uniform_(self.bilinear)

        input_dim = n_fields * embed_dim + n_pairs + numeric_dim
        self.tower = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x_cat, x_num):
        x_cat = x_cat.long()
        embeddings = self.embedding(x_cat)

        field_summary = embeddings.mean(dim=-1)
        field_weights = self.se(field_summary).unsqueeze(-1)
        recalibrated = embeddings * field_weights

        left = recalibrated[:, self.pair_i, :]
        right = recalibrated[:, self.pair_j, :]
        bilinear_products = torch.einsum(
            "bpd,pde,bpe->bp", left, self.bilinear, right
        )

        features = torch.cat(
            [
                recalibrated.flatten(start_dim=1),
                bilinear_products,
                x_num,
            ],
            dim=1,
        )
        wide = self.linear(x_cat).sum(dim=1).squeeze(-1)
        return self.bias + wide + self.tower(features).squeeze(-1)


def train_model(
    model,
    x_cat_np,
    x_num_np,
    y_np,
    epochs,
    learning_rate,
    seed,
):
    x_cat = torch.from_numpy(x_cat_np)
    x_num = torch.from_numpy(x_num_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1.0e-6,
    )
    criterion = nn.BCEWithLogitsLoss()

    generator = torch.Generator()
    generator.manual_seed(seed)
    n = len(y)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for begin in range(0, n, BATCH_SIZE):
            idx = order[begin:begin + BATCH_SIZE]
            logits = model(x_cat[idx], x_num[idx])
            loss = criterion(logits, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.inference_mode()
def predict(model, x_cat_np, x_num_np):
    model.eval()
    x_cat = torch.from_numpy(x_cat_np)
    x_num = torch.from_numpy(x_num_np)
    scores = np.empty(len(x_cat_np), dtype=np.float64)

    for begin in range(0, len(x_cat_np), PRED_BATCH_SIZE):
        end = min(begin + PRED_BATCH_SIZE, len(x_cat_np))
        scores[begin:end] = (
            model(x_cat[begin:end], x_num[begin:end])
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    return scores


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row_id = np.arange(n, dtype=np.int64)

    order = np.lexsort((row_id, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    repeated_starts = np.repeat(starts, sizes)
    denominators = np.repeat(np.maximum(sizes - 1, 1), sizes)
    sorted_ranks = (
        np.arange(n, dtype=np.float64) - repeated_starts
    ) / denominators

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = sorted_ranks
    return ranked


train = load("train")
valid = load("valid")

x_train_cat = categorical_matrix(train)
x_valid_cat = categorical_matrix(valid)

train_num_raw, numeric_names = unscaled_numeric_matrix("train")
valid_num_raw, valid_numeric_names = unscaled_numeric_matrix("valid")
if numeric_names != valid_numeric_names:
    raise RuntimeError("Historical/numeric feature schemas differ across splits")

x_train_num, x_valid_num, numeric_mean, numeric_std = standardize_numeric(
    train_num_raw, valid_num_raw
)
del train_num_raw, valid_num_raw

print(
    "FINDINGS "
    + json.dumps(
        {
            "numeric_feature_count": int(x_train_num.shape[1]),
            "categorical_field_count": len(FIELDS),
            "numeric_schema_match": True,
        },
        separators=(",", ":"),
    )
)

numeric_dim = x_train_num.shape[1]
model_specs = [
    (
        "dcnv2",
        DCNv2Model(
            TOTAL_CARDINALITY,
            len(FIELDS),
            numeric_dim,
            embed_dim=EMBED_DIM,
        ),
        3,
        0.0015,
    ),
    (
        "pnn",
        PNNModel(
            TOTAL_CARDINALITY,
            len(FIELDS),
            numeric_dim,
            embed_dim=EMBED_DIM,
        ),
        3,
        0.0015,
    ),
    (
        "fibinet",
        FiBiNETModel(
            TOTAL_CARDINALITY,
            len(FIELDS),
            numeric_dim,
            embed_dim=EMBED_DIM,
        ),
        2,
        0.0015,
    ),
]

models = {}
valid_predictions = {}

for model_index, (name, model, epochs, lr) in enumerate(model_specs):
    torch.manual_seed(SEED + 101 * model_index)
    model = train_model(
        model,
        x_train_cat,
        x_train_num,
        train.y,
        epochs=epochs,
        learning_rate=lr,
        seed=SEED + 1000 + model_index,
    )
    models[name] = model
    valid_predictions[name] = predict(model, x_valid_cat, x_valid_num)

del x_train_cat, x_train_num

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores are missing")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
incumbent_valid_rank = within_user_rank(
    valid.user_id, incumbent_valid
)

candidate_scores = {}
candidate_metrics = {}
candidate_recipes = {}
candidate_raw = {}

for name, raw_scores in valid_predictions.items():
    standalone_metric = evaluate(valid.user_id, valid.y, raw_scores)
    candidate_scores[name] = raw_scores
    candidate_metrics[name] = float(standalone_metric["primary"])
    candidate_recipes[name] = ("standalone", name, 1.0)
    candidate_raw[name] = raw_scores

    own_rank = within_user_rank(valid.user_id, raw_scores)
    for weight in (0.10, 0.20, 0.35, 0.50, 0.65):
        candidate_name = f"{name}_rankblend_w{weight:.2f}"
        blended = (
            weight * own_rank
            + (1.0 - weight) * incumbent_valid_rank
        )
        blended_metric = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[candidate_name] = blended
        candidate_metrics[candidate_name] = float(
            blended_metric["primary"]
        )
        candidate_recipes[candidate_name] = (
            "rankblend",
            name,
            weight,
        )
        candidate_raw[candidate_name] = raw_scores

family_rank_ensemble = np.mean(
    np.column_stack(
        [
            within_user_rank(valid.user_id, valid_predictions["dcnv2"]),
            within_user_rank(valid.user_id, valid_predictions["pnn"]),
            within_user_rank(valid.user_id, valid_predictions["fibinet"]),
        ]
    ),
    axis=1,
)
ensemble_metric = evaluate(
    valid.user_id, valid.y, family_rank_ensemble
)
candidate_scores["interaction_family_ensemble"] = family_rank_ensemble
candidate_metrics["interaction_family_ensemble"] = float(
    ensemble_metric["primary"]
)
candidate_recipes["interaction_family_ensemble"] = (
    "standalone",
    "ensemble",
    1.0,
)
candidate_raw["interaction_family_ensemble"] = family_rank_ensemble

for weight in (0.10, 0.20, 0.35, 0.50, 0.65):
    candidate_name = f"interaction_ensemble_rankblend_w{weight:.2f}"
    blended = (
        weight * family_rank_ensemble
        + (1.0 - weight) * incumbent_valid_rank
    )
    blended_metric = evaluate(valid.user_id, valid.y, blended)
    candidate_scores[candidate_name] = blended
    candidate_metrics[candidate_name] = float(
        blended_metric["primary"]
    )
    candidate_recipes[candidate_name] = (
        "rankblend",
        "ensemble",
        weight,
    )
    candidate_raw[candidate_name] = family_rank_ensemble

winner_name = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = candidate_scores[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_metrics.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if candidate_recipes[winner_name][0] != "standalone":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner_name], dtype=np.float64),
        )

test = load("test")
x_test_cat = categorical_matrix(test)
test_num_raw, test_numeric_names = unscaled_numeric_matrix("test")
if test_numeric_names != numeric_names:
    raise RuntimeError("Test numeric feature schema differs from train")

x_test_num = (test_num_raw - numeric_mean) / numeric_std
x_test_num = np.ascontiguousarray(
    np.clip(x_test_num, -8.0, 8.0),
    dtype=np.float32,
)
del test_num_raw

recipe_type, recipe_model, weight = candidate_recipes[winner_name]

if recipe_model == "ensemble":
    test_model_predictions = {
        name: predict(model, x_test_cat, x_test_num)
        for name, model in models.items()
    }
    own_test_scores = np.mean(
        np.column_stack(
            [
                within_user_rank(
                    test.user_id, test_model_predictions["dcnv2"]
                ),
                within_user_rank(
                    test.user_id, test_model_predictions["pnn"]
                ),
                within_user_rank(
                    test.user_id, test_model_predictions["fibinet"]
                ),
            ]
        ),
        axis=1,
    )
else:
    own_test_scores = predict(
        models[recipe_model], x_test_cat, x_test_num
    )

if recipe_type == "standalone":
    test_scores = own_test_scores
elif recipe_type == "rankblend":
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError("Trusted incumbent test scores are missing")
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    incumbent_test_rank = within_user_rank(
        test.user_id, incumbent_test
    )

    if recipe_model == "ensemble":
        own_test_rank = own_test_scores
    else:
        own_test_rank = within_user_rank(
            test.user_id, own_test_scores
        )

    test_scores = (
        weight * own_test_rank
        + (1.0 - weight) * incumbent_test_rank
    )
else:
    raise ValueError("Unknown candidate recipe")

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)