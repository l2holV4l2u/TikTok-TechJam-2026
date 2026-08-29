import os
import time
import json
import gc
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260829
EPOCHS = 2
BATCH_SIZE = 8192
EMBED_DIM = 10

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "hour",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "video_type",
    "is_video_author",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
FAMILIES = ["autoint", "pnn", "fibinet", "dcnv2"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))

ART = os.environ["RUN_ARTIFACTS"]
OUT = os.environ.get("ITER_OUT")
if OUT:
    os.makedirs(OUT, exist_ok=True)

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

inc_valid = np.asarray(
    np.load(os.path.join(ART, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(ART, "incumbent_test_scores.npy")

CARDS = np.asarray(
    [int(FEATURE_CARDINALITIES[name]) for name in FIELDS],
    dtype=np.int64,
)
OFFSETS = np.cumsum(
    np.concatenate([
        np.zeros(1, dtype=np.int64),
        CARDS[:-1],
    ])
)
TOTAL_CARD = int(CARDS.sum())
N_FIELDS = len(FIELDS)
N_NUM = len(NUM_FIELDS)

PAIR_I, PAIR_J = np.triu_indices(N_FIELDS, k=1)
PAIR_I_T = torch.from_numpy(PAIR_I.astype(np.int64))
PAIR_J_T = torch.from_numpy(PAIR_J.astype(np.int64))
N_PAIRS = len(PAIR_I)


def make_cat(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64)
            for name in FIELDS
        ]),
        dtype=np.int64,
    )


def raw_numeric(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.num[name], dtype=np.float32)
            for name in NUM_FIELDS
        ]),
        dtype=np.float32,
    )


def fit_numeric_stats(raw):
    x = np.asarray(raw, dtype=np.float64)
    x = np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)
    x = np.log1p(x)
    median = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, median[None, :])
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-3)
    return (
        median.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def transform_numeric(raw, stats):
    median, mean, std = stats
    x = np.asarray(raw, dtype=np.float32)
    x = np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)
    x = np.log1p(x)
    x = np.where(np.isfinite(x), x, median[None, :])
    x = (x - mean[None, :]) / std[None, :]
    return np.ascontiguousarray(
        np.clip(x, -6.0, 6.0),
        dtype=np.float32,
    )


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        users,
    ))
    sorted_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(starts_flag, np.arange(n), 0)
    )

    ends_flag = np.empty(n, dtype=bool)
    ends_flag[-1] = True
    ends_flag[:-1] = sorted_users[:-1] != sorted_users[1:]
    ends = np.minimum.accumulate(
        np.where(ends_flag, np.arange(n), n - 1)[::-1]
    )[::-1]

    denom = ends - starts
    positions = np.arange(n, dtype=np.float64)
    ranked = np.full(n, 0.5, dtype=np.float64)
    useful = denom > 0
    ranked[useful] = (
        positions[useful] - starts[useful]
    ) / denom[useful]

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def initialize_dense(module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.numeric_linear = nn.Linear(N_NUM, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "offsets",
            torch.from_numpy(OFFSETS.copy()).long(),
        )
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.numeric_linear.weight)
        nn.init.zeros_(self.numeric_linear.bias)

    def encode(self, cat, numeric):
        ids = cat + self.offsets
        embeddings = self.embedding(ids)
        linear = (
            self.linear(ids).sum(dim=1).squeeze(1)
            + self.numeric_linear(numeric).squeeze(1)
            + self.bias
        )
        return embeddings, linear


class AutoIntModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.k1 = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.v1 = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.q2 = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.k2 = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.v2 = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.output = nn.Sequential(
            nn.Linear(N_FIELDS * EMBED_DIM + N_NUM, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )
        initialize_dense(self)

    @staticmethod
    def attend(x, q_layer, k_layer, v_layer):
        q = q_layer(x)
        k = k_layer(x)
        v = v_layer(x)
        logits = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(EMBED_DIM)
        weights = torch.softmax(logits, dim=2)
        return torch.bmm(weights, v)

    def forward(self, cat, numeric):
        emb, linear = self.encode(cat, numeric)
        h = self.norm1(
            emb + self.attend(emb, self.q1, self.k1, self.v1)
        )
        h = self.norm2(
            h + self.attend(h, self.q2, self.k2, self.v2)
        )
        nonlinear = self.output(
            torch.cat([h.flatten(1), numeric], dim=1)
        ).squeeze(1)
        return linear + nonlinear


class PNNModel(BaseModel):
    def __init__(self):
        super().__init__()
        input_dim = N_FIELDS * EMBED_DIM + N_PAIRS + N_NUM
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        initialize_dense(self)

    def forward(self, cat, numeric):
        emb, linear = self.encode(cat, numeric)
        pair_products = (
            emb[:, PAIR_I_T, :] * emb[:, PAIR_J_T, :]
        ).sum(dim=2)
        nonlinear = self.network(
            torch.cat([
                emb.flatten(1),
                pair_products,
                numeric,
            ], dim=1)
        ).squeeze(1)
        return linear + nonlinear


class FiBiNETModel(BaseModel):
    def __init__(self):
        super().__init__()
        reduction = max(4, N_FIELDS // 3)
        self.squeeze = nn.Sequential(
            nn.Linear(N_FIELDS, reduction),
            nn.ReLU(),
            nn.Linear(reduction, N_FIELDS),
            nn.Sigmoid(),
        )
        self.field_scale = nn.Parameter(
            torch.ones(N_FIELDS, EMBED_DIM)
        )
        self.network = nn.Sequential(
            nn.Linear(N_PAIRS * EMBED_DIM + N_NUM, 96),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        initialize_dense(self)

    def forward(self, cat, numeric):
        emb, linear = self.encode(cat, numeric)
        field_summary = emb.mean(dim=2)
        gates = self.squeeze(field_summary).unsqueeze(2)
        reweighted = emb * gates * self.field_scale.unsqueeze(0)
        interactions = (
            reweighted[:, PAIR_I_T, :]
            * reweighted[:, PAIR_J_T, :]
        )
        nonlinear = self.network(
            torch.cat([
                interactions.flatten(1),
                numeric,
            ], dim=1)
        ).squeeze(1)
        return linear + nonlinear


class LowRankCrossLayer(nn.Module):
    def __init__(self, dimension, rank=32):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, x0, x):
        crossed = self.up(torch.tanh(self.down(x)))
        return x + x0 * crossed + self.bias


class DCNv2Model(BaseModel):
    def __init__(self):
        super().__init__()
        dimension = N_FIELDS * EMBED_DIM + N_NUM
        self.cross1 = LowRankCrossLayer(dimension, rank=32)
        self.cross2 = LowRankCrossLayer(dimension, rank=32)
        self.cross3 = LowRankCrossLayer(dimension, rank=24)
        self.deep = nn.Sequential(
            nn.Linear(dimension, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.output = nn.Linear(dimension + 48, 1)
        initialize_dense(self)

    def forward(self, cat, numeric):
        emb, linear = self.encode(cat, numeric)
        x0 = torch.cat([emb.flatten(1), numeric], dim=1)
        crossed = self.cross1(x0, x0)
        crossed = self.cross2(x0, crossed)
        crossed = self.cross3(x0, crossed)
        deep = self.deep(x0)
        nonlinear = self.output(
            torch.cat([crossed, deep], dim=1)
        ).squeeze(1)
        return linear + nonlinear


def make_model(family):
    if family == "autoint":
        return AutoIntModel()
    if family == "pnn":
        return PNNModel()
    if family == "fibinet":
        return FiBiNETModel()
    if family == "dcnv2":
        return DCNv2Model()
    raise ValueError("unknown family: " + family)


def fit_model(family, cat, numeric, labels, seed):
    torch.manual_seed(seed)
    model = make_model(family)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0015,
        weight_decay=3e-6,
    )

    cat_tensor = torch.from_numpy(cat)
    numeric_tensor = torch.from_numpy(numeric)
    label_tensor = torch.from_numpy(
        np.asarray(labels, dtype=np.float32)
    )
    generator = torch.Generator().manual_seed(seed + 101)

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(cat), generator=generator)
        total_loss = 0.0

        for start in range(0, len(cat), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                cat_tensor[idx],
                numeric_tensor[idx],
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                label_tensor[idx],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        print(
            "training family=%s epoch=%d loss=%.6f"
            % (family, epoch + 1, total_loss / len(cat)),
            flush=True,
        )

    return model


def predict_model(model, cat, numeric):
    model.eval()
    result = np.empty(len(cat), dtype=np.float64)
    cat_tensor = torch.from_numpy(cat)
    numeric_tensor = torch.from_numpy(numeric)

    with torch.no_grad():
        for start in range(0, len(cat), 32768):
            end = min(start + 32768, len(cat))
            result[start:end] = model(
                cat_tensor[start:end],
                numeric_tensor[start:end],
            ).cpu().numpy()

    return result


x_train = make_cat(train)
x_valid = make_cat(valid)
raw_train_num = raw_numeric(train)
raw_valid_num = raw_numeric(valid)
numeric_stats = fit_numeric_stats(raw_train_num)
n_train = transform_numeric(raw_train_num, numeric_stats)
n_valid = transform_numeric(raw_valid_num, numeric_stats)

inc_rank_valid = within_user_rank(valid.user_id, inc_valid)
candidate_scores = {}
candidate_predictions = {}
candidate_recipes = {}

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_recipes["incumbent"] = {
    "family": None,
    "alpha": 0.0,
}

blend_alphas = [0.25, 0.50, 0.75]
family_raw_predictions = {}

for family_index, family in enumerate(FAMILIES):
    family_seed = SEED + 1000 * (family_index + 1)
    model = fit_model(
        family,
        x_train,
        n_train,
        y_train,
        family_seed,
    )
    raw_pred = predict_model(model, x_valid, n_valid)
    family_raw_predictions[family] = raw_pred

    standalone_metrics = evaluate(
        valid.user_id, y_valid, raw_pred
    )
    candidate_scores[family] = float(
        standalone_metrics["primary"]
    )
    candidate_predictions[family] = raw_pred
    candidate_recipes[family] = {
        "family": family,
        "alpha": 1.0,
    }

    model_rank = within_user_rank(valid.user_id, raw_pred)
    for alpha in blend_alphas:
        blended = (
            alpha * model_rank
            + (1.0 - alpha) * inc_rank_valid
        )
        name = "%s_blend_%.2f" % (family, alpha)
        metrics = evaluate(valid.user_id, y_valid, blended)
        candidate_scores[name] = float(metrics["primary"])
        candidate_predictions[name] = blended
        candidate_recipes[name] = {
            "family": family,
            "alpha": float(alpha),
        }

    del model
    gc.collect()

best_name = max(
    candidate_scores,
    key=lambda name: candidate_scores[name],
)
best_valid_scores = np.asarray(
    candidate_predictions[best_name],
    dtype=np.float64,
)
best_recipe = candidate_recipes[best_name]
best_metrics = evaluate(
    valid.user_id,
    y_valid,
    best_valid_scores,
)

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: round(float(score), 6)
            for name, score in sorted(candidate_scores.items())
        },
        sort_keys=True,
    ),
    flush=True,
)
print(
    "FINDINGS winner=%s family=%s alpha=%.2f"
    % (
        best_name,
        str(best_recipe["family"]),
        float(best_recipe["alpha"]),
    ),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        best_valid_scores,
    )

# Build test predictions. If the incumbent itself wins, reuse its trusted
# test scores. Otherwise refit the identical selected family on train+valid.
test = load("test")

if best_recipe["family"] is None:
    test_scores = np.asarray(
        np.load(inc_test_path),
        dtype=np.float64,
    )
else:
    selected_family = best_recipe["family"]
    selected_alpha = float(best_recipe["alpha"])

    x_test = make_cat(test)
    raw_test_num = raw_numeric(test)

    combined_cat = np.ascontiguousarray(
        np.concatenate([x_train, x_valid], axis=0),
        dtype=np.int64,
    )
    combined_raw_num = np.ascontiguousarray(
        np.concatenate(
            [raw_train_num, raw_valid_num],
            axis=0,
        ),
        dtype=np.float32,
    )
    combined_labels = np.ascontiguousarray(
        np.concatenate([
            y_train,
            y_valid.astype(np.float32),
        ]),
        dtype=np.float32,
    )

    combined_stats = fit_numeric_stats(combined_raw_num)
    combined_num = transform_numeric(
        combined_raw_num,
        combined_stats,
    )
    n_test = transform_numeric(
        raw_test_num,
        combined_stats,
    )

    selected_index = FAMILIES.index(selected_family)
    selected_seed = SEED + 1000 * (selected_index + 1)
    final_model = fit_model(
        selected_family,
        combined_cat,
        combined_num,
        combined_labels,
        selected_seed,
    )
    raw_test_prediction = predict_model(
        final_model,
        x_test,
        n_test,
    )

    if selected_alpha >= 0.999:
        test_scores = raw_test_prediction
    else:
        incumbent_test = np.asarray(
            np.load(inc_test_path),
            dtype=np.float64,
        )
        model_rank_test = within_user_rank(
            test.user_id,
            raw_test_prediction,
        )
        incumbent_rank_test = within_user_rank(
            test.user_id,
            incumbent_test,
        )
        test_scores = (
            selected_alpha * model_rank_test
            + (1.0 - selected_alpha) * incumbent_rank_test
        )

    del final_model
    gc.collect()

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)