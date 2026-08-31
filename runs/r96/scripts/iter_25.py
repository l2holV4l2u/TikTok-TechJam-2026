import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 73129
THREADS = min(16, os.cpu_count() or 1)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

DEVICE = torch.device("cpu")
BATCH_SIZE = 8192
EPOCHS = 3
EMBED_DIM = 12
HALF_LIFE = 4.0

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "hour",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "is_video_author",
    "is_live_streamer",
    "onehot_feat2",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_idx = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = ordered_users[:-1] != ordered_users[1:]
    end_pos = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.asarray([-1], dtype=np.int64),
        end_pos,
    )))
    row_sizes = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - start_idx

    ranked = (
        positions.astype(np.float64) + 0.5
    ) / row_sizes.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def make_categorical_matrix(split):
    return np.ascontiguousarray(
        np.stack([
            np.asarray(split.X[field], dtype=np.int64)
            for field in FIELDS
        ], axis=1),
        dtype=np.int64,
    )


def raw_numeric_matrix(split):
    cols = []
    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(
            value, nan=0.0, posinf=0.0, neginf=0.0
        )
        value = np.log1p(np.maximum(value, 0.0))
        cols.append(value.astype(np.float32))
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)


def standardize_numeric(train_num, query_num):
    center = np.median(train_num, axis=0).astype(np.float32)
    q25 = np.percentile(train_num, 25, axis=0).astype(np.float32)
    q75 = np.percentile(train_num, 75, axis=0).astype(np.float32)
    scale = np.maximum(q75 - q25, 0.1).astype(np.float32)
    result = (query_num - center[None, :]) / scale[None, :]
    result = np.clip(result, -8.0, 8.0)
    return np.ascontiguousarray(result, dtype=np.float32)


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    weights = np.power(
        0.5, age.astype(np.float32) / HALF_LIFE
    )
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


class FieldEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[field]), EMBED_DIM)
            for field in FIELDS
        ])
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[field]), 1)
            for field in FIELDS
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)
        for emb in self.linear_embeddings:
            nn.init.zeros_(emb.weight)

    def embed(self, x_cat):
        return torch.stack([
            emb(x_cat[:, j])
            for j, emb in enumerate(self.embeddings)
        ], dim=1)

    def wide(self, x_cat):
        terms = [
            emb(x_cat[:, j])
            for j, emb in enumerate(self.linear_embeddings)
        ]
        return torch.stack(terms, dim=1).sum(dim=1).squeeze(1)


class FiBiNETModel(FieldEmbedding):
    def __init__(self, n_num):
        super().__init__()
        n_fields = len(FIELDS)
        reduction = max(4, n_fields // 3)

        self.se1 = nn.Linear(n_fields, reduction)
        self.se2 = nn.Linear(reduction, n_fields)
        self.bilinear = nn.Parameter(
            torch.empty(EMBED_DIM, EMBED_DIM)
        )
        nn.init.xavier_uniform_(self.bilinear)

        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i", torch.as_tensor(pair_i, dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.as_tensor(pair_j, dtype=torch.long)
        )
        n_pairs = len(pair_i)

        self.mlp = nn.Sequential(
            nn.Linear(n_pairs + n_num, 128),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x_cat, x_num):
        emb = self.embed(x_cat)
        squeeze = emb.mean(dim=2)
        gates = 2.0 * torch.sigmoid(
            self.se2(F.silu(self.se1(squeeze)))
        )
        recalibrated = emb * gates.unsqueeze(2)

        transformed = torch.matmul(recalibrated, self.bilinear)
        left = transformed[:, self.pair_i, :]
        right = emb[:, self.pair_j, :]
        pair_logits = (left * right).sum(dim=2)

        deep_input = torch.cat([pair_logits, x_num], dim=1)
        return (
            self.wide(x_cat)
            + self.mlp(deep_input).squeeze(1)
            + self.bias
        )


class AFNModel(FieldEmbedding):
    def __init__(self, n_num, logarithmic_neurons=32):
        super().__init__()
        n_fields = len(FIELDS)
        self.exponents = nn.Parameter(
            torch.empty(n_fields, logarithmic_neurons)
        )
        nn.init.normal_(
            self.exponents,
            mean=1.0 / n_fields,
            std=0.035,
        )

        self.mlp = nn.Sequential(
            nn.Linear(logarithmic_neurons * EMBED_DIM + n_num, 160),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(160, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x_cat, x_num):
        emb = self.embed(x_cat)
        log_emb = torch.log(torch.clamp(torch.abs(emb), min=1e-5))
        logarithmic = torch.einsum(
            "bfd,fl->bld", log_emb, self.exponents
        )
        interaction = torch.exp(torch.clamp(logarithmic, -7.0, 7.0))
        interaction = interaction.flatten(start_dim=1)
        deep_input = torch.cat([interaction, x_num], dim=1)
        return (
            self.wide(x_cat)
            + self.mlp(deep_input).squeeze(1)
            + self.bias
        )


class MaskNetModel(FieldEmbedding):
    def __init__(self, n_num):
        super().__init__()
        input_dim = len(FIELDS) * EMBED_DIM + n_num
        hidden = 192

        self.input_projection = nn.Linear(input_dim, hidden)
        self.mask1 = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.SiLU(),
            nn.Linear(96, hidden),
            nn.Sigmoid(),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.mask2 = nn.Sequential(
            nn.Linear(hidden, 96),
            nn.SiLU(),
            nn.Linear(96, hidden),
            nn.Sigmoid(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x_cat, x_num):
        emb = self.embed(x_cat).flatten(start_dim=1)
        raw = torch.cat([emb, x_num], dim=1)

        base = self.input_projection(raw)
        h1 = self.block1(base * (2.0 * self.mask1(raw)))
        h2 = self.block2(h1 * (2.0 * self.mask2(h1)))
        hidden = base + h1 + h2

        return (
            self.wide(x_cat)
            + self.output(hidden).squeeze(1)
            + self.bias
        )


def train_model(model, x_cat, x_num, y, weights, seed):
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0025,
        weight_decay=2e-5,
    )

    n = len(y)
    y_tensor = torch.from_numpy(y)
    w_tensor = torch.from_numpy(weights)

    model.train()
    for epoch in range(EPOCHS):
        generator = torch.Generator()
        generator.manual_seed(seed + 1009 * epoch)
        permutation = torch.randperm(n, generator=generator)

        epoch_loss = 0.0
        epoch_weight = 0
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE].numpy()

            cat_batch = torch.from_numpy(x_cat[idx]).to(DEVICE)
            num_batch = torch.from_numpy(x_num[idx]).to(DEVICE)
            label_batch = y_tensor[idx].to(DEVICE)
            weight_batch = w_tensor[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat_batch, num_batch)
            losses = F.binary_cross_entropy_with_logits(
                logits, label_batch, reduction="none"
            )
            loss = torch.sum(losses * weight_batch) / torch.sum(
                weight_batch
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            epoch_loss += float(loss.detach()) * len(idx)
            epoch_weight += len(idx)

        print(
            "FINDINGS " + json.dumps({
                "model": model.__class__.__name__,
                "epoch": epoch + 1,
                "weighted_logloss": epoch_loss / max(epoch_weight, 1),
            }, sort_keys=True)
        )

    return model


@torch.no_grad()
def predict_model(model, x_cat, x_num):
    model.eval()
    result = np.empty(len(x_cat), dtype=np.float32)
    for start in range(0, len(x_cat), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(x_cat))
        cat_batch = torch.from_numpy(x_cat[start:end]).to(DEVICE)
        num_batch = torch.from_numpy(x_num[start:end]).to(DEVICE)
        logits = model(cat_batch, num_batch)
        result[start:end] = logits.cpu().numpy().astype(np.float32)
    return result


train = load("train")
valid = load("valid")
test = load("test")

xcat_train = make_categorical_matrix(train)
xcat_valid = make_categorical_matrix(valid)
xcat_test = make_categorical_matrix(test)

raw_num_train = raw_numeric_matrix(train)
raw_num_valid = raw_numeric_matrix(valid)
raw_num_test = raw_numeric_matrix(test)

xnum_train = standardize_numeric(raw_num_train, raw_num_train)
xnum_valid = standardize_numeric(raw_num_train, raw_num_valid)
xnum_test = standardize_numeric(raw_num_train, raw_num_test)

del raw_num_train, raw_num_valid, raw_num_test
gc.collect()

y_train = np.asarray(train.y, dtype=np.float32)
sample_weights = recency_weights(train.date)

models = [
    ("fibinet", FiBiNETModel(len(NUM_FIELDS)), SEED + 11),
    ("adaptive_factorization_network", AFNModel(len(NUM_FIELDS)), SEED + 23),
    ("masknet", MaskNetModel(len(NUM_FIELDS)), SEED + 37),
]

own_valid = {}
own_test = {}

for name, model, model_seed in models:
    model = train_model(
        model,
        xcat_train,
        xnum_train,
        y_train,
        sample_weights,
        model_seed,
    )
    own_valid[name] = predict_model(model, xcat_valid, xnum_valid)
    own_test[name] = predict_model(model, xcat_test, xnum_test)
    del model
    gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

candidate_scores = {"trusted_incumbent": inc_valid}
candidate_metrics = {
    "trusted_incumbent": evaluate(valid.user_id, valid.y, inc_valid)
}
candidate_specs = {
    "trusted_incumbent": (None, None)
}

for name in own_valid:
    standalone = np.asarray(own_valid[name], dtype=np.float64)
    candidate_scores[name + "_standalone"] = standalone
    candidate_metrics[name + "_standalone"] = evaluate(
        valid.user_id, valid.y, standalone
    )
    candidate_specs[name + "_standalone"] = (name, None)

    valid_rank = rank_percentile(valid.user_id, standalone)
    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        candidate_name = f"{name}_incumbent_blend_{alpha:.2f}"
        score = (
            alpha * valid_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        candidate_scores[candidate_name] = score
        candidate_metrics[candidate_name] = evaluate(
            valid.user_id, valid.y, score
        )
        candidate_specs[candidate_name] = (name, alpha)

# A rank ensemble also tests whether the three mechanisms make
# complementary errors before blending the result with the incumbent.
all_valid_ranks = np.stack([
    rank_percentile(valid.user_id, own_valid[name])
    for name in own_valid
], axis=1)
all_test_ranks = np.stack([
    rank_percentile(test.user_id, own_test[name])
    for name in own_test
], axis=1)

neural_valid_ensemble = np.mean(all_valid_ranks, axis=1)
neural_test_ensemble = np.mean(all_test_ranks, axis=1)

candidate_scores["three_family_rank_ensemble"] = neural_valid_ensemble
candidate_metrics["three_family_rank_ensemble"] = evaluate(
    valid.user_id, valid.y, neural_valid_ensemble
)
candidate_specs["three_family_rank_ensemble"] = (
    "three_family_rank_ensemble", None
)

for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
    candidate_name = f"three_family_incumbent_blend_{alpha:.2f}"
    score = (
        alpha * neural_valid_ensemble
        + (1.0 - alpha) * inc_valid_rank
    )
    candidate_scores[candidate_name] = score
    candidate_metrics[candidate_name] = evaluate(
        valid.user_id, valid.y, score
    )
    candidate_specs[candidate_name] = (
        "three_family_rank_ensemble", alpha
    )

best_name = max(
    candidate_metrics,
    key=lambda key: float(candidate_metrics[key]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(candidate_scores[best_name], dtype=np.float64)
best_family, best_alpha = candidate_specs[best_name]

if best_name == "trusted_incumbent":
    best_test = inc_test
    raw_valid = np.asarray(own_valid["fibinet"], dtype=np.float64)
elif best_family == "three_family_rank_ensemble":
    raw_valid = neural_valid_ensemble
    if best_alpha is None:
        best_test = neural_test_ensemble
    else:
        best_test = (
            best_alpha * neural_test_ensemble
            + (1.0 - best_alpha) * inc_test_rank
        )
else:
    raw_valid = np.asarray(own_valid[best_family], dtype=np.float64)
    if best_alpha is None:
        best_test = np.asarray(own_test[best_family], dtype=np.float64)
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
    "best_family": best_family,
    "best_blend_alpha": best_alpha,
    "families": list(own_valid.keys()),
    "fields": len(FIELDS),
    "embedding_dim": EMBED_DIM,
    "epochs": EPOCHS,
    "half_life_days": HALF_LIFE,
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
    if best_name == "trusted_incumbent" or best_alpha is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))