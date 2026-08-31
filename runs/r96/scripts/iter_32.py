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
SEED = 314159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "hour", "tag", "upload_type", "music_type", "user_active_degree",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_range",
    "is_live_streamer", "is_video_author",
    "onehot_feat1", "onehot_feat2", "onehot_feat3",
    "onehot_feat7", "onehot_feat8", "video_type",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]

MAX_GROUP = 32
BATCH_USERS = 128
EPOCHS = {
    "lambda_additive": 8,
    "lambda_fm": 8,
    "lambda_deep": 7,
}


def make_offsets():
    offsets = []
    running = 0
    for field in FIELDS:
        offsets.append(running)
        running += int(FEATURE_CARDINALITIES[field])
    return np.asarray(offsets, dtype=np.int64), running


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def fit_numeric_stats(split):
    matrix = []
    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        matrix.append(np.log1p(np.maximum(value, 0.0)))
    matrix = np.stack(matrix, axis=1).astype(np.float32)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-3)
    return mean, std


def make_features(split, mean, std):
    categorical = np.stack(
        [np.asarray(split.X[field], dtype=np.int64) for field in FIELDS],
        axis=1,
    )
    categorical = categorical + OFFSETS[None, :]

    numeric = []
    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        numeric.append(np.log1p(np.maximum(value, 0.0)))
    numeric = np.stack(numeric, axis=1).astype(np.float32)
    numeric = ((numeric - mean[None, :]) / std[None, :]).astype(np.float32)
    numeric = np.clip(numeric, -6.0, 6.0)
    return np.ascontiguousarray(categorical), np.ascontiguousarray(numeric)


def make_group_grid(user_ids, time_ms, max_group):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    time_ms = np.asarray(time_ms, dtype=np.int64)
    n = len(user_ids)
    row_id = np.arange(n, dtype=np.int64)

    order = np.lexsort((row_id, time_ms, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_positions = np.flatnonzero(starts)
    end_positions = np.concatenate(
        [start_positions[1:], np.asarray([n], dtype=np.int64)]
    )
    counts = end_positions - start_positions
    unique_users = sorted_users[start_positions]

    group_number = np.repeat(
        np.arange(len(unique_users), dtype=np.int64), counts
    )
    within = np.arange(n, dtype=np.int64) - np.repeat(start_positions, counts)
    from_end = np.repeat(counts, counts) - 1 - within
    keep = from_end < max_group

    kept_order = order[keep]
    kept_groups = group_number[keep]
    kept_columns = (
        np.repeat(counts, counts)[keep] - 1 - from_end[keep]
    )
    kept_columns = kept_columns - np.maximum(
        np.repeat(counts, counts)[keep] - max_group, 0
    )

    grid = np.full(
        (len(unique_users), max_group), -1, dtype=np.int64
    )
    grid[kept_groups, kept_columns] = kept_order
    return unique_users, grid, counts


class AdditiveScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.num = nn.Linear(len(NUM_FIELDS), 1)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.num.weight)
        nn.init.zeros_(self.num.bias)

    def forward(self, categorical, numeric):
        return (
            self.bias(categorical).squeeze(-1).sum(dim=-1)
            + self.num(numeric).squeeze(-1)
        )


class FMScorer(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, dim)
        self.num_linear = nn.Linear(len(NUM_FIELDS), 1)
        self.num_embedding = nn.Linear(len(NUM_FIELDS), dim, bias=False)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, categorical, numeric):
        emb = self.embedding(categorical)
        num_emb = self.num_embedding(numeric).unsqueeze(-2)
        all_emb = torch.cat([emb, num_emb], dim=-2)
        summed = all_emb.sum(dim=-2)
        interaction = 0.5 * (
            summed.square() - all_emb.square().sum(dim=-2)
        ).sum(dim=-1)
        linear = (
            self.linear(categorical).squeeze(-1).sum(dim=-1)
            + self.num_linear(numeric).squeeze(-1)
        )
        return linear + interaction


class DeepInteractionScorer(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, dim)
        input_dim = len(FIELDS) * dim + len(NUM_FIELDS)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.wide = nn.Embedding(TOTAL_CARDINALITY, 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.wide.weight)

    def forward(self, categorical, numeric):
        shape = categorical.shape
        emb = self.embedding(categorical).reshape(
            *shape[:-1], len(FIELDS) * self.embedding.embedding_dim
        )
        deep = self.network(torch.cat([emb, numeric], dim=-1)).squeeze(-1)
        wide = self.wide(categorical).squeeze(-1).sum(dim=-1)
        return deep + 0.15 * wide


def lambda_top5_loss(scores, labels, mask, row_weights):
    masked_scores = scores.masked_fill(~mask, -1e9)
    batch, slate = scores.shape

    predicted_order = torch.argsort(masked_scores, dim=1, descending=True)
    predicted_rank = torch.empty_like(predicted_order)
    ranks = torch.arange(
        slate, device=scores.device, dtype=torch.long
    ).view(1, -1).expand(batch, -1)
    predicted_rank.scatter_(1, predicted_order, ranks)

    discount_table = torch.zeros(
        slate, dtype=scores.dtype, device=scores.device
    )
    top = min(5, slate)
    discount_table[:top] = 1.0 / torch.log2(
        torch.arange(
            2, top + 2, device=scores.device, dtype=scores.dtype
        )
    )
    predicted_discount = discount_table[predicted_rank]

    positive_count = (labels * mask.float()).sum(dim=1).long()
    ideal_discount_prefix = torch.cumsum(discount_table, dim=0)
    ideal_index = torch.clamp(positive_count, min=1, max=5) - 1
    idcg = ideal_discount_prefix[ideal_index]
    idcg = torch.where(
        positive_count > 0, idcg, torch.ones_like(idcg)
    )

    positive_pair = (
        (labels[:, :, None] > labels[:, None, :])
        & mask[:, :, None]
        & mask[:, None, :]
    )

    score_difference = scores[:, :, None] - scores[:, None, :]
    pair_loss = F.softplus(-score_difference)

    delta_discount = torch.abs(
        predicted_discount[:, :, None]
        - predicted_discount[:, None, :]
    )
    dcg_weight = delta_discount / idcg[:, None, None]
    dcg_weight = dcg_weight * positive_pair.float()
    dcg_weight = dcg_weight / (
        dcg_weight.sum(dim=(1, 2), keepdim=True) + 1e-8
    )

    uniform_weight = positive_pair.float()
    uniform_weight = uniform_weight / (
        uniform_weight.sum(dim=(1, 2), keepdim=True) + 1e-8
    )

    recency_pair = torch.sqrt(
        row_weights[:, :, None] * row_weights[:, None, :]
    )
    combined_weight = (
        0.65 * dcg_weight + 0.35 * uniform_weight
    ) * recency_pair

    user_loss = (pair_loss * combined_weight).sum(dim=(1, 2))
    valid_user = positive_pair.any(dim=2).any(dim=1)
    if valid_user.any():
        return user_loss[valid_user].mean()
    return scores.sum() * 0.0


def train_model(model, cat, num, labels, dates, grid, epochs, seed):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-5
    )
    model.train()

    max_date = int(np.max(dates))
    age = max_date - dates.astype(np.int32)
    recency = np.power(0.5, age.astype(np.float32) / 4.0).astype(np.float32)
    recency /= max(float(recency.mean()), 1e-8)

    user_indices = np.arange(len(grid), dtype=np.int64)
    epoch_losses = []

    for epoch in range(epochs):
        rng.shuffle(user_indices)
        total_loss = 0.0
        total_batches = 0

        for start in range(0, len(user_indices), BATCH_USERS):
            selected = user_indices[start:start + BATCH_USERS]
            rows = grid[selected]
            mask_np = rows >= 0
            safe_rows = np.maximum(rows, 0)

            cat_t = torch.from_numpy(cat[safe_rows])
            num_t = torch.from_numpy(num[safe_rows])
            label_t = torch.from_numpy(labels[safe_rows]).float()
            mask_t = torch.from_numpy(mask_np)
            weight_t = torch.from_numpy(recency[safe_rows])

            scores = model(cat_t, num_t)
            loss = lambda_top5_loss(
                scores, label_t, mask_t, weight_t
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach())
            total_batches += 1

        epoch_losses.append(total_loss / max(total_batches, 1))

    return epoch_losses


@torch.no_grad()
def predict_model(model, cat, num, batch_size=8192):
    model.eval()
    output = np.empty(len(cat), dtype=np.float64)
    for start in range(0, len(cat), batch_size):
        end = min(start + batch_size, len(cat))
        cat_t = torch.from_numpy(cat[start:end])
        num_t = torch.from_numpy(num[start:end])
        output[start:end] = (
            model(cat_t, num_t).cpu().numpy().astype(np.float64)
        )
    return output


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(
        np.concatenate([np.asarray([-1], dtype=np.int64), end_positions])
    )
    row_sizes = np.repeat(sizes, sizes)

    position = np.arange(n, dtype=np.int64) - group_start
    percentile = (position.astype(np.float64) + 0.5) / row_sizes
    result = np.empty(n, dtype=np.float64)
    result[order] = percentile
    return result


train = load("train")
valid = load("valid")
test = load("test")

numeric_mean, numeric_std = fit_numeric_stats(train)
train_cat, train_num = make_features(train, numeric_mean, numeric_std)
valid_cat, valid_num = make_features(valid, numeric_mean, numeric_std)
test_cat, test_num = make_features(test, numeric_mean, numeric_std)

y_train = np.asarray(train.y, dtype=np.float32)
dates_train = np.asarray(train.date, dtype=np.int32)

group_users, group_grid, original_group_sizes = make_group_grid(
    train.user_id, train.time_ms, MAX_GROUP
)

model_factories = {
    "lambda_additive": lambda: AdditiveScorer(),
    "lambda_fm": lambda: FMScorer(dim=16),
    "lambda_deep": lambda: DeepInteractionScorer(dim=6),
}

valid_raw = {}
test_raw = {}
training_losses = {}

for model_number, name in enumerate(model_factories):
    torch.manual_seed(SEED + 100 * model_number)
    model = model_factories[name]()
    losses = train_model(
        model,
        train_cat,
        train_num,
        y_train,
        dates_train,
        group_grid,
        EPOCHS[name],
        SEED + 1000 + model_number,
    )
    training_losses[name] = losses
    valid_raw[name] = predict_model(model, valid_cat, valid_num)
    test_raw[name] = predict_model(model, test_cat, test_num)
    del model
    gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

candidate_valid = {"trusted_incumbent": inc_valid}
candidate_test = {"trusted_incumbent": inc_test}
candidate_raw_name = {"trusted_incumbent": "lambda_fm"}
candidate_metrics = {
    "trusted_incumbent": evaluate(valid.user_id, valid.y, inc_valid)
}

for family in valid_raw:
    standalone = family + "_standalone"
    candidate_valid[standalone] = valid_raw[family]
    candidate_test[standalone] = test_raw[family]
    candidate_raw_name[standalone] = family
    candidate_metrics[standalone] = evaluate(
        valid.user_id, valid.y, valid_raw[family]
    )

    va_rank = rank_percentile(valid.user_id, valid_raw[family])
    te_rank = rank_percentile(test.user_id, test_raw[family])

    for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        name = family + f"_incumbent_blend_{alpha:.2f}"
        candidate_valid[name] = (
            (1.0 - alpha) * inc_valid_rank + alpha * va_rank
        )
        candidate_test[name] = (
            (1.0 - alpha) * inc_test_rank + alpha * te_rank
        )
        candidate_raw_name[name] = family
        candidate_metrics[name] = evaluate(
            valid.user_id, valid.y, candidate_valid[name]
        )

# A rank-aggregated metric-aligned committee is structurally less sensitive
# to the calibration differences among the three heads.
committee_valid_rank = np.mean(
    np.stack([
        rank_percentile(valid.user_id, valid_raw[name])
        for name in model_factories
    ], axis=1),
    axis=1,
)
committee_test_rank = np.mean(
    np.stack([
        rank_percentile(test.user_id, test_raw[name])
        for name in model_factories
    ], axis=1),
    axis=1,
)
candidate_valid["lambda_committee_standalone"] = committee_valid_rank
candidate_test["lambda_committee_standalone"] = committee_test_rank
candidate_raw_name["lambda_committee_standalone"] = "lambda_fm"
candidate_metrics["lambda_committee_standalone"] = evaluate(
    valid.user_id, valid.y, committee_valid_rank
)

for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
    name = f"lambda_committee_incumbent_blend_{alpha:.2f}"
    candidate_valid[name] = (
        (1.0 - alpha) * inc_valid_rank + alpha * committee_valid_rank
    )
    candidate_test[name] = (
        (1.0 - alpha) * inc_test_rank + alpha * committee_test_rank
    )
    candidate_raw_name[name] = "lambda_fm"
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, candidate_valid[name]
    )

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]
raw_family = candidate_raw_name[best_name]

print("CANDIDATES " + json.dumps(
    {
        name: float(metrics["primary"])
        for name, metrics in candidate_metrics.items()
    },
    sort_keys=True,
))
print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "training_losses": {
        name: [float(x) for x in losses]
        for name, losses in training_losses.items()
    },
    "train_users": int(len(group_grid)),
    "median_original_train_group": float(np.median(original_group_sizes)),
    "fraction_train_rows_retained": float(
        np.sum(group_grid >= 0) / len(train_cat)
    ),
    "objective": "0.65 Lambda-nDCG@5 swap weight plus 0.35 pairwise GAUC weight",
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
    if best_name == "trusted_incumbent" or "blend" in best_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(valid_raw[raw_family], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))