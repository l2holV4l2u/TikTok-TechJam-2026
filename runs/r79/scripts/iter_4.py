import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260829
BATCH_SIZE = 8192
EPOCHS = 3
EMBED_DIM = 12

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "video_type",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

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
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
    dtype=np.int64,
)
OFFSETS = np.cumsum(
    np.concatenate([np.zeros(1, dtype=np.int64), CARDS[:-1]])
)
TOTAL_CARD = int(CARDS.sum())


def make_cat(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64)
            for name in FIELDS
        ]),
        dtype=np.int64,
    )


def raw_num(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.num[name], dtype=np.float32)
            for name in NUM_FIELDS
        ]),
        dtype=np.float32,
    )


def fit_numeric_transform(raw):
    x = np.asarray(raw, dtype=np.float64)
    x = np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)
    z = np.log1p(x)
    median = np.nanmedian(z, axis=0)
    z = np.where(np.isfinite(z), z, median[None, :])
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    std = np.maximum(std, 1e-3)
    return median.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def transform_numeric(raw, stats):
    median, mean, std = stats
    x = np.asarray(raw, dtype=np.float32)
    x = np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)
    z = np.log1p(x)
    z = np.where(np.isfinite(z), z, median[None, :])
    z = (z - mean[None, :]) / std[None, :]
    return np.ascontiguousarray(np.clip(z, -6.0, 6.0), dtype=np.float32)


x_train = make_cat(train)
x_valid = make_cat(valid)
raw_train_num = raw_num(train)
raw_valid_num = raw_num(valid)
train_num_stats = fit_numeric_transform(raw_train_num)
n_train = transform_numeric(raw_train_num, train_num_stats)
n_valid = transform_numeric(raw_valid_num, train_num_stats)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    sorted_users = user_ids[order]

    group_start = np.empty(n, dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(group_start, np.arange(n), 0)
    )

    group_end = np.empty(n, dtype=bool)
    group_end[-1] = True
    group_end[:-1] = sorted_users[:-1] != sorted_users[1:]
    ends = np.minimum.accumulate(
        np.where(group_end, np.arange(n), n - 1)[::-1]
    )[::-1]

    denom = ends - starts
    sorted_rank = np.full(n, 0.5, dtype=np.float64)
    useful = denom > 0
    positions = np.arange(n, dtype=np.float64)
    sorted_rank[useful] = (
        positions[useful] - starts[useful]
    ) / denom[useful]

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_rank
    return result


class BaseCategoricalModel(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, embed_dim)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "offsets",
            torch.from_numpy(OFFSETS.copy()).long(),
        )
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def encoded(self, x):
        ids = x + self.offsets
        emb = self.embedding(ids)
        linear = self.linear(ids).sum(dim=1).squeeze(1) + self.bias
        return emb, linear


class NFM(BaseCategoricalModel):
    """
    Neural Factorization Machine: all pairwise embedding interactions are
    pooled into one vector and transformed nonlinearly.
    """
    def __init__(self):
        super().__init__()
        self.numeric_projection = nn.Sequential(
            nn.Linear(len(NUM_FIELDS), 16),
            nn.ReLU(),
        )
        self.network = nn.Sequential(
            nn.Linear(EMBED_DIM + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.reset_dense()

    def reset_dense(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, numeric):
        emb, linear = self.encoded(x)
        summed = emb.sum(dim=1)
        bi = 0.5 * (
            summed * summed - (emb * emb).sum(dim=1)
        )
        num = self.numeric_projection(numeric)
        nonlinear = self.network(
            torch.cat([bi, num], dim=1)
        ).squeeze(1)
        return linear + nonlinear


class DCN(BaseCategoricalModel):
    """
    Deep & Cross Network: cross layers explicitly generate bounded-degree
    multiplicative crosses over all field embeddings and numeric features.
    """
    def __init__(self):
        super().__init__()
        self.input_dim = len(FIELDS) * EMBED_DIM + len(NUM_FIELDS)
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.empty(self.input_dim))
            for _ in range(3)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.input_dim))
            for _ in range(3)
        ])
        self.deep = nn.Sequential(
            nn.Linear(self.input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.output = nn.Linear(self.input_dim + 48, 1)
        self.reset_dense()

    def reset_dense(self):
        for weight in self.cross_weights:
            nn.init.normal_(weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, numeric):
        emb, linear = self.encoded(x)
        x0 = torch.cat([
            emb.reshape(emb.shape[0], -1),
            numeric,
        ], dim=1)
        crossed = x0
        for weight, bias in zip(
            self.cross_weights, self.cross_biases
        ):
            scalar = torch.sum(crossed * weight, dim=1, keepdim=True)
            crossed = x0 * scalar + bias + crossed
        deep = self.deep(x0)
        nonlinear = self.output(
            torch.cat([crossed, deep], dim=1)
        ).squeeze(1)
        return linear + nonlinear


class BPRFM(BaseCategoricalModel):
    """
    FM scoring function optimized with within-user positive-negative pairs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, numeric):
        emb, linear = self.encoded(x)
        summed = emb.sum(dim=1)
        interaction = 0.5 * (
            (summed * summed).sum(dim=1)
            - (emb * emb).sum(dim=(1, 2))
        )
        return linear + interaction


def make_model(family):
    if family == "nfm":
        return NFM()
    if family == "dcn":
        return DCN()
    if family == "bpr_fm":
        return BPRFM()
    raise ValueError(family)


def predict_model(model, cat, numeric):
    model.eval()
    result = np.empty(len(cat), dtype=np.float64)
    cat_tensor = torch.from_numpy(cat)
    num_tensor = torch.from_numpy(numeric)
    with torch.no_grad():
        for start in range(0, len(cat), 65536):
            end = min(start + 65536, len(cat))
            result[start:end] = model(
                cat_tensor[start:end],
                num_tensor[start:end],
            ).cpu().numpy()
    return result


def fit_pointwise(family, cat, numeric, labels, seed):
    torch.manual_seed(seed)
    model = make_model(family)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0015,
        weight_decay=2e-6,
    )

    cat_tensor = torch.from_numpy(cat)
    num_tensor = torch.from_numpy(numeric)
    label_tensor = torch.from_numpy(
        np.asarray(labels, dtype=np.float32)
    )
    generator = torch.Generator().manual_seed(seed + 17)

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(cat), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(cat), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(cat_tensor[idx], num_tensor[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                label_tensor[idx],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
        print(
            "training family=%s epoch=%d loss=%.6f"
            % (family, epoch + 1, loss_sum / len(cat)),
            flush=True,
        )
    return model


def construct_bpr_pairs(cat, labels, seed):
    """
    For every positive row, draw several rows uniformly from the same user's
    logged impressions and retain draws that are negatives. This never treats
    an unlogged catalog item as a negative.
    """
    users = np.asarray(cat[:, 0], dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    n_users = int(CARDS[0])

    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    counts = np.bincount(sorted_users, minlength=n_users)
    starts = np.cumsum(
        np.concatenate([[0], counts[:-1]])
    ).astype(np.int64)

    positive_rows = np.flatnonzero(labels > 0)
    positive_users = users[positive_rows]
    rng = np.random.default_rng(seed)

    repeated_pos = np.repeat(positive_rows, 5)
    repeated_users = np.repeat(positive_users, 5)
    group_counts = counts[repeated_users]

    random_fraction = rng.random(len(repeated_pos))
    offsets = np.minimum(
        (random_fraction * group_counts).astype(np.int64),
        group_counts - 1,
    )
    candidate_positions = starts[repeated_users] + offsets
    negative_rows = order[candidate_positions]

    keep = labels[negative_rows] == 0
    pos = repeated_pos[keep]
    neg = negative_rows[keep]

    if len(pos) > 1500000:
        chosen = rng.choice(
            len(pos), size=1500000, replace=False
        )
        pos = pos[chosen]
        neg = neg[chosen]

    return (
        np.ascontiguousarray(pos, dtype=np.int64),
        np.ascontiguousarray(neg, dtype=np.int64),
    )


def fit_bpr(cat, numeric, labels, seed):
    torch.manual_seed(seed)
    model = BPRFM()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0012,
        weight_decay=3e-6,
    )

    pos_rows, neg_rows = construct_bpr_pairs(cat, labels, seed + 31)
    cat_tensor = torch.from_numpy(cat)
    num_tensor = torch.from_numpy(numeric)
    generator = torch.Generator().manual_seed(seed + 37)

    print(
        "bpr_pairs=%d positives=%d"
        % (len(pos_rows), int(np.sum(labels))),
        flush=True,
    )

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(pos_rows), generator=generator)
        loss_sum = 0.0

        for start in range(0, len(pos_rows), BATCH_SIZE):
            pair_idx = order[start:start + BATCH_SIZE]
            pidx = torch.from_numpy(pos_rows)[pair_idx]
            nidx = torch.from_numpy(neg_rows)[pair_idx]

            optimizer.zero_grad(set_to_none=True)
            positive_score = model(
                cat_tensor[pidx], num_tensor[pidx]
            )
            negative_score = model(
                cat_tensor[nidx], num_tensor[nidx]
            )
            loss = nn.functional.softplus(
                -(positive_score - negative_score)
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(pair_idx)

        print(
            "training family=bpr_fm epoch=%d loss=%.6f"
            % (epoch + 1, loss_sum / len(pos_rows)),
            flush=True,
        )

    return model


families = ["nfm", "dcn", "bpr_fm"]
family_predictions = {}

for family_index, family in enumerate(families):
    seed = SEED + 1000 * (family_index + 1)
    if family == "bpr_fm":
        model = fit_bpr(
            x_train, n_train, y_train, seed
        )
    else:
        model = fit_pointwise(
            family, x_train, n_train, y_train, seed
        )
    family_predictions[family] = predict_model(
        model, x_valid, n_valid
    )
    del model
    gc.collect()

candidate_scores = {}
candidate_predictions = {}
candidate_recipe = {}

inc_metric = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_recipe["incumbent"] = ("incumbent", 0.0)

inc_rank = within_user_rank(valid.user_id, inc_valid)
blend_weights = [0.20, 0.35, 0.50, 0.65, 0.80]

for family, prediction in family_predictions.items():
    prediction = np.asarray(prediction, dtype=np.float64)
    raw_metric = evaluate(valid.user_id, y_valid, prediction)
    candidate_scores[family] = float(raw_metric["primary"])
    candidate_predictions[family] = prediction
    candidate_recipe[family] = (family, 1.0)

    new_rank = within_user_rank(valid.user_id, prediction)
    for new_weight in blend_weights:
        name = "%s_inc_blend_%02d" % (
            family,
            int(round(100 * new_weight)),
        )
        blended = (
            (1.0 - new_weight) * inc_rank
            + new_weight * new_rank
        )
        metric = evaluate(valid.user_id, y_valid, blended)
        candidate_scores[name] = float(metric["primary"])
        candidate_predictions[name] = blended
        candidate_recipe[name] = (family, new_weight)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_predictions[winner]
winner_family, winner_weight = candidate_recipe[winner]
valid_metric = evaluate(valid.user_id, y_valid, valid_scores)

print(
    "CANDIDATES " + json.dumps(
        candidate_scores, sort_keys=True
    ),
    flush=True,
)
print(
    "FINDINGS winner=%s family=%s new_family_weight=%.2f"
    % (winner, winner_family, winner_weight),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Apply the validation-selected recipe to test. A selected new family is
# refitted from scratch on train+validation with the same architecture,
# objective, epoch count, and preprocessing recipe.
test = load("test")

if winner_family == "incumbent":
    test_scores = np.asarray(
        np.load(inc_test_path),
        dtype=np.float64,
    )
else:
    x_test = make_cat(test)
    raw_test_num = raw_num(test)

    x_combined = np.ascontiguousarray(
        np.concatenate([x_train, x_valid], axis=0),
        dtype=np.int64,
    )
    raw_combined_num = np.ascontiguousarray(
        np.concatenate([raw_train_num, raw_valid_num], axis=0),
        dtype=np.float32,
    )
    y_combined = np.ascontiguousarray(
        np.concatenate([
            y_train,
            y_valid.astype(np.float32),
        ]),
        dtype=np.float32,
    )

    combined_num_stats = fit_numeric_transform(raw_combined_num)
    n_combined = transform_numeric(
        raw_combined_num, combined_num_stats
    )
    n_test = transform_numeric(
        raw_test_num, combined_num_stats
    )

    family_index = families.index(winner_family)
    refit_seed = SEED + 1000 * (family_index + 1)
    if winner_family == "bpr_fm":
        final_model = fit_bpr(
            x_combined,
            n_combined,
            y_combined,
            refit_seed,
        )
    else:
        final_model = fit_pointwise(
            winner_family,
            x_combined,
            n_combined,
            y_combined,
            refit_seed,
        )

    new_test_prediction = predict_model(
        final_model, x_test, n_test
    )
    del final_model
    gc.collect()

    if winner_weight >= 0.999:
        test_scores = new_test_prediction
    else:
        incumbent_test = np.asarray(
            np.load(inc_test_path),
            dtype=np.float64,
        )
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        new_test_rank = within_user_rank(
            test.user_id, new_test_prediction
        )
        test_scores = (
            (1.0 - winner_weight) * incumbent_test_rank
            + winner_weight * new_test_rank
        )

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(valid_metric["primary"]),
        "gauc": float(valid_metric["gauc"]),
        "ndcg@5": float(valid_metric["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)