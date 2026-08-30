import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 73129
BATCH_SIZE = 8192
EPOCHS = 2
EMBED_DIM = 12
NEGATIVES = 3
HALF_LIFE_DAYS = 4.0
HASH_SIZE = 1 << 20

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "hour",
    "upload_type",
    "music_type",
    "user_active_degree",
]

NUMERIC_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

USER_TOWER_FIELDS = [0, 9]
ITEM_TOWER_FIELDS = [1, 2, 3, 4, 7, 8]
CONTEXT_FIELDS = [5, 6]
HASH_CROSS_FIELDS = [1, 2, 3, 4, 5, 6, 7, 8]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def categorical_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[name] for name in FIELDS]),
        dtype=np.int64,
    )


def numeric_raw(split):
    columns = []
    for name in NUMERIC_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def fit_numeric(raw):
    mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = raw.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-4)


def normalize_numeric(raw, mean, std):
    return np.ascontiguousarray(
        (raw - mean[None, :]) / std[None, :], dtype=np.float32
    )


def date_ages(date):
    values = np.asarray(date, dtype=np.int64)
    unique = np.unique(values)
    ranks = np.searchsorted(unique, values)
    return (len(unique) - 1 - ranks).astype(np.float32)


def make_data(split, mean=None, std=None):
    x = categorical_matrix(split)
    raw = numeric_raw(split)
    if mean is None:
        mean, std = fit_numeric(raw)
    numeric = normalize_numeric(raw, mean, std)
    return {
        "x": x,
        "numeric": numeric,
        "user": np.asarray(split.user_id, dtype=np.int64),
        "date": np.asarray(split.date, dtype=np.int64),
        "mean": mean,
        "std": std,
    }


def concatenate_training(a, b):
    raw_a = numeric_raw(a)
    raw_b = numeric_raw(b)
    raw = np.concatenate([raw_a, raw_b], axis=0)
    mean, std = fit_numeric(raw)
    return {
        "x": np.concatenate(
            [categorical_matrix(a), categorical_matrix(b)], axis=0
        ),
        "numeric": normalize_numeric(raw, mean, std),
        "user": np.concatenate(
            [
                np.asarray(a.user_id, dtype=np.int64),
                np.asarray(b.user_id, dtype=np.int64),
            ]
        ),
        "date": np.concatenate(
            [
                np.asarray(a.date, dtype=np.int64),
                np.asarray(b.date, dtype=np.int64),
            ]
        ),
        "y": np.concatenate(
            [
                np.asarray(a.y, dtype=np.int8),
                np.asarray(b.y, dtype=np.int8),
            ]
        ),
        "mean": mean,
        "std": std,
    }


def make_hashed_crosses(x):
    user = x[:, 0].astype(np.uint64)
    output = np.empty((len(x), len(HASH_CROSS_FIELDS)), dtype=np.int64)
    mask = np.uint64(HASH_SIZE - 1)
    for j, field_index in enumerate(HASH_CROSS_FIELDS):
        value = x[:, field_index].astype(np.uint64)
        h = (
            user * np.uint64(11995408973635179863)
            + value * np.uint64(10150724397891781847)
            + np.uint64((j + 1) * 2654435761)
        )
        h ^= h >> np.uint64(31)
        output[:, j] = (h & mask).astype(np.int64)
    return np.ascontiguousarray(output)


def build_training_sets(user, y, date):
    user = np.asarray(user, dtype=np.int64)
    y = np.asarray(y, dtype=np.int8)
    n_users = max(FEATURE_CARDINALITIES["user_id"], int(user.max()) + 1)

    positive = np.flatnonzero(y == 1).astype(np.int64)
    negative = np.flatnonzero(y == 0).astype(np.int64)

    negative_order = np.argsort(user[negative], kind="stable")
    negative = negative[negative_order]
    neg_users = user[negative]

    negative_counts = np.bincount(neg_users, minlength=n_users).astype(np.int64)
    negative_starts = np.empty(n_users, dtype=np.int64)
    negative_starts[0] = 0
    np.cumsum(negative_counts[:-1], out=negative_starts[1:])

    valid_positive = negative_counts[user[positive]] > 0
    positive = positive[valid_positive]
    positive_users = user[positive]

    candidates = np.empty((len(positive), NEGATIVES), dtype=np.int64)
    base_hash = positive.astype(np.uint64) * np.uint64(11400714819323198485)
    for k in range(NEGATIVES):
        h = base_hash + np.uint64((k + 1) * 1442695040888963407)
        offset = (h % negative_counts[positive_users].astype(np.uint64)).astype(
            np.int64
        )
        candidates[:, k] = negative_starts[positive_users] + offset
    candidates = negative[candidates]

    ages = date_ages(date)
    row_weight = np.exp2(-ages / HALF_LIFE_DAYS).astype(np.float32)
    pair_weight = np.sqrt(
        row_weight[positive, None] * row_weight[candidates]
    ).mean(axis=1)
    pair_weight /= max(float(pair_weight.mean()), 1e-6)

    return positive, candidates, pair_weight.astype(np.float32)


class SharedCategoricalEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        cards = [FEATURE_CARDINALITIES[name] for name in FIELDS]
        offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )
        self.total = int(sum(cards))
        self.embedding = nn.Embedding(self.total, dim)
        self.linear = nn.Embedding(self.total, 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def encode(self, x):
        indexed = x + self.offsets
        return self.embedding(indexed), self.linear(indexed).sum(1).squeeze(-1)


class NFMRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedCategoricalEncoder(EMBED_DIM)
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM + len(NUMERIC_FIELDS), 48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, numeric):
        embeddings, wide = self.encoder.encode(x)
        summed = embeddings.sum(dim=1)
        biinteraction = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        )
        deep_input = torch.cat([biinteraction, numeric], dim=1)
        return wide + self.head(deep_input).squeeze(-1) + self.bias


class TwoTowerRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedCategoricalEncoder(EMBED_DIM)
        self.user_projection = nn.Sequential(
            nn.Linear(len(USER_TOWER_FIELDS) * EMBED_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, EMBED_DIM),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(
                len(ITEM_TOWER_FIELDS) * EMBED_DIM + len(NUMERIC_FIELDS),
                48,
            ),
            nn.ReLU(),
            nn.Linear(48, EMBED_DIM),
        )
        self.context = nn.Sequential(
            nn.Linear(len(CONTEXT_FIELDS) * EMBED_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, numeric):
        embeddings, wide = self.encoder.encode(x)
        user_part = embeddings[:, USER_TOWER_FIELDS].flatten(1)
        item_part = torch.cat(
            [embeddings[:, ITEM_TOWER_FIELDS].flatten(1), numeric], dim=1
        )
        context_part = embeddings[:, CONTEXT_FIELDS].flatten(1)

        user_vector = F.normalize(self.user_projection(user_part), dim=1)
        item_vector = F.normalize(self.item_projection(item_part), dim=1)
        scale = torch.exp(self.temperature).clamp(0.5, 20.0)
        interaction = scale * (user_vector * item_vector).sum(dim=1)
        return interaction + self.context(context_part).squeeze(-1) + 0.15 * wide


class HashedCrossRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross = nn.Embedding(HASH_SIZE, 1)
        self.field_weight = nn.Parameter(
            torch.ones(len(HASH_CROSS_FIELDS), dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.cross.weight)

    def forward(self, hashed):
        values = self.cross(hashed).squeeze(-1)
        return (values * self.field_weight).sum(dim=1) + self.bias


def create_model(kind):
    if kind == "nfm_hard_bpr":
        return NFMRanker()
    if kind == "two_tower_softmax":
        return TwoTowerRanker()
    if kind == "hashed_cross_bpr":
        return HashedCrossRanker()
    raise ValueError(kind)


def score_rows(model, kind, x, numeric, hashed=None):
    if kind == "hashed_cross_bpr":
        return model(hashed)
    return model(x, numeric)


def fit_model(data, y, kind, seed):
    seed_all(seed)
    positive, negatives, pair_weight = build_training_sets(
        data["user"], y, data["date"]
    )

    x = torch.from_numpy(data["x"])
    numeric = torch.from_numpy(data["numeric"])
    hashed = None
    if kind == "hashed_cross_bpr":
        hashed = torch.from_numpy(make_hashed_crosses(data["x"]))

    pos_tensor = torch.from_numpy(positive)
    neg_tensor = torch.from_numpy(negatives)
    weight_tensor = torch.from_numpy(pair_weight)

    model = create_model(kind)
    if kind == "hashed_cross_bpr":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.012)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.0018, weight_decay=2e-6
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    n = len(positive)

    model.train()
    for epoch in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            selected = permutation[start:start + BATCH_SIZE]
            p = pos_tensor[selected]
            neg = neg_tensor[selected]
            weight = weight_tensor[selected]

            pos_score = score_rows(
                model,
                kind,
                x[p],
                numeric[p],
                None if hashed is None else hashed[p],
            )

            flat_neg = neg.reshape(-1)
            neg_score = score_rows(
                model,
                kind,
                x[flat_neg],
                numeric[flat_neg],
                None if hashed is None else hashed[flat_neg],
            ).reshape(len(selected), NEGATIVES)

            if kind == "two_tower_softmax":
                logits = torch.cat([pos_score[:, None], neg_score], dim=1)
                losses = -F.log_softmax(logits, dim=1)[:, 0]
            elif kind == "nfm_hard_bpr":
                hard_negative = neg_score.max(dim=1).values
                losses = F.softplus(hard_negative - pos_score)
            else:
                losses = F.softplus(
                    neg_score - pos_score[:, None]
                ).mean(dim=1)

            loss = (losses * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict(model, data, kind):
    x = torch.from_numpy(data["x"])
    numeric = torch.from_numpy(data["numeric"])
    hashed = None
    if kind == "hashed_cross_bpr":
        hashed = torch.from_numpy(make_hashed_crosses(data["x"]))

    result = np.empty(len(data["x"]), dtype=np.float32)
    step = BATCH_SIZE * 3
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(result), step):
            end = min(start + step, len(result))
            output = score_rows(
                model,
                kind,
                x[start:end],
                numeric[start:end],
                None if hashed is None else hashed[start:end],
            )
            result[start:end] = output.cpu().numpy()
    return result.astype(np.float64)


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, user_ids))
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    positions = np.arange(n) - np.repeat(starts, sizes)
    denominator = np.maximum(np.repeat(sizes, sizes) - 1, 1)

    rank = np.empty(n, dtype=np.float64)
    rank[order] = positions / denominator
    return rank


def transformed_blend(candidate, incumbent, users, mode, weight):
    if mode == "raw":
        return weight * zscore(candidate) + (1.0 - weight) * zscore(incumbent)
    if mode == "rank":
        return (
            weight * within_user_rank(users, candidate)
            + (1.0 - weight) * within_user_rank(users, incumbent)
        )
    if mode == "standalone":
        return np.asarray(candidate, dtype=np.float64)
    raise ValueError(mode)


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_all(SEED)

    train = load("train")
    valid = load("valid")
    y_train = np.asarray(train.y, dtype=np.int8)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    train_raw = numeric_raw(train)
    mean, std = fit_numeric(train_raw)
    train_data = make_data(train, mean, std)
    valid_data = make_data(valid, mean, std)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid = np.load(
        os.path.join(shared, "incumbent_valid_scores.npy")
    ).astype(np.float64)
    incumbent_test_path = os.path.join(
        shared, "incumbent_test_scores.npy"
    )

    incumbent_metrics = evaluate(
        valid.user_id, y_valid, incumbent_valid
    )
    candidates = {
        "trusted_incumbent": float(incumbent_metrics["primary"])
    }

    best_primary = float(incumbent_metrics["primary"])
    best_valid = incumbent_valid.copy()
    best_descriptor = {
        "kind": "incumbent",
        "mode": "incumbent",
        "weight": 0.0,
    }

    kinds = [
        "nfm_hard_bpr",
        "two_tower_softmax",
        "hashed_cross_bpr",
    ]
    blend_weights = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70]

    for index, kind in enumerate(kinds):
        model = fit_model(
            train_data, y_train, kind, SEED + 503 * index
        )
        prediction = predict(model, valid_data, kind)
        del model
        gc.collect()

        standalone = float(
            evaluate(valid.user_id, y_valid, prediction)["primary"]
        )
        candidates[kind + "_standalone"] = standalone

        if standalone > best_primary:
            best_primary = standalone
            best_valid = prediction.copy()
            best_descriptor = {
                "kind": kind,
                "mode": "standalone",
                "weight": 1.0,
            }

        for mode in ["raw", "rank"]:
            local_best = -np.inf
            local_weight = 0.0
            for weight in blend_weights:
                blended = transformed_blend(
                    prediction,
                    incumbent_valid,
                    valid.user_id,
                    mode,
                    weight,
                )
                primary = float(
                    evaluate(valid.user_id, y_valid, blended)["primary"]
                )
                if primary > local_best:
                    local_best = primary
                    local_weight = weight
                if primary > best_primary:
                    best_primary = primary
                    best_valid = blended.copy()
                    best_descriptor = {
                        "kind": kind,
                        "mode": mode,
                        "weight": weight,
                    }

            candidates[kind + "_" + mode + "_blend"] = local_best
            candidates[kind + "_" + mode + "_weight"] = local_weight

    print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
    print("FINDINGS winner=" + json.dumps(best_descriptor, sort_keys=True))

    metrics = evaluate(valid.user_id, y_valid, best_valid)

    out = os.environ.get("ITER_OUT")
    if out:
        np.save(
            os.path.join(out, "scores_valid.npy"),
            np.asarray(best_valid, dtype=np.float64),
        )

        test = load("test")
        incumbent_test = np.load(incumbent_test_path).astype(np.float64)

        if best_descriptor["kind"] == "incumbent":
            test_scores = incumbent_test
        else:
            combined = concatenate_training(train, valid)
            test_data = make_data(
                test, combined["mean"], combined["std"]
            )
            final_model = fit_model(
                combined,
                combined["y"],
                best_descriptor["kind"],
                SEED
                + 503 * kinds.index(best_descriptor["kind"]),
            )
            test_prediction = predict(
                final_model, test_data, best_descriptor["kind"]
            )
            test_scores = transformed_blend(
                test_prediction,
                incumbent_test,
                test.user_id,
                best_descriptor["mode"],
                best_descriptor["weight"],
            )

        np.save(
            os.path.join(out, "scores_test.npy"),
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
            }
        )
    )


if __name__ == "__main__":
    main()