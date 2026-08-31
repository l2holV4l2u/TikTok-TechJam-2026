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
SEED = 19427
EMBED_DIM = 8
BATCH_SIZE = 12288
EPOCHS = 2
LR = 0.0015

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

AUX_FIELDS = ["is_click", "is_like"]


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
    cols = []
    for name in NUMERIC_FIELDS:
        values = np.asarray(split.num[name], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.log1p(np.maximum(values, 0.0))
        cols.append(values)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def fit_numeric_transform(split):
    raw = numeric_raw(split)
    mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = raw.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def transform_numeric(split, mean, std):
    return np.ascontiguousarray(
        (numeric_raw(split) - mean[None, :]) / std[None, :],
        dtype=np.float32,
    )


def auxiliary_matrix(split):
    cols = []
    for name in AUX_FIELDS:
        if name not in split.aux:
            cols.append(np.zeros(len(split.user_id), dtype=np.float32))
        else:
            value = np.asarray(split.aux[name], dtype=np.float32)
            value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
            cols.append(np.clip(value, 0.0, 1.0))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    positions = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    denominators = np.maximum(np.repeat(sizes, sizes) - 1, 1)

    result = np.empty(n, dtype=np.float64)
    result[order] = positions / denominators
    return result


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        cards = [FEATURE_CARDINALITIES[name] for name in FIELDS]
        offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )
        total = int(sum(cards))

        self.embedding = nn.Embedding(total, EMBED_DIM)
        self.linear = nn.Embedding(total, 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        indexed = x + self.offsets
        embeddings = self.embedding(indexed)
        wide = self.linear(indexed).sum(dim=1).squeeze(-1) + self.bias
        return embeddings, wide


class AutoIntModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.attention1 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, batch_first=True
        )
        self.attention2 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, batch_first=True
        )
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        input_dim = len(FIELDS) * EMBED_DIM + len(NUMERIC_FIELDS)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x, numeric):
        embeddings, wide = self.encoder(x)
        attended, _ = self.attention1(
            embeddings, embeddings, embeddings, need_weights=False
        )
        embeddings = self.norm1(embeddings + attended)
        attended, _ = self.attention2(
            embeddings, embeddings, embeddings, need_weights=False
        )
        embeddings = self.norm2(embeddings + attended)
        deep = torch.cat([embeddings.flatten(1), numeric], dim=1)
        return wide + self.head(deep).squeeze(-1)


class FiBiNETModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        field_count = len(FIELDS)

        self.squeeze = nn.Sequential(
            nn.Linear(field_count, max(4, field_count // 2)),
            nn.ReLU(),
            nn.Linear(max(4, field_count // 2), field_count),
            nn.Sigmoid(),
        )

        pair_i = []
        pair_j = []
        for i in range(field_count):
            for j in range(i + 1, field_count):
                pair_i.append(i)
                pair_j.append(j)
        self.register_buffer(
            "pair_i", torch.tensor(pair_i, dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.tensor(pair_j, dtype=torch.long)
        )

        pair_dim = len(pair_i) * EMBED_DIM
        input_dim = pair_dim + field_count * EMBED_DIM + len(NUMERIC_FIELDS)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, numeric):
        embeddings, wide = self.encoder(x)
        squeeze_stat = embeddings.mean(dim=2)
        gates = self.squeeze(squeeze_stat).unsqueeze(-1)
        gated = embeddings * gates

        pairwise = (
            gated.index_select(1, self.pair_i)
            * embeddings.index_select(1, self.pair_j)
        )
        deep = torch.cat(
            [gated.flatten(1), pairwise.flatten(1), numeric], dim=1
        )
        return wide + self.head(deep).squeeze(-1)


class MMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        input_dim = len(FIELDS) * EMBED_DIM + len(NUMERIC_FIELDS)
        hidden_dim = 64
        expert_count = 4
        task_count = 1 + len(AUX_FIELDS)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            for _ in range(expert_count)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, expert_count)
            for _ in range(task_count)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(task_count)
        ])

    def forward(self, x, numeric):
        embeddings, wide = self.encoder(x)
        base = torch.cat([embeddings.flatten(1), numeric], dim=1)
        experts = torch.stack(
            [expert(base) for expert in self.experts], dim=1
        )

        outputs = []
        for task_index, (gate, tower) in enumerate(
            zip(self.gates, self.towers)
        ):
            weights = torch.softmax(gate(base), dim=1).unsqueeze(-1)
            representation = (experts * weights).sum(dim=1)
            output = tower(representation).squeeze(-1)
            if task_index == 0:
                output = output + wide
            outputs.append(output)
        return outputs


def create_model(kind):
    if kind == "autoint":
        return AutoIntModel()
    if kind == "fibinet":
        return FiBiNETModel()
    if kind == "mmoe":
        return MMoEModel()
    raise ValueError(kind)


def fit_model(x_np, numeric_np, y_np, aux_np, kind, seed):
    seed_all(seed)
    x = torch.from_numpy(x_np)
    numeric = torch.from_numpy(numeric_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    aux = torch.from_numpy(np.asarray(aux_np, dtype=np.float32))

    model = create_model(kind)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = len(y)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            result = model(x[idx], numeric[idx])

            if kind == "mmoe":
                loss = criterion(result[0], y[idx])
                loss = loss + 0.30 * criterion(result[1], aux[idx, 0])
                loss = loss + 0.20 * criterion(result[2], aux[idx, 1])
            else:
                loss = criterion(result, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict(model, x_np, numeric_np, kind):
    x = torch.from_numpy(x_np)
    numeric = torch.from_numpy(numeric_np)
    result = np.empty(len(x_np), dtype=np.float32)
    step = BATCH_SIZE * 2

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_np), step):
            end = min(start + step, len(x_np))
            output = model(x[start:end], numeric[start:end])
            if kind == "mmoe":
                output = output[0]
            result[start:end] = output.cpu().numpy()
    return result


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_all(SEED)

    train = load("train")
    valid = load("valid")
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    x_train = categorical_matrix(train)
    x_valid = categorical_matrix(valid)
    mean, std = fit_numeric_transform(train)
    n_train = transform_numeric(train, mean, std)
    n_valid = transform_numeric(valid, mean, std)
    aux_train = auxiliary_matrix(train)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid = np.load(
        os.path.join(shared, "incumbent_valid_scores.npy")
    ).astype(np.float64)
    incumbent_test_path = os.path.join(
        shared, "incumbent_test_scores.npy"
    )

    incumbent_z = zscore(incumbent_valid)
    incumbent_rank = within_user_rank(valid.user_id, incumbent_valid)

    incumbent_metrics = evaluate(
        valid.user_id, y_valid, incumbent_valid
    )
    candidates = {
        "trusted_incumbent": float(incumbent_metrics["primary"])
    }

    best_primary = float(incumbent_metrics["primary"])
    best_valid = incumbent_valid.copy()
    best_descriptor = ("incumbent", "standalone", 0.0)

    blend_weights = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70]
    kinds = ["autoint", "fibinet", "mmoe"]

    for model_index, kind in enumerate(kinds):
        model = fit_model(
            x_train,
            n_train,
            y_train,
            aux_train,
            kind,
            SEED + 101 * model_index,
        )
        prediction = predict(model, x_valid, n_valid, kind).astype(
            np.float64
        )
        del model
        gc.collect()

        standalone = float(
            evaluate(valid.user_id, y_valid, prediction)["primary"]
        )
        candidates[kind + "_standalone"] = standalone

        if standalone > best_primary:
            best_primary = standalone
            best_valid = prediction.copy()
            best_descriptor = (kind, "standalone", 1.0)

        prediction_z = zscore(prediction)
        prediction_rank = within_user_rank(valid.user_id, prediction)

        best_raw_local = -np.inf
        best_rank_local = -np.inf
        best_raw_weight = 0.0
        best_rank_weight = 0.0

        for weight in blend_weights:
            raw_blend = (
                weight * prediction_z
                + (1.0 - weight) * incumbent_z
            )
            raw_primary = float(
                evaluate(valid.user_id, y_valid, raw_blend)["primary"]
            )
            if raw_primary > best_raw_local:
                best_raw_local = raw_primary
                best_raw_weight = weight
            if raw_primary > best_primary:
                best_primary = raw_primary
                best_valid = raw_blend.copy()
                best_descriptor = (kind, "raw_blend", weight)

            rank_blend = (
                weight * prediction_rank
                + (1.0 - weight) * incumbent_rank
            )
            rank_primary = float(
                evaluate(valid.user_id, y_valid, rank_blend)["primary"]
            )
            if rank_primary > best_rank_local:
                best_rank_local = rank_primary
                best_rank_weight = weight
            if rank_primary > best_primary:
                best_primary = rank_primary
                best_valid = rank_blend.copy()
                best_descriptor = (kind, "rank_blend", weight)

        candidates[kind + "_best_raw_blend"] = best_raw_local
        candidates[kind + "_best_raw_weight"] = best_raw_weight
        candidates[kind + "_best_rank_blend"] = best_rank_local
        candidates[kind + "_best_rank_weight"] = best_rank_weight

    metrics = evaluate(valid.user_id, y_valid, best_valid)

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(best_valid, dtype=np.float64),
        )

    test = load("test")
    incumbent_test = np.load(incumbent_test_path).astype(np.float64)

    selected_kind, selected_mode, selected_weight = best_descriptor

    if selected_kind == "incumbent":
        test_scores = incumbent_test
    else:
        x_combined = np.ascontiguousarray(
            np.concatenate([x_train, x_valid], axis=0),
            dtype=np.int64,
        )
        y_combined = np.concatenate(
            [y_train, y_valid.astype(np.float32)]
        )
        aux_valid = auxiliary_matrix(valid)
        aux_combined = np.ascontiguousarray(
            np.concatenate([aux_train, aux_valid], axis=0),
            dtype=np.float32,
        )

        combined_raw = np.ascontiguousarray(
            np.concatenate([numeric_raw(train), numeric_raw(valid)], axis=0),
            dtype=np.float32,
        )
        combined_mean = combined_raw.mean(
            axis=0, dtype=np.float64
        ).astype(np.float32)
        combined_std = combined_raw.std(
            axis=0, dtype=np.float64
        ).astype(np.float32)
        combined_std = np.maximum(combined_std, 1e-4)
        n_combined = np.ascontiguousarray(
            (combined_raw - combined_mean[None, :])
            / combined_std[None, :],
            dtype=np.float32,
        )

        x_test = categorical_matrix(test)
        n_test = transform_numeric(test, combined_mean, combined_std)

        final_model = fit_model(
            x_combined,
            n_combined,
            y_combined,
            aux_combined,
            selected_kind,
            SEED + 909,
        )
        model_test = predict(
            final_model, x_test, n_test, selected_kind
        ).astype(np.float64)

        if selected_mode == "standalone":
            test_scores = model_test
        elif selected_mode == "raw_blend":
            test_scores = (
                selected_weight * zscore(model_test)
                + (1.0 - selected_weight) * zscore(incumbent_test)
            )
        else:
            test_scores = (
                selected_weight
                * within_user_rank(test.user_id, model_test)
                + (1.0 - selected_weight)
                * within_user_rank(test.user_id, incumbent_test)
            )

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    candidates["selected_primary"] = float(metrics["primary"])
    print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
    print(
        "FINDINGS selected="
        + json.dumps({
            "family": selected_kind,
            "mode": selected_mode,
            "weight": float(selected_weight),
        }, sort_keys=True)
    )

    elapsed = time.time() - START
    final = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }
    print("METRICS " + json.dumps(final))


if __name__ == "__main__":
    main()