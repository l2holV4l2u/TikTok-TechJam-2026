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
SEED = 19437
BATCH_SIZE = 16384
EPOCHS = 3
EMBED_DIM = 10
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
]

AUX_CANDIDATES = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_profile_enter",
    "is_hate",
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[name] for name in FIELDS]),
        dtype=np.int64,
    )


def available_auxiliary_names(train, valid):
    names = []
    for name in AUX_CANDIDATES:
        if name not in train.aux or name not in valid.aux:
            continue
        a = np.asarray(train.aux[name])
        if len(a) != len(train.user_id):
            continue
        finite = a[np.isfinite(a)]
        if len(finite) == 0:
            continue
        unique = np.unique(finite[: min(len(finite), 300000)])
        if np.all(np.isin(unique, [0, 1])):
            names.append(name)
    return names[:5]


def auxiliary_matrix(split, names):
    if not names:
        return np.zeros((len(split.user_id), 0), dtype=np.float32)
    columns = []
    for name in names:
        values = np.asarray(split.aux[name], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        columns.append(np.clip(values, 0.0, 1.0))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    std = max(float(values.std()), 1e-8)
    return (values - values.mean()) / std


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


class LatentMF(nn.Module):
    def __init__(self):
        super().__init__()
        uc = FEATURE_CARDINALITIES["user_id"]
        vc = FEATURE_CARDINALITIES["video_id"]
        ac = FEATURE_CARDINALITIES["author_id"]

        self.user = nn.Embedding(uc, 24)
        self.video = nn.Embedding(vc, 24)
        self.author = nn.Embedding(ac, 24)

        self.user_bias = nn.Embedding(uc, 1)
        self.video_bias = nn.Embedding(vc, 1)
        self.author_bias = nn.Embedding(ac, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        for emb in [self.user, self.video, self.author]:
            nn.init.normal_(emb.weight, std=0.03)
        for emb in [self.user_bias, self.video_bias, self.author_bias]:
            nn.init.zeros_(emb.weight)

    def forward(self, x):
        user_id = x[:, 0]
        video_id = x[:, 1]
        author_id = x[:, 2]

        u = self.user(user_id)
        v = self.video(video_id)
        a = self.author(author_id)

        interaction = (u * v).sum(dim=1)
        interaction += 0.6 * (u * a).sum(dim=1)
        interaction += 0.4 * (v * a).sum(dim=1)

        bias = self.user_bias(user_id).squeeze(-1)
        bias += self.video_bias(video_id).squeeze(-1)
        bias += self.author_bias(author_id).squeeze(-1)
        return (interaction + bias + self.global_bias).unsqueeze(1)


class CategoricalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        cards = [FEATURE_CARDINALITIES[name] for name in FIELDS]
        offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )
        self.embedding = nn.Embedding(int(sum(cards)), EMBED_DIM)
        self.linear = nn.Embedding(int(sum(cards)), 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        indexed = x + self.offsets
        embedding = self.embedding(indexed).flatten(1)
        wide = self.linear(indexed).sum(dim=1)
        return embedding, wide


class CrossLayer(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, x0, x):
        return x0 * self.linear(x) + self.bias + x


class DCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CategoricalEncoder()
        dimension = len(FIELDS) * EMBED_DIM

        self.cross_layers = nn.ModuleList(
            [CrossLayer(dimension) for _ in range(3)]
        )
        self.deep = nn.Sequential(
            nn.Linear(dimension, 80),
            nn.ReLU(),
            nn.Linear(80, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(dimension + 32, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x0, wide = self.encoder(x)
        crossed = x0
        for layer in self.cross_layers:
            crossed = layer(x0, crossed)
        deep = self.deep(x0)
        return wide + self.output(torch.cat([crossed, deep], dim=1)) + self.bias


class MMoEModel(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.encoder = CategoricalEncoder()
        dimension = len(FIELDS) * EMBED_DIM
        expert_dim = 40
        num_experts = 4

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dimension, 72),
                nn.ReLU(),
                nn.Linear(72, expert_dim),
                nn.ReLU(),
            )
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(dimension, num_experts)
            for _ in range(num_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(num_tasks)
        ])
        self.task_wide_scale = nn.Parameter(torch.ones(num_tasks))

    def forward(self, x):
        encoded, wide = self.encoder(x)
        expert_values = torch.stack(
            [expert(encoded) for expert in self.experts], dim=1
        )

        outputs = []
        for task in range(self.num_tasks):
            gate = torch.softmax(self.gates[task](encoded), dim=1)
            mixture = (
                expert_values * gate.unsqueeze(-1)
            ).sum(dim=1)
            task_output = self.towers[task](mixture)
            task_output = (
                task_output
                + self.task_wide_scale[task] * wide
            )
            outputs.append(task_output)
        return torch.cat(outputs, dim=1)


def create_model(kind, n_aux):
    if kind == "latent_mf":
        return LatentMF()
    if kind == "dcn":
        return DCNModel()
    if kind == "mmoe":
        return MMoEModel(1 + n_aux)
    raise ValueError(kind)


def fit_model(x_np, y_np, aux_np, kind, seed):
    seed_all(seed)
    model = create_model(kind, aux_np.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=2e-6
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    aux = torch.from_numpy(aux_np)

    generator = torch.Generator()
    generator.manual_seed(seed)
    n = len(y)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x[idx])

            main_loss = criterion(logits[:, 0], y[idx]).mean()
            if kind == "mmoe" and aux.shape[1] > 0:
                aux_loss = criterion(
                    logits[:, 1:], aux[idx]
                ).mean()
                loss = main_loss + 0.22 * aux_loss
            else:
                loss = main_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict(model, x_np):
    x = torch.from_numpy(x_np)
    scores = np.empty(len(x_np), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_np), BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, len(x_np))
            scores[start:end] = (
                model(x[start:end])[:, 0].cpu().numpy()
            )
    return scores


def score_and_search_blends(
    valid,
    y_valid,
    prediction,
    incumbent,
    family,
    candidates,
):
    standalone = float(
        evaluate(valid.user_id, y_valid, prediction)["primary"]
    )
    candidates[family] = standalone

    best_score = standalone
    best_descriptor = (family, "standalone", 1.0)
    best_values = prediction.astype(np.float64)

    incumbent_z = zscore(incumbent)
    prediction_z = zscore(prediction)
    incumbent_rank = within_user_rank(valid.user_id, incumbent)
    prediction_rank = within_user_rank(valid.user_id, prediction)

    raw_best = -np.inf
    rank_best = -np.inf
    raw_weight = 0.0
    rank_weight = 0.0

    for weight in [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85]:
        raw = weight * prediction_z + (1.0 - weight) * incumbent_z
        raw_score = float(
            evaluate(valid.user_id, y_valid, raw)["primary"]
        )
        if raw_score > raw_best:
            raw_best = raw_score
            raw_weight = weight
        if raw_score > best_score:
            best_score = raw_score
            best_descriptor = (family, "raw", weight)
            best_values = raw.copy()

        ranked = (
            weight * prediction_rank
            + (1.0 - weight) * incumbent_rank
        )
        rank_score = float(
            evaluate(valid.user_id, y_valid, ranked)["primary"]
        )
        if rank_score > rank_best:
            rank_best = rank_score
            rank_weight = weight
        if rank_score > best_score:
            best_score = rank_score
            best_descriptor = (family, "rank", weight)
            best_values = ranked.copy()

    candidates[family + "_raw_blend"] = raw_best
    candidates[family + "_rank_blend"] = rank_best
    candidates[family + "_raw_weight"] = raw_weight
    candidates[family + "_rank_weight"] = rank_weight
    return best_score, best_descriptor, best_values


def apply_recipe(model_scores, incumbent_scores, user_ids, mode, weight):
    if mode == "standalone":
        return np.asarray(model_scores, dtype=np.float64)
    if mode == "raw":
        return (
            weight * zscore(model_scores)
            + (1.0 - weight) * zscore(incumbent_scores)
        )
    if mode == "rank":
        return (
            weight * within_user_rank(user_ids, model_scores)
            + (1.0 - weight)
            * within_user_rank(user_ids, incumbent_scores)
        )
    raise ValueError(mode)


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_all(SEED)

    train = load("train")
    valid = load("valid")
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    aux_names = available_auxiliary_names(train, valid)
    aux_train = auxiliary_matrix(train, aux_names)
    aux_valid = auxiliary_matrix(valid, aux_names)

    x_train = make_matrix(train)
    x_valid = make_matrix(valid)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid = np.load(
        os.path.join(shared, "incumbent_valid_scores.npy")
    ).astype(np.float64)

    incumbent_metrics = evaluate(
        valid.user_id, y_valid, incumbent_valid
    )
    candidates = {
        "trusted_incumbent": float(incumbent_metrics["primary"])
    }

    best_score = float(incumbent_metrics["primary"])
    best_descriptor = ("incumbent", "standalone", 0.0)
    best_valid = incumbent_valid.copy()

    predictions = {}
    families = ["latent_mf", "dcn", "mmoe"]

    for index, family in enumerate(families):
        model = fit_model(
            x_train,
            y_train,
            aux_train,
            family,
            SEED + 101 * index,
        )
        prediction = predict(model, x_valid)
        predictions[family] = prediction

        family_score, descriptor, values = score_and_search_blends(
            valid,
            y_valid,
            prediction,
            incumbent_valid,
            family,
            candidates,
        )
        if family_score > best_score:
            best_score = family_score
            best_descriptor = descriptor
            best_valid = values

        del model
        gc.collect()

    metrics = evaluate(valid.user_id, y_valid, best_valid)

    print(
        "FINDINGS "
        + json.dumps({
            "auxiliary_training_targets": aux_names,
            "selected_recipe": {
                "family": best_descriptor[0],
                "mode": best_descriptor[1],
                "new_model_weight": float(best_descriptor[2]),
            },
        }, sort_keys=True)
    )
    print("CANDIDATES " + json.dumps(candidates, sort_keys=True))

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(best_valid, dtype=np.float64),
        )

    test = load("test")
    incumbent_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)

    if best_descriptor[0] == "incumbent":
        test_scores = incumbent_test
    else:
        selected_family, selected_mode, selected_weight = best_descriptor

        x_combined = np.ascontiguousarray(
            np.concatenate([x_train, x_valid], axis=0),
            dtype=np.int64,
        )
        y_combined = np.concatenate([
            y_train,
            y_valid.astype(np.float32),
        ])
        aux_combined = np.ascontiguousarray(
            np.concatenate([aux_train, aux_valid], axis=0),
            dtype=np.float32,
        )

        selected_model = fit_model(
            x_combined,
            y_combined,
            aux_combined,
            selected_family,
            SEED + 101 * families.index(selected_family),
        )
        x_test = make_matrix(test)
        model_test = predict(selected_model, x_test)
        test_scores = apply_recipe(
            model_test,
            incumbent_test,
            test.user_id,
            selected_mode,
            selected_weight,
        )

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
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


if __name__ == "__main__":
    main()