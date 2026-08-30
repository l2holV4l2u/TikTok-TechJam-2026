import os
import time
import json
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 9127
BATCH_SIZE = 16384
EPOCHS = 2
EMBED_DIM = 12
N_EXPERTS = 3
EXPERT_DIM = 48
SVD_DIM = 32

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
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
]

AUXILIARY_CANDIDATES = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[name] for name in FIELDS]),
        dtype=np.int64,
    )


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(float(x.std()), 1e-8)


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


def available_auxiliary_names(split):
    names = []
    for name in AUXILIARY_CANDIDATES:
        if name not in split.aux:
            continue
        values = np.asarray(split.aux[name])
        if len(values) != len(split.user_id):
            continue
        finite = np.isfinite(values)
        if not finite.all():
            continue
        unique = np.unique(values)
        if len(unique) <= 2 and np.all(np.isin(unique, [0, 1])):
            names.append(name)
    return names


def task_labels(split, aux_names):
    columns = [np.asarray(split.y, dtype=np.float32)]
    for name in aux_names:
        columns.append(np.asarray(split.aux[name], dtype=np.float32))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


class MMoE(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        cards = [FEATURE_CARDINALITIES[name] for name in FIELDS]
        offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
        total_cardinality = int(sum(cards))

        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, std=0.025)

        self.wide = nn.Embedding(total_cardinality, 1)
        nn.init.zeros_(self.wide.weight)

        input_dim = len(FIELDS) * EMBED_DIM
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, EXPERT_DIM),
                    nn.ReLU(),
                    nn.Linear(EXPERT_DIM, EXPERT_DIM),
                    nn.ReLU(),
                )
                for _ in range(N_EXPERTS)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, N_EXPERTS) for _ in range(n_tasks)]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(EXPERT_DIM, 24),
                    nn.ReLU(),
                    nn.Linear(24, 1),
                )
                for _ in range(n_tasks)
            ]
        )
        self.task_bias = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, x):
        ids = x + self.offsets
        embedded = self.embedding(ids).flatten(1)
        expert_values = torch.stack(
            [expert(embedded) for expert in self.experts], dim=1
        )

        outputs = []
        primary_wide = self.wide(ids).sum(dim=1).squeeze(-1)
        for task_index, (gate, tower) in enumerate(
            zip(self.gates, self.towers)
        ):
            weights = torch.softmax(gate(embedded), dim=1)
            mixed = (
                expert_values * weights.unsqueeze(-1)
            ).sum(dim=1)
            logits = tower(mixed).squeeze(-1) + self.task_bias[task_index]
            if task_index == 0:
                logits = logits + primary_wide
            outputs.append(logits)
        return torch.stack(outputs, dim=1)


def fit_mmoe(x_np, labels_np, seed):
    seed_all(seed)
    x = torch.from_numpy(x_np)
    labels = torch.from_numpy(labels_np)
    n_tasks = labels_np.shape[1]

    model = MMoE(n_tasks)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0015, weight_decay=1e-6
    )

    positive = labels_np.sum(axis=0)
    negative = len(labels_np) - positive
    pos_weight = np.sqrt(
        (negative + 1.0) / (positive + 1.0)
    ).clip(0.75, 5.0).astype(np.float32)
    pos_weight = torch.from_numpy(pos_weight)

    task_weight = np.ones(n_tasks, dtype=np.float32)
    if n_tasks > 1:
        task_weight[1:] = 0.30
    task_weight /= task_weight.sum()
    task_weight = torch.from_numpy(task_weight)

    generator = torch.Generator()
    generator.manual_seed(seed)
    n = len(x_np)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x[idx])
            batch_labels = labels[idx]

            element_loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch_labels,
                reduction="none",
                pos_weight=pos_weight,
            )
            loss = (
                element_loss.mean(dim=0) * task_weight
            ).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_mmoe(model, x_np):
    x = torch.from_numpy(x_np)
    output = np.empty(len(x_np), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_np), BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, len(x_np))
            output[start:end] = model(x[start:end])[:, 0].cpu().numpy()
    return output


def fit_implicit_svd(user_ids, video_ids, labels, seed):
    users = np.asarray(user_ids, dtype=np.int64)
    videos = np.asarray(video_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float32)

    n_users = FEATURE_CARDINALITIES["user_id"]
    n_videos = FEATURE_CARDINALITIES["video_id"]

    # Positive implicit interactions, with repeated positive impressions
    # increasing confidence. A small observed-impression term prevents the
    # decomposition from depending only on prolific positive users.
    values = 0.08 + 0.92 * labels
    matrix = sp.coo_matrix(
        (values, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    user_degree = np.asarray(matrix.sum(axis=1)).ravel()
    video_degree = np.asarray(matrix.sum(axis=0)).ravel()
    user_scale = 1.0 / np.sqrt(np.maximum(user_degree, 1.0))
    video_scale = 1.0 / np.sqrt(np.maximum(video_degree, 1.0))
    normalized = sp.diags(user_scale).dot(matrix).dot(
        sp.diags(video_scale)
    )

    rank = min(
        SVD_DIM,
        normalized.shape[0] - 1,
        normalized.shape[1] - 1,
    )
    u, singular, vt = svds(
        normalized,
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=seed,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order]

    user_factors = (
        u * np.sqrt(singular)[None, :]
    ) / np.maximum(user_scale[:, None], 1e-8)
    video_factors = (
        vt.T * np.sqrt(singular)[None, :]
    ) / np.maximum(video_scale[:, None], 1e-8)

    # Smoothed item preference adds the rank-one popularity component that
    # centered low-rank decompositions otherwise represent inefficiently.
    positive_by_video = np.bincount(
        videos, weights=labels, minlength=n_videos
    ).astype(np.float64)
    count_by_video = np.bincount(
        videos, minlength=n_videos
    ).astype(np.float64)
    global_rate = float(labels.mean())
    item_rate = (
        positive_by_video + 30.0 * global_rate
    ) / (count_by_video + 30.0)
    item_logit = np.log(
        np.clip(item_rate, 1e-5, 1 - 1e-5)
        / np.clip(1.0 - item_rate, 1e-5, 1.0)
    )

    return (
        user_factors.astype(np.float32),
        video_factors.astype(np.float32),
        item_logit.astype(np.float32),
    )


def predict_implicit_svd(model, user_ids, video_ids):
    user_factors, video_factors, item_logit = model
    users = np.asarray(user_ids, dtype=np.int64)
    videos = np.asarray(video_ids, dtype=np.int64)
    latent = np.einsum(
        "ij,ij->i",
        user_factors[users],
        video_factors[videos],
        optimize=True,
    )
    return (latent + 0.15 * item_logit[videos]).astype(np.float32)


def concatenate_features(a, b):
    return np.ascontiguousarray(
        np.concatenate([feature_matrix(a), feature_matrix(b)], axis=0),
        dtype=np.int64,
    )


def choose_candidates(
    valid,
    y_valid,
    incumbent,
    family_predictions,
):
    candidates = {}
    incumbent = np.asarray(incumbent, dtype=np.float64)
    candidates["trusted_incumbent"] = float(
        evaluate(valid.user_id, y_valid, incumbent)["primary"]
    )

    incumbent_z = zscore(incumbent)
    incumbent_rank = within_user_rank(valid.user_id, incumbent)

    best_score = candidates["trusted_incumbent"]
    best_scores = incumbent.copy()
    best_descriptor = ("incumbent", "standalone", 0.0)

    weights = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85]

    for family, prediction in family_predictions.items():
        prediction = np.asarray(prediction, dtype=np.float64)
        standalone = float(
            evaluate(valid.user_id, y_valid, prediction)["primary"]
        )
        candidates[family + "_standalone"] = standalone

        if standalone > best_score:
            best_score = standalone
            best_scores = prediction.copy()
            best_descriptor = (family, "standalone", 1.0)

        prediction_z = zscore(prediction)
        prediction_rank = within_user_rank(valid.user_id, prediction)

        best_raw = -np.inf
        best_raw_weight = 0.0
        best_rank = -np.inf
        best_rank_weight = 0.0

        for weight in weights:
            raw = (
                weight * prediction_z
                + (1.0 - weight) * incumbent_z
            )
            raw_score = float(
                evaluate(valid.user_id, y_valid, raw)["primary"]
            )
            if raw_score > best_raw:
                best_raw = raw_score
                best_raw_weight = weight
            if raw_score > best_score:
                best_score = raw_score
                best_scores = raw.copy()
                best_descriptor = (family, "raw_blend", weight)

            ranked = (
                weight * prediction_rank
                + (1.0 - weight) * incumbent_rank
            )
            rank_score = float(
                evaluate(valid.user_id, y_valid, ranked)["primary"]
            )
            if rank_score > best_rank:
                best_rank = rank_score
                best_rank_weight = weight
            if rank_score > best_score:
                best_score = rank_score
                best_scores = ranked.copy()
                best_descriptor = (family, "rank_blend", weight)

        candidates[family + "_raw_blend"] = best_raw
        candidates[family + "_raw_weight"] = best_raw_weight
        candidates[family + "_rank_blend"] = best_rank
        candidates[family + "_rank_weight"] = best_rank_weight

    return candidates, best_descriptor, best_scores


def form_test_scores(
    descriptor,
    family_test_prediction,
    incumbent_test,
    test_user_ids,
):
    family, mode, weight = descriptor
    incumbent_test = np.asarray(incumbent_test, dtype=np.float64)

    if family == "incumbent":
        return incumbent_test.copy()

    prediction = np.asarray(family_test_prediction, dtype=np.float64)
    if mode == "standalone":
        return prediction
    if mode == "raw_blend":
        return (
            weight * zscore(prediction)
            + (1.0 - weight) * zscore(incumbent_test)
        )
    return (
        weight * within_user_rank(test_user_ids, prediction)
        + (1.0 - weight)
        * within_user_rank(test_user_ids, incumbent_test)
    )


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_all(SEED)

    train = load("train")
    valid = load("valid")
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid = np.load(
        os.path.join(shared_dir, "incumbent_valid_scores.npy")
    ).astype(np.float64)
    incumbent_test = np.load(
        os.path.join(shared_dir, "incumbent_test_scores.npy")
    ).astype(np.float64)

    x_train = feature_matrix(train)
    x_valid = feature_matrix(valid)

    aux_names = available_auxiliary_names(train)
    labels_train = task_labels(train, aux_names)

    mmoe = fit_mmoe(x_train, labels_train, SEED)
    mmoe_valid = predict_mmoe(mmoe, x_valid)
    del mmoe
    gc.collect()

    svd_model = fit_implicit_svd(
        train.user_id,
        train.video_id,
        y_train,
        SEED,
    )
    svd_valid = predict_implicit_svd(
        svd_model,
        valid.user_id,
        valid.video_id,
    )
    del svd_model
    gc.collect()

    family_predictions = {
        "multitask_mmoe": mmoe_valid,
        "implicit_svd": svd_valid,
    }

    candidates, descriptor, valid_scores = choose_candidates(
        valid,
        y_valid,
        incumbent_valid,
        family_predictions,
    )
    metrics = evaluate(valid.user_id, y_valid, valid_scores)

    print(
        "FINDINGS "
        + json.dumps(
            {
                "auxiliary_tasks": aux_names,
                "selected_family": descriptor[0],
                "selected_mode": descriptor[1],
                "selected_weight": float(descriptor[2]),
            },
            sort_keys=True,
        )
    )
    print("CANDIDATES " + json.dumps(candidates, sort_keys=True))

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(valid_scores, dtype=np.float64),
        )

    test = load("test")
    selected_family = descriptor[0]
    family_test_prediction = None

    if selected_family == "multitask_mmoe":
        x_combined = concatenate_features(train, valid)
        x_test = feature_matrix(test)

        labels_valid = task_labels(valid, aux_names)
        labels_combined = np.ascontiguousarray(
            np.concatenate([labels_train, labels_valid], axis=0),
            dtype=np.float32,
        )
        final_model = fit_mmoe(
            x_combined,
            labels_combined,
            SEED,
        )
        family_test_prediction = predict_mmoe(final_model, x_test)
        del final_model, x_combined, x_test, labels_combined
        gc.collect()

    elif selected_family == "implicit_svd":
        combined_users = np.concatenate(
            [np.asarray(train.user_id), np.asarray(valid.user_id)]
        )
        combined_videos = np.concatenate(
            [np.asarray(train.video_id), np.asarray(valid.video_id)]
        )
        combined_labels = np.concatenate(
            [y_train, y_valid.astype(np.float32)]
        )
        final_svd = fit_implicit_svd(
            combined_users,
            combined_videos,
            combined_labels,
            SEED,
        )
        family_test_prediction = predict_implicit_svd(
            final_svd,
            test.user_id,
            test.video_id,
        )
        del final_svd
        gc.collect()

    test_scores = form_test_scores(
        descriptor,
        family_test_prediction,
        incumbent_test,
        test.user_id,
    )

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
            }
        )
    )


if __name__ == "__main__":
    main()