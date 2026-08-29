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


SEED = 2026

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "hour",
    "tag",
    "upload_type",
    "music_type",
    "video_type",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

# Explicit memorization paths. Each cross gets its own hashed namespace.
CROSS_PAIRS = [
    ("user_id", "video_id"),
    ("user_id", "author_id"),
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("user_id", "tab"),
    ("video_id", "tab"),
    ("video_id", "hour"),
    ("author_id", "tab"),
    ("tag", "duration_bucket"),
    ("user_active_degree", "tag"),
]

HASH_BUCKETS = 1 << 18
EMBED_DIM = 16
BATCH_SIZE = 8192
EPOCHS = 5


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_categorical(split):
    return np.ascontiguousarray(
        np.column_stack(
            [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS]
        ),
        dtype=np.int64,
    )


def build_crosses(split):
    columns = []
    for namespace, (left, right) in enumerate(CROSS_PAIRS):
        a = np.asarray(split.X[left], dtype=np.int64)
        b = np.asarray(split.X[right], dtype=np.int64)

        # Values and constants are small enough to remain safely in int64.
        hashed = (
            a * np.int64(1000003)
            + b * np.int64(9176)
            + np.int64((namespace + 1) * 104729)
        ) % np.int64(HASH_BUCKETS)
        hashed = hashed + np.int64(namespace * HASH_BUCKETS)
        columns.append(hashed)

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.where(starts, np.arange(n, dtype=np.int64), 0)
    start_positions = np.maximum.accumulate(start_positions)
    positions = np.arange(n, dtype=np.int64) - start_positions

    boundaries = np.flatnonzero(starts)
    ends = np.r_[boundaries[1:], n]
    lengths = ends - boundaries
    sizes = np.repeat(lengths, lengths)

    ranked_sorted = positions.astype(np.float64) / np.maximum(
        sizes - 1, 1
    ).astype(np.float64)
    ranked_sorted[sizes == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def standardization(scores):
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return mean, std


class WideDeepFM(nn.Module):
    def __init__(self, cardinalities, offsets):
        super().__init__()
        self.register_buffer(
            "offsets",
            torch.tensor(offsets, dtype=torch.long),
        )

        total_categories = int(sum(cardinalities))
        n_fields = len(cardinalities)
        n_cross_values = len(CROSS_PAIRS) * HASH_BUCKETS

        self.linear_embedding = nn.Embedding(total_categories, 1)
        self.feature_embedding = nn.Embedding(total_categories, EMBED_DIM)
        self.cross_embedding = nn.Embedding(n_cross_values, 1)

        deep_input = n_fields * EMBED_DIM
        self.deep = nn.Sequential(
            nn.Linear(deep_input, 192),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear_embedding.weight)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.cross_embedding.weight)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Start the nonlinear residual conservatively.
        nn.init.normal_(self.deep[-1].weight, mean=0.0, std=0.01)

    def components(self, x, crosses):
        global_ids = x + self.offsets.unsqueeze(0)

        linear = self.linear_embedding(global_ids).sum(dim=1).squeeze(-1)

        embeddings = self.feature_embedding(global_ids)
        summed = embeddings.sum(dim=1)
        fm = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        ).sum(dim=1)

        deep = self.deep(embeddings.flatten(start_dim=1)).squeeze(-1)
        cross = self.cross_embedding(crosses).sum(dim=1).squeeze(-1)

        generalized = self.bias + linear + fm + deep
        return generalized, cross

    def forward(self, x, crosses):
        generalized, cross = self.components(x, crosses)
        return generalized + cross


def predict_components(model, x_np, cross_np, device):
    model.eval()
    n = x_np.shape[0]
    generalized = np.empty(n, dtype=np.float64)
    cross = np.empty(n, dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            xb = torch.from_numpy(x_np[start:end]).to(device=device)
            cb = torch.from_numpy(cross_np[start:end]).to(device=device)
            gb, wb = model.components(xb, cb)
            generalized[start:end] = gb.cpu().numpy().astype(np.float64)
            cross[start:end] = wb.cpu().numpy().astype(np.float64)

    return generalized, cross


def main():
    wall_start = time.perf_counter()

    requested_device = os.environ["AGENT_DEVICE"].lower()
    if requested_device != "cpu":
        raise RuntimeError("This run is configured for deterministic CPU use")
    device = torch.device("cpu")

    threads = max(
        1,
        int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)),
    )
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    seed_everything(SEED)

    artifacts = os.environ.get("RUN_ARTIFACTS", "")
    incumbent_valid_path = os.path.join(
        artifacts, "incumbent_valid_scores.npy"
    )
    incumbent_test_path = os.path.join(
        artifacts, "incumbent_test_scores.npy"
    )
    if not (
        os.path.isfile(incumbent_valid_path)
        and os.path.isfile(incumbent_test_path)
    ):
        raise RuntimeError("Trusted incumbent predictions are unavailable")

    train = load("train")
    valid = load("valid")

    x_train = build_categorical(train)
    c_train = build_crosses(train)
    y_train = np.asarray(train.y, dtype=np.float32)

    x_valid = build_categorical(valid)
    c_valid = build_crosses(valid)

    cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
    offsets = np.cumsum([0] + cardinalities[:-1]).astype(np.int64)

    model = WideDeepFM(cardinalities, offsets).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-6,
    )

    rng = np.random.default_rng(SEED)
    n_train = len(y_train)
    epoch_losses = []

    model.train()
    for epoch in range(EPOCHS):
        permutation = rng.permutation(n_train)
        loss_sum = 0.0
        seen = 0

        for start in range(0, n_train, BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]

            xb = torch.from_numpy(x_train[indices]).to(device=device)
            cb = torch.from_numpy(c_train[indices]).to(device=device)
            yb = torch.from_numpy(y_train[indices]).to(device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, cb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_n = len(indices)
            loss_sum += float(loss.detach()) * batch_n
            seen += batch_n

        epoch_losses.append(loss_sum / max(seen, 1))

    valid_generalized, valid_cross = predict_components(
        model, x_valid, c_valid, device
    )

    incumbent_valid = np.asarray(
        np.load(incumbent_valid_path),
        dtype=np.float64,
    )
    if incumbent_valid.shape != valid_generalized.shape:
        raise RuntimeError("Incumbent validation prediction shape mismatch")

    incumbent_metrics = evaluate(
        valid.user_id, valid.y, incumbent_valid
    )

    candidate_results = {
        "incumbent": float(incumbent_metrics["primary"]),
    }

    selected_primary = float(incumbent_metrics["primary"])
    selected_wide_scale = 0.0
    selected_blend_method = "incumbent"
    selected_model_weight = 0.0
    valid_scores = incumbent_valid.copy()

    incumbent_mean, incumbent_std = standardization(incumbent_valid)
    incumbent_z = (incumbent_valid - incumbent_mean) / incumbent_std
    incumbent_rank = within_user_rank(valid.user_id, incumbent_valid)

    wide_scales = [0.0, 0.5, 1.0, 1.5]
    blend_weights = np.linspace(0.0, 1.0, 11)

    for wide_scale in wide_scales:
        raw_model_scores = (
            valid_generalized + wide_scale * valid_cross
        )
        raw_metrics = evaluate(
            valid.user_id, valid.y, raw_model_scores
        )
        raw_name = "wide_scale_%.1f_raw" % wide_scale
        candidate_results[raw_name] = float(raw_metrics["primary"])

        model_mean, model_std = standardization(raw_model_scores)
        model_z = (raw_model_scores - model_mean) / model_std
        model_rank = within_user_rank(valid.user_id, raw_model_scores)

        for model_weight_value in blend_weights:
            model_weight = float(model_weight_value)

            if model_weight == 0.0:
                z_scores = incumbent_valid
                rank_scores = incumbent_valid
            elif model_weight == 1.0:
                z_scores = raw_model_scores
                rank_scores = raw_model_scores
            else:
                z_scores = (
                    (1.0 - model_weight) * incumbent_z
                    + model_weight * model_z
                )
                rank_scores = (
                    (1.0 - model_weight) * incumbent_rank
                    + model_weight * model_rank
                )

            z_metrics = evaluate(valid.user_id, valid.y, z_scores)
            z_name = "ws%.1f_zblend_%.1f" % (
                wide_scale,
                model_weight,
            )
            candidate_results[z_name] = float(z_metrics["primary"])

            if float(z_metrics["primary"]) > selected_primary:
                selected_primary = float(z_metrics["primary"])
                selected_wide_scale = float(wide_scale)
                selected_blend_method = "zblend"
                selected_model_weight = model_weight
                valid_scores = np.asarray(
                    z_scores, dtype=np.float64
                ).copy()

            rank_metrics = evaluate(valid.user_id, valid.y, rank_scores)
            rank_name = "ws%.1f_rankblend_%.1f" % (
                wide_scale,
                model_weight,
            )
            candidate_results[rank_name] = float(
                rank_metrics["primary"]
            )

            if float(rank_metrics["primary"]) > selected_primary:
                selected_primary = float(rank_metrics["primary"])
                selected_wide_scale = float(wide_scale)
                selected_blend_method = "rankblend"
                selected_model_weight = model_weight
                valid_scores = np.asarray(
                    rank_scores, dtype=np.float64
                ).copy()

    metrics = evaluate(valid.user_id, valid.y, valid_scores)

    print(
        "FINDINGS epochs=%d final_train_loss=%.6f "
        "incumbent_primary=%.6f selected_wide_scale=%.1f "
        "selected_method=%s selected_model_weight=%.1f "
        "selected_primary=%.6f"
        % (
            EPOCHS,
            epoch_losses[-1],
            float(incumbent_metrics["primary"]),
            selected_wide_scale,
            selected_blend_method,
            selected_model_weight,
            float(metrics["primary"]),
        ),
        flush=True,
    )
    print(
        "CANDIDATES " + json.dumps(candidate_results, sort_keys=True),
        flush=True,
    )

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(valid_scores, dtype=np.float64),
        )

    # Retain validation normalization parameters for applying the identical
    # selected z-score transformation to test predictions.
    selected_valid_raw = (
        valid_generalized + selected_wide_scale * valid_cross
    )
    selected_model_mean, selected_model_std = standardization(
        selected_valid_raw
    )

    del x_train, c_train, y_train
    del x_valid, c_valid
    del train
    gc.collect()

    test = load("test")
    x_test = build_categorical(test)
    c_test = build_crosses(test)

    test_generalized, test_cross = predict_components(
        model, x_test, c_test, device
    )
    raw_test_scores = (
        test_generalized + selected_wide_scale * test_cross
    )

    incumbent_test = np.asarray(
        np.load(incumbent_test_path),
        dtype=np.float64,
    )
    if incumbent_test.shape != raw_test_scores.shape:
        raise RuntimeError("Incumbent test prediction shape mismatch")

    if (
        selected_blend_method == "incumbent"
        or selected_model_weight == 0.0
    ):
        test_scores = incumbent_test
    elif selected_model_weight == 1.0:
        test_scores = raw_test_scores
    elif selected_blend_method == "zblend":
        incumbent_test_z = (
            incumbent_test - incumbent_mean
        ) / incumbent_std
        model_test_z = (
            raw_test_scores - selected_model_mean
        ) / selected_model_std
        test_scores = (
            (1.0 - selected_model_weight) * incumbent_test_z
            + selected_model_weight * model_test_z
        )
    elif selected_blend_method == "rankblend":
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        model_test_rank = within_user_rank(
            test.user_id, raw_test_scores
        )
        test_scores = (
            (1.0 - selected_model_weight) * incumbent_test_rank
            + selected_model_weight * model_test_rank
        )
    else:
        raise RuntimeError("Unknown selected blend method")

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    elapsed = time.perf_counter() - wall_start
    final_payload = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
        "device": "cpu",
    }
    print("METRICS " + json.dumps(final_payload), flush=True)


if __name__ == "__main__":
    main()