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
SEED = 7319
HISTORY_LEN = 6
EMBED_DIM = 12
BATCH_SIZE = 16384
EPOCHS = 3
LR = 0.0015
HALF_LIFE_DAYS = 4.0

CURRENT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "hour",
]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def current_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[f] for f in CURRENT_FIELDS]),
        dtype=np.int64,
    )


def causal_history(split, labels):
    """Previous impressions in (user, time, row-position) order."""
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    n = len(users)

    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, times, users))
    sorted_users = users[order]
    sorted_videos = videos[order]
    sorted_labels = labels[order]

    hist_video_sorted = np.zeros((n, HISTORY_LEN), dtype=np.int64)
    hist_label_sorted = np.full((n, HISTORY_LEN), -1, dtype=np.int8)

    positions = np.arange(n, dtype=np.int64)
    for lag in range(1, HISTORY_LEN + 1):
        dst = positions[lag:]
        src = positions[:-lag]
        good = sorted_users[dst] == sorted_users[src]
        good_dst = dst[good]
        good_src = src[good]
        hist_video_sorted[good_dst, lag - 1] = sorted_videos[good_src]
        hist_label_sorted[good_dst, lag - 1] = sorted_labels[good_src]

    hist_video = np.empty_like(hist_video_sorted)
    hist_label = np.empty_like(hist_label_sorted)
    hist_video[order] = hist_video_sorted
    hist_label[order] = hist_label_sorted
    return hist_video, hist_label


def static_history(reference, reference_labels, target):
    """Last labeled reference impressions for each target user."""
    ref_users = np.asarray(reference.user_id, dtype=np.int64)
    ref_times = np.asarray(reference.time_ms, dtype=np.int64)
    ref_videos = np.asarray(reference.X["video_id"], dtype=np.int64)
    ref_labels = np.asarray(reference_labels, dtype=np.int8)

    rows = np.arange(len(ref_users), dtype=np.int64)
    order = np.lexsort((rows, ref_times, ref_users))
    su = ref_users[order]
    sv = ref_videos[order]
    sy = ref_labels[order]

    max_user = int(
        max(
            int(ref_users.max(initial=0)),
            int(np.asarray(target.user_id).max(initial=0)),
        )
    )
    starts = np.full(max_user + 1, -1, dtype=np.int64)
    ends = np.full(max_user + 1, -1, dtype=np.int64)

    group_starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    group_ends = np.r_[group_starts[1:], len(su)]
    group_users = su[group_starts]
    starts[group_users] = group_starts
    ends[group_users] = group_ends

    tu = np.asarray(target.user_id, dtype=np.int64)
    target_ends = ends[tu]
    target_starts = starts[tu]
    n = len(tu)

    hv = np.zeros((n, HISTORY_LEN), dtype=np.int64)
    hy = np.full((n, HISTORY_LEN), -1, dtype=np.int8)

    for lag in range(1, HISTORY_LEN + 1):
        idx = target_ends - lag
        valid = (target_ends >= 0) & (idx >= target_starts)
        hv[valid, lag - 1] = sv[idx[valid]]
        hy[valid, lag - 1] = sy[idx[valid]]
    return hv, hy


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    weights = np.exp2(-age.astype(np.float32) / HALF_LIFE_DAYS)
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


class SequenceCTR(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

        cards = [FEATURE_CARDINALITIES[f] for f in CURRENT_FIELDS]
        offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
        total = int(sum(cards))
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )

        self.linear = nn.Embedding(total, 1)
        nn.init.zeros_(self.linear.weight)
        self.bias = nn.Parameter(torch.zeros(1))

        self.current_embedding = nn.Embedding(total, EMBED_DIM)
        nn.init.normal_(self.current_embedding.weight, std=0.02)

        video_card = FEATURE_CARDINALITIES["video_id"]
        self.history_video = nn.Embedding(video_card, EMBED_DIM)
        self.outcome_embedding = nn.Embedding(2, EMBED_DIM)
        nn.init.normal_(self.history_video.weight, std=0.02)
        nn.init.normal_(self.outcome_embedding.weight, std=0.02)

        current_dim = len(CURRENT_FIELDS) * EMBED_DIM

        if kind == "din":
            self.attention = nn.Sequential(
                nn.Linear(4 * EMBED_DIM, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            sequence_dim = EMBED_DIM
        elif kind == "gru":
            self.gru = nn.GRU(
                input_size=EMBED_DIM,
                hidden_size=EMBED_DIM,
                batch_first=True,
            )
            sequence_dim = EMBED_DIM
        else:
            sequence_dim = EMBED_DIM

        self.head = nn.Sequential(
            nn.Linear(current_dim + sequence_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def encode_history(self, target, hist_video, hist_label):
        mask = hist_label >= 0
        safe_label = hist_label.clamp(min=0).long()
        h = (
            self.history_video(hist_video)
            + self.outcome_embedding(safe_label)
        )
        mask_float = mask.unsqueeze(-1).float()

        if self.kind == "last_event":
            return h[:, 0, :] * mask_float[:, 0, :]

        if self.kind == "mean_pool":
            numerator = (h * mask_float).sum(dim=1)
            denominator = mask_float.sum(dim=1).clamp(min=1.0)
            return numerator / denominator

        if self.kind == "din":
            expanded_target = target.unsqueeze(1).expand_as(h)
            attention_input = torch.cat(
                [
                    h,
                    expanded_target,
                    h * expanded_target,
                    h - expanded_target,
                ],
                dim=-1,
            )
            logits = self.attention(attention_input).squeeze(-1)
            logits = logits.masked_fill(~mask, -1e4)
            weights = torch.softmax(logits, dim=1) * mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(
                min=1e-6
            )
            return (h * weights.unsqueeze(-1)).sum(dim=1)

        # Histories are stored newest first; GRU consumes oldest first.
        h = torch.flip(h, dims=[1])
        reversed_mask = torch.flip(mask_float, dims=[1])
        h = h * reversed_mask
        output, _ = self.gru(h)
        lengths = mask.sum(dim=1).clamp(min=1)
        last_index = (lengths - 1).view(-1, 1, 1).expand(
            -1, 1, EMBED_DIM
        )
        encoded = output.gather(1, last_index).squeeze(1)
        return encoded * mask.any(dim=1, keepdim=True).float()

    def forward(self, x, hist_video, hist_label):
        z = x + self.offsets
        wide = self.linear(z).sum(dim=1).squeeze(-1) + self.bias
        current = self.current_embedding(z)
        target = current[:, 1, :]
        sequence = self.encode_history(target, hist_video, hist_label)
        deep_input = torch.cat([current.flatten(1), sequence], dim=1)
        return wide + self.head(deep_input).squeeze(-1)


def fit_model(x_np, hv_np, hy_np, y_np, dates, kind, seed):
    seed_all(seed)
    x = torch.from_numpy(x_np)
    hv = torch.from_numpy(hv_np)
    hy = torch.from_numpy(hy_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    weights = torch.from_numpy(recency_weights(dates))

    model = SequenceCTR(kind)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-6
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    generator = torch.Generator()
    generator.manual_seed(seed)
    n = len(y)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x[idx], hv[idx], hy[idx])
            loss = (criterion(logits, y[idx]) * weights[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict(model, x_np, hv_np, hy_np):
    x = torch.from_numpy(x_np)
    hv = torch.from_numpy(hv_np)
    hy = torch.from_numpy(hy_np)
    result = np.empty(len(x_np), dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_np), BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, len(x_np))
            result[start:end] = model(
                x[start:end], hv[start:end], hy[start:end]
            ).cpu().numpy()
    return result


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    positions = np.arange(n) - np.repeat(starts, sizes)
    denominators = np.maximum(np.repeat(sizes, sizes) - 1, 1)

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = positions / denominators
    return ranked


class Combined:
    pass


def combine_splits(train, valid):
    combined = Combined()
    combined.X = {
        f: np.concatenate([train.X[f], valid.X[f]])
        for f in CURRENT_FIELDS
    }
    combined.user_id = np.concatenate([train.user_id, valid.user_id])
    combined.time_ms = np.concatenate([train.time_ms, valid.time_ms])
    combined.date = np.concatenate([train.date, valid.date])
    return combined


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_all(SEED)

    train = load("train")
    valid = load("valid")
    y_train = np.asarray(train.y, dtype=np.int8)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    x_train = current_matrix(train)
    x_valid = current_matrix(valid)
    train_hv, train_hy = causal_history(train, y_train)
    valid_hv, valid_hy = static_history(train, y_train, valid)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid = np.load(
        os.path.join(shared, "incumbent_valid_scores.npy")
    ).astype(np.float64)
    incumbent_z = zscore(incumbent_valid)
    incumbent_rank = within_user_rank(valid.user_id, incumbent_valid)

    kinds = ["last_event", "mean_pool", "din", "gru"]
    predictions = {}
    candidates = {
        "trusted_incumbent": float(
            evaluate(valid.user_id, y_valid, incumbent_valid)["primary"]
        )
    }

    best_primary = candidates["trusted_incumbent"]
    best_descriptor = ("incumbent", "standalone", 0.0)
    best_valid = incumbent_valid.copy()

    blend_weights = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]

    for model_index, kind in enumerate(kinds):
        model = fit_model(
            x_train,
            train_hv,
            train_hy,
            y_train,
            train.date,
            kind,
            SEED + model_index,
        )
        pred = predict(model, x_valid, valid_hv, valid_hy)
        predictions[kind] = pred
        del model
        gc.collect()

        standalone = float(
            evaluate(valid.user_id, y_valid, pred)["primary"]
        )
        candidates[kind] = standalone
        if standalone > best_primary:
            best_primary = standalone
            best_descriptor = (kind, "standalone", 1.0)
            best_valid = pred.astype(np.float64)

        pred_z = zscore(pred)
        pred_rank = within_user_rank(valid.user_id, pred)

        local_raw_score = -np.inf
        local_rank_score = -np.inf
        local_raw_weight = 0.0
        local_rank_weight = 0.0

        for weight in blend_weights:
            raw_blend = weight * pred_z + (1.0 - weight) * incumbent_z
            raw_score = float(
                evaluate(valid.user_id, y_valid, raw_blend)["primary"]
            )
            if raw_score > local_raw_score:
                local_raw_score = raw_score
                local_raw_weight = weight
            if raw_score > best_primary:
                best_primary = raw_score
                best_descriptor = (kind, "raw_blend", weight)
                best_valid = raw_blend.copy()

            rank_blend = (
                weight * pred_rank + (1.0 - weight) * incumbent_rank
            )
            rank_score = float(
                evaluate(valid.user_id, y_valid, rank_blend)["primary"]
            )
            if rank_score > local_rank_score:
                local_rank_score = rank_score
                local_rank_weight = weight
            if rank_score > best_primary:
                best_primary = rank_score
                best_descriptor = (kind, "rank_blend", weight)
                best_valid = rank_blend.copy()

        candidates[kind + "_raw_blend"] = local_raw_score
        candidates[kind + "_rank_blend"] = local_rank_score
        candidates[kind + "_raw_weight"] = local_raw_weight
        candidates[kind + "_rank_weight"] = local_rank_weight

    metrics = evaluate(valid.user_id, y_valid, best_valid)

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(best_valid, dtype=np.float64),
        )

    winner_kind, blend_mode, winner_weight = best_descriptor
    test = load("test")
    incumbent_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)

    if winner_kind == "incumbent":
        test_scores = incumbent_test
    else:
        combined = combine_splits(train, valid)
        y_combined = np.concatenate([y_train, y_valid])
        x_combined = np.concatenate([x_train, x_valid], axis=0)

        combined_hv, combined_hy = causal_history(combined, y_combined)
        test_hv, test_hy = static_history(combined, y_combined, test)

        refit = fit_model(
            x_combined,
            combined_hv,
            combined_hy,
            y_combined,
            combined.date,
            winner_kind,
            SEED + kinds.index(winner_kind),
        )
        test_new = predict(
            refit, current_matrix(test), test_hv, test_hy
        )

        if blend_mode == "standalone":
            test_scores = test_new.astype(np.float64)
        elif blend_mode == "raw_blend":
            test_scores = (
                winner_weight * zscore(test_new)
                + (1.0 - winner_weight) * zscore(incumbent_test)
            )
        else:
            test_scores = (
                winner_weight
                * within_user_rank(test.user_id, test_new)
                + (1.0 - winner_weight)
                * within_user_rank(test.user_id, incumbent_test)
            )

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    valid_history_coverage = float((valid_hy >= 0).any(axis=1).mean())
    mean_valid_history = float((valid_hy >= 0).sum(axis=1).mean())
    print(
        "FINDINGS "
        + json.dumps(
            {
                "winner": winner_kind,
                "blend_mode": blend_mode,
                "new_model_weight": float(winner_weight),
                "valid_history_coverage": valid_history_coverage,
                "mean_valid_history_length": mean_valid_history,
            },
            separators=(",", ":"),
        )
    )
    print(
        "CANDIDATES "
        + json.dumps(candidates, sort_keys=True, separators=(",", ":"))
    )
    print(
        "METRICS "
        + json.dumps(
            {
                "primary": float(metrics["primary"]),
                "gauc": float(metrics["gauc"]),
                "ndcg@5": float(metrics["ndcg@5"]),
                "gpu_seconds": float(time.time() - START),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()