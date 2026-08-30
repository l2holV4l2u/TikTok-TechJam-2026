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


START_TIME = time.time()
SEED = 2025
BATCH_SIZE = 16384
EPOCHS = 5
EMBED_DIM = 12
LR = 0.001

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
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[name] for name in FIELDS]),
        dtype=np.int64,
    )


def recency_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    age = np.max(dates).astype(np.int64) - np.asarray(dates, dtype=np.int64)
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    w /= max(float(w.mean()), 1e-8)
    return w.astype(np.float32)


class CTRModel(nn.Module):
    def __init__(self, cardinalities, kind, embedding_dim=12):
        super().__init__()
        self.kind = kind
        total = int(sum(cardinalities))
        offsets = np.cumsum([0] + list(cardinalities[:-1]), dtype=np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total, 1)
        nn.init.zeros_(self.linear.weight)

        if kind != "wide":
            self.embedding = nn.Embedding(total, embedding_dim)
            nn.init.normal_(self.embedding.weight, std=0.01)

        dim = len(cardinalities) * embedding_dim
        if kind == "deepfm":
            self.deep = nn.Sequential(
                nn.Linear(dim, 96),
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, 1),
            )
        elif kind == "dcn":
            self.cross_w = nn.ParameterList(
                [nn.Parameter(torch.empty(dim)) for _ in range(2)]
            )
            self.cross_b = nn.ParameterList(
                [nn.Parameter(torch.zeros(dim)) for _ in range(2)]
            )
            for w in self.cross_w:
                nn.init.normal_(w, std=0.01)
            self.dcn_out = nn.Linear(dim, 1)

    def forward(self, x):
        z = x + self.offsets
        wide = self.linear(z).sum(dim=1).squeeze(-1) + self.bias

        if self.kind == "wide":
            return wide

        emb = self.embedding(z)
        if self.kind in ("fm", "deepfm"):
            summed = emb.sum(dim=1)
            fm = 0.5 * (
                summed.square() - emb.square().sum(dim=1)
            ).sum(dim=1)
            if self.kind == "fm":
                return wide + fm
            deep = self.deep(emb.flatten(1)).squeeze(-1)
            return wide + fm + deep

        x0 = emb.flatten(1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = (xl * w).sum(dim=1, keepdim=True)
            xl = x0 * scalar + b + xl
        return wide + self.dcn_out(xl).squeeze(-1)


def fit_neural(x_np, y_np, dates, kind, half_life, seed):
    seed_everything(seed)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    weights = torch.from_numpy(recency_weights(dates, half_life))

    cards = [FEATURE_CARDINALITIES[f] for f in FIELDS]
    model = CTRModel(cards, kind, EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    generator = torch.Generator()
    generator.manual_seed(seed)
    n = len(y)

    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            logits = model(x[idx])
            loss = (loss_fn(logits, y[idx]) * weights[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def predict_neural(model, x_np):
    x = torch.from_numpy(x_np)
    out = np.empty(len(x_np), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_np), BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, len(x_np))
            out[start:end] = model(x[start:end]).cpu().numpy()
    return out


def logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def fit_empirical(split, y):
    y = np.asarray(y, dtype=np.float64)
    prior = float(y.mean())
    model = {"prior": prior, "tables": {}}
    specifications = [
        ("video_id", 24.0, 0.50),
        ("author_id", 35.0, 0.30),
        ("tag", 100.0, 0.12),
        ("duration_bucket", 180.0, 0.08),
    ]
    for field, smoothing, coefficient in specifications:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = FEATURE_CARDINALITIES[field]
        count = np.bincount(ids, minlength=card).astype(np.float64)
        positives = np.bincount(ids, weights=y, minlength=card)
        rate = (positives + smoothing * prior) / (count + smoothing)
        model["tables"][field] = (logit(rate), coefficient)
    return model


def predict_empirical(model, split):
    score = np.zeros(len(split.user_id), dtype=np.float64)
    for field, (table, coefficient) in model["tables"].items():
        score += coefficient * table[np.asarray(split.X[field], dtype=np.int64)]
    return score.astype(np.float32)


def merge_splits(a, b):
    class Combined:
        pass
    c = Combined()
    c.X = {
        field: np.concatenate([a.X[field], b.X[field]])
        for field in FIELDS
    }
    c.user_id = np.concatenate([a.user_id, b.user_id])
    return c


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(float(x.std()), 1e-8)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    positions = np.arange(n) - np.repeat(starts, sizes)
    denominators = np.maximum(np.repeat(sizes, sizes) - 1, 1)
    ranked_sorted = positions / denominators

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_everything(SEED)

    train = load("train")
    valid = load("valid")
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)
    x_train = make_matrix(train)
    x_valid = make_matrix(valid)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid = np.load(
        os.path.join(shared, "incumbent_valid_scores.npy")
    ).astype(np.float64)
    incumbent_rank_valid = within_user_rank(valid.user_id, incumbent_valid)
    incumbent_z_valid = zscore(incumbent_valid)

    configurations = {
        "wide_uniform": ("wide", None),
        "fm_uniform": ("fm", None),
        "fm_recency_h4": ("fm", 4.0),
        "fm_recency_h8": ("fm", 8.0),
        "deepfm_uniform": ("deepfm", None),
        "dcn_uniform": ("dcn", None),
    }

    predictions = {}
    for i, (name, (kind, half_life)) in enumerate(configurations.items()):
        model = fit_neural(
            x_train, y_train, train.date, kind, half_life, SEED + i
        )
        predictions[name] = predict_neural(model, x_valid)
        del model
        gc.collect()

    empirical_model = fit_empirical(train, y_train)
    predictions["empirical_bayes"] = predict_empirical(empirical_model, valid)

    candidate_scores = {}
    best_score = -np.inf
    best_descriptor = None
    best_valid_scores = None

    incumbent_metrics = evaluate(
        valid.user_id, y_valid, incumbent_valid
    )
    candidate_scores["trusted_incumbent"] = float(
        incumbent_metrics["primary"]
    )

    blend_alphas = [0.25, 0.40, 0.55, 0.70, 0.85]

    for name, pred in predictions.items():
        standalone_metrics = evaluate(valid.user_id, y_valid, pred)
        standalone_primary = float(standalone_metrics["primary"])
        candidate_scores[name] = standalone_primary

        if standalone_primary > best_score:
            best_score = standalone_primary
            best_descriptor = (name, "standalone", 1.0)
            best_valid_scores = np.asarray(pred, dtype=np.float64)

        pred_z = zscore(pred)
        pred_rank = within_user_rank(valid.user_id, pred)

        local_raw_score = -np.inf
        local_raw_alpha = None
        local_raw_pred = None
        local_rank_score = -np.inf
        local_rank_alpha = None
        local_rank_pred = None

        for alpha in blend_alphas:
            raw_blend = alpha * pred_z + (1.0 - alpha) * incumbent_z_valid
            raw_primary = float(
                evaluate(valid.user_id, y_valid, raw_blend)["primary"]
            )
            if raw_primary > local_raw_score:
                local_raw_score = raw_primary
                local_raw_alpha = alpha
                local_raw_pred = raw_blend

            rank_blend = (
                alpha * pred_rank
                + (1.0 - alpha) * incumbent_rank_valid
            )
            rank_primary = float(
                evaluate(valid.user_id, y_valid, rank_blend)["primary"]
            )
            if rank_primary > local_rank_score:
                local_rank_score = rank_primary
                local_rank_alpha = alpha
                local_rank_pred = rank_blend

        candidate_scores[name + "_rawblend"] = local_raw_score
        candidate_scores[name + "_rankblend"] = local_rank_score

        if local_raw_score > best_score:
            best_score = local_raw_score
            best_descriptor = (name, "rawblend", local_raw_alpha)
            best_valid_scores = np.asarray(local_raw_pred, dtype=np.float64)

        if local_rank_score > best_score:
            best_score = local_rank_score
            best_descriptor = (name, "rankblend", local_rank_alpha)
            best_valid_scores = np.asarray(local_rank_pred, dtype=np.float64)

    final_metrics = evaluate(valid.user_id, y_valid, best_valid_scores)

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(best_valid_scores, dtype=np.float64),
        )

    selected_name, blend_mode, selected_alpha = best_descriptor

    test = load("test")
    incumbent_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)

    if selected_name == "empirical_bayes":
        combined = merge_splits(train, valid)
        y_combined = np.concatenate(
            [y_train, y_valid.astype(np.float32)]
        )
        selected_model = fit_empirical(combined, y_combined)
        new_test_scores = predict_empirical(selected_model, test)
    else:
        kind, half_life = configurations[selected_name]
        x_refit = np.concatenate([x_train, x_valid], axis=0)
        y_refit = np.concatenate(
            [y_train, y_valid.astype(np.float32)]
        )
        date_refit = np.concatenate([train.date, valid.date])
        selected_seed = SEED + list(configurations).index(selected_name)
        selected_model = fit_neural(
            x_refit,
            y_refit,
            date_refit,
            kind,
            half_life,
            selected_seed,
        )
        x_test = make_matrix(test)
        new_test_scores = predict_neural(selected_model, x_test)

    if blend_mode == "standalone":
        test_scores = np.asarray(new_test_scores, dtype=np.float64)
    elif blend_mode == "rawblend":
        test_scores = (
            selected_alpha * zscore(new_test_scores)
            + (1.0 - selected_alpha) * zscore(incumbent_test)
        )
    else:
        test_scores = (
            selected_alpha
            * within_user_rank(test.user_id, new_test_scores)
            + (1.0 - selected_alpha)
            * within_user_rank(test.user_id, incumbent_test)
        )

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    print(
        "FINDINGS "
        + json.dumps(
            {
                "winner": selected_name,
                "blend_mode": blend_mode,
                "new_model_weight": float(selected_alpha),
            },
            separators=(",", ":"),
        )
    )
    print(
        "CANDIDATES "
        + json.dumps(candidate_scores, separators=(",", ":"), sort_keys=True)
    )

    elapsed = time.time() - START_TIME
    payload = {
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }
    print("METRICS " + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()