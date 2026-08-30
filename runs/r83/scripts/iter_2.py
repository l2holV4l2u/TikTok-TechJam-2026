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


START_TIME = time.time()
SEED = 2024
BATCH_SIZE = 8192

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

FAMILY_CONFIGS = {
    "additive": {
        "family": "additive",
        "embedding_dim": 0,
        "lr": 0.004,
        "epochs": 4,
        "half_life": None,
    },
    "fm": {
        "family": "fm",
        "embedding_dim": 16,
        "lr": 0.001,
        "epochs": 5,
        "half_life": None,
    },
    "nfm": {
        "family": "nfm",
        "embedding_dim": 16,
        "lr": 0.001,
        "epochs": 5,
        "half_life": None,
    },
    "fm_recent8d": {
        "family": "fm",
        "embedding_dim": 16,
        "lr": 0.001,
        "epochs": 5,
        "half_life": 8.0,
    },
}

BLEND_WEIGHTS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[name] for name in FIELDS]),
        dtype=np.int64,
    )


def make_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    dates = np.asarray(dates, dtype=np.int64)
    newest = int(dates.max())
    # YYYYMMDD values in this dataset stay within one month for each fit.
    age_days = newest - dates
    weights = np.exp2(-age_days.astype(np.float32) / float(half_life))
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


class OffsetModule(nn.Module):
    def __init__(self, cardinalities):
        super().__init__()
        total = int(sum(cardinalities))
        offsets = np.cumsum([0] + list(cardinalities[:-1]), dtype=np.int64)
        self.total_cardinality = total
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )

    def shifted(self, x):
        return x + self.offsets


class AdditiveModel(OffsetModule):
    def __init__(self, cardinalities):
        super().__init__(cardinalities)
        self.linear = nn.Embedding(self.total_cardinality, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        ids = self.shifted(x)
        return self.bias + self.linear(ids).sum(dim=1).squeeze(-1)


class FMModel(OffsetModule):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__(cardinalities)
        self.linear = nn.Embedding(self.total_cardinality, 1)
        self.embedding = nn.Embedding(
            self.total_cardinality, embedding_dim
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        ids = self.shifted(x)
        linear = self.linear(ids).sum(dim=1).squeeze(-1)
        latent = self.embedding(ids)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class NFMModel(OffsetModule):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__(cardinalities)
        self.linear = nn.Embedding(self.total_cardinality, 1)
        self.embedding = nn.Embedding(
            self.total_cardinality, embedding_dim
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.interaction_network = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        for layer in self.interaction_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        ids = self.shifted(x)
        linear = self.linear(ids).sum(dim=1).squeeze(-1)
        latent = self.embedding(ids)
        summed = latent.sum(dim=1)
        bi_interaction = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        )
        nonlinear = self.interaction_network(
            bi_interaction
        ).squeeze(-1)
        return self.bias + linear + nonlinear


def build_model(config):
    cardinalities = [FEATURE_CARDINALITIES[name] for name in FIELDS]
    family = config["family"]
    if family == "additive":
        return AdditiveModel(cardinalities)
    if family == "fm":
        return FMModel(cardinalities, config["embedding_dim"])
    if family == "nfm":
        return NFMModel(cardinalities, config["embedding_dim"])
    raise ValueError("Unknown family: " + family)


def fit_model(x_np, y_np, dates_np, config, seed):
    seed_everything(seed)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    weights = torch.from_numpy(
        make_weights(dates_np, config["half_life"])
    )

    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed)

    model.train()
    for _ in range(config["epochs"]):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x[idx])
            row_loss = F.softplus(logits) - y[idx] * logits
            batch_weights = weights[idx]
            loss = (row_loss * batch_weights).sum() / batch_weights.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def predict(model, x_np):
    x = torch.from_numpy(x_np)
    scores = np.empty(x.shape[0], dtype=np.float32)
    step = BATCH_SIZE * 2

    model.eval()
    with torch.inference_mode():
        for start in range(0, x.shape[0], step):
            end = min(start + step, x.shape[0])
            scores[start:end] = model(x[start:end]).cpu().numpy()
    return scores


def standardize(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scale = float(scores.std())
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (scores - float(scores.mean())) / scale


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_everything(SEED)

    train = load("train")
    valid = load("valid")

    x_train = make_matrix(train)
    x_valid = make_matrix(valid)
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)
    date_train = np.asarray(train.date, dtype=np.int64)
    date_valid = np.asarray(valid.date, dtype=np.int64)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid_path = os.path.join(
        shared, "incumbent_valid_scores.npy"
    )
    incumbent_test_path = os.path.join(
        shared, "incumbent_test_scores.npy"
    )
    if not os.path.exists(incumbent_valid_path):
        raise FileNotFoundError(incumbent_valid_path)

    incumbent_valid = np.load(incumbent_valid_path)
    incumbent_valid_z = standardize(incumbent_valid)

    family_valid_predictions = {}
    candidate_scores = {}
    best_choice = None
    best_primary = -np.inf

    for family_name, config in FAMILY_CONFIGS.items():
        model = fit_model(
            x_train, y_train, date_train, config, SEED
        )
        raw_scores = predict(model, x_valid)
        family_valid_predictions[family_name] = raw_scores

        standalone_metrics = evaluate(
            valid.user_id, y_valid, raw_scores
        )
        candidate_scores[family_name + "_standalone"] = float(
            standalone_metrics["primary"]
        )

        model_z = standardize(raw_scores)
        family_best_blend = -np.inf
        family_best_alpha = None

        for alpha in BLEND_WEIGHTS:
            blended = (
                alpha * model_z
                + (1.0 - alpha) * incumbent_valid_z
            )
            blend_metrics = evaluate(
                valid.user_id, y_valid, blended
            )
            primary = float(blend_metrics["primary"])

            if primary > family_best_blend:
                family_best_blend = primary
                family_best_alpha = alpha

            if primary > best_primary:
                best_primary = primary
                best_choice = {
                    "family_name": family_name,
                    "alpha": float(alpha),
                    "valid_scores": blended.copy(),
                    "metrics": blend_metrics,
                }

        candidate_scores[
            family_name + "_best_blend"
        ] = float(family_best_blend)
        candidate_scores[
            family_name + "_best_alpha"
        ] = float(family_best_alpha)

        del model
        gc.collect()

    if best_choice is None:
        raise RuntimeError("No candidate was evaluated")

    selected_valid_scores = np.asarray(
        best_choice["valid_scores"], dtype=np.float64
    )
    metrics = evaluate(
        valid.user_id, y_valid, selected_valid_scores
    )

    print(
        "FINDINGS selected_family={} alpha={:.2f}".format(
            best_choice["family_name"], best_choice["alpha"]
        )
    )
    print(
        "CANDIDATES "
        + json.dumps(candidate_scores, sort_keys=True, separators=(",", ":"))
    )

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            selected_valid_scores,
        )

    # Refit the selected recipe on all labels available before test.
    selected_name = best_choice["family_name"]
    selected_config = FAMILY_CONFIGS[selected_name]
    selected_alpha = float(best_choice["alpha"])

    test = load("test")

    if selected_alpha == 0.0:
        if not os.path.exists(incumbent_test_path):
            raise FileNotFoundError(incumbent_test_path)
        test_scores = standardize(np.load(incumbent_test_path))
    else:
        x_refit = np.concatenate([x_train, x_valid], axis=0)
        y_refit = np.concatenate(
            [y_train, y_valid.astype(np.float32)], axis=0
        )
        date_refit = np.concatenate(
            [date_train, date_valid], axis=0
        )

        refit_model = fit_model(
            x_refit,
            y_refit,
            date_refit,
            selected_config,
            SEED,
        )
        x_test = make_matrix(test)
        model_test_scores = predict(refit_model, x_test)
        model_test_z = standardize(model_test_scores)

        if selected_alpha < 1.0:
            if not os.path.exists(incumbent_test_path):
                raise FileNotFoundError(incumbent_test_path)
            incumbent_test = np.load(incumbent_test_path)
            incumbent_test_z = standardize(incumbent_test)
            test_scores = (
                selected_alpha * model_test_z
                + (1.0 - selected_alpha) * incumbent_test_z
            )
        else:
            test_scores = model_test_z

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    elapsed = time.time() - START_TIME
    payload = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }
    print("METRICS " + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()