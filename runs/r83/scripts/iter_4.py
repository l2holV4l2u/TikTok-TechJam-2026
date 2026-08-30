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
EMBED_DIM = 16
BATCH_SIZE = 8192
EPOCHS = 5
LEARNING_RATE = 0.001
FAMILIES = ["fm", "deepfm", "pnn"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[name] for name in FIELDS]),
        dtype=np.int64,
    )


class FeatureEmbeddingBase(nn.Module):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__()
        total = int(sum(cardinalities))
        offsets = np.cumsum([0] + list(cardinalities[:-1]), dtype=np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.linear = nn.Embedding(total, 1)
        self.embedding = nn.Embedding(total, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def components(self, x):
        ids = x + self.offsets
        linear = self.linear(ids).sum(dim=1).squeeze(-1)
        latent = self.embedding(ids)
        return linear, latent

    @staticmethod
    def fm_interaction(latent):
        summed = latent.sum(dim=1)
        return 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)


class ExpandedFM(FeatureEmbeddingBase):
    def forward(self, x):
        linear, latent = self.components(x)
        return self.bias + linear + self.fm_interaction(latent)


class DeepFM(FeatureEmbeddingBase):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__(cardinalities, embedding_dim)
        input_dim = len(cardinalities) * embedding_dim
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        linear, latent = self.components(x)
        fm = self.fm_interaction(latent)
        deep = self.deep(latent.flatten(start_dim=1)).squeeze(-1)
        return self.bias + linear + fm + deep


class ProductNetwork(FeatureEmbeddingBase):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__(cardinalities, embedding_dim)
        left = []
        right = []
        for i in range(len(cardinalities)):
            for j in range(i + 1, len(cardinalities)):
                left.append(i)
                right.append(j)
        self.register_buffer("pair_left", torch.tensor(left, dtype=torch.long))
        self.register_buffer("pair_right", torch.tensor(right, dtype=torch.long))

        input_dim = len(cardinalities) * embedding_dim + len(left)
        self.product_tower = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for module in self.product_tower:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        linear, latent = self.components(x)
        pair_products = (
            latent[:, self.pair_left, :] * latent[:, self.pair_right, :]
        ).sum(dim=2)
        product_input = torch.cat(
            [latent.flatten(start_dim=1), pair_products], dim=1
        )
        nonlinear = self.product_tower(product_input).squeeze(-1)
        return self.bias + linear + nonlinear


def build_model(family):
    cardinalities = [FEATURE_CARDINALITIES[name] for name in FIELDS]
    if family == "fm":
        return ExpandedFM(cardinalities, EMBED_DIM)
    if family == "deepfm":
        return DeepFM(cardinalities, EMBED_DIM)
    if family == "pnn":
        return ProductNetwork(cardinalities, EMBED_DIM)
    raise ValueError(f"Unknown family: {family}")


def fit_model(family, x_np, y_np, seed):
    seed_everything(seed)
    model = build_model(family)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    n = x.shape[0]

    generator = torch.Generator()
    generator.manual_seed(seed)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x[idx])
            loss = criterion(logits, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def predict(model, x_np):
    x = torch.from_numpy(x_np)
    scores = np.empty(x.shape[0], dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for start in range(0, x.shape[0], BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, x.shape[0])
            scores[start:end] = model(x[start:end]).cpu().numpy()
    return scores


def standardize(scores):
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(scores.mean())
    std = float(scores.std())
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (scores - mean) / std


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_everything(SEED)

    train = load("train")
    valid = load("valid")
    x_train = make_matrix(train)
    x_valid = make_matrix(valid)
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
    incumbent_test_path = os.path.join(shared, "incumbent_test_scores.npy")
    incumbent_valid = np.load(incumbent_valid_path).astype(np.float64)
    incumbent_valid_z = standardize(incumbent_valid)

    candidate_primary = {}
    family_predictions = {}
    best_primary = -np.inf
    best_family = None
    best_alpha = None
    best_valid_scores = None
    best_metrics = None

    blend_alphas = np.linspace(0.1, 1.0, 10)

    for family_index, family in enumerate(FAMILIES):
        model = fit_model(
            family,
            x_train,
            y_train,
            SEED + 97 * family_index,
        )
        raw_scores = predict(model, x_valid).astype(np.float64)
        model_scores_z = standardize(raw_scores)
        family_predictions[family] = model_scores_z

        standalone_metrics = evaluate(valid.user_id, y_valid, raw_scores)
        candidate_primary[family] = float(standalone_metrics["primary"])

        for alpha in blend_alphas:
            alpha = float(alpha)
            if alpha >= 1.0 - 1e-12:
                scores = model_scores_z
                name = family
            else:
                scores = (
                    alpha * model_scores_z
                    + (1.0 - alpha) * incumbent_valid_z
                )
                name = f"{family}+inc_a{alpha:.1f}"

            metrics = evaluate(valid.user_id, y_valid, scores)
            primary = float(metrics["primary"])
            candidate_primary[name] = primary

            if primary > best_primary:
                best_primary = primary
                best_family = family
                best_alpha = alpha
                best_valid_scores = np.asarray(scores, dtype=np.float64).copy()
                best_metrics = metrics

        del model
        gc.collect()

    print(
        "CANDIDATES "
        + json.dumps(candidate_primary, sort_keys=True, separators=(",", ":"))
    )
    print(
        "FINDINGS "
        + json.dumps(
            {
                "selected_family": best_family,
                "selected_model_weight": best_alpha,
                "expanded_fields": FIELDS,
            },
            separators=(",", ":"),
        )
    )

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(best_valid_scores, dtype=np.float64),
        )

    # Refit the selected family with all labels available before the test period.
    x_refit = np.concatenate([x_train, x_valid], axis=0)
    y_refit = np.concatenate(
        [y_train, np.asarray(y_valid, dtype=np.float32)],
        axis=0,
    )

    selected_index = FAMILIES.index(best_family)
    refit_model = fit_model(
        best_family,
        x_refit,
        y_refit,
        SEED + 97 * selected_index,
    )

    test = load("test")
    x_test = make_matrix(test)
    model_test_raw = predict(refit_model, x_test)
    model_test_z = standardize(model_test_raw)

    if best_alpha < 1.0 - 1e-12:
        incumbent_test = np.load(incumbent_test_path).astype(np.float64)
        incumbent_test_z = standardize(incumbent_test)
        test_scores = (
            best_alpha * model_test_z
            + (1.0 - best_alpha) * incumbent_test_z
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
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }
    print("METRICS " + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()