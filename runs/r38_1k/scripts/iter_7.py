import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
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
PAIR_FIELDS = [
    "tag",
    "duration_bucket",
    "upload_type",
    "tab",
    "music_type",
    "hour",
    "author_id",
]
PAIR_ALPHAS = {
    "tag": 30.0,
    "duration_bucket": 50.0,
    "upload_type": 40.0,
    "tab": 50.0,
    "music_type": 60.0,
    "hour": 100.0,
    "author_id": 20.0,
}

K = 16
LR = 0.001
EPOCHS = 10
TRAIN_BATCH_SIZE = 8192
PRED_BATCH_SIZE = 131072
EPS = 1e-5
MAX_DENSE_PAIR_SIZE = 100_000_000

np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class FactorizationMachine(torch.nn.Module):
    def __init__(self, cardinalities, k):
        super().__init__()
        offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        self.embedding = torch.nn.Embedding(
            int(sum(cardinalities)), k + 1, sparse=True
        )

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)
            self.embedding.weight[self.offsets].zero_()

    def forward(self, x):
        z = self.embedding(x + self.offsets)
        linear = z[:, :, 0].sum(dim=1)
        factors = z[:, :, 1:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return linear + interaction


def matrix_from_split(split):
    missing = [name for name in FIELDS if name not in split.X]
    if missing:
        raise KeyError("Missing required fields: " + repr(missing))
    return np.column_stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS]
    )


@torch.no_grad()
def predict_fm(model, x, intercept):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        stop = min(start + PRED_BATCH_SIZE, x.shape[0])
        xb = torch.from_numpy(x[start:stop])
        result[start:stop] = (
            model(xb).add(intercept).cpu().numpy().astype(np.float32)
        )
    return result


def clipped_logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def fit_pair_statistics(train, field, alpha):
    user = np.asarray(train.X["user_id"], dtype=np.int64)
    value = np.asarray(train.X[field], dtype=np.int64)
    y = np.asarray(train.y, dtype=np.float64)

    user_card = int(FEATURE_CARDINALITIES["user_id"])
    value_card = int(FEATURE_CARDINALITIES[field])
    pair_size = user_card * value_card
    keys = user * np.int64(value_card) + value

    value_count = np.bincount(
        value, minlength=value_card
    ).astype(np.float64)
    value_pos = np.bincount(
        value, weights=y, minlength=value_card
    ).astype(np.float64)

    global_rate = float(y.mean())
    value_rate = (
        value_pos + 20.0 * global_rate
    ) / (value_count + 20.0)
    value_rate = np.clip(value_rate, EPS, 1.0 - EPS)

    if pair_size <= MAX_DENSE_PAIR_SIZE:
        pair_count = np.bincount(
            keys, minlength=pair_size
        ).astype(np.float32)
        pair_pos = np.bincount(
            keys, weights=y, minlength=pair_size
        ).astype(np.float32)

        prior_rate_for_pair = np.tile(
            value_rate.astype(np.float32), user_card
        )
        posterior = (
            pair_pos + np.float32(alpha) * prior_rate_for_pair
        ) / (pair_count + np.float32(alpha))

        residual = (
            clipped_logit(posterior.astype(np.float64))
            - clipped_logit(prior_rate_for_pair.astype(np.float64))
        ).astype(np.float32)

        del pair_count, pair_pos, prior_rate_for_pair, keys
        return {
            "storage": "dense",
            "value_card": value_card,
            "residual": residual,
        }

    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    pair_pos = np.bincount(
        inverse, weights=y, minlength=unique_keys.size
    ).astype(np.float64)
    pair_count = counts.astype(np.float64)

    unique_values = np.remainder(
        unique_keys, np.int64(value_card)
    ).astype(np.int64)
    prior = value_rate[unique_values]
    posterior = (
        pair_pos + float(alpha) * prior
    ) / (pair_count + float(alpha))

    residual = (
        clipped_logit(posterior)
        - clipped_logit(prior)
    ).astype(np.float32)

    del keys, inverse, counts, pair_pos, pair_count
    del unique_values, prior, posterior

    return {
        "storage": "sparse",
        "value_card": value_card,
        "keys": unique_keys,
        "residual": residual,
    }


def apply_pair_statistics(split, field, stats):
    user = np.asarray(split.X["user_id"], dtype=np.int64)
    value = np.asarray(split.X[field], dtype=np.int64)
    value_card = int(stats["value_card"])
    keys = user * np.int64(value_card) + value
    residual = stats["residual"]

    if stats["storage"] == "dense":
        valid_key = (keys >= 0) & (keys < residual.size)
        out = np.zeros(keys.shape[0], dtype=np.float32)
        out[valid_key] = residual[keys[valid_key]]
        return out

    stored_keys = stats["keys"]
    positions = np.searchsorted(stored_keys, keys)
    out = np.zeros(keys.shape[0], dtype=np.float32)

    in_bounds = positions < stored_keys.size
    row_indices = np.flatnonzero(in_bounds)
    if row_indices.size:
        bounded_positions = positions[row_indices]
        matched = stored_keys[bounded_positions] == keys[row_indices]
        matched_rows = row_indices[matched]
        out[matched_rows] = residual[bounded_positions[matched]]

    return out


def candidate_scores(base, residuals, coefficients):
    score = np.asarray(base, dtype=np.float32).copy()
    for field, coefficient in coefficients.items():
        if coefficient != 0.0:
            score += np.float32(coefficient) * residuals[field]
    return score


def compact_metrics(metrics):
    return {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
    }


def main():
    train = load("train")
    valid = load("valid")

    cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
    x_train = matrix_from_split(train)
    x_valid = matrix_from_split(valid)
    y_train = np.asarray(train.y, dtype=np.float32)

    model = FactorizationMachine(cardinalities, K)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)

    positive_rate = float(
        np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6)
    )
    intercept = float(
        np.log(positive_rate / (1.0 - positive_rate))
    )

    n = x_train.shape[0]
    best_primary = -np.inf
    best_weight = None
    best_epoch = -1

    for epoch in range(EPOCHS):
        model.train()
        order = np.random.permutation(n)

        for start in range(0, n, TRAIN_BATCH_SIZE):
            idx = order[start:min(start + TRAIN_BATCH_SIZE, n)]
            xb = torch.from_numpy(x_train[idx])
            yb = torch.from_numpy(y_train[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb) + intercept
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

        del order

        valid_epoch_scores = predict_fm(model, x_valid, intercept)
        epoch_metrics = evaluate(
            valid.user_id, valid.y, valid_epoch_scores
        )
        epoch_primary = float(epoch_metrics["primary"])

        if epoch_primary > best_primary:
            best_primary = epoch_primary
            best_epoch = epoch + 1
            best_weight = model.embedding.weight.detach().clone()

        del valid_epoch_scores

    if best_weight is None:
        raise RuntimeError("No FM checkpoint was produced.")

    with torch.no_grad():
        model.embedding.weight.copy_(best_weight)
    model.eval()

    base_valid = predict_fm(model, x_valid, intercept)
    base_metrics = evaluate(valid.user_id, valid.y, base_valid)

    del optimizer, best_weight, x_train, y_train
    gc.collect()

    pair_stats = {}
    valid_residuals = {}
    pair_coverage = {}

    for field in PAIR_FIELDS:
        stats = fit_pair_statistics(
            train, field, PAIR_ALPHAS[field]
        )
        pair_stats[field] = stats
        residual = apply_pair_statistics(valid, field, stats)
        valid_residuals[field] = residual
        pair_coverage[field] = float(np.mean(residual != 0.0))
        gc.collect()

    multi_medium = {
        "tag": 0.50,
        "duration_bucket": 0.40,
        "upload_type": 0.40,
        "tab": 0.30,
        "music_type": 0.30,
        "hour": 0.20,
    }

    candidates = {
        "fm": {},
        "fm_tag_025": {
            "tag": 0.25,
        },
        "fm_tag_050": {
            "tag": 0.50,
        },
        "fm_tag_075": {
            "tag": 0.75,
        },
        "fm_multi_low": {
            "tag": 0.25,
            "duration_bucket": 0.20,
            "upload_type": 0.20,
            "tab": 0.15,
            "music_type": 0.15,
            "hour": 0.10,
        },
        "fm_multi_medium": multi_medium,
        "fm_content_medium": {
            "tag": 0.50,
            "duration_bucket": 0.40,
            "upload_type": 0.40,
            "music_type": 0.30,
        },
        "fm_tag_upload": {
            "tag": 0.50,
            "upload_type": 0.35,
        },
        "fm_author_025": {
            "author_id": 0.25,
        },
        "fm_author_050": {
            "author_id": 0.50,
        },
        "fm_author_075": {
            "author_id": 0.75,
        },
        "fm_multi_author_025": {
            **multi_medium,
            "author_id": 0.25,
        },
        "fm_multi_author_050": {
            **multi_medium,
            "author_id": 0.50,
        },
        "fm_multi_author_075": {
            **multi_medium,
            "author_id": 0.75,
        },
        "fm_multi_author_100": {
            **multi_medium,
            "author_id": 1.00,
        },
    }

    candidate_metrics = {}
    best_name = None
    best_coefficients = None
    best_metrics = None

    for name, coefficients in candidates.items():
        if name == "fm":
            scores = base_valid
        else:
            scores = candidate_scores(
                base_valid, valid_residuals, coefficients
            )

        metrics = evaluate(valid.user_id, valid.y, scores)
        candidate_metrics[name] = compact_metrics(metrics)

        if (
            best_metrics is None
            or float(metrics["primary"])
            > float(best_metrics["primary"])
        ):
            best_name = name
            best_coefficients = dict(coefficients)
            best_metrics = metrics

        if name != "fm":
            del scores

    print(
        "FINDINGS "
        + json.dumps(
            {
                "best_fm_epoch": best_epoch,
                "base_primary": float(base_metrics["primary"]),
                "selected": best_name,
                "selected_coefficients": best_coefficients,
                "author_pair_storage": pair_stats[
                    "author_id"
                ]["storage"],
                "author_seen_pair_coverage": pair_coverage[
                    "author_id"
                ],
                "author_observed_pairs": int(
                    pair_stats["author_id"]["residual"].size
                ),
            },
            separators=(",", ":"),
        )
    )
    print(
        "CANDIDATES "
        + json.dumps(
            {
                name: values["primary"]
                for name, values in candidate_metrics.items()
            },
            separators=(",", ":"),
        )
    )

    del x_valid, base_valid, valid_residuals
    del valid
    gc.collect()

    out = os.environ.get("ITER_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        test = load("test")
        x_test = matrix_from_split(test)
        test_scores = predict_fm(model, x_test, intercept)

        if best_coefficients:
            for field, coefficient in best_coefficients.items():
                if coefficient != 0.0:
                    residual = apply_pair_statistics(
                        test, field, pair_stats[field]
                    )
                    test_scores += (
                        np.float32(coefficient) * residual
                    )
                    del residual

        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

        with open(
            os.path.join(out, "selection.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "candidate": best_name,
                    "coefficients": best_coefficients,
                    "best_fm_epoch": best_epoch,
                },
                f,
                separators=(",", ":"),
            )

    payload = {
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": 0.0,
    }
    print("METRICS " + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()