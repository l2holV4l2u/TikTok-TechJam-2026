import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2026

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
    "video_type",
    "register_days_bucket",
]

BASE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
]

NUM_FIELDS = [
    "collect_cnt",
    "comment_cnt",
    "complete_play_cnt",
    "counts",
    "download_cnt",
    "duration_ms",
    "fans_user_num",
    "follow_cnt",
    "follow_user_num",
    "friend_user_num",
    "like_cnt",
    "long_time_play_cnt",
    "play_cnt",
    "play_duration",
    "play_progress",
    "play_user_num",
    "register_days",
    "share_cnt",
    "short_time_play_cnt",
    "show_cnt",
    "show_user_num",
    "valid_play_cnt",
]

# Targeted change: these numeric attributes are also represented by train-only
# quantile-bin embeddings, so they participate directly in FM interactions.
BIN_NUM_FIELDS = [
    "play_progress",
    "long_time_play_cnt",
    "complete_play_cnt",
    "valid_play_cnt",
    "play_duration",
    "play_cnt",
    "play_user_num",
    "like_cnt",
    "counts",
    "collect_cnt",
    "show_cnt",
    "duration_ms",
]

N_QUANTILE_BINS = 16
EMBED_DIM = 10
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 8
LEARNING_RATE = 1.0e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_cat_matrix(split, fields, offsets):
    columns = []
    for name, offset in zip(fields, offsets):
        values = np.asarray(split.X[name], dtype=np.int64)
        columns.append(values + int(offset))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


def safe_raw_numeric(split, name):
    x = np.asarray(split.num[name], dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def signed_log_numeric(split, name):
    x = safe_raw_numeric(split, name)
    return (
        np.sign(x) * np.log1p(np.abs(x))
    ).astype(np.float32, copy=False)


def raw_numeric_features(split):
    raw = {name: safe_raw_numeric(split, name) for name in NUM_FIELDS}

    transformed = []
    names = []

    for name in NUM_FIELDS:
        x = raw[name]
        x = np.sign(x) * np.log1p(np.abs(x))
        transformed.append(x.astype(np.float32, copy=False))
        names.append("log_" + name)

    ratio_specs = [
        ("complete_per_play", "complete_play_cnt", "play_cnt"),
        ("long_per_play", "long_time_play_cnt", "play_cnt"),
        ("valid_per_play", "valid_play_cnt", "play_cnt"),
        ("short_per_play", "short_time_play_cnt", "play_cnt"),
        ("like_per_show", "like_cnt", "show_cnt"),
        ("collect_per_show", "collect_cnt", "show_cnt"),
        ("comment_per_show", "comment_cnt", "show_cnt"),
        ("share_per_show", "share_cnt", "show_cnt"),
        ("follow_per_show", "follow_cnt", "show_cnt"),
        ("play_users_per_show_users", "play_user_num", "show_user_num"),
    ]

    for feature_name, numerator_name, denominator_name in ratio_specs:
        numerator = np.maximum(raw[numerator_name], 0.0).astype(np.float64)
        denominator = np.maximum(raw[denominator_name], 0.0).astype(np.float64)
        ratio = np.log1p(numerator) - np.log1p(denominator)
        transformed.append(ratio.astype(np.float32))
        names.append(feature_name)

    matrix = np.column_stack(transformed).astype(np.float32, copy=False)
    return np.ascontiguousarray(matrix), names


def fit_numeric_transform(split):
    x, names = raw_numeric_features(split)
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1.0e-5, std, 1.0).astype(np.float32)
    x = np.clip((x - mean) / std, -8.0, 8.0)
    return np.ascontiguousarray(x, dtype=np.float32), mean, std, names


def apply_numeric_transform(split, mean, std):
    x, _ = raw_numeric_features(split)
    x = np.clip((x - mean) / std, -8.0, 8.0)
    return np.ascontiguousarray(x, dtype=np.float32)


def fit_quantile_bins(split):
    quantiles = np.arange(1, N_QUANTILE_BINS, dtype=np.float64)
    quantiles /= float(N_QUANTILE_BINS)

    edges_by_field = {}
    cardinalities = []

    for name in BIN_NUM_FIELDS:
        x = signed_log_numeric(split, name)
        edges = np.quantile(x, quantiles)
        edges = np.unique(edges.astype(np.float32))
        edges_by_field[name] = edges
        cardinalities.append(int(len(edges) + 1))

    return edges_by_field, cardinalities


def make_binned_numeric_matrix(split, edges_by_field, offsets):
    columns = []

    for name, offset in zip(BIN_NUM_FIELDS, offsets):
        x = signed_log_numeric(split, name)
        edges = edges_by_field[name]
        bin_ids = np.searchsorted(edges, x, side="right").astype(np.int64)
        columns.append(bin_ids + int(offset))

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


def make_deep_cat_matrix(
    split,
    cat_offsets,
    bin_edges,
    bin_offsets,
):
    categorical = make_cat_matrix(split, CAT_FIELDS, cat_offsets)
    binned = make_binned_numeric_matrix(split, bin_edges, bin_offsets)
    return np.ascontiguousarray(
        np.column_stack([categorical, binned]),
        dtype=np.int64,
    )


class DeepFM(nn.Module):
    def __init__(self, total_cardinality, n_fields, n_numeric, initial_bias):
        super().__init__()
        self.n_fields = n_fields
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.linear = nn.Embedding(total_cardinality, 1)
        self.bias = nn.Parameter(
            torch.tensor(float(initial_bias), dtype=torch.float32)
        )

        deep_input = n_fields * EMBED_DIM + n_numeric
        self.deep = nn.Sequential(
            nn.Linear(deep_input, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.weight)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.deep[-1].weight, mean=0.0, std=0.01)

    def forward(self, categorical, numeric):
        embeddings = self.embedding(categorical)

        linear_term = self.linear(categorical).squeeze(-1).sum(dim=1)

        summed = embeddings.sum(dim=1)
        fm_term = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        ).sum(dim=1)

        deep_input = torch.cat(
            [embeddings.reshape(embeddings.shape[0], -1), numeric],
            dim=1,
        )
        deep_term = self.deep(deep_input).squeeze(1)

        return self.bias + linear_term + fm_term + deep_term


class BaselineFM(nn.Module):
    def __init__(self, total_cardinality, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        linear_term = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


@torch.inference_mode()
def predict_deepfm(model, categorical, numeric):
    model.eval()
    result = np.empty(categorical.shape[0], dtype=np.float32)

    for start in range(0, categorical.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, categorical.shape[0])
        cat_batch = torch.from_numpy(categorical[start:end])
        num_batch = torch.from_numpy(numeric[start:end])
        result[start:end] = model(cat_batch, num_batch).cpu().numpy()

    return result


@torch.inference_mode()
def predict_baseline(model, matrix):
    model.eval()
    result = np.empty(matrix.shape[0], dtype=np.float32)

    for start in range(0, matrix.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, matrix.shape[0])
        result[start:end] = model(
            torch.from_numpy(matrix[start:end])
        ).cpu().numpy()

    return result


def load_baseline(artifacts):
    if not artifacts:
        return None

    path = os.path.join(artifacts, "official_fm_k16.pt")
    if not os.path.exists(path):
        return None

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        fields = checkpoint["fields"]
        cardinalities = [int(x) for x in checkpoint["cardinalities"]]
        offsets = np.asarray(checkpoint["offsets"], dtype=np.int64)
        embedding_dim = int(checkpoint["embedding_dim"])

        model = BaselineFM(sum(cardinalities), embedding_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, fields, offsets
    except Exception as exc:
        print(
            "FINDINGS baseline_artifact_load_failed=" + repr(exc),
            flush=True,
        )
        return None


def evaluate_candidate(valid, scores):
    metrics = evaluate(valid.user_id, valid.y, scores)
    return {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
    }


def main():
    set_seed(SEED)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

    train = load("train")
    valid = load("valid")

    cat_cardinalities = [
        int(FEATURE_CARDINALITIES[name]) for name in CAT_FIELDS
    ]
    cat_offsets = np.cumsum(
        [0] + cat_cardinalities[:-1],
        dtype=np.int64,
    )

    bin_edges, bin_cardinalities = fit_quantile_bins(train)
    bin_start = int(sum(cat_cardinalities))
    bin_offsets = (
        bin_start
        + np.cumsum([0] + bin_cardinalities[:-1], dtype=np.int64)
    )

    total_cardinality = int(
        sum(cat_cardinalities) + sum(bin_cardinalities)
    )
    n_model_fields = len(CAT_FIELDS) + len(BIN_NUM_FIELDS)

    x_train_cat = make_deep_cat_matrix(
        train,
        cat_offsets,
        bin_edges,
        bin_offsets,
    )
    x_valid_cat = make_deep_cat_matrix(
        valid,
        cat_offsets,
        bin_edges,
        bin_offsets,
    )

    x_train_num, num_mean, num_std, numeric_names = fit_numeric_transform(
        train
    )
    x_valid_num = apply_numeric_transform(valid, num_mean, num_std)

    y_train_np = np.asarray(train.y, dtype=np.float32)
    y_train = torch.from_numpy(y_train_np)
    x_train_cat_t = torch.from_numpy(x_train_cat)
    x_train_num_t = torch.from_numpy(x_train_num)

    positive_rate = float(
        np.clip(y_train_np.mean(), 1.0e-6, 1.0 - 1.0e-6)
    )
    initial_bias = math.log(positive_rate / (1.0 - positive_rate))

    model = DeepFM(
        total_cardinality=total_cardinality,
        n_fields=n_model_fields,
        n_numeric=x_train_num.shape[1],
        initial_bias=initial_bias,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1.0e-6,
    )
    criterion = nn.BCEWithLogitsLoss()

    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_state = None
    best_primary = -np.inf
    best_epoch = 0
    stale_epochs = 0
    n_train = len(y_train_np)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(n_train, generator=generator)
        total_loss = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            index = permutation[start:start + BATCH_SIZE]
            cat_batch = x_train_cat_t[index]
            num_batch = x_train_num_t[index]
            label_batch = y_train[index]

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat_batch, num_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach()) * len(index)

        valid_scores_epoch = predict_deepfm(
            model,
            x_valid_cat,
            x_valid_num,
        )
        epoch_metrics = evaluate_candidate(valid, valid_scores_epoch)
        epoch_primary = epoch_metrics["primary"]

        print(
            f"epoch={epoch} "
            f"loss={total_loss / n_train:.6f} "
            f"primary={epoch_primary:.6f} "
            f"gauc={epoch_metrics['gauc']:.6f} "
            f"ndcg5={epoch_metrics['ndcg@5']:.6f}",
            flush=True,
        )

        if epoch_primary > best_primary + 1.0e-5:
            best_primary = epoch_primary
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch >= 5 and stale_epochs >= 2:
            break

    model.load_state_dict(best_state)
    deep_valid_scores = predict_deepfm(
        model,
        x_valid_cat,
        x_valid_num,
    )

    candidate_scores = {"quantile_deepfm": deep_valid_scores}
    baseline_bundle = load_baseline(os.environ.get("RUN_ARTIFACTS"))

    if baseline_bundle is not None:
        baseline_model, baseline_fields, baseline_offsets = baseline_bundle
        x_valid_base = make_cat_matrix(
            valid,
            baseline_fields,
            baseline_offsets,
        )
        baseline_valid_scores = predict_baseline(
            baseline_model,
            x_valid_base,
        )
        candidate_scores["baseline"] = baseline_valid_scores

        for deep_weight in (0.25, 0.50, 0.75):
            name = f"blend_deep_{deep_weight:.2f}"
            candidate_scores[name] = (
                deep_weight * deep_valid_scores
                + (1.0 - deep_weight) * baseline_valid_scores
            )

    candidate_metrics = {}
    best_name = None
    best_metrics = None

    for name, scores in candidate_scores.items():
        metrics = evaluate_candidate(valid, scores)
        candidate_metrics[name] = metrics

        if best_metrics is None or metrics["primary"] > best_metrics["primary"]:
            best_name = name
            best_metrics = metrics

    bin_description = {
        name: int(cardinality)
        for name, cardinality in zip(BIN_NUM_FIELDS, bin_cardinalities)
    }

    print(
        "FINDINGS "
        f"best_epoch={best_epoch} "
        f"continuous_features={len(numeric_names)} "
        f"quantile_fields={len(BIN_NUM_FIELDS)} "
        f"quantile_cardinalities={json.dumps(bin_description, separators=(',', ':'))} "
        f"selected={best_name}",
        flush=True,
    )
    print(
        "CANDIDATES "
        + json.dumps(
            {
                name: round(values["primary"], 7)
                for name, values in candidate_metrics.items()
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    artifacts = os.environ.get("RUN_ARTIFACTS")
    if artifacts:
        os.makedirs(artifacts, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "cat_fields": CAT_FIELDS,
                "cat_cardinalities": cat_cardinalities,
                "cat_offsets": cat_offsets,
                "bin_num_fields": BIN_NUM_FIELDS,
                "bin_edges": bin_edges,
                "bin_cardinalities": bin_cardinalities,
                "bin_offsets": bin_offsets,
                "numeric_names": numeric_names,
                "numeric_mean": num_mean,
                "numeric_std": num_std,
                "embedding_dim": EMBED_DIM,
                "best_epoch": best_epoch,
                "selected_candidate": best_name,
                "validation_candidates": candidate_metrics,
            },
            os.path.join(
                artifacts,
                "deepfm_numeric_quantile_embeddings.pt",
            ),
        )

    out = os.environ.get("ITER_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        test = load("test")

        x_test_cat = make_deep_cat_matrix(
            test,
            cat_offsets,
            bin_edges,
            bin_offsets,
        )
        x_test_num = apply_numeric_transform(test, num_mean, num_std)
        deep_test_scores = predict_deepfm(
            model,
            x_test_cat,
            x_test_num,
        )

        if best_name == "quantile_deepfm" or baseline_bundle is None:
            test_scores = deep_test_scores
        elif best_name == "baseline":
            baseline_model, baseline_fields, baseline_offsets = baseline_bundle
            x_test_base = make_cat_matrix(
                test,
                baseline_fields,
                baseline_offsets,
            )
            test_scores = predict_baseline(
                baseline_model,
                x_test_base,
            )
        else:
            deep_weight = float(best_name.rsplit("_", 1)[1])
            baseline_model, baseline_fields, baseline_offsets = baseline_bundle
            x_test_base = make_cat_matrix(
                test,
                baseline_fields,
                baseline_offsets,
            )
            baseline_test_scores = predict_baseline(
                baseline_model,
                x_test_base,
            )
            test_scores = (
                deep_weight * deep_test_scores
                + (1.0 - deep_weight) * baseline_test_scores
            )

        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    print(
        "METRICS "
        + json.dumps(
            {
                "primary": best_metrics["primary"],
                "gauc": best_metrics["gauc"],
                "ndcg@5": best_metrics["ndcg@5"],
                "gpu_seconds": 0.0,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()