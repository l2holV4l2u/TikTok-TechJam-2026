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


START = time.time()
SEED = 18427
THREADS = min(16, os.cpu_count() or 1)
BATCH_SIZE = 8192
DEV_EPOCHS = 2
FULL_EPOCHS = 4
EMBED_DIM = 16

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "hour",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat2",
    "onehot_feat7",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "is_video_author",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def make_cat(split):
    return np.ascontiguousarray(
        np.stack(
            [
                np.asarray(split.X[name], dtype=np.int64)
                for name in FIELDS
            ],
            axis=1,
        ),
        dtype=np.int64,
    )


def make_raw_num(split):
    columns = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)))
    return np.ascontiguousarray(np.stack(columns, axis=1))


def robust_scale(train_num, query_num):
    center = np.median(train_num, axis=0).astype(np.float32)
    q25 = np.percentile(train_num, 25, axis=0).astype(np.float32)
    q75 = np.percentile(train_num, 75, axis=0).astype(np.float32)
    scale = np.maximum(q75 - q25, 0.1).astype(np.float32)
    result = (query_num - center[None, :]) / scale[None, :]
    return np.ascontiguousarray(
        np.clip(result, -8.0, 8.0), dtype=np.float32
    )


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]

    start_index = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = ordered_users[:-1] != ordered_users[1:]
    end_indices = np.flatnonzero(ends)

    sizes = np.diff(
        np.concatenate(
            [np.asarray([-1], dtype=np.int64), end_indices]
        )
    )
    row_sizes = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - start_index

    ranked_order = (
        positions.astype(np.float64) + 0.5
    ) / row_sizes.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_order
    return result


class RobustFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    int(FEATURE_CARDINALITIES[name]), EMBED_DIM
                )
                for name in FIELDS
            ]
        )
        self.linear_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    int(FEATURE_CARDINALITIES[name]), 1
                )
                for name in FIELDS
            ]
        )
        self.numeric_linear = nn.Linear(len(NUM_FIELDS), 1)
        self.numeric_factor = nn.Linear(
            len(NUM_FIELDS), EMBED_DIM, bias=False
        )
        self.bias = nn.Parameter(torch.zeros(()))

        for emb in self.factor_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)
        for emb in self.linear_embeddings:
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.numeric_linear.weight)
        nn.init.zeros_(self.numeric_linear.bias)
        nn.init.normal_(
            self.numeric_factor.weight, mean=0.0, std=0.02
        )

    def forward(self, x_cat, x_num):
        factors = torch.stack(
            [
                emb(x_cat[:, j])
                for j, emb in enumerate(self.factor_embeddings)
            ],
            dim=1,
        )

        numeric_factor = self.numeric_factor(x_num).unsqueeze(1)
        factors = torch.cat([factors, numeric_factor], dim=1)

        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)

        wide = torch.stack(
            [
                emb(x_cat[:, j]).squeeze(1)
                for j, emb in enumerate(self.linear_embeddings)
            ],
            dim=1,
        ).sum(dim=1)

        return (
            wide
            + interaction
            + self.numeric_linear(x_num).squeeze(1)
            + self.bias
        )


def normalized_weights(weights):
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.maximum(weights, 1e-5)
    return weights / max(float(weights.mean()), 1e-8)


def static_scheme_weights(scheme, dates, users):
    dates = np.asarray(dates, dtype=np.int32)
    users = np.asarray(users, dtype=np.int64)
    age = dates.max() - dates

    if scheme == "uniform":
        weights = np.ones(len(dates), dtype=np.float32)
    elif scheme.startswith("half_life_"):
        half_life = float(scheme.split("_")[-1])
        weights = np.power(
            0.5, age.astype(np.float32) / half_life
        )
    elif scheme == "user_balanced_half_life_4":
        temporal = np.power(
            0.5, age.astype(np.float32) / 4.0
        )
        counts = np.bincount(users)
        user_weight = 1.0 / np.sqrt(
            np.maximum(counts[users], 1).astype(np.float32)
        )
        weights = temporal * user_weight
    elif scheme == "date_balanced_half_life_4":
        temporal = np.power(
            0.5, age.astype(np.float32) / 4.0
        )
        unique_dates, inverse = np.unique(dates, return_inverse=True)
        date_counts = np.bincount(inverse).astype(np.float32)
        weights = temporal / np.maximum(date_counts[inverse], 1.0)
    elif scheme == "group_dro":
        weights = np.ones(len(dates), dtype=np.float32)
    else:
        raise ValueError(scheme)

    return normalized_weights(weights)


def fit_model(
    x_cat,
    x_num,
    labels,
    dates,
    users,
    fit_indices,
    scheme,
    epochs,
    seed,
):
    torch.manual_seed(seed)
    model = RobustFM()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=2e-5
    )

    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    fit_dates = np.asarray(dates[fit_indices], dtype=np.int32)
    fit_users = np.asarray(users[fit_indices], dtype=np.int64)

    unique_dates, local_date_index = np.unique(
        fit_dates, return_inverse=True
    )
    n_dates = len(unique_dates)
    date_counts = np.bincount(
        local_date_index, minlength=n_dates
    ).astype(np.float64)

    group_q = np.full(n_dates, 1.0 / n_dates, dtype=np.float64)
    base_weights = static_scheme_weights(
        scheme, fit_dates, fit_users
    )

    labels_tensor = torch.from_numpy(
        np.asarray(labels, dtype=np.float32)
    )

    n_fit = len(fit_indices)
    for epoch in range(epochs):
        if scheme == "group_dro":
            row_weights = (
                group_q[local_date_index]
                / np.maximum(date_counts[local_date_index], 1.0)
            )
            row_weights = normalized_weights(row_weights)
        else:
            row_weights = base_weights

        generator = torch.Generator()
        generator.manual_seed(seed + 997 * epoch)
        permutation = torch.randperm(
            n_fit, generator=generator
        ).numpy()

        date_loss_sum = np.zeros(n_dates, dtype=np.float64)
        date_loss_count = np.zeros(n_dates, dtype=np.float64)
        total_loss = 0.0

        model.train()
        for start in range(0, n_fit, BATCH_SIZE):
            local = permutation[start:start + BATCH_SIZE]
            idx = fit_indices[local]

            cat_batch = torch.from_numpy(x_cat[idx])
            num_batch = torch.from_numpy(x_num[idx])
            y_batch = labels_tensor[idx]
            w_batch = torch.from_numpy(row_weights[local])

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat_batch, num_batch)
            individual_losses = F.binary_cross_entropy_with_logits(
                logits, y_batch, reduction="none"
            )
            loss = (
                individual_losses * w_batch
            ).sum() / w_batch.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            losses_np = individual_losses.detach().numpy()
            local_groups = local_date_index[local]
            date_loss_sum += np.bincount(
                local_groups,
                weights=losses_np,
                minlength=n_dates,
            )
            date_loss_count += np.bincount(
                local_groups, minlength=n_dates
            )
            total_loss += float(loss.detach()) * len(local)

        if scheme == "group_dro":
            group_losses = date_loss_sum / np.maximum(
                date_loss_count, 1.0
            )
            centered = group_losses - group_losses.mean()
            group_q *= np.exp(8.0 * centered)
            group_q /= group_q.sum()

        print(
            "FINDINGS "
            + json.dumps(
                {
                    "stage": "fit",
                    "scheme": scheme,
                    "epoch": epoch + 1,
                    "loss": total_loss / n_fit,
                    "dro_max_date_weight": (
                        float(group_q.max())
                        if scheme == "group_dro"
                        else None
                    ),
                },
                sort_keys=True,
            )
        )

    return model


@torch.no_grad()
def predict(model, x_cat, x_num, indices=None):
    model.eval()
    if indices is None:
        indices = np.arange(len(x_cat), dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)

    result = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), BATCH_SIZE * 2):
        local = indices[start:start + BATCH_SIZE * 2]
        logits = model(
            torch.from_numpy(x_cat[local]),
            torch.from_numpy(x_num[local]),
        )
        result[start:start + len(local)] = (
            logits.numpy().astype(np.float32)
        )
    return result


train = load("train")
valid = load("valid")
test = load("test")

xcat_train = make_cat(train)
xcat_valid = make_cat(valid)
xcat_test = make_cat(test)

raw_num_train = make_raw_num(train)
raw_num_valid = make_raw_num(valid)
raw_num_test = make_raw_num(test)

xnum_train = robust_scale(raw_num_train, raw_num_train)
xnum_valid = robust_scale(raw_num_train, raw_num_valid)
xnum_test = robust_scale(raw_num_train, raw_num_test)

del raw_num_train, raw_num_valid, raw_num_test
gc.collect()

y_train = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int32)
train_users = np.asarray(train.user_id, dtype=np.int64)

# The last two train days select the drift weighting protocol.
development_mask = train_dates <= 20220419
holdout_mask = train_dates >= 20220420
development_indices = np.flatnonzero(development_mask)
holdout_indices = np.flatnonzero(holdout_mask)

schemes = [
    "uniform",
    "half_life_12",
    "half_life_8",
    "half_life_4",
    "half_life_2",
    "user_balanced_half_life_4",
    "date_balanced_half_life_4",
    "group_dro",
]

holdout_results = {}

for scheme_index, scheme in enumerate(schemes):
    model = fit_model(
        xcat_train,
        xnum_train,
        y_train,
        train_dates,
        train_users,
        development_indices,
        scheme,
        DEV_EPOCHS,
        SEED + 101 * scheme_index,
    )
    holdout_scores = predict(
        model, xcat_train, xnum_train, holdout_indices
    )
    metrics = evaluate(
        train_users[holdout_indices],
        y_train[holdout_indices],
        holdout_scores,
    )
    holdout_results[scheme] = metrics
    del model, holdout_scores
    gc.collect()

ranked_schemes = sorted(
    schemes,
    key=lambda name: float(
        holdout_results[name]["primary"]
    ),
    reverse=True,
)
selected_schemes = ranked_schemes[:2]

print(
    "FINDINGS "
    + json.dumps(
        {
            "selection_split": "train_dates_20220420_20220421",
            "holdout_primary": {
                name: float(holdout_results[name]["primary"])
                for name in schemes
            },
            "selected_schemes": selected_schemes,
        },
        sort_keys=True,
    )
)

all_train_indices = np.arange(len(y_train), dtype=np.int64)
valid_predictions = []
test_predictions = []

for selected_index, scheme in enumerate(selected_schemes):
    model = fit_model(
        xcat_train,
        xnum_train,
        y_train,
        train_dates,
        train_users,
        all_train_indices,
        scheme,
        FULL_EPOCHS,
        SEED + 5000 + 211 * selected_index,
    )
    valid_predictions.append(
        predict(model, xcat_valid, xnum_valid)
    )
    test_predictions.append(
        predict(model, xcat_test, xnum_test)
    )
    del model
    gc.collect()

valid_ranks = [
    within_user_rank(valid.user_id, score)
    for score in valid_predictions
]
test_ranks = [
    within_user_rank(test.user_id, score)
    for score in test_predictions
]

# This ensemble specification is fixed by the train-only holdout.
own_valid = np.mean(np.stack(valid_ranks, axis=1), axis=1)
own_test = np.mean(np.stack(test_ranks, axis=1), axis=1)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_valid = {
    "trusted_incumbent": inc_valid,
    "train_selected_drift_ensemble": own_valid,
}
candidate_test = {
    "trusted_incumbent": inc_test,
    "train_selected_drift_ensemble": own_test,
}

for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
    name = f"drift_ensemble_incumbent_blend_{alpha:.2f}"
    candidate_valid[name] = (
        alpha * own_valid
        + (1.0 - alpha) * inc_valid_rank
    )
    candidate_test[name] = (
        alpha * own_test
        + (1.0 - alpha) * inc_test_rank
    )

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_valid.items()
}

best_name = max(
    candidate_metrics,
    key=lambda name: float(
        candidate_metrics[name]["primary"]
    ),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(
    candidate_valid[best_name], dtype=np.float64
)
best_test = np.asarray(
    candidate_test[best_name], dtype=np.float64
)

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: float(metrics["primary"])
            for name, metrics in candidate_metrics.items()
        },
        sort_keys=True,
    )
)

print(
    "FINDINGS "
    + json.dumps(
        {
            "best_candidate": best_name,
            "selected_by_train_holdout": selected_schemes,
            "own_primary": float(
                candidate_metrics[
                    "train_selected_drift_ensemble"
                ]["primary"]
            ),
            "own_gauc": float(
                candidate_metrics[
                    "train_selected_drift_ensemble"
                ]["gauc"]
            ),
            "own_ndcg5": float(
                candidate_metrics[
                    "train_selected_drift_ensemble"
                ]["ndcg@5"]
            ),
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        best_valid,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        best_test,
    )
    if best_name != "train_selected_drift_ensemble":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)