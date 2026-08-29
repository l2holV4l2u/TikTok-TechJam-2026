import gc
import json
import os

import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import ndtri

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024

FM_FIELDS = [
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

RERANK_COEFFICIENTS = {
    "tag": 0.50,
    "duration_bucket": 0.40,
    "upload_type": 0.40,
    "tab": 0.30,
    "music_type": 0.30,
    "hour": 0.20,
}

TREE_FIELDS = [
    "author_id",
    "tab",
    "tag",
    "upload_type",
    "duration_bucket",
    "music_type",
    "hour",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
    "onehot_feat2",
    "onehot_feat1",
    "register_days_bucket",
    "register_days_range",
    "user_active_degree",
    "fans_user_num_range",
]

K = 16
LR = 0.001
FM_EPOCHS = 10
TRAIN_BATCH_SIZE = 8192
PRED_BATCH_SIZE = 131072
EPS = 1e-5
MAX_DENSE_PAIR_SIZE = 100_000_000

RAW_TREE_WEIGHTS = [
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

RANK_TREE_WEIGHTS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

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
        embedded = self.embedding(x + self.offsets)
        linear = embedded[:, :, 0].sum(dim=1)
        factors = embedded[:, :, 1:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return linear + interaction


def fm_matrix(split):
    return np.column_stack(
        [
            np.asarray(split.X[name], dtype=np.int64)
            for name in FM_FIELDS
        ]
    )


def tree_matrix(split):
    return np.column_stack(
        [
            np.asarray(split.X[name], dtype=np.int32)
            for name in TREE_FIELDS
        ]
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
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def compact_metrics(metrics):
    return {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
    }


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

        prior = np.tile(value_rate.astype(np.float32), user_card)
        posterior = (
            pair_pos + np.float32(alpha) * prior
        ) / (pair_count + np.float32(alpha))

        residual = (
            clipped_logit(posterior) - clipped_logit(prior)
        ).astype(np.float32)

        del keys, pair_count, pair_pos, prior, posterior
        return {
            "storage": "dense",
            "value_card": value_card,
            "residual": residual,
        }

    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    pair_pos = np.bincount(
        inverse,
        weights=y,
        minlength=unique_keys.size,
    ).astype(np.float64)

    unique_values = np.remainder(
        unique_keys, np.int64(value_card)
    ).astype(np.int64)
    prior = value_rate[unique_values]
    posterior = (
        pair_pos + float(alpha) * prior
    ) / (counts.astype(np.float64) + float(alpha))

    residual = (
        clipped_logit(posterior) - clipped_logit(prior)
    ).astype(np.float32)

    del keys, inverse, counts, pair_pos
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

    if stats["storage"] == "dense":
        stored = stats["residual"]
        out = np.zeros(keys.size, dtype=np.float32)
        valid = (keys >= 0) & (keys < stored.size)
        out[valid] = stored[keys[valid]]
        return out

    stored_keys = stats["keys"]
    stored_residual = stats["residual"]
    positions = np.searchsorted(stored_keys, keys)
    out = np.zeros(keys.size, dtype=np.float32)

    in_bounds = positions < stored_keys.size
    rows = np.flatnonzero(in_bounds)
    if rows.size:
        bounded_positions = positions[rows]
        matched = stored_keys[bounded_positions] == keys[rows]
        matched_rows = rows[matched]
        out[matched_rows] = stored_residual[
            bounded_positions[matched]
        ]
    return out


def build_reranked_score(
    split,
    base_score,
    pair_stats,
    author_coefficient,
    collect_coverage=False,
):
    score = np.asarray(base_score, dtype=np.float32).copy()
    coverage = {}

    for field, coefficient in RERANK_COEFFICIENTS.items():
        residual = apply_pair_statistics(
            split, field, pair_stats[field]
        )
        score += np.float32(coefficient) * residual
        if collect_coverage:
            coverage[field] = float(np.mean(residual != 0.0))
        del residual

    author_residual = apply_pair_statistics(
        split, "author_id", pair_stats["author_id"]
    )
    score += np.float32(author_coefficient) * author_residual
    if collect_coverage:
        coverage["author_id"] = float(
            np.mean(author_residual != 0.0)
        )

    return score, author_residual, coverage


def within_user_percentile(user_ids, scores):
    """
    Convert arbitrary scores to ascending within-user mid-rank percentiles.
    Larger input scores receive larger output values.
    """
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size

    order = np.lexsort((scores, user_ids))
    sorted_users = user_ids[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]

    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    repeated_starts = np.repeat(starts, sizes)
    repeated_sizes = np.repeat(sizes, sizes)

    ordinal = np.arange(n, dtype=np.int64) - repeated_starts
    sorted_percentile = (
        ordinal.astype(np.float64) + 0.5
    ) / repeated_sizes.astype(np.float64)

    result = np.empty(n, dtype=np.float32)
    result[order] = sorted_percentile.astype(np.float32)
    return result


def make_combination(
    mode,
    weight,
    incumbent_score,
    tree_score,
    incumbent_pct=None,
    tree_pct=None,
    incumbent_gauss=None,
    tree_gauss=None,
):
    if mode == "raw":
        return (
            np.asarray(incumbent_score, dtype=np.float32)
            + np.float32(weight)
            * np.asarray(tree_score, dtype=np.float32)
        )

    if mode == "borda":
        return (
            np.float32(1.0 - weight) * incumbent_pct
            + np.float32(weight) * tree_pct
        )

    if mode == "gaussian":
        return (
            np.float32(1.0 - weight) * incumbent_gauss
            + np.float32(weight) * tree_gauss
        )

    raise ValueError("Unknown combination mode: " + str(mode))


def main():
    train = load("train")
    valid = load("valid")

    x_train = fm_matrix(train)
    x_valid = fm_matrix(valid)
    y_train = np.asarray(train.y, dtype=np.float32)

    cardinalities = [
        int(FEATURE_CARDINALITIES[name]) for name in FM_FIELDS
    ]
    model = FactorizationMachine(cardinalities, K)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)

    positive_rate = float(
        np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6)
    )
    intercept = float(
        np.log(positive_rate / (1.0 - positive_rate))
    )

    n_train = x_train.shape[0]
    best_fm_primary = -np.inf
    best_fm_epoch = -1
    best_weight = None

    for epoch in range(FM_EPOCHS):
        model.train()
        order = np.random.permutation(n_train)

        for start in range(0, n_train, TRAIN_BATCH_SIZE):
            idx = order[
                start:min(start + TRAIN_BATCH_SIZE, n_train)
            ]
            xb = torch.from_numpy(x_train[idx])
            yb = torch.from_numpy(y_train[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb) + intercept
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

        del order

        epoch_score = predict_fm(model, x_valid, intercept)
        epoch_metrics = evaluate(
            valid.user_id, valid.y, epoch_score
        )
        epoch_primary = float(epoch_metrics["primary"])

        if epoch_primary > best_fm_primary:
            best_fm_primary = epoch_primary
            best_fm_epoch = epoch + 1
            best_weight = model.embedding.weight.detach().clone()

        del epoch_score

    if best_weight is None:
        raise RuntimeError("FM training produced no checkpoint")

    with torch.no_grad():
        model.embedding.weight.copy_(best_weight)

    base_valid = predict_fm(model, x_valid, intercept)

    del optimizer, best_weight, x_train, x_valid, y_train
    gc.collect()

    pair_stats = {}
    for field in PAIR_FIELDS:
        pair_stats[field] = fit_pair_statistics(
            train, field, PAIR_ALPHAS[field]
        )
        gc.collect()

    context_valid, author_residual_valid, coverage = (
        build_reranked_score(
            valid,
            base_valid,
            pair_stats,
            author_coefficient=0.0,
            collect_coverage=True,
        )
    )

    candidate_scores = {}
    best_author_coefficient = 0.0
    best_incumbent_metrics = None

    for coefficient in [0.0, 0.25, 0.50, 0.75, 1.0]:
        score = (
            context_valid
            + np.float32(coefficient) * author_residual_valid
        )
        metrics = evaluate(valid.user_id, valid.y, score)
        name = "inc_author_" + str(coefficient).replace(".", "")
        candidate_scores[name] = float(metrics["primary"])

        if (
            best_incumbent_metrics is None
            or float(metrics["primary"])
            > float(best_incumbent_metrics["primary"])
        ):
            best_incumbent_metrics = metrics
            best_author_coefficient = coefficient

        del score

    incumbent_valid = (
        context_valid
        + np.float32(best_author_coefficient)
        * author_residual_valid
    ).astype(np.float32)

    del context_valid, author_residual_valid, base_valid
    gc.collect()

    x_tree_train = tree_matrix(train)
    x_tree_valid = tree_matrix(valid)

    train_dataset = lgb.Dataset(
        x_tree_train,
        label=np.asarray(train.y, dtype=np.float32),
        categorical_feature=list(range(len(TREE_FIELDS))),
        free_raw_data=True,
    )
    valid_dataset = lgb.Dataset(
        x_tree_valid,
        label=np.asarray(valid.y, dtype=np.float32),
        categorical_feature=list(range(len(TREE_FIELDS))),
        reference=train_dataset,
        free_raw_data=True,
    )

    tree_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.06,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 800,
        "min_sum_hessian_in_leaf": 10.0,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_cat_threshold": 64,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "max_bin": 127,
        "num_threads": min(16, os.cpu_count() or 1),
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "verbose": -1,
    }

    tree_model = lgb.train(
        tree_params,
        train_dataset,
        num_boost_round=350,
        valid_sets=[valid_dataset],
        callbacks=[lgb.early_stopping(35, verbose=False)],
    )

    tree_prob_valid = tree_model.predict(
        x_tree_valid,
        num_iteration=tree_model.best_iteration,
    )
    tree_valid = clipped_logit(tree_prob_valid).astype(np.float32)
    tree_metrics = evaluate(valid.user_id, valid.y, tree_valid)
    candidate_scores["tree_only"] = float(tree_metrics["primary"])

    del tree_prob_valid
    del x_tree_train, x_tree_valid
    del train_dataset, valid_dataset
    gc.collect()

    incumbent_pct = within_user_percentile(
        valid.user_id, incumbent_valid
    )
    tree_pct = within_user_percentile(valid.user_id, tree_valid)

    incumbent_gauss = ndtri(
        np.clip(incumbent_pct.astype(np.float64), 1e-5, 1.0 - 1e-5)
    ).astype(np.float32)
    tree_gauss = ndtri(
        np.clip(tree_pct.astype(np.float64), 1e-5, 1.0 - 1e-5)
    ).astype(np.float32)

    best_metrics = None
    best_mode = None
    best_weight = None
    best_name = None

    for weight in RAW_TREE_WEIGHTS:
        score = make_combination(
            "raw",
            weight,
            incumbent_valid,
            tree_valid,
        )
        metrics = evaluate(valid.user_id, valid.y, score)
        name = "raw_" + str(weight).replace(".", "")
        candidate_scores[name] = float(metrics["primary"])

        if (
            best_metrics is None
            or float(metrics["primary"])
            > float(best_metrics["primary"])
        ):
            best_metrics = metrics
            best_mode = "raw"
            best_weight = weight
            best_name = name

        del score

    for mode in ["borda", "gaussian"]:
        for weight in RANK_TREE_WEIGHTS:
            score = make_combination(
                mode,
                weight,
                incumbent_valid,
                tree_valid,
                incumbent_pct=incumbent_pct,
                tree_pct=tree_pct,
                incumbent_gauss=incumbent_gauss,
                tree_gauss=tree_gauss,
            )
            metrics = evaluate(valid.user_id, valid.y, score)
            name = (
                mode + "_" + str(weight).replace(".", "")
            )
            candidate_scores[name] = float(metrics["primary"])

            if float(metrics["primary"]) > float(
                best_metrics["primary"]
            ):
                best_metrics = metrics
                best_mode = mode
                best_weight = weight
                best_name = name

            del score

    rank_disagreement = float(
        np.mean(np.abs(incumbent_pct - tree_pct))
    )

    print(
        "FINDINGS "
        + json.dumps(
            {
                "best_fm_epoch": best_fm_epoch,
                "best_fm_primary": best_fm_primary,
                "best_author_coefficient": best_author_coefficient,
                "tree_best_iteration": int(
                    tree_model.best_iteration
                ),
                "tree_primary": float(tree_metrics["primary"]),
                "mean_absolute_rank_disagreement": rank_disagreement,
                "selected": best_name,
                "selected_mode": best_mode,
                "selected_tree_weight": best_weight,
                "pair_coverage": coverage,
            },
            separators=(",", ":"),
        )
    )
    print(
        "CANDIDATES "
        + json.dumps(candidate_scores, separators=(",", ":"))
    )

    del incumbent_pct, tree_pct
    del incumbent_gauss, tree_gauss
    del incumbent_valid, tree_valid
    del valid
    del train
    gc.collect()

    te = load("test")
    x_test_fm = fm_matrix(te)
    base_test = predict_fm(model, x_test_fm, intercept)
    del x_test_fm

    incumbent_test, _, _ = build_reranked_score(
        te,
        base_test,
        pair_stats,
        author_coefficient=best_author_coefficient,
        collect_coverage=False,
    )
    del base_test
    gc.collect()

    x_tree_test = tree_matrix(te)
    tree_prob_test = tree_model.predict(
        x_tree_test,
        num_iteration=tree_model.best_iteration,
    )
    tree_test = clipped_logit(tree_prob_test).astype(np.float32)
    del tree_prob_test, x_tree_test
    gc.collect()

    if best_mode == "raw":
        test_scores = make_combination(
            "raw",
            best_weight,
            incumbent_test,
            tree_test,
        )
    else:
        incumbent_test_pct = within_user_percentile(
            te.user_id, incumbent_test
        )
        tree_test_pct = within_user_percentile(
            te.user_id, tree_test
        )

        if best_mode == "borda":
            test_scores = make_combination(
                "borda",
                best_weight,
                incumbent_test,
                tree_test,
                incumbent_pct=incumbent_test_pct,
                tree_pct=tree_test_pct,
            )
        else:
            incumbent_test_gauss = ndtri(
                np.clip(
                    incumbent_test_pct.astype(np.float64),
                    1e-5,
                    1.0 - 1e-5,
                )
            ).astype(np.float32)
            tree_test_gauss = ndtri(
                np.clip(
                    tree_test_pct.astype(np.float64),
                    1e-5,
                    1.0 - 1e-5,
                )
            ).astype(np.float32)

            test_scores = make_combination(
                "gaussian",
                best_weight,
                incumbent_test,
                tree_test,
                incumbent_gauss=incumbent_test_gauss,
                tree_gauss=tree_test_gauss,
            )

    out = os.environ.get("ITER_OUT")
    if out:
        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    final_output = {
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": 0.0,
    }
    print("METRICS " + json.dumps(final_output, separators=(",", ":")))


if __name__ == "__main__":
    main()