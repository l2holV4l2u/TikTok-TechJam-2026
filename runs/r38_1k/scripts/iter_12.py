import gc
import json
import os

import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import ndtri

from pipeline.data import FEATURE_CARDINALITIES, load
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

PAIR_WEIGHTS = {
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
FM_CHECKPOINT_EPOCHS = {8, 10}
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
        embedded = self.embedding(x + self.offsets)
        linear = embedded[:, :, 0].sum(dim=1)
        factors = embedded[:, :, 1:]
        factor_sum = factors.sum(dim=1)
        interaction = 0.5 * (
            factor_sum.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return linear + interaction


def fm_matrix(split):
    return np.column_stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in FM_FIELDS]
    )


def tree_matrix(split):
    return np.column_stack(
        [np.asarray(split.X[name], dtype=np.int32) for name in TREE_FIELDS]
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
        inverse, weights=y, minlength=unique_keys.size
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
        residual = stats["residual"]
        out = np.zeros(keys.size, dtype=np.float32)
        valid = (keys >= 0) & (keys < residual.size)
        out[valid] = residual[keys[valid]]
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
        out[matched_rows] = stored_residual[bounded_positions[matched]]

    return out


def make_incumbent(split, fm_scores, pair_stats, author_coefficient):
    score = np.asarray(fm_scores, dtype=np.float32).copy()

    for field, weight in PAIR_WEIGHTS.items():
        residual = apply_pair_statistics(split, field, pair_stats[field])
        score += np.float32(weight) * residual
        del residual

    if author_coefficient != 0.0:
        residual = apply_pair_statistics(
            split, "author_id", pair_stats["author_id"]
        )
        score += np.float32(author_coefficient) * residual
        del residual

    return score


def within_user_quantile(user_ids, scores):
    """
    Fully vectorized ascending within-user ranks.

    q=(rank+0.5)/group_size is monotone in the original score and maps every
    user's score distribution to the same scale. lexsort performs the only
    global sorting operation; there is no loop over users or rows.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = values.size

    order = np.lexsort((values, users))
    sorted_users = users[order]

    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_users[1:] != sorted_users[:-1]

    group_start_marker = np.zeros(n, dtype=np.int64)
    boundary_positions = np.flatnonzero(boundary)
    group_start_marker[boundary_positions] = boundary_positions
    group_start = np.maximum.accumulate(group_start_marker)

    group_end_marker = np.empty(n, dtype=np.int64)
    group_end_marker.fill(n)
    if boundary_positions.size > 1:
        group_end_marker[boundary_positions[:-1]] = boundary_positions[1:]
    group_end = np.minimum.accumulate(group_end_marker[::-1])[::-1]

    position = np.arange(n, dtype=np.int64) - group_start
    group_size = group_end - group_start

    sorted_quantile = (
        position.astype(np.float64) + 0.5
    ) / group_size.astype(np.float64)

    quantile = np.empty(n, dtype=np.float32)
    quantile[order] = sorted_quantile.astype(np.float32)

    del order, sorted_users, boundary
    del boundary_positions, group_start_marker, group_start
    del group_end_marker, group_end, position, group_size, sorted_quantile
    return quantile


def combine_rank_scores(q_incumbent, q_tree, method, tree_weight):
    w = np.float32(tree_weight)

    if method == "borda":
        return (
            np.float32(1.0 - tree_weight) * q_incumbent
            + w * q_tree
        ).astype(np.float32)

    if method == "gaussian":
        inc = ndtri(
            np.clip(q_incumbent.astype(np.float64), 1e-6, 1.0 - 1e-6)
        )
        tree = ndtri(
            np.clip(q_tree.astype(np.float64), 1e-6, 1.0 - 1e-6)
        )
        out = (
            (1.0 - float(tree_weight)) * inc
            + float(tree_weight) * tree
        ).astype(np.float32)
        del inc, tree
        return out

    raise ValueError("Unknown rank aggregation method: " + str(method))


def train_fm(train, valid):
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

    best_primary = -np.inf
    best_epoch = -1
    best_weight = None
    checkpoint_scores = {}

    n_train = x_train.shape[0]

    for epoch in range(1, FM_EPOCHS + 1):
        model.train()
        order = np.random.permutation(n_train)

        for start in range(0, n_train, TRAIN_BATCH_SIZE):
            idx = order[start:min(start + TRAIN_BATCH_SIZE, n_train)]
            xb = torch.from_numpy(x_train[idx])
            yb = torch.from_numpy(y_train[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb) + intercept
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

        del order

        if epoch in FM_CHECKPOINT_EPOCHS:
            scores = predict_fm(model, x_valid, intercept)
            metrics = evaluate(valid.user_id, valid.y, scores)
            primary = float(metrics["primary"])
            checkpoint_scores[str(epoch)] = primary

            if primary > best_primary:
                best_primary = primary
                best_epoch = epoch
                best_weight = model.embedding.weight.detach().clone()

            del scores

    if best_weight is None:
        raise RuntimeError("FM training produced no checkpoint")

    with torch.no_grad():
        model.embedding.weight.copy_(best_weight)

    valid_scores = predict_fm(model, x_valid, intercept)

    del optimizer, best_weight, x_train, x_valid, y_train
    gc.collect()

    return (
        model,
        intercept,
        valid_scores,
        best_epoch,
        best_primary,
        checkpoint_scores,
    )


def train_tree(train, valid):
    x_train = tree_matrix(train)
    x_valid = tree_matrix(valid)

    train_dataset = lgb.Dataset(
        x_train,
        label=np.asarray(train.y, dtype=np.float32),
        categorical_feature=list(range(len(TREE_FIELDS))),
        free_raw_data=True,
    )
    valid_dataset = lgb.Dataset(
        x_valid,
        label=np.asarray(valid.y, dtype=np.float32),
        categorical_feature=list(range(len(TREE_FIELDS))),
        reference=train_dataset,
        free_raw_data=True,
    )

    params = {
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

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=350,
        valid_sets=[valid_dataset],
        callbacks=[lgb.early_stopping(35, verbose=False)],
    )

    probability = model.predict(
        x_valid, num_iteration=model.best_iteration
    )
    valid_logit = clipped_logit(probability).astype(np.float32)

    del probability, x_train, x_valid, train_dataset, valid_dataset
    gc.collect()

    return model, valid_logit


def main():
    train = load("train")
    valid = load("valid")

    (
        fm_model,
        intercept,
        fm_valid,
        best_fm_epoch,
        best_fm_primary,
        fm_checkpoints,
    ) = train_fm(train, valid)

    pair_stats = {}
    pair_coverage = {}

    for field in PAIR_FIELDS:
        stats = fit_pair_statistics(train, field, PAIR_ALPHAS[field])
        pair_stats[field] = stats
        residual = apply_pair_statistics(valid, field, stats)
        pair_coverage[field] = float(np.mean(residual != 0.0))
        del residual
        gc.collect()

    incumbent_zero = make_incumbent(
        valid, fm_valid, pair_stats, author_coefficient=0.0
    )
    metrics_author_zero = evaluate(
        valid.user_id, valid.y, incumbent_zero
    )

    incumbent_author = make_incumbent(
        valid, fm_valid, pair_stats, author_coefficient=0.25
    )
    metrics_author_quarter = evaluate(
        valid.user_id, valid.y, incumbent_author
    )

    if (
        float(metrics_author_quarter["primary"])
        > float(metrics_author_zero["primary"])
    ):
        incumbent_valid = incumbent_author
        author_coefficient = 0.25
        incumbent_metrics = metrics_author_quarter
        del incumbent_zero
    else:
        incumbent_valid = incumbent_zero
        author_coefficient = 0.0
        incumbent_metrics = metrics_author_zero
        del incumbent_author

    tree_model, tree_valid = train_tree(train, valid)
    tree_metrics = evaluate(valid.user_id, valid.y, tree_valid)

    raw_valid = (
        incumbent_valid + np.float32(0.50) * tree_valid
    ).astype(np.float32)
    raw_metrics = evaluate(valid.user_id, valid.y, raw_valid)

    q_incumbent = within_user_quantile(
        valid.user_id, incumbent_valid
    )
    q_tree = within_user_quantile(valid.user_id, tree_valid)

    candidate_metrics = {
        "incumbent": compact_metrics(incumbent_metrics),
        "tree_only": compact_metrics(tree_metrics),
        "raw_logit_050": compact_metrics(raw_metrics),
    }

    best_name = "raw_logit_050"
    best_metrics = raw_metrics
    best_method = "raw"
    best_rank_weight = 0.50
    best_valid_score = raw_valid

    rank_candidates = [
        ("borda_035", "borda", 0.35),
        ("borda_050", "borda", 0.50),
        ("gaussian_050", "gaussian", 0.50),
    ]

    for name, method, weight in rank_candidates:
        score = combine_rank_scores(
            q_incumbent, q_tree, method, weight
        )
        metrics = evaluate(valid.user_id, valid.y, score)
        candidate_metrics[name] = compact_metrics(metrics)

        if float(metrics["primary"]) > float(best_metrics["primary"]):
            if best_valid_score is not raw_valid:
                del best_valid_score
            best_name = name
            best_metrics = metrics
            best_method = method
            best_rank_weight = weight
            best_valid_score = score
        else:
            del score

    print(
        "FINDINGS "
        + json.dumps(
            {
                "best_fm_epoch": int(best_fm_epoch),
                "best_fm_primary": float(best_fm_primary),
                "fm_checkpoint_primary": fm_checkpoints,
                "author_coefficient": float(author_coefficient),
                "tree_best_iteration": int(tree_model.best_iteration),
                "tree_primary": float(tree_metrics["primary"]),
                "selected": best_name,
                "selected_method": best_method,
                "selected_tree_weight": float(best_rank_weight),
                "pair_coverage": pair_coverage,
            },
            separators=(",", ":"),
        )
    )
    print(
        "CANDIDATES "
        + json.dumps(
            {
                name: float(values["primary"])
                for name, values in candidate_metrics.items()
            },
            separators=(",", ":"),
        )
    )

    del q_incumbent, q_tree, best_valid_score, raw_valid
    del incumbent_valid, tree_valid, fm_valid
    del valid
    gc.collect()

    # All model and combiner choices are now fixed using validation only.
    del train
    gc.collect()

    out = os.environ.get("ITER_OUT")
    if out:
        test = load("test")

        x_test_fm = fm_matrix(test)
        fm_test = predict_fm(fm_model, x_test_fm, intercept)
        del x_test_fm

        incumbent_test = make_incumbent(
            test,
            fm_test,
            pair_stats,
            author_coefficient=author_coefficient,
        )
        del fm_test

        x_test_tree = tree_matrix(test)
        tree_probability_test = tree_model.predict(
            x_test_tree,
            num_iteration=tree_model.best_iteration,
        )
        tree_test = clipped_logit(
            tree_probability_test
        ).astype(np.float32)
        del x_test_tree, tree_probability_test

        if best_method == "raw":
            test_scores = (
                incumbent_test + np.float32(0.50) * tree_test
            ).astype(np.float32)
        else:
            q_inc_test = within_user_quantile(
                test.user_id, incumbent_test
            )
            q_tree_test = within_user_quantile(
                test.user_id, tree_test
            )
            test_scores = combine_rank_scores(
                q_inc_test,
                q_tree_test,
                best_method,
                best_rank_weight,
            )
            del q_inc_test, q_tree_test

        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

        del test_scores, incumbent_test, tree_test, test
        gc.collect()

    final = {
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": 0.0,
    }
    print("METRICS " + json.dumps(final, separators=(",", ":")))


if __name__ == "__main__":
    main()