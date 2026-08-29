import gc
import json
import os

import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F

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
AUX_LR = 0.0005
FM_EPOCHS = 10
AUX_EPOCHS = 3
AUX_WEIGHT = 0.20
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


class MultiTaskFactorizationMachine(torch.nn.Module):
    def __init__(self, cardinalities, k):
        super().__init__()
        offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        self.embedding = torch.nn.Embedding(
            int(sum(cardinalities)), k + 2, sparse=True
        )
        with torch.no_grad():
            self.embedding.weight[:, :2].zero_()
            self.embedding.weight[:, 2:].normal_(mean=0.0, std=0.01)
            self.embedding.weight[self.offsets].zero_()

    def components(self, x):
        embedded = self.embedding(x + self.offsets)
        long_linear = embedded[:, :, 0].sum(dim=1)
        click_linear = embedded[:, :, 1].sum(dim=1)
        factors = embedded[:, :, 2:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return long_linear, click_linear, interaction

    def forward_long(self, x):
        long_linear, _, interaction = self.components(x)
        return long_linear + interaction

    def forward_both(self, x):
        long_linear, click_linear, interaction = self.components(x)
        return long_linear + interaction, click_linear + interaction


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
            model.forward_long(xb)
            .add(intercept)
            .cpu()
            .numpy()
            .astype(np.float32)
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


def binary_association(y, z):
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    yc = y - y.mean()
    zc = z - z.mean()
    denominator = np.sqrt(np.dot(yc, yc) * np.dot(zc, zc))
    if denominator <= 0:
        return 0.0
    return float(np.dot(yc, zc) / denominator)


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
        out[matched_rows] = stored_residual[
            bounded_positions[matched]
        ]
    return out


def add_reranker(base, residuals, author_coefficient):
    score = np.asarray(base, dtype=np.float32).copy()
    score += np.float32(0.50) * residuals["tag"]
    score += np.float32(0.40) * residuals["duration_bucket"]
    score += np.float32(0.40) * residuals["upload_type"]
    score += np.float32(0.30) * residuals["tab"]
    score += np.float32(0.30) * residuals["music_type"]
    score += np.float32(0.20) * residuals["hour"]
    if author_coefficient != 0.0:
        score += (
            np.float32(author_coefficient) * residuals["author_id"]
        )
    return score


def main():
    train = load("train")
    valid = load("valid")

    x_train = fm_matrix(train)
    x_valid = fm_matrix(valid)
    y_train = np.asarray(train.y, dtype=np.float32)
    click_train = np.asarray(train.aux["is_click"], dtype=np.float32)

    long_rate = float(np.clip(y_train.mean(), EPS, 1.0 - EPS))
    click_rate = float(np.clip(click_train.mean(), EPS, 1.0 - EPS))
    long_intercept = float(np.log(long_rate / (1.0 - long_rate)))
    click_intercept = float(np.log(click_rate / (1.0 - click_rate)))

    click_correlation = binary_association(y_train, click_train)

    cardinalities = [
        int(FEATURE_CARDINALITIES[name]) for name in FM_FIELDS
    ]
    model = MultiTaskFactorizationMachine(cardinalities, K)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)

    n_train = x_train.shape[0]
    best_fm_primary = -np.inf
    best_fm_epoch = -1
    standard_weight = None

    for epoch in range(FM_EPOCHS):
        model.train()
        order = np.random.permutation(n_train)

        for start in range(0, n_train, TRAIN_BATCH_SIZE):
            idx = order[start:min(start + TRAIN_BATCH_SIZE, n_train)]
            xb = torch.from_numpy(x_train[idx])
            yb = torch.from_numpy(y_train[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_long(xb) + long_intercept
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

        del order

        epoch_scores = predict_fm(model, x_valid, long_intercept)
        epoch_metrics = evaluate(valid.user_id, valid.y, epoch_scores)
        epoch_primary = float(epoch_metrics["primary"])

        if epoch_primary > best_fm_primary:
            best_fm_primary = epoch_primary
            best_fm_epoch = epoch + 1
            standard_weight = model.embedding.weight.detach().clone()

        del epoch_scores

    if standard_weight is None:
        raise RuntimeError("Standard FM training produced no checkpoint")

    del optimizer
    gc.collect()

    with torch.no_grad():
        model.embedding.weight.copy_(standard_weight)

    standard_valid = predict_fm(model, x_valid, long_intercept)
    standard_raw_metrics = evaluate(
        valid.user_id, valid.y, standard_valid
    )

    aux_optimizer = torch.optim.SparseAdam(
        model.parameters(), lr=AUX_LR
    )

    best_aux_primary = -np.inf
    best_aux_epoch = 0
    best_aux_weight = None
    best_aux_valid = None
    aux_epoch_primaries = []

    for epoch in range(AUX_EPOCHS):
        model.train()
        order = np.random.permutation(n_train)

        for start in range(0, n_train, TRAIN_BATCH_SIZE):
            idx = order[start:min(start + TRAIN_BATCH_SIZE, n_train)]
            xb = torch.from_numpy(x_train[idx])
            yb = torch.from_numpy(y_train[idx])
            cb = torch.from_numpy(click_train[idx])

            aux_optimizer.zero_grad(set_to_none=True)
            long_logits, click_logits = model.forward_both(xb)
            long_loss = F.binary_cross_entropy_with_logits(
                long_logits + long_intercept, yb
            )
            click_loss = F.binary_cross_entropy_with_logits(
                click_logits + click_intercept, cb
            )
            loss = long_loss + AUX_WEIGHT * click_loss
            loss.backward()
            aux_optimizer.step()

        del order

        aux_valid = predict_fm(model, x_valid, long_intercept)
        aux_metrics = evaluate(valid.user_id, valid.y, aux_valid)
        aux_primary = float(aux_metrics["primary"])
        aux_epoch_primaries.append(aux_primary)

        if aux_primary > best_aux_primary:
            best_aux_primary = aux_primary
            best_aux_epoch = epoch + 1
            best_aux_weight = model.embedding.weight.detach().clone()
            if best_aux_valid is not None:
                del best_aux_valid
            best_aux_valid = aux_valid
        else:
            del aux_valid

    del aux_optimizer, x_train, y_train, click_train
    gc.collect()

    pair_stats = {}
    valid_residuals = {}
    coverage = {}

    for field in PAIR_FIELDS:
        stats = fit_pair_statistics(
            train, field, PAIR_ALPHAS[field]
        )
        pair_stats[field] = stats
        residual = apply_pair_statistics(valid, field, stats)
        valid_residuals[field] = residual
        coverage[field] = float(np.mean(residual != 0.0))
        gc.collect()

    candidate_metrics = {
        "standard_raw": compact_metrics(standard_raw_metrics)
    }

    standard_reranked_candidates = {}
    best_author_coefficient = 0.0
    best_standard_reranked = None
    best_standard_reranked_metrics = None

    for author_coefficient in [0.0, 0.25, 0.50, 0.75, 1.0]:
        score = add_reranker(
            standard_valid, valid_residuals, author_coefficient
        )
        metrics = evaluate(valid.user_id, valid.y, score)
        name = "standard_author_" + str(author_coefficient).replace(".", "")
        candidate_metrics[name] = compact_metrics(metrics)
        standard_reranked_candidates[author_coefficient] = (
            score,
            metrics,
        )

        if (
            best_standard_reranked_metrics is None
            or float(metrics["primary"])
            > float(best_standard_reranked_metrics["primary"])
        ):
            best_author_coefficient = author_coefficient
            best_standard_reranked = score
            best_standard_reranked_metrics = metrics

    for coefficient, (score, _) in list(
        standard_reranked_candidates.items()
    ):
        if score is not best_standard_reranked:
            del score
    del standard_reranked_candidates

    aux_reranked = add_reranker(
        best_aux_valid, valid_residuals, best_author_coefficient
    )
    aux_reranked_metrics = evaluate(
        valid.user_id, valid.y, aux_reranked
    )
    candidate_metrics["aux_raw"] = compact_metrics(
        evaluate(valid.user_id, valid.y, best_aux_valid)
    )
    candidate_metrics["aux_reranked"] = compact_metrics(
        aux_reranked_metrics
    )

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
    tree_logit_valid = clipped_logit(
        tree_prob_valid
    ).astype(np.float32)

    tree_metrics = evaluate(
        valid.user_id, valid.y, tree_logit_valid
    )
    candidate_metrics["tree_only"] = compact_metrics(tree_metrics)

    model_candidates = {
        "standard": (
            best_standard_reranked,
            standard_weight,
            best_standard_reranked_metrics,
        ),
        "multitask": (
            aux_reranked,
            best_aux_weight,
            aux_reranked_metrics,
        ),
    }

    best_name = None
    best_metrics = None
    best_tree_weight = 0.0
    best_model_name = None
    best_valid_score = None

    for model_name, (base_score, _, base_metrics) in model_candidates.items():
        base_name = model_name + "_blend_00"
        candidate_metrics[base_name] = compact_metrics(base_metrics)

        if (
            best_metrics is None
            or float(base_metrics["primary"])
            > float(best_metrics["primary"])
        ):
            best_name = base_name
            best_metrics = base_metrics
            best_tree_weight = 0.0
            best_model_name = model_name
            best_valid_score = base_score

        for weight in [0.20, 0.30, 0.40, 0.50, 0.60]:
            score = base_score + np.float32(weight) * tree_logit_valid
            metrics = evaluate(valid.user_id, valid.y, score)
            name = (
                model_name
                + "_blend_"
                + str(weight).replace(".", "")
            )
            candidate_metrics[name] = compact_metrics(metrics)

            if float(metrics["primary"]) > float(best_metrics["primary"]):
                best_name = name
                best_metrics = metrics
                best_tree_weight = weight
                best_model_name = model_name
                best_valid_score = score
            else:
                del score

    selected_weight = (
        standard_weight
        if best_model_name == "standard"
        else best_aux_weight
    )

    print(
        "FINDINGS "
        + json.dumps(
            {
                "click_rate": click_rate,
                "long_view_click_correlation": click_correlation,
                "best_fm_epoch": best_fm_epoch,
                "best_fm_primary": best_fm_primary,
                "aux_weight": AUX_WEIGHT,
                "aux_epoch_primaries": aux_epoch_primaries,
                "best_aux_epoch": best_aux_epoch,
                "best_aux_raw_primary": best_aux_primary,
                "best_author_coefficient": best_author_coefficient,
                "tree_best_iteration": int(tree_model.best_iteration),
                "tree_primary": float(tree_metrics["primary"]),
                "selected": best_name,
                "selected_model": best_model_name,
                "selected_tree_weight": best_tree_weight,
                "pair_coverage": coverage,
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

    del x_tree_train, x_tree_valid
    del train_dataset, valid_dataset
    del tree_prob_valid
    del standard_valid, best_aux_valid
    del best_standard_reranked, aux_reranked
    del best_valid_score, tree_logit_valid
    del valid_residuals, valid
    gc.collect()

    with torch.no_grad():
        model.embedding.weight.copy_(selected_weight)

    del standard_weight, best_aux_weight, selected_weight
    gc.collect()

    out = os.environ.get("ITER_OUT")
    if out:
        test = load("test")
        x_test_fm = fm_matrix(test)
        base_test = predict_fm(model, x_test_fm, long_intercept)
        del x_test_fm

        test_residuals = {}
        for field in PAIR_FIELDS:
            test_residuals[field] = apply_pair_statistics(
                test, field, pair_stats[field]
            )

        test_score = add_reranker(
            base_test,
            test_residuals,
            best_author_coefficient,
        )

        if best_tree_weight != 0.0:
            x_tree_test = tree_matrix(test)
            tree_prob_test = tree_model.predict(
                x_tree_test,
                num_iteration=tree_model.best_iteration,
            )
            tree_logit_test = clipped_logit(
                tree_prob_test
            ).astype(np.float32)
            test_score += (
                np.float32(best_tree_weight) * tree_logit_test
            )
            del x_tree_test, tree_prob_test, tree_logit_test

        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_score, dtype=np.float64),
        )

    final = {
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": 0.0,
    }
    print("METRICS " + json.dumps(final, separators=(",", ":")))


if __name__ == "__main__":
    main()