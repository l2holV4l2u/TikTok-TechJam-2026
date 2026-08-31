import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2022
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
CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
BATCH_SIZE = 4096
FM_EPOCHS = 5
FM_DIM = 16
FM_LR = 0.001

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


def make_matrix(split):
    return np.stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS],
        axis=1,
    )


def standardize_scores(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        return x - x.mean()
    return (x - x.mean()) / sd


class ExpandedFM(nn.Module):
    def __init__(self):
        super().__init__()
        offsets = np.cumsum([0] + CARDINALITIES[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        total = int(sum(CARDINALITIES))
        self.linear = nn.Embedding(total, 1)
        self.embedding = nn.Embedding(total, FM_DIM)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        z = x + self.offsets
        linear = self.linear(z).sum(dim=1).squeeze(1)
        factors = self.embedding(z)
        interactions = 0.5 * (
            factors.sum(dim=1).square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interactions


def fit_fm(x_np, y_np, seed):
    torch.manual_seed(seed)
    model = ExpandedFM()
    optimizer = torch.optim.Adam(model.parameters(), lr=FM_LR)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(np.ascontiguousarray(x_np))
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    n = len(y_np)
    generator = torch.Generator()
    generator.manual_seed(seed)

    model.train()
    for _ in range(FM_EPOCHS):
        order = torch.randperm(n, generator=generator)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:lo + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
    return model


def predict_fm(model, x_np):
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(x_np))
    out = np.empty(len(x_np), dtype=np.float64)
    with torch.inference_mode():
        for lo in range(0, len(x_np), BATCH_SIZE * 2):
            hi = min(len(x_np), lo + BATCH_SIZE * 2)
            out[lo:hi] = model(x[lo:hi]).cpu().numpy().astype(np.float64)
    return out


def fit_lgbm(x_np, y_np):
    dataset = lgb.Dataset(
        np.asarray(x_np, dtype=np.int32),
        label=np.asarray(y_np, dtype=np.float32),
        categorical_feature=list(range(len(FIELDS))),
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.06,
        "num_leaves": 63,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_cat_to_onehot": 16,
        "cat_smooth": 30.0,
        "cat_l2": 10.0,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "num_threads": max(1, min(16, os.cpu_count() or 1)),
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=260)


EB_ALPHAS = np.asarray(
    [20.0, 80.0, 80.0, 400.0, 250.0, 250.0, 300.0, 300.0, 400.0],
    dtype=np.float64,
)


def entity_statistics(x_np, y_np):
    prior = float(np.mean(y_np))
    stats = []
    y64 = np.asarray(y_np, dtype=np.float64)
    for j, card in enumerate(CARDINALITIES):
        ids = x_np[:, j]
        counts = np.bincount(ids, minlength=card).astype(np.float64)
        sums = np.bincount(ids, weights=y64, minlength=card).astype(np.float64)
        stats.append((counts, sums))
    return prior, stats


def target_features(x_np, prior, stats, y_loo=None):
    n, p = x_np.shape
    result = np.empty((n, p), dtype=np.float32)
    for j, (counts, sums) in enumerate(stats):
        ids = x_np[:, j]
        alpha = EB_ALPHAS[j]
        if y_loo is None:
            numer = sums[ids] + alpha * prior
            denom = counts[ids] + alpha
        else:
            numer = sums[ids] - y_loo + alpha * prior
            denom = counts[ids] - 1.0 + alpha
        result[:, j] = (numer / denom).astype(np.float32)
    return result


def fit_empirical_bayes(x_np, y_np):
    y = np.asarray(y_np, dtype=np.float64)
    prior, stats = entity_statistics(x_np, y)
    features = target_features(x_np, prior, stats, y_loo=y)

    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale = np.maximum(scale, 1e-4)
    z = (features.astype(np.float64) - mean) / scale

    design = np.empty((len(y), len(FIELDS) + 1), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1:] = z

    beta = np.zeros(design.shape[1], dtype=np.float64)
    beta[0] = np.log((prior + 1e-6) / (1.0 - prior + 1e-6))
    ridge = np.diag(np.r_[0.01, np.full(len(FIELDS), 1.0)])

    for _ in range(10):
        logits = np.clip(design @ beta, -20.0, 20.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        weight = np.maximum(prob * (1.0 - prob), 1e-5)
        gradient = design.T @ (prob - y) + ridge @ beta
        hessian = design.T @ (design * weight[:, None]) + ridge
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-6:
            break

    del features, z, design
    return {
        "prior": prior,
        "stats": stats,
        "mean": mean,
        "scale": scale,
        "beta": beta,
    }


def predict_empirical_bayes(model, x_np):
    features = target_features(
        x_np, model["prior"], model["stats"], y_loo=None
    ).astype(np.float64)
    z = (features - model["mean"]) / model["scale"]
    return model["beta"][0] + z @ model["beta"][1:]


def evaluate_primary(valid, scores):
    return evaluate(valid.user_id, valid.y, scores)


train = load("train")
valid = load("valid")
x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)

family_valid = {}

fm_model = fit_fm(x_train, y_train, SEED)
family_valid["expanded_fm"] = predict_fm(fm_model, x_valid)
del fm_model
gc.collect()

lgb_model = fit_lgbm(x_train, y_train)
family_valid["lightgbm_binary"] = lgb_model.predict(
    np.asarray(x_valid, dtype=np.int32)
).astype(np.float64)
del lgb_model
gc.collect()

eb_model = fit_empirical_bayes(x_train, y_train)
family_valid["empirical_bayes"] = predict_empirical_bayes(eb_model, x_valid)
del eb_model
gc.collect()

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")
have_incumbent = os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)

candidate_scores = {}
candidate_metrics = {}
best_name = None
best_scores = None
best_metrics = None
best_family = None
best_weight = None

for family_name, scores in family_valid.items():
    met = evaluate_primary(valid, scores)
    candidate_scores[family_name] = float(met["primary"])
    candidate_metrics[family_name] = met
    if best_metrics is None or met["primary"] > best_metrics["primary"]:
        best_name = family_name
        best_scores = scores.copy()
        best_metrics = met
        best_family = family_name
        best_weight = None

if have_incumbent:
    incumbent_valid = np.asarray(
        np.load(inc_valid_path), dtype=np.float64
    )
    inc_z = standardize_scores(incumbent_valid)

    blend_weights = np.arange(0.15, 0.91, 0.05)
    for family_name, scores in family_valid.items():
        own_z = standardize_scores(scores)
        family_best_met = None
        family_best_scores = None
        family_best_weight = None

        for weight in blend_weights:
            # weight is the contribution of the new family.
            blend = weight * own_z + (1.0 - weight) * inc_z
            met = evaluate_primary(valid, blend)
            if (
                family_best_met is None
                or met["primary"] > family_best_met["primary"]
            ):
                family_best_met = met
                family_best_scores = blend.copy()
                family_best_weight = float(weight)

        blend_name = family_name + "_incumbent_blend"
        candidate_scores[blend_name] = float(family_best_met["primary"])
        candidate_metrics[blend_name] = family_best_met

        if family_best_met["primary"] > best_metrics["primary"]:
            best_name = blend_name
            best_scores = family_best_scores
            best_metrics = family_best_met
            best_family = family_name
            best_weight = family_best_weight

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS " + json.dumps(
        {
            "winner": best_name,
            "new_family_weight": best_weight,
            "fields": FIELDS,
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_weight is not None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(family_valid[best_family], dtype=np.float64),
        )

# Refit the selected family on train + validation using the identical recipe.
y_valid = np.asarray(valid.y, dtype=np.float32)
x_combined = np.concatenate([x_train, x_valid], axis=0)
y_combined = np.concatenate([y_train, y_valid], axis=0)

test = load("test")
x_test = make_matrix(test)

if best_family == "expanded_fm":
    final_model = fit_fm(x_combined, y_combined, SEED)
    own_test_scores = predict_fm(final_model, x_test)
    del final_model
elif best_family == "lightgbm_binary":
    final_model = fit_lgbm(x_combined, y_combined)
    own_test_scores = final_model.predict(
        np.asarray(x_test, dtype=np.int32)
    ).astype(np.float64)
    del final_model
elif best_family == "empirical_bayes":
    final_model = fit_empirical_bayes(x_combined, y_combined)
    own_test_scores = predict_empirical_bayes(final_model, x_test)
    del final_model
else:
    raise RuntimeError("Unknown selected family: " + str(best_family))

if best_weight is None:
    test_scores = own_test_scores
else:
    incumbent_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    test_scores = (
        best_weight * standardize_scores(own_test_scores)
        + (1.0 - best_weight) * standardize_scores(incumbent_test)
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)