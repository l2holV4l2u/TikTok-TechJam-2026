import os
import time
import math
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2025
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
BATCH_SIZE = 8192
PRED_BATCH = 32768
FM_EPOCHS = 5
LINEAR_EPOCHS = 4
DEEP_EPOCHS = 4
FM_RANK = 16
DEEP_RANK = 12

torch.set_num_threads(min(16, os.cpu_count() or 1))
torch.manual_seed(SEED)
np.random.seed(SEED)

OFFSETS = {}
offset = 0
for name in FIELDS:
    OFFSETS[name] = offset
    offset += int(FEATURE_CARDINALITIES[name])
TOTAL_CARDINALITY = offset


def make_matrix(split, dtype=np.int64):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS)), dtype=dtype)
    for j, name in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[name], dtype=dtype) + OFFSETS[name]
    return x


def make_lgb_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int32) for name in FIELDS
    ])


def initial_logit(y):
    p = float(np.mean(y))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class AdditiveModel(nn.Module):
    def __init__(self, cardinality, intercept):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, 1, sparse=True)
        self.register_buffer(
            "intercept", torch.tensor(float(intercept), dtype=torch.float32)
        )
        nn.init.zeros_(self.embedding.weight)

    def forward(self, x):
        return self.intercept + self.embedding(x).squeeze(-1).sum(dim=1)


class FMModel(nn.Module):
    def __init__(self, cardinality, rank, intercept):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        self.register_buffer(
            "intercept", torch.tensor(float(intercept), dtype=torch.float32)
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        sv = v.sum(dim=1)
        interaction = 0.5 * (
            sv.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.intercept + linear + interaction


class DeepFMModel(nn.Module):
    def __init__(self, cardinality, n_fields, rank, intercept):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_fields * rank, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.register_buffer(
            "intercept", torch.tensor(float(intercept), dtype=torch.float32)
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        sv = v.sum(dim=1)
        fm = 0.5 * (sv.square() - v.square().sum(dim=1)).sum(dim=1)
        deep = self.mlp(v.reshape(v.shape[0], -1)).squeeze(1)
        return self.intercept + linear + fm + deep


def fit_sparse_model(kind, x_np, y_np, seed):
    torch.manual_seed(seed)
    x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.int64))
    y = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.float32))
    intercept = initial_logit(y_np)

    if kind == "additive":
        model = AdditiveModel(TOTAL_CARDINALITY, intercept)
        epochs = LINEAR_EPOCHS
        sparse_opt = torch.optim.SparseAdam(model.parameters(), lr=0.006)
        dense_opt = None
    elif kind == "fm":
        model = FMModel(TOTAL_CARDINALITY, FM_RANK, intercept)
        epochs = FM_EPOCHS
        sparse_opt = torch.optim.SparseAdam(model.parameters(), lr=0.001)
        dense_opt = None
    elif kind == "deepfm":
        model = DeepFMModel(
            TOTAL_CARDINALITY, len(FIELDS), DEEP_RANK, intercept
        )
        epochs = DEEP_EPOCHS
        sparse_opt = torch.optim.SparseAdam(
            model.embedding.parameters(), lr=0.0015
        )
        dense_opt = torch.optim.Adam(
            model.mlp.parameters(), lr=0.001, weight_decay=1e-6
        )
    else:
        raise ValueError(kind)

    generator = torch.Generator()
    generator.manual_seed(seed + 7919)
    n = x.shape[0]
    model.train()

    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])

            sparse_opt.zero_grad(set_to_none=True)
            if dense_opt is not None:
                dense_opt.zero_grad(set_to_none=True)
            loss.backward()
            sparse_opt.step()
            if dense_opt is not None:
                dense_opt.step()

    return model


@torch.no_grad()
def predict_torch(model, x_np):
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.int64))
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], PRED_BATCH):
        end = min(start + PRED_BATCH, x.shape[0])
        result[start:end] = (
            model(x[start:end]).cpu().numpy().astype(np.float64)
        )
    return result


LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "num_threads": min(16, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "verbose": -1,
}


def fit_lgb(x, y):
    dset = lgb.Dataset(
        x,
        label=np.asarray(y, dtype=np.float32),
        categorical_feature=list(range(len(FIELDS))),
        free_raw_data=True,
    )
    return lgb.train(LGB_PARAMS, dset, num_boost_round=220)


def predict_lgb(model, x):
    p = model.predict(x, num_iteration=model.best_iteration)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def smoothed_logit(sum_y, count, prior, strength):
    rate = (sum_y + strength * prior) / (count + strength)
    rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
    return np.log(rate / (1.0 - rate))


def fit_eb(split, y):
    y64 = np.asarray(y, dtype=np.float64)
    prior = float(y64.mean())

    def aggregate(ids, cardinality):
        ids = np.asarray(ids, dtype=np.int64)
        count = np.bincount(ids, minlength=cardinality).astype(np.float64)
        sums = np.bincount(
            ids, weights=y64, minlength=cardinality
        ).astype(np.float64)
        return count, sums

    cv, sv = aggregate(
        split.X["video_id"], int(FEATURE_CARDINALITIES["video_id"])
    )
    ca, sa = aggregate(
        split.X["author_id"], int(FEATURE_CARDINALITIES["author_id"])
    )
    ct, st = aggregate(
        split.X["tag"], int(FEATURE_CARDINALITIES["tag"])
    )

    tag_card = int(FEATURE_CARDINALITIES["tag"])
    user_card = int(FEATURE_CARDINALITIES["user_id"])
    user_tag = (
        np.asarray(split.X["user_id"], dtype=np.int64) * tag_card
        + np.asarray(split.X["tag"], dtype=np.int64)
    )
    cut, sut = aggregate(user_tag, user_card * tag_card)

    return {
        "prior": prior,
        "video": smoothed_logit(sv, cv, prior, 35.0),
        "author": smoothed_logit(sa, ca, prior, 100.0),
        "tag": smoothed_logit(st, ct, prior, 300.0),
        "user_tag": smoothed_logit(sut, cut, prior, 18.0),
        "tag_card": tag_card,
    }


def predict_eb(model, split):
    base = math.log(model["prior"] / (1.0 - model["prior"]))
    video = model["video"][np.asarray(split.X["video_id"], dtype=np.int64)]
    author = model["author"][np.asarray(split.X["author_id"], dtype=np.int64)]
    tag = model["tag"][np.asarray(split.X["tag"], dtype=np.int64)]
    user_tag_ids = (
        np.asarray(split.X["user_id"], dtype=np.int64) * model["tag_card"]
        + np.asarray(split.X["tag"], dtype=np.int64)
    )
    user_tag = model["user_tag"][user_tag_ids]
    return (
        0.80 * video
        + 0.45 * author
        + 0.30 * tag
        + 0.75 * (user_tag - base)
    ).astype(np.float64)


def fit_family(name, split, x_torch, x_lgb, y, seed):
    if name in ("additive", "fm", "deepfm"):
        return fit_sparse_model(name, x_torch, y, seed)
    if name == "lightgbm":
        return fit_lgb(x_lgb, y)
    if name == "empirical_bayes":
        return fit_eb(split, y)
    raise ValueError(name)


def predict_family(name, model, split, x_torch, x_lgb):
    if name in ("additive", "fm", "deepfm"):
        return predict_torch(model, x_torch)
    if name == "lightgbm":
        return predict_lgb(model, x_lgb)
    if name == "empirical_bayes":
        return predict_eb(model, split)
    raise ValueError(name)


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train_torch = make_matrix(train)
x_valid_torch = make_matrix(valid)
x_train_lgb = make_lgb_matrix(train)
x_valid_lgb = make_lgb_matrix(valid)

artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

families = ["additive", "fm", "deepfm", "lightgbm", "empirical_bayes"]
alphas = [1.0, 0.75, 0.50, 0.25]

candidate_scores = {}
best_primary = -np.inf
best_scores = None
best_family = None
best_alpha = None
best_metrics = None

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metrics["primary"])
best_primary = float(inc_metrics["primary"])
best_scores = inc_valid.copy()
best_family = "incumbent"
best_alpha = 0.0
best_metrics = inc_metrics

for fi, family in enumerate(families):
    model = fit_family(
        family, train, x_train_torch, x_train_lgb, y_train, SEED + 101 * fi
    )
    family_valid = predict_family(
        family, model, valid, x_valid_torch, x_valid_lgb
    )

    for alpha in alphas:
        if alpha == 1.0:
            scores = family_valid
            name = family
        else:
            scores = alpha * family_valid + (1.0 - alpha) * inc_valid
            name = "%s_blend_%.2f" % (family, alpha)

        met = evaluate(valid.user_id, y_valid, scores)
        primary = float(met["primary"])
        candidate_scores[name] = primary
        if primary > best_primary:
            best_primary = primary
            best_scores = np.asarray(scores, dtype=np.float64).copy()
            best_family = family
            best_alpha = alpha
            best_metrics = met

    del model, family_valid
    gc.collect()

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

# Refit only the validation-selected family on train + validation.
test = load("test")

if best_family == "incumbent":
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
else:
    x_joint_torch = np.concatenate(
        (x_train_torch, x_valid_torch), axis=0
    )
    x_joint_lgb = np.concatenate(
        (x_train_lgb, x_valid_lgb), axis=0
    )
    y_joint = np.concatenate(
        (y_train, np.asarray(valid.y, dtype=np.float32)), axis=0
    )

    class JointSplit:
        pass

    joint = JointSplit()
    joint.X = {
        name: np.concatenate(
            (
                np.asarray(train.X[name], dtype=np.int64),
                np.asarray(valid.X[name], dtype=np.int64),
            )
        )
        for name in ["user_id", "video_id", "author_id", "tag"]
    }

    selected_index = families.index(best_family)
    joint_model = fit_family(
        best_family,
        joint,
        x_joint_torch,
        x_joint_lgb,
        y_joint,
        SEED + 101 * selected_index,
    )

    x_test_torch = make_matrix(test)
    x_test_lgb = make_lgb_matrix(test)
    family_test = predict_family(
        best_family, joint_model, test, x_test_torch, x_test_lgb
    )

    if best_alpha < 1.0:
        inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
        test_scores = (
            best_alpha * family_test + (1.0 - best_alpha) * inc_test
        )
    else:
        test_scores = family_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected_family=%s blend_new_weight=%.2f validation_primary=%.6f"
    % (best_family, float(best_alpha), float(best_metrics["primary"]))
)

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)