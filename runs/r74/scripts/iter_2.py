import os
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2026
THREADS = max(1, min(16, os.cpu_count() or 1))
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536

torch.set_num_threads(THREADS)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ARTIFACTS = os.environ.get("RUN_ARTIFACTS", "")
OUT_DIR = os.environ.get("ITER_OUT")
if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)

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

LGB_CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "video_type",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "register_days_bucket",
]
LGB_NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_neural_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) + off
            for f, off in zip(FIELDS, offsets)
        ]),
        dtype=np.int64,
    )


def make_lgb_matrix(split):
    cols = [
        np.asarray(split.X[f], dtype=np.float32)
        for f in LGB_CAT_FIELDS
    ]
    for f in LGB_NUM_FIELDS:
        v = np.asarray(split.num[f], dtype=np.float32)
        v = np.nan_to_num(v, nan=-1.0, posinf=-1.0, neginf=-1.0)
        cols.append(np.log1p(np.maximum(v, 0.0)).astype(np.float32))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


class WideModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(-1)

    def sparse_parameters(self):
        return [self.linear.weight]

    def dense_parameters(self):
        return [self.bias]


class FMModel(nn.Module):
    def __init__(self, n_features, k=16):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.factors = nn.Embedding(n_features, k, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.factors(x)
        summed = v.sum(dim=1)
        pairwise = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + pairwise

    def sparse_parameters(self):
        return [self.linear.weight, self.factors.weight]

    def dense_parameters(self):
        return [self.bias]


class DeepFMModel(nn.Module):
    def __init__(self, n_features, n_fields, k=12):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.factors = nn.Embedding(n_features, k, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(n_fields * k, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.factors(x)
        summed = v.sum(dim=1)
        pairwise = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.mlp(v.reshape(v.shape[0], -1)).squeeze(-1)
        return self.bias + linear + pairwise + deep

    def sparse_parameters(self):
        return [self.linear.weight, self.factors.weight]

    def dense_parameters(self):
        return [self.bias] + list(self.mlp.parameters())


@torch.no_grad()
def torch_predict(model, x):
    model.eval()
    xt = torch.from_numpy(x)
    ans = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x))
        ans[start:end] = model(xt[start:end]).cpu().numpy()
    return ans


def build_optimizers(model, lr):
    sparse_opt = torch.optim.SparseAdam(model.sparse_parameters(), lr=lr)
    dense_opt = torch.optim.Adam(model.dense_parameters(), lr=lr)
    return sparse_opt, dense_opt


def train_torch_candidate(
    model_factory,
    x_train,
    y_train,
    x_valid,
    valid_user,
    y_valid,
    epochs,
    lr,
    seed,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = model_factory()
    sparse_opt, dense_opt = build_optimizers(model, lr)
    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train)
    generator = torch.Generator().manual_seed(seed + 91)

    best_primary = -np.inf
    best_scores = None
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(x_train), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            sparse_opt.zero_grad(set_to_none=True)
            dense_opt.zero_grad(set_to_none=True)
            logits = model(xt[idx])
            loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
            loss.backward()
            sparse_opt.step()
            dense_opt.step()

        pred = torch_predict(model, x_valid)
        metric = evaluate(valid_user, y_valid, pred)
        if metric["primary"] > best_primary:
            best_primary = float(metric["primary"])
            best_scores = pred.copy()
            best_epoch = epoch

    del model, sparse_opt, dense_opt, xt, yt
    gc.collect()
    return best_scores, best_epoch


def refit_torch(model_factory, x_fit, y_fit, x_test, epochs, lr, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = model_factory()
    sparse_opt, dense_opt = build_optimizers(model, lr)
    xt = torch.from_numpy(x_fit)
    yt = torch.from_numpy(y_fit)
    generator = torch.Generator().manual_seed(seed + 91)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x_fit), generator=generator)
        for start in range(0, len(x_fit), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            sparse_opt.zero_grad(set_to_none=True)
            dense_opt.zero_grad(set_to_none=True)
            logits = model(xt[idx])
            loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
            loss.backward()
            sparse_opt.step()
            dense_opt.step()

    pred = torch_predict(model, x_test)
    del model, sparse_opt, dense_opt, xt, yt
    gc.collect()
    return pred


def smoothed_rate(train_ids, labels, query_ids, cardinality, alpha, prior):
    counts = np.bincount(train_ids, minlength=cardinality).astype(np.float64)
    positives = np.bincount(
        train_ids, weights=labels, minlength=cardinality
    ).astype(np.float64)
    rate = (positives + alpha * prior) / (counts + alpha)
    return rate[query_ids]


def empirical_bayes_scores(fit_split, labels, query_split):
    prior = float(np.mean(labels))
    specs = [
        ("video_id", 30.0, 0.45),
        ("author_id", 45.0, 0.28),
        ("tag", 80.0, 0.12),
        ("duration_bucket", 100.0, 0.08),
        ("upload_type", 100.0, 0.04),
        ("music_type", 100.0, 0.03),
    ]
    score = np.zeros(len(query_split.user_id), dtype=np.float64)
    eps = 1e-5
    for field, alpha, weight in specs:
        train_ids = np.asarray(fit_split.X[field], dtype=np.int64)
        query_ids = np.asarray(query_split.X[field], dtype=np.int64)
        rate = smoothed_rate(
            train_ids,
            labels,
            query_ids,
            int(FEATURE_CARDINALITIES[field]),
            alpha,
            prior,
        )
        rate = np.clip(rate, eps, 1.0 - eps)
        score += weight * np.log(rate / (1.0 - rate))
    return score.astype(np.float32)


def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (x - float(np.mean(x))) / sd


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train_nn = make_neural_matrix(train)
x_valid_nn = make_neural_matrix(valid)

families = {}
family_meta = {}

wide_scores, wide_epoch = train_torch_candidate(
    lambda: WideModel(total_cardinality),
    x_train_nn,
    y_train,
    x_valid_nn,
    valid.user_id,
    y_valid,
    epochs=5,
    lr=0.002,
    seed=SEED + 1,
)
families["wide_additive"] = wide_scores
family_meta["wide_additive"] = {"epoch": wide_epoch}

fm_scores, fm_epoch = train_torch_candidate(
    lambda: FMModel(total_cardinality, k=16),
    x_train_nn,
    y_train,
    x_valid_nn,
    valid.user_id,
    y_valid,
    epochs=9,
    lr=0.001,
    seed=SEED + 2,
)
families["expanded_fm"] = fm_scores
family_meta["expanded_fm"] = {"epoch": fm_epoch}

deep_scores, deep_epoch = train_torch_candidate(
    lambda: DeepFMModel(total_cardinality, len(FIELDS), k=12),
    x_train_nn,
    y_train,
    x_valid_nn,
    valid.user_id,
    y_valid,
    epochs=5,
    lr=0.001,
    seed=SEED + 3,
)
families["deepfm"] = deep_scores
family_meta["deepfm"] = {"epoch": deep_epoch}

eb_scores = empirical_bayes_scores(train, y_train, valid)
families["empirical_bayes"] = eb_scores
family_meta["empirical_bayes"] = {}

x_train_lgb = make_lgb_matrix(train)
x_valid_lgb = make_lgb_matrix(valid)
cat_indices = list(range(len(LGB_CAT_FIELDS)))

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "seed": SEED + 4,
    "num_threads": THREADS,
    "verbose": -1,
}
dtrain = lgb.Dataset(
    x_train_lgb,
    label=y_train,
    categorical_feature=cat_indices,
    free_raw_data=False,
)
dvalid = lgb.Dataset(
    x_valid_lgb,
    label=y_valid,
    categorical_feature=cat_indices,
    reference=dtrain,
    free_raw_data=False,
)
lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=350,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(30, verbose=False)],
)
lgb_rounds = int(lgb_model.best_iteration or 350)
lgb_scores = lgb_model.predict(
    x_valid_lgb, num_iteration=lgb_rounds
).astype(np.float32)
families["lightgbm"] = lgb_scores
family_meta["lightgbm"] = {"rounds": lgb_rounds}

inc_valid_path = os.path.join(ARTIFACTS, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(ARTIFACTS, "incumbent_test_scores.npy")
have_incumbent = (
    ARTIFACTS
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
)

if have_incumbent:
    incumbent_valid = np.asarray(
        np.load(inc_valid_path), dtype=np.float64
    )
    if len(incumbent_valid) == len(y_valid):
        families["trusted_incumbent"] = incumbent_valid
    else:
        have_incumbent = False

candidate_metrics = {}
for name, pred in families.items():
    candidate_metrics[name] = evaluate(
        valid.user_id, y_valid, pred
    )

best_name = max(
    candidate_metrics,
    key=lambda n: candidate_metrics[n]["primary"],
)
best_scores = np.asarray(families[best_name], dtype=np.float64)
best_metrics = candidate_metrics[best_name]
best_kind = "pure"
best_alt = best_name
best_weight = 0.0

# Search only a coarse validation blend grid to avoid selecting sub-noise
# fluctuations from an excessively dense grid.
if have_incumbent:
    inc_z = normalized(incumbent_valid)
    for alt_name in [
        "wide_additive",
        "expanded_fm",
        "deepfm",
        "empirical_bayes",
        "lightgbm",
    ]:
        alt_z = normalized(families[alt_name])
        local_best_metric = None
        local_best_scores = None
        local_best_w = None
        for w_inc in [0.20, 0.35, 0.50, 0.65, 0.80, 0.90]:
            blended = w_inc * inc_z + (1.0 - w_inc) * alt_z
            metric = evaluate(valid.user_id, y_valid, blended)
            if (
                local_best_metric is None
                or metric["primary"] > local_best_metric["primary"]
            ):
                local_best_metric = metric
                local_best_scores = blended
                local_best_w = w_inc

        blend_name = "incumbent_plus_" + alt_name
        candidate_metrics[blend_name] = local_best_metric
        if local_best_metric["primary"] > best_metrics["primary"]:
            best_name = blend_name
            best_scores = np.asarray(local_best_scores, dtype=np.float64)
            best_metrics = local_best_metric
            best_kind = "blend"
            best_alt = alt_name
            best_weight = float(local_best_w)

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

candidate_primary = {
    name: round(float(metric["primary"]), 6)
    for name, metric in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(candidate_primary, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "winner": best_name,
        "winner_kind": best_kind,
        "blend_incumbent_weight": best_weight,
        "wide_epoch": wide_epoch,
        "fm_epoch": fm_epoch,
        "deepfm_epoch": deep_epoch,
        "lightgbm_rounds": lgb_rounds,
    }, sort_keys=True)
)

# Build test scores only after all choices have been made using validation.
test = load("test")

if best_kind == "pure" and best_name == "trusted_incumbent":
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
else:
    need_alt = best_alt
    y_fit = np.ascontiguousarray(
        np.concatenate([
            np.asarray(train.y, dtype=np.float32),
            np.asarray(valid.y, dtype=np.float32),
        ])
    )

    if need_alt in {"wide_additive", "expanded_fm", "deepfm"}:
        x_fit_nn = np.ascontiguousarray(
            np.concatenate([x_train_nn, x_valid_nn], axis=0)
        )
        x_test_nn = make_neural_matrix(test)

        if need_alt == "wide_additive":
            alt_test = refit_torch(
                lambda: WideModel(total_cardinality),
                x_fit_nn,
                y_fit,
                x_test_nn,
                family_meta[need_alt]["epoch"],
                0.002,
                SEED + 1,
            )
        elif need_alt == "expanded_fm":
            alt_test = refit_torch(
                lambda: FMModel(total_cardinality, k=16),
                x_fit_nn,
                y_fit,
                x_test_nn,
                family_meta[need_alt]["epoch"],
                0.001,
                SEED + 2,
            )
        else:
            alt_test = refit_torch(
                lambda: DeepFMModel(
                    total_cardinality, len(FIELDS), k=12
                ),
                x_fit_nn,
                y_fit,
                x_test_nn,
                family_meta[need_alt]["epoch"],
                0.001,
                SEED + 3,
            )

    elif need_alt == "empirical_bayes":
        class CombinedSplit:
            pass

        combined = CombinedSplit()
        combined.X = {
            f: np.concatenate([
                np.asarray(train.X[f]),
                np.asarray(valid.X[f]),
            ])
            for f in [
                "video_id",
                "author_id",
                "tag",
                "duration_bucket",
                "upload_type",
                "music_type",
            ]
        }
        alt_test = empirical_bayes_scores(combined, y_fit, test)

    elif need_alt == "lightgbm":
        x_fit_lgb = np.ascontiguousarray(
            np.concatenate([x_train_lgb, x_valid_lgb], axis=0),
            dtype=np.float32,
        )
        x_test_lgb = make_lgb_matrix(test)
        dfit = lgb.Dataset(
            x_fit_lgb,
            label=y_fit,
            categorical_feature=cat_indices,
            free_raw_data=True,
        )
        final_lgb = lgb.train(
            lgb_params,
            dfit,
            num_boost_round=family_meta[need_alt]["rounds"],
        )
        alt_test = final_lgb.predict(
            x_test_lgb,
            num_iteration=family_meta[need_alt]["rounds"],
        ).astype(np.float32)
    else:
        raise RuntimeError("Unknown selected family: " + str(need_alt))

    if best_kind == "blend":
        incumbent_test = np.asarray(
            np.load(inc_test_path), dtype=np.float64
        )
        test_scores = (
            best_weight * normalized(incumbent_test)
            + (1.0 - best_weight) * normalized(alt_test)
        )
    else:
        test_scores = np.asarray(alt_test, dtype=np.float64)

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))