import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2026
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
K = 16
BATCH_SIZE = 4096
FM_EPOCHS = 7
DEEP_EPOCHS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))

ART = os.environ["RUN_ARTIFACTS"]
OUT = os.environ.get("ITER_OUT")
if OUT:
    os.makedirs(OUT, exist_ok=True)

inc_valid = np.asarray(
    np.load(os.path.join(ART, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)

train = load("train")
valid = load("valid")
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)


def make_x(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[f] for f in FIELDS]),
        dtype=np.int64,
    )


x_train_np = make_x(train)
x_valid_np = make_x(valid)

CARDS = np.asarray(
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS], dtype=np.int64
)
OFFSETS = np.cumsum(
    np.concatenate([np.zeros(1, dtype=np.int64), CARDS[:-1]])
)
TOTAL = int(CARDS.sum())


def within_user_rank(user_ids, scores):
    """Ascending percentile rank within each user, fully vectorized."""
    u = np.asarray(user_ids)
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    order = np.lexsort((np.arange(n, dtype=np.int64), s, u))
    us = u[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = us[1:] != us[:-1]
    start_values = np.where(starts_flag, np.arange(n), 0)
    starts = np.maximum.accumulate(start_values)

    ends_flag = np.empty(n, dtype=bool)
    ends_flag[-1] = True
    ends_flag[:-1] = us[:-1] != us[1:]
    end_values = np.where(ends_flag, np.arange(n), n - 1)
    ends = np.minimum.accumulate(end_values[::-1])[::-1]

    denom = ends - starts
    ranks_sorted = np.empty(n, dtype=np.float64)
    multi = denom > 0
    ranks_sorted[multi] = (
        np.arange(n, dtype=np.float64)[multi] - starts[multi]
    ) / denom[multi]
    ranks_sorted[~multi] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks_sorted
    return result


class ExpandedFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(TOTAL, 1, sparse=True)
        self.latent = nn.Embedding(TOTAL, K, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "offsets", torch.from_numpy(OFFSETS.copy()).long()
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, std=0.01)

    def forward(self, x):
        ids = x + self.offsets
        linear = self.linear(ids).squeeze(-1).sum(dim=1)
        v = self.latent(ids)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(TOTAL, 1)
        self.latent = nn.Embedding(TOTAL, K)
        self.deep = nn.Sequential(
            nn.Linear(len(FIELDS) * K, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "offsets", torch.from_numpy(OFFSETS.copy()).long()
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, std=0.01)
        for layer in self.deep:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        ids = x + self.offsets
        linear = self.linear(ids).squeeze(-1).sum(dim=1)
        v = self.latent(ids)
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(v.reshape(v.shape[0], -1)).squeeze(1)
        return self.bias + linear + fm + deep


def predict_torch(model, x_np, batch=65536):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for st in range(0, len(x_np), batch):
            en = min(st + batch, len(x_np))
            result[st:en] = (
                model(x[st:en]).detach().cpu().numpy().astype(np.float64)
            )
    return result


def fit_fm(x_np, y_np, epochs, valid_x=None):
    torch.manual_seed(SEED)
    model = ExpandedFM()
    sparse_opt = torch.optim.SparseAdam(
        [model.linear.weight, model.latent.weight], lr=0.001
    )
    dense_opt = torch.optim.Adam([model.bias], lr=0.001)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    gen = torch.Generator().manual_seed(SEED + 11)

    best_score = -np.inf
    best_epoch = epochs
    best_state = None
    best_pred = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=gen)
        loss_sum = 0.0
        for st in range(0, len(x), BATCH_SIZE):
            idx = order[st:st + BATCH_SIZE]
            sparse_opt.zero_grad(set_to_none=True)
            dense_opt.zero_grad(set_to_none=True)
            logits = model(x[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx]
            )
            loss.backward()
            sparse_opt.step()
            dense_opt.step()
            loss_sum += float(loss.detach()) * len(idx)

        if valid_x is not None:
            pred = predict_torch(model, valid_x)
            met = evaluate(valid.user_id, y_valid, pred)
            print(
                "fm epoch=%d loss=%.6f primary=%.6f"
                % (epoch, loss_sum / len(x), met["primary"]),
                flush=True,
            )
            if met["primary"] > best_score:
                best_score = float(met["primary"])
                best_epoch = epoch
                best_pred = pred.copy()
                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }

    if valid_x is not None:
        model.load_state_dict(best_state)
        return model, best_pred, best_epoch
    return model, None, epochs


def fit_deepfm(x_np, y_np, epochs, valid_x=None):
    torch.manual_seed(SEED + 101)
    model = DeepFM()
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=1e-6
    )
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    gen = torch.Generator().manual_seed(SEED + 103)

    best_score = -np.inf
    best_epoch = epochs
    best_state = None
    best_pred = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=gen)
        loss_sum = 0.0
        for st in range(0, len(x), BATCH_SIZE):
            idx = order[st:st + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            logits = model(x[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)

        if valid_x is not None:
            pred = predict_torch(model, valid_x)
            met = evaluate(valid.user_id, y_valid, pred)
            print(
                "deepfm epoch=%d loss=%.6f primary=%.6f"
                % (epoch, loss_sum / len(x), met["primary"]),
                flush=True,
            )
            if met["primary"] > best_score:
                best_score = float(met["primary"])
                best_epoch = epoch
                best_pred = pred.copy()
                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }

    if valid_x is not None:
        model.load_state_dict(best_state)
        return model, best_pred, best_epoch
    return model, None, epochs


def empirical_bayes_fit_predict(fit_x, fit_y, pred_x):
    global_rate = float(np.mean(fit_y))
    global_logit = np.log(global_rate / (1.0 - global_rate))
    smooth = {
        "user_id": 18.0,
        "video_id": 80.0,
        "author_id": 80.0,
        "tab": 300.0,
        "duration_bucket": 300.0,
        "tag": 250.0,
        "upload_type": 300.0,
        "music_type": 300.0,
        "hour": 400.0,
    }
    field_weights = {
        "user_id": 1.00,
        "video_id": 0.85,
        "author_id": 0.75,
        "tab": 0.45,
        "duration_bucket": 0.35,
        "tag": 0.55,
        "upload_type": 0.30,
        "music_type": 0.25,
        "hour": 0.30,
    }

    score = np.full(len(pred_x), global_logit, dtype=np.float64)
    total_weight = 1.0

    for j, name in enumerate(FIELDS):
        card = int(CARDS[j])
        ids = fit_x[:, j]
        counts = np.bincount(ids, minlength=card).astype(np.float64)
        positives = np.bincount(
            ids, weights=fit_y, minlength=card
        ).astype(np.float64)
        rate = (
            positives + smooth[name] * global_rate
        ) / (counts + smooth[name])
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        logits = np.log(rate / (1.0 - rate))
        w = field_weights[name]
        score += w * logits[pred_x[:, j]]
        total_weight += w

    return score / total_weight


# Family 1: expanded FM.
fm_model, fm_valid, fm_best_epoch = fit_fm(
    x_train_np, y_train_np, FM_EPOCHS, x_valid_np
)
del fm_model
gc.collect()

# Family 2: DeepFM.
deep_model, deep_valid, deep_best_epoch = fit_deepfm(
    x_train_np, y_train_np, DEEP_EPOCHS, x_valid_np
)
del deep_model
gc.collect()

# Family 3: categorical gradient boosting.
lgb_train_x = np.asarray(x_train_np, dtype=np.int32)
lgb_valid_x = np.asarray(x_valid_np, dtype=np.int32)
dtrain = lgb.Dataset(
    lgb_train_x,
    label=y_train_np,
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=False,
)
dvalid = lgb.Dataset(
    lgb_valid_x,
    label=y_valid,
    categorical_feature=list(range(len(FIELDS))),
    reference=dtrain,
    free_raw_data=False,
)
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 150,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_cat_to_onehot": 16,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "verbosity": -1,
    "num_threads": min(16, max(1, os.cpu_count() or 1)),
    "seed": SEED,
}
lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=220,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(25, verbose=False)],
)
lgb_rounds = int(lgb_model.best_iteration or 220)
lgb_valid = np.asarray(
    lgb_model.predict(lgb_valid_x, num_iteration=lgb_rounds),
    dtype=np.float64,
)
del lgb_model, dtrain, dvalid
gc.collect()

# Family 4: non-parametric empirical Bayes.
eb_valid = empirical_bayes_fit_predict(
    x_train_np, y_train_np.astype(np.float64), x_valid_np
)

family_predictions = {
    "expanded_fm": fm_valid,
    "deepfm": deep_valid,
    "lightgbm_binary": lgb_valid,
    "empirical_bayes": eb_valid,
}

candidate_scores = {}
candidate_predictions = {}
candidate_recipe = {}

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_recipe["incumbent"] = ("incumbent", 1.0)

inc_rank = within_user_rank(valid.user_id, inc_valid)
blend_weights = [0.25, 0.40, 0.55, 0.70, 0.85]

for family_name, pred in family_predictions.items():
    met = evaluate(valid.user_id, y_valid, pred)
    candidate_scores[family_name] = float(met["primary"])
    candidate_predictions[family_name] = pred
    candidate_recipe[family_name] = (family_name, 1.0)

    family_rank = within_user_rank(valid.user_id, pred)
    for alpha in blend_weights:
        blend_name = "%s_blend_%02d" % (
            family_name, int(round(alpha * 100))
        )
        blend = alpha * family_rank + (1.0 - alpha) * inc_rank
        blend_met = evaluate(valid.user_id, y_valid, blend)
        candidate_scores[blend_name] = float(blend_met["primary"])
        candidate_predictions[blend_name] = blend
        candidate_recipe[blend_name] = (family_name, alpha)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(
    candidate_predictions[winner], dtype=np.float64
)
best_metrics = evaluate(valid.user_id, y_valid, valid_scores)

print(
    "FINDINGS fm_best_epoch=%d deepfm_best_epoch=%d lgb_rounds=%d winner=%s"
    % (fm_best_epoch, deep_best_epoch, lgb_rounds, winner),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(
        candidate_scores, sort_keys=True, separators=(",", ":")
    ),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )

# Refit only the selected family on train + validation.
test = load("test")
x_test_np = make_x(test)
inc_test = np.asarray(
    np.load(os.path.join(ART, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

selected_family, selected_alpha = candidate_recipe[winner]

if selected_family == "incumbent":
    family_test = inc_test
elif selected_family == "expanded_fm":
    x_combined = np.ascontiguousarray(
        np.concatenate([x_train_np, x_valid_np], axis=0),
        dtype=np.int64,
    )
    y_combined = np.ascontiguousarray(
        np.concatenate(
            [y_train_np, np.asarray(valid.y, dtype=np.float32)]
        ),
        dtype=np.float32,
    )
    final_model, _, _ = fit_fm(
        x_combined, y_combined, fm_best_epoch, None
    )
    family_test = predict_torch(final_model, x_test_np)
    del final_model, x_combined, y_combined
elif selected_family == "deepfm":
    x_combined = np.ascontiguousarray(
        np.concatenate([x_train_np, x_valid_np], axis=0),
        dtype=np.int64,
    )
    y_combined = np.ascontiguousarray(
        np.concatenate(
            [y_train_np, np.asarray(valid.y, dtype=np.float32)]
        ),
        dtype=np.float32,
    )
    final_model, _, _ = fit_deepfm(
        x_combined, y_combined, deep_best_epoch, None
    )
    family_test = predict_torch(final_model, x_test_np)
    del final_model, x_combined, y_combined
elif selected_family == "lightgbm_binary":
    x_combined = np.ascontiguousarray(
        np.concatenate([lgb_train_x, lgb_valid_x], axis=0),
        dtype=np.int32,
    )
    y_combined = np.ascontiguousarray(
        np.concatenate(
            [y_train_np, np.asarray(valid.y, dtype=np.float32)]
        ),
        dtype=np.float32,
    )
    dfinal = lgb.Dataset(
        x_combined,
        label=y_combined,
        categorical_feature=list(range(len(FIELDS))),
        free_raw_data=False,
    )
    final_lgb = lgb.train(
        lgb_params,
        dfinal,
        num_boost_round=lgb_rounds,
    )
    family_test = np.asarray(
        final_lgb.predict(
            np.asarray(x_test_np, dtype=np.int32),
            num_iteration=lgb_rounds,
        ),
        dtype=np.float64,
    )
    del final_lgb, dfinal, x_combined, y_combined
elif selected_family == "empirical_bayes":
    x_combined = np.ascontiguousarray(
        np.concatenate([x_train_np, x_valid_np], axis=0),
        dtype=np.int64,
    )
    y_combined = np.ascontiguousarray(
        np.concatenate(
            [
                y_train_np.astype(np.float64),
                np.asarray(valid.y, dtype=np.float64),
            ]
        )
    )
    family_test = empirical_bayes_fit_predict(
        x_combined, y_combined, x_test_np
    )
    del x_combined, y_combined
else:
    raise RuntimeError("Unknown selected family: " + selected_family)

if selected_family == "incumbent":
    test_scores = inc_test
elif selected_alpha >= 0.999:
    test_scores = np.asarray(family_test, dtype=np.float64)
else:
    test_scores = (
        selected_alpha
        * within_user_rank(test.user_id, family_test)
        + (1.0 - selected_alpha)
        * within_user_rank(test.user_id, inc_test)
    )

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(",", ":")))