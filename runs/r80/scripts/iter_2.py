import os
import time
import copy
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
BATCH_SIZE = 4096
THREADS = max(1, min(8, os.cpu_count() or 1))

# These add content and presentation context that varies within a user's slate.
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
NUMERIC = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)


def make_offsets():
    offsets = []
    total = 0
    for name in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def make_sparse_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64) + OFFSETS[j]
            for j, name in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def make_lgb_matrix(split):
    cats = np.column_stack([
        np.asarray(split.X[name], dtype=np.int32) for name in FIELDS
    ])
    nums = []
    for name in NUMERIC:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.where(np.isfinite(x), x, np.nan)
        x = np.log1p(np.maximum(x, 0.0))
        nums.append(x.astype(np.float32))
    return np.ascontiguousarray(
        np.column_stack([cats] + nums),
        dtype=np.float32,
    )


class WideModel(nn.Module):
    def __init__(self, cardinality, prevalence):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, 1, sparse=True)
        with torch.no_grad():
            self.embedding.weight.zero_()
        p = float(np.clip(prevalence, 1e-5, 1.0 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )

    def forward(self, x):
        return self.bias + self.embedding(x).squeeze(-1).sum(dim=1)

    def sparse_params(self):
        return [self.embedding.weight]

    def dense_params(self):
        return [self.bias]


class FMModel(nn.Module):
    def __init__(self, cardinality, prevalence, rank=16):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)
        p = float(np.clip(prevalence, 1e-5, 1.0 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction

    def sparse_params(self):
        return [self.embedding.weight]

    def dense_params(self):
        return [self.bias]


class DeepFMModel(nn.Module):
    def __init__(self, cardinality, prevalence, n_fields, rank=12):
        super().__init__()
        self.rank = rank
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)

        self.mlp = nn.Sequential(
            nn.Linear(n_fields * rank, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        p = float(np.clip(prevalence, 1e-5, 1.0 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.mlp(v.reshape(v.shape[0], -1)).squeeze(1)
        return self.bias + linear + interaction + deep

    def sparse_params(self):
        return [self.embedding.weight]

    def dense_params(self):
        return [self.bias] + list(self.mlp.parameters())


def predict_torch(model, x):
    model.eval()
    scores = np.empty(len(x), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x), 32768):
            end = min(start + 32768, len(x))
            xb = torch.from_numpy(x[start:end])
            scores[start:end] = (
                model(xb).cpu().numpy().astype(np.float64)
            )
    return scores


def train_epoch(model, x, y, sparse_opt, dense_opt, rng):
    model.train()
    order = rng.permutation(len(y))
    total_loss = 0.0
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = torch.from_numpy(x[idx])
        yb = torch.from_numpy(y[idx])

        sparse_opt.zero_grad(set_to_none=True)
        dense_opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        sparse_opt.step()
        dense_opt.step()
        total_loss += float(loss.detach()) * len(idx)
    return total_loss / len(y)


def new_torch_model(family, prevalence):
    torch.manual_seed(SEED)
    if family == "wide":
        return WideModel(TOTAL_CARDINALITY, prevalence)
    if family == "fm":
        return FMModel(TOTAL_CARDINALITY, prevalence, rank=16)
    if family == "deepfm":
        return DeepFMModel(
            TOTAL_CARDINALITY, prevalence, len(FIELDS), rank=12
        )
    raise ValueError(family)


def fit_torch_valid(family, x_tr, y_tr, x_va, y_va, users):
    max_epochs = {"wide": 6, "fm": 8, "deepfm": 5}[family]
    model = new_torch_model(family, float(y_tr.mean()))
    sparse_opt = torch.optim.SparseAdam(
        model.sparse_params(), lr=0.001
    )
    dense_opt = torch.optim.Adam(model.dense_params(), lr=0.001)
    rng = np.random.default_rng(SEED)

    best_score = -np.inf
    best_epoch = 1
    best_state = None
    best_predictions = None

    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(
            model, x_tr, y_tr, sparse_opt, dense_opt, rng
        )
        predictions = predict_torch(model, x_va)
        result = evaluate(users, y_va, predictions)
        print(
            "FIT family=%s epoch=%d loss=%.6f primary=%.6f"
            % (family, epoch, loss, result["primary"]),
            flush=True,
        )
        if result["primary"] > best_score:
            best_score = float(result["primary"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_predictions = predictions.copy()

    model.load_state_dict(best_state)
    return best_predictions, best_epoch


def fit_torch_fixed(family, x, y, epochs):
    model = new_torch_model(family, float(y.mean()))
    sparse_opt = torch.optim.SparseAdam(
        model.sparse_params(), lr=0.001
    )
    dense_opt = torch.optim.Adam(model.dense_params(), lr=0.001)
    rng = np.random.default_rng(SEED)
    for _ in range(epochs):
        train_epoch(model, x, y, sparse_opt, dense_opt, rng)
    return model


def empirical_bayes_fit_scores(train_split, labels, target_split):
    global_rate = float(np.mean(labels))
    global_logit = np.log(
        np.clip(global_rate, 1e-6, 1 - 1e-6)
        / np.clip(1 - global_rate, 1e-6, 1 - 1e-6)
    )
    components = []

    for field, strength, coefficient in [
        ("video_id", 20.0, 0.60),
        ("author_id", 35.0, 0.40),
    ]:
        ids = np.asarray(train_split.X[field], dtype=np.int64)
        target_ids = np.asarray(target_split.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, minlength=card).astype(np.float64)
        positives = np.bincount(
            ids, weights=labels, minlength=card
        ).astype(np.float64)
        rates = (
            positives + strength * global_rate
        ) / (counts + strength)
        rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
        logits = np.log(rates / (1.0 - rates))
        components.append(coefficient * logits[target_ids])

    return global_logit + sum(components)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd < 1e-12:
        return x - float(np.mean(x))
    return (x - float(np.mean(x))) / sd


tr = load("train")
va = load("valid")
y_tr = np.asarray(tr.y, dtype=np.float32)
y_va_float = np.asarray(va.y, dtype=np.float32)
y_va = np.asarray(va.y, dtype=np.int8)
valid_users = np.asarray(va.user_id)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

x_tr = make_sparse_matrix(tr)
x_va = make_sparse_matrix(va)

family_predictions = {}
family_recipe = {}

# Additive wide memorization.
wide_pred, wide_epoch = fit_torch_valid(
    "wide", x_tr, y_tr, x_va, y_va, valid_users
)
family_predictions["wide"] = wide_pred
family_recipe["wide"] = {"epochs": wide_epoch}

# Expanded pairwise factorization machine.
fm_pred, fm_epoch = fit_torch_valid(
    "fm", x_tr, y_tr, x_va, y_va, valid_users
)
family_predictions["fm"] = fm_pred
family_recipe["fm"] = {"epochs": fm_epoch}

# Nonlinear high-order interaction model.
deep_pred, deep_epoch = fit_torch_valid(
    "deepfm", x_tr, y_tr, x_va, y_va, valid_users
)
family_predictions["deepfm"] = deep_pred
family_recipe["deepfm"] = {"epochs": deep_epoch}

# Non-parametric smoothed entity response statistics.
eb_pred = empirical_bayes_fit_scores(tr, y_tr, va)
family_predictions["empirical_bayes"] = eb_pred
family_recipe["empirical_bayes"] = {}

# Gradient boosting uses categorical splits plus robust log-scaled numeric values.
lgb_tr = make_lgb_matrix(tr)
lgb_va = make_lgb_matrix(va)
lgb_train = lgb.Dataset(
    lgb_tr,
    label=y_tr,
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=False,
)
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}
LGB_ROUNDS = 180
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=LGB_ROUNDS,
)
lgb_pred = lgb_model.predict(
    lgb_va, num_iteration=LGB_ROUNDS
).astype(np.float64)
family_predictions["lightgbm"] = lgb_pred
family_recipe["lightgbm"] = {"rounds": LGB_ROUNDS}

candidate_scores = {}
candidate_payload = {}

inc_metric = evaluate(valid_users, y_va, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_payload["incumbent"] = ("incumbent", 0.0, inc_valid)

# Score every standalone model and a validation-selected standardized blend
# with the trusted incumbent.
inc_z = zscore(inc_valid)
for family, pred in family_predictions.items():
    standalone = evaluate(valid_users, y_va, pred)
    standalone_name = family + "_standalone"
    candidate_scores[standalone_name] = float(standalone["primary"])
    candidate_payload[standalone_name] = (family, 1.0, pred)

    pred_z = zscore(pred)
    best_blend_score = -np.inf
    best_alpha = 0.0
    best_blend = None

    for alpha in np.linspace(0.1, 0.9, 9):
        blended = alpha * pred_z + (1.0 - alpha) * inc_z
        result = evaluate(valid_users, y_va, blended)
        if result["primary"] > best_blend_score:
            best_blend_score = float(result["primary"])
            best_alpha = float(alpha)
            best_blend = blended.copy()

    blend_name = family + "_blend"
    candidate_scores[blend_name] = best_blend_score
    candidate_payload[blend_name] = (
        family,
        best_alpha,
        best_blend,
    )

winner_name = max(candidate_scores, key=candidate_scores.get)
winner_family, winner_alpha, valid_scores = candidate_payload[winner_name]
metrics = evaluate(valid_users, y_va, valid_scores)

print(
    "CANDIDATES "
    + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner_name,
            "family": winner_family,
            "incumbent_weight": float(1.0 - winner_alpha),
            "new_family_weight": float(winner_alpha),
            "selected_recipe": family_recipe.get(winner_family, {}),
        },
        sort_keys=True,
    ),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit only the selected new family on train + validation. If validation
# retained the incumbent, reuse its already-refitted test predictions.
te = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winner_family == "incumbent" or winner_alpha == 0.0:
    test_scores = inc_test.copy()
else:
    y_combined = np.concatenate([y_tr, y_va_float], axis=0)

    if winner_family in ("wide", "fm", "deepfm"):
        x_combined = np.concatenate([x_tr, x_va], axis=0)
        selected_model = fit_torch_fixed(
            winner_family,
            x_combined,
            y_combined,
            int(family_recipe[winner_family]["epochs"]),
        )
        x_te = make_sparse_matrix(te)
        family_test = predict_torch(selected_model, x_te)
        del selected_model, x_te, x_combined

    elif winner_family == "empirical_bayes":
        class CombinedSplit:
            pass

        combined = CombinedSplit()
        combined.X = {
            field: np.concatenate([
                np.asarray(tr.X[field]),
                np.asarray(va.X[field]),
            ])
            for field in ("video_id", "author_id")
        }
        family_test = empirical_bayes_fit_scores(
            combined, y_combined, te
        )

    elif winner_family == "lightgbm":
        lgb_combined = np.concatenate([lgb_tr, lgb_va], axis=0)
        lgb_te = make_lgb_matrix(te)
        combined_dataset = lgb.Dataset(
            lgb_combined,
            label=y_combined,
            categorical_feature=list(range(len(FIELDS))),
            free_raw_data=False,
        )
        selected_model = lgb.train(
            lgb_params,
            combined_dataset,
            num_boost_round=int(
                family_recipe["lightgbm"]["rounds"]
            ),
        )
        family_test = selected_model.predict(
            lgb_te,
            num_iteration=int(
                family_recipe["lightgbm"]["rounds"]
            ),
        ).astype(np.float64)
        del selected_model, combined_dataset, lgb_combined, lgb_te

    else:
        raise ValueError("Unknown winner family: " + winner_family)

    if winner_alpha >= 1.0 - 1e-12:
        test_scores = family_test
    else:
        test_scores = (
            winner_alpha * zscore(family_test)
            + (1.0 - winner_alpha) * zscore(inc_test)
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

gc.collect()
elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(", ", ": "),
    )
)