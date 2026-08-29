import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260829
PRED_BATCH = 65536

torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_FIELDS = [
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
RICH_FIELDS = BASE_FIELDS + [
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
]
RICHER_FIELDS = RICH_FIELDS + [
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
]

FIELD_SETS = {
    "base9": BASE_FIELDS,
    "rich13": RICH_FIELDS,
    "rich16": RICHER_FIELDS,
}

CONFIGS = [
    dict(name="k16_lr10_base", fields="base9", k=16, lr=0.0010,
         batch=8192, wd=0.0, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k8_lr10_base", fields="base9", k=8, lr=0.0010,
         batch=8192, wd=0.0, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k24_lr10_base", fields="base9", k=24, lr=0.0010,
         batch=8192, wd=0.0, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k32_lr10_base", fields="base9", k=32, lr=0.0010,
         batch=8192, wd=0.0, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k16_lr06_base", fields="base9", k=16, lr=0.0006,
         batch=8192, wd=0.0, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k16_lr15_base", fields="base9", k=16, lr=0.0015,
         batch=8192, wd=0.0, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k16_lr20_base", fields="base9", k=16, lr=0.0020,
         batch=8192, wd=0.0, user_exp=0.0, pos_weight=1.0, schedule="decay"),
    dict(name="k16_bs4096", fields="base9", k=16, lr=0.0010,
         batch=4096, wd=0.0, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k16_bs16384", fields="base9", k=16, lr=0.0010,
         batch=16384, wd=0.0, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k16_wd1e6", fields="base9", k=16, lr=0.0010,
         batch=8192, wd=1e-6, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k16_wd1e5", fields="base9", k=16, lr=0.0010,
         batch=8192, wd=1e-5, user_exp=0.0, pos_weight=1.0, schedule="constant"),
    dict(name="k16_user025", fields="base9", k=16, lr=0.0010,
         batch=8192, wd=0.0, user_exp=0.25, pos_weight=1.0, schedule="constant"),
    dict(name="k16_user050", fields="base9", k=16, lr=0.0010,
         batch=8192, wd=0.0, user_exp=0.50, pos_weight=1.0, schedule="constant"),
    dict(name="k16_pos080", fields="base9", k=16, lr=0.0010,
         batch=8192, wd=0.0, user_exp=0.0, pos_weight=0.80, schedule="constant"),
    dict(name="k16_rich13", fields="rich13", k=16, lr=0.0010,
         batch=8192, wd=1e-6, user_exp=0.25, pos_weight=1.0, schedule="constant"),
    dict(name="k16_rich16", fields="rich16", k=16, lr=0.0010,
         batch=8192, wd=1e-6, user_exp=0.25, pos_weight=1.0, schedule="constant"),
]
CONFIG_BY_NAME = {c["name"]: c for c in CONFIGS}


def initial_logit(y):
    p = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    return float(np.log(p / (1.0 - p)))


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    std = float(x.std())
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return (x - mean) / std


def field_spec(field_key):
    fields = FIELD_SETS[field_key]
    cards = np.asarray(
        [int(FEATURE_CARDINALITIES[f]) for f in fields],
        dtype=np.int64,
    )
    offsets = np.concatenate([
        np.zeros(1, dtype=np.int64),
        np.cumsum(cards[:-1], dtype=np.int64),
    ])
    return fields, offsets, int(cards.sum())


def make_matrix(split, field_key):
    fields, offsets, _ = field_spec(field_key)
    x = np.column_stack([
        np.asarray(split.X[f], dtype=np.int64) for f in fields
    ])
    x += offsets[None, :]
    return np.ascontiguousarray(x, dtype=np.int64)


def row_weights(user_ids, exponent):
    if exponent <= 0:
        return np.ones(len(user_ids), dtype=np.float32)
    ids = np.asarray(user_ids, dtype=np.int64)
    counts = np.bincount(ids)
    w = np.power(
        np.maximum(counts[ids], 1).astype(np.float64),
        -float(exponent),
    )
    w /= np.mean(w)
    return np.asarray(w, dtype=np.float32)


class FM(nn.Module):
    def __init__(self, total_cardinality, k, bias):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, k)
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        e = self.embedding(x)
        summed = e.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - e.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


def set_epoch_lr(optimizer, cfg, epoch):
    multiplier = 1.0
    if cfg["schedule"] == "decay":
        multipliers = [1.0, 1.0, 0.75, 0.55, 0.40, 0.30]
        multiplier = multipliers[min(epoch - 1, len(multipliers) - 1)]
    for group in optimizer.param_groups:
        group["lr"] = float(cfg["lr"]) * multiplier


class Trial:
    def __init__(self, cfg, y_train):
        self.cfg = cfg
        _, _, total = field_spec(cfg["fields"])
        torch.manual_seed(SEED)
        self.model = FM(total, int(cfg["k"]), initial_logit(y_train))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["wd"]),
        )
        self.generator = torch.Generator()
        self.generator.manual_seed(SEED + 37)
        self.epoch = 0
        self.last_scores = None
        self.last_primary = -np.inf
        self.epoch_results = {}

    def train_to(self, target_epoch, x_np, y_np, weights_np,
                 x_valid, y_valid, valid_users):
        xt = torch.from_numpy(x_np)
        yt = torch.from_numpy(y_np)
        wt = torch.from_numpy(weights_np)
        n = len(y_np)
        batch_size = int(self.cfg["batch"])
        pos_weight = float(self.cfg["pos_weight"])

        while self.epoch < target_epoch:
            self.epoch += 1
            set_epoch_lr(self.optimizer, self.cfg, self.epoch)
            self.model.train()
            permutation = torch.randperm(n, generator=self.generator)
            total_loss = 0.0
            total_weight = 0.0

            for start in range(0, n, batch_size):
                idx = permutation[start:start + batch_size]
                xb = xt.index_select(0, idx)
                yb = yt.index_select(0, idx)
                wb = wt.index_select(0, idx)

                if pos_weight != 1.0:
                    wb = wb * (1.0 + (pos_weight - 1.0) * yb)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                losses = F.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )
                denom = wb.sum().clamp_min(1e-8)
                loss = (losses * wb).sum() / denom
                loss.backward()
                self.optimizer.step()

                total_loss += float((losses.detach() * wb).sum())
                total_weight += float(denom)

            scores = predict(self.model, x_valid)
            metrics = evaluate(valid_users, y_valid, scores)
            primary = float(metrics["primary"])
            self.last_scores = scores
            self.last_primary = primary
            self.epoch_results[self.epoch] = (
                primary,
                scores.copy(),
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
            )
            print(
                "trial=%s epoch=%d loss=%.6f primary=%.6f "
                "gauc=%.6f ndcg5=%.6f"
                % (
                    self.cfg["name"],
                    self.epoch,
                    total_loss / max(total_weight, 1e-8),
                    primary,
                    float(metrics["gauc"]),
                    float(metrics["ndcg@5"]),
                ),
                flush=True,
            )


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    xt = torch.from_numpy(x_np)
    scores = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), PRED_BATCH):
        end = min(start + PRED_BATCH, len(x_np))
        scores[start:end] = (
            model(xt[start:end]).cpu().numpy().astype(np.float64, copy=False)
        )
    return scores


def fit_fixed(cfg, x_np, y_np, users_np, epochs):
    _, _, total = field_spec(cfg["fields"])
    torch.manual_seed(SEED)
    model = FM(total, int(cfg["k"]), initial_logit(y_np))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["wd"]),
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + 37)

    xt = torch.from_numpy(x_np)
    yt = torch.from_numpy(y_np)
    weights_np = row_weights(users_np, float(cfg["user_exp"]))
    wt = torch.from_numpy(weights_np)

    n = len(y_np)
    batch_size = int(cfg["batch"])
    pos_weight = float(cfg["pos_weight"])

    for epoch in range(1, int(epochs) + 1):
        set_epoch_lr(optimizer, cfg, epoch)
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt.index_select(0, idx)
            yb = yt.index_select(0, idx)
            wb = wt.index_select(0, idx)
            if pos_weight != 1.0:
                wb = wb * (1.0 + (pos_weight - 1.0) * yb)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            denom = wb.sum().clamp_min(1e-8)
            loss = (losses * wb).sum() / denom
            loss.backward()
            optimizer.step()

            total_loss += float((losses.detach() * wb).sum())
            total_weight += float(denom)

        print(
            "refit=%s epoch=%d loss=%.6f"
            % (
                cfg["name"],
                epoch,
                total_loss / max(total_weight, 1e-8),
            ),
            flush=True,
        )

    del optimizer, xt, yt, wt
    gc.collect()
    return model


train = load("train")
valid = load("valid")

y_train = np.ascontiguousarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

artifact_dir = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifact_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(artifact_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
inc_metrics = evaluate(valid_users, y_valid, inc_valid)
inc_primary = float(inc_metrics["primary"])
inc_z = standardize(inc_valid)

needed_field_sets = sorted(set(c["fields"] for c in CONFIGS))
x_train_cache = {}
x_valid_cache = {}
for key in needed_field_sets:
    x_train_cache[key] = make_matrix(train, key)
    x_valid_cache[key] = make_matrix(valid, key)

weight_cache = {}
for exponent in sorted(set(float(c["user_exp"]) for c in CONFIGS)):
    weight_cache[exponent] = row_weights(train_users, exponent)

# Rung 1: all 16 configurations, one full-data epoch.
trials = {}
rung1 = {}
for cfg in CONFIGS:
    trial = Trial(cfg, y_train)
    trial.train_to(
        1,
        x_train_cache[cfg["fields"]],
        y_train,
        weight_cache[float(cfg["user_exp"])],
        x_valid_cache[cfg["fields"]],
        y_valid,
        valid_users,
    )
    trials[cfg["name"]] = trial
    rung1[cfg["name"]] = float(trial.last_primary)

top6_names = sorted(rung1, key=rung1.get, reverse=True)[:6]
print(
    "FINDINGS rung1_top6=" + json.dumps(top6_names)
    + " scores=" + json.dumps(rung1, sort_keys=True),
    flush=True,
)

# Rung 2: continue the strongest six through epoch 3.
rung2 = {}
for name in top6_names:
    trial = trials[name]
    cfg = trial.cfg
    trial.train_to(
        3,
        x_train_cache[cfg["fields"]],
        y_train,
        weight_cache[float(cfg["user_exp"])],
        x_valid_cache[cfg["fields"]],
        y_valid,
        valid_users,
    )
    rung2[name] = float(trial.last_primary)

top2_names = sorted(rung2, key=rung2.get, reverse=True)[:2]
print(
    "FINDINGS rung2_top2=" + json.dumps(top2_names)
    + " scores=" + json.dumps(rung2, sort_keys=True),
    flush=True,
)

# Release models that cannot reach rung 3.
for name in list(trials):
    if name not in top2_names:
        del trials[name]
gc.collect()

# Rung 3: continue the two finalists through epoch 6.
rung3 = {}
final_model_scores = {}
final_model_epochs = {}

for name in top2_names:
    trial = trials[name]
    cfg = trial.cfg
    trial.train_to(
        6,
        x_train_cache[cfg["fields"]],
        y_train,
        weight_cache[float(cfg["user_exp"])],
        x_valid_cache[cfg["fields"]],
        y_valid,
        valid_users,
    )

    eligible = {
        epoch: values for epoch, values in trial.epoch_results.items()
        if epoch >= 3
    }
    best_epoch = max(eligible, key=lambda e: eligible[e][0])
    best_values = eligible[best_epoch]
    final_model_epochs[name] = int(best_epoch)
    final_model_scores[name] = np.asarray(best_values[1], dtype=np.float64)
    rung3[name] = float(best_values[0])

print(
    "FINDINGS rung3_scores=" + json.dumps(rung3, sort_keys=True)
    + " best_epochs=" + json.dumps(final_model_epochs, sort_keys=True),
    flush=True,
)

# Compare stable finalist fusion and incumbent blends.
base_components = {}
recipes = {}

for name in top2_names:
    base_components[name] = standardize(final_model_scores[name])
    recipes[name] = {
        "models": [(name, 1.0)],
    }

ensemble_name = "rung3_ensemble"
base_components[ensemble_name] = (
    standardize(final_model_scores[top2_names[0]])
    + standardize(final_model_scores[top2_names[1]])
) / 2.0
recipes[ensemble_name] = {
    "models": [(top2_names[0], 0.5), (top2_names[1], 0.5)],
}

candidate_values = {"incumbent": inc_primary}
candidate_arrays = {"incumbent": inc_valid.copy()}
candidate_recipes = {
    "incumbent": {"models": [], "inc_weight": 1.0}
}

blend_weights = [0.25, 0.50, 0.75, 1.00]
for component_name, component_scores in base_components.items():
    for model_weight in blend_weights:
        candidate_name = "%s_m%.2f" % (component_name, model_weight)
        scores = (
            model_weight * component_scores
            + (1.0 - model_weight) * inc_z
        )
        metrics = evaluate(valid_users, y_valid, scores)
        candidate_values[candidate_name] = float(metrics["primary"])
        candidate_arrays[candidate_name] = scores
        candidate_recipes[candidate_name] = {
            "models": recipes[component_name]["models"],
            "inc_weight": float(1.0 - model_weight),
        }

best_noninc = max(
    (k for k in candidate_values if k != "incumbent"),
    key=candidate_values.get,
)

# Prefer the two-finalist ensemble when it is effectively tied with an
# isolated finalist peak.
ensemble_candidates = [
    k for k in candidate_values if k.startswith(ensemble_name + "_")
]
best_ensemble = max(ensemble_candidates, key=candidate_values.get)
if candidate_values[best_ensemble] >= candidate_values[best_noninc] - 0.0004:
    selected_noninc = best_ensemble
else:
    selected_noninc = best_noninc

# Do not ship a configuration-search fluctuation smaller than the specified
# 0.0008 noise guard.
if candidate_values[selected_noninc] >= inc_primary + 0.0008:
    winner = selected_noninc
else:
    winner = "incumbent"

valid_scores = np.asarray(candidate_arrays[winner], dtype=np.float64)
valid_metrics = evaluate(valid_users, y_valid, valid_scores)

print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_values.items()},
        sort_keys=True,
    ),
    flush=True,
)
print(
    "FINDINGS selected=%s incumbent=%.6f selected_noninc=%s "
    "selected_noninc_primary=%.6f noise_guard=0.0008"
    % (
        winner,
        inc_primary,
        selected_noninc,
        candidate_values[selected_noninc],
    ),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores.astype(np.float64, copy=False),
    )

# Construct test scores using the identical selected recipe refit on
# train+validation. The test labels are never accessed.
if winner == "incumbent":
    test_scores = inc_test.copy()
else:
    recipe = candidate_recipes[winner]
    test = load("test")

    y_combined = np.ascontiguousarray(
        np.concatenate([
            y_train,
            np.asarray(valid.y, dtype=np.float32),
        ]),
        dtype=np.float32,
    )
    users_combined = np.ascontiguousarray(
        np.concatenate([
            train_users,
            np.asarray(valid.user_id, dtype=np.int64),
        ]),
        dtype=np.int64,
    )

    model_test_components = []
    model_component_weights = []

    for model_name, component_weight in recipe["models"]:
        cfg = CONFIG_BY_NAME[model_name]
        field_key = cfg["fields"]

        x_combined = np.ascontiguousarray(
            np.concatenate([
                x_train_cache[field_key],
                x_valid_cache[field_key],
            ], axis=0),
            dtype=np.int64,
        )
        x_test = make_matrix(test, field_key)

        model = fit_fixed(
            cfg,
            x_combined,
            y_combined,
            users_combined,
            final_model_epochs[model_name],
        )
        raw_test = predict(model, x_test)
        model_test_components.append(standardize(raw_test))
        model_component_weights.append(float(component_weight))

        del model, x_combined, x_test, raw_test
        gc.collect()

    combined_model_test = np.zeros(len(model_test_components[0]), dtype=np.float64)
    for weight, component in zip(
        model_component_weights, model_test_components
    ):
        combined_model_test += weight * component

    inc_weight = float(recipe["inc_weight"])
    model_weight = 1.0 - inc_weight
    test_scores = (
        model_weight * combined_model_test
        + inc_weight * standardize(inc_test)
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(valid_metrics["primary"]),
        "gauc": float(valid_metrics["gauc"]),
        "ndcg@5": float(valid_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    }),
    flush=True,
)