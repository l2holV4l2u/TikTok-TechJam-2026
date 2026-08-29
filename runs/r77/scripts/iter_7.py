import os
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
PRED_BATCH = 65536
THREADS = min(8, os.cpu_count() or 1)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

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
FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}
CARDS = np.asarray(
    [int(FEATURE_CARDINALITIES[f]) for f in FIELDS],
    dtype=np.int64,
)
OFFSETS = np.cumsum(
    np.concatenate([np.zeros(1, dtype=np.int64), CARDS[:-1]])
)
TOTAL_CARDINALITY = int(CARDS.sum())


def make_x(split):
    x = np.column_stack([
        np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
    ])
    x += OFFSETS[None, :]
    return np.ascontiguousarray(x, dtype=np.int64)


def recent_half_mask(dates):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    threshold = unique_dates[len(unique_dates) // 2]
    return dates >= threshold


def row_weights(y, users, dates, pos_weight, user_power, recency):
    y = np.asarray(y, dtype=np.float32)
    users = np.asarray(users, dtype=np.int64)
    dates = np.asarray(dates, dtype=np.int32)

    counts = np.bincount(users, minlength=int(users.max()) + 1)
    w = np.power(
        np.maximum(counts[users], 1).astype(np.float32),
        -float(user_power),
    )

    if recency > 0:
        date_values = np.unique(dates)
        date_to_age = {
            int(d): len(date_values) - 1 - i
            for i, d in enumerate(date_values)
        }
        ages = np.fromiter(
            (date_to_age[int(d)] for d in dates),
            dtype=np.float32,
            count=len(dates),
        )
        w *= np.exp(-float(recency) * ages)

    w *= np.where(y > 0.5, float(pos_weight), 1.0).astype(np.float32)
    w /= max(float(w.mean()), 1e-8)
    return np.ascontiguousarray(w, dtype=np.float32)


class DCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.active = list(config["active"])
        self.n_active = len(self.active)
        self.k = int(config["k"])

        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, self.k)
        self.bias = nn.Parameter(torch.zeros(1))

        input_dim = self.n_active * self.k
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim))
            for _ in range(int(config["cross_layers"]))
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(int(config["cross_layers"]))
        ])

        layers = []
        previous = input_dim
        for width in config["deep"]:
            layers.append(nn.Linear(previous, int(width)))
            layers.append(nn.ReLU())
            if config["dropout"] > 0:
                layers.append(nn.Dropout(float(config["dropout"])))
            previous = int(width)
        self.deep = nn.Sequential(*layers)

        self.output = nn.Linear(input_dim + previous, 1)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)

    def forward(self, x):
        xa = x[:, self.active]
        embeddings = self.embedding(xa)
        x0 = embeddings.flatten(start_dim=1)
        cross = x0

        for w, b in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(cross * w, dim=1, keepdim=True)
            cross = x0 * scalar + b + cross

        deep = self.deep(x0)
        wide = self.linear(xa).sum(dim=1).squeeze(1)
        interaction = self.output(
            torch.cat([cross, deep], dim=1)
        ).squeeze(1)
        return self.bias + wide + interaction


ALL = tuple(range(9))
NO_TIME = tuple(i for i, f in enumerate(FIELDS) if f != "hour")
CORE7 = tuple(
    i for i, f in enumerate(FIELDS)
    if f not in ("hour", "music_type")
)

# Deliberately arranged in local neighborhoods rather than sixteen unrelated
# points, making broad plateaus preferable to isolated validation peaks.
CONFIGS = [
    dict(name="base_a", k=12, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0010, wd=1e-6, batch=4096,
         pos_weight=1.0, user_power=0.0, recency=0.0, active=ALL),
    dict(name="base_b", k=16, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0010, wd=1e-6, batch=4096,
         pos_weight=1.0, user_power=0.0, recency=0.0, active=ALL),
    dict(name="base_c", k=12, cross_layers=3, deep=(128, 64),
         dropout=0.05, lr=0.0010, wd=1e-6, batch=4096,
         pos_weight=1.0, user_power=0.0, recency=0.0, active=ALL),
    dict(name="base_d", k=16, cross_layers=3, deep=(128, 64),
         dropout=0.10, lr=0.0008, wd=1e-5, batch=4096,
         pos_weight=1.0, user_power=0.0, recency=0.0, active=ALL),

    dict(name="userbal_a", k=12, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0010, wd=1e-6, batch=4096,
         pos_weight=1.0, user_power=0.15, recency=0.0, active=ALL),
    dict(name="userbal_b", k=16, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0010, wd=1e-6, batch=4096,
         pos_weight=1.0, user_power=0.15, recency=0.0, active=ALL),
    dict(name="userbal_c", k=12, cross_layers=2, deep=(128, 64),
         dropout=0.10, lr=0.0008, wd=1e-5, batch=8192,
         pos_weight=1.0, user_power=0.25, recency=0.0, active=ALL),
    dict(name="userbal_d", k=16, cross_layers=3, deep=(128, 64),
         dropout=0.10, lr=0.0008, wd=1e-5, batch=8192,
         pos_weight=1.0, user_power=0.25, recency=0.0, active=ALL),

    dict(name="recent_a", k=12, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0010, wd=1e-6, batch=4096,
         pos_weight=1.0, user_power=0.0, recency=0.035, active=ALL),
    dict(name="recent_b", k=16, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0010, wd=1e-6, batch=4096,
         pos_weight=1.0, user_power=0.0, recency=0.035, active=ALL),
    dict(name="recent_bal_a", k=12, cross_layers=2, deep=(128, 64),
         dropout=0.05, lr=0.0010, wd=1e-5, batch=4096,
         pos_weight=1.0, user_power=0.15, recency=0.035, active=ALL),
    dict(name="recent_bal_b", k=16, cross_layers=3, deep=(128, 64),
         dropout=0.10, lr=0.0008, wd=1e-5, batch=8192,
         pos_weight=1.0, user_power=0.15, recency=0.035, active=ALL),

    dict(name="poslow_all", k=12, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0012, wd=1e-6, batch=4096,
         pos_weight=0.85, user_power=0.10, recency=0.02, active=ALL),
    dict(name="poshigh_all", k=12, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0012, wd=1e-6, batch=4096,
         pos_weight=1.20, user_power=0.10, recency=0.02, active=ALL),
    dict(name="notime", k=12, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0010, wd=1e-6, batch=4096,
         pos_weight=1.0, user_power=0.15, recency=0.02, active=NO_TIME),
    dict(name="core7", k=16, cross_layers=2, deep=(96, 48),
         dropout=0.05, lr=0.0010, wd=1e-6, batch=4096,
         pos_weight=1.0, user_power=0.15, recency=0.02, active=CORE7),
]


def make_model(config, seed):
    torch.manual_seed(seed)
    return DCN(config)


def train_epochs(model, config, x_np, y_np, weights_np, epochs, seed):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["wd"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs))
    )

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    weights = torch.from_numpy(np.asarray(weights_np, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(seed)

    batch_size = int(config["batch"])
    for _ in range(int(epochs)):
        order = torch.randperm(x.shape[0], generator=generator)
        for begin in range(0, x.shape[0], batch_size):
            idx = order[begin:begin + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            loss = torch.mean(losses * weights[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    del optimizer, scheduler, x, y, weights
    gc.collect()


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    for begin in range(0, x_np.shape[0], PRED_BATCH):
        end = min(begin + PRED_BATCH, x_np.shape[0])
        xb = torch.from_numpy(x_np[begin:end])
        result[begin:end] = model(xb).cpu().numpy().astype(np.float64)
    return result


def metric_primary(valid, scores):
    return float(evaluate(valid.user_id, valid.y, scores)["primary"])


def normalized(scores):
    scores = np.asarray(scores, dtype=np.float64)
    sd = float(scores.std())
    if not np.isfinite(sd) or sd < 1e-12:
        return scores - float(scores.mean())
    return (scores - float(scores.mean())) / sd


train = load("train")
valid = load("valid")

x_train = make_x(train)
x_valid = make_x(valid)
y_train = np.asarray(train.y, dtype=np.float32)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")
inc_primary = metric_primary(valid, inc_valid)

recent_mask = recent_half_mask(train.date)
x_recent = x_train[recent_mask]
y_recent = y_train[recent_mask]
u_recent = np.asarray(train.user_id)[recent_mask]
d_recent = np.asarray(train.date)[recent_mask]

states = {}
rung1 = {}

# Rung 1: one epoch on the recent half of the fitting period.
for i, config in enumerate(CONFIGS):
    model = make_model(config, SEED + i)
    weights = row_weights(
        y_recent, u_recent, d_recent,
        config["pos_weight"],
        config["user_power"],
        config["recency"],
    )
    train_epochs(
        model, config, x_recent, y_recent, weights,
        epochs=1, seed=SEED + 1000 + i,
    )
    scores = predict(model, x_valid)
    rung1[config["name"]] = metric_primary(valid, scores)
    states[i] = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    del model, scores, weights
    gc.collect()

top6 = sorted(
    range(len(CONFIGS)),
    key=lambda i: rung1[CONFIGS[i]["name"]],
    reverse=True,
)[:6]

rung2 = {}
states2 = {}

# Rung 2: two additional epochs on all training rows.
for rank, i in enumerate(top6):
    config = CONFIGS[i]
    model = make_model(config, SEED + i)
    model.load_state_dict(states[i])
    weights = row_weights(
        y_train, train.user_id, train.date,
        config["pos_weight"],
        config["user_power"],
        config["recency"],
    )
    train_epochs(
        model, config, x_train, y_train, weights,
        epochs=2, seed=SEED + 2000 + i,
    )
    scores = predict(model, x_valid)
    rung2[config["name"]] = metric_primary(valid, scores)
    states2[i] = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    del model, scores, weights
    gc.collect()

states.clear()
top2 = sorted(
    top6,
    key=lambda i: rung2[CONFIGS[i]["name"]],
    reverse=True,
)[:2]

rung3 = {}
final_states = {}
final_scores = {}

# Rung 3: two more full-data epochs, for five epochs total.
for i in top2:
    config = CONFIGS[i]
    model = make_model(config, SEED + i)
    model.load_state_dict(states2[i])
    weights = row_weights(
        y_train, train.user_id, train.date,
        config["pos_weight"],
        config["user_power"],
        config["recency"],
    )
    train_epochs(
        model, config, x_train, y_train, weights,
        epochs=2, seed=SEED + 3000 + i,
    )
    scores = predict(model, x_valid)
    rung3[config["name"]] = metric_primary(valid, scores)
    final_scores[i] = scores
    final_states[i] = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    del model, weights
    gc.collect()

# Search only fusion weights of the already-established incumbent/component
# bundle. Scores are standardized to prevent arbitrary logit scale deciding
# the blend.
inc_z = normalized(inc_valid)
blend_candidates = {}
best_primary = inc_primary
best_i = None
best_alpha = 0.0
best_valid_scores = inc_valid.copy()

for i in top2:
    candidate_z = normalized(final_scores[i])
    name = CONFIGS[i]["name"]
    for alpha in np.linspace(0.1, 1.0, 10):
        blended = (1.0 - alpha) * inc_z + alpha * candidate_z
        primary = metric_primary(valid, blended)
        blend_candidates[f"{name}@{alpha:.1f}"] = primary
        if primary > best_primary:
            best_primary = primary
            best_i = i
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()

# Do not ship a validation-selected fluctuation smaller than the stated
# within-run noise threshold.
if best_primary <= inc_primary + 0.0008:
    best_i = None
    best_alpha = 0.0
    best_primary = inc_primary
    best_valid_scores = inc_valid.copy()

validation_metrics = evaluate(
    valid.user_id, valid.y, best_valid_scores
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Produce test scores. If no configuration clears the gate, preserve the
# trusted incumbent exactly. Otherwise refit the selected recipe on
# train+validation without reading any test labels.
test = load("test")

if best_i is None:
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
    winner_name = "trusted_incumbent"
else:
    config = CONFIGS[best_i]
    x_test = make_x(test)

    x_fit = np.concatenate([x_train, x_valid], axis=0)
    y_fit = np.concatenate([
        y_train,
        np.asarray(valid.y, dtype=np.float32),
    ])
    u_fit = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    d_fit = np.concatenate([
        np.asarray(train.date, dtype=np.int32),
        np.asarray(valid.date, dtype=np.int32),
    ])

    fit_recent_mask = recent_half_mask(d_fit)
    model = make_model(config, SEED + best_i)

    recent_weights = row_weights(
        y_fit[fit_recent_mask],
        u_fit[fit_recent_mask],
        d_fit[fit_recent_mask],
        config["pos_weight"],
        config["user_power"],
        config["recency"],
    )
    train_epochs(
        model,
        config,
        x_fit[fit_recent_mask],
        y_fit[fit_recent_mask],
        recent_weights,
        epochs=1,
        seed=SEED + 1000 + best_i,
    )

    full_weights = row_weights(
        y_fit, u_fit, d_fit,
        config["pos_weight"],
        config["user_power"],
        config["recency"],
    )
    train_epochs(
        model, config, x_fit, y_fit, full_weights,
        epochs=4, seed=SEED + 2000 + best_i,
    )

    candidate_test = predict(model, x_test)
    incumbent_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
    test_scores = (
        (1.0 - best_alpha) * normalized(incumbent_test)
        + best_alpha * normalized(candidate_test)
    )
    winner_name = f"{config['name']}@{best_alpha:.1f}"

    del model, x_fit, y_fit, u_fit, d_fit, x_test
    gc.collect()

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("FINDINGS " + json.dumps({
    "incumbent_primary": inc_primary,
    "rung1_best": max(rung1.items(), key=lambda z: z[1]),
    "rung2": rung2,
    "rung3": rung3,
    "winner": winner_name,
    "selected_gain": best_primary - inc_primary,
}, sort_keys=True))

compact_candidates = {
    "incumbent": inc_primary,
    **{f"r3_{k}": v for k, v in rung3.items()},
    "selected": float(validation_metrics["primary"]),
}
print("CANDIDATES " + json.dumps(compact_candidates, sort_keys=True))

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(validation_metrics["primary"]),
    "gauc": float(validation_metrics["gauc"]),
    "ndcg@5": float(validation_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))