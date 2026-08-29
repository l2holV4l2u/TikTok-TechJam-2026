import os
import gc
import json
import time
import random
import shutil
import tempfile
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

UNION_FIELDS = BASE_FIELDS + [
    "onehot_feat3",
    "onehot_feat8",
    "video_type",
    "user_active_degree",
]

FIELD_SETS = {
    "core7": [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "duration_bucket",
        "tag",
        "upload_type",
    ],
    "base9": BASE_FIELDS,
    "plus10": BASE_FIELDS + ["onehot_feat3"],
    "plus11": BASE_FIELDS + ["onehot_feat3", "onehot_feat8"],
    "plus13": UNION_FIELDS,
}

cards = [int(FEATURE_CARDINALITIES[f]) for f in UNION_FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
field_position = {name: i for i, name in enumerate(UNION_FIELDS)}
field_indices = {
    name: np.asarray([field_position[f] for f in fields], dtype=np.int64)
    for name, fields in FIELD_SETS.items()
}


def make_features(split):
    x = np.column_stack([
        np.asarray(split.X[f], dtype=np.int64) for f in UNION_FIELDS
    ])
    x += offsets[None, :]
    return np.ascontiguousarray(x, dtype=np.int64)


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x0, x):
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x + x0 * scale + self.bias


class DCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.field_idx = torch.from_numpy(
            field_indices[cfg["fields"]].copy()
        )
        self.n_fields = len(self.field_idx)
        self.k = int(cfg["k"])
        dim = self.n_fields * self.k

        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, self.k)
        self.bias = nn.Parameter(torch.zeros(1))

        self.cross_layers = nn.ModuleList([
            CrossLayer(dim) for _ in range(int(cfg["cross_depth"]))
        ])

        hidden = list(cfg["hidden"])
        layers = []
        in_dim = dim
        for width in hidden:
            layers.append(nn.Linear(in_dim, int(width)))
            layers.append(nn.ReLU())
            if cfg["dropout"] > 0:
                layers.append(nn.Dropout(float(cfg["dropout"])))
            in_dim = int(width)
        self.deep = nn.Sequential(*layers)
        self.deep_out_dim = in_dim

        self.output = nn.Linear(dim + self.deep_out_dim, 1)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x_all):
        idx = self.field_idx.to(x_all.device)
        x = x_all.index_select(1, idx)

        emb = self.embedding(x)
        x0 = emb.flatten(start_dim=1)

        cross = x0
        for layer in self.cross_layers:
            cross = layer(x0, cross)

        deep = self.deep(x0)
        wide = self.linear(x).sum(dim=1).squeeze(1)
        interaction = self.output(
            torch.cat([cross, deep], dim=1)
        ).squeeze(1)

        return self.bias + wide + interaction


def make_model(cfg):
    torch.manual_seed(SEED)
    return DCN(cfg)


def make_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )


def train_one_epoch(model, optimizer, x, y, cfg, epoch_number):
    model.train()
    generator = torch.Generator()
    generator.manual_seed(SEED + 1009 * int(epoch_number))
    order = torch.randperm(x.shape[0], generator=generator)

    batch_size = int(cfg["batch_size"])
    pos_weight = torch.tensor(
        float(cfg["pos_weight"]), dtype=torch.float32
    )

    for begin in range(0, x.shape[0], batch_size):
        idx = order[begin:begin + batch_size]
        xb = x[idx]
        yb = y[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            yb,
            pos_weight=pos_weight,
        )
        loss.backward()

        clip = float(cfg.get("grad_clip", 0.0))
        if clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    out = np.empty(x_np.shape[0], dtype=np.float64)
    for begin in range(0, x_np.shape[0], PRED_BATCH):
        end = min(begin + PRED_BATCH, x_np.shape[0])
        xb = torch.from_numpy(x_np[begin:end])
        out[begin:end] = model(xb).cpu().numpy().astype(np.float64)
    return out


BLEND_ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]


def evaluate_blends(valid, candidate, incumbent):
    results = {}
    best_score = -np.inf
    best_alpha = 1.0
    best_pred = None
    best_metrics = None

    for alpha in BLEND_ALPHAS:
        pred = alpha * candidate + (1.0 - alpha) * incumbent
        metrics = evaluate(valid.user_id, valid.y, pred)
        primary = float(metrics["primary"])
        results[str(alpha)] = primary
        if primary > best_score:
            best_score = primary
            best_alpha = float(alpha)
            best_pred = pred.copy()
            best_metrics = metrics

    return best_score, best_alpha, best_pred, best_metrics, results


def save_checkpoint(path, model, optimizer, completed_epochs):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epochs": int(completed_epochs),
        },
        path,
    )


def load_checkpoint(path, cfg):
    model = make_model(cfg)
    optimizer = make_optimizer(model, cfg)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return model, optimizer, int(obj["completed_epochs"])


CONFIGS = [
    {
        "name": "central_k16_c2",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "central_k12_c2",
        "fields": "base9", "k": 12, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "central_k20_c2",
        "fields": "base9", "k": 20, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "k16_c3",
        "fields": "base9", "k": 16, "cross_depth": 3,
        "hidden": [128, 64], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "k16_c1",
        "fields": "base9", "k": 16, "cross_depth": 1,
        "hidden": [128, 64], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "narrow_96_48",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [96, 48], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "wide_192_96",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [192, 96], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "deep_128_64_32",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [128, 64, 32], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "drop010",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.10,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "drop020_wd",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.20,
        "lr": 0.0010, "weight_decay": 1e-5,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "lr0007_wd",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.05,
        "lr": 0.0007, "weight_decay": 1e-6,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "lr0014",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.0,
        "lr": 0.0014, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
        "grad_clip": 5.0,
    },
    {
        "name": "batch8192",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.0,
        "lr": 0.0014, "weight_decay": 0.0,
        "batch_size": 8192, "pos_weight": 1.0,
    },
    {
        "name": "positive095",
        "fields": "base9", "k": 16, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 0.95,
    },
    {
        "name": "fields_plus10",
        "fields": "plus10", "k": 16, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.05,
        "lr": 0.0010, "weight_decay": 1e-6,
        "batch_size": 4096, "pos_weight": 1.0,
    },
    {
        "name": "fields_core7",
        "fields": "core7", "k": 16, "cross_depth": 2,
        "hidden": [128, 64], "dropout": 0.0,
        "lr": 0.0010, "weight_decay": 0.0,
        "batch_size": 4096, "pos_weight": 1.0,
    },
]

config_by_name = {c["name"]: c for c in CONFIGS}

train = load("train")
valid = load("valid")

x_train_np = make_features(train)
x_valid_np = make_features(valid)
x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
inc_primary = float(inc_metrics["primary"])

checkpoint_dir = tempfile.mkdtemp(prefix="dcn_halving_")
checkpoint_paths = {}

rung1 = {}

# Rung 1: all 16 configurations, one complete epoch.
for cfg in CONFIGS:
    model = make_model(cfg)
    optimizer = make_optimizer(model, cfg)
    train_one_epoch(model, optimizer, x_train, y_train, cfg, 1)

    raw = predict(model, x_valid_np)
    score, alpha, _, _, blend_details = evaluate_blends(
        valid, raw, inc_valid
    )
    rung1[cfg["name"]] = {
        "score": float(score),
        "alpha": float(alpha),
        "blends": blend_details,
    }

    path = os.path.join(checkpoint_dir, cfg["name"] + ".pt")
    save_checkpoint(path, model, optimizer, 1)
    checkpoint_paths[cfg["name"]] = path

    del model, optimizer, raw
    gc.collect()

survivors1 = sorted(
    rung1,
    key=lambda n: rung1[n]["score"],
    reverse=True,
)[:6]

for cfg in CONFIGS:
    if cfg["name"] not in survivors1:
        try:
            os.remove(checkpoint_paths[cfg["name"]])
        except OSError:
            pass

print("FINDINGS rung1 " + json.dumps({
    n: round(rung1[n]["score"], 6)
    for n in sorted(rung1, key=lambda x: rung1[x]["score"], reverse=True)
}))

rung2 = {}

# Rung 2: top six continue to a total of three epochs.
for name in survivors1:
    cfg = config_by_name[name]
    model, optimizer, completed = load_checkpoint(
        checkpoint_paths[name], cfg
    )

    for epoch in range(completed + 1, 4):
        train_one_epoch(
            model, optimizer, x_train, y_train, cfg, epoch
        )

    raw = predict(model, x_valid_np)
    score, alpha, _, _, blend_details = evaluate_blends(
        valid, raw, inc_valid
    )
    rung2[name] = {
        "score": float(score),
        "alpha": float(alpha),
        "blends": blend_details,
    }

    save_checkpoint(checkpoint_paths[name], model, optimizer, 3)
    del model, optimizer, raw
    gc.collect()

survivors2 = sorted(
    rung2,
    key=lambda n: rung2[n]["score"],
    reverse=True,
)[:2]

for name in survivors1:
    if name not in survivors2:
        try:
            os.remove(checkpoint_paths[name])
        except OSError:
            pass

print("FINDINGS rung2 " + json.dumps({
    n: round(rung2[n]["score"], 6)
    for n in sorted(rung2, key=lambda x: rung2[x]["score"], reverse=True)
}))

rung3 = {}
rung3_predictions = {}

# Rung 3: top two continue through epochs 4, 5, and 6. Retain each
# candidate's best full-budget epoch rather than selecting a single noisy
# endpoint.
for name in survivors2:
    cfg = config_by_name[name]
    model, optimizer, completed = load_checkpoint(
        checkpoint_paths[name], cfg
    )

    epoch_records = {}
    best_score = -np.inf
    best_epoch = None
    best_alpha = None
    best_pred = None
    best_metrics = None

    for epoch in range(completed + 1, 7):
        train_one_epoch(
            model, optimizer, x_train, y_train, cfg, epoch
        )
        raw = predict(model, x_valid_np)
        score, alpha, pred, metrics, blend_details = evaluate_blends(
            valid, raw, inc_valid
        )
        epoch_records[str(epoch)] = {
            "score": float(score),
            "alpha": float(alpha),
            "blends": blend_details,
        }

        if score > best_score:
            best_score = float(score)
            best_epoch = int(epoch)
            best_alpha = float(alpha)
            best_pred = pred.copy()
            best_metrics = metrics

        del raw, pred
        gc.collect()

    rung3[name] = {
        "score": best_score,
        "epoch": best_epoch,
        "alpha": best_alpha,
        "epochs": epoch_records,
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
    }
    rung3_predictions[name] = best_pred

    del model, optimizer
    gc.collect()

print("FINDINGS rung3 " + json.dumps(rung3))

winner_name = max(rung3, key=lambda n: rung3[n]["score"])
winner_record = rung3[winner_name]
winner_gain = float(winner_record["score"] - inc_primary)

candidate_log = {
    "incumbent": inc_primary,
}
for name in rung3:
    candidate_log[
        name + "_epoch" + str(rung3[name]["epoch"])
    ] = float(rung3[name]["score"])

# The requested safeguard: do not ship a configuration-search fluctuation
# unless it clears 0.0008 over the trusted incumbent.
use_winner = winner_gain > 0.0008

if use_winner:
    valid_scores = rung3_predictions[winner_name]
    chosen_cfg = config_by_name[winner_name]
    chosen_epoch = int(winner_record["epoch"])
    chosen_alpha = float(winner_record["alpha"])
    chosen_name = winner_name
else:
    valid_scores = inc_valid.copy()
    chosen_cfg = None
    chosen_epoch = 0
    chosen_alpha = 0.0
    chosen_name = "incumbent"

metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("FINDINGS selection " + json.dumps({
    "winner": winner_name,
    "winner_gain": winner_gain,
    "threshold": 0.0008,
    "shipped": chosen_name,
    "epoch": chosen_epoch,
    "alpha": chosen_alpha,
}))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Produce test scores. If the tuned configuration wins, refit its identical
# recipe from scratch on train+validation for the selected epoch count.
test = load("test")

if use_winner:
    x_test_np = make_features(test)
    x_combined_np = np.concatenate(
        [x_train_np, x_valid_np], axis=0
    )
    y_combined_np = np.concatenate([
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ])

    x_combined = torch.from_numpy(x_combined_np)
    y_combined = torch.from_numpy(y_combined_np)

    final_model = make_model(chosen_cfg)
    final_optimizer = make_optimizer(final_model, chosen_cfg)

    for epoch in range(1, chosen_epoch + 1):
        train_one_epoch(
            final_model,
            final_optimizer,
            x_combined,
            y_combined,
            chosen_cfg,
            epoch,
        )

    raw_test = predict(final_model, x_test_np)
    te_scores = (
        chosen_alpha * raw_test
        + (1.0 - chosen_alpha) * inc_test
    )

    del final_model, final_optimizer
    del x_combined, y_combined, x_combined_np, y_combined_np
    del x_test_np, raw_test
else:
    if len(test.user_id) != len(inc_test):
        raise RuntimeError("Incumbent test prediction length mismatch")
    te_scores = inc_test.copy()

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(te_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidate_log))

try:
    shutil.rmtree(checkpoint_dir)
except OSError:
    pass

elapsed = float(time.time() - START)
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}))