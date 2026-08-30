import os
import time
import json
import gc
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 6187
THREADS = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(THREADS)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type",
    "onehot_feat3", "onehot_feat8", "onehot_feat1",
    "onehot_feat7", "user_active_degree",
    "register_days_bucket", "register_days_range",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "hour", "is_live_streamer",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
AUX_KEYS = [
    "is_click", "is_like", "is_follow",
    "is_comment", "is_forward", "is_profile_enter",
]

EMBED_DIM = 8
BATCH_SIZE = 8192
EPOCHS = 2
AUX_WEIGHT = 0.18

cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(CAT_FIELDS)
n_tasks = 1 + len(AUX_KEYS)


def make_cat(parts):
    cols = []
    for f, off in zip(CAT_FIELDS, offsets):
        if len(parts) == 1:
            x = np.asarray(parts[0].X[f], dtype=np.int64)
        else:
            x = np.concatenate([
                np.asarray(p.X[f], dtype=np.int64) for p in parts
            ])
        cols.append(x + off)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


def numeric_statistics(parts):
    means = []
    scales = []
    for f in NUM_FIELDS:
        if len(parts) == 1:
            x = np.asarray(parts[0].num[f], dtype=np.float64)
        else:
            x = np.concatenate([
                np.asarray(p.num[f], dtype=np.float64) for p in parts
            ])
        x = np.log1p(np.maximum(np.nan_to_num(
            x, nan=0.0, posinf=0.0, neginf=0.0
        ), 0.0))
        med = float(np.median(x))
        q25, q75 = np.percentile(x, [25.0, 75.0])
        scale = max(float(q75 - q25), 0.25)
        means.append(med)
        scales.append(scale)
    return np.asarray(means, np.float32), np.asarray(scales, np.float32)


def make_num(parts, center, scale):
    cols = []
    for j, f in enumerate(NUM_FIELDS):
        if len(parts) == 1:
            x = np.asarray(parts[0].num[f], dtype=np.float32)
        else:
            x = np.concatenate([
                np.asarray(p.num[f], dtype=np.float32) for p in parts
            ])
        x = np.log1p(np.maximum(np.nan_to_num(
            x, nan=0.0, posinf=0.0, neginf=0.0
        ), 0.0))
        x = np.clip((x - center[j]) / scale[j], -6.0, 6.0)
        cols.append(x.astype(np.float32))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def train_targets(train_split):
    # Only training-split feedback outcomes are read, and only as labels.
    main = np.asarray(train_split.y, dtype=np.float32)
    aux_cols = []
    for key in AUX_KEYS:
        x = np.asarray(train_split.aux[key], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        aux_cols.append((x > 0).astype(np.float32))
    return np.ascontiguousarray(
        np.column_stack([main] + aux_cols), dtype=np.float32
    )


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.wide = nn.Embedding(total_cardinality, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.015)
        nn.init.zeros_(self.wide.weight)

    @property
    def output_dim(self):
        return n_fields * EMBED_DIM + len(NUM_FIELDS)

    def forward(self, cat, num):
        emb = self.embedding(cat).reshape(cat.shape[0], -1)
        features = torch.cat([emb, num], dim=1)
        wide = self.wide(cat).squeeze(-1).sum(dim=1)
        return features, wide


class SingleTaskMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        d = self.encoder.output_dim
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, cat, num):
        x, wide = self.encoder(cat, num)
        main = self.net(x).squeeze(1) + wide + self.bias
        return main[:, None]


class HardSharedMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        d = self.encoder.output_dim
        self.bottom = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 72),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(72, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(n_tasks)
        ])
        self.main_bias = nn.Parameter(torch.zeros(1))

    def forward(self, cat, num):
        x, wide = self.encoder(cat, num)
        h = self.bottom(x)
        outputs = torch.cat([head(h) for head in self.heads], dim=1)
        outputs[:, 0] = outputs[:, 0] + wide + self.main_bias
        return outputs


class MMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        d = self.encoder.output_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 96),
                nn.ReLU(),
                nn.Linear(96, 64),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(d, 4) for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(n_tasks)
        ])
        self.main_bias = nn.Parameter(torch.zeros(1))

    def forward(self, cat, num):
        x, wide = self.encoder(cat, num)
        expert_values = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )
        outputs = []
        for task in range(n_tasks):
            gate = torch.softmax(self.gates[task](x), dim=1)
            task_h = torch.sum(expert_values * gate[:, :, None], dim=1)
            outputs.append(self.towers[task](task_h))
        outputs = torch.cat(outputs, dim=1)
        outputs[:, 0] = outputs[:, 0] + wide + self.main_bias
        return outputs


def make_model(family):
    if family == "single_task_mlp":
        return SingleTaskMLP()
    if family == "hard_shared_multitask":
        return HardSharedMultiTask()
    if family == "mmoe":
        return MMoE()
    raise ValueError(family)


def task_positive_weights(targets):
    weights = np.ones(targets.shape[1], dtype=np.float32)
    for j in range(1, targets.shape[1]):
        p = float(np.mean(targets[:, j]))
        if 0.0 < p < 1.0:
            weights[j] = np.float32(np.clip((1.0 - p) / p, 1.0, 8.0))
    return weights


def fit_model(family, cat, num, targets, aux_mask, seed):
    torch.manual_seed(seed)
    model = make_model(family)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0022, weight_decay=2.0e-6
    )

    cat_t = torch.from_numpy(cat)
    num_t = torch.from_numpy(num)
    target_t = torch.from_numpy(targets)
    mask_t = torch.from_numpy(np.asarray(aux_mask, dtype=np.float32))
    pos_weights = torch.from_numpy(task_positive_weights(targets))

    n = len(cat)
    for epoch in range(EPOCHS):
        gen = torch.Generator()
        gen.manual_seed(seed + 31 * epoch + 1)
        order = torch.randperm(n, generator=gen)

        for st in range(0, n, BATCH_SIZE):
            idx = order[st:st + BATCH_SIZE]
            logits = model(cat_t[idx], num_t[idx])

            main_loss = nn.functional.binary_cross_entropy_with_logits(
                logits[:, 0], target_t[idx, 0]
            )
            loss = main_loss

            if logits.shape[1] > 1:
                auxiliary = nn.functional.binary_cross_entropy_with_logits(
                    logits[:, 1:],
                    target_t[idx, 1:],
                    reduction="none",
                    pos_weight=pos_weights[1:],
                )
                row_mask = mask_t[idx, None]
                denom = torch.clamp(
                    row_mask.sum() * auxiliary.shape[1], min=1.0
                )
                auxiliary_loss = (auxiliary * row_mask).sum() / denom
                loss = loss + AUX_WEIGHT * auxiliary_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict(model, cat, num):
    model.eval()
    cat_t = torch.from_numpy(cat)
    num_t = torch.from_numpy(num)
    out = np.empty(len(cat), dtype=np.float64)
    with torch.no_grad():
        for st in range(0, len(cat), BATCH_SIZE * 2):
            en = min(st + BATCH_SIZE * 2, len(cat))
            logits = model(cat_t[st:en], num_t[st:en])
            out[st:en] = logits[:, 0].cpu().numpy()
    return out


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")

center, scale = numeric_statistics([train])
cat_train = make_cat([train])
num_train = make_num([train], center, scale)
cat_valid = make_cat([valid])
num_valid = make_num([valid], center, scale)

targets_train = train_targets(train)
aux_mask_train = np.ones(len(train.user_id), dtype=np.float32)

families = [
    "single_task_mlp",
    "hard_shared_multitask",
    "mmoe",
]
raw_predictions = {}

for i, family in enumerate(families):
    model = fit_model(
        family,
        cat_train,
        num_train,
        targets_train,
        aux_mask_train,
        SEED + 100 * i,
    )
    raw_predictions[family] = predict(model, cat_valid, num_valid)
    del model
    gc.collect()

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores = {"incumbent": float(inc_metrics["primary"])}
candidate_specs = {"incumbent": ("incumbent", 0.0, 1.0)}

best_name = "incumbent"
best_scores = inc_valid.copy()
best_raw = None
best_metrics = inc_metrics
best_spec = candidate_specs["incumbent"]

inc_std = max(float(np.std(inc_valid)), 1.0e-8)
blend_alphas = [0.15, 0.25, 0.35, 0.50, 0.65]

for family, raw in raw_predictions.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_scores[family] = float(raw_metrics["primary"])
    candidate_specs[family] = (family, 1.0, 1.0)

    if raw_metrics["primary"] > best_metrics["primary"]:
        best_name = family
        best_scores = raw.copy()
        best_raw = raw.copy()
        best_metrics = raw_metrics
        best_spec = candidate_specs[family]

    raw_std = max(float(np.std(raw)), 1.0e-8)
    score_scale = inc_std / raw_std
    scaled_raw = raw * score_scale

    for alpha in blend_alphas:
        blended = (1.0 - alpha) * inc_valid + alpha * scaled_raw
        metrics = evaluate(valid.user_id, y_valid, blended)
        name = "%s_blend_%.2f" % (family, alpha)
        candidate_scores[name] = float(metrics["primary"])
        candidate_specs[name] = (family, alpha, score_scale)

        if metrics["primary"] > best_metrics["primary"]:
            best_name = name
            best_scores = blended.copy()
            best_raw = raw.copy()
            best_metrics = metrics
            best_spec = candidate_specs[name]

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)
print(
    "FINDINGS train_aux_targets_only=true valid_aux_read=false "
    "single=%.6f hard_shared=%.6f mmoe=%.6f selected=%s"
    % (
        candidate_scores["single_task_mlp"],
        candidate_scores["hard_shared_multitask"],
        candidate_scores["mmoe"],
        best_name,
    ),
    flush=True,
)

selected_family, selected_alpha, selected_scale = best_spec
test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)

if selected_family == "incumbent":
    test_scores = inc_test.copy()
else:
    # Refit on train + validation main labels. Validation auxiliary outcomes
    # are deliberately never accessed; their auxiliary loss is masked out.
    y_tv = np.concatenate([
        y_train, y_valid.astype(np.float32)
    ]).astype(np.float32)

    center_tv, scale_tv = numeric_statistics([train, valid])
    cat_tv = make_cat([train, valid])
    num_tv = make_num([train, valid], center_tv, scale_tv)

    n_train = len(y_train)
    n_valid = len(y_valid)
    targets_tv = np.zeros((n_train + n_valid, n_tasks), dtype=np.float32)
    targets_tv[:n_train] = targets_train
    targets_tv[:, 0] = y_tv

    aux_mask_tv = np.concatenate([
        np.ones(n_train, dtype=np.float32),
        np.zeros(n_valid, dtype=np.float32),
    ])

    cat_test = make_cat([test])
    num_test = make_num([test], center_tv, scale_tv)

    family_index = families.index(selected_family)
    refit_model = fit_model(
        selected_family,
        cat_tv,
        num_tv,
        targets_tv,
        aux_mask_tv,
        SEED + 100 * family_index,
    )
    test_raw = predict(refit_model, cat_test, num_test)

    if selected_alpha >= 1.0:
        test_scores = test_raw
    else:
        test_scores = (
            (1.0 - selected_alpha) * inc_test
            + selected_alpha * selected_scale * test_raw
        )

    del refit_model, cat_tv, num_tv, cat_test, num_test
    gc.collect()

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if selected_family != "incumbent" and selected_alpha < 1.0:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        best_metrics["primary"],
        best_metrics["gauc"],
        best_metrics["ndcg@5"],
        elapsed,
    )
)