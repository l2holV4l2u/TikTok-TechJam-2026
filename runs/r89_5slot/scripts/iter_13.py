import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 28471
BATCH = 8192
PRED_BATCH = 65536
EPOCHS = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def make_features(s):
    cats = np.column_stack([
        np.asarray(s.X[name], dtype=np.int64) for name in CAT_FIELDS
    ]).astype(np.int64)

    nums = []
    for name in NUM_FIELDS:
        x = np.asarray(s.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        nums.append(x)

    hour_id = np.asarray(s.X["hour"], dtype=np.float32)
    hour = np.mod(np.maximum(hour_id - 1.0, 0.0), 24.0)
    angle = 2.0 * np.pi * hour / 24.0

    duration = np.nan_to_num(
        np.asarray(s.num["duration_ms"], dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    duration_log = np.log1p(np.maximum(duration, 0.0))

    nums.extend([
        np.sin(angle).astype(np.float32),
        np.cos(angle).astype(np.float32),
        (duration_log < np.log1p(15000.0)).astype(np.float32),
        (duration_log > np.log1p(120000.0)).astype(np.float32),
    ])

    x = np.column_stack(nums).astype(np.float32)
    return cats, x


def fit_scaler(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def scale_features(x, mean, std):
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


class FeatureEncoder(nn.Module):
    def __init__(self, cards, n_num, emb_dim=8):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim) for card in cards
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)
        self.output_dim = len(cards) * emb_dim + n_num

    def forward(self, cats, nums):
        embs = [
            self.embeddings[j](cats[:, j])
            for j in range(cats.shape[1])
        ]
        return torch.cat(embs + [nums], dim=1)


def make_expert(inp, hidden=64):
    return nn.Sequential(
        nn.Linear(inp, 96),
        nn.SiLU(),
        nn.Linear(96, hidden),
        nn.SiLU(),
    )


class SharedBottom(nn.Module):
    def __init__(self, cards, n_num, priors):
        super().__init__()
        self.encoder = FeatureEncoder(cards, n_num)
        d = self.encoder.output_dim
        self.bottom = nn.Sequential(
            nn.Linear(d, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(3)])
        self._init_heads(priors)

    def _init_heads(self, priors):
        for head, p in zip(self.heads, priors):
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(
                head.bias,
                float(np.log(p / (1.0 - p)))
            )

    def forward(self, cats, nums):
        h = self.bottom(self.encoder(cats, nums))
        return torch.cat([head(h) for head in self.heads], dim=1)


class MMoE(nn.Module):
    def __init__(self, cards, n_num, priors):
        super().__init__()
        self.encoder = FeatureEncoder(cards, n_num)
        d = self.encoder.output_dim
        self.experts = nn.ModuleList([make_expert(d) for _ in range(4)])
        self.gates = nn.ModuleList([nn.Linear(d, 4) for _ in range(3)])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            for _ in range(3)
        ])
        for tower, p in zip(self.towers, priors):
            nn.init.constant_(
                tower[-1].bias,
                float(np.log(p / (1.0 - p)))
            )

    def forward(self, cats, nums):
        z = self.encoder(cats, nums)
        experts = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(z), dim=1).unsqueeze(2)
            h = (weights * experts).sum(dim=1)
            outputs.append(tower(h))
        return torch.cat(outputs, dim=1)


class PLE(nn.Module):
    def __init__(self, cards, n_num, priors):
        super().__init__()
        self.encoder = FeatureEncoder(cards, n_num)
        d = self.encoder.output_dim

        self.shared_experts = nn.ModuleList([
            make_expert(d) for _ in range(2)
        ])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([make_expert(d) for _ in range(2)])
            for _ in range(3)
        ])
        self.gates = nn.ModuleList([nn.Linear(d, 4) for _ in range(3)])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            for _ in range(3)
        ])

        for tower, p in zip(self.towers, priors):
            nn.init.constant_(
                tower[-1].bias,
                float(np.log(p / (1.0 - p)))
            )

    def forward(self, cats, nums):
        z = self.encoder(cats, nums)
        shared = [expert(z) for expert in self.shared_experts]
        outputs = []
        for task in range(3):
            specific = [
                expert(z) for expert in self.task_experts[task]
            ]
            experts = torch.stack(shared + specific, dim=1)
            weights = torch.softmax(
                self.gates[task](z), dim=1
            ).unsqueeze(2)
            h = (weights * experts).sum(dim=1)
            outputs.append(self.towers[task](h))
        return torch.cat(outputs, dim=1)


def build_model(family, cards, n_num, priors):
    if family == "shared_bottom":
        return SharedBottom(cards, n_num, priors)
    if family == "mmoe":
        return MMoE(cards, n_num, priors)
    if family == "ple":
        return PLE(cards, n_num, priors)
    raise ValueError(family)


def fit_model(family, cats, nums, targets, aux_mask, seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]

    masked_aux = targets[aux_mask, 1:]
    priors = [
        float(np.clip(targets[:, 0].mean(), 1e-4, 1.0 - 1e-4)),
        float(np.clip(masked_aux[:, 0].mean(), 1e-4, 1.0 - 1e-4)),
        float(np.clip(masked_aux[:, 1].mean(), 1e-4, 1.0 - 1e-4)),
    ]

    model = build_model(family, cards, nums.shape[1], priors)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0018, weight_decay=2e-5
    )

    ct = torch.from_numpy(cats)
    xt = torch.from_numpy(nums)
    yt = torch.from_numpy(targets.astype(np.float32, copy=False))
    mt = torch.from_numpy(aux_mask.astype(np.bool_, copy=False))

    n = len(targets)
    generator = torch.Generator()
    generator.manual_seed(SEED + 1000 + seed_offset)

    for epoch in range(EPOCHS):
        perm = torch.randperm(n, generator=generator)
        model.train()

        for st in range(0, n, BATCH):
            idx = perm[st:min(st + BATCH, n)]
            logits = model(ct[idx], xt[idx])

            primary_loss = F.binary_cross_entropy_with_logits(
                logits[:, 0], yt[idx, 0]
            )

            batch_mask = mt[idx]
            if bool(batch_mask.any()):
                click_loss = F.binary_cross_entropy_with_logits(
                    logits[batch_mask, 1], yt[idx[batch_mask], 1]
                )
                like_loss = F.binary_cross_entropy_with_logits(
                    logits[batch_mask, 2], yt[idx[batch_mask], 2]
                )
                loss = primary_loss + 0.18 * click_loss + 0.18 * like_loss
            else:
                loss = primary_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, cats, nums):
    model.eval()
    out = np.empty(len(cats), dtype=np.float32)
    for st in range(0, len(cats), PRED_BATCH):
        en = min(st + PRED_BATCH, len(cats))
        logits = model(
            torch.from_numpy(cats[st:en]),
            torch.from_numpy(nums[st:en])
        )
        out[st:en] = logits[:, 0].cpu().numpy()
    return out


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        sd = 1.0
    return (x - float(x.mean())) / sd


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(new_user)
    counts = np.diff(np.r_[starts, n])

    positions = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, counts)
    )
    repeated_counts = np.repeat(counts, counts)
    ranks = positions / np.maximum(repeated_counts - 1, 1)
    ranks[repeated_counts == 1] = 0.5

    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


# Only the training split's post-impression outcomes are accessed.
train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_click = (
    np.asarray(train.aux["is_click"]).reshape(-1) > 0
).astype(np.float32)
train_like = (
    np.asarray(train.aux["is_like"]).reshape(-1) > 0
).astype(np.float32)

train_targets = np.column_stack([
    train_y, train_click, train_like
]).astype(np.float32)
train_aux_mask = np.ones(len(train_y), dtype=bool)

train_cat, train_num_raw = make_features(train)
valid_cat, valid_num_raw = make_features(valid)
mean, std = fit_scaler(train_num_raw)
train_num = scale_features(train_num_raw, mean, std)
valid_num = scale_features(valid_num_raw, mean, std)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64
)
inc_valid_z = zscore(inc_valid)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

candidate_scores = {}
candidate_specs = {}
raw_predictions = {}

inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_specs["incumbent"] = ("incumbent", 0.0, "raw")

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_primary = float(inc_metrics["primary"])

families = ["shared_bottom", "mmoe", "ple"]
seed_offsets = {
    "shared_bottom": 11,
    "mmoe": 37,
    "ple": 71,
}

for family in families:
    model = fit_model(
        family,
        train_cat,
        train_num,
        train_targets,
        train_aux_mask,
        seed_offsets[family],
    )
    pred = predict_model(model, valid_cat, valid_num)
    raw_predictions[family] = pred

    raw_metrics = evaluate(valid_users, valid_y, pred)
    candidate_scores[family] = float(raw_metrics["primary"])
    candidate_specs[family] = (family, 0.0, "raw")

    if float(raw_metrics["primary"]) > best_primary:
        best_primary = float(raw_metrics["primary"])
        best_name = family
        best_scores = pred.astype(np.float64)
        best_metrics = raw_metrics

    pred_z = zscore(pred)
    pred_rank = within_user_rank(valid_users, pred)

    for weight in (0.08, 0.14, 0.20, 0.28, 0.36, 0.45):
        z_blend = (1.0 - weight) * inc_valid_z + weight * pred_z
        z_name = f"{family}_zblend_{weight:.2f}"
        z_metrics = evaluate(valid_users, valid_y, z_blend)
        candidate_scores[z_name] = float(z_metrics["primary"])
        candidate_specs[z_name] = (family, float(weight), "z")

        if float(z_metrics["primary"]) > best_primary:
            best_primary = float(z_metrics["primary"])
            best_name = z_name
            best_scores = z_blend.copy()
            best_metrics = z_metrics

        rank_blend = (
            (1.0 - weight) * inc_valid_rank + weight * pred_rank
        )
        rank_name = f"{family}_rankblend_{weight:.2f}"
        rank_metrics = evaluate(valid_users, valid_y, rank_blend)
        candidate_scores[rank_name] = float(rank_metrics["primary"])
        candidate_specs[rank_name] = (family, float(weight), "rank")

        if float(rank_metrics["primary"]) > best_primary:
            best_primary = float(rank_metrics["primary"])
            best_name = rank_name
            best_scores = rank_blend.copy()
            best_metrics = rank_metrics

print("CANDIDATES " + json.dumps(
    {k: round(v, 6) for k, v in candidate_scores.items()},
    sort_keys=True
))
print("FINDINGS selected=" + best_name)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )

selected_family, selected_weight, selected_mode = candidate_specs[best_name]

if (
    out_dir
    and selected_family != "incumbent"
    and selected_mode != "raw"
):
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(raw_predictions[selected_family], dtype=np.float64)
    )

# Refit the selected recipe on train + validation primary labels.
# Auxiliary outcomes remain train-only: valid.aux is never read.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)

if selected_family == "incumbent":
    test_scores = np.asarray(
        np.load(os.path.join(shared, "incumbent_test_scores.npy")),
        dtype=np.float64
    )
else:
    valid_y_float = valid_y.astype(np.float32)
    combined_cat = np.concatenate([train_cat, valid_cat], axis=0)
    combined_num_raw = np.concatenate(
        [train_num_raw, valid_num_raw], axis=0
    )

    combined_targets = np.zeros(
        (len(train_y) + len(valid_y_float), 3), dtype=np.float32
    )
    combined_targets[:len(train_y)] = train_targets
    combined_targets[len(train_y):, 0] = valid_y_float

    combined_aux_mask = np.zeros(len(combined_targets), dtype=bool)
    combined_aux_mask[:len(train_y)] = True

    combined_mean, combined_std = fit_scaler(combined_num_raw)
    combined_num = scale_features(
        combined_num_raw, combined_mean, combined_std
    )

    test_cat, test_num_raw = make_features(test)
    test_num = scale_features(
        test_num_raw, combined_mean, combined_std
    )

    final_model = fit_model(
        selected_family,
        combined_cat,
        combined_num,
        combined_targets,
        combined_aux_mask,
        seed_offsets[selected_family],
    )
    test_raw = predict_model(final_model, test_cat, test_num)

    if selected_mode == "raw":
        test_scores = test_raw.astype(np.float64)
    else:
        inc_test = np.asarray(
            np.load(os.path.join(shared, "incumbent_test_scores.npy")),
            dtype=np.float64
        )
        if selected_mode == "z":
            test_scores = (
                (1.0 - selected_weight) * zscore(inc_test)
                + selected_weight * zscore(test_raw)
            )
        else:
            test_scores = (
                (1.0 - selected_weight)
                * within_user_rank(test_users, inc_test)
                + selected_weight
                * within_user_rank(test_users, test_raw)
            )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))