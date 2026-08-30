import gc
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "hour",
    "upload_type",
]
BLEND_WEIGHTS = [0.10, 0.20, 0.30, 0.40, 0.50]
EPOCHS = 2
BATCH_SIZE = 8192
EMBED_DIM = 12
AUX_WEIGHT = 0.20


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(starts_mask, np.arange(n, dtype=np.int64), 0)
    )
    positions = np.arange(n, dtype=np.float64) - starts

    start_idx = np.flatnonzero(starts_mask)
    end_idx = np.r_[start_idx[1:], n]
    sizes = end_idx - start_idx
    repeated_sizes = np.repeat(sizes, sizes).astype(np.float64)

    ranked = positions / np.maximum(repeated_sizes - 1.0, 1.0)
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def make_arrays(split, duration_stats=None):
    xcat = np.column_stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS]
    )

    duration = np.asarray(split.num["duration_ms"], dtype=np.float32)
    duration = np.log1p(np.maximum(np.nan_to_num(duration, nan=0.0), 0.0))

    if duration_stats is None:
        mean = float(duration.mean())
        std = float(duration.std())
        std = max(std, 1e-4)
        duration_stats = (mean, std)
    mean, std = duration_stats
    xnum = ((duration - mean) / std).astype(np.float32)[:, None]
    return xcat, xnum, duration_stats


def select_aux_names(split):
    preferred = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_profile_enter",
    ]
    selected = [name for name in preferred if name in split.aux][:2]
    if len(selected) < 2:
        for name in sorted(split.aux.keys()):
            if name not in selected:
                arr = np.asarray(split.aux[name])
                if arr.ndim == 1 and len(arr) == len(split.y):
                    selected.append(name)
                if len(selected) == 2:
                    break
    if len(selected) < 2:
        raise RuntimeError("At least two auxiliary training outcomes are required")
    return selected


def make_targets(split, aux_names):
    targets = [np.asarray(split.y, dtype=np.float32)]
    for name in aux_names:
        a = np.asarray(split.aux[name])
        if np.issubdtype(a.dtype, np.floating):
            a = np.nan_to_num(a, nan=0.0)
        targets.append((a > 0).astype(np.float32))
    return np.column_stack(targets).astype(np.float32)


def recency_weights(dates, half_life):
    n = len(dates)
    if half_life is None:
        return np.ones(n, dtype=np.float32)
    dates = np.asarray(dates, dtype=np.float64)
    age = float(np.max(dates)) - dates
    weights = np.exp2(-age / float(half_life))
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(int(FEATURE_CARDINALITIES[f]), EMBED_DIM)
                for f in FIELDS
            ]
        )
        self.output_dim = len(FIELDS) * EMBED_DIM + 1

        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.03)
            with torch.no_grad():
                emb.weight[0].zero_()

    def forward(self, xcat, xnum):
        parts = [emb(xcat[:, j]) for j, emb in enumerate(self.embeddings)]
        return torch.cat(parts + [xnum], dim=1)


class SharedBottom(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.encoder = FeatureEncoder()
        d = self.encoder.output_dim
        self.bottom = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(n_tasks)])

    def forward(self, xcat, xnum):
        h = self.bottom(self.encoder(xcat, xnum))
        return torch.cat([head(h) for head in self.heads], dim=1)


class MMoE(nn.Module):
    def __init__(self, n_tasks, n_experts=4):
        super().__init__()
        self.encoder = FeatureEncoder()
        d = self.encoder.output_dim
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, 96),
                    nn.ReLU(),
                    nn.Linear(96, 64),
                    nn.ReLU(),
                )
                for _ in range(n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(d, n_experts) for _ in range(n_tasks)]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )
                for _ in range(n_tasks)
            ]
        )

    def forward(self, xcat, xnum):
        z = self.encoder(xcat, xnum)
        experts = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(z), dim=1).unsqueeze(2)
            mixture = torch.sum(experts * weights, dim=1)
            outputs.append(tower(mixture))
        return torch.cat(outputs, dim=1)


class PLE(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.encoder = FeatureEncoder()
        d = self.encoder.output_dim
        self.n_tasks = n_tasks
        self.shared_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, 96),
                    nn.ReLU(),
                    nn.Linear(96, 64),
                    nn.ReLU(),
                )
                for _ in range(2)
            ]
        )
        self.task_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, 96),
                    nn.ReLU(),
                    nn.Linear(96, 64),
                    nn.ReLU(),
                )
                for _ in range(n_tasks)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(d, 3) for _ in range(n_tasks)]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )
                for _ in range(n_tasks)
            ]
        )

    def forward(self, xcat, xnum):
        z = self.encoder(xcat, xnum)
        shared = [expert(z) for expert in self.shared_experts]
        outputs = []
        for task in range(self.n_tasks):
            candidates = torch.stack(
                [shared[0], shared[1], self.task_experts[task](z)], dim=1
            )
            gate = torch.softmax(self.gates[task](z), dim=1).unsqueeze(2)
            representation = torch.sum(candidates * gate, dim=1)
            outputs.append(self.towers[task](representation))
        return torch.cat(outputs, dim=1)


def build_model(family, n_tasks):
    if family == "shared_bottom":
        return SharedBottom(n_tasks)
    if family == "mmoe":
        return MMoE(n_tasks)
    if family == "ple":
        return PLE(n_tasks)
    raise ValueError("Unknown family: " + family)


def train_model(
    family,
    xcat,
    xnum,
    targets,
    sample_weights,
    seed,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(family, targets.shape[1])
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-6
    )
    n = len(targets)
    task_weights = torch.tensor(
        [1.0] + [AUX_WEIGHT] * (targets.shape[1] - 1),
        dtype=torch.float32,
    )

    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            bxcat = torch.from_numpy(xcat[idx])
            bxnum = torch.from_numpy(xnum[idx])
            by = torch.from_numpy(targets[idx])
            bw = torch.from_numpy(sample_weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(bxcat, bxnum)
            losses = F.binary_cross_entropy_with_logits(
                logits, by, reduction="none"
            )
            per_row = torch.sum(losses * task_weights[None, :], dim=1)
            loss = torch.sum(per_row * bw) / torch.sum(bw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict(model, xcat, xnum):
    model.eval()
    result = np.empty(len(xcat), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(xcat), BATCH_SIZE * 2):
            stop = min(start + BATCH_SIZE * 2, len(xcat))
            logits = model(
                torch.from_numpy(xcat[start:stop]),
                torch.from_numpy(xnum[start:stop]),
            )
            result[start:stop] = logits[:, 0].cpu().numpy()
    return result


train = load("train")
valid = load("valid")

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
inc_rank = within_user_rank(valid.user_id, inc_valid)

aux_names = select_aux_names(train)
train_xcat, train_xnum, duration_stats = make_arrays(train)
valid_xcat, valid_xnum, _ = make_arrays(valid, duration_stats)
train_targets = make_targets(train, aux_names)

recipes = [
    ("shared_uniform", "shared_bottom", None, 1101),
    ("shared_recency4", "shared_bottom", 4.0, 1102),
    ("mmoe_recency4", "mmoe", 4.0, 1201),
    ("ple_recency4", "ple", 4.0, 1301),
]

candidate_scores = {"incumbent": float(inc_metrics["primary"])}
candidate_predictions = {"incumbent": inc_valid}
candidate_recipes = {
    "incumbent": {
        "family": None,
        "half_life": None,
        "seed": None,
        "blend": None,
    }
}
rank_correlations = {}

for recipe_name, family, half_life, seed in recipes:
    weights = recency_weights(train.date, half_life)
    model = train_model(
        family,
        train_xcat,
        train_xnum,
        train_targets,
        weights,
        seed,
    )
    raw = predict(model, valid_xcat, valid_xnum)
    raw_rank = within_user_rank(valid.user_id, raw)

    raw_metrics = evaluate(valid.user_id, valid.y, raw)
    candidate_scores[recipe_name] = float(raw_metrics["primary"])
    candidate_predictions[recipe_name] = raw
    candidate_recipes[recipe_name] = {
        "family": family,
        "half_life": half_life,
        "seed": seed,
        "blend": None,
    }

    rank_correlations[recipe_name] = float(
        np.corrcoef(raw_rank, inc_rank)[0, 1]
    )

    for alpha in BLEND_WEIGHTS:
        name = "%s_rankblend%.2f" % (recipe_name, alpha)
        blended = alpha * raw_rank + (1.0 - alpha) * inc_rank
        met = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[name] = float(met["primary"])
        candidate_predictions[name] = blended
        candidate_recipes[name] = {
            "family": family,
            "half_life": half_life,
            "seed": seed,
            "blend": alpha,
        }

    del model, raw, raw_rank
    gc.collect()

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(candidate_predictions[winner], dtype=np.float64)
winner_recipe = candidate_recipes[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS auxiliary_training_targets=%s"
    % ",".join(aux_names)
)
print(
    "FINDINGS rank_correlations_with_incumbent="
    + json.dumps(rank_correlations, sort_keys=True)
)
print(
    "FINDINGS winner=%s family=%s half_life=%s blend=%s delta_incumbent=%+.6f"
    % (
        winner,
        str(winner_recipe["family"]),
        str(winner_recipe["half_life"]),
        str(winner_recipe["blend"]),
        float(metrics["primary"] - inc_metrics["primary"]),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winner == "incumbent":
    test_scores = inc_test
else:
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {
        f: np.concatenate(
            [
                np.asarray(train.X[f], dtype=np.int64),
                np.asarray(valid.X[f], dtype=np.int64),
            ]
        )
        for f in FIELDS
    }
    combined.num = {
        "duration_ms": np.concatenate(
            [
                np.asarray(train.num["duration_ms"], dtype=np.float32),
                np.asarray(valid.num["duration_ms"], dtype=np.float32),
            ]
        )
    }
    combined.y = np.concatenate(
        [
            np.asarray(train.y, dtype=np.int8),
            np.asarray(valid.y, dtype=np.int8),
        ]
    )
    combined.date = np.concatenate(
        [
            np.asarray(train.date),
            np.asarray(valid.date),
        ]
    )
    combined.aux = {
        name: np.concatenate(
            [
                np.asarray(train.aux[name]),
                np.asarray(valid.aux[name]),
            ]
        )
        for name in aux_names
    }

    combined_xcat, combined_xnum, final_duration_stats = make_arrays(combined)
    test_xcat, test_xnum, _ = make_arrays(test, final_duration_stats)
    combined_targets = make_targets(combined, aux_names)
    final_weights = recency_weights(
        combined.date, winner_recipe["half_life"]
    )

    final_model = train_model(
        winner_recipe["family"],
        combined_xcat,
        combined_xnum,
        combined_targets,
        final_weights,
        int(winner_recipe["seed"]),
    )
    raw_test = predict(final_model, test_xcat, test_xnum)

    if winner_recipe["blend"] is None:
        test_scores = raw_test
    else:
        alpha = float(winner_recipe["blend"])
        raw_test_rank = within_user_rank(test.user_id, raw_test)
        inc_test_rank = within_user_rank(test.user_id, inc_test)
        test_scores = alpha * raw_test_rank + (1.0 - alpha) * inc_test_rank

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)