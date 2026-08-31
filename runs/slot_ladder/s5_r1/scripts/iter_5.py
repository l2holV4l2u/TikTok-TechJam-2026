import os
import time
import json
import random
import numpy as np
import torch
from torch import nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 7341
BATCH = 8192
PRED_BATCH = 32768
HIST_LEN = 8
FIELDS = [
    "user_id", "video_id", "author_id", "tab",
    "duration_bucket", "tag", "upload_type", "onehot_feat1",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

OFFSETS = []
_total = 0
for f in FIELDS:
    OFFSETS.append(_total)
    _total += int(FEATURE_CARDINALITIES[f])
OFFSETS = np.asarray(OFFSETS, dtype=np.int64)
TOTAL_CARD = _total


def matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) + OFFSETS[j]
            for j, f in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def make_sequence_histories(train):
    """Prior positive videos for train rows and final train history per user."""
    n = len(train.user_id)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, np.asarray(train.time_ms), np.asarray(train.user_id)))
    su = np.asarray(train.user_id, dtype=np.int64)[order]
    sy = np.asarray(train.y, dtype=np.int8)[order]
    sv = np.asarray(train.video_id, dtype=np.int64)[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = su[1:] != su[:-1]
    group_starts = np.flatnonzero(new_group)
    group_ends = np.r_[group_starts[1:], n]

    global_cum = np.cumsum(sy, dtype=np.int64)
    before_group = np.zeros(len(group_starts), dtype=np.int64)
    mask = group_starts > 0
    before_group[mask] = global_cum[group_starts[mask] - 1]
    base_per_row = np.repeat(before_group, group_ends - group_starts)
    prior_count = global_cum - base_per_row - sy.astype(np.int64)

    positive_positions = np.flatnonzero(sy)
    hist_sorted = np.zeros((n, HIST_LEN), dtype=np.int64)
    for k in range(1, HIST_LEN + 1):
        ok = prior_count >= k
        pos_ordinal = base_per_row[ok] + prior_count[ok] - k
        hist_sorted[ok, k - 1] = sv[positive_positions[pos_ordinal]]

    hist_train = np.zeros_like(hist_sorted)
    hist_train[order] = hist_sorted

    user_card = int(FEATURE_CARDINALITIES["user_id"])
    final_hist = np.zeros((user_card, HIST_LEN), dtype=np.int64)
    total_pos = global_cum[group_ends - 1] - before_group

    for k in range(1, HIST_LEN + 1):
        ok = total_pos >= k
        ordinals = before_group[ok] + total_pos[ok] - k
        final_hist[su[group_starts[ok]], k - 1] = sv[positive_positions[ordinals]]

    return np.ascontiguousarray(hist_train), final_hist


def histories_for_split(split, final_hist):
    uid = np.asarray(split.user_id, dtype=np.int64)
    result = np.zeros((len(uid), HIST_LEN), dtype=np.int64)
    ok = (uid >= 0) & (uid < len(final_hist))
    result[ok] = final_hist[uid[ok]]
    return np.ascontiguousarray(result)


class WideModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Embedding(TOTAL_CARD, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.weight.weight)

    def forward(self, x):
        return self.bias + self.weight(x).sum(1).squeeze(-1)


class DINModel(nn.Module):
    def __init__(self, emb_dim=16):
        super().__init__()
        self.cat_emb = nn.Embedding(TOTAL_CARD, emb_dim)
        self.video_emb = nn.Embedding(
            int(FEATURE_CARDINALITIES["video_id"]), emb_dim, padding_idx=0
        )
        nn.init.normal_(self.cat_emb.weight, std=0.025)
        nn.init.normal_(self.video_emb.weight, std=0.025)
        with torch.no_grad():
            self.video_emb.weight[0].zero_()

        # Flattened categorical embeddings plus target-conditioned history summary.
        in_dim = len(FIELDS) * emb_dim + 4 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x, history):
        cats = self.cat_emb(x)
        target = self.video_emb(x[:, 1] - int(OFFSETS[1]))
        h = self.video_emb(history)
        mask = history.ne(0)

        scale = target.shape[-1] ** -0.5
        att = (h * target[:, None, :]).sum(-1) * scale
        weights = torch.exp(torch.clamp(att, -12.0, 12.0)) * mask.float()
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        summary = (weights[:, :, None] * h).sum(1)

        interaction = torch.cat([
            summary,
            target,
            summary * target,
            torch.abs(summary - target),
        ], dim=1)
        z = torch.cat([cats.flatten(1), interaction], dim=1)
        return self.net(z).squeeze(-1)


class MMoEModel(nn.Module):
    def __init__(self, emb_dim=12, num_experts=4, num_tasks=3):
        super().__init__()
        self.emb = nn.Embedding(TOTAL_CARD, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.025)
        input_dim = len(FIELDS) * emb_dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
            )
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts) for _ in range(num_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        z = self.emb(x).flatten(1)
        experts = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            w = torch.softmax(gate(z), dim=1)
            task_rep = (experts * w[:, :, None]).sum(1)
            outputs.append(tower(task_rep).squeeze(-1))
        return torch.stack(outputs, dim=1)


def train_wide(x, y, epochs=3):
    model = WideModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-7)
    loss_fn = nn.BCEWithLogitsLoss()
    tx = torch.from_numpy(x)
    ty = torch.from_numpy(np.asarray(y, dtype=np.float32))
    gen = torch.Generator().manual_seed(SEED + 1)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(tx), generator=gen)
        for start in range(0, len(tx), BATCH):
            idx = order[start:start + BATCH]
            loss = loss_fn(model(tx[idx]), ty[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def train_din(x, histories, y, epochs=3):
    model = DINModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    tx = torch.from_numpy(x)
    th = torch.from_numpy(histories)
    ty = torch.from_numpy(np.asarray(y, dtype=np.float32))
    gen = torch.Generator().manual_seed(SEED + 2)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(tx), generator=gen)
        for start in range(0, len(tx), BATCH):
            idx = order[start:start + BATCH]
            loss = loss_fn(model(tx[idx], th[idx]), ty[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def train_mmoe(x, targets, epochs=3):
    model = MMoEModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-6)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    tx = torch.from_numpy(x)
    tt = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    task_weights = torch.tensor([1.0, 0.22, 0.18], dtype=torch.float32)
    gen = torch.Generator().manual_seed(SEED + 3)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(tx), generator=gen)
        for start in range(0, len(tx), BATCH):
            idx = order[start:start + BATCH]
            losses = loss_fn(model(tx[idx]), tt[idx]).mean(0)
            loss = (losses * task_weights).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def predict_wide(model, x):
    model.eval()
    tx = torch.from_numpy(x)
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), PRED_BATCH):
        end = min(start + PRED_BATCH, len(x))
        out[start:end] = model(tx[start:end]).numpy().astype(np.float64)
    return out


@torch.inference_mode()
def predict_din(model, x, histories):
    model.eval()
    tx = torch.from_numpy(x)
    th = torch.from_numpy(histories)
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), PRED_BATCH):
        end = min(start + PRED_BATCH, len(x))
        out[start:end] = model(
            tx[start:end], th[start:end]
        ).numpy().astype(np.float64)
    return out


@torch.inference_mode()
def predict_mmoe(model, x):
    model.eval()
    tx = torch.from_numpy(x)
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), PRED_BATCH):
        end = min(start + PRED_BATCH, len(x))
        out[start:end] = model(tx[start:end])[:, 0].numpy().astype(np.float64)
    return out


def metric_primary(valid, scores):
    return float(evaluate(valid.user_id, valid.y, scores)["primary"])


train = load("train")
valid = load("valid")
x_train = matrix(train)
x_valid = matrix(valid)

train_hist, final_hist = make_sequence_histories(train)
valid_hist = histories_for_split(valid, final_hist)

click_target = np.asarray(train.aux["is_click"], dtype=np.float32)
like_target = np.asarray(train.aux["is_like"], dtype=np.float32)
multi_targets = np.column_stack([
    np.asarray(train.y, dtype=np.float32),
    click_target,
    like_target,
]).astype(np.float32)

wide = train_wide(x_train, train.y)
din = train_din(x_train, train_hist, train.y)
mmoe = train_mmoe(x_train, multi_targets)

valid_predictions = {
    "wide": predict_wide(wide, x_valid),
    "din": predict_din(din, x_valid, valid_hist),
    "mmoe": predict_mmoe(mmoe, x_valid),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)

candidate_scores = {}
candidate_arrays = {}
candidate_specs = {}

for name, pred in valid_predictions.items():
    p = metric_primary(valid, pred)
    candidate_scores[name] = p
    candidate_arrays[name] = pred
    candidate_specs[name] = (name, None)

    # The API explicitly permits choosing an incumbent blend weight on validation.
    for w in (0.15, 0.25, 0.35, 0.50, 0.65):
        blended = (1.0 - w) * inc_valid + w * pred
        key = name + "_blend_" + str(w)
        bp = metric_primary(valid, blended)
        candidate_scores[key] = bp
        candidate_arrays[key] = blended
        candidate_specs[key] = (name, w)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_arrays[winner]
winner_family, winner_weight = candidate_specs[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(
    {k: float(v) for k, v in candidate_scores.items()},
    sort_keys=True,
    separators=(",", ":"),
))
print("FINDINGS " + json.dumps({
    "winner": winner,
    "train_prior_history_nonempty": float(np.mean(train_hist[:, 0] != 0)),
    "valid_train_history_nonempty": float(np.mean(valid_hist[:, 0] != 0)),
    "click_rate": float(click_target.mean()),
    "like_rate": float(like_target.mean()),
}, separators=(",", ":")))

test = load("test")
x_test = matrix(test)
test_hist = histories_for_split(test, final_hist)

if winner_family == "wide":
    own_test_scores = predict_wide(wide, x_test)
elif winner_family == "din":
    own_test_scores = predict_din(din, x_test, test_hist)
else:
    own_test_scores = predict_mmoe(mmoe, x_test)

if winner_weight is None:
    test_scores = own_test_scores
    own_valid_scores = valid_predictions[winner_family]
else:
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_scores = (
        (1.0 - winner_weight) * inc_test
        + winner_weight * own_test_scores
    )
    own_valid_scores = valid_predictions[winner_family]

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if winner_weight is not None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(own_valid_scores, dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}, separators=(",", ":")))