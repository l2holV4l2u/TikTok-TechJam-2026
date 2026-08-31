import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
BATCH = 8192
PRED_BATCH = 16384
THREADS = max(1, min(16, os.cpu_count() or 1))
torch.set_num_threads(THREADS)
torch.manual_seed(SEED)
np.random.seed(SEED)

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
HISTORY_LENGTH = 20


def make_matrix(split, mask=None):
    arrays = [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS]
    if mask is not None:
        arrays = [a[mask] for a in arrays]
    return np.ascontiguousarray(np.stack(arrays, axis=1))


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    su = user_ids[order]

    start_mask = np.empty(n, dtype=bool)
    start_mask[0] = True
    start_mask[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(start_mask)
    groups = np.cumsum(start_mask) - 1
    positions = np.arange(n) - starts[groups]
    sizes = np.diff(np.r_[starts, n])
    denominator = np.maximum(sizes[groups] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denominator
    ranked_sorted[sizes[groups] == 1] = 0.5
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def blend_search(user_ids, labels, own, incumbent):
    own_rank = within_user_rank(user_ids, own)
    incumbent_rank = within_user_rank(user_ids, incumbent)
    best = None
    for alpha in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.80, 0.90, 1.0]:
        score = alpha * own_rank + (1.0 - alpha) * incumbent_rank
        metrics = evaluate(user_ids, labels, score)
        record = (float(metrics["primary"]), float(alpha), score, metrics)
        if best is None or record[0] > best[0]:
            best = record
    return best


class FieldEmbedding(nn.Module):
    def __init__(self, cards, dim):
        super().__init__()
        offsets = np.cumsum([0] + cards[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        self.embedding = nn.Embedding(int(sum(cards)), dim)
        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, x):
        return self.embedding(x + self.offsets)


class DCN(nn.Module):
    def __init__(self, cards, dim=12):
        super().__init__()
        self.fields = FieldEmbedding(cards, dim)
        width = len(cards) * dim

        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(width)) for _ in range(3)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(width)) for _ in range(3)
        ])
        for w in self.cross_w:
            nn.init.normal_(w, std=0.02)

        self.deep = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(width + 64, 1)

    def forward(self, x):
        x0 = self.fields(x).flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = (cross * w).sum(dim=1, keepdim=True)
            cross = x0 * scalar + b + cross
        deep = self.deep(x0)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(1)


class MMoE(nn.Module):
    def __init__(self, cards, dim=10, tasks=3, experts=4):
        super().__init__()
        self.tasks = tasks
        self.fields = FieldEmbedding(cards, dim)
        width = len(cards) * dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, 96),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(96, 48),
                nn.ReLU(),
            )
            for _ in range(experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(width, experts) for _ in range(tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(tasks)
        ])

    def forward(self, x):
        flat = self.fields(x).flatten(1)
        expert_values = torch.stack(
            [expert(flat) for expert in self.experts], dim=1
        )
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(flat), dim=1).unsqueeze(2)
            representation = (expert_values * weights).sum(dim=1)
            outputs.append(tower(representation).squeeze(1))
        return outputs


class DIN(nn.Module):
    def __init__(self, cards, dim=14):
        super().__init__()
        self.dim = dim
        self.fields = FieldEmbedding(cards, dim)
        total = int(sum(cards))
        self.history_embedding = self.fields.embedding
        self.author_offset = int(sum(cards[:2]))
        self.tag_offset = int(sum(cards[:5]))
        width = len(cards) * dim + 2 * dim

        self.mlp = nn.Sequential(
            nn.Linear(width, 160),
            nn.PReLU(),
            nn.Dropout(0.08),
            nn.Linear(160, 64),
            nn.PReLU(),
            nn.Linear(64, 1),
        )

    def attend(self, candidate, history_ids, offset):
        mask = history_ids != 0
        global_ids = history_ids + offset
        history = self.history_embedding(global_ids)
        logits = (history * candidate.unsqueeze(1)).sum(dim=2)
        logits = logits / np.sqrt(float(self.dim))
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (history * weights.unsqueeze(2)).sum(dim=1)

    def forward(self, x, author_history, tag_history):
        embedded = self.fields(x)
        candidate_author = embedded[:, 2]
        candidate_tag = embedded[:, 5]
        author_interest = self.attend(
            candidate_author, author_history, self.author_offset
        )
        tag_interest = self.attend(
            candidate_tag, tag_history, self.tag_offset
        )
        joined = torch.cat(
            [embedded.flatten(1), author_interest, tag_interest], dim=1
        )
        return self.mlp(joined).squeeze(1)


def build_positive_histories(split, y, prefix_mask):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    authors = np.asarray(split.X["author_id"], dtype=np.int64)
    tags = np.asarray(split.X["tag"], dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    row = np.arange(len(users), dtype=np.int64)

    selected = np.flatnonzero(prefix_mask & (np.asarray(y) > 0))
    order_local = np.lexsort(
        (row[selected], times[selected], users[selected])
    )
    selected = selected[order_local]
    sorted_users = users[selected]

    author_history = np.zeros(
        (USER_CARD, HISTORY_LENGTH), dtype=np.int64
    )
    tag_history = np.zeros(
        (USER_CARD, HISTORY_LENGTH), dtype=np.int64
    )
    if len(selected) == 0:
        return author_history, tag_history

    end_mask = np.empty(len(selected), dtype=bool)
    end_mask[-1] = True
    end_mask[:-1] = sorted_users[:-1] != sorted_users[1:]
    ends = np.flatnonzero(end_mask) + 1

    start_mask = np.empty(len(selected), dtype=bool)
    start_mask[0] = True
    start_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    group = np.cumsum(start_mask) - 1
    reverse_position = ends[group] - 1 - np.arange(len(selected))
    keep = reverse_position < HISTORY_LENGTH

    chosen = selected[keep]
    slots = reverse_position[keep]
    chosen_users = users[chosen]
    author_history[chosen_users, slots] = authors[chosen]
    tag_history[chosen_users, slots] = tags[chosen]
    return author_history, tag_history


def train_standard(model, x, y, epochs, seed, aux=None):
    torch.manual_seed(seed)
    model.train()
    xt = torch.from_numpy(np.ascontiguousarray(x))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    aux_tensors = None
    if aux is not None:
        aux_tensors = [
            torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in aux
        ]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )
    generator = torch.Generator().manual_seed(seed)
    n = len(xt)

    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for lo in range(0, n, BATCH):
            idx = order[lo:lo + BATCH]
            optimizer.zero_grad(set_to_none=True)
            output = model(xt[idx])
            if aux_tensors is None:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    output, yt[idx]
                )
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    output[0], yt[idx]
                )
                loss = loss + 0.18 * nn.functional.binary_cross_entropy_with_logits(
                    output[1], aux_tensors[0][idx]
                )
                loss = loss + 0.12 * nn.functional.binary_cross_entropy_with_logits(
                    output[2], aux_tensors[1][idx]
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_standard(model, x, primary_head=False):
    model.eval()
    xt = torch.from_numpy(np.ascontiguousarray(x))
    result = np.empty(len(xt), dtype=np.float64)
    with torch.inference_mode():
        for lo in range(0, len(xt), PRED_BATCH):
            hi = min(len(xt), lo + PRED_BATCH)
            output = model(xt[lo:hi])
            if primary_head:
                output = output[0]
            result[lo:hi] = output.numpy().astype(np.float64)
    return result


def train_din(model, x, y, users, author_history, tag_history,
              epochs, seed):
    torch.manual_seed(seed)
    model.train()
    xt = torch.from_numpy(np.ascontiguousarray(x))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    users = np.asarray(users, dtype=np.int64)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )
    generator = torch.Generator().manual_seed(seed)
    n = len(xt)

    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for lo in range(0, n, BATCH):
            idx = order[lo:lo + BATCH]
            idx_np = idx.numpy()
            batch_users = users[idx_np]
            ah = torch.from_numpy(
                np.ascontiguousarray(author_history[batch_users])
            )
            th = torch.from_numpy(
                np.ascontiguousarray(tag_history[batch_users])
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[idx], ah, th)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_din(model, x, users, author_history, tag_history):
    model.eval()
    xt = torch.from_numpy(np.ascontiguousarray(x))
    users = np.asarray(users, dtype=np.int64)
    result = np.empty(len(xt), dtype=np.float64)
    with torch.inference_mode():
        for lo in range(0, len(xt), PRED_BATCH):
            hi = min(len(xt), lo + PRED_BATCH)
            bu = users[lo:hi]
            ah = torch.from_numpy(
                np.ascontiguousarray(author_history[bu])
            )
            th = torch.from_numpy(
                np.ascontiguousarray(tag_history[bu])
            )
            result[lo:hi] = model(
                xt[lo:hi], ah, th
            ).numpy().astype(np.float64)
    return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.float32)
x_train = make_matrix(train)
x_valid = make_matrix(valid)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

predictions = {}

# Family 1: explicit bounded feature crosses.
dcn = DCN(CARDS)
train_standard(dcn, x_train, y_train, epochs=3, seed=SEED)
predictions["dcn_cross"] = predict_standard(dcn, x_valid)
del dcn
gc.collect()

# Family 2: multi-task experts. Auxiliary outcomes are training targets only.
click_train = np.asarray(train.aux["is_click"], dtype=np.float32)
like_train = np.asarray(train.aux["is_like"], dtype=np.float32)
mmoe = MMoE(CARDS)
train_standard(
    mmoe, x_train, y_train, epochs=3, seed=SEED + 1,
    aux=[click_train, like_train]
)
predictions["mmoe_auxiliary"] = predict_standard(
    mmoe, x_valid, primary_head=True
)
del mmoe
gc.collect()

# Family 3: candidate-conditioned attention over positive prefix history.
unique_dates = np.unique(np.asarray(train.date))
cutoff = unique_dates[-5]
prefix_mask = np.asarray(train.date) < cutoff
target_mask = ~prefix_mask
author_hist, tag_hist = build_positive_histories(
    train, y_train, prefix_mask
)
x_din_train = make_matrix(train, target_mask)
din_users_train = np.asarray(train.X["user_id"], dtype=np.int64)[target_mask]
din = DIN(CARDS)
train_din(
    din,
    x_din_train,
    y_train[target_mask],
    din_users_train,
    author_hist,
    tag_hist,
    epochs=4,
    seed=SEED + 2,
)
valid_users = np.asarray(valid.X["user_id"], dtype=np.int64)
predictions["din_prefix_attention"] = predict_din(
    din, x_valid, valid_users, author_hist, tag_hist
)
del din
gc.collect()

candidate_summary = {}
raw_summary = {}
choices = {}

for name, pred in predictions.items():
    raw_metrics = evaluate(valid.user_id, valid.y, pred)
    raw_summary[name] = float(raw_metrics["primary"])
    blend = blend_search(
        valid.user_id, valid.y, pred, inc_valid
    )
    candidate_summary[name + "_blend"] = float(blend[0])
    choices[name] = blend

winner = max(choices, key=lambda name: choices[name][0])
best_primary, best_alpha, valid_scores, metrics = choices[winner]
raw_valid = predictions[winner]

print("FINDINGS raw_family_primary=" + json.dumps(
    raw_summary, sort_keys=True
))
print(
    "FINDINGS selected_family=%s own_rank_weight=%.2f history_target_rows=%d"
    % (winner, best_alpha, int(target_mask.sum()))
)
print("CANDIDATES " + json.dumps(candidate_summary, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(raw_valid, dtype=np.float64)
    )

# Permitted identical-recipe refit on train + validation.
test = load("test")
x_test = make_matrix(test)
test_users = np.asarray(test.X["user_id"], dtype=np.int64)

combined_y = np.concatenate([y_train, y_valid])
combined_x = np.concatenate([x_train, x_valid], axis=0)

if winner == "dcn_cross":
    final_model = DCN(CARDS)
    train_standard(
        final_model, combined_x, combined_y,
        epochs=3, seed=SEED
    )
    own_test = predict_standard(final_model, x_test)

elif winner == "mmoe_auxiliary":
    combined_click = np.concatenate([
        click_train,
        np.asarray(valid.aux["is_click"], dtype=np.float32)
    ])
    combined_like = np.concatenate([
        like_train,
        np.asarray(valid.aux["is_like"], dtype=np.float32)
    ])
    final_model = MMoE(CARDS)
    train_standard(
        final_model, combined_x, combined_y,
        epochs=3, seed=SEED + 1,
        aux=[combined_click, combined_like]
    )
    own_test = predict_standard(
        final_model, x_test, primary_head=True
    )

else:
    class Combined:
        pass

    combined = Combined()
    combined.X = {
        field: np.concatenate([
            np.asarray(train.X[field]),
            np.asarray(valid.X[field])
        ])
        for field in FIELDS
    }
    combined.date = np.concatenate([
        np.asarray(train.date), np.asarray(valid.date)
    ])
    combined.time_ms = np.concatenate([
        np.asarray(train.time_ms), np.asarray(valid.time_ms)
    ])

    combined_dates_unique = np.unique(combined.date)
    combined_cutoff = combined_dates_unique[-5]
    combined_prefix = np.asarray(combined.date) < combined_cutoff
    combined_target = ~combined_prefix

    final_author_hist, final_tag_hist = build_positive_histories(
        combined, combined_y, combined_prefix
    )
    final_x = make_matrix(combined, combined_target)
    final_users = np.asarray(
        combined.X["user_id"], dtype=np.int64
    )[combined_target]

    final_model = DIN(CARDS)
    train_din(
        final_model,
        final_x,
        combined_y[combined_target],
        final_users,
        final_author_hist,
        final_tag_hist,
        epochs=4,
        seed=SEED + 2,
    )
    own_test = predict_din(
        final_model, x_test, test_users,
        final_author_hist, final_tag_hist
    )

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
test_scores = (
    best_alpha * within_user_rank(test.user_id, own_test)
    + (1.0 - best_alpha) * within_user_rank(test.user_id, inc_test)
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))