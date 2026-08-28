import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FM_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
FIBI_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]
TREE_CAT_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "music_type",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
]
NUM_FIELDS = [
    "collect_cnt",
    "comment_cnt",
    "complete_play_cnt",
    "counts",
    "download_cnt",
    "duration_ms",
    "follow_cnt",
    "like_cnt",
    "long_time_play_cnt",
    "play_cnt",
    "play_duration",
    "play_progress",
    "play_user_num",
    "share_cnt",
    "short_time_play_cnt",
    "show_cnt",
    "show_user_num",
    "valid_play_cnt",
]

FM_RANK = 16
FM_LR = 0.001
FM_BATCH = 4096
FM_EPOCHS = 6

FIBI_DIM = 8
FIBI_LR = 0.001
FIBI_BATCH = 8192
FIBI_EPOCHS = 5
CLICK_WEIGHT = 0.20
N_EXPERTS = 3

N_THREADS = max(1, min(16, os.cpu_count() or 1))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(N_THREADS)


def make_offset_matrix(split, fields):
    cards = [int(FEATURE_CARDINALITIES[name]) for name in fields]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    x = np.stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in fields],
        axis=1,
    )
    return np.ascontiguousarray(x + offsets[None, :]), int(sum(cards))


class FactorizationMachine(nn.Module):
    def __init__(self, n_tokens, rank):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)

    def forward(self, x):
        z = self.embedding(x)
        linear = z[:, :, 0].sum(dim=1)
        factors = z[:, :, 1:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class FiBiMMoE(nn.Module):
    def __init__(self, n_tokens, n_fields, dim, n_experts):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, dim)
        self.long_wide = nn.Embedding(n_tokens, 1)
        self.click_wide = nn.Embedding(n_tokens, 1)
        self.long_bias = nn.Parameter(torch.zeros(()))
        self.click_bias = nn.Parameter(torch.zeros(()))

        hidden = max(3, n_fields // 3)
        self.senet = nn.Sequential(
            nn.Linear(n_fields, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_fields),
            nn.Sigmoid(),
        )
        self.bilinear = nn.Linear(dim, dim, bias=False)

        left = []
        right = []
        for i in range(n_fields):
            for j in range(i + 1, n_fields):
                left.append(i)
                right.append(j)
        self.register_buffer("pair_left", torch.tensor(left, dtype=torch.long))
        self.register_buffer("pair_right", torch.tensor(right, dtype=torch.long))

        n_pairs = len(left)
        input_dim = (n_fields + n_pairs) * dim

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 72),
                    nn.ReLU(),
                    nn.Dropout(0.08),
                    nn.Linear(72, 32),
                    nn.ReLU(),
                )
                for _ in range(n_experts)
            ]
        )
        self.long_gate = nn.Linear(input_dim, n_experts)
        self.click_gate = nn.Linear(input_dim, n_experts)
        self.long_tower = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(24, 1),
        )
        self.click_tower = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(24, 1),
        )

        nn.init.normal_(self.embedding.weight, 0.0, 0.03)
        nn.init.zeros_(self.long_wide.weight)
        nn.init.zeros_(self.click_wide.weight)
        nn.init.xavier_uniform_(self.bilinear.weight)
        nn.init.zeros_(self.long_gate.bias)
        nn.init.zeros_(self.click_gate.bias)

    def forward(self, x, return_click=False):
        e = self.embedding(x)
        scale = 2.0 * self.senet(e.mean(dim=2))
        se = e * scale.unsqueeze(2)

        transformed = self.bilinear(se)
        pair_features = (
            transformed.index_select(1, self.pair_left)
            * se.index_select(1, self.pair_right)
        )
        features = torch.cat(
            [se.flatten(start_dim=1), pair_features.flatten(start_dim=1)],
            dim=1,
        )

        experts = torch.stack(
            [expert(features) for expert in self.experts],
            dim=1,
        )
        long_gate = torch.softmax(self.long_gate(features), dim=1)
        long_shared = (experts * long_gate.unsqueeze(2)).sum(dim=1)
        long_logit = (
            self.long_bias
            + self.long_wide(x).squeeze(2).sum(dim=1)
            + self.long_tower(long_shared).squeeze(1)
        )

        if not return_click:
            return long_logit

        click_gate = torch.softmax(self.click_gate(features), dim=1)
        click_shared = (experts * click_gate.unsqueeze(2)).sum(dim=1)
        click_logit = (
            self.click_bias
            + self.click_wide(x).squeeze(2).sum(dim=1)
            + self.click_tower(click_shared).squeeze(1)
        )
        return long_logit, click_logit


def predict_torch(model, x, batch_size=32768):
    model.eval()
    out = np.empty(len(x), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            out[start:end] = (
                model(torch.from_numpy(x[start:end]))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    return out


def train_fm(model, x, y):
    sparse_opt = torch.optim.SparseAdam(
        [model.embedding.weight], lr=FM_LR
    )
    bias_opt = torch.optim.Adam([model.bias], lr=FM_LR)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)

    for epoch in range(FM_EPOCHS):
        model.train()
        permutation = rng.permutation(len(y))
        total = 0.0
        for start in range(0, len(y), FM_BATCH):
            idx = permutation[start:start + FM_BATCH]
            xb = torch.from_numpy(x[idx])
            yb = torch.from_numpy(y[idx])

            sparse_opt.zero_grad(set_to_none=True)
            bias_opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            sparse_opt.step()
            bias_opt.step()
            total += float(loss.detach()) * len(idx)

        print(
            "train model=FM epoch=%d loss=%.6f"
            % (epoch + 1, total / len(y)),
            flush=True,
        )


def train_fibi(model, x, y_long, y_click):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FIBI_LR, weight_decay=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED + 17)

    for epoch in range(FIBI_EPOCHS):
        model.train()
        permutation = rng.permutation(len(y_long))
        total = 0.0
        total_long = 0.0
        total_click = 0.0

        for start in range(0, len(y_long), FIBI_BATCH):
            idx = permutation[start:start + FIBI_BATCH]
            xb = torch.from_numpy(x[idx])
            ylb = torch.from_numpy(y_long[idx])
            ycb = torch.from_numpy(y_click[idx])

            optimizer.zero_grad(set_to_none=True)
            long_logits, click_logits = model(xb, return_click=True)
            long_loss = criterion(long_logits, ylb)
            click_loss = criterion(click_logits, ycb)
            loss = long_loss + CLICK_WEIGHT * click_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total += float(loss.detach()) * len(idx)
            total_long += float(long_loss.detach()) * len(idx)
            total_click += float(click_loss.detach()) * len(idx)

        print(
            "train model=FiBiMMoE epoch=%d loss=%.6f long=%.6f click=%.6f"
            % (
                epoch + 1,
                total / len(y_long),
                total_long / len(y_long),
                total_click / len(y_long),
            ),
            flush=True,
        )


def signed_log1p(x):
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


def safe_ratio(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    result = a / (np.abs(b) + 1.0)
    result[~np.isfinite(a) | ~np.isfinite(b)] = np.nan
    return result.astype(np.float32)


def make_tree_matrix(split):
    columns = [
        np.asarray(split.X[name], dtype=np.float32)
        for name in TREE_CAT_FIELDS
    ]
    columns.extend(
        signed_log1p(split.num[name]) for name in NUM_FIELDS
    )

    n = split.num
    columns.extend(
        [
            safe_ratio(n["long_time_play_cnt"], n["show_cnt"]),
            safe_ratio(n["complete_play_cnt"], n["show_cnt"]),
            safe_ratio(n["valid_play_cnt"], n["show_cnt"]),
            safe_ratio(n["play_cnt"], n["show_cnt"]),
            safe_ratio(n["play_user_num"], n["show_user_num"]),
            safe_ratio(n["like_cnt"], n["show_cnt"]),
            safe_ratio(n["comment_cnt"], n["show_cnt"]),
            safe_ratio(n["follow_cnt"], n["show_cnt"]),
            safe_ratio(n["share_cnt"], n["show_cnt"]),
            safe_ratio(n["collect_cnt"], n["show_cnt"]),
            safe_ratio(n["play_duration"], n["play_cnt"]),
            safe_ratio(
                n["play_duration"],
                np.asarray(n["play_cnt"]) * np.asarray(n["duration_ms"]),
            ),
            safe_ratio(n["long_time_play_cnt"], n["play_cnt"]),
            safe_ratio(n["complete_play_cnt"], n["play_cnt"]),
            safe_ratio(n["valid_play_cnt"], n["play_cnt"]),
        ]
    )
    return np.ascontiguousarray(
        np.column_stack(columns), dtype=np.float32
    )


def grouped_positions(new_group):
    n = len(new_group)
    starts = np.flatnonzero(new_group)
    lengths = np.diff(np.r_[starts, n])
    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    pos = np.arange(n, dtype=np.int64) - repeated_starts
    return pos, repeated_lengths


def make_context_matrix(split):
    n = len(split.y)
    row = np.arange(n, dtype=np.int64)
    user = np.asarray(split.user_id, dtype=np.int64)
    time_ms = np.asarray(split.time_ms, dtype=np.int64)
    date = np.asarray(split.date, dtype=np.int64)

    order = np.lexsort((row, time_ms, user))
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)

    us = user[order]
    ts = time_ms[order]
    ds = date[order]

    user_new = np.r_[True, us[1:] != us[:-1]]
    user_pos, user_len = grouped_positions(user_new)

    day_new = np.r_[
        True,
        (us[1:] != us[:-1]) | (ds[1:] != ds[:-1]),
    ]
    day_pos, day_len = grouped_positions(day_new)

    raw_gap = np.r_[0, ts[1:] - ts[:-1]]
    same_user_prev = np.r_[False, us[1:] == us[:-1]]
    raw_gap[~same_user_prev] = 0

    session_new = day_new | (raw_gap > 30 * 60 * 1000)
    session_pos, session_len = grouped_positions(session_new)

    batch_new = np.r_[
        True,
        (us[1:] != us[:-1]) | (ts[1:] != ts[:-1]),
    ]
    batch_pos, batch_len = grouped_positions(batch_new)

    next_gap = np.r_[ts[1:] - ts[:-1], 0]
    same_user_next = np.r_[us[:-1] == us[1:], False]
    next_gap[~same_user_next] = 0

    prev_gap_log = np.log1p(
        np.maximum(raw_gap, 0).astype(np.float64) / 1000.0
    )
    next_gap_log = np.log1p(
        np.maximum(next_gap, 0).astype(np.float64) / 1000.0
    )

    video = np.asarray(split.video_id, dtype=np.int64)[order]
    author = np.asarray(split.X["author_id"], dtype=np.int64)[order]
    tag = np.asarray(split.X["tag"], dtype=np.int64)[order]
    tab = np.asarray(split.X["tab"], dtype=np.int64)[order]
    duration = np.asarray(
        split.X["duration_bucket"], dtype=np.int64
    )[order]

    valid_prev = ~user_new
    valid_next = np.r_[us[:-1] == us[1:], False]

    same_video_prev = np.zeros(n, dtype=np.float32)
    same_author_prev = np.zeros(n, dtype=np.float32)
    same_tag_prev = np.zeros(n, dtype=np.float32)
    same_video_next = np.zeros(n, dtype=np.float32)
    same_author_next = np.zeros(n, dtype=np.float32)
    same_tag_next = np.zeros(n, dtype=np.float32)

    same_video_prev[1:] = (
        (us[1:] == us[:-1]) & (video[1:] == video[:-1])
    )
    same_author_prev[1:] = (
        (us[1:] == us[:-1]) & (author[1:] == author[:-1])
    )
    same_tag_prev[1:] = (
        (us[1:] == us[:-1]) & (tag[1:] == tag[:-1])
    )
    same_video_next[:-1] = (
        (us[:-1] == us[1:]) & (video[:-1] == video[1:])
    )
    same_author_next[:-1] = (
        (us[:-1] == us[1:]) & (author[:-1] == author[1:])
    )
    same_tag_next[:-1] = (
        (us[:-1] == us[1:]) & (tag[:-1] == tag[1:])
    )

    user_fraction = user_pos / np.maximum(user_len - 1, 1)
    day_fraction = day_pos / np.maximum(day_len - 1, 1)
    session_fraction = session_pos / np.maximum(session_len - 1, 1)
    batch_fraction = batch_pos / np.maximum(batch_len - 1, 1)

    hour = np.asarray(split.X["hour"], dtype=np.float32)[order]
    hour_angle = 2.0 * np.pi * hour / 24.0

    sorted_columns = [
        np.minimum(user_pos, 100),
        np.log1p(user_len),
        user_fraction,
        np.minimum(day_pos, 50),
        np.log1p(day_len),
        day_fraction,
        np.minimum(session_pos, 50),
        np.log1p(session_len),
        session_fraction,
        np.minimum(batch_pos, 20),
        np.log1p(batch_len),
        batch_fraction,
        prev_gap_log,
        next_gap_log,
        (raw_gap > 5 * 60 * 1000).astype(np.float32),
        (raw_gap > 30 * 60 * 1000).astype(np.float32),
        same_video_prev,
        same_author_prev,
        same_tag_prev,
        same_video_next,
        same_author_next,
        same_tag_next,
        valid_prev.astype(np.float32),
        valid_next.astype(np.float32),
        np.sin(hour_angle),
        np.cos(hour_angle),
        tab,
        duration,
    ]

    sorted_matrix = np.column_stack(sorted_columns).astype(np.float32)
    return np.ascontiguousarray(sorted_matrix[inverse])


def metric_primary(split, scores):
    return float(
        evaluate(split.user_id, split.y, scores)["primary"]
    )


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
click_train = np.asarray(train.aux["is_click"], dtype=np.float32)

# Established FM component.
xtr_fm, fm_tokens = make_offset_matrix(train, FM_FIELDS)
xva_fm, _ = make_offset_matrix(valid, FM_FIELDS)

fm = FactorizationMachine(fm_tokens, FM_RANK)
train_fm(fm, xtr_fm, y_train)
fm_valid = predict_torch(fm, xva_fm)
del xtr_fm, xva_fm

# Established FiBiNET/MMoE component.
xtr_fibi, fibi_tokens = make_offset_matrix(train, FIBI_FIELDS)
xva_fibi, _ = make_offset_matrix(valid, FIBI_FIELDS)

torch.manual_seed(SEED + 17)
fibi = FiBiMMoE(
    fibi_tokens, len(FIBI_FIELDS), FIBI_DIM, N_EXPERTS
)
train_fibi(fibi, xtr_fibi, y_train, click_train)
fibi_valid = predict_torch(fibi, xva_fibi)
del xtr_fibi, xva_fibi

# Established safe numeric/categorical tree component.
xtr_tree = make_tree_matrix(train)
xva_tree = make_tree_matrix(valid)
tree_cat_indices = list(range(len(TREE_CAT_FIELDS)))

tree_train = lgb.Dataset(
    xtr_tree,
    label=y_train,
    categorical_feature=tree_cat_indices,
    free_raw_data=False,
)
tree_valid_set = lgb.Dataset(
    xva_tree,
    label=np.asarray(valid.y, dtype=np.float32),
    categorical_feature=tree_cat_indices,
    reference=tree_train,
    free_raw_data=False,
)
tree_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "cat_smooth": 30.0,
    "cat_l2": 15.0,
    "max_cat_to_onehot": 16,
    "seed": SEED + 91,
    "feature_fraction_seed": SEED + 92,
    "bagging_seed": SEED + 93,
    "num_threads": N_THREADS,
    "verbose": -1,
}
tree = lgb.train(
    tree_params,
    tree_train,
    num_boost_round=500,
    valid_sets=[tree_valid_set],
    callbacks=[lgb.early_stopping(40, verbose=False)],
)
tree_valid = tree.predict(
    xva_tree,
    num_iteration=tree.best_iteration,
    raw_score=True,
).astype(np.float64)
del xtr_tree, xva_tree, tree_train, tree_valid_set

# Select the established base ensemble.
base_candidates = {}
best_base_primary = -np.inf
best_base_weights = None
best_base_valid = None

for wt in [0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.50]:
    remaining = 1.0 - wt
    for fibi_fraction in [0.60, 0.70, 0.75, 0.80, 0.90]:
        wfm = remaining * (1.0 - fibi_fraction)
        wfibi = remaining * fibi_fraction
        score = (
            wfm * fm_valid
            + wfibi * fibi_valid
            + wt * tree_valid
        )
        primary = metric_primary(valid, score)
        name = "base_fm%.3f_fibi%.3f_tree%.3f" % (
            wfm, wfibi, wt
        )
        base_candidates[name] = primary
        if primary > best_base_primary:
            best_base_primary = primary
            best_base_weights = (wfm, wfibi, wt)
            best_base_valid = score.copy()

# New direction: train-only logged-context model.
xtr_context = make_context_matrix(train)
xva_context = make_context_matrix(valid)

context_train = lgb.Dataset(
    xtr_context,
    label=y_train,
    free_raw_data=False,
)
context_valid_set = lgb.Dataset(
    xva_context,
    label=np.asarray(valid.y, dtype=np.float32),
    reference=context_train,
    free_raw_data=False,
)
context_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.035,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "seed": SEED + 301,
    "feature_fraction_seed": SEED + 302,
    "bagging_seed": SEED + 303,
    "num_threads": N_THREADS,
    "verbose": -1,
}
context_model = lgb.train(
    context_params,
    context_train,
    num_boost_round=400,
    valid_sets=[context_valid_set],
    callbacks=[lgb.early_stopping(35, verbose=False)],
)
context_valid = context_model.predict(
    xva_context,
    num_iteration=context_model.best_iteration,
    raw_score=True,
).astype(np.float64)
del xtr_context, xva_context, context_train, context_valid_set

context_only_primary = metric_primary(valid, context_valid)

# Add context as a residual. Zero is included, so the established base
# remains available if the temporal mechanism does not transfer.
candidate_scores = {
    "fm": metric_primary(valid, fm_valid),
    "fibi": metric_primary(valid, fibi_valid),
    "tree": metric_primary(valid, tree_valid),
    "base": best_base_primary,
    "context_only": context_only_primary,
}
best_primary = best_base_primary
best_context_weight = 0.0
best_valid_scores = best_base_valid.copy()

for wc in [
    -0.20, -0.12, -0.08, -0.04, 0.0,
    0.04, 0.08, 0.12, 0.16, 0.20,
    0.28, 0.36, 0.45, 0.55,
]:
    scores = best_base_valid + wc * context_valid
    primary = metric_primary(valid, scores)
    candidate_scores["base_context_%+.2f" % wc] = primary
    if primary > best_primary:
        best_primary = primary
        best_context_weight = wc
        best_valid_scores = scores.copy()

best_metrics = evaluate(
    valid.user_id, valid.y, best_valid_scores
)

print(
    "FINDINGS context_only_primary=%.6f base_primary=%.6f "
    "selected_context_weight=%.3f context_best_iteration=%d"
    % (
        context_only_primary,
        best_base_primary,
        best_context_weight,
        context_model.best_iteration,
    ),
    flush=True,
)
print(
    "FINDINGS selected_base_weights fm=%.6f fibi=%.6f tree=%.6f"
    % best_base_weights,
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

# Required test scoring, using only validation-selected fixed weights.
test = load("test")

xte_fm, _ = make_offset_matrix(test, FM_FIELDS)
fm_test = predict_torch(fm, xte_fm)
del xte_fm

xte_fibi, _ = make_offset_matrix(test, FIBI_FIELDS)
fibi_test = predict_torch(fibi, xte_fibi)
del xte_fibi

xte_tree = make_tree_matrix(test)
tree_test = tree.predict(
    xte_tree,
    num_iteration=tree.best_iteration,
    raw_score=True,
).astype(np.float64)
del xte_tree

xte_context = make_context_matrix(test)
context_test = context_model.predict(
    xte_context,
    num_iteration=context_model.best_iteration,
    raw_score=True,
).astype(np.float64)
del xte_context

wfm, wfibi, wtree = best_base_weights
test_scores = (
    wfm * fm_test
    + wfibi * fibi_test
    + wtree * tree_test
    + best_context_weight * context_test
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        }
    ),
    flush=True,
)