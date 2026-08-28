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
CLICK_LOSS_WEIGHT = 0.20
N_EXPERTS = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


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
    def __init__(self, n_tokens, n_fields, embed_dim, n_experts):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, embed_dim)
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
        self.bilinear = nn.Linear(embed_dim, embed_dim, bias=False)

        left, right = [], []
        for i in range(n_fields):
            for j in range(i + 1, n_fields):
                left.append(i)
                right.append(j)
        self.register_buffer(
            "pair_left", torch.tensor(left, dtype=torch.long)
        )
        self.register_buffer(
            "pair_right", torch.tensor(right, dtype=torch.long)
        )

        n_pairs = len(left)
        mmoe_input = (n_fields + n_pairs) * embed_dim

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(mmoe_input, 72),
                    nn.ReLU(),
                    nn.Dropout(0.08),
                    nn.Linear(72, 32),
                    nn.ReLU(),
                )
                for _ in range(n_experts)
            ]
        )
        self.long_gate = nn.Linear(mmoe_input, n_experts)
        self.click_gate = nn.Linear(mmoe_input, n_experts)

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

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.03)
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
        pairs = (
            transformed.index_select(1, self.pair_left)
            * se.index_select(1, self.pair_right)
        )
        features = torch.cat(
            [se.flatten(start_dim=1), pairs.flatten(start_dim=1)],
            dim=1,
        )

        expert_outputs = torch.stack(
            [expert(features) for expert in self.experts],
            dim=1,
        )

        long_weights = torch.softmax(self.long_gate(features), dim=1)
        long_shared = torch.sum(
            expert_outputs * long_weights.unsqueeze(2), dim=1
        )
        long_logit = (
            self.long_bias
            + self.long_wide(x).squeeze(2).sum(dim=1)
            + self.long_tower(long_shared).squeeze(1)
        )

        if not return_click:
            return long_logit

        click_weights = torch.softmax(self.click_gate(features), dim=1)
        click_shared = torch.sum(
            expert_outputs * click_weights.unsqueeze(2), dim=1
        )
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


def signed_log1p(x):
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


def safe_ratio(a, b, scale=1.0):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    result = (scale * a) / (np.abs(b) + 1.0)
    invalid = ~np.isfinite(a) | ~np.isfinite(b)
    result[invalid] = np.nan
    return np.asarray(result, dtype=np.float32)


def make_tree_matrix(split):
    columns = []

    for name in TREE_CAT_FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.float32))

    for name in NUM_FIELDS:
        columns.append(signed_log1p(split.num[name]).astype(np.float32))

    n = split.num
    ratios = [
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
        safe_ratio(n["play_duration"], n["play_cnt"] * n["duration_ms"]),
        safe_ratio(n["long_time_play_cnt"], n["play_cnt"]),
        safe_ratio(n["complete_play_cnt"], n["play_cnt"]),
        safe_ratio(n["valid_play_cnt"], n["play_cnt"]),
        safe_ratio(
            n["like_cnt"]
            + n["comment_cnt"]
            + n["share_cnt"]
            + n["collect_cnt"]
            + n["follow_cnt"],
            n["show_cnt"],
        ),
    ]
    columns.extend(ratios)

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def train_multitask(
    model,
    optimizer,
    x,
    y_long,
    y_click,
    epochs,
    batch_size,
    seed,
):
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    n = len(y_long)

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        total_loss = 0.0
        total_long = 0.0
        total_click = 0.0

        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.from_numpy(x[idx])
            ylb = torch.from_numpy(y_long[idx])
            ycb = torch.from_numpy(y_click[idx])

            optimizer.zero_grad(set_to_none=True)
            long_logits, click_logits = model(xb, return_click=True)
            long_loss = criterion(long_logits, ylb)
            click_loss = criterion(click_logits, ycb)
            loss = long_loss + CLICK_LOSS_WEIGHT * click_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_n = len(idx)
            total_loss += float(loss.detach()) * batch_n
            total_long += float(long_loss.detach()) * batch_n
            total_click += float(click_loss.detach()) * batch_n

        print(
            "train model=FiBiMMoE epoch=%d loss=%.6f long=%.6f click=%.6f"
            % (
                epoch + 1,
                total_loss / n,
                total_long / n,
                total_click / n,
            ),
            flush=True,
        )


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
click_train = np.asarray(train.aux["is_click"], dtype=np.float32)

# Reliable categorical FM component.
xtr_fm, fm_tokens = make_offset_matrix(train, FM_FIELDS)
xva_fm, _ = make_offset_matrix(valid, FM_FIELDS)

fm = FactorizationMachine(fm_tokens, FM_RANK)
fm_sparse_opt = torch.optim.SparseAdam(
    [fm.embedding.weight], lr=FM_LR
)
fm_bias_opt = torch.optim.Adam([fm.bias], lr=FM_LR)

criterion = nn.BCEWithLogitsLoss()
rng_fm = np.random.default_rng(SEED)
for epoch in range(FM_EPOCHS):
    fm.train()
    permutation = rng_fm.permutation(len(y_train))
    total_loss = 0.0

    for start in range(0, len(y_train), FM_BATCH):
        idx = permutation[start:start + FM_BATCH]
        xb = torch.from_numpy(xtr_fm[idx])
        yb = torch.from_numpy(y_train[idx])

        fm_sparse_opt.zero_grad(set_to_none=True)
        fm_bias_opt.zero_grad(set_to_none=True)
        loss = criterion(fm(xb), yb)
        loss.backward()
        fm_sparse_opt.step()
        fm_bias_opt.step()
        total_loss += float(loss.detach()) * len(idx)

    print(
        "train model=FM epoch=%d loss=%.6f"
        % (epoch + 1, total_loss / len(y_train)),
        flush=True,
    )

fm_valid = predict_torch(fm, xva_fm)
fm_metrics = evaluate(valid.user_id, valid.y, fm_valid)

# Replace only the FiBiNET supervision/tower with click-regularized MMoE.
xtr_fibi, fibi_tokens = make_offset_matrix(train, FIBI_FIELDS)
xva_fibi, _ = make_offset_matrix(valid, FIBI_FIELDS)

torch.manual_seed(SEED + 17)
fibi = FiBiMMoE(
    fibi_tokens,
    len(FIBI_FIELDS),
    FIBI_DIM,
    N_EXPERTS,
)
fibi_opt = torch.optim.AdamW(
    fibi.parameters(),
    lr=FIBI_LR,
    weight_decay=1e-6,
)
train_multitask(
    fibi,
    fibi_opt,
    xtr_fibi,
    y_train,
    click_train,
    FIBI_EPOCHS,
    FIBI_BATCH,
    SEED + 17,
)
fibi_valid = predict_torch(fibi, xva_fibi)
fibi_metrics = evaluate(valid.user_id, valid.y, fibi_valid)

# Preserve the previously selected categorical blend as the starting point.
base_valid = 0.25 * fm_valid + 0.75 * fibi_valid
base_metrics = evaluate(valid.user_id, valid.y, base_valid)

# Preserve the complementary numeric LightGBM component.
xtr_tree = make_tree_matrix(train)
xva_tree = make_tree_matrix(valid)
cat_indices = list(range(len(TREE_CAT_FIELDS)))

dtrain = lgb.Dataset(
    xtr_tree,
    label=np.asarray(train.y, dtype=np.float32),
    categorical_feature=cat_indices,
    free_raw_data=False,
)
dvalid = lgb.Dataset(
    xva_tree,
    label=np.asarray(valid.y, dtype=np.float32),
    categorical_feature=cat_indices,
    reference=dtrain,
    free_raw_data=False,
)

tree_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "max_depth": -1,
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
    "num_threads": max(1, min(16, os.cpu_count() or 1)),
    "verbose": -1,
}

tree = lgb.train(
    tree_params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(40, verbose=False)],
)
tree_valid = tree.predict(
    xva_tree,
    num_iteration=tree.best_iteration,
    raw_score=True,
).astype(np.float64)
tree_metrics = evaluate(valid.user_id, valid.y, tree_valid)

# Validation-select model weights, retaining the 25/75 categorical blend.
best_primary = float(base_metrics["primary"])
best_weights = (0.25, 0.75, 0.0)
best_valid = base_valid.copy()

tree_weights = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.50]
fibi_fractions = [0.60, 0.70, 0.75, 0.80, 0.90]

for wt in tree_weights:
    remaining = 1.0 - wt
    for fibi_fraction in fibi_fractions:
        wf = remaining * (1.0 - fibi_fraction)
        wfi = remaining * fibi_fraction
        scores = wf * fm_valid + wfi * fibi_valid + wt * tree_valid
        metrics = evaluate(valid.user_id, valid.y, scores)
        primary = float(metrics["primary"])
        if primary > best_primary:
            best_primary = primary
            best_weights = (wf, wfi, wt)
            best_valid = scores.copy()

final_metrics = evaluate(valid.user_id, valid.y, best_valid)

print(
    "CANDIDATES "
    + json.dumps(
        {
            "fm": float(fm_metrics["primary"]),
            "click_mmoe_fibinet": float(fibi_metrics["primary"]),
            "categorical_ensemble": float(base_metrics["primary"]),
            "numeric_lightgbm": float(tree_metrics["primary"]),
            "selected_three_way": float(final_metrics["primary"]),
        },
        separators=(",", ":"),
    ),
    flush=True,
)
print(
    "FINDINGS tree_best_iteration=%d click_loss_weight=%.3f "
    "selected_weights_fm=%.4f_fibi_mmoe=%.4f_tree=%.4f"
    % (
        int(tree.best_iteration),
        CLICK_LOSS_WEIGHT,
        best_weights[0],
        best_weights[1],
        best_weights[2],
    ),
    flush=True,
)

# Score hidden test using only validation-selected settings. Test outcomes are
# neither read nor used.
test = load("test")
xte_fm, _ = make_offset_matrix(test, FM_FIELDS)
xte_fibi, _ = make_offset_matrix(test, FIBI_FIELDS)
xte_tree = make_tree_matrix(test)

fm_test = predict_torch(fm, xte_fm)
fibi_test = predict_torch(fibi, xte_fibi)
tree_test = tree.predict(
    xte_tree,
    num_iteration=tree.best_iteration,
    raw_score=True,
).astype(np.float64)

test_scores = (
    best_weights[0] * fm_test
    + best_weights[1] * fibi_test
    + best_weights[2] * tree_test
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
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        },
        separators=(",", ":"),
    )
)