import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
THREADS = min(8, os.cpu_count() or 1)
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 16384
EPOCHS = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
n_train = y_train.shape[0]

# Fixed recency weighting: recent training days better approximate the
# date-split evaluation distribution.
train_dates = np.asarray(train.date, dtype=np.int64)
age_days = train_dates.max() - train_dates
w_train = np.power(0.5, age_days.astype(np.float32) / 4.0).astype(np.float32)
w_train /= w_train.mean()

cards = [int(FEATURE_CARDINALITIES[name]) for name in CAT_FIELDS]
offsets = np.zeros(len(cards), dtype=np.int64)
offsets[1:] = np.cumsum(cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_categorical(split):
    result = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, name in enumerate(CAT_FIELDS):
        result[:, j] = np.asarray(split.X[name], dtype=np.int64) + offsets[j]
    return result


xcat_train = make_categorical(train)
xcat_valid = make_categorical(valid)
xcat_test = make_categorical(test)


def make_raw_numeric(split_name, split):
    columns = []

    for name in NUM_FIELDS:
        values = np.asarray(split.num[name], dtype=np.float32)
        if name != "user_register_days":
            values = np.log1p(np.maximum(values, 0.0))
        columns.append(values)

    # The API guarantees that these are train-only statistics. Training
    # values are leave-one-out and evaluation values use the full train split.
    for entity in ["video_id", "author_id"]:
        history = historical_features(split_name, key=entity)
        for name in sorted(history):
            columns.append(np.asarray(history[name], dtype=np.float32))

    return np.column_stack(columns).astype(np.float32)


raw_num_train = make_raw_numeric("train", train)
raw_num_valid = make_raw_numeric("valid", valid)
raw_num_test = make_raw_numeric("test", test)

finite_train = np.where(np.isfinite(raw_num_train), raw_num_train, np.nan)
num_mean = np.nanmean(finite_train, axis=0).astype(np.float32)
num_mean = np.where(np.isfinite(num_mean), num_mean, 0.0).astype(np.float32)

num_std = np.nanstd(finite_train, axis=0).astype(np.float32)
num_std = np.where(
    np.isfinite(num_std) & (num_std > 1e-5), num_std, 1.0
).astype(np.float32)


def normalize_numeric(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.where(np.isfinite(values), values, num_mean[None, :])
    values = (values - num_mean[None, :]) / num_std[None, :]
    return np.clip(values, -8.0, 8.0).astype(np.float32)


xnum_train = normalize_numeric(raw_num_train)
xnum_valid = normalize_numeric(raw_num_valid)
xnum_test = normalize_numeric(raw_num_test)
n_num = xnum_train.shape[1]

del raw_num_train, raw_num_valid, raw_num_test, finite_train

base_rate = float(np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6))
base_logit = float(np.log(base_rate / (1.0 - base_rate)))


class XDeepFM(nn.Module):
    """Deep tower plus a corrected compressed interaction network."""

    def __init__(self, rank=12, cin_widths=(12, 12)):
        super().__init__()
        self.n_fields = len(CAT_FIELDS)
        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.cin_layers = nn.ModuleList()
        previous_width = self.n_fields
        for width in cin_widths:
            self.cin_layers.append(
                nn.Conv1d(
                    in_channels=self.n_fields * previous_width,
                    out_channels=width,
                    kernel_size=1,
                )
            )
            previous_width = width

        deep_dim = self.n_fields * rank + n_num
        self.deep = nn.Sequential(
            nn.Linear(deep_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(64 + sum(cin_widths), 1)
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum):
        x0 = self.embedding(xcat)  # B x F x D
        xk = x0
        cin_outputs = []

        for layer in self.cin_layers:
            # Correct CIN construction. The failed implementation used an
            # einsum whose recurrent field dimension was not kept compatible.
            # This explicitly forms every x0-field by xk-channel product.
            products = x0.unsqueeze(2) * xk.unsqueeze(1)
            products = products.reshape(
                products.shape[0],
                self.n_fields * xk.shape[1],
                products.shape[3],
            )
            xk = F.relu(layer(products))
            cin_outputs.append(xk.sum(dim=2))

        deep_input = torch.cat([x0.flatten(1), xnum], dim=1)
        deep_output = self.deep(deep_input)
        cin_output = torch.cat(cin_outputs, dim=1)
        return self.bias + self.output(
            torch.cat([deep_output, cin_output], dim=1)
        ).squeeze(1)


class ProductNN(nn.Module):
    """A PNN using explicit pairwise inner products before its MLP."""

    def __init__(self, rank=14):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.pairs = [
            (i, j)
            for i in range(len(CAT_FIELDS))
            for j in range(i + 1, len(CAT_FIELDS))
        ]
        input_dim = len(CAT_FIELDS) * rank + len(self.pairs) + n_num
        self.network = nn.Sequential(
            nn.Linear(input_dim, 144),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(144, 72),
            nn.ReLU(),
            nn.Linear(72, 1),
        )
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum):
        embeddings = self.embedding(xcat)
        products = [
            (embeddings[:, i] * embeddings[:, j]).sum(dim=1, keepdim=True)
            for i, j in self.pairs
        ]
        product_features = torch.cat(products, dim=1)
        features = torch.cat(
            [embeddings.flatten(1), product_features, xnum], dim=1
        )
        return self.bias + self.network(features).squeeze(1)


class FiBiNET(nn.Module):
    """Field recalibration followed by field-specific bilinear interactions."""

    def __init__(self, rank=12):
        super().__init__()
        self.n_fields = len(CAT_FIELDS)
        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.02)

        squeeze_hidden = max(4, self.n_fields)
        self.senet = nn.Sequential(
            nn.Linear(self.n_fields, squeeze_hidden),
            nn.ReLU(),
            nn.Linear(squeeze_hidden, self.n_fields),
            nn.Sigmoid(),
        )

        self.pairs = [
            (i, j)
            for i in range(self.n_fields)
            for j in range(i + 1, self.n_fields)
        ]
        self.bilinear = nn.ModuleList(
            [nn.Linear(rank, rank, bias=False) for _ in self.pairs]
        )

        interaction_dim = len(self.pairs) * rank
        input_dim = self.n_fields * rank + interaction_dim + n_num
        self.network = nn.Sequential(
            nn.Linear(input_dim, 144),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(144, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.tensor(base_logit, dtype=torch.float32))

    def forward(self, xcat, xnum):
        embeddings = self.embedding(xcat)
        squeeze = embeddings.mean(dim=2)
        gates = self.senet(squeeze).unsqueeze(2)
        recalibrated = embeddings * gates

        interactions = []
        for layer, (i, j) in zip(self.bilinear, self.pairs):
            interactions.append(
                layer(recalibrated[:, i]) * recalibrated[:, j]
            )

        interaction_features = torch.cat(interactions, dim=1)
        features = torch.cat(
            [recalibrated.flatten(1), interaction_features, xnum], dim=1
        )
        return self.bias + self.network(features).squeeze(1)


class PLE(nn.Module):
    """
    One-level progressive layered extraction model.

    Long-view, click, and like receive task-specific experts while shared
    experts provide transfer. Only the long-view head is used at inference.
    """

    def __init__(self, rank=10, expert_dim=48, experts_per_group=2):
        super().__init__()
        self.n_tasks = 3
        self.experts_per_group = experts_per_group

        self.embedding = nn.Embedding(total_cardinality, rank)
        nn.init.normal_(self.embedding.weight, std=0.02)

        input_dim = len(CAT_FIELDS) * rank + n_num

        def expert():
            return nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(96, expert_dim),
                nn.ReLU(),
            )

        self.shared_experts = nn.ModuleList(
            [expert() for _ in range(experts_per_group)]
        )
        self.task_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [expert() for _ in range(experts_per_group)]
                )
                for _ in range(self.n_tasks)
            ]
        )

        gate_width = 2 * experts_per_group
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, gate_width) for _ in range(self.n_tasks)]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(expert_dim, 1) for _ in range(self.n_tasks)]
        )
        self.long_bias = nn.Parameter(
            torch.tensor(base_logit, dtype=torch.float32)
        )

    def forward(self, xcat, xnum):
        features = torch.cat(
            [self.embedding(xcat).flatten(1), xnum], dim=1
        )

        shared = torch.stack(
            [expert(features) for expert in self.shared_experts], dim=1
        )

        outputs = []
        for task in range(self.n_tasks):
            specific = torch.stack(
                [expert(features) for expert in self.task_experts[task]], dim=1
            )
            all_experts = torch.cat([specific, shared], dim=1)
            gate = torch.softmax(self.gates[task](features), dim=1).unsqueeze(2)
            representation = (all_experts * gate).sum(dim=1)
            logit = self.heads[task](representation).squeeze(1)
            if task == 0:
                logit = logit + self.long_bias
            outputs.append(logit)

        return tuple(outputs)


aux_click = np.asarray(train.aux["is_click"], dtype=np.float32)
aux_like = np.asarray(train.aux["is_like"], dtype=np.float32)


def train_model(model, seed, multitask=False):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.25e-3, weight_decay=2e-6
    )

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(n_train)
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]

            cat = torch.from_numpy(xcat_train[idx])
            num = torch.from_numpy(xnum_train[idx])
            target = torch.from_numpy(y_train[idx])
            weight = torch.from_numpy(w_train[idx])

            optimizer.zero_grad(set_to_none=True)
            output = model(cat, num)

            if multitask:
                long_logit, click_logit, like_logit = output

                long_loss = F.binary_cross_entropy_with_logits(
                    long_logit, target, reduction="none"
                )
                click_target = torch.from_numpy(aux_click[idx])
                like_target = torch.from_numpy(aux_like[idx])

                click_loss = F.binary_cross_entropy_with_logits(
                    click_logit, click_target, reduction="none"
                )
                like_loss = F.binary_cross_entropy_with_logits(
                    like_logit, like_target, reduction="none"
                )

                # Auxiliary outcomes are targets only, never input features.
                row_loss = long_loss + 0.16 * click_loss + 0.10 * like_loss
            else:
                row_loss = F.binary_cross_entropy_with_logits(
                    output, target, reduction="none"
                )

            loss = (row_loss * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float((row_loss.detach() * weight).sum())
            total_weight += float(weight.sum())

        print(
            "TRAIN family=%s epoch=%d loss=%.6f"
            % (
                model.__class__.__name__,
                epoch + 1,
                total_loss / total_weight,
            ),
            flush=True,
        )

    return model


def predict_model(model, xcat, xnum, multitask=False):
    result = np.empty(xcat.shape[0], dtype=np.float64)
    model.eval()

    with torch.inference_mode():
        for start in range(0, xcat.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, xcat.shape[0])
            output = model(
                torch.from_numpy(xcat[start:end]),
                torch.from_numpy(xnum[start:end]),
            )
            if multitask:
                output = output[0]
            result[start:end] = output.cpu().numpy()

    return result


def weighted_entity_rate(ids, cardinality, smoothing):
    ids = np.asarray(ids, dtype=np.int64)
    weighted_count = np.bincount(
        ids, weights=w_train, minlength=cardinality
    ).astype(np.float64)
    weighted_positive = np.bincount(
        ids, weights=w_train * y_train, minlength=cardinality
    ).astype(np.float64)

    weighted_global = float(np.sum(w_train * y_train) / np.sum(w_train))
    return (
        weighted_positive + smoothing * weighted_global
    ) / (weighted_count + smoothing)


def empirical_bayes_scores(split):
    video_ids = np.asarray(split.X["video_id"], dtype=np.int64)
    author_ids = np.asarray(split.X["author_id"], dtype=np.int64)
    duration_ids = np.asarray(split.X["duration_bucket"], dtype=np.int64)
    tab_ids = np.asarray(split.X["tab"], dtype=np.int64)

    probability = (
        0.52 * video_rate[video_ids]
        + 0.28 * author_rate[author_ids]
        + 0.12 * duration_rate[duration_ids]
        + 0.08 * tab_rate[tab_ids]
    )
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


video_rate = weighted_entity_rate(
    train.X["video_id"], FEATURE_CARDINALITIES["video_id"], smoothing=18.0
)
author_rate = weighted_entity_rate(
    train.X["author_id"], FEATURE_CARDINALITIES["author_id"], smoothing=24.0
)
duration_rate = weighted_entity_rate(
    train.X["duration_bucket"],
    FEATURE_CARDINALITIES["duration_bucket"],
    smoothing=100.0,
)
tab_rate = weighted_entity_rate(
    train.X["tab"], FEATURE_CARDINALITIES["tab"], smoothing=150.0
)


families = [
    ("xdeepfm", lambda: XDeepFM(rank=12, cin_widths=(12, 12)), False),
    ("pnn", lambda: ProductNN(rank=14), False),
    ("fibinet", lambda: FiBiNET(rank=12), False),
    ("ple", lambda: PLE(rank=10, expert_dim=48, experts_per_group=2), True),
]

predictions = {}

for family_index, (name, constructor, multitask) in enumerate(families):
    torch.manual_seed(SEED + 101 * family_index)
    model = constructor()
    model = train_model(
        model,
        seed=SEED + 1009 * family_index,
        multitask=multitask,
    )
    valid_prediction = predict_model(
        model, xcat_valid, xnum_valid, multitask=multitask
    )
    test_prediction = predict_model(
        model, xcat_test, xnum_test, multitask=multitask
    )
    predictions[name] = (valid_prediction, test_prediction)
    del model

predictions["empirical_bayes"] = (
    empirical_bayes_scores(valid),
    empirical_bayes_scores(test),
)


def within_user_rank(user_ids, scores):
    """
    Vectorized average ranks within each user. Tied scores receive their
    average rank, avoiding arbitrary row-order effects for empirical Bayes.
    """
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]

    order = np.lexsort((scores, user_ids))
    sorted_users = user_ids[order]
    sorted_scores = scores[order]

    user_start_flag = np.empty(n, dtype=bool)
    user_start_flag[0] = True
    user_start_flag[1:] = sorted_users[1:] != sorted_users[:-1]

    positions = np.arange(n, dtype=np.int64)
    user_starts = np.maximum.accumulate(
        np.where(user_start_flag, positions, 0)
    )

    user_start_indices = np.flatnonzero(user_start_flag)
    user_ends = np.r_[user_start_indices[1:], n]
    user_sizes = user_ends - user_start_indices
    repeated_user_sizes = np.repeat(user_sizes, user_sizes)

    tie_start_flag = np.empty(n, dtype=bool)
    tie_start_flag[0] = True
    tie_start_flag[1:] = (
        (sorted_users[1:] != sorted_users[:-1])
        | (sorted_scores[1:] != sorted_scores[:-1])
    )

    tie_starts_idx = np.flatnonzero(tie_start_flag)
    tie_ends_idx = np.r_[tie_starts_idx[1:], n]
    tie_sizes = tie_ends_idx - tie_starts_idx
    repeated_tie_starts = np.repeat(tie_starts_idx, tie_sizes)
    repeated_tie_sizes = np.repeat(tie_sizes, tie_sizes)

    average_position = (
        repeated_tie_starts
        - user_starts
        + 0.5 * (repeated_tie_sizes - 1)
    ).astype(np.float64)

    denominator = np.maximum(repeated_user_sizes - 1, 1)
    sorted_ranks = average_position / denominator
    sorted_ranks[repeated_user_sizes == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_ranks
    return result


shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")

has_incumbent = (
    os.path.isfile(inc_valid_path) and os.path.isfile(inc_test_path)
)

candidate_scores = {}
candidate_payload = {}

valid_users = np.asarray(valid.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)
test_users = np.asarray(test.user_id, dtype=np.int64)

if has_incumbent:
    incumbent_valid = np.asarray(
        np.load(inc_valid_path), dtype=np.float64
    )
    incumbent_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
    incumbent_valid_rank = within_user_rank(valid_users, incumbent_valid)
    incumbent_test_rank = within_user_rank(test_users, incumbent_test)

    incumbent_metrics = evaluate(
        valid_users, valid_labels, incumbent_valid
    )
    candidate_scores["trusted_incumbent"] = float(
        incumbent_metrics["primary"]
    )
    candidate_payload["trusted_incumbent"] = (
        incumbent_valid,
        incumbent_test,
        None,
        "incumbent",
    )

for family_name, (family_valid, family_test) in predictions.items():
    raw_metrics = evaluate(valid_users, valid_labels, family_valid)
    raw_name = family_name + "_raw"
    candidate_scores[raw_name] = float(raw_metrics["primary"])
    candidate_payload[raw_name] = (
        family_valid,
        family_test,
        None,
        family_name,
    )

    if has_incumbent:
        family_valid_rank = within_user_rank(valid_users, family_valid)
        family_test_rank = within_user_rank(test_users, family_test)

        for alpha in (0.25, 0.50, 0.75):
            blend_valid = (
                alpha * family_valid_rank
                + (1.0 - alpha) * incumbent_valid_rank
            )
            blend_test = (
                alpha * family_test_rank
                + (1.0 - alpha) * incumbent_test_rank
            )
            blend_name = "%s_blend_a%.2f" % (family_name, alpha)
            blend_metrics = evaluate(
                valid_users, valid_labels, blend_valid
            )
            candidate_scores[blend_name] = float(
                blend_metrics["primary"]
            )
            candidate_payload[blend_name] = (
                blend_valid,
                blend_test,
                family_valid,
                family_name,
            )

winner_name = max(candidate_scores, key=candidate_scores.get)
valid_scores, test_scores, raw_winner_scores, winner_family = (
    candidate_payload[winner_name]
)
metrics = evaluate(valid_users, valid_labels, valid_scores)

print(
    "FINDINGS winner=%s family=%s primary=%.6f"
    % (winner_name, winner_family, metrics["primary"]),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if raw_winner_scores is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_winner_scores, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        metrics["primary"],
        metrics["gauc"],
        metrics["ndcg@5"],
        elapsed,
    )
)