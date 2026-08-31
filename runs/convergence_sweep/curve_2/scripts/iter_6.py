import os
import time
import json
import math
import random
import gc

import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "register_days_bucket",
    "music_type",
    "video_type",
]
CARDS = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))
N_FIELDS = len(FIELDS)
EMBED_DIM = 10


def make_cat(split):
    return np.ascontiguousarray(
        np.stack(
            [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS],
            axis=1,
        ),
        dtype=np.int64,
    )


x_train_np = make_cat(train)
x_valid_np = make_cat(valid)
x_test_np = make_cat(test)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

click_train_np = np.nan_to_num(
    np.asarray(train.aux["is_click"], dtype=np.float32),
    nan=0.0,
    posinf=1.0,
    neginf=0.0,
)
like_train_np = np.nan_to_num(
    np.asarray(train.aux["is_like"], dtype=np.float32),
    nan=0.0,
    posinf=1.0,
    neginf=0.0,
)
click_train_np = np.clip(click_train_np, 0.0, 1.0).astype(np.float32)
like_train_np = np.clip(like_train_np, 0.0, 1.0).astype(np.float32)
click_train = torch.from_numpy(click_train_np)
like_train = torch.from_numpy(like_train_np)

# The same moderate train-only recency weighting is used for every family,
# ensuring the family comparison is not confounded by a different date policy.
max_train_date = int(np.max(np.asarray(train.date, dtype=np.int32)))
train_age = (
    max_train_date - np.asarray(train.date, dtype=np.int32)
).astype(np.float32)
sample_weight_np = np.exp(
    -math.log(2.0) * train_age / 6.0
).astype(np.float32)
sample_weight_np /= float(sample_weight_np.mean())
sample_weight = torch.from_numpy(sample_weight_np)

offset_tensor = torch.from_numpy(OFFSETS.copy())
pair_index = torch.triu_indices(N_FIELDS, N_FIELDS, offset=1)
N_PAIRS = int(pair_index.shape[1])


class XDeepFM(nn.Module):
    """
    A compact xDeepFM: the CIN explicitly forms vector-wise field crosses,
    while the deep branch models implicit nonlinear interactions.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)

        self.cin1 = nn.Linear(N_FIELDS * N_FIELDS, 12)
        self.cin2 = nn.Linear(N_FIELDS * 12, 10)

        self.deep = nn.Sequential(
            nn.Linear(N_FIELDS * EMBED_DIM, 96),
            nn.PReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 40),
            nn.PReLU(),
        )
        self.output = nn.Linear(12 + 10 + 40, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def cin_layer(self, x0, hidden, projection):
        # x0: B,F,D and hidden: B,H,D. Crosses are projected separately
        # for each embedding coordinate, preserving the CIN vector structure.
        outer = torch.einsum("bfd,bhd->bfhd", x0, hidden)
        b, f, h, d = outer.shape
        outer = outer.reshape(b, f * h, d).transpose(1, 2)
        value = torch.relu(projection(outer))
        return value.transpose(1, 2)

    def forward(self, x):
        ids = x + self.offsets
        emb = self.embedding(ids)
        linear = self.linear(ids).sum(dim=1).squeeze(1)

        h1 = self.cin_layer(emb, emb, self.cin1)
        h2 = self.cin_layer(emb, h1, self.cin2)
        cin = torch.cat([h1.sum(dim=2), h2.sum(dim=2)], dim=1)

        deep = self.deep(emb.flatten(1))
        interaction = self.output(torch.cat([cin, deep], dim=1)).squeeze(1)
        return self.bias + linear + interaction


class FiBiNET(nn.Module):
    """
    SENet reweights fields conditional on the current impression, followed by
    field-pair-specific bilinear interactions rather than a shared FM product.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())
        self.register_buffer("pair_i", pair_index[0].clone())
        self.register_buffer("pair_j", pair_index[1].clone())

        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)

        self.senet = nn.Sequential(
            nn.Linear(N_FIELDS, 8),
            nn.ReLU(),
            nn.Linear(8, N_FIELDS),
            nn.Sigmoid(),
        )
        self.bilinear = nn.Parameter(
            torch.empty(N_PAIRS, EMBED_DIM, EMBED_DIM)
        )
        self.output = nn.Sequential(
            nn.Linear(N_FIELDS * EMBED_DIM + N_PAIRS, 96),
            nn.PReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)
        nn.init.xavier_uniform_(self.bilinear)

    def forward(self, x):
        ids = x + self.offsets
        emb = self.embedding(ids)
        squeeze = emb.mean(dim=2)
        field_weight = 0.5 + self.senet(squeeze)
        reweighted = emb * field_weight.unsqueeze(2)

        left = reweighted[:, self.pair_i, :]
        right = reweighted[:, self.pair_j, :]
        transformed = torch.einsum(
            "bpd,pde->bpe", left, self.bilinear
        )
        pair_scores = (transformed * right).sum(dim=2)

        linear = self.linear(ids).sum(dim=1).squeeze(1)
        z = torch.cat([reweighted.flatten(1), pair_scores], dim=1)
        return self.bias + linear + self.output(z).squeeze(1)


class ESMM(nn.Module):
    """
    Models P(long_view) as P(click) * P(long_view | click) over the whole
    impression space. Click is used only as a train-time target.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)

        in_dim = N_FIELDS * EMBED_DIM
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 112),
            nn.PReLU(),
            nn.Dropout(0.04),
            nn.Linear(112, 56),
            nn.PReLU(),
        )
        self.click_tower = nn.Sequential(
            nn.Linear(56, 28),
            nn.PReLU(),
            nn.Linear(28, 1),
        )
        self.conditional_tower = nn.Sequential(
            nn.Linear(56, 28),
            nn.PReLU(),
            nn.Linear(28, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.025)

    def probabilities(self, x):
        z = self.embedding(x + self.offsets).flatten(1)
        shared = self.shared(z)
        click_logit = self.click_tower(shared).squeeze(1)
        conditional_logit = self.conditional_tower(shared).squeeze(1)
        click_prob = torch.sigmoid(click_logit)
        conditional_prob = torch.sigmoid(conditional_logit)
        long_prob = click_prob * conditional_prob
        return click_logit, conditional_logit, long_prob

    def forward(self, x):
        _, _, probability = self.probabilities(x)
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(probability)


class PLE(nn.Module):
    """
    A one-level Progressive Layered Extraction network. Long-view, click,
    and like each receive task-specific experts plus shared experts, allowing
    the long-view task to reject harmful auxiliary-task information.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", offset_tensor.clone())
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        in_dim = N_FIELDS * EMBED_DIM
        hidden = 48
        self.n_tasks = 3

        def expert():
            return nn.Sequential(
                nn.Linear(in_dim, 72),
                nn.PReLU(),
                nn.Dropout(0.03),
                nn.Linear(72, hidden),
                nn.PReLU(),
            )

        self.shared_experts = nn.ModuleList([expert(), expert()])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([expert(), expert()])
            for _ in range(self.n_tasks)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, 4) for _ in range(self.n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 24),
                nn.PReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(self.n_tasks)
        ])
        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, x):
        z = self.embedding(x + self.offsets).flatten(1)
        shared = [expert(z) for expert in self.shared_experts]
        outputs = []

        for task in range(self.n_tasks):
            specific = [
                expert(z) for expert in self.task_experts[task]
            ]
            candidates = torch.stack(specific + shared, dim=1)
            gate = torch.softmax(self.gates[task](z), dim=1).unsqueeze(2)
            representation = (candidates * gate).sum(dim=1)
            outputs.append(self.towers[task](representation).squeeze(1))

        return torch.stack(outputs, dim=1)


@torch.no_grad()
def predict(model, x_np, batch_size=16384):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    for begin in range(0, len(x_np), batch_size):
        end = min(begin + batch_size, len(x_np))
        xb = torch.from_numpy(x_np[begin:end])
        output = model(xb)
        if output.ndim == 2:
            output = output[:, 0]
        result[begin:end] = output.detach().cpu().numpy()
    return result


def weighted_mean(loss_vector, weights):
    return (loss_vector * weights).sum() / weights.sum()


def training_loss(model, xb, yb, cb, lb, weights):
    if isinstance(model, ESMM):
        click_logit, _, long_probability = model.probabilities(xb)
        long_probability = long_probability.clamp(1e-6, 1.0 - 1e-6)
        long_loss = nn.functional.binary_cross_entropy(
            long_probability, yb, reduction="none"
        )
        click_loss = nn.functional.binary_cross_entropy_with_logits(
            click_logit, cb, reduction="none"
        )
        return weighted_mean(long_loss + 0.30 * click_loss, weights)

    output = model(xb)
    if isinstance(model, PLE):
        long_loss = nn.functional.binary_cross_entropy_with_logits(
            output[:, 0], yb, reduction="none"
        )
        click_loss = nn.functional.binary_cross_entropy_with_logits(
            output[:, 1], cb, reduction="none"
        )
        like_loss = nn.functional.binary_cross_entropy_with_logits(
            output[:, 2], lb, reduction="none"
        )
        combined = long_loss + 0.22 * click_loss + 0.10 * like_loss
        return weighted_mean(combined, weights)

    loss = nn.functional.binary_cross_entropy_with_logits(
        output, yb, reduction="none"
    )
    return weighted_mean(loss, weights)


def train_family(model, name, epochs=2):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.15e-3,
        weight_decay=3e-6,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + sum(ord(c) for c in name))

    best_primary = -1.0
    best_state = None
    epoch_scores = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(
            len(y_train_np), generator=generator
        )
        for begin in range(0, len(y_train_np), 4096):
            idx = permutation[begin:begin + 4096]
            xb = x_train[idx]
            loss = training_loss(
                model,
                xb,
                y_train[idx],
                click_train[idx],
                like_train[idx],
                sample_weight[idx],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        validation_scores = predict(model, x_valid_np)
        metrics = evaluate(
            valid.user_id, valid.y, validation_scores
        )
        primary = float(metrics["primary"])
        epoch_scores.append(primary)

        if primary > best_primary:
            best_primary = primary
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    validation_scores = predict(model, x_valid_np)
    test_scores = predict(model, x_test_np)

    print(
        "FINDINGS %s_epoch_primary=%s"
        % (name, json.dumps(epoch_scores))
    )
    return validation_scores, test_scores


def within_user_rank(user_ids, scores):
    """
    Rank normalization is performed separately inside every logged user set.
    It requires no labels and makes rank aggregation insensitive to family-
    specific logit calibration.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(len(values)), values, users))
    sorted_users = users[order]

    new_group = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.maximum.accumulate(
        np.where(new_group, np.arange(len(values)), 0)
    )
    position = np.arange(len(values)) - starts

    _, counts = np.unique(sorted_users, return_counts=True)
    repeated_counts = np.repeat(counts, counts)
    ranked_sorted = (position + 0.5) / repeated_counts

    result = np.empty(len(values), dtype=np.float64)
    result[order] = ranked_sorted
    return result


families = [
    ("xdeepfm", XDeepFM),
    ("fibinet", FiBiNET),
    ("esmm", ESMM),
    ("ple", PLE),
]

valid_predictions = {}
test_predictions = {}

for family_name, constructor in families:
    model = constructor()
    va_scores, te_scores = train_family(model, family_name, epochs=2)
    valid_predictions[family_name] = va_scores
    test_predictions[family_name] = te_scores
    del model
    gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_log = {}
best_primary = -1.0
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_name = None
best_metric = None
best_is_blend = False

blend_weights = [0.25, 0.50, 0.75, 0.90]

for family_name, _ in families:
    raw_valid = valid_predictions[family_name]
    raw_test = test_predictions[family_name]

    standalone_metric = evaluate(
        valid.user_id, valid.y, raw_valid
    )
    standalone_primary = float(standalone_metric["primary"])
    candidate_log[family_name] = standalone_primary

    if standalone_primary > best_primary:
        best_primary = standalone_primary
        best_valid_scores = raw_valid.astype(np.float64)
        best_test_scores = raw_test.astype(np.float64)
        best_raw_valid = raw_valid.astype(np.float64)
        best_name = family_name
        best_metric = standalone_metric
        best_is_blend = False

    family_valid_rank = within_user_rank(
        valid.user_id, raw_valid
    )
    family_test_rank = within_user_rank(
        test.user_id, raw_test
    )

    family_best_blend_primary = -1.0
    family_best_alpha = None

    for alpha in blend_weights:
        blended_valid = (
            alpha * family_valid_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        metric = evaluate(
            valid.user_id, valid.y, blended_valid
        )
        primary = float(metric["primary"])

        if primary > family_best_blend_primary:
            family_best_blend_primary = primary
            family_best_alpha = alpha

        if primary > best_primary:
            blended_test = (
                alpha * family_test_rank
                + (1.0 - alpha) * inc_test_rank
            )
            best_primary = primary
            best_valid_scores = blended_valid.astype(np.float64)
            best_test_scores = blended_test.astype(np.float64)
            best_raw_valid = raw_valid.astype(np.float64)
            best_name = "%s_rankblend_%.2f" % (family_name, alpha)
            best_metric = metric
            best_is_blend = True

    candidate_log[
        family_name + "_best_incumbent_blend"
    ] = family_best_blend_primary
    print(
        "FINDINGS %s_best_blend_alpha=%.2f"
        % (family_name, family_best_alpha)
    )

print("FINDINGS selected_candidate=%s" % best_name)
print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    if best_is_blend:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        float(best_metric["primary"]),
        float(best_metric["gauc"]),
        float(best_metric["ndcg@5"]),
        elapsed,
    )
)