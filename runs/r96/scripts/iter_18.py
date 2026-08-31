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
SEED = 91731
EPOCHS = 2
PAIR_BATCH_SIZE = 4096
INFER_BATCH_SIZE = 8192
EMBED_DIM = 12
TOWER_DIM = 48

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

USER_FIELDS = [
    "user_id",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "register_days_bucket",
]

ITEM_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
]

CONTEXT_FIELDS = [
    "tab",
    "hour",
    "is_live_streamer",
    "is_lowactive_period",
    "is_video_author",
]

CAT_FIELDS = USER_FIELDS + ITEM_FIELDS + CONTEXT_FIELDS

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

FIELD_INDEX = {name: i for i, name in enumerate(CAT_FIELDS)}
USER_INDEX = [FIELD_INDEX[name] for name in USER_FIELDS]
ITEM_INDEX = [FIELD_INDEX[name] for name in ITEM_FIELDS]
CONTEXT_INDEX = [FIELD_INDEX[name] for name in CONTEXT_FIELDS]

USER_NUM_INDEX = [1, 2, 3, 4]
ITEM_NUM_INDEX = [0]


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=np.bool_)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    first = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=np.bool_)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.array([-1], dtype=np.int64),
        end_positions,
    )))
    row_sizes = np.repeat(sizes, sizes)
    position = np.arange(n, dtype=np.int64) - first

    ranked = (position.astype(np.float64) + 0.5) / row_sizes
    output = np.empty(n, dtype=np.float64)
    output[order] = ranked
    return output


def fit_numeric_state(train):
    state = {}
    for name in NUM_FIELDS:
        x = np.asarray(train.num[name], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        mean = float(x.mean())
        std = max(float(x.std()), 1e-5)
        state[name] = (mean, std)
    return state


def make_arrays(split, numeric_state):
    cats = np.column_stack([
        np.asarray(split.X[name], dtype=np.int32)
        for name in CAT_FIELDS
    ])

    nums = np.empty((len(split), len(NUM_FIELDS)), dtype=np.float32)
    for j, name in enumerate(NUM_FIELDS):
        x = np.asarray(split.num[name], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        mean, std = numeric_state[name]
        nums[:, j] = np.clip((x - mean) / std, -8.0, 8.0)

    return cats, nums


def make_pairs(users, labels, rng):
    n = len(labels)
    random_key = rng.random(n)
    order = np.lexsort((random_key, users))

    positives = []
    negatives = []

    for offset in (1, 2, 4):
        left = order[:-offset]
        right = order[offset:]

        different_label = labels[left] != labels[right]
        same_user = users[left] == users[right]
        keep = same_user & different_label

        left = left[keep]
        right = right[keep]

        left_positive = labels[left] > labels[right]
        pos = np.where(left_positive, left, right)
        neg = np.where(left_positive, right, left)

        positives.append(pos.astype(np.int64, copy=False))
        negatives.append(neg.astype(np.int64, copy=False))

    pos = np.concatenate(positives)
    neg = np.concatenate(negatives)

    permutation = rng.permutation(len(pos))
    return pos[permutation], neg[permutation]


class TwoTowerBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[name]),
                EMBED_DIM,
                padding_idx=0,
            )
            for name in CAT_FIELDS
        ])

        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[name]),
                1,
                padding_idx=0,
            )
            for name in CAT_FIELDS
        ])

        user_input = len(USER_FIELDS) * EMBED_DIM + len(USER_NUM_INDEX)
        item_input = len(ITEM_FIELDS) * EMBED_DIM + len(ITEM_NUM_INDEX)
        context_input = len(CONTEXT_FIELDS) * EMBED_DIM

        self.user_tower = nn.Sequential(
            nn.Linear(user_input, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(128, TOWER_DIM),
        )

        self.item_tower = nn.Sequential(
            nn.Linear(item_input, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(160, TOWER_DIM),
        )

        self.context_projection = nn.Sequential(
            nn.Linear(context_input, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

        self.numeric_linear = nn.Linear(len(NUM_FIELDS), 1)
        self.global_bias = nn.Parameter(torch.zeros(()))

    def encode(self, cats, nums):
        embedded = [
            embedding(cats[:, j])
            for j, embedding in enumerate(self.embeddings)
        ]

        user_input = torch.cat(
            [embedded[j] for j in USER_INDEX]
            + [nums[:, USER_NUM_INDEX]],
            dim=1,
        )
        item_input = torch.cat(
            [embedded[j] for j in ITEM_INDEX]
            + [nums[:, ITEM_NUM_INDEX]],
            dim=1,
        )
        context_input = torch.cat(
            [embedded[j] for j in CONTEXT_INDEX],
            dim=1,
        )

        user_vector = self.user_tower(user_input)
        item_vector = self.item_tower(item_input)
        context_score = self.context_projection(context_input).squeeze(1)

        wide = self.global_bias + self.numeric_linear(nums).squeeze(1)
        for j, embedding in enumerate(self.linear_embeddings):
            wide = wide + embedding(cats[:, j]).squeeze(1)

        return user_vector, item_vector, context_score + wide


class DotProductTower(TwoTowerBase):
    def forward(self, cats, nums):
        user_vector, item_vector, side_score = self.encode(cats, nums)
        match = (
            user_vector * item_vector
        ).sum(dim=1) / np.sqrt(float(TOWER_DIM))
        return match + side_score


class MetricDistanceTower(TwoTowerBase):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.user_metric = nn.Linear(TOWER_DIM, TOWER_DIM, bias=False)
        self.item_metric = nn.Linear(TOWER_DIM, TOWER_DIM, bias=False)

    def forward(self, cats, nums):
        user_vector, item_vector, side_score = self.encode(cats, nums)
        user_vector = self.user_metric(user_vector)
        item_vector = self.item_metric(item_vector)
        distance = torch.square(user_vector - item_vector).mean(dim=1)
        scale = torch.exp(self.log_scale).clamp(0.05, 20.0)
        return -scale * distance + side_score


class NonlinearMatchingTower(TwoTowerBase):
    def __init__(self):
        super().__init__()
        self.matcher = nn.Sequential(
            nn.Linear(TOWER_DIM * 4, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(160, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cats, nums):
        user_vector, item_vector, side_score = self.encode(cats, nums)
        interaction = torch.cat([
            user_vector,
            item_vector,
            user_vector * item_vector,
            torch.abs(user_vector - item_vector),
        ], dim=1)
        return self.matcher(interaction).squeeze(1) + side_score


def train_model(model, train_cats, train_nums, users, labels,
                recency_weights, seed_offset):
    torch.manual_seed(SEED + seed_offset)
    rng = np.random.default_rng(SEED + seed_offset)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0018,
        weight_decay=3e-5,
    )

    losses = []
    pair_counts = []

    model.train()
    for epoch in range(EPOCHS):
        pos, neg = make_pairs(users, labels, rng)
        pair_counts.append(int(len(pos)))

        epoch_loss_sum = 0.0
        epoch_weight_sum = 0.0

        for start in range(0, len(pos), PAIR_BATCH_SIZE):
            p = pos[start:start + PAIR_BATCH_SIZE]
            q = neg[start:start + PAIR_BATCH_SIZE]

            p_cats = torch.from_numpy(train_cats[p]).long()
            p_nums = torch.from_numpy(train_nums[p])
            q_cats = torch.from_numpy(train_cats[q]).long()
            q_nums = torch.from_numpy(train_nums[q])

            p_score = model(p_cats, p_nums)
            q_score = model(q_cats, q_nums)

            pair_weight_np = np.sqrt(
                recency_weights[p] * recency_weights[q]
            ).astype(np.float32)
            pair_weight = torch.from_numpy(pair_weight_np)

            pair_loss = nn.functional.softplus(-(p_score - q_score))

            # A small calibrated auxiliary term stabilizes entity biases while
            # leaving the within-user pairwise objective dominant.
            positive_bce = nn.functional.softplus(-p_score)
            negative_bce = nn.functional.softplus(q_score)
            element_loss = (
                pair_loss
                + 0.075 * positive_bce
                + 0.075 * negative_bce
            )

            denominator = pair_weight.sum().clamp_min(1.0)
            loss = (element_loss * pair_weight).sum() / denominator

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss_sum += float(
                (element_loss * pair_weight).sum().detach()
            )
            epoch_weight_sum += float(denominator.detach())

        losses.append(epoch_loss_sum / max(epoch_weight_sum, 1.0))

    return losses, pair_counts


@torch.no_grad()
def predict_model(model, cats, nums):
    model.eval()
    output = np.empty(len(cats), dtype=np.float64)

    for start in range(0, len(cats), INFER_BATCH_SIZE):
        end = min(start + INFER_BATCH_SIZE, len(cats))
        batch_cats = torch.from_numpy(cats[start:end]).long()
        batch_nums = torch.from_numpy(nums[start:end])
        output[start:end] = model(
            batch_cats, batch_nums
        ).cpu().numpy()

    return output


train = load("train")
valid = load("valid")
test = load("test")

numeric_state = fit_numeric_state(train)
train_cats, train_nums = make_arrays(train, numeric_state)
valid_cats, valid_nums = make_arrays(valid, numeric_state)
test_cats, test_nums = make_arrays(test, numeric_state)

train_users = np.asarray(train.user_id, dtype=np.int64)
train_labels = np.asarray(train.y, dtype=np.int8)
train_dates = np.asarray(train.date, dtype=np.int32)

age_days = np.maximum(
    int(train_dates.max()) - train_dates,
    0,
).astype(np.float32)
recency_weights = np.power(0.5, age_days / 4.0).astype(np.float32)
recency_weights /= max(float(recency_weights.mean()), 1e-6)

constructors = [
    ("dot_product_tower", DotProductTower),
    ("metric_distance_tower", MetricDistanceTower),
    ("nonlinear_matching_tower", NonlinearMatchingTower),
]

family_valid = {}
family_test = {}
training_findings = {}
failures = {}

for family_index, (name, constructor) in enumerate(constructors):
    try:
        model = constructor()
        loss, pair_counts = train_model(
            model,
            train_cats,
            train_nums,
            train_users,
            train_labels,
            recency_weights,
            seed_offset=1000 * (family_index + 1),
        )
        family_valid[name] = predict_model(model, valid_cats, valid_nums)
        family_test[name] = predict_model(model, test_cats, test_nums)
        training_findings[name] = {
            "loss": loss,
            "pair_counts": pair_counts,
        }
        del model
        gc.collect()
    except Exception as exc:
        failures[name] = repr(exc)
        gc.collect()

if not family_valid:
    raise RuntimeError("All two-tower families failed: " + repr(failures))

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

family_valid_rank = {
    name: rank_percentile(valid.user_id, score)
    for name, score in family_valid.items()
}
family_test_rank = {
    name: rank_percentile(test.user_id, score)
    for name, score in family_test.items()
}

candidate_valid = {"incumbent": inc_valid}
candidate_test = {"incumbent": inc_test}
candidate_raw = {"incumbent": inc_valid}

for name in family_valid:
    candidate_valid[name + "_standalone"] = family_valid[name]
    candidate_test[name + "_standalone"] = family_test[name]
    candidate_raw[name + "_standalone"] = family_valid[name]

    for alpha in (0.10, 0.20, 0.30, 0.40, 0.55, 0.70):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * family_valid_rank[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * family_test_rank[name]
        )
        candidate_raw[key] = family_valid[name]

if len(family_valid) >= 2:
    names = sorted(family_valid)
    ensemble_valid = np.mean(
        np.stack([family_valid_rank[name] for name in names]),
        axis=0,
    )
    ensemble_test = np.mean(
        np.stack([family_test_rank[name] for name in names]),
        axis=0,
    )

    candidate_valid["two_tower_family_ensemble"] = ensemble_valid
    candidate_test["two_tower_family_ensemble"] = ensemble_test
    candidate_raw["two_tower_family_ensemble"] = ensemble_valid

    for alpha in (0.15, 0.25, 0.35, 0.50, 0.65):
        key = f"two_tower_ensemble_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * ensemble_valid
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * ensemble_test
        )
        candidate_raw[key] = ensemble_valid

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, score)
    for name, score in candidate_valid.items()
}

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]

inc_correlations = {
    name: float(np.corrcoef(
        inc_valid_rank,
        family_valid_rank[name],
    )[0, 1])
    for name in family_valid_rank
}

pairwise_correlations = {}
family_names = sorted(family_valid_rank)
for i in range(len(family_names)):
    for j in range(i + 1, len(family_names)):
        left = family_names[i]
        right = family_names[j]
        pairwise_correlations[left + "__" + right] = float(np.corrcoef(
            family_valid_rank[left],
            family_valid_rank[right],
        )[0, 1])

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "training": training_findings,
    "failures": failures,
    "rank_correlations_with_incumbent": inc_correlations,
    "pairwise_family_rank_correlations": pairwise_correlations,
    "recency_weight_min": float(recency_weights.min()),
    "recency_weight_max": float(recency_weights.max()),
    "recency_weight_mean": float(recency_weights.mean()),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_name != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[best_name], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))