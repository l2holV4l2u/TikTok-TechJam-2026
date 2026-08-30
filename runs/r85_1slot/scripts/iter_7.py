import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 46291
THREADS = min(8, os.cpu_count() or 1)
DEVICE = torch.device("cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "hour",
    "user_active_degree",
]

NUM_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]

AUXILIARY_CANDIDATES = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
]

EMBED_DIM = 8
HIDDEN_DIM = 64
BATCH_SIZE = 8192
TRAIN_EPOCHS = 2
PRED_BATCH = 32768


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]
    starts_mask = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.flatnonzero(starts_mask)
    group_id = np.cumsum(starts_mask) - 1
    group_starts = starts[group_id]
    positions = np.arange(n, dtype=np.int64) - group_starts
    group_sizes = np.diff(np.r_[starts, n])
    denominators = np.maximum(group_sizes[group_id] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denominators
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def make_cats(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64) for name in CAT_FIELDS
    ])


def raw_nums(split):
    columns = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)))
    return np.column_stack(columns).astype(np.float32, copy=False)


def fit_num_scaler(nums):
    mean = np.mean(nums, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(nums, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-3)
    return mean, std


def scale_nums(nums, mean, std):
    return np.clip((nums - mean) / std, -6.0, 6.0).astype(
        np.float32, copy=False
    )


def multitask_targets(split, auxiliary_names):
    targets = [np.asarray(split.y, dtype=np.float32)]
    for name in auxiliary_names:
        targets.append(np.asarray(split.aux[name], dtype=np.float32))
    return np.column_stack(targets).astype(np.float32, copy=False)


def choose_auxiliary_names(train):
    result = []
    n = len(train.user_id)
    for name in AUXILIARY_CANDIDATES:
        if name not in train.aux:
            continue
        value = np.asarray(train.aux[name])
        if value.shape != (n,):
            continue
        finite = np.isfinite(value)
        if not np.all(finite):
            continue
        unique = np.unique(value)
        if unique.size <= 2 and np.all((unique == 0) | (unique == 1)):
            result.append(name)
    return result[:5]


def causal_last_positive_context(split, labels):
    """
    For every training row, return the video/author/tag from the most recent
    strictly preceding positive impression for the same user.

    The implementation is vectorized after the chronological sort. The large
    per-group offset makes np.maximum.accumulate reset at each user boundary.
    """
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    n = users.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    sy = labels[order]

    starts_mask = np.r_[True, su[1:] != su[:-1]]
    starts = np.flatnonzero(starts_mask)
    group_id = np.cumsum(starts_mask) - 1
    group_start = starts[group_id]
    local_position = np.arange(n, dtype=np.int64) - group_start

    stride = np.int64(n + 1)
    base = group_id.astype(np.int64) * stride
    marker = base + np.where(sy == 1, local_position + 1, 0)
    inclusive_last = np.maximum.accumulate(marker) - base

    prior_last = np.empty(n, dtype=np.int64)
    prior_last[0] = 0
    prior_last[1:] = inclusive_last[:-1]
    prior_last[starts] = 0

    source_sorted = group_start + np.maximum(prior_last - 1, 0)
    has_history = prior_last > 0

    context_sorted = np.zeros((n, 3), dtype=np.int64)
    source_rows = order[source_sorted]

    video = np.asarray(split.video_id, dtype=np.int64)
    author = np.asarray(split.X["author_id"], dtype=np.int64)
    tag = np.asarray(split.X["tag"], dtype=np.int64)

    context_sorted[has_history, 0] = video[source_rows[has_history]]
    context_sorted[has_history, 1] = author[source_rows[has_history]]
    context_sorted[has_history, 2] = tag[source_rows[has_history]]

    context = np.empty_like(context_sorted)
    context[order] = context_sorted

    # State after the complete split, used for the following date window.
    user_cardinality = int(FEATURE_CARDINALITIES["user_id"])
    final_context = np.zeros((user_cardinality, 3), dtype=np.int64)
    ends = np.r_[starts[1:] - 1, n - 1]
    final_rel = inclusive_last[ends]
    final_users = su[ends]
    valid_final = final_rel > 0
    final_sources = starts[valid_final] + final_rel[valid_final] - 1
    final_source_rows = order[final_sources]

    final_context[final_users[valid_final], 0] = video[final_source_rows]
    final_context[final_users[valid_final], 1] = author[final_source_rows]
    final_context[final_users[valid_final], 2] = tag[final_source_rows]

    return context, final_context


def static_context(split, final_state):
    users = np.asarray(split.user_id, dtype=np.int64)
    result = np.zeros((users.size, 3), dtype=np.int64)
    valid = (users >= 0) & (users < final_state.shape[0])
    result[valid] = final_state[users[valid]]
    return result


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[name]), EMBED_DIM)
            for name in CAT_FIELDS
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.025)
            with torch.no_grad():
                emb.weight[0].zero_()

    def forward(self, cats, nums):
        embedded = [
            emb(cats[:, index])
            for index, emb in enumerate(self.embeddings)
        ]
        return torch.cat(embedded + [nums], dim=1)


class MMoEModel(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.encoder = FeatureEncoder()
        input_dim = len(CAT_FIELDS) * EMBED_DIM + len(NUM_FIELDS)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(0.08),
                nn.Linear(HIDDEN_DIM, 32),
                nn.ReLU(),
            )
            for _ in range(3)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, len(self.experts))
            for _ in range(num_tasks)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(num_tasks)
        ])

    def forward(self, cats, nums, context=None):
        x = self.encoder(cats, nums)
        expert_values = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )

        outputs = []
        for gate, head in zip(self.gates, self.heads):
            weights = torch.softmax(gate(x), dim=1).unsqueeze(2)
            task_representation = torch.sum(
                expert_values * weights, dim=1
            )
            outputs.append(head(task_representation).squeeze(1))
        return torch.stack(outputs, dim=1)


class LastPositiveDIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()

        self.history_video = nn.Embedding(
            int(FEATURE_CARDINALITIES["video_id"]), EMBED_DIM
        )
        self.history_author = nn.Embedding(
            int(FEATURE_CARDINALITIES["author_id"]), EMBED_DIM
        )
        self.history_tag = nn.Embedding(
            int(FEATURE_CARDINALITIES["tag"]), EMBED_DIM
        )

        for emb in (
            self.history_video,
            self.history_author,
            self.history_tag,
        ):
            nn.init.normal_(emb.weight, std=0.025)
            with torch.no_grad():
                emb.weight[0].zero_()

        # This explicit calculation fixes the previous attempt's predictor
        # dimension mismatch.
        base_dim = len(CAT_FIELDS) * EMBED_DIM + len(NUM_FIELDS)
        history_dim = 3 * EMBED_DIM
        match_dim = 3
        predictor_input_dim = base_dim + history_dim + match_dim

        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.video_index = CAT_FIELDS.index("video_id")
        self.author_index = CAT_FIELDS.index("author_id")
        self.tag_index = CAT_FIELDS.index("tag")

    def forward(self, cats, nums, context):
        base = self.encoder(cats, nums)

        history = torch.cat([
            self.history_video(context[:, 0]),
            self.history_author(context[:, 1]),
            self.history_tag(context[:, 2]),
        ], dim=1)

        video_match = (
            (context[:, 0] != 0)
            & (context[:, 0] == cats[:, self.video_index])
        ).float()
        author_match = (
            (context[:, 1] != 0)
            & (context[:, 1] == cats[:, self.author_index])
        ).float()
        tag_match = (
            (context[:, 2] != 0)
            & (context[:, 2] == cats[:, self.tag_index])
        ).float()
        matches = torch.stack(
            [video_match, author_match, tag_match], dim=1
        )

        combined = torch.cat([base, history, matches], dim=1)
        return self.predictor(combined).squeeze(1)


def train_model(model, cats, nums, targets, context, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-6
    )
    n = cats.shape[0]

    for _ in range(TRAIN_EPOCHS):
        permutation = rng.permutation(n)
        model.train()

        for begin in range(0, n, BATCH_SIZE):
            idx = permutation[begin:begin + BATCH_SIZE]
            tc = torch.from_numpy(cats[idx]).to(DEVICE)
            tn = torch.from_numpy(nums[idx]).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if isinstance(model, MMoEModel):
                logits = model(tc, tn)
                target = torch.from_numpy(targets[idx]).to(DEVICE)
                per_task = F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                ).mean(dim=0)
                # Keep the scored long_view task dominant while auxiliary
                # outcomes regularize the shared experts.
                if per_task.numel() > 1:
                    loss = per_task[0] + 0.20 * per_task[1:].mean()
                else:
                    loss = per_task[0]
            else:
                tx = torch.from_numpy(context[idx]).to(DEVICE)
                logits = model(tc, tn, tx)
                target = torch.from_numpy(
                    targets[idx].reshape(-1)
                ).to(DEVICE)
                loss = F.binary_cross_entropy_with_logits(logits, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, cats, nums, context=None):
    model.eval()
    result = np.empty(cats.shape[0], dtype=np.float32)

    for begin in range(0, cats.shape[0], PRED_BATCH):
        end = min(begin + PRED_BATCH, cats.shape[0])
        tc = torch.from_numpy(cats[begin:end]).to(DEVICE)
        tn = torch.from_numpy(nums[begin:end]).to(DEVICE)

        if isinstance(model, MMoEModel):
            logits = model(tc, tn)[:, 0]
        else:
            tx = torch.from_numpy(context[begin:end]).to(DEVICE)
            logits = model(tc, tn, tx)

        result[begin:end] = logits.cpu().numpy()

    return result


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_cats = make_cats(train)
valid_cats = make_cats(valid)

train_nums_raw = raw_nums(train)
valid_nums_raw = raw_nums(valid)
num_mean, num_std = fit_num_scaler(train_nums_raw)
train_nums = scale_nums(train_nums_raw, num_mean, num_std)
valid_nums = scale_nums(valid_nums_raw, num_mean, num_std)

auxiliary_names = choose_auxiliary_names(train)
mmoe_train_targets = multitask_targets(train, auxiliary_names)

train_context, train_final_state = causal_last_positive_context(
    train, y_train
)
valid_context = static_context(valid, train_final_state)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.size != y_valid.size:
    raise ValueError("Incumbent validation prediction length mismatch")
inc_valid_rank = within_user_rank(valid_users, inc_valid)

models = {}
raw_predictions = {}

mmoe = MMoEModel(num_tasks=1 + len(auxiliary_names))
mmoe = train_model(
    mmoe,
    train_cats,
    train_nums,
    mmoe_train_targets,
    None,
    SEED + 10,
)
raw_predictions["mmoe"] = predict_model(
    mmoe, valid_cats, valid_nums
).astype(np.float64)
models["mmoe"] = mmoe

din = LastPositiveDIN()
din = train_model(
    din,
    train_cats,
    train_nums,
    y_train.astype(np.float32),
    train_context,
    SEED + 20,
)
raw_predictions["last_positive_din"] = predict_model(
    din, valid_cats, valid_nums, valid_context
).astype(np.float64)
models["last_positive_din"] = din

candidate_scores = {}
selected_family = None
selected_weight = None
selected_scores = None
selected_raw = None
selected_metrics = None

for family, raw_scores in raw_predictions.items():
    raw_metrics = evaluate(valid_users, y_valid, raw_scores)
    candidate_scores[family + "_raw"] = float(raw_metrics["primary"])

    own_rank = within_user_rank(valid_users, raw_scores)
    local_scores = raw_scores
    local_metrics = raw_metrics
    local_weight = 1.0

    for weight in np.linspace(0.0, 1.0, 21):
        blended = weight * own_rank + (1.0 - weight) * inc_valid_rank
        metrics = evaluate(valid_users, y_valid, blended)
        if float(metrics["primary"]) > float(local_metrics["primary"]):
            local_metrics = metrics
            local_scores = blended.copy()
            local_weight = float(weight)

    candidate_scores[family + "_best_blend"] = float(
        local_metrics["primary"]
    )

    if (
        selected_metrics is None
        or float(local_metrics["primary"])
        > float(selected_metrics["primary"])
    ):
        selected_family = family
        selected_weight = local_weight
        selected_scores = np.asarray(local_scores, dtype=np.float64)
        selected_raw = np.asarray(raw_scores, dtype=np.float64)
        selected_metrics = local_metrics

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": selected_family,
            "own_rank_weight": selected_weight,
            "auxiliary_tasks": auxiliary_names,
            "train_epochs": TRAIN_EPOCHS,
            "history_nonzero_train": float(
                np.mean(train_context[:, 0] != 0)
            ),
            "history_nonzero_valid": float(
                np.mean(valid_context[:, 0] != 0)
            ),
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        selected_scores.astype(np.float64),
    )
    if selected_weight < 1.0 - 1e-12:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            selected_raw.astype(np.float64),
        )

# Refit the selected recipe on train + validation and score test.
del mmoe, din
models.clear()
gc.collect()

test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)

refit_cats = np.concatenate([train_cats, valid_cats], axis=0)
test_cats = make_cats(test)

refit_nums_raw = np.concatenate(
    [train_nums_raw, valid_nums_raw], axis=0
)
test_nums_raw = raw_nums(test)
refit_mean, refit_std = fit_num_scaler(refit_nums_raw)
refit_nums = scale_nums(refit_nums_raw, refit_mean, refit_std)
test_nums = scale_nums(test_nums_raw, refit_mean, refit_std)

y_refit = np.concatenate([y_train, y_valid]).astype(np.int8)

if selected_family == "mmoe":
    valid_aux_targets = multitask_targets(valid, auxiliary_names)
    refit_targets = np.concatenate(
        [mmoe_train_targets, valid_aux_targets], axis=0
    )

    refit_model = MMoEModel(num_tasks=1 + len(auxiliary_names))
    refit_model = train_model(
        refit_model,
        refit_cats,
        refit_nums,
        refit_targets,
        None,
        SEED + 110,
    )
    own_test = predict_model(
        refit_model, test_cats, test_nums
    ).astype(np.float64)
else:
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.user_id = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    combined.time_ms = np.concatenate([
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(valid.time_ms, dtype=np.int64),
    ])
    combined.video_id = np.concatenate([
        np.asarray(train.video_id, dtype=np.int64),
        np.asarray(valid.video_id, dtype=np.int64),
    ])
    combined.X = {
        "author_id": np.concatenate([
            np.asarray(train.X["author_id"], dtype=np.int64),
            np.asarray(valid.X["author_id"], dtype=np.int64),
        ]),
        "tag": np.concatenate([
            np.asarray(train.X["tag"], dtype=np.int64),
            np.asarray(valid.X["tag"], dtype=np.int64),
        ]),
    }

    refit_context, refit_final_state = causal_last_positive_context(
        combined, y_refit
    )
    test_context = static_context(test, refit_final_state)

    refit_model = LastPositiveDIN()
    refit_model = train_model(
        refit_model,
        refit_cats,
        refit_nums,
        y_refit.astype(np.float32),
        refit_context,
        SEED + 120,
    )
    own_test = predict_model(
        refit_model, test_cats, test_nums, test_context
    ).astype(np.float64)

if selected_weight < 1.0 - 1e-12:
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    if inc_test.size != test_users.size:
        raise ValueError("Incumbent test prediction length mismatch")
    own_test_rank = within_user_rank(test_users, own_test)
    inc_test_rank = within_user_rank(test_users, inc_test)
    test_scores = (
        selected_weight * own_test_rank
        + (1.0 - selected_weight) * inc_test_rank
    )
else:
    test_scores = own_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(selected_metrics["primary"]),
            "gauc": float(selected_metrics["gauc"]),
            "ndcg@5": float(selected_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)