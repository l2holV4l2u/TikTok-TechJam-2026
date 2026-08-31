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
SEED = 84173
MAX_LEN = 32
BATCH_SIZE = 256
INFER_BATCH_SIZE = 512
EPOCHS = 2
EMBED_DIM = 8
MODEL_DIM = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

DEVICE = torch.device("cpu")

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "hour",
    "upload_type",
    "music_type",
    "video_type",
    "user_active_degree",
    "is_video_author",
    "is_live_streamer",
    "onehot_feat3",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


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

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    first = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
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


def fit_numeric_normalizer(train):
    state = {}
    for field in NUM_FIELDS:
        value = np.asarray(train.num[field], dtype=np.float64)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        value = np.sign(value) * np.log1p(np.abs(value))
        mean = float(value.mean())
        std = float(value.std())
        state[field] = (mean, max(std, 1e-5))
    return state


def numeric_matrix(split, normalizer):
    result = np.empty((len(split), len(NUM_FIELDS)), dtype=np.float32)
    for j, field in enumerate(NUM_FIELDS):
        value = np.asarray(split.num[field], dtype=np.float64)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        value = np.sign(value) * np.log1p(np.abs(value))
        mean, std = normalizer[field]
        result[:, j] = np.clip((value - mean) / std, -8.0, 8.0)
    return result


def make_padded(split, normalizer, include_labels):
    n = len(split)
    row = np.arange(n, dtype=np.int64)
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)

    order = np.lexsort((row, times, users))
    ordered_users = users[order]

    user_start = np.r_[
        0,
        np.flatnonzero(ordered_users[1:] != ordered_users[:-1]) + 1,
    ]
    user_end = np.r_[user_start[1:], n]

    segment_starts = []
    segment_lengths = []
    for start, end in zip(user_start.tolist(), user_end.tolist()):
        positions = np.arange(start, end, MAX_LEN, dtype=np.int64)
        segment_starts.extend(positions.tolist())
        segment_lengths.extend(
            np.minimum(MAX_LEN, end - positions).tolist()
        )

    segment_starts = np.asarray(segment_starts, dtype=np.int64)
    segment_lengths = np.asarray(segment_lengths, dtype=np.int32)
    n_segments = len(segment_starts)

    cats = np.zeros(
        (n_segments, MAX_LEN, len(CAT_FIELDS)),
        dtype=np.int32,
    )
    nums = np.zeros(
        (n_segments, MAX_LEN, len(NUM_FIELDS)),
        dtype=np.float32,
    )
    mask = np.zeros((n_segments, MAX_LEN), dtype=np.bool_)
    rows = np.full(
        (n_segments, MAX_LEN),
        -1,
        dtype=np.int64,
    )

    source_cats = np.column_stack([
        np.asarray(split.X[field], dtype=np.int32)
        for field in CAT_FIELDS
    ])
    source_nums = numeric_matrix(split, normalizer)

    labels = None
    weights = None
    if include_labels:
        labels = np.zeros((n_segments, MAX_LEN), dtype=np.float32)
        weights = np.zeros((n_segments, MAX_LEN), dtype=np.float32)
        source_labels = np.asarray(split.y, dtype=np.float32)

        dates = np.asarray(split.date, dtype=np.int32)
        age = np.maximum(int(dates.max()) - dates, 0).astype(np.float32)
        source_weights = np.power(0.5, age / 4.0).astype(np.float32)
        source_weights /= max(float(source_weights.mean()), 1e-6)

    for i in range(n_segments):
        length = int(segment_lengths[i])
        ordered_slice = order[
            segment_starts[i]:segment_starts[i] + length
        ]
        cats[i, :length] = source_cats[ordered_slice]
        nums[i, :length] = source_nums[ordered_slice]
        mask[i, :length] = True
        rows[i, :length] = ordered_slice

        if include_labels:
            labels[i, :length] = source_labels[ordered_slice]
            weights[i, :length] = source_weights[ordered_slice]

    return {
        "cats": cats,
        "nums": nums,
        "mask": mask,
        "rows": rows,
        "labels": labels,
        "weights": weights,
        "n_rows": n,
        "n_segments": n_segments,
        "mean_segment_length": float(segment_lengths.mean()),
    }


class TokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[field]),
                EMBED_DIM,
                padding_idx=0,
            )
            for field in CAT_FIELDS
        ])
        input_dim = len(CAT_FIELDS) * EMBED_DIM + len(NUM_FIELDS)
        self.project = nn.Sequential(
            nn.Linear(input_dim, MODEL_DIM),
            nn.LayerNorm(MODEL_DIM),
            nn.SiLU(),
            nn.Dropout(0.08),
        )
        self.position = nn.Embedding(MAX_LEN, MODEL_DIM)

    def forward(self, cats, nums):
        embedded = [
            embedding(cats[:, :, j])
            for j, embedding in enumerate(self.embeddings)
        ]
        x = torch.cat(embedded + [nums], dim=-1)
        x = self.project(x)

        position = torch.arange(
            x.shape[1], device=x.device, dtype=torch.long
        )
        return x + self.position(position)[None, :, :]


class ContextConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TokenEncoder()
        self.blocks = nn.ModuleList()
        for dilation in (1, 2, 4):
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(MODEL_DIM),
                "conv": nn.Conv1d(
                    MODEL_DIM,
                    MODEL_DIM,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                "gate": nn.Conv1d(
                    MODEL_DIM,
                    MODEL_DIM,
                    kernel_size=1,
                ),
            }))
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_DIM),
            nn.Linear(MODEL_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, cats, nums, mask):
        x = self.encoder(cats, nums)
        active = mask.unsqueeze(-1).to(x.dtype)
        x = x * active

        for block in self.blocks:
            z = block["norm"](x).transpose(1, 2)
            value = torch.tanh(block["conv"](z))
            gate = torch.sigmoid(block["gate"](z))
            x = x + (value * gate).transpose(1, 2)
            x = x * active

        return self.head(x).squeeze(-1)


class BidirectionalGRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TokenEncoder()
        self.gru = nn.GRU(
            MODEL_DIM,
            MODEL_DIM // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_DIM),
            nn.Linear(MODEL_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, cats, nums, mask):
        x = self.encoder(cats, nums)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        x, _ = self.gru(x)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.head(x).squeeze(-1)


class SlateTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TokenEncoder()
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=4,
            dim_feedforward=160,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_DIM),
            nn.Linear(MODEL_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, cats, nums, mask):
        x = self.encoder(cats, nums)
        x = self.transformer(
            x,
            src_key_padding_mask=~mask,
        )
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.head(x).squeeze(-1)


def train_model(model, data, seed_offset):
    torch.manual_seed(SEED + seed_offset)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=2e-5,
    )

    n_segments = data["n_segments"]
    rng = np.random.default_rng(SEED + seed_offset)

    model.train()
    epoch_losses = []
    for epoch in range(EPOCHS):
        permutation = rng.permutation(n_segments)
        loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, n_segments, BATCH_SIZE):
            index = permutation[start:start + BATCH_SIZE]

            cats = torch.from_numpy(data["cats"][index]).long()
            nums = torch.from_numpy(data["nums"][index])
            mask = torch.from_numpy(data["mask"][index])
            labels = torch.from_numpy(data["labels"][index])
            weights = torch.from_numpy(data["weights"][index])

            logits = model(cats, nums, mask)
            element_loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )

            effective_weight = weights * mask.to(weights.dtype)
            denominator = effective_weight.sum().clamp_min(1.0)
            loss = (element_loss * effective_weight).sum() / denominator

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float(
                (element_loss * effective_weight).sum().detach()
            )
            weight_sum += float(denominator.detach())

        epoch_losses.append(loss_sum / max(weight_sum, 1.0))

    return epoch_losses


@torch.no_grad()
def predict_model(model, data):
    model.eval()
    result = np.empty(data["n_rows"], dtype=np.float64)
    n_segments = data["n_segments"]

    for start in range(0, n_segments, INFER_BATCH_SIZE):
        end = min(start + INFER_BATCH_SIZE, n_segments)
        cats = torch.from_numpy(data["cats"][start:end]).long()
        nums = torch.from_numpy(data["nums"][start:end])
        mask = torch.from_numpy(data["mask"][start:end])

        logits = model(cats, nums, mask).cpu().numpy()
        rows = data["rows"][start:end]
        active = data["mask"][start:end]
        result[rows[active]] = logits[active]

    return result


train = load("train")
valid = load("valid")
test = load("test")

normalizer = fit_numeric_normalizer(train)
train_data = make_padded(train, normalizer, include_labels=True)
valid_data = make_padded(valid, normalizer, include_labels=False)
test_data = make_padded(test, normalizer, include_labels=False)

family_constructors = [
    ("context_conv", ContextConvModel),
    ("bidirectional_gru", BidirectionalGRUModel),
    ("slate_transformer", SlateTransformerModel),
]

family_valid = {}
family_test = {}
training_losses = {}
model_failures = {}

for model_index, (name, constructor) in enumerate(family_constructors):
    try:
        model = constructor()
        training_losses[name] = train_model(
            model,
            train_data,
            seed_offset=100 * (model_index + 1),
        )
        family_valid[name] = predict_model(model, valid_data)
        family_test[name] = predict_model(model, test_data)
        del model
        gc.collect()
    except Exception as exc:
        model_failures[name] = repr(exc)
        gc.collect()

if not family_valid:
    raise RuntimeError("All contextual model families failed: " + repr(model_failures))

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
    name: rank_percentile(valid.user_id, scores)
    for name, scores in family_valid.items()
}
family_test_rank = {
    name: rank_percentile(test.user_id, scores)
    for name, scores in family_test.items()
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

# A rank-average across structurally different contextual mechanisms is also
# tested because convolution, recurrence, and attention impose different
# locality and order biases.
if len(family_valid_rank) >= 2:
    available_names = sorted(family_valid_rank)
    contextual_ensemble_valid = np.mean(
        np.stack([family_valid_rank[n] for n in available_names]),
        axis=0,
    )
    contextual_ensemble_test = np.mean(
        np.stack([family_test_rank[n] for n in available_names]),
        axis=0,
    )

    candidate_valid["contextual_family_ensemble"] = contextual_ensemble_valid
    candidate_test["contextual_family_ensemble"] = contextual_ensemble_test
    candidate_raw["contextual_family_ensemble"] = contextual_ensemble_valid

    for alpha in (0.20, 0.35, 0.50, 0.65):
        key = f"contextual_ensemble_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * contextual_ensemble_valid
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * contextual_ensemble_test
        )
        candidate_raw[key] = contextual_ensemble_valid

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_valid.items()
}

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]

correlations = {}
for name, scores in family_valid_rank.items():
    correlations[name] = float(np.corrcoef(
        inc_valid_rank,
        scores,
    )[0, 1])

family_names = sorted(family_valid_rank)
pairwise_correlations = {}
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
    "training_losses": training_losses,
    "model_failures": model_failures,
    "rank_correlations_with_incumbent": correlations,
    "pairwise_contextual_rank_correlations": pairwise_correlations,
    "train_segments": int(train_data["n_segments"]),
    "valid_segments": int(valid_data["n_segments"]),
    "test_segments": int(test_data["n_segments"]),
    "train_mean_segment_length": train_data["mean_segment_length"],
    "valid_mean_segment_length": valid_data["mean_segment_length"],
    "test_mean_segment_length": test_data["mean_segment_length"],
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