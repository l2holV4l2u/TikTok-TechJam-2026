import os
import time
import json
import gc
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_start_time = time.time()

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 20260828
EPOCHS = 6
BATCH_SIZE = 32768
LEARNING_RATE = 0.035
HASH_SIZE = 1 << 20
HASH_MASK = HASH_SIZE - 1

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

DEVICE = torch.device("cpu")

BASE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "hour",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row_id = np.arange(n, dtype=np.int64)

    order = np.lexsort((row_id, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    sorted_ranks = positions / denominators

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def standardize(reference, values):
    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(reference))
    std = max(float(np.std(reference)), 1e-8)
    return (values - mean) / std


class SparseWideCrossModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_offsets = {}
        running = 0
        for name in BASE_FIELDS:
            self.base_offsets[name] = running
            running += int(FEATURE_CARDINALITIES[name])
        self.base_size = running

        user_card = int(FEATURE_CARDINALITIES["user_id"])
        tag_card = int(FEATURE_CARDINALITIES["tag"])
        tab_card = int(FEATURE_CARDINALITIES["tab"])
        duration_card = int(FEATURE_CARDINALITIES["duration_bucket"])
        hour_card = int(FEATURE_CARDINALITIES["hour"])

        self.tag_card = tag_card
        self.tab_card = tab_card
        self.duration_card = duration_card
        self.hour_card = hour_card

        self.base = nn.Embedding(self.base_size, 1, sparse=True)
        self.user_video = nn.Embedding(HASH_SIZE, 1, sparse=True)
        self.user_author = nn.Embedding(HASH_SIZE, 1, sparse=True)
        self.user_tag = nn.Embedding(
            user_card * tag_card, 1, sparse=True
        )
        self.user_tab = nn.Embedding(
            user_card * tab_card, 1, sparse=True
        )
        self.user_duration = nn.Embedding(
            user_card * duration_card, 1, sparse=True
        )
        self.user_hour = nn.Embedding(
            user_card * hour_card, 1, sparse=True
        )
        self.video_tab = nn.Embedding(HASH_SIZE, 1, sparse=True)
        self.author_tag = nn.Embedding(HASH_SIZE, 1, sparse=True)
        self.bias = nn.Embedding(1, 1, sparse=True)

        for parameter in self.parameters():
            nn.init.zeros_(parameter)

    @staticmethod
    def hashed_pair(a, b, salt):
        h = (
            a * (1000003 + 97 * salt)
            + b * (9176 + 193 * salt)
            + (0x9E3779B1 + 104729 * salt)
        ) & HASH_MASK

        sign_bit = (
            (a * (2147483647 - 131 * salt))
            ^ (b * (2654435761 + 17 * salt))
            ^ (97531 * salt)
        ) & 1
        sign = sign_bit.to(torch.float32).mul_(2.0).sub_(1.0)
        return h, sign

    def forward(self, batch):
        n = batch["user_id"].shape[0]
        score = self.bias(
            torch.zeros(n, dtype=torch.long, device=DEVICE)
        ).squeeze(1)

        for name in BASE_FIELDS:
            index = batch[name] + self.base_offsets[name]
            score = score + self.base(index).squeeze(1)

        user = batch["user_id"]
        video = batch["video_id"]
        author = batch["author_id"]
        tag = batch["tag"]
        tab = batch["tab"]
        duration = batch["duration_bucket"]
        hour = batch["hour"]

        index, sign = self.hashed_pair(user, video, 1)
        score = score + sign * self.user_video(index).squeeze(1)

        index, sign = self.hashed_pair(user, author, 2)
        score = score + sign * self.user_author(index).squeeze(1)

        score = score + self.user_tag(
            user * self.tag_card + tag
        ).squeeze(1)
        score = score + self.user_tab(
            user * self.tab_card + tab
        ).squeeze(1)
        score = score + self.user_duration(
            user * self.duration_card + duration
        ).squeeze(1)
        score = score + self.user_hour(
            user * self.hour_card + hour
        ).squeeze(1)

        index, sign = self.hashed_pair(video, tab, 3)
        score = score + sign * self.video_tab(index).squeeze(1)

        index, sign = self.hashed_pair(author, tag, 4)
        score = score + sign * self.author_tag(index).squeeze(1)

        return score


def tensorize(split):
    result = {}
    for name in BASE_FIELDS:
        result[name] = torch.from_numpy(
            np.asarray(split.X[name], dtype=np.int64)
        )
    return result


@torch.no_grad()
def predict_model(model, tensors, batch_size=65536):
    model.eval()
    n = len(tensors["user_id"])
    output = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = {
            name: value[start:end]
            for name, value in tensors.items()
        }
        output[start:end] = (
            model(batch).detach().cpu().numpy().astype(np.float64)
        )
    return output


def evaluate_blends(
    user_ids,
    labels,
    wide_scores,
    incumbent_scores,
    candidate_prefix,
):
    candidates = []

    wide_metrics = evaluate(user_ids, labels, wide_scores)
    candidates.append(
        (
            float(wide_metrics["primary"]),
            candidate_prefix + "_wide",
            "wide",
            1.0,
            wide_scores.copy(),
            wide_metrics,
        )
    )

    wide_z = standardize(wide_scores, wide_scores)
    incumbent_z = standardize(incumbent_scores, incumbent_scores)

    wide_rank = within_user_rank(user_ids, wide_scores)
    incumbent_rank = within_user_rank(user_ids, incumbent_scores)

    for alpha in np.arange(0.05, 0.651, 0.05):
        alpha = float(alpha)

        z_scores = alpha * wide_z + (1.0 - alpha) * incumbent_z
        z_metrics = evaluate(user_ids, labels, z_scores)
        candidates.append(
            (
                float(z_metrics["primary"]),
                "%s_z_%.2f" % (candidate_prefix, alpha),
                "zblend",
                alpha,
                z_scores,
                z_metrics,
            )
        )

        rank_scores = (
            alpha * wide_rank + (1.0 - alpha) * incumbent_rank
        )
        rank_metrics = evaluate(user_ids, labels, rank_scores)
        candidates.append(
            (
                float(rank_metrics["primary"]),
                "%s_rank_%.2f" % (candidate_prefix, alpha),
                "rankblend",
                alpha,
                rank_scores,
                rank_metrics,
            )
        )

    return candidates


artifacts = os.environ.get("RUN_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

if not (
    os.path.isfile(incumbent_valid_path)
    and os.path.isfile(incumbent_test_path)
):
    raise FileNotFoundError(
        "Trusted incumbent validation/test predictions are required"
    )

train = load("train")
valid = load("valid")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.y):
    raise ValueError("Incumbent validation prediction length mismatch")

incumbent_metrics = evaluate(
    valid.user_id, valid.y, incumbent_valid
)

train_tensors = tensorize(train)
valid_tensors = tensorize(valid)
train_labels = torch.from_numpy(
    np.asarray(train.y, dtype=np.float32)
)

# Mild temporal adaptation, while keeping every training impression.
train_dates = np.asarray(train.date, dtype=np.int32)
days_old = np.maximum(
    int(train_dates.max()) - train_dates, 0
).astype(np.float32)
train_weights = torch.from_numpy(
    np.exp(-0.025 * days_old).astype(np.float32)
)

model = SparseWideCrossModel().to(DEVICE)
optimizer = torch.optim.SparseAdam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.995),
    eps=1e-8,
)

all_candidates = [
    (
        float(incumbent_metrics["primary"]),
        "incumbent",
        "incumbent",
        0.0,
        incumbent_valid.copy(),
        incumbent_metrics,
        0,
    )
]
candidate_log = {
    "incumbent": float(incumbent_metrics["primary"])
}

best_primary = float(incumbent_metrics["primary"])
best_epoch = 0
best_kind = "incumbent"
best_alpha = 0.0
best_valid_scores = incumbent_valid.copy()
best_metrics = incumbent_metrics

checkpoint_dir = os.environ.get("ITER_OUT") or tempfile.gettempdir()
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(
    checkpoint_dir, "wide_cross_best_state.pt"
)

n_train = len(train_labels)

for epoch in range(1, EPOCHS + 1):
    model.train()
    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)
    permutation = torch.randperm(n_train, generator=generator)

    epoch_loss_sum = 0.0
    epoch_weight_sum = 0.0

    for start in range(0, n_train, BATCH_SIZE):
        batch_index = permutation[start:start + BATCH_SIZE]
        batch = {
            name: values[batch_index]
            for name, values in train_tensors.items()
        }
        labels = train_labels[batch_index]
        weights = train_weights[batch_index]

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        point_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        loss = torch.sum(point_loss * weights) / torch.sum(weights)
        loss.backward()
        optimizer.step()

        epoch_loss_sum += float(
            torch.sum(point_loss.detach() * weights).item()
        )
        epoch_weight_sum += float(torch.sum(weights).item())

    wide_valid = predict_model(model, valid_tensors)
    epoch_candidates = evaluate_blends(
        valid.user_id,
        valid.y,
        wide_valid,
        incumbent_valid,
        "epoch%d" % epoch,
    )

    epoch_candidates.sort(key=lambda x: x[0], reverse=True)
    epoch_best = epoch_candidates[0]
    epoch_loss = epoch_loss_sum / max(epoch_weight_sum, 1e-12)

    for item in epoch_candidates:
        primary, name, kind, alpha, scores, metrics = item
        candidate_log[name] = primary
        all_candidates.append(
            (
                primary,
                name,
                kind,
                alpha,
                scores,
                metrics,
                epoch,
            )
        )

    print(
        "FINDINGS epoch=%d weighted_logloss=%.6f wide_primary=%.6f "
        "best_primary=%.6f best_kind=%s alpha=%.2f"
        % (
            epoch,
            epoch_loss,
            float(
                next(
                    x[0]
                    for x in epoch_candidates
                    if x[2] == "wide"
                )
            ),
            float(epoch_best[0]),
            epoch_best[2],
            float(epoch_best[3]),
        )
    )

    if float(epoch_best[0]) > best_primary:
        best_primary = float(epoch_best[0])
        best_epoch = epoch
        best_kind = epoch_best[2]
        best_alpha = float(epoch_best[3])
        best_valid_scores = np.asarray(
            epoch_best[4], dtype=np.float64
        ).copy()
        best_metrics = epoch_best[5]
        torch.save(model.state_dict(), checkpoint_path)

all_candidates.sort(key=lambda x: x[0], reverse=True)
global_best = all_candidates[0]

best_primary = float(global_best[0])
best_name = global_best[1]
best_kind = global_best[2]
best_alpha = float(global_best[3])
best_valid_scores = np.asarray(global_best[4], dtype=np.float64).copy()
best_metrics = global_best[5]
best_epoch = int(global_best[6])

top_candidates = sorted(
    candidate_log.items(), key=lambda item: item[1], reverse=True
)[:15]
print(
    "CANDIDATES "
    + json.dumps(
        {name: score for name, score in top_candidates},
        separators=(", ", ": "),
    )
)
print(
    "FINDINGS selected=%s epoch=%d kind=%s alpha=%.2f "
    "incumbent=%.6f selected_primary=%.6f"
    % (
        best_name,
        best_epoch,
        best_kind,
        best_alpha,
        float(incumbent_metrics["primary"]),
        best_primary,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        best_valid_scores,
    )

# Validation selection is complete. Test labels are never accessed.
del permutation
gc.collect()

test = load("test")
incumbent_test = np.asarray(
    np.load(incumbent_test_path), dtype=np.float64
)
if len(incumbent_test) != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

if best_kind == "incumbent":
    test_scores = incumbent_test.copy()
else:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Selected wide model checkpoint was not written"
        )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )
    test_tensors = tensorize(test)
    wide_test = predict_model(model, test_tensors)

    # Recover the selected epoch's validation-wide prediction so the test
    # normalization is anchored to exactly the selected validation model.
    selected_epoch_entry = None
    for entry in all_candidates:
        if (
            int(entry[6]) == best_epoch
            and entry[2] == "wide"
        ):
            selected_epoch_entry = entry
            break
    if selected_epoch_entry is None:
        raise RuntimeError("Selected epoch wide prediction not found")

    selected_wide_valid = np.asarray(
        selected_epoch_entry[4], dtype=np.float64
    )

    if best_kind == "wide":
        test_scores = wide_test
    elif best_kind == "zblend":
        wide_test_z = standardize(
            selected_wide_valid, wide_test
        )
        incumbent_test_z = standardize(
            incumbent_valid, incumbent_test
        )
        test_scores = (
            best_alpha * wide_test_z
            + (1.0 - best_alpha) * incumbent_test_z
        )
    elif best_kind == "rankblend":
        wide_test_rank = within_user_rank(
            test.user_id, wide_test
        )
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        test_scores = (
            best_alpha * wide_test_rank
            + (1.0 - best_alpha) * incumbent_test_rank
        )
    else:
        raise RuntimeError("Unknown selected prediction kind")

test_scores = np.asarray(test_scores, dtype=np.float64)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        test_scores,
    )

elapsed = time.time() - _start_time
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))