import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18427
HISTORY_LEN = 8
EMBED_DIM = 20
DIN_EPOCHS = 3
BATCH_SIZE = 8192
PRED_BATCH = 32768
LR = 0.003
SVD_DIM = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])
AUTHOR_CARD = int(FEATURE_CARDINALITIES["author_id"])
TAG_CARD = int(FEATURE_CARDINALITIES["tag"])
TAB_CARD = int(FEATURE_CARDINALITIES["tab"])
DURATION_CARD = int(FEATURE_CARDINALITIES["duration_bucket"])


def ordered_rows(split):
    n = len(split.user_id)
    return np.lexsort((
        np.arange(n, dtype=np.int64),
        np.asarray(split.time_ms, dtype=np.int64),
        np.asarray(split.user_id, dtype=np.int64)
    ))


def training_positive_histories(split, labels):
    """For every training row, return the preceding K positive videos."""
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)
    n = len(users)

    order = ordered_rows(split)
    sorted_users = users[order]
    sorted_y = y[order]

    boundaries = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    group = np.cumsum(boundaries, dtype=np.int64) - 1
    n_groups = int(group[-1]) + 1

    global_cumulative = np.cumsum(sorted_y, dtype=np.int64)
    starts = np.flatnonzero(boundaries)
    base = global_cumulative[starts] - sorted_y[starts]
    local_cumulative = global_cumulative - base[group]
    positives_before = local_cumulative - sorted_y

    positive_counts = np.bincount(
        group, weights=sorted_y, minlength=n_groups
    ).astype(np.int64)
    positive_starts = np.zeros(n_groups, dtype=np.int64)
    if n_groups > 1:
        positive_starts[1:] = np.cumsum(
            positive_counts[:-1], dtype=np.int64
        )

    positive_video_stream = videos[order[sorted_y == 1]]
    offsets = np.arange(HISTORY_LEN, dtype=np.int64)[None, :]
    locations = (
        positive_starts[group, None]
        + positives_before[:, None]
        - 1
        - offsets
    )
    valid = positives_before[:, None] > offsets
    safe_locations = np.maximum(locations, 0)

    histories_sorted = np.zeros(
        (n, HISTORY_LEN), dtype=np.int64
    )
    if len(positive_video_stream):
        histories_sorted[valid] = positive_video_stream[
            safe_locations[valid]
        ]

    histories = np.empty_like(histories_sorted)
    histories[order] = histories_sorted
    return histories


def final_positive_history_table(split, labels):
    """Last K positive videos per user after consuming an entire fit split."""
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)

    order = ordered_rows(split)
    positive_rows = order[y[order] == 1]
    positive_users = users[positive_rows]
    positive_videos = videos[positive_rows]

    counts = np.bincount(
        positive_users, minlength=USER_CARD
    ).astype(np.int64)
    ends = np.cumsum(counts, dtype=np.int64)

    table = np.zeros((USER_CARD, HISTORY_LEN), dtype=np.int64)
    all_users = np.arange(USER_CARD, dtype=np.int64)
    for k in range(HISTORY_LEN):
        valid = counts > k
        locations = ends - 1 - k
        table[all_users[valid], k] = positive_videos[locations[valid]]
    return table


def make_context(split):
    return (
        np.asarray(split.video_id, dtype=np.int64),
        np.asarray(split.X["author_id"], dtype=np.int64),
        np.asarray(split.X["tag"], dtype=np.int64),
        np.asarray(split.X["tab"], dtype=np.int64),
        np.asarray(split.X["duration_bucket"], dtype=np.int64)
    )


class DINHistoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_embedding = nn.Embedding(
            VIDEO_CARD, EMBED_DIM, padding_idx=0
        )
        self.author_linear = nn.Embedding(AUTHOR_CARD, 1)
        self.tag_linear = nn.Embedding(TAG_CARD, 1)
        self.tab_linear = nn.Embedding(TAB_CARD, 1)
        self.duration_linear = nn.Embedding(DURATION_CARD, 1)
        self.video_linear = nn.Embedding(VIDEO_CARD, 1)

        self.attention = nn.Sequential(
            nn.Linear(4 * EMBED_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.scorer = nn.Sequential(
            nn.Linear(3 * EMBED_DIM, 48),
            nn.ReLU(),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        nn.init.normal_(self.video_embedding.weight, std=0.03)
        with torch.no_grad():
            self.video_embedding.weight[0].zero_()
        for emb in (
            self.author_linear, self.tag_linear, self.tab_linear,
            self.duration_linear, self.video_linear
        ):
            nn.init.zeros_(emb.weight)

    def forward(self, video, author, tag, tab, duration, history):
        candidate = self.video_embedding(video)
        history_embedding = self.video_embedding(history)
        mask = history.ne(0)

        expanded_candidate = candidate[:, None, :].expand_as(
            history_embedding
        )
        attention_input = torch.cat([
            expanded_candidate,
            history_embedding,
            expanded_candidate - history_embedding,
            expanded_candidate * history_embedding
        ], dim=2)
        attention_logits = self.attention(
            attention_input
        ).squeeze(2)
        attention_logits = attention_logits.masked_fill(~mask, -1.0e4)

        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = attention_weights * mask.float()
        attention_weights = attention_weights / (
            attention_weights.sum(dim=1, keepdim=True) + 1.0e-8
        )
        history_vector = (
            attention_weights[:, :, None] * history_embedding
        ).sum(dim=1)

        deep_input = torch.cat([
            candidate,
            history_vector,
            candidate * history_vector
        ], dim=1)
        deep_score = self.scorer(deep_input).squeeze(1)

        linear_score = (
            self.video_linear(video).squeeze(1)
            + self.author_linear(author).squeeze(1)
            + self.tag_linear(tag).squeeze(1)
            + self.tab_linear(tab).squeeze(1)
            + self.duration_linear(duration).squeeze(1)
        )
        return deep_score + linear_score


def fit_din(split, labels, histories):
    torch.manual_seed(SEED + 41)
    model = DINHistoryModel()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=2e-6
    )

    video, author, tag, tab, duration = make_context(split)
    y = np.asarray(labels, dtype=np.float32)
    n = len(y)
    rng = np.random.default_rng(SEED + 73)

    video_t = torch.from_numpy(video)
    author_t = torch.from_numpy(author)
    tag_t = torch.from_numpy(tag)
    tab_t = torch.from_numpy(tab)
    duration_t = torch.from_numpy(duration)
    history_t = torch.from_numpy(histories)
    y_t = torch.from_numpy(y)

    for _ in range(DIN_EPOCHS):
        permutation = rng.permutation(n)
        model.train()
        for start in range(0, n, BATCH_SIZE):
            idx_np = permutation[start:start + BATCH_SIZE]
            idx = torch.from_numpy(idx_np)

            logits = model(
                video_t[idx], author_t[idx], tag_t[idx],
                tab_t[idx], duration_t[idx], history_t[idx]
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict_din(model, split, histories):
    model.eval()
    video, author, tag, tab, duration = make_context(split)
    n = len(video)
    scores = np.empty(n, dtype=np.float32)

    for start in range(0, n, PRED_BATCH):
        end = min(start + PRED_BATCH, n)
        scores[start:end] = model(
            torch.from_numpy(video[start:end]),
            torch.from_numpy(author[start:end]),
            torch.from_numpy(tag[start:end]),
            torch.from_numpy(tab[start:end]),
            torch.from_numpy(duration[start:end]),
            torch.from_numpy(histories[start:end])
        ).cpu().numpy()
    return scores.astype(np.float64)


def fit_spectral(split, labels):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)
    positive = y == 1

    matrix = sp.coo_matrix(
        (
            np.ones(int(positive.sum()), dtype=np.float32),
            (users[positive], videos[positive])
        ),
        shape=(USER_CARD, VIDEO_CARD),
        dtype=np.float32
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1.0

    item_users = np.asarray(
        matrix.getnnz(axis=0), dtype=np.float32
    )
    idf = 1.0 / np.sqrt(np.maximum(item_users, 1.0))
    weighted = matrix @ sp.diags(idf, format="csr")

    k = min(SVD_DIM, min(weighted.shape) - 1)
    u, singular, vt = svds(
        weighted, k=k, which="LM",
        random_state=SEED + 109
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order]

    user_factors = (
        u * np.sqrt(singular)[None, :]
    ).astype(np.float32)
    item_factors = (
        vt.T * np.sqrt(singular)[None, :]
    ).astype(np.float32)
    item_factors *= idf[:, None]
    return user_factors, item_factors


def predict_spectral(factors, split):
    user_factors, item_factors = factors
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    return np.einsum(
        "ij,ij->i",
        user_factors[users],
        item_factors[videos],
        optimize=True
    ).astype(np.float64)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((scores, users))
    sorted_users = users[order]
    boundaries = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.flatnonzero(boundaries)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = np.arange(n, dtype=np.int64) - np.repeat(
        starts, lengths
    )
    ranked = positions / np.maximum(
        np.repeat(lengths, lengths) - 1, 1
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def candidate_set(name, new_scores, incumbent_scores, users):
    result = [
        (
            name + "_standalone",
            np.asarray(new_scores, dtype=np.float64),
            (name, "standalone", 0.0)
        )
    ]
    new_rank = within_user_rank(users, new_scores)
    incumbent_rank = within_user_rank(users, incumbent_scores)
    for alpha in (0.50, 0.70, 0.80, 0.90, 0.95):
        result.append((
            name + "_rankblend_inc%.2f" % alpha,
            alpha * incumbent_rank + (1.0 - alpha) * new_rank,
            (name, "rankblend", alpha)
        ))
    return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
), dtype=np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

# Candidate-conditioned sequence family.
train_histories = training_positive_histories(train, y_train)
train_final_history = final_positive_history_table(train, y_train)
valid_histories = train_final_history[
    np.asarray(valid.user_id, dtype=np.int64)
]
din_model = fit_din(train, y_train, train_histories)
din_valid = predict_din(din_model, valid, valid_histories)

# Closed-form latent spectral family.
spectral_model = fit_spectral(train, y_train)
spectral_valid = predict_spectral(spectral_model, valid)

all_candidates = [
    (
        "trusted_incumbent",
        inc_valid,
        ("incumbent", "standalone", 1.0)
    )
]
all_candidates.extend(candidate_set(
    "din_history", din_valid, inc_valid, valid_users
))
all_candidates.extend(candidate_set(
    "spectral_mf", spectral_valid, inc_valid, valid_users
))

candidate_metrics = {}
best_name = None
best_scores = None
best_spec = None
best_primary = -np.inf

for name, scores, spec in all_candidates:
    result = evaluate(valid_users, y_valid, scores)
    primary = float(result["primary"])
    candidate_metrics[name] = primary
    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_scores = np.asarray(scores, dtype=np.float64)
        best_spec = spec

metrics = evaluate(valid_users, y_valid, best_scores)

print("CANDIDATES " + json.dumps(
    candidate_metrics, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "winner": best_name,
    "trusted_incumbent": candidate_metrics["trusted_incumbent"],
    "din_history_standalone":
        candidate_metrics["din_history_standalone"],
    "spectral_mf_standalone":
        candidate_metrics["spectral_mf_standalone"],
    "din_users_with_history_fraction": float(
        np.mean(np.any(valid_histories != 0, axis=1))
    )
}, sort_keys=True))

test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

family, transform, alpha = best_spec

if family == "incumbent":
    test_scores = inc_test.copy()
else:
    # Refit the identical selected recipe on train + validation.
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.user_id = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64)
    ])
    combined.video_id = np.concatenate([
        np.asarray(train.video_id, dtype=np.int64),
        np.asarray(valid.video_id, dtype=np.int64)
    ])
    combined.time_ms = np.concatenate([
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(valid.time_ms, dtype=np.int64)
    ])
    combined.X = {}
    for field in (
        "author_id", "tag", "tab", "duration_bucket"
    ):
        combined.X[field] = np.concatenate([
            np.asarray(train.X[field], dtype=np.int64),
            np.asarray(valid.X[field], dtype=np.int64)
        ])

    y_combined = np.concatenate([y_train, y_valid])

    if family == "din_history":
        combined_histories = training_positive_histories(
            combined, y_combined
        )
        final_history = final_positive_history_table(
            combined, y_combined
        )
        test_histories = final_history[test_users]
        final_model = fit_din(
            combined, y_combined, combined_histories
        )
        new_test = predict_din(
            final_model, test, test_histories
        )
    elif family == "spectral_mf":
        final_model = fit_spectral(combined, y_combined)
        new_test = predict_spectral(final_model, test)
    else:
        raise RuntimeError("Unknown selected family")

    if transform == "standalone":
        test_scores = new_test
    elif transform == "rankblend":
        test_scores = (
            float(alpha) * within_user_rank(test_users, inc_test)
            + (1.0 - float(alpha))
            * within_user_rank(test_users, new_test)
        )
    else:
        raise RuntimeError("Unknown selected transform")

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

wall = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(wall)
}))