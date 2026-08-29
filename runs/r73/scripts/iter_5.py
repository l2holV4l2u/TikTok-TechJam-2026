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


START_TIME = time.time()
SEED = 20260829
HISTORY_LENGTH = 8
EMBED_DIM = 12
HIDDEN_DIM = 64
EPOCHS = 3
BATCH_SIZE = 8192
LEARNING_RATE = 2.0e-3
WEIGHT_DECAY = 1.0e-6

ITEM_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
]
CONTEXT_FIELDS = [
    "tab",
    "hour",
    "user_active_degree",
]
ALL_FIELDS = ITEM_FIELDS + CONTEXT_FIELDS

BLEND_ALPHAS = [0.0, 0.20, 0.40, 0.60, 0.75, 0.85, 0.92, 0.97, 1.0]


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, max(1, os.cpu_count() or 1)))


class CombinedSplit:
    def __init__(self, first, second):
        self.user_id = np.concatenate([
            np.asarray(first.user_id, dtype=np.int64),
            np.asarray(second.user_id, dtype=np.int64),
        ])
        self.time_ms = np.concatenate([
            np.asarray(first.time_ms, dtype=np.int64),
            np.asarray(second.time_ms, dtype=np.int64),
        ])
        self.X = {}
        for field in ALL_FIELDS:
            self.X[field] = np.concatenate([
                np.asarray(first.X[field], dtype=np.int64),
                np.asarray(second.X[field], dtype=np.int64),
            ])


def make_feature_arrays(split):
    item = np.stack(
        [np.asarray(split.X[f], dtype=np.int32) for f in ITEM_FIELDS],
        axis=1,
    )
    context = np.stack(
        [np.asarray(split.X[f], dtype=np.int32) for f in CONTEXT_FIELDS],
        axis=1,
    )
    users = np.asarray(split.user_id, dtype=np.int32)
    return item, context, users


def positive_sequence_table(split, labels):
    """
    Return the positive row indices in user/time order, together with dense
    per-user offsets into that array.
    """
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    n = len(users)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        times,
        users,
    ))
    ordered_users = users[order]
    ordered_labels = labels[order]

    positive_rows = order[ordered_labels != 0].astype(np.int32, copy=False)

    user_size = max(
        int(users.max(initial=0)) + 1,
        int(FEATURE_CARDINALITIES["user_id"]),
    )
    positive_counts = np.bincount(
        users[labels != 0],
        minlength=user_size,
    ).astype(np.int64)

    offsets = np.empty(user_size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(positive_counts, out=offsets[1:])

    return order, ordered_users, ordered_labels, positive_rows, offsets


def causal_history_indices(split, labels, history_length):
    """
    For every fitting row, find the preceding positive rows for the same user.
    Ordering is exactly (user_id, time_ms, original row position).
    """
    labels = np.asarray(labels, dtype=np.int8)
    users = np.asarray(split.user_id, dtype=np.int64)
    n = len(users)

    (
        order,
        ordered_users,
        ordered_labels,
        positive_rows,
        offsets,
    ) = positive_sequence_table(split, labels)

    cumulative = np.cumsum(ordered_labels, dtype=np.int64)

    starts = np.r_[
        0,
        np.flatnonzero(ordered_users[1:] != ordered_users[:-1]) + 1,
    ]
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    before_group = cumulative[starts] - ordered_labels[starts]
    group_base = np.repeat(before_group, lengths)

    prior_ordered = cumulative - group_base - ordered_labels
    prior = np.empty(n, dtype=np.int64)
    prior[order] = prior_ordered

    history = np.full((n, history_length), -1, dtype=np.int32)
    base = offsets[users]

    for k in range(history_length):
        sequence_position = base + prior - (k + 1)
        valid = sequence_position >= base
        history[valid, k] = positive_rows[sequence_position[valid]]

    del order, ordered_users, ordered_labels, cumulative
    del starts, ends, lengths, before_group, group_base, prior_ordered
    gc.collect()
    return history


def static_history_indices(fit_split, fit_labels, pred_split, history_length):
    """
    Histories for a later date split use all positives from the fitting split,
    but never any outcome from the split being scored.
    """
    fit_labels = np.asarray(fit_labels, dtype=np.int8)
    pred_users = np.asarray(pred_split.user_id, dtype=np.int64)

    _, _, _, positive_rows, offsets = positive_sequence_table(
        fit_split, fit_labels
    )

    user_size = len(offsets) - 1
    safe_users = np.minimum(pred_users, user_size - 1)
    known_user = pred_users < user_size

    starts = offsets[safe_users]
    counts = offsets[safe_users + 1] - starts

    history = np.full(
        (len(pred_users), history_length),
        -1,
        dtype=np.int32,
    )

    for k in range(history_length):
        sequence_position = starts + counts - (k + 1)
        valid = known_user & (counts > k)
        history[valid, k] = positive_rows[sequence_position[valid]]

    return history


class CausalDIN(nn.Module):
    def __init__(self):
        super().__init__()

        self.item_embeddings = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[field]),
                EMBED_DIM,
                padding_idx=0,
            )
            for field in ITEM_FIELDS
        ])
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(
                int(FEATURE_CARDINALITIES[field]),
                EMBED_DIM,
                padding_idx=0,
            )
            for field in CONTEXT_FIELDS
        ])
        self.user_embedding = nn.Embedding(
            int(FEATURE_CARDINALITIES["user_id"]),
            EMBED_DIM,
            padding_idx=0,
        )

        input_dim = EMBED_DIM * 6 + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.025)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def embed_items(self, values):
        result = 0.0
        for j, embedding in enumerate(self.item_embeddings):
            result = result + embedding(values[..., j])
        return result / np.sqrt(float(len(self.item_embeddings)))

    def forward(self, candidate_item, context, users, history_item, history_mask):
        candidate = self.embed_items(candidate_item)
        historical = self.embed_items(history_item)

        scale = float(EMBED_DIM) ** -0.5
        attention_logits = (
            historical * candidate.unsqueeze(1)
        ).sum(dim=-1) * scale

        attention_logits = attention_logits.masked_fill(
            ~history_mask, -1.0e4
        )
        attention = torch.softmax(attention_logits, dim=1)
        attention = attention * history_mask.float()
        attention = attention / attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)

        attended = (
            attention.unsqueeze(-1) * historical
        ).sum(dim=1)

        context_vector = 0.0
        for j, embedding in enumerate(self.context_embeddings):
            context_vector = context_vector + embedding(context[:, j])
        context_vector = context_vector / np.sqrt(
            float(len(self.context_embeddings))
        )

        user_vector = self.user_embedding(users)
        interaction = candidate * attended
        difference = torch.abs(candidate - attended)

        has_history = history_mask.any(dim=1).float().unsqueeze(1)
        history_count = torch.log1p(
            history_mask.sum(dim=1).float()
        ).unsqueeze(1)

        features = torch.cat([
            candidate,
            attended,
            interaction,
            difference,
            context_vector,
            user_vector,
            has_history,
            history_count,
        ], dim=1)

        return self.mlp(features).squeeze(1)


def batch_tensors(
    row_indices,
    item_array,
    context_array,
    user_array,
    history_indices,
    history_source_items,
):
    row_indices = np.asarray(row_indices, dtype=np.int64)

    hist_rows = history_indices[row_indices]
    mask = hist_rows >= 0
    safe_hist_rows = np.maximum(hist_rows, 0)

    candidate = torch.from_numpy(
        item_array[row_indices].astype(np.int64, copy=False)
    )
    context = torch.from_numpy(
        context_array[row_indices].astype(np.int64, copy=False)
    )
    users = torch.from_numpy(
        user_array[row_indices].astype(np.int64, copy=False)
    )
    history_item = torch.from_numpy(
        history_source_items[safe_hist_rows].astype(np.int64, copy=False)
    )
    history_mask = torch.from_numpy(mask)

    return candidate, context, users, history_item, history_mask


def fit_model(fit_split, labels, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    item, context, users = make_feature_arrays(fit_split)
    labels = np.asarray(labels, dtype=np.float32)
    history = causal_history_indices(
        fit_split, labels.astype(np.int8), HISTORY_LENGTH
    )

    model = CausalDIN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    n = len(labels)
    model.train()

    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        epoch_loss = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            rows = permutation[start:start + BATCH_SIZE]
            tensors = batch_tensors(
                rows,
                item,
                context,
                users,
                history,
                item,
            )
            target = torch.from_numpy(labels[rows])

            optimizer.zero_grad(set_to_none=True)
            logits = model(*tensors)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_n = len(rows)
            epoch_loss += float(loss.detach()) * batch_n
            seen += batch_n

        print(
            "FINDINGS "
            + json.dumps({
                "fit_epoch": epoch + 1,
                "fit_rows": n,
                "mean_bce": epoch_loss / max(seen, 1),
            }, sort_keys=True)
        )

    del history
    gc.collect()
    return model, item


def predict_model(model, fit_split, fit_labels, pred_split, fit_item):
    pred_item, pred_context, pred_users = make_feature_arrays(pred_split)
    history = static_history_indices(
        fit_split,
        fit_labels,
        pred_split,
        HISTORY_LENGTH,
    )

    scores = np.empty(len(pred_users), dtype=np.float64)
    model.eval()

    with torch.no_grad():
        for start in range(0, len(pred_users), BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, len(pred_users))
            rows = np.arange(start, end, dtype=np.int64)
            tensors = batch_tensors(
                rows,
                pred_item,
                pred_context,
                pred_users,
                history,
                fit_item,
            )
            scores[start:end] = model(*tensors).numpy().astype(
                np.float64, copy=False
            )

    history_coverage = float(np.mean(np.any(history >= 0, axis=1)))
    mean_history = float(np.mean(np.sum(history >= 0, axis=1)))

    del history, pred_item, pred_context, pred_users
    gc.collect()
    return scores, history_coverage, mean_history


def within_user_standardize(values, user_ids):
    values = np.asarray(values, dtype=np.float64)
    users = np.asarray(user_ids, dtype=np.int64)
    size = int(users.max(initial=0)) + 1

    counts = np.bincount(users, minlength=size).astype(np.float64)
    sums = np.bincount(
        users, weights=values, minlength=size
    ).astype(np.float64)
    squares = np.bincount(
        users, weights=values * values, minlength=size
    ).astype(np.float64)

    means = sums / np.maximum(counts, 1.0)
    variance = squares / np.maximum(counts, 1.0) - means * means
    standard_deviation = np.sqrt(np.maximum(variance, 1.0e-8))

    standardized = (
        values - means[users]
    ) / standard_deviation[users]
    return standardized


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(y_valid):
    raise RuntimeError("Incumbent validation score length mismatch")

valid_model, train_item = fit_model(train, y_train, SEED)
din_valid, valid_history_coverage, valid_mean_history = predict_model(
    valid_model,
    train,
    y_train,
    valid,
    train_item,
)

inc_component = within_user_standardize(
    inc_valid, valid.user_id
)
din_component = within_user_standardize(
    din_valid, valid.user_id
)

candidate_metrics = {}
candidate_scores = {}

raw_din_metrics = evaluate(
    valid.user_id, y_valid, din_valid
)
candidate_metrics["din_raw"] = float(raw_din_metrics["primary"])
candidate_scores["din_raw"] = din_valid

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = evaluate(
    valid.user_id, y_valid, best_scores
)
candidate_metrics["incumbent"] = float(best_metrics["primary"])

best_alpha = 1.0

for alpha in BLEND_ALPHAS:
    blended = (
        alpha * inc_component
        + (1.0 - alpha) * din_component
    )
    name = "blend_inc_{:.2f}".format(alpha)
    metrics = evaluate(
        valid.user_id, y_valid, blended
    )
    candidate_metrics[name] = float(metrics["primary"])
    candidate_scores[name] = blended

    if metrics["primary"] > best_metrics["primary"]:
        best_name = name
        best_scores = blended.copy()
        best_metrics = metrics
        best_alpha = float(alpha)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "selected": best_name,
        "selected_incumbent_alpha": best_alpha,
        "history_length": HISTORY_LENGTH,
        "validation_history_coverage": valid_history_coverage,
        "validation_mean_positive_history": valid_mean_history,
        "din_primary": float(raw_din_metrics["primary"]),
        "incumbent_primary": candidate_metrics["incumbent"],
        "best_primary": float(best_metrics["primary"]),
    }, sort_keys=True)
)

del valid_model, train_item, din_valid
gc.collect()

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("Incumbent test score length mismatch")

if best_alpha >= 1.0 - 1.0e-12:
    test_scores = inc_test
else:
    combined = CombinedSplit(train, valid)
    y_combined = np.concatenate([
        y_train,
        y_valid,
    ]).astype(np.int8, copy=False)

    test_model, combined_item = fit_model(
        combined,
        y_combined,
        SEED,
    )
    din_test, test_history_coverage, test_mean_history = predict_model(
        test_model,
        combined,
        y_combined,
        test,
        combined_item,
    )

    inc_test_component = within_user_standardize(
        inc_test, test.user_id
    )
    din_test_component = within_user_standardize(
        din_test, test.user_id
    )
    test_scores = (
        best_alpha * inc_test_component
        + (1.0 - best_alpha) * din_test_component
    )

    print(
        "FINDINGS "
        + json.dumps({
            "test_history_coverage": test_history_coverage,
            "test_mean_positive_history": test_mean_history,
            "test_refit_rows": int(len(y_combined)),
        }, sort_keys=True)
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)