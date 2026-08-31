import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 19427
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

DEVICE = torch.device("cpu")
HIST_LEN = 10
BATCH_SIZE = 8192
MAX_EPOCHS = 3

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "hour",
]
VIDEO_COL = CAT_FIELDS.index("video_id")

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])

field_cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
field_offsets = np.cumsum([0] + field_cards[:-1], dtype=np.int64)
TOTAL_CAT_CARD = int(sum(field_cards))


def cat_matrix(split):
    x = np.column_stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in CAT_FIELDS]
    )
    x += field_offsets[None, :]
    return np.ascontiguousarray(x, dtype=np.int64)


def sequence_histories(split, labels, hist_len=HIST_LEN):
    """
    For every fitted row, return the last hist_len positive video IDs that
    occurred strictly before that row for the same user. Sorting uses the
    prescribed (user_id, time_ms, row_position) ordering.
    Padding is zero; real video IDs are shifted by one.
    """
    labels = np.asarray(labels, dtype=np.int8)
    n = len(labels)
    row_position = np.arange(n, dtype=np.int64)
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)

    order = np.lexsort((row_position, times, users))
    su = users[order]
    sy = labels[order]
    sv = videos[order]

    positive_counts = np.bincount(
        su[sy > 0], minlength=USER_CARD
    ).astype(np.int64)
    positive_starts = np.empty(USER_CARD, dtype=np.int64)
    positive_starts[0] = 0
    np.cumsum(positive_counts[:-1], out=positive_starts[1:])

    global_positive_before = (
        np.cumsum(sy, dtype=np.int64) - sy.astype(np.int64)
    )
    user_start_mask = np.empty(n, dtype=bool)
    user_start_mask[0] = True
    user_start_mask[1:] = su[1:] != su[:-1]

    base_at_start = np.zeros(n, dtype=np.int64)
    base_at_start[user_start_mask] = global_positive_before[user_start_mask]
    user_base = np.maximum.accumulate(base_at_start)
    prior_count = global_positive_before - user_base

    positive_videos = sv[sy > 0]
    starts_for_rows = positive_starts[su]

    hist_sorted = np.zeros((n, hist_len), dtype=np.int32)
    for j in range(hist_len):
        lag = hist_len - j
        idx = starts_for_rows + prior_count - lag
        ok = idx >= starts_for_rows
        hist_sorted[ok, j] = positive_videos[idx[ok]].astype(np.int32) + 1

    result = np.empty_like(hist_sorted)
    result[order] = hist_sorted
    return result


def static_histories(fit_splits, fit_labels, target, hist_len=HIST_LEN):
    """
    Build each target user's history from all positive rows in fit_splits.
    No target outcomes are consulted.
    """
    all_users = np.concatenate(
        [np.asarray(s.user_id, dtype=np.int64) for s in fit_splits]
    )
    all_times = np.concatenate(
        [np.asarray(s.time_ms, dtype=np.int64) for s in fit_splits]
    )
    all_videos = np.concatenate(
        [np.asarray(s.video_id, dtype=np.int64) for s in fit_splits]
    )
    labels = np.concatenate(
        [np.asarray(y, dtype=np.int8) for y in fit_labels]
    )
    row_position = np.arange(len(labels), dtype=np.int64)

    positive = labels > 0
    pu = all_users[positive]
    pt = all_times[positive]
    pv = all_videos[positive]
    pp = row_position[positive]

    order = np.lexsort((pp, pt, pu))
    pu = pu[order]
    pv = pv[order]

    counts = np.bincount(pu, minlength=USER_CARD).astype(np.int64)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = ends - counts

    target_users = np.asarray(target.user_id, dtype=np.int64)
    result = np.zeros((len(target_users), hist_len), dtype=np.int32)
    starts_for_rows = starts[target_users]
    counts_for_rows = counts[target_users]

    for j in range(hist_len):
        lag = hist_len - j
        ok = counts_for_rows >= lag
        idx = starts_for_rows + counts_for_rows - lag
        result[ok, j] = pv[idx[ok]].astype(np.int32) + 1
    return result


class DINLite(nn.Module):
    def __init__(self):
        super().__init__()
        cat_dim = 8
        video_dim = 16

        self.cat_embedding = nn.Embedding(TOTAL_CAT_CARD, cat_dim)
        self.video_embedding = nn.Embedding(
            VIDEO_CARD + 1, video_dim, padding_idx=0
        )
        self.linear = nn.Embedding(TOTAL_CAT_CARD, 1)

        self.attention = nn.Sequential(
            nn.Linear(video_dim * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        deep_input = len(CAT_FIELDS) * cat_dim + video_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(deep_input, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.cat_embedding.weight, std=0.015)
        nn.init.normal_(self.video_embedding.weight, std=0.015)
        nn.init.zeros_(self.video_embedding.weight[0])
        nn.init.zeros_(self.linear.weight)

    def forward(self, cats, history):
        cat_emb = self.cat_embedding(cats)
        flat_cat = cat_emb.flatten(1)

        candidate_raw = (
            cats[:, VIDEO_COL] - int(field_offsets[VIDEO_COL]) + 1
        )
        candidate_emb = self.video_embedding(candidate_raw)
        history_emb = self.video_embedding(history)

        expanded_candidate = candidate_emb.unsqueeze(1).expand_as(history_emb)
        attention_input = torch.cat(
            [
                history_emb,
                expanded_candidate,
                history_emb - expanded_candidate,
                history_emb * expanded_candidate,
            ],
            dim=2,
        )
        attention_logits = self.attention(attention_input).squeeze(2)
        mask = history != 0
        attention_logits = attention_logits.masked_fill(mask.logical_not(), -1e4)
        weights = torch.softmax(attention_logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        interest = torch.sum(history_emb * weights.unsqueeze(2), dim=1)
        interaction = interest * candidate_emb

        deep_input = torch.cat(
            [flat_cat, candidate_emb, interest, interaction], dim=1
        )
        deep = self.mlp(deep_input).squeeze(1)
        wide = self.linear(cats).sum(dim=1).squeeze(1)
        return self.bias + wide + deep


@torch.no_grad()
def predict_din(model, cats, histories):
    model.eval()
    result = np.empty(len(cats), dtype=np.float64)
    for start in range(0, len(cats), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(cats))
        xb = torch.from_numpy(cats[start:end]).to(DEVICE)
        hb = torch.from_numpy(
            histories[start:end].astype(np.int64, copy=False)
        ).to(DEVICE)
        result[start:end] = (
            model(xb, hb).cpu().numpy().astype(np.float64)
        )
    return result


def fit_din_train_valid():
    xtr = cat_matrix(train)
    xva = cat_matrix(valid)
    htr = sequence_histories(train, y_train)
    hva = static_histories([train], [y_train], valid)

    model = DINLite().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0015, weight_decay=2e-6
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 11)

    y_tensor = torch.from_numpy(y_train)
    n = len(y_train)
    best_score = None
    best_state = None
    best_epoch = 1
    best_primary = -np.inf
    epoch_scores = {}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=generator)

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            idx_np = idx.numpy()
            xb = torch.from_numpy(xtr[idx_np]).to(DEVICE)
            hb = torch.from_numpy(
                htr[idx_np].astype(np.int64, copy=False)
            ).to(DEVICE)
            yb = y_tensor[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yb
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        score = predict_din(model, xva, hva)
        met = evaluate(valid.user_id, y_valid, score)
        epoch_scores[str(epoch)] = float(met["primary"])

        if float(met["primary"]) > best_primary:
            best_primary = float(met["primary"])
            best_epoch = epoch
            best_score = score.copy()
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    del model, optimizer, xtr, xva, htr, hva, y_tensor
    gc.collect()
    return best_score, best_epoch, epoch_scores


def transition_model_scores(fit_splits, fit_labels, target):
    """
    Empirical-Bayes score for candidate video conditioned on the user's most
    recent positive video. The fallback is a smoothed candidate-video rate.
    """
    fit_users = np.concatenate(
        [np.asarray(s.user_id, dtype=np.int64) for s in fit_splits]
    )
    fit_times = np.concatenate(
        [np.asarray(s.time_ms, dtype=np.int64) for s in fit_splits]
    )
    fit_videos = np.concatenate(
        [np.asarray(s.video_id, dtype=np.int64) for s in fit_splits]
    )
    labels = np.concatenate(
        [np.asarray(y, dtype=np.int8) for y in fit_labels]
    )
    row_position = np.arange(len(labels), dtype=np.int64)

    order = np.lexsort((row_position, fit_times, fit_users))
    su = fit_users[order]
    sv = fit_videos[order]
    sy = labels[order]

    positive_counts = np.bincount(
        su[sy > 0], minlength=USER_CARD
    ).astype(np.int64)
    positive_starts = np.empty(USER_CARD, dtype=np.int64)
    positive_starts[0] = 0
    np.cumsum(positive_counts[:-1], out=positive_starts[1:])

    global_before = np.cumsum(sy, dtype=np.int64) - sy.astype(np.int64)
    starts_mask = np.empty(len(sy), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = su[1:] != su[:-1]
    bases = np.zeros(len(sy), dtype=np.int64)
    bases[starts_mask] = global_before[starts_mask]
    bases = np.maximum.accumulate(bases)
    prior_count = global_before - bases

    positive_videos = sv[sy > 0]
    prior_index = positive_starts[su] + prior_count - 1
    has_prior = prior_count > 0
    last_video = np.zeros(len(sy), dtype=np.int64)
    last_video[has_prior] = positive_videos[prior_index[has_prior]] + 1

    # Candidate-video prior.
    item_n = np.bincount(sv, minlength=VIDEO_CARD).astype(np.float64)
    item_p = np.bincount(
        sv, weights=sy.astype(np.float64), minlength=VIDEO_CARD
    )
    global_rate = float(sy.mean())
    item_rate = (item_p + 20.0 * global_rate) / (item_n + 20.0)

    # Sparse transition table keyed by (last positive video, candidate).
    usable = has_prior
    keys = last_video[usable] * np.int64(VIDEO_CARD) + sv[usable]
    key_order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[key_order]
    sorted_y = sy[usable][key_order].astype(np.float64)

    unique_keys, first, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )
    positives = np.add.reduceat(sorted_y, first)

    previous_video = unique_keys // np.int64(VIDEO_CARD)
    candidate_video = unique_keys % np.int64(VIDEO_CARD)
    priors = item_rate[candidate_video]
    transition_rates = (positives + 8.0 * priors) / (counts + 8.0)

    target_history = static_histories(
        fit_splits, fit_labels, target, hist_len=1
    )[:, 0].astype(np.int64)
    target_video = np.asarray(target.video_id, dtype=np.int64)
    target_keys = (
        target_history * np.int64(VIDEO_CARD) + target_video
    )

    positions = np.searchsorted(unique_keys, target_keys)
    found = positions < len(unique_keys)
    safe_positions = np.minimum(positions, max(len(unique_keys) - 1, 0))
    if len(unique_keys):
        found &= unique_keys[safe_positions] == target_keys
    else:
        found[:] = False

    score = item_rate[target_video].astype(np.float64)
    if len(unique_keys):
        score[found] = transition_rates[safe_positions[found]]

    del (
        fit_users, fit_times, fit_videos, labels, order, su, sv, sy,
        keys, key_order, sorted_keys, sorted_y
    )
    gc.collect()
    return score


def standardized_blend(own, incumbent, weight):
    own = np.asarray(own, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    own_scale = max(float(np.std(own)), 1e-8)
    inc_scale = max(float(np.std(incumbent)), 1e-8)
    return weight * own / own_scale + (1.0 - weight) * incumbent / inc_scale


shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

din_valid, din_epoch, din_epoch_scores = fit_din_train_valid()
transition_valid = transition_model_scores([train], [y_train], valid)

families = {
    "din_attention": din_valid,
    "last_positive_transition": transition_valid,
}

candidate_scores = {}
best_primary = -np.inf
best_family = None
best_weight = 1.0
best_valid = None
best_raw_valid = None
best_metrics = None

for family, own_scores in families.items():
    own_metrics = evaluate(valid.user_id, y_valid, own_scores)
    own_primary = float(own_metrics["primary"])
    candidate_scores[family + "_standalone"] = own_primary

    if own_primary > best_primary:
        best_primary = own_primary
        best_family = family
        best_weight = 1.0
        best_valid = own_scores.copy()
        best_raw_valid = own_scores.copy()
        best_metrics = own_metrics

    family_best_primary = -np.inf
    family_best_weight = None
    family_best_scores = None
    family_best_metrics = None

    for weight in [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80]:
        blended = standardized_blend(own_scores, inc_valid, weight)
        metrics = evaluate(valid.user_id, y_valid, blended)
        primary = float(metrics["primary"])
        if primary > family_best_primary:
            family_best_primary = primary
            family_best_weight = weight
            family_best_scores = blended
            family_best_metrics = metrics

    candidate_scores[family + "_incumbent_blend"] = family_best_primary
    if family_best_primary > best_primary:
        best_primary = family_best_primary
        best_family = family
        best_weight = float(family_best_weight)
        best_valid = family_best_scores.copy()
        best_raw_valid = own_scores.copy()
        best_metrics = family_best_metrics

corr_din_inc = float(np.corrcoef(din_valid, inc_valid)[0, 1])
corr_transition_inc = float(
    np.corrcoef(transition_valid, inc_valid)[0, 1]
)
history_coverage = float(
    np.mean(
        static_histories([train], [y_train], valid, hist_len=1)[:, 0] != 0
    )
)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS din_epochs=%s selected_epoch=%d history_coverage=%.4f "
    "corr_din_inc=%.4f corr_transition_inc=%.4f selected=%s weight=%.2f"
    % (
        json.dumps(din_epoch_scores, sort_keys=True),
        din_epoch,
        history_coverage,
        corr_din_inc,
        corr_transition_inc,
        best_family,
        best_weight,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    if best_weight < 0.999999:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

# Refit the selected recipe on train + validation, then score test without
# reading any test outcome.
test = load("test")
combined_labels = np.concatenate(
    [y_train, y_valid.astype(np.float32)]
)

if best_family == "last_positive_transition":
    raw_test = transition_model_scores(
        [train, valid], [y_train, y_valid], test
    )
else:
    x_train = cat_matrix(train)
    x_valid = cat_matrix(valid)
    x_combined = np.ascontiguousarray(
        np.concatenate([x_train, x_valid], axis=0),
        dtype=np.int64,
    )

    h_train = sequence_histories(train, y_train)
    h_valid_local = sequence_histories(valid, y_valid)

    # Validation training histories must also contain positives from train.
    train_static_for_valid = static_histories(
        [train], [y_train], valid, hist_len=HIST_LEN
    )
    merged_valid_hist = np.zeros_like(h_valid_local)

    # Combine the fixed train tail with each row's preceding validation tail.
    for j in range(HIST_LEN):
        pass
    concatenated = np.concatenate(
        [train_static_for_valid, h_valid_local], axis=1
    )
    nonzero = concatenated != 0
    # Vectorized stable compaction to retain the last HIST_LEN nonzero events.
    ranks = np.cumsum(nonzero, axis=1)
    totals = ranks[:, -1]
    for j in range(HIST_LEN):
        desired = totals - (HIST_LEN - 1 - j)
        match = nonzero & (ranks == desired[:, None])
        rows, cols = np.nonzero(match)
        merged_valid_hist[rows, j] = concatenated[rows, cols]

    h_combined = np.ascontiguousarray(
        np.concatenate([h_train, merged_valid_hist], axis=0),
        dtype=np.int32,
    )
    y_combined_tensor = torch.from_numpy(
        combined_labels.astype(np.float32)
    )

    torch.manual_seed(SEED)
    final_model = DINLite().to(DEVICE)
    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=0.0015, weight_decay=2e-6
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 11)
    n = len(combined_labels)

    for epoch in range(din_epoch):
        final_model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            idx_np = idx.numpy()
            xb = torch.from_numpy(x_combined[idx_np]).to(DEVICE)
            hb = torch.from_numpy(
                h_combined[idx_np].astype(np.int64, copy=False)
            ).to(DEVICE)
            yb = y_combined_tensor[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = final_model(xb, hb)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yb
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 5.0)
            optimizer.step()

    x_test = cat_matrix(test)
    h_test = static_histories(
        [train, valid], [y_train, y_valid], test
    )
    raw_test = predict_din(final_model, x_test, h_test)

if best_weight < 0.999999:
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_scores = standardized_blend(raw_test, inc_test, best_weight)
else:
    test_scores = raw_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)