import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 27419
EPOCHS = 2
MAX_USERS_PER_BATCH = 256
MAX_TOKENS_PER_BATCH = 32768
PRED_MAX_TOKENS = 65536

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

FIELDS = [
    "video_id", "author_id", "tag", "tab",
    "duration_bucket", "upload_type", "music_type", "hour"
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]


class Joined:
    pass


def join_splits(a, b, include_labels=True):
    z = Joined()
    z.X = {
        f: np.concatenate([
            np.asarray(a.X[f], dtype=np.int64),
            np.asarray(b.X[f], dtype=np.int64)
        ])
        for f in a.X
    }
    z.user_id = np.concatenate([
        np.asarray(a.user_id, dtype=np.int64),
        np.asarray(b.user_id, dtype=np.int64)
    ])
    z.time_ms = np.concatenate([
        np.asarray(a.time_ms, dtype=np.int64),
        np.asarray(b.time_ms, dtype=np.int64)
    ])
    z.date = np.concatenate([
        np.asarray(a.date, dtype=np.int32),
        np.asarray(b.date, dtype=np.int32)
    ])
    if include_labels:
        z.y = np.concatenate([
            np.asarray(a.y, dtype=np.int8),
            np.asarray(b.y, dtype=np.int8)
        ])
    return z


def sequence_arrays(split):
    n = len(split.user_id)
    row = np.arange(n, dtype=np.int64)
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    order = np.lexsort((row, times, users))

    su = users[order]
    st = times[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    same_prev = np.r_[False, su[1:] == su[:-1]]
    gap = np.zeros(n, dtype=np.float32)
    raw_gap = np.zeros(n, dtype=np.float64)
    raw_gap[1:] = np.maximum(
        (st[1:] - st[:-1]).astype(np.float64) / 1000.0, 0.0
    )
    raw_gap[~same_prev] = 0.0
    gap[:] = np.log1p(np.minimum(raw_gap, 86400.0)).astype(np.float32)

    reset = (~same_prev) | (raw_gap > 1800.0)
    reset_locations = np.where(reset, np.arange(n), 0)
    last_reset = np.maximum.accumulate(reset_locations)
    session_position = np.arange(n) - last_reset

    numerical_sorted = np.column_stack([
        gap / np.float32(np.log1p(86400.0)),
        np.log1p(session_position).astype(np.float32) / np.float32(np.log(128.0)),
        reset.astype(np.float32)
    ]).astype(np.float32)

    numerical = np.empty_like(numerical_sorted)
    numerical[order] = numerical_sorted

    cats = np.column_stack([
        np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
    ])
    for j, card in enumerate(CARDS):
        cats[:, j] = np.clip(cats[:, j], 0, card - 1)

    return {
        "cats": np.ascontiguousarray(cats),
        "nums": np.ascontiguousarray(numerical),
        "order": order,
        "starts": starts,
        "ends": ends,
        "lengths": lengths
    }


def make_batches(seq, token_limit, shuffle=False, seed=0):
    lengths = seq["lengths"]
    group_order = np.argsort(lengths, kind="stable")
    batches = []
    current = []
    current_max = 0

    for g in group_order:
        length = int(lengths[g])
        proposed_max = max(current_max, length)
        proposed_users = len(current) + 1
        if current and (
            proposed_users > MAX_USERS_PER_BATCH
            or proposed_max * proposed_users > token_limit
        ):
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
            current_max = 0
        current.append(int(g))
        current_max = max(current_max, length)

    if current:
        batches.append(np.asarray(current, dtype=np.int64))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(batches)
    return batches


def materialize_batch(seq, groups, labels=None, weights=None):
    starts = seq["starts"]
    ends = seq["ends"]
    order = seq["order"]
    lengths = ends[groups] - starts[groups]
    batch_size = len(groups)
    max_len = int(lengths.max())

    cats = np.zeros((batch_size, max_len, len(FIELDS)), dtype=np.int64)
    nums = np.zeros((batch_size, max_len, 3), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    ys = None if labels is None else np.zeros(
        (batch_size, max_len), dtype=np.float32
    )
    ws = None if weights is None else np.zeros(
        (batch_size, max_len), dtype=np.float32
    )
    original_ids = []

    for i, g in enumerate(groups):
        ids = order[starts[g]:ends[g]]
        length = len(ids)
        cats[i, :length] = seq["cats"][ids]
        nums[i, :length] = seq["nums"][ids]
        mask[i, :length] = 1.0
        if ys is not None:
            ys[i, :length] = labels[ids]
        if ws is not None:
            ws[i, :length] = weights[ids]
        original_ids.append(ids)

    return cats, nums, mask, ys, ws, original_ids


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, 4, padding_idx=0) for card in CARDS
        ])
        dimension = 4 * len(FIELDS) + 3
        self.projection = nn.Sequential(
            nn.Linear(dimension, 32),
            nn.LayerNorm(32),
            nn.SiLU()
        )

    def forward(self, cats, nums):
        parts = [
            emb(cats[:, :, j])
            for j, emb in enumerate(self.embeddings)
        ]
        parts.append(nums)
        return self.projection(torch.cat(parts, dim=-1))


class CausalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.gru = nn.GRU(32, 24, batch_first=True)
        self.skip = nn.Linear(32, 1)
        self.head = nn.Sequential(
            nn.Linear(24, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, cats, nums):
        x = self.encoder(cats, nums)
        h, _ = self.gru(x)
        return (self.head(h) + self.skip(x)).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation
        )
        self.norm = nn.GroupNorm(4, channels)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        padded = F.pad(x, (2 * self.dilation, 0))
        z = self.conv(padded)
        z = F.silu(self.norm(z))
        z = self.out(z)
        return F.silu(x + z)


class CausalTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.blocks = nn.ModuleList([
            CausalBlock(32, 1),
            CausalBlock(32, 2),
            CausalBlock(32, 4),
            CausalBlock(32, 8)
        ])
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, cats, nums):
        encoded = self.encoder(cats, nums)
        x = encoded.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.head(x.transpose(1, 2)).squeeze(-1)


def make_model(family):
    if family == "gru":
        return CausalGRU()
    if family == "tcn":
        return CausalTCN()
    raise ValueError(family)


def training_weights(dates):
    dates = np.asarray(dates)
    unique = np.unique(dates)
    day = np.searchsorted(unique, dates)
    age = day.max() - day
    w = np.exp2(-age / 10.0).astype(np.float32)
    return w / max(float(w.mean()), 1e-8)


def fit_sequence_model(split, family):
    torch.manual_seed(SEED + (11 if family == "gru" else 29))
    seq = sequence_arrays(split)
    labels = np.asarray(split.y, dtype=np.float32)
    weights = training_weights(split.date)
    model = make_model(family)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=7e-5
    )

    for epoch in range(EPOCHS):
        batches = make_batches(
            seq, MAX_TOKENS_PER_BATCH, shuffle=True,
            seed=SEED + epoch + (100 if family == "tcn" else 0)
        )
        model.train()
        for groups in batches:
            c, n, mask, y, w, _ = materialize_batch(
                seq, groups, labels, weights
            )
            ct = torch.from_numpy(c)
            nt = torch.from_numpy(n)
            mt = torch.from_numpy(mask)
            yt = torch.from_numpy(y)
            wt = torch.from_numpy(w)

            logits = model(ct, nt)
            losses = F.binary_cross_entropy_with_logits(
                logits, yt, reduction="none"
            )
            effective = mt * wt
            loss = (losses * effective).sum() / effective.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.inference_mode()
def predict_sequence(model, split):
    seq = sequence_arrays(split)
    result = np.empty(len(split.user_id), dtype=np.float32)
    batches = make_batches(seq, PRED_MAX_TOKENS, shuffle=False)

    model.eval()
    for groups in batches:
        c, n, mask, _, _, original_ids = materialize_batch(
            seq, groups, None, None
        )
        logits = model(torch.from_numpy(c), torch.from_numpy(n)).cpu().numpy()
        for i, ids in enumerate(original_ids):
            result[ids] = logits[i, :len(ids)]
    return result.astype(np.float64)


def previous_values(split, field):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    values = np.asarray(split.X[field], dtype=np.int64)
    row = np.arange(len(users), dtype=np.int64)
    order = np.lexsort((row, times, users))
    su = users[order]
    sv = values[order]

    prev_sorted = np.zeros(len(users), dtype=np.int64)
    same = np.r_[False, su[1:] == su[:-1]]
    prev_sorted[1:] = np.where(same[1:], sv[:-1], 0)

    prev = np.empty_like(prev_sorted)
    prev[order] = prev_sorted
    return prev


def smoothed_code_rates(ref_codes, ref_y, eval_codes, alpha, prior):
    unique, inverse = np.unique(ref_codes, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    positive = np.bincount(
        inverse, weights=ref_y, minlength=len(unique)
    ).astype(np.float64)

    loc = np.searchsorted(unique, eval_codes)
    safe = np.minimum(loc, len(unique) - 1)
    known = (loc < len(unique)) & (unique[safe] == eval_codes)

    ec = np.zeros(len(eval_codes), dtype=np.float64)
    ep = np.zeros(len(eval_codes), dtype=np.float64)
    ec[known] = count[safe[known]]
    ep[known] = positive[safe[known]]
    return (ep + alpha * prior) / (ec + alpha)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def transition_bayes(reference, evaluation):
    y = np.asarray(reference.y, dtype=np.float64)
    prior = float(y.mean())
    joined = join_splits(reference, evaluation, include_labels=False)
    offset = len(reference.user_id)

    score = np.zeros(len(evaluation.user_id), dtype=np.float64)
    total = 0.0

    entity_specs = [
        ("video_id", 1.0, 28.0),
        ("author_id", 0.8, 28.0),
        ("tag", 0.45, 40.0),
        ("tab", 0.35, 40.0)
    ]
    for field, weight, alpha in entity_specs:
        ref_code = np.asarray(reference.X[field], dtype=np.int64)
        ev_code = np.asarray(evaluation.X[field], dtype=np.int64)
        rate = smoothed_code_rates(ref_code, y, ev_code, alpha, prior)
        score += weight * logit(rate)
        total += weight

    transition_specs = [
        ("tag", "tag", 0.85, 35.0),
        ("tab", "tag", 0.55, 45.0),
        ("duration_bucket", "tag", 0.45, 45.0),
        ("tag", "duration_bucket", 0.40, 45.0),
        ("author_id", "tag", 0.50, 55.0),
        ("upload_type", "tag", 0.30, 55.0)
    ]

    for previous_field, current_field, weight, alpha in transition_specs:
        ref_prev = previous_values(reference, previous_field)
        joined_prev = previous_values(joined, previous_field)[offset:]
        current_card = int(FEATURE_CARDINALITIES[current_field])

        ref_current = np.asarray(
            reference.X[current_field], dtype=np.int64
        )
        eval_current = np.asarray(
            evaluation.X[current_field], dtype=np.int64
        )
        ref_code = ref_prev * np.int64(current_card) + ref_current
        eval_code = joined_prev * np.int64(current_card) + eval_current

        rate = smoothed_code_rates(ref_code, y, eval_code, alpha, prior)
        score += weight * logit(rate)
        total += weight

    return score / max(total, 1e-8)


def within_user_rank(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    lengths = ends - starts
    positions = np.arange(n) - np.repeat(starts, lengths)
    denominators = np.repeat(np.maximum(lengths - 1, 1), lengths)
    ranked = positions / denominators
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = float(x.std())
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - float(x.mean())) / s


def candidate_variants(name, scores, incumbent, users):
    variants = [(name + "_raw", scores, (name, "raw", 0.0))]
    nr = within_user_rank(users, scores)
    ir = within_user_rank(users, incumbent)
    nz = zscore(scores)
    iz = zscore(incumbent)

    for alpha in (0.25, 0.50, 0.75, 0.90):
        variants.append((
            "%s_rank_inc%.2f" % (name, alpha),
            alpha * ir + (1.0 - alpha) * nr,
            (name, "rank", alpha)
        ))
        variants.append((
            "%s_z_inc%.2f" % (name, alpha),
            alpha * iz + (1.0 - alpha) * nz,
            (name, "z", alpha)
        ))
    return variants


train = load("train")
valid = load("valid")
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(os.path.join(shared, "incumbent_valid_scores.npy"))
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

train_valid_features = join_splits(train, valid, include_labels=True)
valid_offset = len(train.user_id)

family_valid = {}

bayes_valid = transition_bayes(train, valid)
family_valid["transition_bayes"] = bayes_valid

trained_models = {}
for family in ("gru", "tcn"):
    model = fit_sequence_model(train, family)
    trained_models[family] = model
    all_predictions = predict_sequence(model, train_valid_features)
    family_valid[family] = all_predictions[valid_offset:]

all_candidates = [
    ("incumbent", np.asarray(inc_valid, dtype=np.float64),
     ("incumbent", "raw", 1.0))
]
for family, scores in family_valid.items():
    all_candidates.extend(
        candidate_variants(
            family, scores, inc_valid, valid_users
        )
    )

candidate_scores = {}
best_name = None
best_scores = None
best_recipe = None
best_metric = -np.inf

for name, scores, recipe in all_candidates:
    metrics = evaluate(valid_users, valid_y, scores)
    candidate_scores[name] = float(metrics["primary"])
    if metrics["primary"] > best_metric:
        best_metric = float(metrics["primary"])
        best_name = name
        best_scores = np.asarray(scores, dtype=np.float64)
        best_recipe = recipe

inc_rank = within_user_rank(valid_users, inc_valid)
findings = {}
for family, scores in family_valid.items():
    corr = np.corrcoef(
        inc_rank, within_user_rank(valid_users, scores)
    )[0, 1]
    findings[family + "_rank_corr_inc"] = float(corr)

print("FINDINGS " + json.dumps(findings, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

final_metrics = evaluate(valid_users, valid_y, best_scores)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )

test = load("test")
inc_test = np.load(inc_test_path)

selected_family, combine_method, alpha = best_recipe

if selected_family == "incumbent":
    test_scores = np.asarray(inc_test, dtype=np.float64)
else:
    combined_reference = join_splits(train, valid, include_labels=True)

    if selected_family == "transition_bayes":
        new_test = transition_bayes(combined_reference, test)
    else:
        refit_model = fit_sequence_model(combined_reference, selected_family)
        reference_test_features = join_splits(
            combined_reference, test, include_labels=False
        )
        all_test_predictions = predict_sequence(
            refit_model, reference_test_features
        )
        new_test = all_test_predictions[len(combined_reference.user_id):]

    if combine_method == "raw":
        test_scores = np.asarray(new_test, dtype=np.float64)
    elif combine_method == "rank":
        test_scores = (
            alpha * within_user_rank(test.user_id, inc_test)
            + (1.0 - alpha) * within_user_rank(test.user_id, new_test)
        )
    elif combine_method == "z":
        test_scores = alpha * zscore(inc_test) + (1.0 - alpha) * zscore(new_test)
    else:
        raise ValueError(combine_method)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))