import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
THREADS = min(8, os.cpu_count() or 8)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)

ntr = len(ytr)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "music_type",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

# ----------------------------------------------------------------------
# Inputs: stationary categorical/context fields, continuous quantities,
# and organizer-provided train-only entity histories.
# ----------------------------------------------------------------------
offsets = []
running = 0
for field in CAT_FIELDS:
    offsets.append(running)
    running += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)
TOTAL_CARDINALITY = int(running)


def categorical_matrix(sample):
    x = np.column_stack([
        np.asarray(sample.X[f], dtype=np.int64) for f in CAT_FIELDS
    ])
    x += offsets[None, :]
    return x


def raw_numeric_matrix(sample):
    cols = []
    for field in NUM_FIELDS:
        value = np.asarray(sample.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(value, 0.0)))

    return np.column_stack(cols).astype(np.float32, copy=False)


def history_matrix(split_name):
    cols = []
    names = []
    for key in ("video_id", "author_id"):
        hist = historical_features(split_name, key=key)
        for name in sorted(hist):
            value = np.asarray(hist[name], dtype=np.float32)
            value = np.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Counts can be very heavy-tailed, whereas rates are bounded.
            if (
                "count" in name.lower()
                or "num" in name.lower()
                or np.nanmax(value) > 50.0
            ):
                value = np.sign(value) * np.log1p(np.abs(value))
            cols.append(value)
            names.append(key + ":" + name)

    if not cols:
        length = (
            len(train.user_id) if split_name == "train"
            else len(valid.user_id) if split_name == "valid"
            else len(test.user_id)
        )
        return np.zeros((length, 0), dtype=np.float32), names

    return np.column_stack(cols).astype(np.float32, copy=False), names


xtr_cat = categorical_matrix(train)
xva_cat = categorical_matrix(valid)
xte_cat = categorical_matrix(test)

htr, history_names = history_matrix("train")
hva, _ = history_matrix("valid")
hte, _ = history_matrix("test")

xtr_num = np.column_stack([raw_numeric_matrix(train), htr]).astype(
    np.float32, copy=False
)
xva_num = np.column_stack([raw_numeric_matrix(valid), hva]).astype(
    np.float32, copy=False
)
xte_num = np.column_stack([raw_numeric_matrix(test), hte]).astype(
    np.float32, copy=False
)

# Robust scaling is fit on train only.
center = np.median(xtr_num, axis=0).astype(np.float32)
q25 = np.quantile(xtr_num, 0.25, axis=0).astype(np.float32)
q75 = np.quantile(xtr_num, 0.75, axis=0).astype(np.float32)
scale = np.maximum(q75 - q25, 1.0e-3).astype(np.float32)

xtr_num = np.clip((xtr_num - center) / scale, -8.0, 8.0).astype(
    np.float32, copy=False
)
xva_num = np.clip((xva_num - center) / scale, -8.0, 8.0).astype(
    np.float32, copy=False
)
xte_num = np.clip((xte_num - center) / scale, -8.0, 8.0).astype(
    np.float32, copy=False
)

print("FINDINGS history_feature_count=%d" % len(history_names))

# ----------------------------------------------------------------------
# Build bounded local slates from each user-day feed sequence.
#
# Long user-day sequences are divided into consecutive chunks. This retains
# true ordering and gives a tractable maximum action set for PL sampling.
# Only mixed-label chunks contribute because homogeneous chunks contain no
# ranking information.
# ----------------------------------------------------------------------
SLATE_SIZE = 16

users = np.asarray(train.user_id, dtype=np.int64)
dates = np.asarray(train.date, dtype=np.int64)
times = np.asarray(train.time_ms, dtype=np.int64)
rows = np.arange(ntr, dtype=np.int64)

order = np.lexsort((rows, times, dates, users))
ou = users[order]
od = dates[order]

group_start = np.flatnonzero(
    np.r_[True, (ou[1:] != ou[:-1]) | (od[1:] != od[:-1])]
)
group_end = np.r_[group_start[1:], ntr]
group_len = group_end - group_start

within_group = (
    np.arange(ntr, dtype=np.int64)
    - np.repeat(group_start, group_len)
)
base_gid = np.repeat(np.arange(len(group_start), dtype=np.int64), group_len)
chunk_within = within_group // SLATE_SIZE

# A collision-free integer key for (base group, local chunk).
chunk_key = base_gid * (
    int(np.max(chunk_within)) + 1
) + chunk_within
chunk_boundary = np.r_[True, chunk_key[1:] != chunk_key[:-1]]
chunk_id = np.cumsum(chunk_boundary, dtype=np.int64) - 1
chunk_starts = np.flatnonzero(chunk_boundary)
chunk_lengths = np.diff(np.r_[chunk_starts, ntr]).astype(np.int64)
num_chunks = len(chunk_starts)

position = (
    np.arange(ntr, dtype=np.int64)
    - np.repeat(chunk_starts, chunk_lengths)
)

slate_rows = np.full(
    (num_chunks, SLATE_SIZE), -1, dtype=np.int64
)
slate_rows[chunk_id, position] = order

safe_rows = np.maximum(slate_rows, 0)
slate_labels = ytr[safe_rows]
slate_mask = slate_rows >= 0
slate_labels *= slate_mask

slate_positive = slate_labels.sum(axis=1).astype(np.int64)
mixed = (
    (slate_positive > 0)
    & (slate_positive < chunk_lengths)
)

slate_rows = slate_rows[mixed]
slate_labels = slate_labels[mixed].astype(np.float32, copy=False)
slate_mask = slate_mask[mixed]
slate_sizes = chunk_lengths[mixed]
slate_positive = slate_positive[mixed]

# Four-day train-only temporal half-life.
train_unique_dates = np.unique(np.asarray(train.date, dtype=np.int64))
train_day = np.searchsorted(
    train_unique_dates, np.asarray(train.date, dtype=np.int64)
)
train_age = (
    len(train_unique_dates) - 1 - train_day
).astype(np.float32)
row_recency = np.exp2(-train_age / 4.0).astype(np.float32)
row_recency /= row_recency.mean()

slate_recency = row_recency[np.maximum(slate_rows[:, 0], 0)]
slate_recency /= np.mean(slate_recency)

print(
    "FINDINGS pl_mixed_slates=%d mean_size=%.3f mean_positive=%.3f"
    % (
        len(slate_rows),
        float(np.mean(slate_sizes)),
        float(np.mean(slate_positive)),
    )
)

discount_np = (
    1.0 / np.log2(np.arange(2, 7, dtype=np.float32))
).astype(np.float32)
ideal_prefix = np.r_[0.0, np.cumsum(discount_np)].astype(np.float32)
slate_ideal = ideal_prefix[np.minimum(slate_positive, 5)]

# ----------------------------------------------------------------------
# Three structurally different random-utility scorers.
# ----------------------------------------------------------------------
class WideUtility(nn.Module):
    """Additive random utility with no latent feature interactions."""

    def __init__(self, cardinality, numeric_dim):
        super().__init__()
        self.wide = nn.Embedding(cardinality, 1)
        self.numeric = nn.Linear(numeric_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.wide.weight)

    def forward(self, cat, num):
        return (
            self.bias
            + self.wide(cat).sum(dim=1).squeeze(-1)
            + self.numeric(num).squeeze(-1)
        )


class FactorizedUtility(nn.Module):
    """Second-order low-rank utility over all selected fields."""

    def __init__(self, cardinality, numeric_dim):
        super().__init__()
        dim = 10
        self.wide = nn.Embedding(cardinality, 1)
        self.factor = nn.Embedding(cardinality, dim)
        self.numeric = nn.Linear(numeric_dim, 1)
        self.num_factor = nn.Linear(numeric_dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.wide.weight)
        nn.init.normal_(self.factor.weight, std=0.018)

    def forward(self, cat, num):
        emb = self.factor(cat)
        summed = emb.sum(dim=1) + self.num_factor(num)
        squared_sum = summed.square()
        sum_squared = emb.square().sum(dim=1) + self.num_factor(num).square()
        interaction = 0.5 * (squared_sum - sum_squared).sum(dim=1)
        return (
            self.bias
            + self.wide(cat).sum(dim=1).squeeze(-1)
            + self.numeric(num).squeeze(-1)
            + interaction
        )


class MixtureUtility(nn.Module):
    """
    Mixture random utility: several context-dependent latent preference
    regimes produce utilities, then a gate marginalizes them by log-sum-exp.
    """

    def __init__(self, cardinality, fields, numeric_dim):
        super().__init__()
        dim = 8
        experts = 4
        self.embedding = nn.Embedding(cardinality, dim)
        self.wide_heads = nn.Embedding(cardinality, experts)
        self.numeric_heads = nn.Linear(numeric_dim, experts)
        self.gate = nn.Sequential(
            nn.Linear(fields * dim + numeric_dim, 64),
            nn.SiLU(),
            nn.Linear(64, experts),
        )
        self.residual = nn.Sequential(
            nn.Linear(fields * dim + numeric_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.018)
        nn.init.zeros_(self.wide_heads.weight)

    def forward(self, cat, num):
        representation = torch.cat(
            [self.embedding(cat).flatten(1), num], dim=1
        )
        expert_utility = (
            self.wide_heads(cat).sum(dim=1)
            + self.numeric_heads(num)
        )
        log_gate = torch.log_softmax(self.gate(representation), dim=1)
        marginal = torch.logsumexp(
            log_gate + expert_utility, dim=1
        )
        return marginal + self.residual(representation).squeeze(-1)


def pl_policy_loss(logits, labels, mask, sizes, ideal, samples=3):
    """
    REINFORCE gradient for exact Plackett-Luce trajectories.

    A leave-batch sample baseline (mean reward across trajectories from the
    same slate) reduces variance. The sampled reward is normalized DCG@5.
    """
    batch, width = logits.shape

    utility = logits[:, None, :].expand(batch, samples, width).reshape(
        batch * samples, width
    )
    gains = labels[:, None, :].expand(batch, samples, width).reshape(
        batch * samples, width
    )
    available = mask[:, None, :].expand(
        batch, samples, width
    ).reshape(batch * samples, width).clone()

    repeated_sizes = sizes[:, None].expand(batch, samples).reshape(-1)
    repeated_ideal = ideal[:, None].expand(batch, samples).reshape(-1)

    log_probability = torch.zeros(
        batch * samples, dtype=logits.dtype
    )
    reward = torch.zeros(batch * samples, dtype=logits.dtype)
    entropy_total = torch.zeros(batch * samples, dtype=logits.dtype)

    discounts = [1.0 / np.log2(r + 2.0) for r in range(5)]

    for rank in range(5):
        active = repeated_sizes > rank
        if not torch.any(active):
            break

        active_indices = torch.nonzero(active, as_tuple=False).squeeze(1)
        active_utility = utility[active_indices].masked_fill(
            ~available[active_indices], -1.0e9
        )
        probability = torch.softmax(active_utility, dim=1)
        selected = torch.multinomial(
            probability, num_samples=1
        ).squeeze(1)

        chosen_probability = probability.gather(
            1, selected[:, None]
        ).squeeze(1).clamp_min(1.0e-9)

        log_probability[active_indices] += torch.log(chosen_probability)
        reward[active_indices] += (
            gains[active_indices, selected] * float(discounts[rank])
        )
        entropy_total[active_indices] += -(
            probability * torch.log(probability.clamp_min(1.0e-9))
        ).sum(dim=1)

        available[active_indices, selected] = False

    reward = reward / repeated_ideal.clamp_min(1.0e-6)
    reward_matrix = reward.reshape(batch, samples)
    baseline = reward_matrix.mean(dim=1, keepdim=True)
    advantage = (reward_matrix - baseline).reshape(-1).detach()

    policy = -(advantage * log_probability).mean()
    entropy = entropy_total.mean()
    return policy - 0.001 * entropy, reward.mean().detach()


def fit_model(model, model_seed):
    rng = np.random.default_rng(model_seed)
    torch.manual_seed(model_seed)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0020, weight_decay=2.0e-6
    )
    batch_size = 768
    reward_observations = []

    model.train()
    for epoch in range(3):
        permutation = rng.permutation(len(slate_rows))

        for begin in range(0, len(permutation), batch_size):
            chosen = permutation[begin:begin + batch_size]
            rows_batch = slate_rows[chosen]
            mask_np = slate_mask[chosen]
            safe = np.maximum(rows_batch, 0)

            flat_rows = safe.reshape(-1)
            cat = torch.from_numpy(xtr_cat[flat_rows])
            num = torch.from_numpy(xtr_num[flat_rows])
            labels = torch.from_numpy(slate_labels[chosen])
            mask = torch.from_numpy(mask_np)
            sizes = torch.from_numpy(slate_sizes[chosen])
            ideal = torch.from_numpy(slate_ideal[chosen])
            recency = torch.from_numpy(slate_recency[chosen])

            logits = model(cat, num).reshape(
                len(chosen), SLATE_SIZE
            )
            logits = logits.masked_fill(~mask, -1.0e9)

            policy_loss, sampled_reward = pl_policy_loss(
                logits, labels, mask, sizes, ideal, samples=3
            )

            # A small pointwise anchor prevents utility drift and supplies
            # low-variance gradients before sampled trajectories specialize
            # the top of the slate.
            valid_logits = logits[mask]
            valid_labels = labels[mask]
            bce = nn.functional.binary_cross_entropy_with_logits(
                valid_logits, valid_labels, reduction="none"
            )

            repeated_recency = recency[:, None].expand_as(mask)[mask]
            bce = (bce * repeated_recency).sum() / repeated_recency.sum()

            loss = policy_loss + 0.18 * bce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if begin % (batch_size * 20) == 0:
                reward_observations.append(float(sampled_reward))

    return float(np.mean(reward_observations[-20:]))


def predict_model(model, cat_matrix, num_matrix):
    result = np.empty(len(cat_matrix), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(cat_matrix), 32768):
            end = min(begin + 32768, len(cat_matrix))
            cat = torch.from_numpy(cat_matrix[begin:end])
            num = torch.from_numpy(num_matrix[begin:end])
            result[begin:end] = model(cat, num).cpu().numpy()
    return result


def user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    row = np.arange(len(scores), dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    position = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominator = np.maximum(
        np.repeat(lengths, lengths) - 1, 1
    )
    ranked = position / denominator

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = ranked
    return result


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)

inc_valid_rank = user_rank(valid.user_id, inc_valid)
inc_test_rank = user_rank(test.user_id, inc_test)

families = [
    (
        "pl_wide",
        WideUtility(TOTAL_CARDINALITY, xtr_num.shape[1]),
    ),
    (
        "pl_factorized",
        FactorizedUtility(TOTAL_CARDINALITY, xtr_num.shape[1]),
    ),
    (
        "pl_mixture",
        MixtureUtility(
            TOTAL_CARDINALITY,
            len(CAT_FIELDS),
            xtr_num.shape[1],
        ),
    ),
]

candidate_scores = {}
family_outputs = {}

best_primary = -np.inf
best_name = None
best_valid = None
best_test = None
best_raw_valid = None
best_metric = None
best_alpha = None

for family_index, (name, model) in enumerate(families):
    sampled_reward = fit_model(model, SEED + 1000 * (family_index + 1))

    raw_valid = predict_model(model, xva_cat, xva_num)
    raw_test = predict_model(model, xte_cat, xte_num)

    raw_metric = evaluate(uva, yva, raw_valid)
    candidate_scores[name + "_raw"] = float(raw_metric["primary"])

    model_valid_rank = user_rank(valid.user_id, raw_valid)
    model_test_rank = user_rank(test.user_id, raw_test)

    family_best_primary = float(evaluate(uva, yva, inc_valid)["primary"])
    family_best_alpha = 0.0
    family_best_valid = inc_valid
    family_best_test = inc_test
    family_best_metric = evaluate(uva, yva, inc_valid)

    # Rank normalization ensures that a blend coefficient describes ordering
    # contribution rather than arbitrary utility scale.
    for alpha in (
        0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 1.00
    ):
        blended_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * model_valid_rank
        )
        metric = evaluate(uva, yva, blended_valid)

        if metric["primary"] > family_best_primary:
            family_best_primary = float(metric["primary"])
            family_best_alpha = float(alpha)
            family_best_valid = blended_valid
            family_best_test = (
                (1.0 - alpha) * inc_test_rank
                + alpha * model_test_rank
            )
            family_best_metric = metric

    candidate_scores[name + "_blend"] = family_best_primary
    print(
        "FINDINGS %s sampled_ndcg=%.6f raw_primary=%.6f "
        "blend_primary=%.6f alpha=%.2f"
        % (
            name,
            sampled_reward,
            raw_metric["primary"],
            family_best_primary,
            family_best_alpha,
        )
    )

    family_outputs[name] = (
        raw_valid,
        raw_test,
        family_best_valid,
        family_best_test,
        family_best_metric,
        family_best_alpha,
    )

    if family_best_primary > best_primary:
        best_primary = family_best_primary
        best_name = name
        best_valid = np.asarray(family_best_valid, dtype=np.float64)
        best_test = np.asarray(family_best_test, dtype=np.float64)
        best_raw_valid = np.asarray(raw_valid, dtype=np.float64)
        best_metric = family_best_metric
        best_alpha = family_best_alpha

    del model, raw_valid, raw_test
    gc.collect()

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS winner=%s alpha=%.2f"
    % (best_name, best_alpha)
)

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
    # The reported result can contain the trusted incumbent, so expose the
    # selected newly trained model's standalone predictions as required.
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, '
    '"ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        best_metric["primary"],
        best_metric["gauc"],
        best_metric["ndcg@5"],
        elapsed,
    )
)