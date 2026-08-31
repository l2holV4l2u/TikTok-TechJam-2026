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
SEED = 314159
BATCH = 4096
MAX_SLATE = 96

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))
rng = np.random.default_rng(SEED)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
dates = np.asarray(train.date, dtype=np.int64)

# Recency weighting is fixed before validation inspection. It makes training
# examples near the date boundary dominate because the evaluation period
# continues the observed temporal drift.
row_weight = np.exp2(
    (dates - dates.max()).astype(np.float32) / 4.0
)
row_weight /= row_weight.mean()
row_weight = row_weight.astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Ascending score gives positions 0..n-1. Row position is a deterministic
    # tie breaker.
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    su = user_ids[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked_sorted[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


# ---------------------------------------------------------------------------
# Family 1: field-aware factorization trained with hard-negative RankNet.
# Unlike an FM, each field has a different representation depending on the
# field with which it interacts.
# ---------------------------------------------------------------------------

FFM_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "onehot_feat3",
]
ffm_cards = [int(FEATURE_CARDINALITIES[f]) for f in FFM_FIELDS]


def make_local_matrix(split, fields, cards):
    cols = []
    for field, card in zip(fields, cards):
        x = np.asarray(split.X[field], dtype=np.int64)
        if x.size and (x.min() < 0 or x.max() >= card):
            raise ValueError("Out-of-range ID for " + field)
        cols.append(x)
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int64)


xffm_tr = make_local_matrix(train, FFM_FIELDS, ffm_cards)
xffm_va = make_local_matrix(valid, FFM_FIELDS, ffm_cards)
xffm_te = make_local_matrix(test, FFM_FIELDS, ffm_cards)


class FieldAwareRanker(nn.Module):
    def __init__(self, cards, rank=8):
        super().__init__()
        self.n_fields = len(cards)
        self.rank = rank
        self.linear = nn.ModuleList([
            nn.Embedding(card, 1) for card in cards
        ])
        # Table i stores a separate rank-dimensional representation of its
        # categories for every possible partner field.
        self.field_vectors = nn.ModuleList([
            nn.Embedding(card, self.n_fields * rank) for card in cards
        ])
        self.bias = nn.Parameter(torch.zeros(1))

        for table in self.linear:
            nn.init.zeros_(table.weight)
        for table in self.field_vectors:
            nn.init.normal_(table.weight, std=0.025)

    def forward(self, x):
        linear = self.bias.expand(x.shape[0])
        vectors = []

        for i in range(self.n_fields):
            linear = linear + self.linear[i](x[:, i]).squeeze(1)
            vectors.append(
                self.field_vectors[i](x[:, i]).view(
                    x.shape[0], self.n_fields, self.rank
                )
            )

        interaction = torch.zeros_like(linear)
        for i in range(self.n_fields):
            for j in range(i + 1, self.n_fields):
                interaction = interaction + (
                    vectors[i][:, j, :] * vectors[j][:, i, :]
                ).sum(dim=1)

        return linear + interaction


@torch.inference_mode()
def predict_rows(model, x):
    model.eval()
    ans = np.empty(len(x), dtype=np.float64)
    for lo in range(0, len(x), 16384):
        hi = min(lo + 16384, len(x))
        z = model(torch.from_numpy(x[lo:hi]))
        ans[lo:hi] = z.cpu().numpy().astype(np.float64)
    return ans


# Same-user negative lookup, built without row loops.
train_users = np.asarray(train.X["user_id"], dtype=np.int64)
negative_rows = np.flatnonzero(ytr < 0.5)
negative_order = np.argsort(
    train_users[negative_rows], kind="stable"
)
negative_rows = negative_rows[negative_order]
negative_users = train_users[negative_rows]

n_users = int(FEATURE_CARDINALITIES["user_id"])
negative_counts = np.bincount(
    negative_users, minlength=n_users
).astype(np.int64)
negative_starts = np.cumsum(
    np.r_[np.int64(0), negative_counts[:-1]], dtype=np.int64
)

positive_rows = np.flatnonzero(ytr > 0.5)
positive_rows = positive_rows[
    negative_counts[train_users[positive_rows]] > 0
]

# A hard-negative prior based only on train labels. Among several random
# negatives from the same user, select the one whose video/author historically
# receives the highest smoothed positive rate.
video_ids = np.asarray(train.X["video_id"], dtype=np.int64)
author_ids = np.asarray(train.X["author_id"], dtype=np.int64)

video_card = int(FEATURE_CARDINALITIES["video_id"])
author_card = int(FEATURE_CARDINALITIES["author_id"])

video_count = np.bincount(
    video_ids, minlength=video_card
).astype(np.float32)
video_pos = np.bincount(
    video_ids,
    weights=ytr,
    minlength=video_card,
).astype(np.float32)

author_count = np.bincount(
    author_ids, minlength=author_card
).astype(np.float32)
author_pos = np.bincount(
    author_ids,
    weights=ytr,
    minlength=author_card,
).astype(np.float32)

global_rate = float(ytr.mean())
video_rate = (
    video_pos + 20.0 * global_rate
) / (video_count + 20.0)
author_rate = (
    author_pos + 30.0 * global_rate
) / (author_count + 30.0)

hard_prior = (
    0.65 * video_rate[video_ids]
    + 0.35 * author_rate[author_ids]
).astype(np.float32)


def train_ffm_ranker(model, epochs=4, lr=0.0025):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=2e-6
    )
    model.train()

    for _ in range(epochs):
        pos = rng.permutation(positive_rows)
        users = train_users[pos]
        counts = negative_counts[users]

        # Four random same-user candidates, followed by vectorized hard choice.
        offsets = (
            rng.random((len(pos), 4)) * counts[:, None]
        ).astype(np.int64)
        candidates = negative_rows[
            negative_starts[users, None] + offsets
        ]
        best_column = np.argmax(hard_prior[candidates], axis=1)
        neg = candidates[np.arange(len(pos)), best_column]

        for lo in range(0, len(pos), BATCH):
            hi = min(lo + BATCH, len(pos))
            pi = pos[lo:hi]
            ni = neg[lo:hi]

            xp = torch.from_numpy(xffm_tr[pi])
            xn = torch.from_numpy(xffm_tr[ni])
            wp = torch.from_numpy(row_weight[pi])

            optimizer.zero_grad(set_to_none=True)
            positive_score = model(xp)
            negative_score = model(xn)

            rank_loss = F.softplus(
                -(positive_score - negative_score)
            )
            # A small calibrated component stabilizes category biases while
            # RankNet remains the dominant objective.
            point_loss = (
                F.binary_cross_entropy_with_logits(
                    positive_score,
                    torch.ones_like(positive_score),
                    reduction="none",
                )
                + F.binary_cross_entropy_with_logits(
                    negative_score,
                    torch.zeros_like(negative_score),
                    reduction="none",
                )
            )

            loss = (
                (rank_loss + 0.10 * point_loss) * wp
            ).sum() / wp.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


torch.manual_seed(SEED + 1)
ffm = FieldAwareRanker(ffm_cards, rank=8)
train_ffm_ranker(ffm)
ffm_valid = predict_rows(ffm, xffm_va)
ffm_test = predict_rows(ffm, xffm_te)
del ffm


# ---------------------------------------------------------------------------
# Family 2: setwise slate transformer.
#
# Its score for an impression is conditioned on the other impressions logged
# for that user. The model can therefore learn relative notions such as "this
# candidate has unusually attractive duration/author attributes within this
# user's available slate," which a pointwise CTR model cannot form.
# ---------------------------------------------------------------------------

SET_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "onehot_feat3",
]
set_cards = [int(FEATURE_CARDINALITIES[f]) for f in SET_FIELDS]
set_offsets = np.cumsum(
    [0] + set_cards[:-1], dtype=np.int64
)
set_total_card = int(sum(set_cards))


def make_offset_matrix(split):
    cols = []
    for field, card, offset in zip(
        SET_FIELDS, set_cards, set_offsets
    ):
        x = np.asarray(split.X[field], dtype=np.int64)
        if x.size and (x.min() < 0 or x.max() >= card):
            raise ValueError("Out-of-range ID for " + field)
        cols.append(x + offset)
    return np.ascontiguousarray(
        np.stack(cols, axis=1), dtype=np.int64
    )


xset_tr = make_offset_matrix(train)
xset_va = make_offset_matrix(valid)
xset_te = make_offset_matrix(test)


def build_slates(split, max_slate=MAX_SLATE):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)

    # Stable chronological ordering inside a user. The chunk number only
    # limits quadratic attention cost for unusually active train users.
    order = np.lexsort((rows, times, users))
    sorted_users = users[order]
    n = len(order)

    user_starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    user_ends = np.r_[user_starts[1:], n]
    user_lengths = user_ends - user_starts

    repeated_user_starts = np.repeat(user_starts, user_lengths)
    within_user_position = (
        np.arange(n, dtype=np.int64) - repeated_user_starts
    )
    chunk_number = within_user_position // max_slate

    new_chunk = np.r_[
        True,
        (sorted_users[1:] != sorted_users[:-1])
        | (chunk_number[1:] != chunk_number[:-1]),
    ]
    chunk_starts = np.flatnonzero(new_chunk)
    chunk_ends = np.r_[chunk_starts[1:], n]
    chunk_lengths = chunk_ends - chunk_starts

    slate_rows = np.full(
        (len(chunk_starts), max_slate), -1, dtype=np.int64
    )
    repeated_chunk_starts = np.repeat(
        chunk_starts, chunk_lengths
    )
    slots = np.arange(n, dtype=np.int64) - repeated_chunk_starts
    chunk_ids = np.repeat(
        np.arange(len(chunk_starts), dtype=np.int64),
        chunk_lengths,
    )
    slate_rows[chunk_ids, slots] = order
    return slate_rows


train_slates = build_slates(train)
valid_slates = build_slates(valid)
test_slates = build_slates(test)


class SlateTransformer(nn.Module):
    def __init__(self, total_card, n_fields, dim=12):
        super().__init__()
        self.embedding = nn.Embedding(total_card, dim)
        self.field_embedding = nn.Parameter(
            torch.empty(n_fields, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=3,
            dim_feedforward=48,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.local_tower = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.output = nn.Linear(dim + 16, 1)

        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.normal_(self.field_embedding, std=0.02)

    def forward(self, x, padding_mask):
        # Sum field-aware token components to form one representation per
        # candidate impression.
        embedded = self.embedding(x)
        embedded = embedded + self.field_embedding[None, None, :, :]
        token = self.input_norm(embedded.sum(dim=2))

        contextual = self.encoder(
            token, src_key_padding_mask=padding_mask
        )
        local = self.local_tower(token)
        logits = self.output(
            torch.cat([contextual, local], dim=-1)
        ).squeeze(-1)
        return logits


def train_slate_model(model, epochs=2, slate_batch=48):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0015, weight_decay=1e-5
    )
    model.train()

    for _ in range(epochs):
        slate_order = rng.permutation(len(train_slates))

        for lo in range(0, len(slate_order), slate_batch):
            ids = slate_order[lo:lo + slate_batch]
            rows = train_slates[ids]
            mask_np = rows < 0
            safe_rows = np.maximum(rows, 0)

            xb = torch.from_numpy(xset_tr[safe_rows])
            mask = torch.from_numpy(mask_np)
            target = torch.from_numpy(ytr[safe_rows])
            weight = torch.from_numpy(row_weight[safe_rows])
            valid_mask = ~mask

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, mask)

            point_loss = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            point_numerator = (
                point_loss * weight * valid_mask
            ).sum()
            point_denominator = (
                weight * valid_mask
            ).sum().clamp_min(1.0)
            point_loss = point_numerator / point_denominator

            # All positive-negative comparisons within each slate. This is
            # vectorized over slate batches and directly trains ordering.
            pair_mask = (
                (target.unsqueeze(2) > target.unsqueeze(1))
                & valid_mask.unsqueeze(2)
                & valid_mask.unsqueeze(1)
            )
            margins = logits.unsqueeze(2) - logits.unsqueeze(1)
            pair_loss_all = F.softplus(-margins)
            pair_weight = weight.unsqueeze(2)

            pair_denominator = (
                pair_weight * pair_mask
            ).sum().clamp_min(1.0)
            pair_loss = (
                pair_loss_all * pair_weight * pair_mask
            ).sum() / pair_denominator

            loss = pair_loss + 0.25 * point_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


@torch.inference_mode()
def predict_slates(model, x, slate_rows, slate_batch=64):
    model.eval()
    result = np.empty(len(x), dtype=np.float64)

    for lo in range(0, len(slate_rows), slate_batch):
        rows = slate_rows[lo:lo + slate_batch]
        mask_np = rows < 0
        safe_rows = np.maximum(rows, 0)

        xb = torch.from_numpy(x[safe_rows])
        mask = torch.from_numpy(mask_np)
        logits = model(xb, mask).cpu().numpy()

        good = ~mask_np
        result[rows[good]] = logits[good].astype(np.float64)

    return result


torch.manual_seed(SEED + 2)
slate_model = SlateTransformer(
    set_total_card, len(SET_FIELDS), dim=12
)
train_slate_model(slate_model)
slate_valid = predict_slates(
    slate_model, xset_va, valid_slates
)
slate_test = predict_slates(
    slate_model, xset_te, test_slates
)
del slate_model


# ---------------------------------------------------------------------------
# Compare raw families and rank aggregates with the trusted incumbent.
# Rank aggregation puts models with unrelated score scales onto a common
# within-user scale, exactly matching the invariances of both scored metrics.
# ---------------------------------------------------------------------------

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

raw_models = {
    "hard_negative_ffm": (ffm_valid, ffm_test),
    "setwise_transformer": (slate_valid, slate_test),
}

candidate_valid = {}
candidate_test = {}
candidate_raw = {}
candidate_scores = {}

for name, (va_raw, te_raw) in raw_models.items():
    candidate_valid[name] = va_raw
    candidate_test[name] = te_raw
    candidate_raw[name] = va_raw
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, va_raw)["primary"]
    )

    va_rank = within_user_rank(valid.user_id, va_raw)
    te_rank = within_user_rank(test.user_id, te_raw)

    # Validation selection of the combination weight is explicitly permitted
    # for the reusable trusted incumbent; the identical weight is applied to
    # test predictions.
    for alpha in (0.25, 0.50, 0.75):
        blend_name = "%s_incblend_%.2f" % (name, alpha)
        blend_va = (
            alpha * va_rank + (1.0 - alpha) * inc_valid_rank
        )
        blend_te = (
            alpha * te_rank + (1.0 - alpha) * inc_test_rank
        )
        candidate_valid[blend_name] = blend_va
        candidate_test[blend_name] = blend_te
        candidate_raw[blend_name] = va_raw
        candidate_scores[blend_name] = float(
            evaluate(
                valid.user_id, valid.y, blend_va
            )["primary"]
        )

# A genuinely multi-family Borda aggregate. Its "own model" score excludes the
# incumbent and is saved separately if this candidate wins.
ffm_va_rank = within_user_rank(valid.user_id, ffm_valid)
ffm_te_rank = within_user_rank(test.user_id, ffm_test)
slate_va_rank = within_user_rank(valid.user_id, slate_valid)
slate_te_rank = within_user_rank(test.user_id, slate_test)

own_ensemble_valid = 0.5 * (
    ffm_va_rank + slate_va_rank
)
own_ensemble_test = 0.5 * (
    ffm_te_rank + slate_te_rank
)

candidate_valid["two_family_borda"] = own_ensemble_valid
candidate_test["two_family_borda"] = own_ensemble_test
candidate_raw["two_family_borda"] = own_ensemble_valid
candidate_scores["two_family_borda"] = float(
    evaluate(
        valid.user_id, valid.y, own_ensemble_valid
    )["primary"]
)

for alpha in (0.25, 0.50, 0.75):
    name = "two_family_incblend_%.2f" % alpha
    va = (
        alpha * own_ensemble_valid
        + (1.0 - alpha) * inc_valid_rank
    )
    te = (
        alpha * own_ensemble_test
        + (1.0 - alpha) * inc_test_rank
    )
    candidate_valid[name] = va
    candidate_test[name] = te
    candidate_raw[name] = own_ensemble_valid
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, va)["primary"]
    )

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_valid[winner]
test_scores = candidate_test[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

    if "blend" in winner:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
        )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "winner": winner,
        "train_slates": int(len(train_slates)),
        "valid_slates": int(len(valid_slates)),
        "test_slates": int(len(test_slates)),
        "ffm_slate_rank_correlation": float(
            np.corrcoef(ffm_va_rank, slate_va_rank)[0, 1]
        ),
        "ffm_inc_rank_correlation": float(
            np.corrcoef(ffm_va_rank, inc_valid_rank)[0, 1]
        ),
        "slate_inc_rank_correlation": float(
            np.corrcoef(slate_va_rank, inc_valid_rank)[0, 1]
        ),
    }, sort_keys=True)
)

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, '
    '"ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)