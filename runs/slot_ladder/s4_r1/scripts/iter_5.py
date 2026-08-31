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
SEED = 2026
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
BATCH = 4096

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_matrix(split):
    cols = []
    for field, offset, card in zip(FIELDS, offsets, cards):
        x = np.asarray(split.X[field], dtype=np.int64)
        if x.size and (x.min() < 0 or x.max() >= card):
            raise ValueError("%s has an out-of-range ID" % field)
        cols.append(x + offset)
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int64)


xtr = make_matrix(train)
xva = make_matrix(valid)
xte = make_matrix(test)
ytr = np.asarray(train.y, dtype=np.float32)

# Fixed four-day half-life, motivated by the known continuation of temporal drift.
dates = np.asarray(train.date, dtype=np.int64)
sample_weight = np.exp2((dates - dates.max()).astype(np.float32) / 4.0)
sample_weight /= sample_weight.mean()
sample_weight = sample_weight.astype(np.float32)

rng = np.random.default_rng(SEED)


@torch.inference_mode()
def predict_model(model, x, output_index=None):
    model.eval()
    ans = np.empty(len(x), dtype=np.float64)
    for lo in range(0, len(x), 16384):
        hi = min(lo + 16384, len(x))
        z = model(torch.from_numpy(x[lo:hi]))
        if output_index is not None:
            z = z[:, output_index]
        ans[lo:hi] = z.detach().cpu().numpy().astype(np.float64)
    return ans


def train_pointwise(model, epochs, lr, multitask_targets=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(ytr)
    model.train()

    for _ in range(epochs):
        order = rng.permutation(n)
        for lo in range(0, n, BATCH):
            idx = order[lo:lo + BATCH]
            xb = torch.from_numpy(xtr[idx])
            wb = torch.from_numpy(sample_weight[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)

            if multitask_targets is None:
                target = torch.from_numpy(ytr[idx])
                loss_row = F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
            else:
                target = torch.from_numpy(multitask_targets[idx])
                task_loss = F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                # long_view remains dominant; auxiliary outcomes are training
                # targets only and are never supplied as prediction inputs.
                task_weights = torch.tensor(
                    [1.0, 0.30, 0.15], dtype=task_loss.dtype
                )
                loss_row = (task_loss * task_weights).sum(dim=1)

            loss = (loss_row * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()


class WeightedFM(nn.Module):
    def __init__(self, n_features, rank=20):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.latent = nn.Embedding(n_features, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.latent(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class LowRankCrossLayer(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.u = nn.Linear(dim, rank, bias=False)
        self.v = nn.Linear(rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        return x + x0 * (self.v(self.u(x)) + self.bias)


class DCNV2(nn.Module):
    def __init__(self, n_features, n_fields, emb_dim=12):
        super().__init__()
        self.embedding = nn.Embedding(n_features, emb_dim)
        dim = n_fields * emb_dim
        self.cross1 = LowRankCrossLayer(dim, 24)
        self.cross2 = LowRankCrossLayer(dim, 24)
        self.deep = nn.Sequential(
            nn.Linear(dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 48, 1)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        xc = self.cross1(x0, x0)
        xc = self.cross2(x0, xc)
        xd = self.deep(x0)
        return self.output(torch.cat([xc, xd], dim=1)).squeeze(1)


class MMoE(nn.Module):
    def __init__(self, n_features, n_fields, emb_dim=12,
                 n_experts=4, n_tasks=3):
        super().__init__()
        self.embedding = nn.Embedding(n_features, emb_dim)
        dim = n_fields * emb_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(dim, n_experts) for _ in range(n_tasks)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(n_tasks)
        ])
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        base = self.embedding(x).flatten(1)
        expert_values = torch.stack(
            [expert(base) for expert in self.experts], dim=1
        )
        outputs = []
        for gate, head in zip(self.gates, self.heads):
            weights = torch.softmax(gate(base), dim=1).unsqueeze(-1)
            representation = (weights * expert_values).sum(dim=1)
            outputs.append(head(representation).squeeze(1))
        return torch.stack(outputs, dim=1)


class PairwiseMF(nn.Module):
    def __init__(self, rank=32):
        super().__init__()
        self.user = nn.Embedding(cards[0], rank)
        self.video = nn.Embedding(cards[1], rank)
        self.author = nn.Embedding(cards[2], rank)
        self.video_bias = nn.Embedding(cards[1], 1)
        self.author_bias = nn.Embedding(cards[2], 1)
        self.tab_bias = nn.Embedding(cards[3], 1)
        self.duration_bias = nn.Embedding(cards[4], 1)
        self.scale = rank ** -0.5

        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.video.weight, std=0.05)
        nn.init.normal_(self.author.weight, std=0.05)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.author_bias.weight)
        nn.init.zeros_(self.tab_bias.weight)
        nn.init.zeros_(self.duration_bias.weight)

    def forward(self, x):
        # Convert offset IDs back to field-local IDs.
        u = x[:, 0] - int(offsets[0])
        v = x[:, 1] - int(offsets[1])
        a = x[:, 2] - int(offsets[2])
        tab = x[:, 3] - int(offsets[3])
        dur = x[:, 4] - int(offsets[4])

        item = self.video(v) + 0.5 * self.author(a)
        score = (self.user(u) * item).sum(dim=1) * self.scale
        score = score + self.video_bias(v).squeeze(1)
        score = score + self.author_bias(a).squeeze(1)
        score = score + self.tab_bias(tab).squeeze(1)
        score = score + self.duration_bias(dur).squeeze(1)
        return score


def train_pairwise(model, epochs=5, lr=0.008):
    user_local = np.asarray(train.X["user_id"], dtype=np.int64)
    negatives = np.flatnonzero(ytr == 0)
    neg_order = np.argsort(user_local[negatives], kind="stable")
    negatives = negatives[neg_order]
    neg_users = user_local[negatives]

    n_users = cards[0]
    neg_counts = np.bincount(neg_users, minlength=n_users).astype(np.int64)
    neg_starts = np.cumsum(
        np.r_[np.int64(0), neg_counts[:-1]], dtype=np.int64
    )

    positives = np.flatnonzero(ytr > 0.5)
    positives = positives[neg_counts[user_local[positives]] > 0]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        pos_order = rng.permutation(positives)
        pos_users = user_local[pos_order]
        count = neg_counts[pos_users]
        random_offset = (rng.random(len(pos_order)) * count).astype(np.int64)
        neg_idx = negatives[neg_starts[pos_users] + random_offset]

        for lo in range(0, len(pos_order), BATCH):
            hi = min(lo + BATCH, len(pos_order))
            pi = pos_order[lo:hi]
            ni = neg_idx[lo:hi]

            xp = torch.from_numpy(xtr[pi])
            xn = torch.from_numpy(xtr[ni])
            wb = torch.from_numpy(sample_weight[pi])

            optimizer.zero_grad(set_to_none=True)
            margin = model(xp) - model(xn)
            loss_row = F.softplus(-margin)
            loss = (loss_row * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    boundary = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    group_start_for_row = np.repeat(starts, lengths)
    group_length_for_row = np.repeat(lengths, lengths)
    position = np.arange(n, dtype=np.int64) - group_start_for_row

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    multi = group_length_for_row > 1
    ranked_sorted[multi] = (
        position[multi] / (group_length_for_row[multi] - 1.0)
    )

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


valid_predictions = {}
test_predictions = {}

# Family 1: temporally weighted factorization.
fm = WeightedFM(total_cardinality, rank=20)
train_pointwise(fm, epochs=5, lr=0.001)
valid_predictions["weighted_fm"] = predict_model(fm, xva)
test_predictions["weighted_fm"] = predict_model(fm, xte)
del fm

# Family 2: explicit low-rank cross network plus nonlinear tower.
torch.manual_seed(SEED + 1)
dcn = DCNV2(total_cardinality, len(FIELDS))
train_pointwise(dcn, epochs=3, lr=0.0015)
valid_predictions["dcnv2"] = predict_model(dcn, xva)
test_predictions["dcnv2"] = predict_model(dcn, xte)
del dcn

# Family 3: multi-task mixture of experts. Auxiliary signals are targets only.
aux_targets = np.stack([
    ytr,
    np.asarray(train.aux["is_click"], dtype=np.float32),
    np.asarray(train.aux["is_like"], dtype=np.float32),
], axis=1)
torch.manual_seed(SEED + 2)
mmoe = MMoE(total_cardinality, len(FIELDS))
train_pointwise(mmoe, epochs=3, lr=0.0015,
                multitask_targets=aux_targets)
valid_predictions["mmoe"] = predict_model(mmoe, xva, output_index=0)
test_predictions["mmoe"] = predict_model(mmoe, xte, output_index=0)
del mmoe, aux_targets

# Family 4: latent pairwise model trained directly on positive-negative
# comparisons among each user's logged impressions.
torch.manual_seed(SEED + 3)
bpr = PairwiseMF(rank=32)
train_pairwise(bpr, epochs=5, lr=0.008)
valid_predictions["pairwise_bpr"] = predict_model(bpr, xva)
test_predictions["pairwise_bpr"] = predict_model(bpr, xte)
del bpr

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(os.path.join(shared, "incumbent_valid_scores.npy"))
inc_test = np.load(os.path.join(shared, "incumbent_test_scores.npy"))

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_scores = {}
candidate_valid_arrays = {}
candidate_test_arrays = {}
candidate_raw_arrays = {}

for name in valid_predictions:
    raw_valid = valid_predictions[name]
    raw_test = test_predictions[name]

    raw_metrics = evaluate(valid.user_id, valid.y, raw_valid)
    candidate_scores[name] = float(raw_metrics["primary"])
    candidate_valid_arrays[name] = raw_valid
    candidate_test_arrays[name] = raw_test
    candidate_raw_arrays[name] = raw_valid

    # Fixed equal-weight rank aggregation avoids choosing a scale or weight
    # using validation and matches the fact that every metric is rank-only.
    own_valid_rank = within_user_rank(valid.user_id, raw_valid)
    own_test_rank = within_user_rank(test.user_id, raw_test)
    blend_valid = 0.5 * inc_valid_rank + 0.5 * own_valid_rank
    blend_test = 0.5 * inc_test_rank + 0.5 * own_test_rank

    blend_name = name + "_rankblend"
    blend_metrics = evaluate(valid.user_id, valid.y, blend_valid)
    candidate_scores[blend_name] = float(blend_metrics["primary"])
    candidate_valid_arrays[blend_name] = blend_valid
    candidate_test_arrays[blend_name] = blend_test
    candidate_raw_arrays[blend_name] = raw_valid

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_valid_arrays[winner]
test_scores = candidate_test_arrays[winner]
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
    if winner.endswith("_rankblend"):
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw_arrays[winner], dtype=np.float64),
        )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print("FINDINGS winner=%s fixed_rank_blend=%s" % (
    winner, str(winner.endswith("_rankblend")).lower()
))

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        elapsed,
    )
)