import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7321
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour"
]
EMBED_DIM = 20
EPOCHS = 5
BATCH_SIZE = 4096
PRED_BATCH = 32768
LR = 0.004
LISTWISE_NEGATIVES = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

cards = np.asarray([int(FEATURE_CARDINALITIES[f]) for f in FIELDS], dtype=np.int64)
offsets = np.cumsum(np.r_[0, cards[:-1]], dtype=np.int64)
TOTAL_CARDINALITY = int(cards.sum())
USER_CARDINALITY = int(FEATURE_CARDINALITIES["user_id"])


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) + offsets[j]
            for j, f in enumerate(FIELDS)
        ]),
        dtype=np.int64
    )


def prepare_pair_sampler(X, y):
    y = np.asarray(y, dtype=np.int8)
    user_local = X[:, 0] - offsets[0]

    neg_rows = np.flatnonzero(y == 0)
    neg_users = user_local[neg_rows]
    neg_order = np.argsort(neg_users, kind="stable")
    neg_rows_sorted = neg_rows[neg_order]
    neg_users_sorted = neg_users[neg_order]

    neg_counts = np.bincount(
        neg_users_sorted, minlength=USER_CARDINALITY
    ).astype(np.int64)
    neg_starts = np.zeros(USER_CARDINALITY, dtype=np.int64)
    if USER_CARDINALITY > 1:
        neg_starts[1:] = np.cumsum(neg_counts[:-1], dtype=np.int64)

    pos_rows = np.flatnonzero(y == 1)
    pos_users = user_local[pos_rows]
    usable = neg_counts[pos_users] > 0
    pos_rows = pos_rows[usable]
    pos_users = pos_users[usable]

    return pos_rows, pos_users, neg_rows_sorted, neg_starts, neg_counts


def sample_negative_rows(rng, pos_users, neg_rows_sorted, neg_starts,
                         neg_counts, n_negatives):
    counts = neg_counts[pos_users]
    random_fraction = rng.random(
        (len(pos_users), n_negatives), dtype=np.float32
    )
    offsets_in_group = np.minimum(
        (random_fraction * counts[:, None]).astype(np.int64),
        counts[:, None] - 1
    )
    locations = neg_starts[pos_users, None] + offsets_in_group
    return neg_rows_sorted[locations]


class PairwiseFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)
        return linear + interaction


class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.context_gate = nn.Parameter(torch.ones(len(FIELDS) - 1))
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        latent = self.embedding(x)
        user_vector = latent[:, 0, :]
        context = latent[:, 1:, :]
        gates = torch.softmax(self.context_gate, dim=0)
        context_vector = (context * gates[None, :, None]).sum(dim=1)
        dot = (user_vector * context_vector).sum(dim=1)
        linear = self.linear(x[:, 1:]).sum(dim=1).squeeze(1)
        return dot + linear


@torch.inference_mode()
def predict(model, X):
    model.eval()
    result = np.empty(len(X), dtype=np.float32)
    xt = torch.from_numpy(X)
    for start in range(0, len(X), PRED_BATCH):
        end = min(start + PRED_BATCH, len(X))
        result[start:end] = model(xt[start:end]).cpu().numpy()
    return result


def fit_bpr_fm(X, y):
    torch.manual_seed(SEED + 11)
    model = PairwiseFM()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=2e-6
    )
    sampler = prepare_pair_sampler(X, y)
    pos_rows, pos_users, neg_rows_sorted, neg_starts, neg_counts = sampler
    rng = np.random.default_rng(SEED + 101)
    xt = torch.from_numpy(X)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(pos_rows))
        ordered_pos = pos_rows[order]
        ordered_users = pos_users[order]
        neg = sample_negative_rows(
            rng, ordered_users, neg_rows_sorted, neg_starts, neg_counts, 1
        )[:, 0]

        model.train()
        for start in range(0, len(ordered_pos), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(ordered_pos))
            pi = torch.from_numpy(ordered_pos[start:end])
            ni = torch.from_numpy(neg[start:end])
            margin = model(xt[pi]) - model(xt[ni])
            loss = torch.nn.functional.softplus(-margin).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def fit_listwise_tower(X, y):
    torch.manual_seed(SEED + 23)
    model = TwoTower()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=2e-6
    )
    sampler = prepare_pair_sampler(X, y)
    pos_rows, pos_users, neg_rows_sorted, neg_starts, neg_counts = sampler
    rng = np.random.default_rng(SEED + 211)
    xt = torch.from_numpy(X)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(pos_rows))
        ordered_pos = pos_rows[order]
        ordered_users = pos_users[order]
        negatives = sample_negative_rows(
            rng, ordered_users, neg_rows_sorted, neg_starts,
            neg_counts, LISTWISE_NEGATIVES
        )

        model.train()
        for start in range(0, len(ordered_pos), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(ordered_pos))
            pos_idx = torch.from_numpy(ordered_pos[start:end])
            neg_idx = torch.from_numpy(negatives[start:end])

            positive_score = model(xt[pos_idx])[:, None]
            flat_negative = neg_idx.reshape(-1)
            negative_score = model(xt[flat_negative]).reshape(
                end - start, LISTWISE_NEGATIVES
            )
            logits = torch.cat([positive_score, negative_score], dim=1)
            target = torch.zeros(end - start, dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(logits, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - float(x.mean())) / sd


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((scores, user_ids))
    sorted_users = user_ids[order]
    boundaries = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    group_start = np.maximum.accumulate(
        np.where(boundaries, np.arange(n, dtype=np.int64), 0)
    )
    position = np.arange(n, dtype=np.int64) - group_start

    next_boundary = np.r_[
        np.flatnonzero(boundaries)[1:],
        n
    ]
    group_lengths = np.repeat(
        next_boundary - np.flatnonzero(boundaries),
        next_boundary - np.flatnonzero(boundaries)
    )
    ranked_sorted = position / np.maximum(group_lengths - 1, 1)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def candidate_variants(name, new_scores, incumbent_scores, users):
    variants = []
    new_z = zscore(new_scores)
    inc_z = zscore(incumbent_scores)
    new_rank = within_user_rank(users, new_scores)
    inc_rank = within_user_rank(users, incumbent_scores)

    variants.append((name + "_standalone", np.asarray(new_scores, dtype=np.float64),
                     ("standalone", 0.0)))

    for alpha in (0.25, 0.50, 0.75):
        variants.append((
            name + "_zblend_inc%.2f" % alpha,
            alpha * inc_z + (1.0 - alpha) * new_z,
            ("zblend", alpha)
        ))
        variants.append((
            name + "_rankblend_inc%.2f" % alpha,
            alpha * inc_rank + (1.0 - alpha) * new_rank,
            ("rankblend", alpha)
        ))
    return variants


train = load("train")
valid = load("valid")
X_train = make_matrix(train)
X_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

bpr_model = fit_bpr_fm(X_train, y_train)
bpr_valid = predict(bpr_model, X_valid).astype(np.float64)

list_model = fit_listwise_tower(X_train, y_train)
list_valid = predict(list_model, X_valid).astype(np.float64)

all_candidates = [
    ("trusted_pointwise_fm", inc_valid, ("incumbent", None, None))
]
for name, scores, specification in candidate_variants(
        "bpr_fm", bpr_valid, inc_valid, valid_users):
    all_candidates.append(
        (name, scores, ("bpr", specification[0], specification[1]))
    )
for name, scores, specification in candidate_variants(
        "listwise_tower", list_valid, inc_valid, valid_users):
    all_candidates.append(
        (name, scores, ("listwise", specification[0], specification[1]))
    )

candidate_scores = {}
best_name = None
best_scores = None
best_spec = None
best_primary = -np.inf

for name, scores, spec in all_candidates:
    result = evaluate(valid_users, y_valid, scores)
    primary = float(result["primary"])
    candidate_scores[name] = primary
    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_scores = np.asarray(scores, dtype=np.float64)
        best_spec = spec

metrics = evaluate(valid_users, y_valid, best_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": best_name,
    "bpr_fm_raw": candidate_scores["bpr_fm_standalone"],
    "listwise_tower_raw": candidate_scores["listwise_tower_standalone"],
    "trusted_pointwise_fm": candidate_scores["trusted_pointwise_fm"]
}, sort_keys=True))

# Refit the selected new family on train + validation. If the trusted incumbent
# wins outright, its already-published test predictions are the exact recipe.
test = load("test")
test_users = np.asarray(test.user_id)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if best_spec[0] == "incumbent":
    test_scores = inc_test.copy()
else:
    X_combined = np.concatenate([X_train, X_valid], axis=0)
    y_combined = np.concatenate([y_train, y_valid], axis=0)
    X_test = make_matrix(test)

    del bpr_model, list_model, X_train, X_valid

    family, transform, alpha = best_spec
    if family == "bpr":
        final_model = fit_bpr_fm(X_combined, y_combined)
    else:
        final_model = fit_listwise_tower(X_combined, y_combined)

    new_test = predict(final_model, X_test).astype(np.float64)

    if transform == "standalone":
        test_scores = new_test
    elif transform == "zblend":
        test_scores = (
            float(alpha) * zscore(inc_test)
            + (1.0 - float(alpha)) * zscore(new_test)
        )
    elif transform == "rankblend":
        test_scores = (
            float(alpha) * within_user_rank(test_users, inc_test)
            + (1.0 - float(alpha)) * within_user_rank(test_users, new_test)
        )
    else:
        raise RuntimeError("Unknown selected transformation")

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