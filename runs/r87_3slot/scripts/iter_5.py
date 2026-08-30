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
SEED = 7319
THREADS = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(THREADS)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

LATENT_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
]
DIM = 24
EPOCHS = 4
BATCH = 8192


def concat_field(parts, field):
    if len(parts) == 1:
        return np.asarray(parts[0].X[field], dtype=np.int64)
    return np.concatenate([
        np.asarray(p.X[field], dtype=np.int64) for p in parts
    ]).astype(np.int64, copy=False)


class PairwiseLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.user = nn.Embedding(
            int(FEATURE_CARDINALITIES["user_id"]), DIM, sparse=True
        )
        self.entities = nn.ModuleDict({
            f: nn.Embedding(
                int(FEATURE_CARDINALITIES[f]), DIM, sparse=True
            )
            for f in LATENT_FIELDS
        })
        self.biases = nn.ModuleDict({
            f: nn.Embedding(
                int(FEATURE_CARDINALITIES[f]), 1, sparse=True
            )
            for f in LATENT_FIELDS
        })

        std = 0.04 / np.sqrt(DIM)
        nn.init.normal_(self.user.weight, std=std)
        for emb in self.entities.values():
            nn.init.normal_(emb.weight, std=std)
        for emb in self.biases.values():
            nn.init.zeros_(emb.weight)

    def score(self, users, fields):
        u = self.user(users)
        candidate = torch.zeros_like(u)
        bias = torch.zeros(len(users), dtype=u.dtype, device=u.device)
        for f in LATENT_FIELDS:
            candidate = candidate + self.entities[f](fields[f])
            bias = bias + self.biases[f](fields[f]).squeeze(-1)
        return (u * candidate).sum(dim=1) + bias


def make_arrays(parts):
    arrays = {
        "user_id": concat_field(parts, "user_id"),
    }
    for f in LATENT_FIELDS:
        arrays[f] = concat_field(parts, f)
    return arrays


def prepare_user_sampling(users):
    users = np.asarray(users, dtype=np.int64)
    order = np.argsort(users, kind="stable")
    card = int(FEATURE_CARDINALITIES["user_id"])
    counts = np.bincount(users, minlength=card).astype(np.int64)
    starts = np.zeros(card, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    return order, starts, counts


def sample_same_user_negatives(pos_rows, users, labels, order, starts,
                               counts, rng):
    pu = users[pos_rows]
    offsets = (rng.random(len(pos_rows)) * counts[pu]).astype(np.int64)
    neg_rows = order[starts[pu] + offsets]

    for _ in range(8):
        bad = labels[neg_rows] != 0
        if not np.any(bad):
            break
        ub = pu[bad]
        offsets = (rng.random(np.sum(bad)) * counts[ub]).astype(np.int64)
        neg_rows[bad] = order[starts[ub] + offsets]

    good = labels[neg_rows] == 0
    return pos_rows[good], neg_rows[good]


def fit_bpr(parts, labels, epochs, seed):
    arrays = make_arrays(parts)
    labels = np.asarray(labels, dtype=np.int8)
    users = arrays["user_id"]
    positive_rows = np.flatnonzero(labels == 1).astype(np.int64)

    order, starts, counts = prepare_user_sampling(users)
    model = PairwiseLatent()
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.018)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        shuffled = positive_rows[rng.permutation(len(positive_rows))]
        pos, neg = sample_same_user_negatives(
            shuffled, users, labels, order, starts, counts, rng
        )

        model.train()
        for st in range(0, len(pos), BATCH):
            en = min(st + BATCH, len(pos))
            p = pos[st:en]
            n = neg[st:en]

            tu = torch.from_numpy(users[p])
            pf = {
                f: torch.from_numpy(arrays[f][p])
                for f in LATENT_FIELDS
            }
            nf = {
                f: torch.from_numpy(arrays[f][n])
                for f in LATENT_FIELDS
            }

            optimizer.zero_grad(set_to_none=True)
            positive_score = model.score(tu, pf)
            negative_score = model.score(tu, nf)
            difference = positive_score - negative_score

            # The small squared-margin term prevents a handful of easy pairs
            # from developing extreme scores without densifying gradients.
            loss = torch.nn.functional.softplus(-difference).mean()
            loss = loss + 2e-5 * torch.square(difference).mean()
            loss.backward()
            optimizer.step()

    return model


def predict_bpr(model, split):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    fields = {
        f: np.asarray(split.X[f], dtype=np.int64)
        for f in LATENT_FIELDS
    }
    result = np.empty(len(users), dtype=np.float64)
    model.eval()

    with torch.no_grad():
        for st in range(0, len(users), BATCH * 2):
            en = min(st + BATCH * 2, len(users))
            tu = torch.from_numpy(users[st:en])
            tf = {
                f: torch.from_numpy(fields[f][st:en])
                for f in LATENT_FIELDS
            }
            result[st:en] = model.score(tu, tf).cpu().numpy()
    return result


def clipped_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


class SmoothedTable:
    def __init__(self, keys, labels, prior, strength):
        keys = np.asarray(keys, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)
        unique, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )
        sums = np.bincount(
            inverse, weights=labels, minlength=len(unique)
        )
        rate = (sums + strength * prior) / (counts + strength)
        self.keys = unique
        self.values = clipped_logit(rate)
        self.default = float(clipped_logit(prior))

    def lookup(self, keys):
        keys = np.asarray(keys, dtype=np.int64)
        idx = np.searchsorted(self.keys, keys)
        valid = idx < len(self.keys)
        safe = np.minimum(idx, len(self.keys) - 1)
        valid &= self.keys[safe] == keys
        result = np.full(len(keys), self.default, dtype=np.float64)
        result[valid] = self.values[safe[valid]]
        return result


class EmpiricalBayesModel:
    def __init__(self, parts, labels):
        labels = np.asarray(labels, dtype=np.int8)
        self.prior = float(np.mean(labels))
        self.author_card = int(FEATURE_CARDINALITIES["author_id"])

        user = concat_field(parts, "user_id")
        video = concat_field(parts, "video_id")
        author = concat_field(parts, "author_id")
        tag = concat_field(parts, "tag")
        tab = concat_field(parts, "tab")
        ua = user * np.int64(self.author_card) + author

        self.video = SmoothedTable(video, labels, self.prior, 18.0)
        self.author = SmoothedTable(author, labels, self.prior, 35.0)
        self.tag = SmoothedTable(tag, labels, self.prior, 90.0)
        self.tab = SmoothedTable(tab, labels, self.prior, 120.0)
        self.user_author = SmoothedTable(ua, labels, self.prior, 9.0)

    def components(self, split):
        user = np.asarray(split.X["user_id"], dtype=np.int64)
        video = np.asarray(split.X["video_id"], dtype=np.int64)
        author = np.asarray(split.X["author_id"], dtype=np.int64)
        tag = np.asarray(split.X["tag"], dtype=np.int64)
        tab = np.asarray(split.X["tab"], dtype=np.int64)
        ua = user * np.int64(self.author_card) + author

        global_logit = float(clipped_logit(self.prior))
        video_score = self.video.lookup(video)
        author_score = self.author.lookup(author)
        tag_score = self.tag.lookup(tag)
        tab_score = self.tab.lookup(tab)
        ua_score = self.user_author.lookup(ua)

        item_score = (
            video_score
            + 0.45 * (author_score - global_logit)
            + 0.15 * (tag_score - global_logit)
            + 0.10 * (tab_score - global_logit)
        )
        personalized_score = (
            0.55 * item_score
            + 0.45 * ua_score
        )
        return item_score, personalized_score


def metric(users, labels, scores):
    return evaluate(
        users, labels, np.asarray(scores, dtype=np.float64)
    )


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

# Family 1: pairwise latent matrix factorization.
bpr_model = fit_bpr([train], y_train, EPOCHS, SEED)
pred_bpr = predict_bpr(bpr_model, valid)

# Family 2: non-parametric empirical-Bayes item and personalized statistics.
eb_model = EmpiricalBayesModel([train], y_train)
pred_eb_item, pred_eb_personal = eb_model.components(valid)

inc_std = max(float(np.std(inc_valid)), 1e-8)
bpr_scale = inc_std / max(float(np.std(pred_bpr)), 1e-8)
item_scale = inc_std / max(float(np.std(pred_eb_item)), 1e-8)
personal_scale = inc_std / max(float(np.std(pred_eb_personal)), 1e-8)

# Family 3 is a heterogeneous latent/non-parametric ensemble, formed before
# considering the trusted incumbent.
pred_bpr_eb = (
    0.55 * bpr_scale * pred_bpr
    + 0.45 * personal_scale * pred_eb_personal
)
ensemble_scale = inc_std / max(float(np.std(pred_bpr_eb)), 1e-8)

raw_families = {
    "bpr_latent": pred_bpr,
    "eb_item": pred_eb_item,
    "eb_user_author": pred_eb_personal,
    "bpr_eb_ensemble": pred_bpr_eb,
}
family_scales = {
    "bpr_latent": bpr_scale,
    "eb_item": item_scale,
    "eb_user_author": personal_scale,
    "bpr_eb_ensemble": ensemble_scale,
}

candidate_scores = {}
inc_metrics = metric(valid.user_id, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])

best_name = "incumbent"
best_family = "incumbent"
best_alpha = 0.0
best_scale = 1.0
best_scores = inc_valid.copy()
best_raw = None
best_metrics = inc_metrics

alphas = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85]

for family, raw in raw_families.items():
    raw_metrics = metric(valid.user_id, y_valid, raw)
    candidate_scores[family] = float(raw_metrics["primary"])

    if raw_metrics["primary"] > best_metrics["primary"]:
        best_name = family
        best_family = family
        best_alpha = 1.0
        best_scale = 1.0
        best_scores = raw.copy()
        best_raw = raw.copy()
        best_metrics = raw_metrics

    scale = family_scales[family]
    scaled = scale * raw
    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid + alpha * scaled
        blended_metrics = metric(valid.user_id, y_valid, blended)
        name = "%s_blend_%.2f" % (family, alpha)
        candidate_scores[name] = float(blended_metrics["primary"])

        if blended_metrics["primary"] > best_metrics["primary"]:
            best_name = name
            best_family = family
            best_alpha = alpha
            best_scale = scale
            best_scores = blended.copy()
            best_raw = raw.copy()
            best_metrics = blended_metrics

print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS selected=%s bpr=%.6f eb_item=%.6f eb_user_author=%.6f "
    "bpr_eb=%.6f incumbent=%.6f"
    % (
        best_name,
        candidate_scores["bpr_latent"],
        candidate_scores["eb_item"],
        candidate_scores["eb_user_author"],
        candidate_scores["bpr_eb_ensemble"],
        candidate_scores["incumbent"],
    ),
    flush=True,
)

# Refit exactly the selected recipe using train+validation, then score test.
test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)
y_tv = np.concatenate([y_train, y_valid]).astype(np.int8)

if best_family == "incumbent":
    test_scores = inc_test.copy()
else:
    test_bpr = None
    test_eb_item = None
    test_eb_personal = None

    if best_family in ("bpr_latent", "bpr_eb_ensemble"):
        del bpr_model
        gc.collect()
        bpr_refit = fit_bpr([train, valid], y_tv, EPOCHS, SEED)
        test_bpr = predict_bpr(bpr_refit, test)
        del bpr_refit
        gc.collect()

    if best_family in (
        "eb_item", "eb_user_author", "bpr_eb_ensemble"
    ):
        eb_refit = EmpiricalBayesModel([train, valid], y_tv)
        test_eb_item, test_eb_personal = eb_refit.components(test)
        del eb_refit
        gc.collect()

    if best_family == "bpr_latent":
        test_raw = test_bpr
    elif best_family == "eb_item":
        test_raw = test_eb_item
    elif best_family == "eb_user_author":
        test_raw = test_eb_personal
    else:
        test_raw = (
            0.55 * bpr_scale * test_bpr
            + 0.45 * personal_scale * test_eb_personal
        )

    if best_alpha >= 1.0:
        test_scores = test_raw
    else:
        test_scores = (
            (1.0 - best_alpha) * inc_test
            + best_alpha * best_scale * test_raw
        )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_family != "incumbent" and best_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        best_metrics["primary"],
        best_metrics["gauc"],
        best_metrics["ndcg@5"],
        elapsed,
    )
)