import os
import gc
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")
START = time.time()
SEED = 7319
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")
if OUT:
    os.makedirs(OUT, exist_ok=True)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "music_type",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
HALF_LIFE = 4.0
BATCH_SIZE = 32768
PRED_BATCH = 131072

offsets = np.cumsum(
    [0] + [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS[:-1]]
).astype(np.int64)
TOTAL_CARD = int(sum(int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS))


def finite32(x):
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def choose_history_keys(d):
    preferred_terms = (
        "long_view_rate",
        "count_log1p",
        "is_click_rate",
        "play_time_ms_logmean",
        "comment_stay_time_logmean",
    )
    keys = [
        k for k in sorted(d)
        if any(term in k.lower() for term in preferred_terms)
    ]
    if not keys:
        keys = sorted(d)[:5]
    return keys[:5]


def build_cat(split):
    return np.column_stack([
        np.asarray(split.X[f], dtype=np.int32) + int(off)
        for f, off in zip(CAT_FIELDS, offsets)
    ]).astype(np.int32, copy=False)


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
n_train = len(train.user_id)

hv_train = historical_features("train", key="video_id")
ha_train = historical_features("train", key="author_id")
VIDEO_KEYS = choose_history_keys(hv_train)
AUTHOR_KEYS = choose_history_keys(ha_train)


def build_num(split, hv, ha):
    cols = []
    for field in NUM_FIELDS:
        z = np.maximum(finite32(split.num[field]), 0.0)
        cols.append(np.log1p(z).astype(np.float32))
    for key in VIDEO_KEYS:
        cols.append(finite32(hv[key]))
    for key in AUTHOR_KEYS:
        cols.append(finite32(ha[key]))

    # Causal context available without row outcomes: position in the user's
    # chronological stream, position within day, and feed-batch size.
    uid = np.asarray(split.user_id)
    tm = np.asarray(split.time_ms)
    date = np.asarray(split.date)
    row = np.arange(len(uid), dtype=np.int64)
    order = np.lexsort((row, tm, uid))
    su = uid[order]
    sd = date[order]
    st = tm[order]

    new_user = np.empty(len(uid), dtype=bool)
    new_user[0] = True
    new_user[1:] = su[1:] != su[:-1]
    user_start = np.maximum.accumulate(
        np.where(new_user, np.arange(len(uid)), 0)
    )
    user_pos_sorted = np.arange(len(uid)) - user_start

    new_day = np.empty(len(uid), dtype=bool)
    new_day[0] = True
    new_day[1:] = (su[1:] != su[:-1]) | (sd[1:] != sd[:-1])
    day_start = np.maximum.accumulate(
        np.where(new_day, np.arange(len(uid)), 0)
    )
    day_pos_sorted = np.arange(len(uid)) - day_start

    new_batch = np.empty(len(uid), dtype=bool)
    new_batch[0] = True
    new_batch[1:] = (su[1:] != su[:-1]) | (st[1:] != st[:-1])
    batch_group = np.cumsum(new_batch, dtype=np.int64) - 1
    batch_sizes = np.bincount(batch_group).astype(np.float32)
    batch_size_sorted = batch_sizes[batch_group]

    user_pos = np.empty(len(uid), dtype=np.float32)
    day_pos = np.empty(len(uid), dtype=np.float32)
    batch_size = np.empty(len(uid), dtype=np.float32)
    user_pos[order] = np.log1p(user_pos_sorted).astype(np.float32)
    day_pos[order] = np.log1p(day_pos_sorted).astype(np.float32)
    batch_size[order] = np.log1p(batch_size_sorted).astype(np.float32)

    cols.extend([user_pos, day_pos, batch_size])
    return np.column_stack(cols).astype(np.float32, copy=False)


cat_train = build_cat(train)
num_train = build_num(train, hv_train, ha_train)
del hv_train, ha_train
gc.collect()

num_mean = np.mean(num_train, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.std(num_train, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.maximum(num_std, 1e-3)
num_train = np.clip(
    (num_train - num_mean) / num_std, -8.0, 8.0
).astype(np.float32)

max_date = int(np.max(train.date))
age = (
    max_date - np.asarray(train.date, dtype=np.int32)
).astype(np.float32)
sample_weight = np.power(0.5, age / HALF_LIFE).astype(np.float32)
sample_weight /= np.mean(sample_weight)

aux_keys = set(train.aux.keys())
click_key = "is_click" if "is_click" in aux_keys else None
like_key = "is_like" if "is_like" in aux_keys else None
click_y = (
    np.asarray(train.aux[click_key], dtype=np.float32)
    if click_key else train_y.copy()
)
like_y = (
    np.asarray(train.aux[like_key], dtype=np.float32)
    if like_key else train_y.copy()
)
click_y = np.clip(click_y, 0.0, 1.0)
like_y = np.clip(like_y, 0.0, 1.0)

print(
    "FINDINGS auxiliary_targets click=%s like=%s history_video=%s history_author=%s"
    % (
        str(click_key),
        str(like_key),
        ",".join(VIDEO_KEYS),
        ",".join(AUTHOR_KEYS),
    ),
    flush=True,
)


class MMoE(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank=12, n_experts=4):
        super().__init__()
        self.n_fields = n_fields
        self.rank = rank
        self.emb = nn.Embedding(total_card, rank, sparse=True)
        nn.init.normal_(self.emb.weight, std=0.018)

        dim = n_fields * rank + n_num
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(dim, n_experts) for _ in range(3)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(3)
        ])

    def forward(self, cat, num):
        e = self.emb(cat).reshape(cat.shape[0], -1)
        x = torch.cat([e, num], dim=1)
        expert = torch.stack([m(x) for m in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(x), dim=1).unsqueeze(2)
            mixed = torch.sum(weights * expert, dim=1)
            outputs.append(tower(mixed).squeeze(1))
        return torch.stack(outputs, dim=1)


class PairwiseScorer(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank=14):
        super().__init__()
        self.emb = nn.Embedding(total_card, rank, sparse=True)
        nn.init.normal_(self.emb.weight, std=0.018)
        dim = n_fields * rank + n_num
        self.scorer = nn.Sequential(
            nn.Linear(dim, 192),
            nn.SiLU(),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )

    def forward(self, cat, num):
        e = self.emb(cat).reshape(cat.shape[0], -1)
        return self.scorer(torch.cat([e, num], dim=1)).squeeze(1)


def optimizers(model, dense_lr):
    sparse = torch.optim.SparseAdam([model.emb.weight], lr=0.004)
    dense = [
        p for name, p in model.named_parameters()
        if name != "emb.weight"
    ]
    opt = torch.optim.AdamW(dense, lr=dense_lr, weight_decay=1e-5)
    return sparse, opt, dense


# Family 1: MMoE with long-view, click and like towers.
mmoe = MMoE(
    TOTAL_CARD, len(CAT_FIELDS), num_train.shape[1],
    rank=12, n_experts=4
)
m_sparse, m_dense, m_dense_params = optimizers(mmoe, 0.0018)
generator = torch.Generator().manual_seed(SEED)

mmoe.train()
for epoch in range(2):
    perm = torch.randperm(n_train, generator=generator).numpy()
    running = 0.0
    seen = 0
    for start in range(0, n_train, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        cat = torch.from_numpy(
            cat_train[idx].astype(np.int64, copy=False)
        )
        num = torch.from_numpy(num_train[idx])
        targets = torch.from_numpy(np.column_stack([
            train_y[idx], click_y[idx], like_y[idx]
        ]).astype(np.float32, copy=False))
        wb = torch.from_numpy(sample_weight[idx])

        m_sparse.zero_grad(set_to_none=True)
        m_dense.zero_grad(set_to_none=True)
        logits = mmoe(cat, num)
        losses = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        row_loss = (
            losses[:, 0]
            + 0.18 * losses[:, 1]
            + 0.07 * losses[:, 2]
        )
        loss = torch.sum(row_loss * wb) / torch.sum(wb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_dense_params, 5.0)
        m_sparse.step()
        m_dense.step()
        running += float(loss.detach()) * len(idx)
        seen += len(idx)

    print(
        "mmoe_epoch=%d weighted_loss=%.6f"
        % (epoch + 1, running / max(seen, 1)),
        flush=True,
    )
    del perm
    gc.collect()


# Family 2: pairwise neural ranker. Construct one temporally weighted
# positive-negative comparison per positive within the same user.
uid_train = np.asarray(train.user_id)
sort_idx = np.argsort(uid_train, kind="stable")
sorted_uid = uid_train[sort_idx]
cuts = np.r_[
    0,
    np.flatnonzero(sorted_uid[1:] != sorted_uid[:-1]) + 1,
    len(sort_idx),
]

positive_parts = []
negative_parts = []
for left, right in zip(cuts[:-1], cuts[1:]):
    rows = sort_idx[left:right]
    pos = rows[train_y[rows] > 0.5]
    neg = rows[train_y[rows] < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        continue

    # Recent positives are naturally emphasized by weighted resampling.
    pw = sample_weight[pos].astype(np.float64)
    pw /= pw.sum()
    take = min(len(pos), 25000)
    if len(pos) > take:
        pos = rng.choice(pos, size=take, replace=False, p=pw)
    sampled_neg = rng.choice(neg, size=len(pos), replace=True)
    positive_parts.append(pos.astype(np.int64))
    negative_parts.append(sampled_neg.astype(np.int64))

pos_idx = np.concatenate(positive_parts)
neg_idx = np.concatenate(negative_parts)
del positive_parts, negative_parts, sort_idx, sorted_uid, cuts
gc.collect()

# Cap only after user-stratified construction, retaining broad user coverage.
MAX_PAIRS = 1800000
if len(pos_idx) > MAX_PAIRS:
    keep_prob = sample_weight[pos_idx].astype(np.float64)
    keep_prob /= keep_prob.sum()
    chosen = rng.choice(
        len(pos_idx), size=MAX_PAIRS, replace=False, p=keep_prob
    )
    pos_idx = pos_idx[chosen]
    neg_idx = neg_idx[chosen]

pair_model = PairwiseScorer(
    TOTAL_CARD, len(CAT_FIELDS), num_train.shape[1], rank=14
)
p_sparse, p_dense, p_dense_params = optimizers(pair_model, 0.0016)

pair_model.train()
pair_perm = rng.permutation(len(pos_idx))
running = 0.0
seen = 0
for start in range(0, len(pair_perm), BATCH_SIZE):
    select = pair_perm[start:start + BATCH_SIZE]
    pi = pos_idx[select]
    ni = neg_idx[select]

    pc = torch.from_numpy(
        cat_train[pi].astype(np.int64, copy=False)
    )
    pn = torch.from_numpy(num_train[pi])
    nc = torch.from_numpy(
        cat_train[ni].astype(np.int64, copy=False)
    )
    nnumer = torch.from_numpy(num_train[ni])
    wb = torch.from_numpy(sample_weight[pi])

    p_sparse.zero_grad(set_to_none=True)
    p_dense.zero_grad(set_to_none=True)
    positive_score = pair_model(pc, pn)
    negative_score = pair_model(nc, nnumer)
    loss_vec = F.softplus(-(positive_score - negative_score))
    loss = torch.sum(loss_vec * wb) / torch.sum(wb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(p_dense_params, 5.0)
    p_sparse.step()
    p_dense.step()

    running += float(loss.detach()) * len(select)
    seen += len(select)

print(
    "pairwise_pairs=%d pairwise_loss=%.6f"
    % (len(pos_idx), running / max(seen, 1)),
    flush=True,
)

del pos_idx, neg_idx, pair_perm
del click_y, like_y, sample_weight, age
gc.collect()


@torch.inference_mode()
def predict_mmoe(cat, num):
    mmoe.eval()
    result = np.empty(len(cat), dtype=np.float32)
    for start in range(0, len(cat), PRED_BATCH):
        end = min(start + PRED_BATCH, len(cat))
        c = torch.from_numpy(
            cat[start:end].astype(np.int64, copy=False)
        )
        n = torch.from_numpy(num[start:end])
        result[start:end] = mmoe(c, n)[:, 0].cpu().numpy()
    return result


@torch.inference_mode()
def predict_pair(cat, num):
    pair_model.eval()
    result = np.empty(len(cat), dtype=np.float32)
    for start in range(0, len(cat), PRED_BATCH):
        end = min(start + PRED_BATCH, len(cat))
        c = torch.from_numpy(
            cat[start:end].astype(np.int64, copy=False)
        )
        n = torch.from_numpy(num[start:end])
        result[start:end] = pair_model(c, n).cpu().numpy()
    return result


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n), 0)
    )
    local = np.arange(n, dtype=np.float32) - start_pos.astype(np.float32)

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_idx = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_idx]).astype(np.float32)
    group = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group] - 1.0, 1.0)

    ranked = np.empty(n, dtype=np.float32)
    ranked[order] = local / denom
    return ranked


def prepare_split(name):
    split = load(name)
    hv = historical_features(name, key="video_id")
    ha = historical_features(name, key="author_id")
    cat = build_cat(split)
    num = build_num(split, hv, ha)
    num = np.clip(
        (num - num_mean) / num_std, -8.0, 8.0
    ).astype(np.float32)
    del hv, ha
    gc.collect()
    return split, cat, num


valid, cat_valid, num_valid = prepare_split("valid")
mmoe_valid = predict_mmoe(cat_valid, num_valid)
pair_valid = predict_pair(cat_valid, num_valid)
valid_uid = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y)

families = {
    "multitask_mmoe": mmoe_valid,
    "within_user_pairwise_nn": pair_valid,
}

candidate_scores = {}
best_name = None
best_family = None
best_weight = 1.0
best_scores = None
best_raw = None
best_primary = -np.inf

for name, score in families.items():
    met = evaluate(valid_uid, valid_y, score)
    candidate_scores[name] = float(met["primary"])
    if float(met["primary"]) > best_primary:
        best_primary = float(met["primary"])
        best_name = name
        best_family = name
        best_weight = 1.0
        best_scores = score.copy()
        best_raw = score.copy()

inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)
has_incumbent = bool(
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
)

if has_incumbent:
    incumbent_valid = np.asarray(
        np.load(inc_valid_path, mmap_mode="r"), dtype=np.float32
    )
    incumbent_met = evaluate(valid_uid, valid_y, incumbent_valid)
    candidate_scores["trusted_incumbent"] = float(
        incumbent_met["primary"]
    )
    inc_rank = within_user_rank(valid_uid, incumbent_valid)

    for name, raw in families.items():
        raw_rank = within_user_rank(valid_uid, raw)
        local_best = -np.inf
        local_weight = 0.0
        for weight in (0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80):
            blend = (
                weight * raw_rank + (1.0 - weight) * inc_rank
            ).astype(np.float32)
            met = evaluate(valid_uid, valid_y, blend)
            primary = float(met["primary"])
            if primary > local_best:
                local_best = primary
                local_weight = weight
            if primary > best_primary:
                best_primary = primary
                best_name = "%s_rankblend_w%.2f" % (name, weight)
                best_family = name
                best_weight = float(weight)
                best_scores = blend.copy()
                best_raw = raw.copy()

        candidate_scores[name + "_best_blend"] = local_best
        print(
            "FINDINGS %s best_blend_weight=%.2f primary=%.6f"
            % (name, local_weight, local_best),
            flush=True,
        )

    if float(incumbent_met["primary"]) > best_primary:
        best_primary = float(incumbent_met["primary"])
        best_name = "trusted_incumbent"
        best_family = "trusted_incumbent"
        best_weight = 0.0
        best_scores = incumbent_valid.copy()
        best_raw = mmoe_valid.copy()

final_metrics = evaluate(valid_uid, valid_y, best_scores)
print(
    "FINDINGS winner=%s pairwise_vs_mmoe_rank_correlation=%.6f"
    % (
        best_name,
        float(np.corrcoef(
            within_user_rank(valid_uid, mmoe_valid),
            within_user_rank(valid_uid, pair_valid),
        )[0, 1]),
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

del cat_valid, num_valid, valid_y, best_scores
del mmoe_valid, pair_valid
del cat_train, num_train, train_y, train
gc.collect()

test, cat_test, num_test = prepare_split("test")

if best_family == "trusted_incumbent":
    test_scores = np.asarray(
        np.load(inc_test_path, mmap_mode="r"), dtype=np.float32
    ).copy()
else:
    if best_family == "multitask_mmoe":
        own_test = predict_mmoe(cat_test, num_test)
    elif best_family == "within_user_pairwise_nn":
        own_test = predict_pair(cat_test, num_test)
    else:
        raise RuntimeError("Unknown family: " + str(best_family))

    if best_weight < 1.0 and has_incumbent:
        incumbent_test = np.asarray(
            np.load(inc_test_path, mmap_mode="r"), dtype=np.float32
        )
        own_rank = within_user_rank(test.user_id, own_test)
        incumbent_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        test_scores = (
            best_weight * own_rank
            + (1.0 - best_weight) * incumbent_rank
        ).astype(np.float32)
    else:
        test_scores = own_test

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)