import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2026
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

train = load("train")
valid = load("valid")

tr_uid = np.asarray(train.X["user_id"], dtype=np.int64)
tr_vid = np.asarray(train.X["video_id"], dtype=np.int64)
tr_aid = np.asarray(train.X["author_id"], dtype=np.int64)
tr_tag = np.asarray(train.X["tag"], dtype=np.int64)
tr_tab = np.asarray(train.X["tab"], dtype=np.int64)
tr_dur = np.asarray(train.X["duration_bucket"], dtype=np.int64)
tr_y = np.asarray(train.y, dtype=np.int8)
tr_date = np.asarray(train.date, dtype=np.int32)
tr_time = np.asarray(train.time_ms, dtype=np.int64)

va_uid = np.asarray(valid.X["user_id"], dtype=np.int64)
va_vid = np.asarray(valid.X["video_id"], dtype=np.int64)
va_aid = np.asarray(valid.X["author_id"], dtype=np.int64)
va_tag = np.asarray(valid.X["tag"], dtype=np.int64)
va_tab = np.asarray(valid.X["tab"], dtype=np.int64)
va_dur = np.asarray(valid.X["duration_bucket"], dtype=np.int64)

N_USERS = int(FEATURE_CARDINALITIES["user_id"])
N_VIDEOS = int(FEATURE_CARDINALITIES["video_id"])
N_AUTHORS = int(FEATURE_CARDINALITIES["author_id"])
N_TAGS = int(FEATURE_CARDINALITIES["tag"])
N_TABS = int(FEATURE_CARDINALITIES["tab"])
N_DURS = int(FEATURE_CARDINALITIES["duration_bucket"])


def within_user_rank(user_ids, scores):
    """Ascending percentile rank within each logged impression set."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Row index provides deterministic tie-breaking.
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_u = user_ids[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = sorted_u[1:] != sorted_u[:-1]

    starts = np.maximum.accumulate(
        np.where(first, np.arange(n, dtype=np.int64), 0)
    )
    position = np.arange(n, dtype=np.int64) - starts

    group_starts = np.flatnonzero(first)
    group_ends = np.r_[group_starts[1:], n]
    group_sizes = group_ends - group_starts
    denom_sorted = np.repeat(np.maximum(group_sizes - 1, 1), group_sizes)

    ranked_sorted = position.astype(np.float64) / denom_sorted
    ranked_sorted[np.repeat(group_sizes == 1, group_sizes)] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


# ----------------------------------------------------------------------
# Family 1: pairwise BPR over logged positive-negative impressions.
# ----------------------------------------------------------------------
class PairwiseBPR(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.user = nn.Embedding(N_USERS, k)
        self.video = nn.Embedding(N_VIDEOS, k)
        self.author = nn.Embedding(N_AUTHORS, k)
        self.video_bias = nn.Embedding(N_VIDEOS, 1)
        self.author_bias = nn.Embedding(N_AUTHORS, 1)
        self.tag_bias = nn.Embedding(N_TAGS, 1)
        self.tab_bias = nn.Embedding(N_TABS, 1)
        self.duration_bias = nn.Embedding(N_DURS, 1)
        self.scale = float(k) ** -0.5

        with torch.no_grad():
            self.user.weight.normal_(0.0, 0.06)
            self.video.weight.normal_(0.0, 0.06)
            self.author.weight.normal_(0.0, 0.04)
            self.video_bias.weight.zero_()
            self.author_bias.weight.zero_()
            self.tag_bias.weight.zero_()
            self.tab_bias.weight.zero_()
            self.duration_bias.weight.zero_()

    def score(self, u, v, a, tag, tab, dur):
        uv = (self.user(u) * self.video(v)).sum(dim=1)
        ua = (self.user(u) * self.author(a)).sum(dim=1)
        return (
            (uv + 0.65 * ua) * self.scale
            + self.video_bias(v).squeeze(1)
            + 0.5 * self.author_bias(a).squeeze(1)
            + self.tag_bias(tag).squeeze(1)
            + self.tab_bias(tab).squeeze(1)
            + self.duration_bias(dur).squeeze(1)
        )


neg_rows = np.flatnonzero(tr_y == 0)
neg_order = np.argsort(tr_uid[neg_rows], kind="stable")
neg_rows = neg_rows[neg_order]
neg_counts = np.bincount(tr_uid[neg_rows], minlength=N_USERS)
neg_starts = np.cumsum(np.r_[0, neg_counts[:-1]])

pos_rows_all = np.flatnonzero(tr_y == 1)
usable = neg_counts[tr_uid[pos_rows_all]] > 0
pos_rows = pos_rows_all[usable]
pos_users = tr_uid[pos_rows]

# A moderate recency half-life emphasizes preference order near the boundary.
days_ago = 21 - (tr_date[pos_rows] % 100)
pair_weight = np.exp2(-days_ago.astype(np.float32) / 5.0)
pair_weight = pair_weight / max(float(pair_weight.mean()), 1e-6)

bpr = PairwiseBPR(k=32)
optimizer = torch.optim.AdamW(bpr.parameters(), lr=0.012, weight_decay=2e-6)
batch_size = 8192
epochs = 7

pos_rows_t = torch.from_numpy(pos_rows)
pair_weight_t = torch.from_numpy(pair_weight.astype(np.float32))

for epoch in range(epochs):
    offsets = rng.integers(
        0,
        neg_counts[pos_users],
        size=len(pos_rows),
        endpoint=False,
    )
    sampled_neg = neg_rows[neg_starts[pos_users] + offsets]
    sampled_neg_t = torch.from_numpy(sampled_neg.astype(np.int64))

    permutation = torch.randperm(
        len(pos_rows), generator=torch.Generator().manual_seed(SEED + epoch)
    )

    bpr.train()
    for begin in range(0, len(pos_rows), batch_size):
        take = permutation[begin:begin + batch_size]
        pr = pos_rows_t[take]
        nr = sampled_neg_t[take]
        w = pair_weight_t[take]

        u = torch.from_numpy(tr_uid[pos_rows[take.numpy()]])
        ps = bpr.score(
            u,
            torch.from_numpy(tr_vid[pos_rows[take.numpy()]]),
            torch.from_numpy(tr_aid[pos_rows[take.numpy()]]),
            torch.from_numpy(tr_tag[pos_rows[take.numpy()]]),
            torch.from_numpy(tr_tab[pos_rows[take.numpy()]]),
            torch.from_numpy(tr_dur[pos_rows[take.numpy()]]),
        )
        ns = bpr.score(
            u,
            torch.from_numpy(tr_vid[sampled_neg[take.numpy()]]),
            torch.from_numpy(tr_aid[sampled_neg[take.numpy()]]),
            torch.from_numpy(tr_tag[sampled_neg[take.numpy()]]),
            torch.from_numpy(tr_tab[sampled_neg[take.numpy()]]),
            torch.from_numpy(tr_dur[sampled_neg[take.numpy()]]),
        )

        loss = (torch.nn.functional.softplus(-(ps - ns)) * w).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def predict_bpr(split, batch=65536):
    uid = np.asarray(split.X["user_id"], dtype=np.int64)
    vid = np.asarray(split.X["video_id"], dtype=np.int64)
    aid = np.asarray(split.X["author_id"], dtype=np.int64)
    tag = np.asarray(split.X["tag"], dtype=np.int64)
    tab = np.asarray(split.X["tab"], dtype=np.int64)
    dur = np.asarray(split.X["duration_bucket"], dtype=np.int64)

    result = np.empty(len(uid), dtype=np.float64)
    bpr.eval()
    with torch.inference_mode():
        for begin in range(0, len(uid), batch):
            end = min(begin + batch, len(uid))
            result[begin:end] = bpr.score(
                torch.from_numpy(uid[begin:end]),
                torch.from_numpy(vid[begin:end]),
                torch.from_numpy(aid[begin:end]),
                torch.from_numpy(tag[begin:end]),
                torch.from_numpy(tab[begin:end]),
                torch.from_numpy(dur[begin:end]),
            ).numpy()
    return result


bpr_valid = predict_bpr(valid)

# ----------------------------------------------------------------------
# Family 2: implicit collaborative SVD of positive user-video interactions.
# ----------------------------------------------------------------------
positive_rows = np.flatnonzero(tr_y == 1)
ui = sparse.coo_matrix(
    (
        np.ones(len(positive_rows), dtype=np.float32),
        (tr_uid[positive_rows], tr_vid[positive_rows]),
    ),
    shape=(N_USERS, N_VIDEOS),
).tocsr()
ui.sum_duplicates()
ui.data[:] = 1.0

row_degree = np.asarray(ui.sum(axis=1)).ravel()
col_degree = np.asarray(ui.sum(axis=0)).ravel()
row_scale = np.zeros_like(row_degree, dtype=np.float32)
nz_row = row_degree > 0
row_scale[nz_row] = np.power(row_degree[nz_row], -0.5)
col_scale = np.zeros_like(col_degree, dtype=np.float32)
nz_col = col_degree > 0
col_scale[nz_col] = np.power(col_degree[nz_col], -0.25)

ui_norm = sparse.diags(row_scale) @ ui @ sparse.diags(col_scale)

try:
    svd_u, svd_s, svd_vt = svds(
        ui_norm.astype(np.float32),
        k=32,
        which="LM",
        random_state=SEED,
        maxiter=500,
    )
    descending = np.argsort(svd_s)[::-1]
    svd_s = svd_s[descending].astype(np.float32)
    svd_u = svd_u[:, descending].astype(np.float32)
    svd_vt = svd_vt[descending].astype(np.float32)
    svd_user = svd_u * svd_s[None, :]
except Exception as exc:
    print("FINDINGS collaborative_svd_failed=" + repr(exc))
    svd_user = np.zeros((N_USERS, 1), dtype=np.float32)
    svd_vt = np.zeros((1, N_VIDEOS), dtype=np.float32)


def predict_svd(split):
    uid = np.asarray(split.X["user_id"], dtype=np.int64)
    vid = np.asarray(split.X["video_id"], dtype=np.int64)
    return np.einsum(
        "ij,ij->i",
        svd_user[uid],
        svd_vt[:, vid].T,
        optimize=True,
    ).astype(np.float64)


svd_valid = predict_svd(valid)

# ----------------------------------------------------------------------
# Family 3: low-rank positive-transition sequence model.
# It predicts candidates from each user's final observed positive video.
# ----------------------------------------------------------------------
pos_sort = np.lexsort(
    (
        positive_rows,
        tr_time[positive_rows],
        tr_uid[positive_rows],
    )
)
ordered_pos_rows = positive_rows[pos_sort]
ordered_pos_uid = tr_uid[ordered_pos_rows]
ordered_pos_vid = tr_vid[ordered_pos_rows]

same_user = ordered_pos_uid[1:] == ordered_pos_uid[:-1]
source_vid = ordered_pos_vid[:-1][same_user]
target_vid = ordered_pos_vid[1:][same_user]

transition = sparse.coo_matrix(
    (
        np.ones(len(source_vid), dtype=np.float32),
        (source_vid, target_vid),
    ),
    shape=(N_VIDEOS, N_VIDEOS),
).tocsr()
transition.sum_duplicates()
transition.data = np.log1p(transition.data)

trans_row_degree = np.asarray(transition.sum(axis=1)).ravel()
trans_col_degree = np.asarray(transition.sum(axis=0)).ravel()
trs = np.zeros(N_VIDEOS, dtype=np.float32)
tcs = np.zeros(N_VIDEOS, dtype=np.float32)
mask = trans_row_degree > 0
trs[mask] = np.power(trans_row_degree[mask], -0.5)
mask = trans_col_degree > 0
tcs[mask] = np.power(trans_col_degree[mask], -0.25)
transition_norm = sparse.diags(trs) @ transition @ sparse.diags(tcs)

try:
    trans_u, trans_s, trans_vt = svds(
        transition_norm.astype(np.float32),
        k=24,
        which="LM",
        random_state=SEED + 1,
        maxiter=500,
    )
    descending = np.argsort(trans_s)[::-1]
    trans_s = trans_s[descending].astype(np.float32)
    trans_u = trans_u[:, descending].astype(np.float32)
    trans_vt = trans_vt[descending].astype(np.float32)
    trans_source = trans_u * trans_s[None, :]
except Exception as exc:
    print("FINDINGS transition_svd_failed=" + repr(exc))
    trans_source = np.zeros((N_VIDEOS, 1), dtype=np.float32)
    trans_vt = np.zeros((1, N_VIDEOS), dtype=np.float32)

last_pos_index = np.full(N_USERS, -1, dtype=np.int64)
np.maximum.at(
    last_pos_index,
    ordered_pos_uid,
    np.arange(len(ordered_pos_uid), dtype=np.int64),
)
last_positive_video = np.zeros(N_USERS, dtype=np.int64)
has_history = last_pos_index >= 0
last_positive_video[has_history] = ordered_pos_vid[last_pos_index[has_history]]


def predict_transition(split):
    uid = np.asarray(split.X["user_id"], dtype=np.int64)
    vid = np.asarray(split.X["video_id"], dtype=np.int64)
    src = last_positive_video[uid]
    result = np.einsum(
        "ij,ij->i",
        trans_source[src],
        trans_vt[:, vid].T,
        optimize=True,
    ).astype(np.float64)
    result[~has_history[uid]] = 0.0
    return result


transition_valid = predict_transition(valid)

# ----------------------------------------------------------------------
# Fixed rank aggregation with the trusted incumbent.
# Rank aggregation avoids choosing score calibration on validation.
# ----------------------------------------------------------------------
shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if os.path.exists(inc_valid_path):
    incumbent_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
else:
    incumbent_valid = bpr_valid.copy()

inc_rank_valid = within_user_rank(valid.user_id, incumbent_valid)
bpr_rank_valid = within_user_rank(valid.user_id, bpr_valid)
svd_rank_valid = within_user_rank(valid.user_id, svd_valid)
transition_rank_valid = within_user_rank(valid.user_id, transition_valid)

latent_consensus_valid = (
    0.55 * bpr_rank_valid
    + 0.30 * svd_rank_valid
    + 0.15 * transition_rank_valid
)

candidate_scores = {
    "bpr_pairwise": bpr_valid,
    "collaborative_svd": svd_valid,
    "transition_sequence": transition_valid,
    "incumbent_bpr_rank_blend": 0.80 * inc_rank_valid + 0.20 * bpr_rank_valid,
    "incumbent_svd_rank_blend": 0.80 * inc_rank_valid + 0.20 * svd_rank_valid,
    "incumbent_transition_rank_blend": (
        0.80 * inc_rank_valid + 0.20 * transition_rank_valid
    ),
    "incumbent_latent_consensus": (
        0.78 * inc_rank_valid + 0.22 * latent_consensus_valid
    ),
}

candidate_metrics = {}
for name, scores in candidate_scores.items():
    candidate_metrics[name] = evaluate(valid.user_id, valid.y, scores)

candidate_primary = {
    name: float(m["primary"]) for name, m in candidate_metrics.items()
}
winner_name = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner_name]
metrics = candidate_metrics[winner_name]

print("CANDIDATES " + json.dumps(candidate_primary, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner_name,
            "bpr_pairs": int(len(pos_rows)),
            "ui_positive_nnz": int(ui.nnz),
            "transition_nnz": int(transition.nnz),
            "users_with_positive_history": int(has_history.sum()),
        },
        sort_keys=True,
    )
)

# Test features only; test labels are never accessed.
test = load("test")
bpr_test = predict_bpr(test)
svd_test = predict_svd(test)
transition_test = predict_transition(test)

if os.path.exists(inc_test_path):
    incumbent_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
else:
    incumbent_test = bpr_test.copy()

inc_rank_test = within_user_rank(test.user_id, incumbent_test)
bpr_rank_test = within_user_rank(test.user_id, bpr_test)
svd_rank_test = within_user_rank(test.user_id, svd_test)
transition_rank_test = within_user_rank(test.user_id, transition_test)
latent_consensus_test = (
    0.55 * bpr_rank_test
    + 0.30 * svd_rank_test
    + 0.15 * transition_rank_test
)

test_candidates = {
    "bpr_pairwise": bpr_test,
    "collaborative_svd": svd_test,
    "transition_sequence": transition_test,
    "incumbent_bpr_rank_blend": 0.80 * inc_rank_test + 0.20 * bpr_rank_test,
    "incumbent_svd_rank_blend": 0.80 * inc_rank_test + 0.20 * svd_rank_test,
    "incumbent_transition_rank_blend": (
        0.80 * inc_rank_test + 0.20 * transition_rank_test
    ),
    "incumbent_latent_consensus": (
        0.78 * inc_rank_test + 0.22 * latent_consensus_test
    ),
}
test_scores = test_candidates[winner_name]

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
    if winner_name.startswith("incumbent_"):
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(bpr_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(", ", ": "),
    )
)