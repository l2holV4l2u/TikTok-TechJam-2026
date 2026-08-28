"""KuaiRand-Pure Track 2 metrics. Definitions pinned to the organizers' spec text.

GAUC: per-user AUC, weighted mean by positive count, users with 0 or all-positive
impressions excluded (0 < positives < impressions required).

nDCG@k: gain = 2**rel - 1. Denominator for nDCG@5 is ALL users -- a user with zero
positives has IDCG=0 and contributes nDCG=0 to the average (not skipped). nDCG@10
follows the same all-users-included rule for consistency with nDCG@5.

recall@50: fraction of a user's positives in their top-50 by score, averaged only
over users with >=1 positive (zero-positive users have no recall defined, excluded).

primary = mean(gauc, ndcg@5).

Tie-break rules (documented, deterministic):
- AUC (GAUC): ties get the averaged rank (standard AUC tie handling); tie order does
  not affect the result so no explicit tiebreak key is needed.
- Top-K ranking (nDCG, recall@50): ties in score are broken by original row order
  (stable: the row appearing first in the input arrays ranks higher). Ideal-order
  ranking (IDCG) sorts by true relevance only; tie order there never affects the sum
  since swapping two equal-gain items leaves the DCG sum unchanged.
"""
import numpy as np


def _sorted_by_user(user_ids: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Row order: user_id ascending, then `key` ascending, ties by original row order.

    Identical to np.lexsort((arange(n), key, user_ids)) but ~3x faster on millions of rows: a
    stable sort on `key` leaves equal keys in original order, and a second stable sort on the
    user preserves that ordering inside each user block. lexsort's n-key path is the single
    biggest cost in evaluate(), which the agent's scripts now call once per training epoch.
    """
    o = np.argsort(key, kind="stable")
    return o[np.argsort(user_ids[o], kind="stable")]


def _new_group_mask(sorted_keys: np.ndarray) -> np.ndarray:
    n = sorted_keys.size
    mask = np.empty(n, dtype=bool)
    if n:
        mask[0] = True
        mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return mask


def _group_by_user(user_ids: np.ndarray, rank_key: np.ndarray, idx: np.ndarray):
    """Sort rows into per-user blocks ordered by rank_key descending, idx (original row order) as tiebreak.
    Returns (order, group_id, rank1, n_groups). group_id numbering = ascending user_id order, stable across calls."""
    order = _sorted_by_user(user_ids, -rank_key)
    u = user_ids[order]
    new_user = _new_group_mask(u)
    group_id = np.cumsum(new_user) - 1
    group_start = np.flatnonzero(new_user)
    n_groups = group_start.size
    rank1 = np.arange(u.size) - group_start[group_id] + 1
    return order, group_id, rank1, n_groups


def _gauc(user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray, idx: np.ndarray,
          per_user: bool = False):
    n = user_ids.size
    order = _sorted_by_user(user_ids, scores)  # ascending score within user; tie order irrelevant (averaged)
    u = user_ids[order]
    s = scores[order]
    y = labels[order]

    new_user = _new_group_mask(u)
    group_id = np.cumsum(new_user) - 1
    group_start = np.flatnonzero(new_user)
    n_groups = group_start.size
    seq_rank = (np.arange(n) - group_start[group_id] + 1).astype(np.float64)

    new_tie = np.empty(n, dtype=bool)
    if n:
        new_tie[0] = True
        new_tie[1:] = (u[1:] != u[:-1]) | (s[1:] != s[:-1])
    tie_id = np.cumsum(new_tie) - 1
    tie_start = np.flatnonzero(new_tie)
    tie_end = np.r_[tie_start[1:] - 1, n - 1]
    avg_rank = 0.5 * (seq_rank[tie_start] + seq_rank[tie_end])
    rank = avg_rank[tie_id]

    group_size = np.bincount(group_id, minlength=n_groups).astype(np.float64)
    n_pos = np.bincount(group_id, weights=y, minlength=n_groups)
    sum_rank_pos = np.bincount(group_id, weights=rank * y, minlength=n_groups)
    n_neg = group_size - n_pos

    valid = (n_pos > 0) & (n_neg > 0)
    if not np.any(valid):
        return (float("nan"), np.full(n_groups, np.nan)) if per_user else float("nan")
    auc = (sum_rank_pos[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2.0) / (n_pos[valid] * n_neg[valid])
    w = n_pos[valid]
    overall = float(np.sum(auc * w) / np.sum(w))
    if not per_user:
        return overall
    # NaN marks a user excluded from GAUC (all-positive or all-negative), matching the spec
    full = np.full(n_groups, np.nan)
    full[valid] = auc
    return overall, full


def _is_binary(labels: np.ndarray) -> bool:
    """True when every label is 0 or 1, which is the case for the scored `long_view` column."""
    return bool(np.all((labels == 0.0) | (labels == 1.0)))


def _idcg_binary(n_pos: np.ndarray, k: int) -> np.ndarray:
    """IDCG@k for 0/1 labels, without sorting.

    The ideal ranking puts every positive first, so the top-k gains are 1 for the first
    min(k, n_pos) ranks and 0 after -- IDCG@k is a prefix sum of 1/log2(r+1) indexed by the
    user's positive count. Sorting several million rows to rediscover that is the third of
    evaluate()'s three sorts, and this removes it.
    """
    prefix = np.concatenate(([0.0], np.cumsum(1.0 / np.log2(np.arange(1, k + 1) + 1.0))))
    return prefix[np.minimum(n_pos, k).astype(np.int64)]


def _dcg_sum(labels_sorted: np.ndarray, rank1: np.ndarray, group_id: np.ndarray, n_groups: int, k: int) -> np.ndarray:
    # only the top-k rows per user contribute; computing 2**x and log2 over every row to then
    # discard 99.8% of them is the second largest cost. k=5 over ~1000 users is ~5000 rows.
    sel = rank1 <= k
    if not sel.any():
        return np.zeros(n_groups)
    gain = 2.0 ** labels_sorted[sel] - 1.0  # spec: nDCG gain = 2^rel - 1
    discount = 1.0 / np.log2(rank1[sel].astype(np.float64) + 1.0)
    return np.bincount(group_id[sel], weights=gain * discount, minlength=n_groups)


def evaluate(user_ids, labels, scores, per_user: bool = False) -> dict:
    """Official GAUC / nDCG@5 scoring. `per_user=True` additionally returns the per-user arrays.

    The aggregate alone cannot say WHERE a model is losing. Both metrics are means over users,
    so the breakdown is already computed internally; returning it costs nothing and saves
    reimplementing the metric (and getting the tie handling or the zero-positive convention
    subtly wrong) just to look at a segment.
    """
    user_ids = np.asarray(user_ids)
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if not (user_ids.shape[0] == labels.shape[0] == scores.shape[0]):
        raise ValueError("user_ids, labels, scores must have the same length")

    n = user_ids.shape[0]
    if n == 0:
        nan = float("nan")
        return {"gauc": nan, "ndcg@5": nan, "primary": nan, "ndcg@10": nan, "recall@50": nan}

    idx = np.arange(n)
    gauc_out = _gauc(user_ids, labels, scores, idx, per_user=per_user)
    gauc, auc_per_user = gauc_out if per_user else (gauc_out, None)

    order_pred, group_id, rank1_pred, n_groups = _group_by_user(user_ids, scores, idx)
    # group_id and group_id_ideal share the same user<->index mapping: both lexsorts use
    # user_ids as the primary key, so group order is always ascending user_id regardless
    # of the secondary ranking key.

    labels_pred_sorted = labels[order_pred]
    n_pos_group = np.bincount(group_id, weights=labels_pred_sorted, minlength=n_groups)

    dcg5 = _dcg_sum(labels_pred_sorted, rank1_pred, group_id, n_groups, 5)
    dcg10 = _dcg_sum(labels_pred_sorted, rank1_pred, group_id, n_groups, 10)
    if _is_binary(labels):
        idcg5 = _idcg_binary(n_pos_group, 5)
        idcg10 = _idcg_binary(n_pos_group, 10)
    else:
        order_ideal, group_id_ideal, rank1_ideal, _ = _group_by_user(user_ids, labels, idx)
        labels_ideal_sorted = labels[order_ideal]
        idcg5 = _dcg_sum(labels_ideal_sorted, rank1_ideal, group_id_ideal, n_groups, 5)
        idcg10 = _dcg_sum(labels_ideal_sorted, rank1_ideal, group_id_ideal, n_groups, 10)

    ndcg5 = np.zeros(n_groups)
    mask5 = idcg5 > 0
    ndcg5[mask5] = dcg5[mask5] / idcg5[mask5]  # IDCG=0 (zero positives) -> nDCG=0, included in mean per spec

    ndcg10 = np.zeros(n_groups)
    mask10 = idcg10 > 0
    ndcg10[mask10] = dcg10[mask10] / idcg10[mask10]

    total_pos = n_pos_group
    hits50 = np.bincount(group_id, weights=np.where(rank1_pred <= 50, labels_pred_sorted, 0.0), minlength=n_groups)
    has_pos = total_pos > 0
    recall50 = float(np.mean(hits50[has_pos] / total_pos[has_pos])) if np.any(has_pos) else float("nan")

    ndcg5_mean = float(np.mean(ndcg5))
    ndcg10_mean = float(np.mean(ndcg10))
    primary = float("nan") if np.isnan(gauc) else (gauc + ndcg5_mean) / 2.0

    out = {"gauc": gauc, "ndcg@5": ndcg5_mean, "primary": primary, "ndcg@10": ndcg10_mean,
           "recall@50": recall50}
    if per_user:
        uniq = user_ids[order_pred][rank1_pred == 1]
        out["per_user"] = {
            "user_id": uniq,
            "n_impressions": np.bincount(group_id, minlength=n_groups),
            "n_positives": total_pos,
            "auc": auc_per_user,          # NaN where the user is excluded from GAUC
            "ndcg@5": ndcg5,
            "ndcg@10": ndcg10,
        }
    return out
