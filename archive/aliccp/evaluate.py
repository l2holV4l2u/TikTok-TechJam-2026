"""CTR AUC over all impressions, CVR AUC over the clicked subset only. No sklearn."""
import warnings

import numpy as np


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """1-indexed ranks with ties resolved to the average rank of their group."""
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    a_sorted = a[sorter]
    new_group = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = new_group.cumsum()[inv]
    group_end = np.r_[np.nonzero(new_group)[0], len(new_group)]
    return 0.5 * (group_end[dense] + group_end[dense - 1] + 1)


def _auc(y: np.ndarray, p: np.ndarray, label: str) -> float:
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        warnings.warn(f"{label} AUC undefined: {n_pos} positives, {n_neg} negatives")
        return float("nan")
    ranks = _rankdata_avg(np.asarray(p, dtype=np.float64))
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ctr_cvr_auc(y_click, y_conv, p_ctr, p_cvr) -> dict:
    y_click = np.asarray(y_click)
    y_conv = np.asarray(y_conv)
    p_ctr = np.asarray(p_ctr)
    p_cvr = np.asarray(p_cvr)

    ctr_auc = _auc(y_click, p_ctr, "ctr")

    clicked = y_click == 1
    # CVR = P(conversion | click): score only where a click happened, never the full impression space.
    cvr_auc = _auc(y_conv[clicked], p_cvr[clicked], "cvr")

    mean_delta_ready = not (np.isnan(ctr_auc) or np.isnan(cvr_auc))
    return {"ctr_auc": ctr_auc, "cvr_auc": cvr_auc, "mean_delta_ready": mean_delta_ready}
