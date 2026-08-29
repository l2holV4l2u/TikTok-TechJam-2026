#!/usr/bin/env python3
# Complete runnable inspection script for the KuaiRand-Pure benchmark.
# Prints concise, quantitative facts about train and valid that should
# directly guide modeling decisions. No training, no METRICS output.

import os
import time
import numpy as np
from collections import Counter

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

np.set_printoptions(precision=4, suppress=True)

def pct(x, total):
    return 100.0 * (x / total) if total else 0.0

def uniq_counts(a):
    vals, cnt = np.unique(a, return_counts=True)
    order = np.argsort(-cnt)
    vals, cnt = vals[order], cnt[order]
    return vals, cnt

def summarize_split(name):
    s = load(name)
    n = len(s.y) if hasattr(s, "y") else len(s.user_id)
    print(f"=== {name.upper()} ===")
    print(f"rows: {n}")
    # dates
    dates, dcnt = np.unique(s.date, return_counts=True)
    print(f"dates: {dates.tolist()} counts: {dcnt.tolist()}")
    if hasattr(s, "y"):
        pos = int(np.sum(s.y))
        print(f"positives: {pos}  rate: {pos/n:0.5f}")
    # rows per-hour distribution from 'hour' categorical (0-23 expected)
    hr = s.X["hour"]
    hrs, hcnt = np.unique(hr, return_counts=True)
    # show top hours
    top_hours_idx = np.argsort(-hcnt)[:5]
    top_hours = list(zip(hrs[top_hours_idx].tolist(), hcnt[top_hours_idx].tolist()))
    print(f"hour: top5 (hour,count): {top_hours}")
    # check shape of categorical arrays
    cat_keys = list(s.X.keys())
    lens_ok = all(s.X[k].ndim == 1 and len(s.X[k]) == n for k in cat_keys)
    print(f"categorical fields: {len(cat_keys)}  arrays 1D and length==rows: {lens_ok}")
    # numeric features
    for k, arr in s.num.items():
        a = np.asarray(arr)
        n_nan = int(np.count_nonzero(np.isnan(a)))
        valid = a[~np.isnan(a)]
        if valid.size:
            mn, sd = float(np.mean(valid)), float(np.std(valid))
            p05, p50, p95 = np.percentile(valid, [5,50,95])
            print(f"num {k}: nan%={pct(n_nan,n):.2f} mean={mn:0.3f} sd={sd:0.3f} p05/p50/p95={p05:0.1f}/{p50:0.1f}/{p95:0.1f}")
        else:
            print(f"num {k}: all NaN")
    return s

def user_video_stats(s, name):
    n = len(s.user_id)
    print(f"-- user/video summary for {name} --")
    # users
    uids, inv = np.unique(s.user_id, return_inverse=True)
    n_users = len(uids)
    impressions_per_user = np.bincount(inv)
    med_imp = float(np.median(impressions_per_user))
    mean_imp = float(np.mean(impressions_per_user))
    pct_one = pct(int(np.count_nonzero(impressions_per_user==1)), n_users)
    print(f"users: {n_users} unique; impressions/user mean={mean_imp:0.3f} median={med_imp:.1f} %users_with_1imp={pct_one:0.2f}")
    # positives per user
    if hasattr(s, "y"):
        pos_per_user = np.bincount(inv, weights=s.y.astype(np.int64))
        zero_pos = int(np.count_nonzero(pos_per_user==0))
        all_pos = int(np.count_nonzero(pos_per_user == impressions_per_user))
        partial = n_users - zero_pos - all_pos
        print(f" per-user positives: zero={zero_pos} ({pct(zero_pos,n_users):.2f}%) all_pos={all_pos} ({pct(all_pos,n_users):.2f}%) partial={partial} ({pct(partial,n_users):.2f}%)")
        avg_pos_per_user = float(np.mean(pos_per_user))
        print(f" average positives/user (raw): {avg_pos_per_user:0.4f}")
        # how many users valid for GAUC: 0 < pos < impressions
        gauc_users = int(np.count_nonzero((pos_per_user>0) & (pos_per_user<impressions_per_user)))
        print(f" users contributing to GAUC condition (0<pos<imp): {gauc_users} ({pct(gauc_users,n_users):.2f}%)")
    # videos
    vids, invv = np.unique(s.video_id, return_inverse=True)
    n_vids = len(vids)
    imps_per_video = np.bincount(invv)
    med_vid_imp = float(np.median(imps_per_video))
    print(f"videos: {n_vids} unique; impressions/video median={med_vid_imp:.1f} mean={float(np.mean(imps_per_video)):0.3f}")
    return {
        "n": n, "n_users": n_users, "imps_per_user": impressions_per_user,
        "n_vids": n_vids, "imps_per_video": imps_per_video
    }

def cat_field_stats(s_train, s_valid, max_print=25):
    keys = list(s_train.X.keys())
    n_train = len(s_train.user_id)
    n_valid = len(s_valid.user_id)
    print("=== categorical field cardinalities and valid-coverage ===")
    for k in keys[:max_print]:
        arr_tr = s_train.X[k]
        arr_va = s_valid.X[k]
        card_total = FEATURE_CARDINALITIES.get(k, None)
        uniq_tr, cnt_tr = np.unique(arr_tr, return_counts=True)
        uniq_va = np.unique(arr_va)
        used_tr = len(uniq_tr)
        used_frac = pct(used_tr, card_total) if card_total else None
        # top-1 (exclude zero if present)
        mask_nz = uniq_tr != 0
        if np.any(mask_nz):
            vals, cnts = uniq_tr[mask_nz], cnt_tr[mask_nz]
            top1_idx = np.argmax(cnts)
            top1_val, top1_cnt = vals[top1_idx], cnts[top1_idx]
            top1_frac = pct(top1_cnt, n_train)
        else:
            top1_val, top1_cnt, top1_frac = uniq_tr[0], cnt_tr[0], pct(cnt_tr[0], n_train)
        # fraction of valid ids unseen in train (excluding 0)
        if uniq_va.size:
            va_nonzero = uniq_va[uniq_va != 0]
            unseen = 0
            if va_nonzero.size:
                unseen = int(np.count_nonzero(~np.isin(va_nonzero, uniq_tr)))
                unseen_frac = pct(unseen, len(va_nonzero))
            else:
                unseen_frac = 0.0
        else:
            unseen_frac = 0.0
        print(f"{k}: FEATURE_CARD={card_total} used_in_train={used_tr} ({used_frac:0.2f}%) top1={int(top1_val)}@{top1_frac:0.2f}% valid_unseen_ids%={unseen_frac:0.2f}")
    if len(keys) > max_print:
        print(f"... ({len(keys)-max_print} more fields)")

def compare_train_valid_ids(s_train, s_valid):
    # how many users/videos in valid are unseen in train
    tr_users = np.unique(s_train.user_id)
    va_users = np.unique(s_valid.user_id)
    unseen_users = int(np.count_nonzero(~np.isin(va_users, tr_users)))
    print(f"valid users: {len(va_users)}; unseen in train: {unseen_users} ({pct(unseen_users,len(va_users)):.2f}%)")
    tr_vids = np.unique(s_train.video_id)
    va_vids = np.unique(s_valid.video_id)
    unseen_vids = int(np.count_nonzero(~np.isin(va_vids, tr_vids)))
    print(f"valid videos: {len(va_vids)}; unseen in train: {unseen_vids} ({pct(unseen_vids,len(va_vids)):.2f}%)")

def time_ordering_check(s):
    # Are time_ms strictly ordering impressions per user? Count ties.
    n = len(s.user_id)
    # lexsort keys: primary user_id, secondary time_ms, tertiary original index
    idx = np.lexsort((np.arange(n), s.time_ms, s.user_id))
    user_sorted = s.user_id[idx]
    time_sorted = s.time_ms[idx]
    # find boundaries for users
    same_user = np.concatenate([[False], user_sorted[1:] == user_sorted[:-1]])
    # equal time adjacent within same user
    tie = (time_sorted[1:] == time_sorted[:-1]) & (user_sorted[1:] == user_sorted[:-1])
    n_ties = int(np.count_nonzero(tie))
    users_with_tie = int(np.count_nonzero(np.bincount(np.searchsorted(np.unique(user_sorted[tie+0]), user_sorted[1:][tie]))>0)) if n_ties>0 else 0
    pct_tied_rows = pct(n_ties, n)
    print(f"time_ms tie-adjacent rows: {n_ties} ({pct_tied_rows:0.4f}% of rows). Users with any tie-adjacent: {users_with_tie}")
    # check if any strict decreases (shouldn't happen)
    decrease = (time_sorted[1:] < time_sorted[:-1]) & (user_sorted[1:] == user_sorted[:-1])
    n_decrease = int(np.count_nonzero(decrease))
    print(f"time_ms decreases within-user (should be 0): {n_decrease}")

def show_history_keys():
    # show which historical features are available for video and author
    hv = historical_features("train", key="video_id")
    ha = historical_features("train", key="author_id")
    print("historical_features keys (video_id):", sorted(hv.keys()))
    print("historical_features keys (author_id):", sorted(ha.keys()))
    # show shapes (they are arrays aligned to rows of valid/test when used)
    for k, v in list(hv.items())[:10]:
        print(f" video hist {k}: dtype={v.dtype} shape={v.shape} min/max={np.nanmin(v):.4f}/{np.nanmax(v):.4f}")
    for k, v in list(ha.items())[:10]:
        print(f" author hist {k}: dtype={v.dtype} shape={v.shape} min/max={np.nanmin(v):.4f}/{np.nanmax(v):.4f}")

def main():
    t0 = time.time()
    train = summarize_split("train")
    valid = summarize_split("valid")
    # basic counts
    print("")
    # per-user/video summaries
    utr = user_video_stats(train, "train")
    uva = user_video_stats(valid, "valid")
    print("")
    # categorical field stats & coverage
    cat_field_stats(train, valid)
    print("")
    compare_train_valid_ids(train, valid)
    print("")
    # time ordering
    time_ordering_check(train)
    time_ordering_check(valid)
    print("")
    # FEATURE_CARDINALITIES for reference for some important fields
    interesting = ["user_id", "video_id", "author_id", "tag", "upload_type", "hour"]
    print("FEATURE_CARDINALITIES (selected):")
    for k in interesting:
        print(f" {k}: declared_cardinality={FEATURE_CARDINALITIES.get(k)}")
    print("")
    # numeric summary already printed; show how many numeric NaNs appear in train
    for k, arr in train.num.items():
        a = np.asarray(arr)
        print(f"train.num {k}: NaN%={pct(int(np.count_nonzero(np.isnan(a))), len(a)):.2f}")
    print("")
    # historical features availability and quick stats
    show_history_keys()
    t1 = time.time()
    print(f"\nINSPECTION_SECONDS: {t1-t0:0.2f}")

if __name__ == "__main__":
    main()