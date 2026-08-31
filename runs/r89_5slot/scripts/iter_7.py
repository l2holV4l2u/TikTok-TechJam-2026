import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
EPS = 1e-5

ENTITY_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "onehot_feat3",
    "duration_bucket",
    "upload_type",
]
PAIR_FIELDS = ["video_id", "author_id", "tag"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd < 1e-12:
        sd = 1.0
    return (x - float(np.mean(x))) / sd


def within_user_rank(user_ids, scores):
    """Ordinal percentile ranks, computed without Python loops."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    us = user_ids[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = us[1:] != us[:-1]
    starts = np.flatnonzero(new_user)
    counts = np.diff(np.r_[starts, n])
    starts_per_row = np.repeat(starts, counts)

    pos = np.arange(n, dtype=np.float64) - starts_per_row
    denom = np.maximum(np.repeat(counts, counts) - 1, 1)
    ranked = pos / denom

    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def extract(split):
    x = {}
    for f in ENTITY_FIELDS:
        x[f] = np.asarray(split.X[f], dtype=np.int64)
    x["user_id"] = np.asarray(split.user_id, dtype=np.int64)
    return {
        "x": x,
        "user": np.asarray(split.user_id, dtype=np.int64),
        "date": np.asarray(split.date, dtype=np.int32),
    }


def concatenate_data(a, b):
    x = {}
    for f in list(ENTITY_FIELDS) + ["user_id"]:
        x[f] = np.concatenate([a["x"][f], b["x"][f]])
    return {
        "x": x,
        "user": x["user_id"],
        "date": np.concatenate([a["date"], b["date"]]),
    }


def day_number(yyyymmdd):
    # All dates are in one month in this benchmark.
    return np.asarray(yyyymmdd, dtype=np.int64) % 100


def recency_weights(date, half_life):
    if half_life is None:
        return np.ones(len(date), dtype=np.float64)
    d = day_number(date)
    age = np.max(d) - d
    w = np.exp2(-age.astype(np.float64) / float(half_life))
    # Keep smoothing constants comparable between half-lives.
    w /= max(float(np.mean(w)), 1e-12)
    return w


class PreparedKeys:
    def __init__(self, data):
        self.data = data
        self.pairs = {}
        u = data["x"]["user_id"]
        for f in PAIR_FIELDS:
            mult = int(FEATURE_CARDINALITIES[f])
            key = u * np.int64(mult) + data["x"][f]
            unique, inverse = np.unique(key, return_inverse=True)
            self.pairs[f] = {
                "unique": unique,
                "inverse": inverse.astype(np.int32, copy=False),
                "mult": mult,
            }


def smoothed_marginal(ids, y, w, size, alpha, global_rate):
    cnt = np.bincount(ids, weights=w, minlength=size).astype(np.float64)
    pos = np.bincount(ids, weights=w * y, minlength=size).astype(np.float64)
    return ((pos + alpha * global_rate) / (cnt + alpha)).astype(np.float32)


def fit_empirical_bayes(prepared, y, half_life):
    data = prepared.data
    y = np.asarray(y, dtype=np.float64)
    w = recency_weights(data["date"], half_life)
    global_rate = float(np.sum(w * y) / np.sum(w))

    alphas = {
        "video_id": 25.0,
        "author_id": 45.0,
        "tag": 150.0,
        "onehot_feat3": 100.0,
        "duration_bucket": 250.0,
        "upload_type": 180.0,
    }

    marginals = {}
    for f in ENTITY_FIELDS:
        size = int(FEATURE_CARDINALITIES[f])
        marginals[f] = smoothed_marginal(
            data["x"][f], y, w, size, alphas[f], global_rate
        )

    pair_alpha = {
        "video_id": 8.0,
        "author_id": 15.0,
        "tag": 24.0,
    }
    pair_tables = {}

    for f in PAIR_FIELDS:
        info = prepared.pairs[f]
        inv = info["inverse"]
        size = len(info["unique"])
        cnt = np.bincount(inv, weights=w, minlength=size).astype(np.float64)
        pos = np.bincount(inv, weights=w * y, minlength=size).astype(np.float64)

        # Each sparse pair shrinks toward its content entity's posterior.
        entity_for_unique = (
            info["unique"] % np.int64(info["mult"])
        ).astype(np.int64)
        prior = marginals[f][entity_for_unique].astype(np.float64)
        alpha = pair_alpha[f]
        rate = (pos + alpha * prior) / (cnt + alpha)

        pair_tables[f] = {
            "unique": info["unique"],
            "rate": rate.astype(np.float32),
            "count": cnt.astype(np.float32),
            "mult": info["mult"],
        }

    return {
        "global": global_rate,
        "marginals": marginals,
        "pairs": pair_tables,
        "half_life": half_life,
    }


def lookup_pair(table, user, entity, fallback):
    key = user * np.int64(table["mult"]) + entity
    unique = table["unique"]
    idx = np.searchsorted(unique, key)
    ok = idx < len(unique)
    safe = np.minimum(idx, len(unique) - 1)
    ok &= unique[safe] == key

    rate = np.asarray(fallback, dtype=np.float64).copy()
    count = np.zeros(len(key), dtype=np.float64)
    if np.any(ok):
        rate[ok] = table["rate"][safe[ok]]
        count[ok] = table["count"][safe[ok]]
    return rate, count


def predict_families(model, data):
    x = data["x"]
    m = model["marginals"]

    lv = logit(m["video_id"][x["video_id"]])
    la = logit(m["author_id"][x["author_id"]])
    lt = logit(m["tag"][x["tag"]])
    lo = logit(m["onehot_feat3"][x["onehot_feat3"]])
    ld = logit(m["duration_bucket"][x["duration_bucket"]])
    lu = logit(m["upload_type"][x["upload_type"]])

    # Family 1: stationary content hierarchy, no user identity.
    content = (
        0.48 * lv
        + 0.22 * la
        + 0.12 * lt
        + 0.08 * lo
        + 0.06 * ld
        + 0.04 * lu
    )

    uv, cuv = lookup_pair(
        model["pairs"]["video_id"],
        x["user_id"],
        x["video_id"],
        m["video_id"][x["video_id"]],
    )
    ua, cua = lookup_pair(
        model["pairs"]["author_id"],
        x["user_id"],
        x["author_id"],
        m["author_id"][x["author_id"]],
    )
    ut, cut = lookup_pair(
        model["pairs"]["tag"],
        x["user_id"],
        x["tag"],
        m["tag"][x["tag"]],
    )

    pair_memory = (
        0.48 * logit(uv)
        + 0.34 * logit(ua)
        + 0.18 * logit(ut)
    )

    # Family 2: confidence-adaptive sparse memory. With no repeated pair it
    # reduces to the stationary content score; repeated interactions gradually
    # replace the corresponding content prior.
    confidence = 1.0 - np.exp(
        -(cuv / 5.0 + cua / 9.0 + cut / 14.0)
    )
    adaptive = (1.0 - confidence) * content + confidence * pair_memory

    # Family 3: conservative residual formation, retaining content-side
    # generalization while adding only half of the memorized deviation.
    conservative = content + 0.50 * (pair_memory - (0.48 * lv + 0.34 * la + 0.18 * lt))

    return {
        "content_hierarchy": np.asarray(content, dtype=np.float64),
        "sparse_pair_memory": np.asarray(pair_memory, dtype=np.float64),
        "adaptive_memory": np.asarray(adaptive, dtype=np.float64),
        "conservative_residual": np.asarray(conservative, dtype=np.float64),
    }


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float64)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_data = extract(train)
valid_data = extract(valid)
prepared_train = PreparedKeys(train_data)

shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

inc_z = standardize(inc_valid)
inc_rank = within_user_rank(valid_users, inc_valid)

candidate_metrics = {}
candidate_spec = {}
raw_by_key = {}

best_name = "trusted_incumbent"
best_scores = inc_valid.copy()
best_metrics = evaluate(valid_users, valid_y, best_scores)
best_spec = {
    "half_life": None,
    "family": "content_hierarchy",
    "mode": "incumbent",
    "weight": 0.0,
}
candidate_metrics[best_name] = float(best_metrics["primary"])

half_lives = [None, 5.0, 2.5]

for half_life in half_lives:
    model = fit_empirical_bayes(prepared_train, train_y, half_life)
    families = predict_families(model, valid_data)
    hname = "all" if half_life is None else str(half_life).replace(".", "p")

    for family, raw in families.items():
        raw_key = (half_life, family)
        raw_by_key[raw_key] = raw
        own_name = "%s_h%s" % (family, hname)

        own_met = evaluate(valid_users, valid_y, raw)
        candidate_metrics[own_name] = float(own_met["primary"])
        if float(own_met["primary"]) > float(best_metrics["primary"]):
            best_name = own_name
            best_scores = raw.copy()
            best_metrics = own_met
            best_spec = {
                "half_life": half_life,
                "family": family,
                "mode": "own",
                "weight": 1.0,
            }

        raw_z = standardize(raw)
        raw_rank = within_user_rank(valid_users, raw)

        for w in (0.10, 0.18, 0.26, 0.35, 0.45, 0.55):
            score = (1.0 - w) * inc_z + w * raw_z
            name = "%s_zblend_%.2f" % (own_name, w)
            met = evaluate(valid_users, valid_y, score)
            candidate_metrics[name] = float(met["primary"])
            if float(met["primary"]) > float(best_metrics["primary"]):
                best_name = name
                best_scores = score.copy()
                best_metrics = met
                best_spec = {
                    "half_life": half_life,
                    "family": family,
                    "mode": "zblend",
                    "weight": float(w),
                }

        for w in (0.10, 0.20, 0.30, 0.40):
            score = (1.0 - w) * inc_rank + w * raw_rank
            name = "%s_rankblend_%.2f" % (own_name, w)
            met = evaluate(valid_users, valid_y, score)
            candidate_metrics[name] = float(met["primary"])
            if float(met["primary"]) > float(best_metrics["primary"]):
                best_name = name
                best_scores = score.copy()
                best_metrics = met
                best_spec = {
                    "half_life": half_life,
                    "family": family,
                    "mode": "rankblend",
                    "weight": float(w),
                }

print(
    "CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS selected=%s half_life=%s family=%s mode=%s own_weight=%.2f"
    % (
        best_name,
        str(best_spec["half_life"]),
        best_spec["family"],
        best_spec["mode"],
        best_spec["weight"],
    ),
    flush=True,
)

# Refit the exact selected empirical-Bayes recipe on train + validation.
valid_extract = valid_data
combined_data = concatenate_data(train_data, valid_extract)
combined_y = np.concatenate(
    [train_y, valid_y.astype(np.float64, copy=False)]
)

test = load("test")
test_data = extract(test)
test_users = np.asarray(test.user_id, dtype=np.int64)

if best_spec["mode"] == "incumbent":
    own_valid_selected = raw_by_key[(None, "content_hierarchy")]
    own_test = None
    test_scores = np.load(inc_test_path).astype(np.float64)
else:
    prepared_combined = PreparedKeys(combined_data)
    final_model = fit_empirical_bayes(
        prepared_combined, combined_y, best_spec["half_life"]
    )
    own_test = predict_families(
        final_model, test_data
    )[best_spec["family"]]
    own_valid_selected = raw_by_key[
        (best_spec["half_life"], best_spec["family"])
    ]

    w = best_spec["weight"]
    if best_spec["mode"] == "own":
        test_scores = own_test
    else:
        inc_test = np.load(inc_test_path).astype(np.float64)
        if best_spec["mode"] == "zblend":
            test_scores = (
                (1.0 - w) * standardize(inc_test)
                + w * standardize(own_test)
            )
        else:
            test_scores = (
                (1.0 - w) * within_user_rank(test_users, inc_test)
                + w * within_user_rank(test_users, own_test)
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
    if best_spec["mode"] in ("incumbent", "zblend", "rankblend"):
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_valid_selected, dtype=np.float64),
        )

elapsed = float(time.time() - START)
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))