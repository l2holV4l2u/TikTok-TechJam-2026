import os
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
HALF_LIFE = 8.0
EPS = 1e-5

ENTITY_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
]

PAIR_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
]

SMOOTH_ENTITY = 18.0
SMOOTH_PAIR = 12.0
SVD_RANK = 24


def rank_within_user(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    su = user_ids[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    mask = repeated_counts > 1
    ranked[mask] = positions[mask] / (repeated_counts[mask] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def date_weights(splits):
    all_dates = np.concatenate([np.asarray(s.date) for s in splits])
    unique_dates = np.unique(all_dates)
    day_index = np.searchsorted(unique_dates, all_dates).astype(np.float32)
    age = float(len(unique_dates) - 1) - day_index
    w = np.exp2(-age / HALF_LIFE).astype(np.float64)
    w /= max(float(w.mean()), EPS)
    return w


def concatenate_labels(splits):
    return np.concatenate([
        np.asarray(s.y, dtype=np.float64) for s in splits
    ])


def concatenate_cat(splits, field):
    return np.concatenate([
        np.asarray(s.X[field], dtype=np.int64) for s in splits
    ])


def weighted_global(y, w):
    return float(np.sum(y * w) / max(np.sum(w), EPS))


def safe_logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def fit_single_table(ids, y, w, cardinality, smoothing, prior):
    counts = np.bincount(
        ids, weights=w, minlength=cardinality
    ).astype(np.float64)
    positives = np.bincount(
        ids, weights=w * y, minlength=cardinality
    ).astype(np.float64)
    rates = (positives + smoothing * prior) / (
        counts + smoothing
    )
    reliability = counts / (counts + smoothing)
    return safe_logit(rates), reliability


def fit_sparse_pair_table(left, right, right_card, y, w, smoothing, prior):
    keys = (
        left.astype(np.int64) * np.int64(right_card)
        + right.astype(np.int64)
    )
    order = np.argsort(keys, kind="mergesort")
    sk = keys[order]
    sy = y[order]
    sw = w[order]

    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    unique_keys = sk[starts]
    counts = np.add.reduceat(sw, starts)
    positives = np.add.reduceat(sw * sy, starts)

    rates = (positives + smoothing * prior) / (
        counts + smoothing
    )
    logits = safe_logit(rates)
    reliability = counts / (counts + smoothing)
    return unique_keys, logits, reliability


def sparse_lookup(keys, table_keys, table_values, default=0.0):
    pos = np.searchsorted(table_keys, keys)
    valid = pos < len(table_keys)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx):
        valid[valid_idx] = (
            table_keys[pos[valid_idx]] == keys[valid_idx]
        )
    out = np.full(len(keys), default, dtype=np.float64)
    out[valid] = table_values[pos[valid]]
    return out


class EntityBayes:
    def fit(self, splits):
        y = concatenate_labels(splits)
        w = date_weights(splits)
        self.prior = weighted_global(y, w)
        self.prior_logit = safe_logit(self.prior)
        self.tables = {}

        for field in ENTITY_FIELDS:
            ids = concatenate_cat(splits, field)
            logits, reliability = fit_single_table(
                ids,
                y,
                w,
                int(FEATURE_CARDINALITIES[field]),
                SMOOTH_ENTITY,
                self.prior,
            )
            self.tables[field] = (logits, reliability)
        return self

    def predict(self, split):
        score = np.zeros(len(split.user_id), dtype=np.float64)
        total_weight = np.zeros(len(split.user_id), dtype=np.float64)

        field_weights = {
            "video_id": 1.45,
            "author_id": 1.25,
            "tag": 0.85,
            "tab": 0.85,
            "duration_bucket": 0.55,
            "upload_type": 0.55,
            "music_type": 0.35,
            "onehot_feat3": 0.55,
            "onehot_feat8": 0.50,
        }

        for field in ENTITY_FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            logits, reliability = self.tables[field]
            strength = field_weights[field] * reliability[ids]
            score += strength * (logits[ids] - self.prior_logit)
            total_weight += strength

        return score / np.maximum(total_weight, 0.25)


class PersonalizedBayes:
    def fit(self, splits):
        y = concatenate_labels(splits)
        w = date_weights(splits)
        users = concatenate_cat(splits, "user_id")
        self.prior = weighted_global(y, w)
        self.prior_logit = safe_logit(self.prior)
        self.entity_tables = {}
        self.pair_tables = {}

        for field in PAIR_FIELDS:
            ids = concatenate_cat(splits, field)
            logits, reliability = fit_single_table(
                ids,
                y,
                w,
                int(FEATURE_CARDINALITIES[field]),
                SMOOTH_ENTITY,
                self.prior,
            )
            self.entity_tables[field] = (logits, reliability)

            right_card = int(FEATURE_CARDINALITIES[field])
            keys, pair_logits, pair_rel = fit_sparse_pair_table(
                users,
                ids,
                right_card,
                y,
                w,
                SMOOTH_PAIR,
                self.prior,
            )
            self.pair_tables[field] = (
                right_card,
                keys,
                pair_logits,
                pair_rel,
            )
        return self

    def predict(self, split):
        users = np.asarray(split.X["user_id"], dtype=np.int64)
        numerator = np.zeros(len(users), dtype=np.float64)
        denominator = np.zeros(len(users), dtype=np.float64)

        field_weights = {
            "video_id": 1.35,
            "author_id": 1.30,
            "tag": 0.90,
            "tab": 0.85,
            "duration_bucket": 0.65,
            "upload_type": 0.55,
        }

        for field in PAIR_FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            entity_logits, entity_rel = self.entity_tables[field]
            right_card, table_keys, pair_logits, pair_rel = (
                self.pair_tables[field]
            )

            query_keys = (
                users.astype(np.int64) * np.int64(right_card)
                + ids.astype(np.int64)
            )
            pair_delta = sparse_lookup(
                query_keys,
                table_keys,
                pair_logits - self.prior_logit,
                default=0.0,
            )
            pair_strength = sparse_lookup(
                query_keys,
                table_keys,
                pair_rel,
                default=0.0,
            )

            entity_delta = entity_logits[ids] - self.prior_logit
            # Unseen or weak user-context pairs back off continuously to
            # the corresponding global entity propensity.
            delta = (
                pair_strength * pair_delta
                + (1.0 - pair_strength) * entity_delta
            )
            strength = field_weights[field] * (
                0.35 + 0.65 * np.maximum(
                    pair_strength, entity_rel[ids]
                )
            )
            numerator += strength * delta
            denominator += strength

        return numerator / np.maximum(denominator, 0.25)


class LatentSVD:
    def fit(self, splits):
        y = concatenate_labels(splits)
        w = date_weights(splits)
        users = concatenate_cat(splits, "user_id")
        videos = concatenate_cat(splits, "video_id")

        self.n_users = int(FEATURE_CARDINALITIES["user_id"])
        self.n_videos = int(FEATURE_CARDINALITIES["video_id"])
        self.prior = weighted_global(y, w)

        # Remove the recency-weighted global video propensity before SVD,
        # so the factors represent personalized deviations rather than
        # merely reproducing item popularity.
        video_logits, video_rel = fit_single_table(
            videos,
            y,
            w,
            self.n_videos,
            SMOOTH_ENTITY,
            self.prior,
        )
        self.video_score = video_logits - safe_logit(self.prior)
        self.video_rel = video_rel

        video_rate = 1.0 / (1.0 + np.exp(-video_logits[videos]))
        residual = (y - video_rate) * np.sqrt(w)

        sum_matrix = sparse.coo_matrix(
            (residual, (users, videos)),
            shape=(self.n_users, self.n_videos),
        ).tocsr()
        count_matrix = sparse.coo_matrix(
            (np.sqrt(w), (users, videos)),
            shape=(self.n_users, self.n_videos),
        ).tocsr()

        inv_counts = count_matrix.copy()
        inv_counts.data = 1.0 / np.maximum(inv_counts.data, EPS)
        matrix = sum_matrix.multiply(inv_counts)

        k = min(
            SVD_RANK,
            self.n_users - 1,
            self.n_videos - 1,
        )
        u, singular, vt = svds(
            matrix.astype(np.float32),
            k=k,
            tol=2e-3,
            maxiter=350,
            return_singular_vectors=True,
            random_state=9473,
        )
        order = np.argsort(singular)[::-1]
        singular = singular[order]
        u = u[:, order]
        vt = vt[order]

        root_s = np.sqrt(np.maximum(singular, 0.0))
        self.user_factors = (
            u * root_s[None, :]
        ).astype(np.float32)
        self.video_factors = (
            vt.T * root_s[None, :]
        ).astype(np.float32)
        return self

    def predict(self, split):
        users = np.asarray(split.X["user_id"], dtype=np.int64)
        videos = np.asarray(split.X["video_id"], dtype=np.int64)

        latent = np.sum(
            self.user_factors[users]
            * self.video_factors[videos],
            axis=1,
            dtype=np.float32,
        ).astype(np.float64)

        return (
            0.75 * self.video_score[videos]
            + 2.0 * latent
        )


def fit_family(name, splits):
    if name == "entity_bayes":
        return EntityBayes().fit(splits)
    if name == "personalized_bayes":
        return PersonalizedBayes().fit(splits)
    if name == "latent_svd":
        return LatentSVD().fit(splits)
    raise ValueError(name)


train = load("train")
valid = load("valid")

valid_users = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent predictions unavailable")

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
inc_valid_rank = rank_within_user(valid_users, inc_valid)

family_names = [
    "entity_bayes",
    "personalized_bayes",
    "latent_svd",
]

raw_valid_scores = {}
candidate_log = {}
best_primary = -np.inf
best_name = None
best_alpha = None
best_valid_scores = None
best_raw_scores = None
best_metrics = None

alphas = np.linspace(0.0, 1.0, 9)

for family_name in family_names:
    model = fit_family(family_name, [train])
    raw = model.predict(valid)
    raw_valid_scores[family_name] = raw

    raw_metrics = evaluate(valid_users, valid_y, raw)
    candidate_log[
        family_name + "_standalone"
    ] = float(raw_metrics["primary"])

    raw_rank = rank_within_user(valid_users, raw)
    local_best = -np.inf
    local_alpha = 0.0

    for alpha in alphas:
        blended = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * raw_rank
        )
        metrics = evaluate(valid_users, valid_y, blended)
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = family_name
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()
            best_raw_scores = raw.copy()
            best_metrics = metrics

    candidate_log[
        family_name + "_best_blend"
    ] = float(local_best)
    candidate_log[
        family_name + "_alpha"
    ] = float(local_alpha)

# Also test rank aggregation of the two explicitly non-parametric
# estimators before choosing the final family recipe.
entity_rank = rank_within_user(
    valid_users, raw_valid_scores["entity_bayes"]
)
personal_rank = rank_within_user(
    valid_users, raw_valid_scores["personalized_bayes"]
)
np_ensemble_rank = 0.4 * entity_rank + 0.6 * personal_rank

for alpha in alphas:
    blended = (
        (1.0 - alpha) * inc_valid_rank
        + alpha * np_ensemble_rank
    )
    metrics = evaluate(valid_users, valid_y, blended)
    primary = float(metrics["primary"])
    if primary > best_primary:
        best_primary = primary
        best_name = "nonparametric_ensemble"
        best_alpha = float(alpha)
        best_valid_scores = blended.copy()
        best_raw_scores = np_ensemble_rank.copy()
        best_metrics = metrics

candidate_log["winner_alpha"] = float(best_alpha)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "winner_alpha": best_alpha,
            "winner_primary": best_primary,
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_raw_scores, dtype=np.float64),
    )

# Refit the identical selected recipe on train+validation and score test.
test = load("test")
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)
inc_test_rank = rank_within_user(
    np.asarray(test.user_id), inc_test
)

if best_name == "nonparametric_ensemble":
    entity_model = fit_family(
        "entity_bayes", [train, valid]
    )
    personal_model = fit_family(
        "personalized_bayes", [train, valid]
    )
    entity_test_rank = rank_within_user(
        np.asarray(test.user_id),
        entity_model.predict(test),
    )
    personal_test_rank = rank_within_user(
        np.asarray(test.user_id),
        personal_model.predict(test),
    )
    raw_test_rank = (
        0.4 * entity_test_rank
        + 0.6 * personal_test_rank
    )
else:
    final_model = fit_family(best_name, [train, valid])
    raw_test = final_model.predict(test)
    raw_test_rank = rank_within_user(
        np.asarray(test.user_id), raw_test
    )

test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * raw_test_rank
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)