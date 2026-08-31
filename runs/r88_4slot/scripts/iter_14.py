import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
EPS = 1e-9

NB_FIELDS = [
    "duration_bucket",
    "tab",
    "tag",
    "upload_type",
    "music_type",
    "video_type",
    "hour",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
]

PROFILE_FIELDS = [
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
]

GRAPH_META_FIELDS = [
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
]


def concatenate(splits, getter, dtype=None):
    arrays = [np.asarray(getter(s)) for s in splits]
    out = np.concatenate(arrays)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def concat_cat(splits, field):
    return concatenate(
        splits, lambda s: s.X[field], np.int64
    )


def concat_y(splits):
    return concatenate(splits, lambda s: s.y, np.float64)


def recency_weights(splits, half_life=10.0):
    dates = concatenate(splits, lambda s: s.date, np.int64)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = (len(unique_dates) - 1) - day_index
    w = np.exp2(-age.astype(np.float64) / half_life)
    w /= max(float(np.mean(w)), EPS)
    return w


def safe_logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def rank_within_user(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.r_[
        0,
        np.flatnonzero(
            sorted_users[1:] != sorted_users[:-1]
        ) + 1,
    ]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    positions = (
        np.arange(n, dtype=np.float64)
        - np.repeat(starts, counts)
    )
    denominators = np.repeat(counts - 1, counts)

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    mask = denominators > 0
    ranked_sorted[mask] = (
        positions[mask] / denominators[mask]
    )

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def sparse_pair_fit(
    left, right, right_card, y, w, smoothing, prior
):
    keys = (
        left.astype(np.int64) * np.int64(right_card)
        + right.astype(np.int64)
    )
    order = np.argsort(keys, kind="mergesort")
    sk = keys[order]
    sy = y[order]
    sw = w[order]

    starts = np.r_[
        0, np.flatnonzero(sk[1:] != sk[:-1]) + 1
    ]
    unique_keys = sk[starts]
    counts = np.add.reduceat(sw, starts)
    positives = np.add.reduceat(sw * sy, starts)

    rates = (
        positives + smoothing * prior
    ) / (counts + smoothing)
    reliability = counts / (counts + smoothing)
    return (
        unique_keys,
        safe_logit(rates) - safe_logit(prior),
        reliability,
    )


def sparse_lookup(query, keys, values, default=0.0):
    pos = np.searchsorted(keys, query)
    valid = pos < len(keys)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        valid[valid_indices] = (
            keys[pos[valid_indices]]
            == query[valid_indices]
        )

    out = np.full(
        len(query), default, dtype=np.float64
    )
    out[valid] = values[pos[valid]]
    return out


class DayConsensusNaiveBayes:
    """
    A generative classifier. Each field contributes a class-conditional
    log likelihood ratio. Effects are estimated independently by day,
    then combined using a median/mean consensus and shrunk according to
    their temporal disagreement.
    """

    def fit(self, splits):
        y = concat_y(splits)
        dates = concatenate(
            splits, lambda s: s.date, np.int64
        )
        unique_dates = np.unique(dates)
        self.tables = {}

        global_prior = (
            np.sum(y) + 20.0 * 0.33
        ) / (len(y) + 20.0)
        self.prior_log_odds = safe_logit(global_prior)

        for field in NB_FIELDS:
            ids = concat_cat(splits, field)
            card = int(FEATURE_CARDINALITIES[field])
            daily_effects = []

            for date in unique_dates:
                mask = dates == date
                yd = y[mask]
                xd = ids[mask]
                n_pos = float(np.sum(yd))
                n_neg = float(len(yd) - n_pos)

                pos_counts = np.bincount(
                    xd,
                    weights=yd,
                    minlength=card,
                ).astype(np.float64)
                neg_counts = np.bincount(
                    xd,
                    weights=1.0 - yd,
                    minlength=card,
                ).astype(np.float64)

                alpha = 1.5
                log_p_pos = np.log(
                    (pos_counts + alpha)
                    / (n_pos + alpha * card)
                )
                log_p_neg = np.log(
                    (neg_counts + alpha)
                    / (n_neg + alpha * card)
                )
                daily_effects.append(log_p_pos - log_p_neg)

            daily = np.stack(daily_effects, axis=0)
            median_effect = np.median(daily, axis=0)
            mean_effect = np.mean(daily, axis=0)
            temporal_scale = np.std(daily, axis=0)

            consensus = (
                0.65 * median_effect + 0.35 * mean_effect
            )
            stability = 1.0 / (
                1.0 + 1.75 * temporal_scale
            )

            total_counts = np.bincount(
                ids, minlength=card
            ).astype(np.float64)
            support = total_counts / (total_counts + 30.0)

            self.tables[field] = (
                consensus * stability * support
            )

        return self

    def predict(self, split):
        score = np.full(
            len(split.user_id),
            self.prior_log_odds,
            dtype=np.float64,
        )

        for field in NB_FIELDS:
            ids = np.asarray(
                split.X[field], dtype=np.int64
            )
            score += self.tables[field][ids]

        return score


class PreferenceKernel:
    """
    Each user gets a positive-versus-negative preference profile over
    several content spaces. A candidate is scored by its similarity to
    those learned profiles, with global content propensity as backoff.
    """

    def fit(self, splits):
        y = concat_y(splits)
        w = recency_weights(splits, half_life=9.0)
        users = concat_cat(splits, "user_id")
        self.prior = float(
            np.sum(w * y) / np.sum(w)
        )
        self.tables = {}

        for field in PROFILE_FIELDS:
            ids = concat_cat(splits, field)
            card = int(FEATURE_CARDINALITIES[field])

            counts = np.bincount(
                ids, weights=w, minlength=card
            ).astype(np.float64)
            positives = np.bincount(
                ids, weights=w * y, minlength=card
            ).astype(np.float64)
            global_rate = (
                positives + 16.0 * self.prior
            ) / (counts + 16.0)
            global_delta = (
                safe_logit(global_rate)
                - safe_logit(self.prior)
            )
            global_rel = counts / (counts + 16.0)

            keys, pair_delta, pair_rel = sparse_pair_fit(
                users,
                ids,
                card,
                y,
                w,
                smoothing=7.0,
                prior=self.prior,
            )
            self.tables[field] = (
                card,
                global_delta,
                global_rel,
                keys,
                pair_delta,
                pair_rel,
            )

        return self

    def predict(self, split):
        users = np.asarray(
            split.X["user_id"], dtype=np.int64
        )
        numerator = np.zeros(len(users), dtype=np.float64)
        denominator = np.zeros(len(users), dtype=np.float64)

        field_weight = {
            "author_id": 1.35,
            "tag": 1.10,
            "tab": 0.90,
            "duration_bucket": 0.80,
            "upload_type": 0.65,
            "music_type": 0.50,
            "video_type": 0.45,
            "onehot_feat3": 0.65,
            "onehot_feat8": 0.65,
        }

        for field in PROFILE_FIELDS:
            ids = np.asarray(
                split.X[field], dtype=np.int64
            )
            (
                card,
                global_delta,
                global_rel,
                keys,
                pair_delta,
                pair_rel,
            ) = self.tables[field]

            query = (
                users.astype(np.int64) * np.int64(card)
                + ids.astype(np.int64)
            )
            personalized = sparse_lookup(
                query, keys, pair_delta
            )
            reliability = sparse_lookup(
                query, keys, pair_rel
            )

            # Personalized profile similarity, continuously backed off
            # to the population content preference.
            effect = (
                reliability * personalized
                + (1.0 - reliability) * global_delta[ids]
            )
            strength = field_weight[field] * (
                0.20
                + 0.80
                * np.maximum(
                    reliability, global_rel[ids]
                )
            )
            numerator += strength * effect
            denominator += strength

        return numerator / np.maximum(denominator, 0.2)


class HeterogeneousGraphPropagation:
    """
    Video labels initialize a heterogeneous graph. Repeated message
    passing video -> metadata -> video smooths noisy video propensities
    toward metadata neighborhoods without imposing a low-rank factor
    model.
    """

    def fit(self, splits):
        y = concat_y(splits)
        w = recency_weights(splits, half_life=11.0)
        videos = concat_cat(splits, "video_id")
        n_video = int(FEATURE_CARDINALITIES["video_id"])

        self.prior = float(
            np.sum(w * y) / np.sum(w)
        )
        prior_logit = safe_logit(self.prior)

        v_count = np.bincount(
            videos, weights=w, minlength=n_video
        ).astype(np.float64)
        v_pos = np.bincount(
            videos, weights=w * y, minlength=n_video
        ).astype(np.float64)
        v_rate = (
            v_pos + 20.0 * self.prior
        ) / (v_count + 20.0)
        video_signal = safe_logit(v_rate) - prior_logit
        video_support = v_count / (v_count + 20.0)

        field_arrays = {
            field: concat_cat(splits, field)
            for field in GRAPH_META_FIELDS
        }

        # Aggregate each video's training rows to define its metadata
        # membership. In this dataset video metadata is nearly constant;
        # bincount-weighted propagation remains robust to exceptions.
        for _ in range(4):
            meta_messages = {}
            for field in GRAPH_META_FIELDS:
                ids = field_arrays[field]
                card = int(FEATURE_CARDINALITIES[field])

                row_message = video_signal[videos]
                meta_count = np.bincount(
                    ids, weights=w, minlength=card
                ).astype(np.float64)
                meta_sum = np.bincount(
                    ids,
                    weights=w * row_message,
                    minlength=card,
                ).astype(np.float64)
                meta_messages[field] = (
                    meta_sum / np.maximum(meta_count, EPS)
                )

            propagated_sum = np.zeros(
                n_video, dtype=np.float64
            )
            propagated_weight = np.zeros(
                n_video, dtype=np.float64
            )

            for field in GRAPH_META_FIELDS:
                ids = field_arrays[field]
                msg = meta_messages[field][ids]
                propagated_sum += np.bincount(
                    videos,
                    weights=w * msg,
                    minlength=n_video,
                )
                propagated_weight += np.bincount(
                    videos, weights=w, minlength=n_video
                )

            propagated = propagated_sum / np.maximum(
                propagated_weight, EPS
            )
            video_signal = (
                0.62 * video_support * video_signal
                + (1.0 - 0.62 * video_support)
                * propagated
            )

        self.video_signal = video_signal
        self.video_support = video_support
        self.meta_tables = {}

        for field in GRAPH_META_FIELDS:
            ids = field_arrays[field]
            card = int(FEATURE_CARDINALITIES[field])
            counts = np.bincount(
                ids, weights=w, minlength=card
            ).astype(np.float64)
            sums = np.bincount(
                ids,
                weights=w * video_signal[videos],
                minlength=card,
            ).astype(np.float64)
            self.meta_tables[field] = (
                sums / np.maximum(counts, EPS),
                counts / (counts + 25.0),
            )

        return self

    def predict(self, split):
        videos = np.asarray(
            split.X["video_id"], dtype=np.int64
        )
        direct = self.video_signal[videos]
        support = self.video_support[videos]

        meta_sum = np.zeros(len(videos), dtype=np.float64)
        meta_weight = np.zeros(len(videos), dtype=np.float64)

        for field in GRAPH_META_FIELDS:
            ids = np.asarray(
                split.X[field], dtype=np.int64
            )
            table, reliability = self.meta_tables[field]
            rel = reliability[ids]
            meta_sum += rel * table[ids]
            meta_weight += rel

        meta = meta_sum / np.maximum(meta_weight, 0.25)
        return support * direct + (1.0 - support) * meta


def fit_family(name, splits):
    if name == "day_consensus_nb":
        return DayConsensusNaiveBayes().fit(splits)
    if name == "preference_kernel":
        return PreferenceKernel().fit(splits)
    if name == "graph_propagation":
        return HeterogeneousGraphPropagation().fit(splits)
    raise ValueError(name)


train = load("train")
valid = load("valid")
valid_users = np.asarray(valid.user_id, dtype=np.int64)
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
    raise RuntimeError(
        "Trusted incumbent predictions are unavailable"
    )

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
inc_valid_rank = rank_within_user(
    valid_users, inc_valid
)

family_names = [
    "day_consensus_nb",
    "preference_kernel",
    "graph_propagation",
]
alphas = np.linspace(0.0, 0.75, 13)

candidate_log = {}
raw_valid_by_family = {}

best_primary = -np.inf
best_name = None
best_alpha = None
best_scores = None
best_raw = None
best_metrics = None

for name in family_names:
    model = fit_family(name, [train])
    raw = model.predict(valid)
    raw_valid_by_family[name] = raw

    raw_metrics = evaluate(
        valid_users, valid_y, raw
    )
    candidate_log[name + "_standalone"] = float(
        raw_metrics["primary"]
    )

    raw_rank = rank_within_user(valid_users, raw)
    local_best = -np.inf
    local_alpha = 0.0

    for alpha in alphas:
        blended = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * raw_rank
        )
        metrics = evaluate(
            valid_users, valid_y, blended
        )
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_alpha = float(alpha)
            best_scores = blended.copy()
            best_raw = raw.copy()
            best_metrics = metrics

    candidate_log[name + "_best_blend"] = float(
        local_best
    )
    candidate_log[name + "_alpha"] = float(
        local_alpha
    )

# A cross-family rank ensemble is itself a distinct consensus prediction:
# it rewards candidates supported by all three mechanisms and suppresses
# idiosyncratic high scores from one estimator.
family_ranks = [
    rank_within_user(valid_users, raw_valid_by_family[n])
    for n in family_names
]
consensus_rank = (
    0.34 * family_ranks[0]
    + 0.38 * family_ranks[1]
    + 0.28 * family_ranks[2]
)

consensus_raw_metrics = evaluate(
    valid_users, valid_y, consensus_rank
)
candidate_log["cross_family_consensus_standalone"] = float(
    consensus_raw_metrics["primary"]
)

for alpha in alphas:
    blended = (
        (1.0 - alpha) * inc_valid_rank
        + alpha * consensus_rank
    )
    metrics = evaluate(valid_users, valid_y, blended)
    primary = float(metrics["primary"])
    if primary > best_primary:
        best_primary = primary
        best_name = "cross_family_consensus"
        best_alpha = float(alpha)
        best_scores = blended.copy()
        best_raw = consensus_rank.copy()
        best_metrics = metrics

candidate_log["winner_alpha"] = float(best_alpha)
candidate_log["winner_primary"] = float(best_primary)

print(
    "CANDIDATES "
    + json.dumps(candidate_log, sort_keys=True)
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "blend_alpha": best_alpha,
            "standalone_scores": {
                n: candidate_log[n + "_standalone"]
                for n in family_names
            },
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )

# Refit the identical selected recipe on train + validation, then apply
# the validation-selected incumbent blend weight to test predictions.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)
inc_test_rank = rank_within_user(test_users, inc_test)

if best_name == "cross_family_consensus":
    test_family_ranks = []
    for name in family_names:
        model = fit_family(name, [train, valid])
        raw_test = model.predict(test)
        test_family_ranks.append(
            rank_within_user(test_users, raw_test)
        )
    own_test_rank = (
        0.34 * test_family_ranks[0]
        + 0.38 * test_family_ranks[1]
        + 0.28 * test_family_ranks[2]
    )
else:
    final_model = fit_family(
        best_name, [train, valid]
    )
    raw_test = final_model.predict(test)
    own_test_rank = rank_within_user(
        test_users, raw_test
    )

test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * own_test_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
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