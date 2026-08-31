import os
import time
import json
import warnings

import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
warnings.filterwarnings("ignore")
np.random.seed(20260831)

train = load("train")
valid = load("valid")
test = load("test")

train_y = np.asarray(train.y, dtype=np.float64)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)

if inc_valid.shape[0] != valid_users.size:
    raise RuntimeError("Incumbent validation prediction length mismatch")
if inc_test.shape[0] != test_users.size:
    raise RuntimeError("Incumbent test prediction length mismatch")


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def recency_weights(dates, half_life=5.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    day_idx = np.searchsorted(unique_dates, dates)
    age = unique_dates.size - 1 - day_idx
    return np.exp2(-age.astype(np.float64) / float(half_life))


def sparse_lookup(keys, fitted_keys, fitted_values, default=0.0):
    keys = np.asarray(keys, dtype=np.int64)
    pos = np.searchsorted(fitted_keys, keys)
    valid_pos = pos < fitted_keys.size
    out = np.full(keys.size, default, dtype=np.float64)

    rows = np.flatnonzero(valid_pos)
    if rows.size:
        exact = fitted_keys[pos[rows]] == keys[rows]
        rows = rows[exact]
        out[rows] = fitted_values[pos[rows]]
    return out


def within_user_rank(user_ids, scores, tie_breaker=None):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    if tie_breaker is None:
        tie_breaker = rows
    tie_breaker = np.asarray(tie_breaker, dtype=np.int64)

    order = np.lexsort((rows, tie_breaker, scores, user_ids))
    ordered_users = user_ids[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = ordered_users[1:] != ordered_users[:-1]
    starts = np.flatnonzero(starts_flag)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranks = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranks[multi] = (
        positions[multi]
        / (repeated_lengths[multi].astype(np.float64) - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


weights = recency_weights(train.date, half_life=5.0)
global_rate = float(np.sum(weights * train_y) / np.sum(weights))
global_logit = float(logit(global_rate))

train_user = np.asarray(train.X["user_id"], dtype=np.int64)
num_users = FEATURE_CARDINALITIES["user_id"]


# ----------------------------------------------------------------------
# Family 1: behavioral archetype co-clustering.
#
# A low-dimensional representation of each training user's signed response
# distribution over several content fields is clustered into behavioral
# archetypes. Archetype/entity rates then personalize videos and authors
# without fitting a separate high-variance embedding for every interaction.
# ----------------------------------------------------------------------

ARCHETYPE_FIELDS = (
    ("tab", 32),
    ("duration_bucket", 32),
    ("tag", 64),
    ("upload_type", 24),
    ("onehot_feat3", 96),
)

row_parts = []
col_parts = []
value_parts = []
column_offset = 0
centered_label = train_y - global_rate

for field, bins in ARCHETYPE_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    hashed = np.mod(
        ids * np.int64(2654435761) + np.int64(97531),
        np.int64(bins),
    )
    row_parts.append(train_user)
    col_parts.append(hashed + column_offset)
    value_parts.append(weights * centered_label)
    column_offset += bins

profile_matrix = sparse.coo_matrix(
    (
        np.concatenate(value_parts).astype(np.float32),
        (
            np.concatenate(row_parts),
            np.concatenate(col_parts),
        ),
    ),
    shape=(num_users, column_offset),
    dtype=np.float32,
).tocsr()

user_weight = np.bincount(
    train_user, weights=weights, minlength=num_users
).astype(np.float64)
normalizer = 1.0 / np.sqrt(np.maximum(user_weight, 1.0))
profile_matrix = sparse.diags(normalizer.astype(np.float32)).dot(profile_matrix)

active_users = np.flatnonzero(user_weight > 0)
svd = TruncatedSVD(
    n_components=18,
    algorithm="randomized",
    n_iter=4,
    random_state=20260831,
)
latent_all = svd.fit_transform(profile_matrix).astype(np.float32)

kmeans = MiniBatchKMeans(
    n_clusters=48,
    batch_size=4096,
    n_init=3,
    max_iter=80,
    random_state=20260831,
)
active_clusters = kmeans.fit_predict(latent_all[active_users])
user_cluster = np.zeros(num_users, dtype=np.int64)
user_cluster[active_users] = active_clusters.astype(np.int64)

cluster_mass = np.bincount(active_clusters, minlength=48)
fallback_cluster = int(np.argmax(cluster_mass))
user_cluster[user_weight == 0] = fallback_cluster

ARCHETYPE_ENTITY_SPECS = (
    ("video_id", 28.0, 65.0, 0.56, 0.42),
    ("author_id", 38.0, 85.0, 0.27, 0.27),
    ("tag", 55.0, 110.0, 0.10, 0.17),
    ("duration_bucket", 70.0, 130.0, 0.08, 0.12),
)

archetype_models = {}
train_clusters = user_cluster[train_user]

for field, entity_prior, cross_prior, base_coef, cross_coef in ARCHETYPE_ENTITY_SPECS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = FEATURE_CARDINALITIES[field]

    entity_count = np.bincount(
        ids, weights=weights, minlength=card
    ).astype(np.float64)
    entity_positive = np.bincount(
        ids, weights=weights * train_y, minlength=card
    ).astype(np.float64)
    parent_rate = (
        entity_positive + entity_prior * global_rate
    ) / (entity_count + entity_prior)
    parent_score = logit(parent_rate) - global_logit

    joint_key = train_clusters * card + ids
    unique_key, inverse = np.unique(joint_key, return_inverse=True)
    joint_count = np.bincount(
        inverse, weights=weights, minlength=unique_key.size
    ).astype(np.float64)
    joint_positive = np.bincount(
        inverse,
        weights=weights * train_y,
        minlength=unique_key.size,
    ).astype(np.float64)

    parent_for_joint = parent_rate[unique_key % card]
    joint_rate = (
        joint_positive + cross_prior * parent_for_joint
    ) / (joint_count + cross_prior)
    joint_deviation = logit(joint_rate) - logit(parent_for_joint)

    archetype_models[field] = {
        "card": card,
        "parent_score": parent_score.astype(np.float32),
        "keys": unique_key,
        "deviation": joint_deviation.astype(np.float32),
        "base_coef": base_coef,
        "cross_coef": cross_coef,
    }


def predict_archetype(split):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    clusters = user_cluster[np.minimum(users, num_users - 1)]
    result = np.zeros(users.size, dtype=np.float64)

    for field, model in archetype_models.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += model["base_coef"] * model["parent_score"][ids]

        key = clusters * model["card"] + ids
        deviation = sparse_lookup(
            key, model["keys"], model["deviation"], default=0.0
        )
        result += model["cross_coef"] * deviation

    return result


archetype_valid = predict_archetype(valid)
archetype_test = predict_archetype(test)


# ----------------------------------------------------------------------
# Family 2: explicit personalized content-preference tables.
#
# This non-parametric model estimates each user's lift for content values,
# backing every user/value rate off to its global content-value rate. Unlike
# latent co-clustering, it preserves sharp preferences such as a particular
# user's duration or tab affinity.
# ----------------------------------------------------------------------

PREFERENCE_SPECS = (
    ("tab", 18.0, 0.30),
    ("duration_bucket", 22.0, 0.30),
    ("tag", 30.0, 0.22),
    ("upload_type", 28.0, 0.12),
    ("onehot_feat1", 25.0, 0.10),
)

preference_models = {}

for field, cross_prior, coefficient in PREFERENCE_SPECS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = FEATURE_CARDINALITIES[field]

    field_count = np.bincount(
        ids, weights=weights, minlength=card
    ).astype(np.float64)
    field_positive = np.bincount(
        ids, weights=weights * train_y, minlength=card
    ).astype(np.float64)
    field_rate = (
        field_positive + 45.0 * global_rate
    ) / (field_count + 45.0)

    keys = train_user * card + ids
    unique_key, inverse = np.unique(keys, return_inverse=True)
    joint_count = np.bincount(
        inverse, weights=weights, minlength=unique_key.size
    ).astype(np.float64)
    joint_positive = np.bincount(
        inverse,
        weights=weights * train_y,
        minlength=unique_key.size,
    ).astype(np.float64)

    parent = field_rate[unique_key % card]
    joint_rate = (
        joint_positive + cross_prior * parent
    ) / (joint_count + cross_prior)

    deviation = logit(joint_rate) - logit(parent)
    reliability = np.sqrt(joint_count / (joint_count + cross_prior))
    deviation *= reliability

    preference_models[field] = {
        "card": card,
        "keys": unique_key,
        "deviation": deviation.astype(np.float32),
        "coefficient": coefficient,
    }


def predict_preferences(split):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    result = np.zeros(users.size, dtype=np.float64)

    for field, model in preference_models.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        key = users * model["card"] + ids
        result += model["coefficient"] * sparse_lookup(
            key, model["keys"], model["deviation"], default=0.0
        )

    return result


preference_valid = predict_preferences(valid)
preference_test = predict_preferences(test)


# ----------------------------------------------------------------------
# Family 3: relational graph smoothing of video quality.
#
# Video target rates are repeatedly shrunk toward weighted neighborhoods
# induced by shared author, tag, duration and upload type. This transfers
# evidence between related videos but retains direct video evidence, forming
# predictions through graph diffusion rather than user embeddings or exact
# user/value tables.
# ----------------------------------------------------------------------

video_train = np.asarray(train.X["video_id"], dtype=np.int64)
video_card = FEATURE_CARDINALITIES["video_id"]

video_count = np.bincount(
    video_train, weights=weights, minlength=video_card
).astype(np.float64)
video_positive = np.bincount(
    video_train, weights=weights * train_y, minlength=video_card
).astype(np.float64)
video_rate = (
    video_positive + 22.0 * global_rate
) / (video_count + 22.0)
video_score = logit(video_rate) - global_logit

GRAPH_FIELDS = (
    ("author_id", 0.38),
    ("tag", 0.24),
    ("duration_bucket", 0.18),
    ("upload_type", 0.12),
    ("tab", 0.08),
)

# Associate each video with the count-weighted modal value of each relation.
video_relations = {}
for field, _ in GRAPH_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = FEATURE_CARDINALITIES[field]
    pair_key = video_train * card + ids
    unique_pair, inverse = np.unique(pair_key, return_inverse=True)
    pair_mass = np.bincount(
        inverse, weights=weights, minlength=unique_pair.size
    ).astype(np.float64)

    pair_video = unique_pair // card
    pair_value = unique_pair % card

    order = np.lexsort((-pair_mass, pair_video))
    ordered_video = pair_video[order]
    first = np.empty(order.size, dtype=bool)
    first[0] = True
    first[1:] = ordered_video[1:] != ordered_video[:-1]
    chosen = order[first]

    relation = np.zeros(video_card, dtype=np.int64)
    relation[pair_video[chosen]] = pair_value[chosen]
    video_relations[field] = relation

direct_score = video_score.copy()
for _ in range(5):
    neighborhood = np.zeros(video_card, dtype=np.float64)
    total_coef = 0.0

    for field, coefficient in GRAPH_FIELDS:
        relation = video_relations[field]
        card = FEATURE_CARDINALITIES[field]

        group_mass = np.bincount(
            relation,
            weights=np.maximum(video_count, 0.1),
            minlength=card,
        ).astype(np.float64)
        group_total = np.bincount(
            relation,
            weights=np.maximum(video_count, 0.1) * video_score,
            minlength=card,
        ).astype(np.float64)
        group_score = group_total / np.maximum(group_mass, 1e-8)

        neighborhood += coefficient * group_score[relation]
        total_coef += coefficient

    neighborhood /= max(total_coef, 1e-8)
    confidence = video_count / (video_count + 40.0)
    video_score = confidence * direct_score + (1.0 - confidence) * neighborhood


def predict_graph(split):
    video = np.asarray(split.X["video_id"], dtype=np.int64)
    return video_score[video].astype(np.float64)


graph_valid = predict_graph(valid)
graph_test = predict_graph(test)


# ----------------------------------------------------------------------
# Compare each family alone, each family blended with the trusted incumbent,
# and a cross-family rank ensemble. Rank aggregation puts heterogeneous
# prediction scales on a common within-user scale, matching the metric.
# ----------------------------------------------------------------------

valid_video = np.asarray(valid.video_id, dtype=np.int64)
test_video = np.asarray(test.video_id, dtype=np.int64)

valid_rank_inc = within_user_rank(valid_users, inc_valid, valid_video)
test_rank_inc = within_user_rank(test_users, inc_test, test_video)

families = {
    "archetype_coclustering": (archetype_valid, archetype_test),
    "personalized_content_tables": (preference_valid, preference_test),
    "graph_smoothed_video_quality": (graph_valid, graph_test),
}

valid_family_ranks = {}
test_family_ranks = {}
for name, (va_score, te_score) in families.items():
    valid_family_ranks[name] = within_user_rank(
        valid_users, va_score, valid_video
    )
    test_family_ranks[name] = within_user_rank(
        test_users, te_score, test_video
    )

valid_family_ranks["cross_family_ensemble"] = (
    0.45 * valid_family_ranks["archetype_coclustering"]
    + 0.30 * valid_family_ranks["personalized_content_tables"]
    + 0.25 * valid_family_ranks["graph_smoothed_video_quality"]
)
test_family_ranks["cross_family_ensemble"] = (
    0.45 * test_family_ranks["archetype_coclustering"]
    + 0.30 * test_family_ranks["personalized_content_tables"]
    + 0.25 * test_family_ranks["graph_smoothed_video_quality"]
)

candidate_summary = {}
all_candidates = {}
blend_alphas = (0.0, 0.08, 0.15, 0.25, 0.40)

inc_metric = evaluate(valid_users, valid_y, valid_rank_inc)
candidate_summary["trusted_incumbent"] = float(inc_metric["primary"])
all_candidates["trusted_incumbent"] = {
    "valid": valid_rank_inc,
    "test": test_rank_inc,
    "raw_valid": archetype_valid,
    "metric": inc_metric,
}

for name in valid_family_ranks:
    raw_metric = evaluate(valid_users, valid_y, valid_family_ranks[name])
    candidate_summary[name + "_raw"] = float(raw_metric["primary"])

    best_name = None
    best_metric = None
    best_valid_score = None
    best_test_score = None

    for alpha in blend_alphas:
        blend_valid = (
            (1.0 - alpha) * valid_rank_inc
            + alpha * valid_family_ranks[name]
        )
        metric = evaluate(valid_users, valid_y, blend_valid)

        if best_metric is None or metric["primary"] > best_metric["primary"]:
            best_name = name + "_blend_" + str(alpha)
            best_metric = metric
            best_valid_score = blend_valid
            best_test_score = (
                (1.0 - alpha) * test_rank_inc
                + alpha * test_family_ranks[name]
            )

    candidate_summary[name + "_best_blend"] = float(best_metric["primary"])

    raw_valid_for_save = (
        archetype_valid
        if name == "archetype_coclustering"
        else preference_valid
        if name == "personalized_content_tables"
        else graph_valid
        if name == "graph_smoothed_video_quality"
        else valid_family_ranks[name]
    )

    all_candidates[best_name] = {
        "valid": best_valid_score,
        "test": best_test_score,
        "raw_valid": raw_valid_for_save,
        "metric": best_metric,
    }

winner_name = max(
    all_candidates,
    key=lambda key: all_candidates[key]["metric"]["primary"],
)
winner = all_candidates[winner_name]
valid_scores = np.asarray(winner["valid"], dtype=np.float64)
test_scores = np.asarray(winner["test"], dtype=np.float64)
metrics = evaluate(valid_users, valid_y, valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "svd_explained_variance": float(
                np.sum(svd.explained_variance_ratio_)
            ),
            "behavioral_archetypes": 48,
            "active_training_users": int(active_users.size),
            "winner": winner_name,
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_summary, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        test_scores.astype(np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(winner["raw_valid"], dtype=np.float64),
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
        }
    )
)