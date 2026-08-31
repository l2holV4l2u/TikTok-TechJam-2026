import os
import time
import json
import numpy as np
import lightgbm as lgb

from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()

CONTENT_FIELDS = [
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "hour",
]

EB_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
    "onehot_feat3",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.015, 0.985)
    return np.log(p) - np.log1p(-p)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    sizes = np.diff(np.r_[starts, n])

    ranks_sorted = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks_sorted
    return result


def extract_history(split_name):
    video_hist = historical_features(split_name, key="video_id")
    author_hist = historical_features(split_name, key="author_id")

    selected = []
    selected_names = []

    for prefix, hist in (("video", video_hist), ("author", author_hist)):
        preferred = [
            k for k in hist
            if (
                "count_log1p" in k
                or "long_view_rate" in k
                or "is_like_rate" in k
                or "is_click_rate" in k
                or "is_follow_rate" in k
                or "is_hate_rate" in k
            )
        ]
        preferred = sorted(preferred)

        for key in preferred:
            arr = np.asarray(hist[key], dtype=np.float32)
            selected.append(arr)
            selected_names.append(prefix + ":" + key)

    if not selected:
        raise RuntimeError("No historical features were returned")

    return np.column_stack(selected).astype(np.float32), selected_names


def make_content_matrix(split, split_name):
    cats = np.column_stack(
        [np.asarray(split.X[f], dtype=np.float32) for f in CONTENT_FIELDS]
    )

    numeric = []
    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float64)
        finite = np.isfinite(x)
        clean = np.where(finite, np.maximum(x, 0.0), 0.0)
        numeric.append(np.log1p(clean).astype(np.float32))
        numeric.append((~finite).astype(np.float32))

    numeric = np.column_stack(numeric)
    hist, hist_names = extract_history(split_name)

    return np.ascontiguousarray(
        np.column_stack([cats, numeric, hist]), dtype=np.float32
    ), hist_names


class EmpiricalBayesModel:
    def __init__(self, train, sample_weight):
        y = np.asarray(train.y, dtype=np.float64)
        w = np.asarray(sample_weight, dtype=np.float64)

        self.global_rate = float(np.dot(w, y) / np.sum(w))
        self.global_logit = float(safe_logit(self.global_rate))
        self.tables = {}
        self.coefficients = {
            "video_id": 0.95,
            "author_id": 0.75,
            "tag": 0.45,
            "duration_bucket": 0.30,
            "tab": 0.45,
            "upload_type": 0.30,
            "onehot_feat3": 0.35,
        }
        smoothing = {
            "video_id": 80.0,
            "author_id": 100.0,
            "tag": 250.0,
            "duration_bucket": 400.0,
            "tab": 500.0,
            "upload_type": 400.0,
            "onehot_feat3": 180.0,
        }

        for field in EB_FIELDS:
            ids = np.asarray(train.X[field], dtype=np.int64)
            card = FEATURE_CARDINALITIES[field]
            counts = np.bincount(
                ids, weights=w, minlength=card
            ).astype(np.float64)
            positives = np.bincount(
                ids, weights=w * y, minlength=card
            ).astype(np.float64)

            alpha = smoothing[field]
            rate = (
                positives + alpha * self.global_rate
            ) / (counts + alpha)
            self.tables[field] = safe_logit(rate) - self.global_logit

    def predict(self, split):
        score = np.zeros(len(split.user_id), dtype=np.float64)
        for field in EB_FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            table = self.tables[field]
            ids = np.minimum(ids, len(table) - 1)
            score += self.coefficients[field] * table[ids]
        return score


class SupervisedLatentSVD:
    def __init__(self, train, sample_weight, rank=36):
        users = np.asarray(train.user_id, dtype=np.int64)
        videos = np.asarray(train.video_id, dtype=np.int64)
        y = np.asarray(train.y, dtype=np.float64)
        w = np.asarray(sample_weight, dtype=np.float64)

        self.n_users = FEATURE_CARDINALITIES["user_id"]
        self.n_videos = FEATURE_CARDINALITIES["video_id"]

        global_rate = float(np.dot(w, y) / np.sum(w))

        user_count = np.bincount(
            users, weights=w, minlength=self.n_users
        ).astype(np.float64)
        user_pos = np.bincount(
            users, weights=w * y, minlength=self.n_users
        ).astype(np.float64)
        user_rate = (
            user_pos + 25.0 * global_rate
        ) / (user_count + 25.0)

        residual = y - user_rate[users]
        values = w * residual

        matrix = sparse.coo_matrix(
            (values, (users, videos)),
            shape=(self.n_users, self.n_videos),
            dtype=np.float32,
        ).tocsr()
        matrix.sum_duplicates()

        self.model = TruncatedSVD(
            n_components=rank,
            algorithm="randomized",
            n_iter=5,
            random_state=1927,
        )
        self.user_factors = self.model.fit_transform(matrix).astype(np.float32)
        self.video_factors = self.model.components_.T.astype(np.float32)

        user_norm = np.sqrt(
            np.sum(self.user_factors * self.user_factors, axis=1)
        )
        video_norm = np.sqrt(
            np.sum(self.video_factors * self.video_factors, axis=1)
        )
        self.user_scale = np.maximum(user_norm, 1e-4).astype(np.float32)
        self.video_scale = np.maximum(video_norm, 1e-4).astype(np.float32)

    def predict(self, split):
        users = np.asarray(split.user_id, dtype=np.int64)
        videos = np.asarray(split.video_id, dtype=np.int64)

        valid_u = users < self.n_users
        valid_v = videos < self.n_videos
        valid = valid_u & valid_v

        score = np.zeros(len(users), dtype=np.float64)
        if np.any(valid):
            u = users[valid]
            v = videos[valid]
            dot = np.sum(
                self.user_factors[u] * self.video_factors[v], axis=1
            )
            denom = np.sqrt(self.user_scale[u] * self.video_scale[v])
            score[valid] = dot / np.maximum(denom, 1e-4)
        return score


def evaluate_candidates(valid, incumbent_rank, own_ranks):
    candidate_scores = {"incumbent": incumbent_rank}
    candidate_own = {"incumbent": np.zeros_like(incumbent_rank)}
    recipes = {"incumbent": ("incumbent", (), 0.0)}

    own_sets = {}
    for name, values in own_ranks.items():
        own_sets[name] = values

    names = list(own_ranks)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name = names[i] + "+" + names[j]
            own_sets[name] = 0.5 * (
                own_ranks[names[i]] + own_ranks[names[j]]
            )

    if len(names) >= 3:
        own_sets["all_three"] = np.mean(
            np.column_stack([own_ranks[n] for n in names]), axis=1
        )

    for name, own_score in own_sets.items():
        standalone = "standalone_" + name
        candidate_scores[standalone] = own_score
        candidate_own[standalone] = own_score
        recipes[standalone] = ("standalone", tuple(name.split("+")), 1.0)

        for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            candidate_name = f"blend_{name}_{alpha:.2f}"
            blended = (
                (1.0 - alpha) * incumbent_rank + alpha * own_score
            )
            candidate_scores[candidate_name] = blended
            candidate_own[candidate_name] = own_score
            recipes[candidate_name] = (
                "blend",
                tuple(name.split("+")),
                alpha,
            )

    metrics = {}
    for name, score in candidate_scores.items():
        metrics[name] = float(
            evaluate(valid.user_id, valid.y, score)["primary"]
        )

    winner = max(metrics, key=metrics.get)
    return (
        winner,
        candidate_scores[winner],
        candidate_own[winner],
        recipes[winner],
        metrics,
    )


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
max_train_date = int(np.max(train.date))
age_days = max_train_date - np.asarray(train.date, dtype=np.int64)
recent_weight = np.power(
    0.5, age_days.astype(np.float64) / 4.0
).astype(np.float32)

X_train, history_names = make_content_matrix(train, "train")
X_valid, _ = make_content_matrix(valid, "valid")

categorical_indices = list(range(len(CONTENT_FIELDS)))

tree_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.05,
    "num_leaves": 47,
    "min_data_in_leaf": 350,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 4.0,
    "max_bin": 127,
    "num_threads": -1,
    "seed": 3181,
    "verbose": -1,
}

tree_dataset = lgb.Dataset(
    X_train,
    label=y_train,
    weight=recent_weight,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
tree_model = lgb.train(
    tree_params,
    tree_dataset,
    num_boost_round=260,
)
tree_valid_raw = tree_model.predict(X_valid)

eb_model = EmpiricalBayesModel(train, recent_weight)
eb_valid_raw = eb_model.predict(valid)

svd_model = SupervisedLatentSVD(
    train, recent_weight, rank=36
)
svd_valid_raw = svd_model.predict(valid)

own_valid_ranks = {
    "content_tree": within_user_rank(valid.user_id, tree_valid_raw),
    "latent_svd": within_user_rank(valid.user_id, svd_valid_raw),
    "empirical_bayes": within_user_rank(valid.user_id, eb_valid_raw),
}

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)

if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores missing")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test scores missing")

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

(
    winner,
    valid_scores,
    own_valid_winner,
    winner_recipe,
    candidate_primary,
) = evaluate_candidates(valid, inc_valid_rank, own_valid_ranks)

metrics = evaluate(valid.user_id, valid.y, valid_scores)

standalone_metrics = {
    name: float(
        evaluate(valid.user_id, valid.y, score)["primary"]
    )
    for name, score in own_valid_ranks.items()
}

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "recipe": winner_recipe,
            "standalone": standalone_metrics,
            "history_feature_count": len(history_names),
            "svd_rank": 36,
            "recency_half_life_days": 4.0,
        },
        separators=(",", ":"),
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        candidate_primary,
        sort_keys=True,
        separators=(",", ":"),
    )
)

test = load("test")
X_test, _ = make_content_matrix(test, "test")

tree_test_raw = tree_model.predict(X_test)
eb_test_raw = eb_model.predict(test)
svd_test_raw = svd_model.predict(test)

own_test_ranks = {
    "content_tree": within_user_rank(test.user_id, tree_test_raw),
    "latent_svd": within_user_rank(test.user_id, svd_test_raw),
    "empirical_bayes": within_user_rank(test.user_id, eb_test_raw),
}

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")
inc_test_rank = within_user_rank(test.user_id, inc_test)

recipe_type, recipe_members, alpha = winner_recipe

if winner == "incumbent":
    own_test_winner = np.zeros_like(inc_test_rank)
    test_scores = inc_test_rank
else:
    if recipe_members == ("all_three",):
        own_test_winner = np.mean(
            np.column_stack(list(own_test_ranks.values())), axis=1
        )
    else:
        member_scores = [
            own_test_ranks[name] for name in recipe_members
        ]
        own_test_winner = np.mean(
            np.column_stack(member_scores), axis=1
        )

    if recipe_type == "standalone":
        test_scores = own_test_winner
    else:
        test_scores = (
            (1.0 - alpha) * inc_test_rank
            + alpha * own_test_winner
        )

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(own_valid_winner, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
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
        separators=(",", ":"),
    )
)