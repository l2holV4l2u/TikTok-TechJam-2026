import os
import time
import json
import gc
import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
    "onehot_feat3",
    "hour",
    "user_active_degree",
    "register_days_bucket",
]

PAIR_FIELDS = [
    ("user_id", "video_id"),
    ("user_id", "author_id"),
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("user_id", "tab"),
    ("user_id", "hour"),
    ("video_id", "tag"),
    ("author_id", "tag"),
]

HASH_DIM = 1 << 18


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, user_ids))
    su = user_ids[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    ranked = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.015, 0.985)
    return np.log(p) - np.log1p(-p)


OFFSETS = {}
offset = 0
for field in FIELDS:
    OFFSETS[field] = offset
    offset += int(FEATURE_CARDINALITIES[field])
BASE_DIM = offset
TOTAL_DIM = BASE_DIM + HASH_DIM


def pair_hash(a, b, pair_index):
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    x = (
        a * np.uint64(11995408973635179863)
        + b * np.uint64(10150724397891781847)
        + np.uint64((pair_index + 1) * 2654435761)
    )
    x ^= x >> np.uint64(29)
    return np.asarray(x & np.uint64(HASH_DIM - 1), dtype=np.int64)


def make_sparse(split):
    n = len(split.user_id)
    nnz_per_row = len(FIELDS) + len(PAIR_FIELDS)
    rows = np.repeat(np.arange(n, dtype=np.int32), nnz_per_row)
    cols = np.empty(n * nnz_per_row, dtype=np.int32)
    data = np.ones(n * nnz_per_row, dtype=np.float32)

    position = 0
    for field in FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        cols[position::nnz_per_row] = (
            OFFSETS[field] + ids
        ).astype(np.int32)
        position += 1

    for j, (left, right) in enumerate(PAIR_FIELDS):
        h = pair_hash(split.X[left], split.X[right], j)
        cols[position::nnz_per_row] = (BASE_DIM + h).astype(np.int32)
        position += 1

    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n, TOTAL_DIM),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix


class AdditiveCategoricalLikelihood:
    def __init__(self, train, weights, alpha=80.0):
        y = np.asarray(train.y, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        self.global_rate = float(np.dot(weights, y) / weights.sum())
        self.global_logit = float(safe_logit(self.global_rate))
        self.tables = {}

        for field in FIELDS:
            ids = np.asarray(train.X[field], dtype=np.int64)
            card = int(FEATURE_CARDINALITIES[field])
            counts = np.bincount(
                ids, weights=weights, minlength=card
            ).astype(np.float64)
            positives = np.bincount(
                ids, weights=weights * y, minlength=card
            ).astype(np.float64)
            rates = (
                positives + alpha * self.global_rate
            ) / (counts + alpha)
            reliability = counts / (counts + 2.0 * alpha)
            self.tables[field] = (
                safe_logit(rates) - self.global_logit
            ) * reliability

    def predict(self, split):
        score = np.zeros(len(split.user_id), dtype=np.float64)
        for field in FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            table = self.tables[field]
            ids = np.minimum(ids, len(table) - 1)
            score += table[ids]
        return score


class RandomConjunctionEnsemble:
    def __init__(
        self,
        train,
        weights,
        n_rules=60,
        hash_size=16384,
        alpha=35.0,
        seed=2026,
    ):
        self.hash_size = int(hash_size)
        self.mask = np.uint64(hash_size - 1)
        self.fields = list(FIELDS)
        self.rules = []
        self.tables = []

        y = np.asarray(train.y, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        global_rate = float(np.dot(weights, y) / weights.sum())
        global_logit = float(safe_logit(global_rate))
        rng = np.random.default_rng(seed)

        for rule_index in range(n_rules):
            order = 2 if rule_index % 3 else 3
            chosen = tuple(
                rng.choice(len(self.fields), size=order, replace=False).tolist()
            )
            salts = rng.integers(
                1, np.iinfo(np.uint32).max, size=order, dtype=np.uint64
            )

            code = np.full(
                len(train.user_id),
                np.uint64(1469598103934665603),
                dtype=np.uint64,
            )
            for index, salt in zip(chosen, salts):
                value = np.asarray(
                    train.X[self.fields[index]], dtype=np.uint64
                )
                code ^= value + salt
                code *= np.uint64(1099511628211)
                code ^= code >> np.uint64(27)

            bins = np.asarray(code & self.mask, dtype=np.int64)
            counts = np.bincount(
                bins, weights=weights, minlength=hash_size
            ).astype(np.float64)
            positives = np.bincount(
                bins, weights=weights * y, minlength=hash_size
            ).astype(np.float64)
            rates = (
                positives + alpha * global_rate
            ) / (counts + alpha)
            reliability = counts / (counts + 2.0 * alpha)
            table = (safe_logit(rates) - global_logit) * reliability

            self.rules.append((chosen, salts))
            self.tables.append(table.astype(np.float32))

    def predict(self, split):
        result = np.zeros(len(split.user_id), dtype=np.float64)
        for (chosen, salts), table in zip(self.rules, self.tables):
            code = np.full(
                len(split.user_id),
                np.uint64(1469598103934665603),
                dtype=np.uint64,
            )
            for index, salt in zip(chosen, salts):
                value = np.asarray(
                    split.X[self.fields[index]], dtype=np.uint64
                )
                code ^= value + salt
                code *= np.uint64(1099511628211)
                code ^= code >> np.uint64(27)
            bins = np.asarray(code & self.mask, dtype=np.int64)
            result += table[bins]
        result /= np.sqrt(float(len(self.tables)))
        return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)

max_date = int(np.max(train.date))
age = max_date - np.asarray(train.date, dtype=np.int64)

half_lives = (2.0, 4.0, 8.0)
weight_sets = {
    h: np.power(0.5, age.astype(np.float64) / h).astype(np.float32)
    for h in half_lives
}
for h in half_lives:
    weight_sets[h] /= np.mean(weight_sets[h])

X_train = make_sparse(train)
X_valid = make_sparse(valid)

family_valid_raw = {}
linear_models = {}

for h in half_lives:
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=2.5e-7,
        l1_ratio=0.02,
        fit_intercept=True,
        max_iter=10,
        tol=1e-4,
        shuffle=True,
        average=True,
        random_state=3100 + int(h),
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=weight_sets[h])
    name = f"crossed_sparse_logistic_h{int(h)}"
    family_valid_raw[name] = model.decision_function(X_valid)
    linear_models[name] = model

del X_train
gc.collect()

likelihood_models = {}
for h in (2.0, 4.0, 8.0):
    model = AdditiveCategoricalLikelihood(
        train, weight_sets[h], alpha=80.0
    )
    name = f"categorical_likelihood_h{int(h)}"
    family_valid_raw[name] = model.predict(valid)
    likelihood_models[name] = model

conjunction_models = {}
for h, seed in ((2.0, 811), (4.0, 977), (8.0, 1231)):
    model = RandomConjunctionEnsemble(
        train,
        weight_sets[h],
        n_rules=60,
        hash_size=16384,
        alpha=35.0,
        seed=seed,
    )
    name = f"random_conjunction_h{int(h)}"
    family_valid_raw[name] = model.predict(valid)
    conjunction_models[name] = model

del X_valid
gc.collect()

family_valid_rank = {
    name: within_user_rank(valid.user_id, score)
    for name, score in family_valid_raw.items()
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation predictions missing")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test predictions missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {"incumbent": inc_valid_rank}
candidate_primary = {
    "incumbent": float(
        evaluate(valid.user_id, valid.y, inc_valid_rank)["primary"]
    )
}
candidate_recipe = {"incumbent": ("incumbent", None, 0.0)}

blend_weights = (0.10, 0.20, 0.30, 0.40, 0.50)

for family, own_rank in family_valid_rank.items():
    standalone = family + "_standalone"
    candidate_scores[standalone] = own_rank
    candidate_primary[standalone] = float(
        evaluate(valid.user_id, valid.y, own_rank)["primary"]
    )
    candidate_recipe[standalone] = ("standalone", family, 1.0)

    for alpha in blend_weights:
        name = f"{family}_blend_{alpha:.2f}"
        score = (1.0 - alpha) * inc_valid_rank + alpha * own_rank
        candidate_scores[name] = score
        candidate_primary[name] = float(
            evaluate(valid.user_id, valid.y, score)["primary"]
        )
        candidate_recipe[name] = ("blend", family, alpha)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)
recipe_type, winner_family, winner_alpha = candidate_recipe[winner]

standalone_names = [
    name for name in candidate_primary if name.endswith("_standalone")
]
best_own_candidate = max(
    standalone_names, key=lambda name: candidate_primary[name]
)
best_own_family = candidate_recipe[best_own_candidate][1]

if winner_family is None:
    raw_family_for_save = best_own_family
else:
    raw_family_for_save = winner_family

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "best_own_family": best_own_family,
            "best_own_primary": float(
                candidate_primary[best_own_candidate]
            ),
            "incumbent_primary": float(candidate_primary["incumbent"]),
            "selected_half_life_family": raw_family_for_save,
        },
        separators=(",", ":"),
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_primary.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

test = load("test")


def predict_family(family, split):
    if family in linear_models:
        X = make_sparse(split)
        prediction = linear_models[family].decision_function(X)
        del X
        gc.collect()
        return prediction
    if family in likelihood_models:
        return likelihood_models[family].predict(split)
    if family in conjunction_models:
        return conjunction_models[family].predict(split)
    raise KeyError(f"Unknown family: {family}")


if winner_family is None:
    test_own_raw = predict_family(raw_family_for_save, test)
else:
    test_own_raw = predict_family(winner_family, test)

test_own_rank = within_user_rank(test.user_id, test_own_raw)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if recipe_type == "incumbent":
    test_scores = inc_test_rank
elif recipe_type == "standalone":
    test_scores = test_own_rank
else:
    test_scores = (
        (1.0 - winner_alpha) * inc_test_rank
        + winner_alpha * test_own_rank
    )

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if recipe_type in ("blend", "incumbent"):
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(
                family_valid_rank[raw_family_for_save],
                dtype=np.float64,
            ),
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