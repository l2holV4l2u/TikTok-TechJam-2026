import os
import gc
import json
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
OUT_DIR = os.environ.get("ITER_OUT")
ARTIFACTS = os.environ.get("RUN_ARTIFACTS", "")

if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)

N_USERS = int(FEATURE_CARDINALITIES["user_id"])
MAX_RANK = 64
RANKS = [8, 16, 32, 64]
WEIGHTS = [
    -0.25, -0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30,
    0.40, 0.55, 0.70, 0.90, 1.20, 1.50
]

CONFIGS = {
    "latent_video": ["video_id"],
    "latent_content": [
        "video_id",
        "author_id",
        "tag",
        "duration_bucket",
        "upload_type",
        "music_type",
    ],
}


def global_zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if not np.isfinite(std) or std < 1e-10:
        std = 1.0
    return (x - mean) / std


def within_user_zscore(user_ids, x):
    """
    Affine-normalize scores separately inside each logged impression set.
    This preserves each component's ranking while preventing users with
    unusually wide score ranges from dominating a fixed blend coefficient.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(x, dtype=np.float64)

    n = max(N_USERS, int(users.max()) + 1)
    count = np.bincount(users, minlength=n).astype(np.float64)
    total = np.bincount(users, weights=values, minlength=n)
    total2 = np.bincount(users, weights=values * values, minlength=n)

    safe_count = np.maximum(count, 1.0)
    mean = total / safe_count
    var = np.maximum(total2 / safe_count - mean * mean, 0.0)
    std = np.sqrt(var)

    # Singleton and constant-score groups cannot be standardized. Centering
    # them at zero is harmless because no within-user ordering is available.
    denom = np.where(std > 1e-8, std, 1.0)
    result = (values - mean[users]) / denom[users]
    result[std[users] <= 1e-8] = 0.0
    return result


def user_centered_residuals(users, labels):
    users = np.asarray(users, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)

    count = np.bincount(users, minlength=N_USERS).astype(np.float64)
    positive = np.bincount(
        users, weights=labels, minlength=N_USERS
    ).astype(np.float64)

    global_rate = float(np.mean(labels))
    # Mild shrinkage stabilizes users with very little history.
    user_rate = (positive + 5.0 * global_rate) / (count + 5.0)
    return labels - user_rate[users]


def averaged_user_entity_matrix(users, entities, residuals, cardinality):
    """
    Construct one sparse cell per observed user/entity pair. Repeated
    impressions are averaged instead of summed, so frequently exposed pairs
    do not automatically receive larger collaborative-filtering targets.
    """
    users = np.asarray(users, dtype=np.int64)
    entities = np.asarray(entities, dtype=np.int64)
    residuals = np.asarray(residuals, dtype=np.float64)

    shape = (N_USERS, int(cardinality))
    sums = sp.coo_matrix(
        (residuals.astype(np.float32), (users, entities)),
        shape=shape,
        dtype=np.float32,
    ).tocsr()
    counts = sp.coo_matrix(
        (
            np.ones(len(users), dtype=np.float32),
            (users, entities),
        ),
        shape=shape,
        dtype=np.float32,
    ).tocsr()

    # COO-to-CSR aggregation produces matching sparsity patterns.
    sums.data /= np.maximum(counts.data, 1.0)
    sums.eliminate_zeros()
    return sums


class LatentHistoryModel:
    def __init__(self, fields, max_rank=64):
        self.fields = list(fields)
        self.max_rank = int(max_rank)
        self.offsets = None
        self.u = None
        self.s = None
        self.vt = None

    def fit(self, split, labels):
        users = np.asarray(split.user_id, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)
        residuals = user_centered_residuals(users, labels)

        matrices = []
        offsets = {}
        offset = 0

        for field in self.fields:
            cardinality = int(FEATURE_CARDINALITIES[field])
            matrix = averaged_user_entity_matrix(
                users,
                np.asarray(split.X[field], dtype=np.int64),
                residuals,
                cardinality,
            )

            # Equalize broad and fine-grained fields so a field's number of
            # populated coordinates does not alone determine the SVD.
            field_scale = np.sqrt(max(1, len(self.fields)))
            if field_scale != 1.0:
                matrix.data /= field_scale

            matrices.append(matrix)
            offsets[field] = offset
            offset += cardinality

        joint = sp.hstack(matrices, format="csr", dtype=np.float32)
        del matrices, residuals
        gc.collect()

        feasible_rank = min(
            self.max_rank,
            joint.shape[0] - 1,
            joint.shape[1] - 1,
        )
        if feasible_rank < 2:
            raise RuntimeError("Collaborative matrix is too small for SVD")

        u, s, vt = svds(
            joint,
            k=feasible_rank,
            which="LM",
            tol=1e-3,
            maxiter=500,
            return_singular_vectors=True,
            random_state=2026,
        )

        # scipy returns singular values in ascending order.
        order = np.argsort(s)[::-1]
        self.u = np.asarray(u[:, order], dtype=np.float32)
        self.s = np.asarray(s[order], dtype=np.float32)
        self.vt = np.asarray(vt[order, :], dtype=np.float32)
        self.offsets = offsets

        del joint, u, s, vt
        gc.collect()
        return self

    def predict(self, split, rank):
        rank = min(int(rank), len(self.s))
        users = np.asarray(split.user_id, dtype=np.int64)

        # U*S is the user-side representation of the reconstructed matrix.
        user_factors = (
            self.u[users, :rank] * self.s[None, :rank]
        ).astype(np.float32, copy=False)

        result = np.zeros(len(users), dtype=np.float64)
        for field in self.fields:
            entity = np.asarray(split.X[field], dtype=np.int64)
            columns = self.offsets[field] + entity
            entity_factors = self.vt[:rank, columns].T
            result += np.einsum(
                "ij,ij->i",
                user_factors,
                entity_factors,
                optimize=True,
            )

        result /= float(len(self.fields))
        return result


def combined_split(train, valid, fields):
    class Combined:
        pass

    result = Combined()
    result.user_id = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    result.X = {}
    for field in fields:
        result.X[field] = np.concatenate([
            np.asarray(train.X[field], dtype=np.int64),
            np.asarray(valid.X[field], dtype=np.int64),
        ])
    return result


inc_valid_path = os.path.join(
    ARTIFACTS, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    ARTIFACTS, "incumbent_test_scores.npy"
)

if not (
    ARTIFACTS
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError(
        "Trusted incumbent predictions are missing from RUN_ARTIFACTS"
    )

train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float64)
y_valid = np.asarray(valid.y, dtype=np.int8)

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
if len(inc_valid) != len(y_valid):
    raise RuntimeError("Invalid incumbent validation prediction length")

inc_valid_global = global_zscore(inc_valid)
inc_valid_user = within_user_zscore(valid.user_id, inc_valid)

base_metrics = evaluate(
    valid.user_id, y_valid, inc_valid_global
)
best_metrics = base_metrics
best_scores = inc_valid_global.copy()
best_config = None
best_rank = None
best_weight = 0.0
best_normalization = None

candidate_summary = {
    "trusted_incumbent": round(float(base_metrics["primary"]), 6)
}
config_diagnostics = {}

for config_name, fields in CONFIGS.items():
    model = LatentHistoryModel(fields, max_rank=MAX_RANK)
    model.fit(train, y_train)

    config_best = -np.inf
    config_best_raw = -np.inf

    for rank in RANKS:
        raw_cf = model.predict(valid, rank)
        cf_global = global_zscore(raw_cf)
        cf_user = within_user_zscore(valid.user_id, raw_cf)

        raw_metrics = evaluate(valid.user_id, y_valid, raw_cf)
        config_best_raw = max(
            config_best_raw, float(raw_metrics["primary"])
        )

        for normalization, incumbent_component, cf_component in [
            (
                "global",
                inc_valid_global,
                cf_global,
            ),
            (
                "within_user",
                inc_valid_user,
                cf_user,
            ),
        ]:
            for weight in WEIGHTS:
                if weight == 0.0:
                    scores = incumbent_component
                else:
                    scores = (
                        incumbent_component
                        + float(weight) * cf_component
                    )

                metrics = evaluate(
                    valid.user_id, y_valid, scores
                )
                primary = float(metrics["primary"])
                config_best = max(config_best, primary)

                if primary > float(best_metrics["primary"]):
                    best_metrics = metrics
                    best_scores = np.asarray(
                        scores, dtype=np.float64
                    ).copy()
                    best_config = config_name
                    best_rank = int(rank)
                    best_weight = float(weight)
                    best_normalization = normalization

        del raw_cf, cf_global, cf_user
        gc.collect()

    candidate_summary[config_name] = round(config_best, 6)
    config_diagnostics[config_name] = {
        "standalone_best": round(config_best_raw, 6),
        "fused_best": round(config_best, 6),
    }

    del model
    gc.collect()

print(
    "FINDINGS "
    + json.dumps(
        {
            "incumbent_primary": round(
                float(base_metrics["primary"]), 6
            ),
            "latent_diagnostics": config_diagnostics,
            "selected_config": best_config,
            "selected_rank": best_rank,
            "selected_weight": best_weight,
            "selected_normalization": best_normalization,
            "selected_primary": round(
                float(best_metrics["primary"]), 6
            ),
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(candidate_summary, sort_keys=True)
)

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

# Apply exactly the validation-selected recipe to test. If latent history was
# selected, refit it on train+validation labels, which is allowed for the
# final test model. Test labels are never accessed.
test = load("test")
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("Invalid incumbent test prediction length")

if (
    best_config is None
    or best_weight == 0.0
    or best_rank is None
):
    test_scores = global_zscore(inc_test)
else:
    selected_fields = CONFIGS[best_config]
    fit_split = combined_split(
        train, valid, selected_fields
    )
    fit_labels = np.concatenate([
        y_train,
        y_valid.astype(np.float64),
    ])

    test_model = LatentHistoryModel(
        selected_fields, max_rank=MAX_RANK
    )
    test_model.fit(fit_split, fit_labels)
    test_cf_raw = test_model.predict(test, best_rank)

    if best_normalization == "within_user":
        test_inc_component = within_user_zscore(
            test.user_id, inc_test
        )
        test_cf_component = within_user_zscore(
            test.user_id, test_cf_raw
        )
    else:
        test_inc_component = global_zscore(inc_test)
        test_cf_component = global_zscore(test_cf_raw)

    test_scores = (
        test_inc_component
        + best_weight * test_cf_component
    )

    del test_model, test_cf_raw, fit_split, fit_labels
    gc.collect()

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START_TIME)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    )
)