import os
import gc
import json
import time
import warnings
import numpy as np
from scipy.special import ndtri, logsumexp

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

np.random.seed(27419)

HALF_LIFE_DAYS = 4.0
N_HIST_BINS = 24
GMM_COMPONENTS = 6
GMM_SAMPLE_PER_CLASS = 220000
GMM_EM_ITERATIONS = 4
PRED_BATCH = 131072

TE_FIELDS = [
    "user_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "music_type",
]

RAW_NUMERIC = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

HISTORY_SUFFIXES = (
    "train_count_log1p",
    "long_view_rate",
    "is_click_rate",
    "play_time_ms_logmean",
    "comment_stay_time_logmean",
)

PRIOR_STRENGTHS = {
    "user_id": 150.0,
    "tag": 1000.0,
    "tab": 1000.0,
    "duration_bucket": 1000.0,
    "upload_type": 700.0,
    "onehot_feat3": 200.0,
    "onehot_feat8": 200.0,
    "user_active_degree": 700.0,
    "music_type": 1000.0,
}


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = (
        np.arange(n, dtype=np.float64)
        - start_positions.astype(np.float64)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]

    sizes = np.diff(np.r_[-1, np.flatnonzero(ends)]).astype(np.float64)
    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = (local_rank / denom).astype(np.float32)
    return result


def copula_score(rank):
    p = np.clip(np.asarray(rank, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return ndtri(p).astype(np.float32)


def load_selected_history(split_name):
    columns = []
    names = []

    for key in ("video_id", "author_id"):
        hist = historical_features(split_name, key=key)
        for name in sorted(hist):
            if any(name.endswith(suffix) for suffix in HISTORY_SUFFIXES):
                x = np.asarray(hist[name], dtype=np.float32)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                columns.append(x)
                names.append(name)

    if not columns:
        raise RuntimeError("No selected historical features found")

    return np.column_stack(columns).astype(np.float32), names


def load_raw_numeric(split):
    columns = []

    for name in RAW_NUMERIC:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    hour = np.mod(hour, 24.0)
    angle = 2.0 * np.pi * hour / 24.0
    columns.append(np.sin(angle).astype(np.float32))
    columns.append(np.cos(angle).astype(np.float32))

    return np.column_stack(columns).astype(np.float32)


def fit_target_tables(train, labels):
    global_prior = float(np.mean(labels))
    tables = {}

    for field in TE_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        ids = np.asarray(train.X[field], dtype=np.int64)
        ids = np.where((ids >= 0) & (ids < card), ids, 0)

        counts = np.bincount(ids, minlength=card).astype(np.float32)
        sums = np.bincount(
            ids, weights=labels, minlength=card
        ).astype(np.float32)

        tables[field] = (
            counts,
            sums,
            global_prior,
            float(PRIOR_STRENGTHS[field]),
        )

    return tables


def target_encoding_features(split, tables, labels=None):
    columns = []

    for field in TE_FIELDS:
        counts, sums, prior, strength = tables[field]
        card = len(counts)

        ids = np.asarray(split.X[field], dtype=np.int64)
        ids = np.where((ids >= 0) & (ids < card), ids, 0)

        c = counts[ids]
        s = sums[ids]

        if labels is not None:
            c = np.maximum(c - 1.0, 0.0)
            s = s - labels

        rate = (s + strength * prior) / (c + strength)
        columns.append(rate.astype(np.float32))
        columns.append(np.log1p(c).astype(np.float32))

    return np.column_stack(columns).astype(np.float32)


def make_features(split, split_name, tables, labels=None,
                  expected_history_names=None):
    history, history_names = load_selected_history(split_name)

    if (
        expected_history_names is not None
        and history_names != expected_history_names
    ):
        raise RuntimeError("History feature order mismatch")

    raw = load_raw_numeric(split)
    target = target_encoding_features(split, tables, labels=labels)

    matrix = np.column_stack([history, raw, target]).astype(np.float32)
    matrix = np.nan_to_num(
        matrix, nan=0.0, posinf=0.0, neginf=0.0
    )

    del history, raw, target
    gc.collect()
    return matrix, history_names


def fit_scaler(x):
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)
    return mean, std


def scale_inplace(x, mean, std):
    x -= mean
    x /= std
    np.clip(x, -8.0, 8.0, out=x)
    return x.astype(np.float32, copy=False)


class HistogramLikelihoodRatio:
    """
    Additive non-parametric density-ratio model. Each continuous input is
    quantile discretized, then class-conditional weighted densities are
    accumulated as smoothed log likelihood ratios.
    """
    def __init__(self, n_bins=24, smoothing=30.0):
        self.n_bins = int(n_bins)
        self.smoothing = float(smoothing)
        self.edges = None
        self.log_ratio = None
        self.log_prior_odds = 0.0

    def fit(self, x, y, weights):
        n, d = x.shape
        rng = np.random.default_rng(11491)
        sample_n = min(n, 700000)
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        sample = x[sample_idx]

        quantiles = np.linspace(
            0.0, 1.0, self.n_bins + 1
        )[1:-1]

        self.edges = []
        self.log_ratio = np.empty(
            (d, self.n_bins), dtype=np.float32
        )

        total_pos = float(np.sum(weights * y))
        total_neg = float(np.sum(weights * (1.0 - y)))
        self.log_prior_odds = float(
            np.log((total_pos + 1.0) / (total_neg + 1.0))
        )

        for j in range(d):
            edges = np.unique(
                np.quantile(sample[:, j], quantiles)
            ).astype(np.float32)
            self.edges.append(edges)

            bins = np.searchsorted(
                edges, x[:, j], side="right"
            )
            effective_bins = len(edges) + 1

            pos = np.bincount(
                bins,
                weights=weights * y,
                minlength=effective_bins,
            ).astype(np.float64)
            neg = np.bincount(
                bins,
                weights=weights * (1.0 - y),
                minlength=effective_bins,
            ).astype(np.float64)

            alpha = self.smoothing / effective_bins
            p_pos = (pos + alpha) / (
                np.sum(pos) + self.smoothing
            )
            p_neg = (neg + alpha) / (
                np.sum(neg) + self.smoothing
            )

            ratios = np.log(p_pos / p_neg)
            self.log_ratio[j, :] = 0.0
            self.log_ratio[j, :effective_bins] = ratios.astype(
                np.float32
            )

        return self

    def predict(self, x):
        n, d = x.shape
        scores = np.full(
            n, self.log_prior_odds, dtype=np.float32
        )

        for j in range(d):
            bins = np.searchsorted(
                self.edges[j], x[:, j], side="right"
            )
            scores += self.log_ratio[j, bins]

        return scores


class RegularizedQDA:
    """
    Full-covariance Gaussian discriminant model. Unlike an additive density
    model, it scores correlations and rotated feature directions separately
    for positives and negatives.
    """
    def __init__(self, shrinkage=0.18):
        self.shrinkage = float(shrinkage)
        self.means = []
        self.precisions = []
        self.logdets = []
        self.log_priors = []

    def fit(self, x, y, weights):
        d = x.shape[1]
        class_weight_totals = []

        for label in (0, 1):
            class_w = weights * (y == label)
            total = float(np.sum(class_w))
            class_weight_totals.append(total)

            mean = np.sum(
                x * class_w[:, None],
                axis=0,
                dtype=np.float64,
            ) / max(total, 1.0)

            covariance = np.zeros((d, d), dtype=np.float64)

            for start in range(0, len(x), PRED_BATCH):
                end = min(start + PRED_BATCH, len(x))
                w = class_w[start:end].astype(np.float64)
                centered = (
                    x[start:end].astype(np.float64)
                    - mean[None, :]
                )
                covariance += (
                    centered * w[:, None]
                ).T @ centered

            covariance /= max(total, 1.0)
            diagonal = np.diag(np.diag(covariance))
            covariance = (
                (1.0 - self.shrinkage) * covariance
                + self.shrinkage * diagonal
            )
            covariance += np.eye(d, dtype=np.float64) * 0.025

            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise RuntimeError("QDA covariance is not positive definite")

            precision = np.linalg.inv(covariance)

            self.means.append(mean.astype(np.float32))
            self.precisions.append(precision.astype(np.float32))
            self.logdets.append(float(logdet))

        total_weight = sum(class_weight_totals)
        self.log_priors = [
            float(np.log(max(v / total_weight, 1e-12)))
            for v in class_weight_totals
        ]
        return self

    def predict(self, x):
        result = np.empty(len(x), dtype=np.float32)

        for start in range(0, len(x), PRED_BATCH):
            end = min(start + PRED_BATCH, len(x))
            xb = x[start:end]
            class_scores = []

            for label in (0, 1):
                centered = xb - self.means[label][None, :]
                quadratic = np.einsum(
                    "bi,ij,bj->b",
                    centered,
                    self.precisions[label],
                    centered,
                    optimize=True,
                )
                score = (
                    self.log_priors[label]
                    - 0.5 * self.logdets[label]
                    - 0.5 * quadratic
                )
                class_scores.append(score)

            result[start:end] = (
                class_scores[1] - class_scores[0]
            ).astype(np.float32)

        return result


class DiagonalGaussianMixtureClassifier:
    """
    A class-conditional mixture model. Multiple prototypes per class capture
    distinct modes such as short-video, long-video, and different engagement
    regimes without imposing one global boundary.
    """
    def __init__(self, n_components=6, iterations=4, variance_floor=0.08):
        self.n_components = int(n_components)
        self.iterations = int(iterations)
        self.variance_floor = float(variance_floor)
        self.params = {}
        self.log_priors = None

    def _fit_class(self, x, weights, seed):
        rng = np.random.default_rng(seed)
        n, d = x.shape
        k = self.n_components

        init_idx = rng.choice(n, size=k, replace=False)
        means = x[init_idx].astype(np.float64, copy=True)
        global_var = np.var(x, axis=0, dtype=np.float64) + 0.1
        variances = np.tile(global_var[None, :], (k, 1))
        mixture = np.full(k, 1.0 / k, dtype=np.float64)

        for iteration in range(self.iterations):
            nk = np.zeros(k, dtype=np.float64)
            sx = np.zeros((k, d), dtype=np.float64)
            sx2 = np.zeros((k, d), dtype=np.float64)
            weighted_ll = 0.0
            weight_sum = 0.0

            for start in range(0, n, 32768):
                end = min(start + 32768, n)
                xb = x[start:end].astype(np.float64)
                wb = weights[start:end].astype(np.float64)

                diff = xb[:, None, :] - means[None, :, :]
                log_prob = (
                    np.log(mixture + 1e-12)[None, :]
                    - 0.5 * np.sum(
                        np.log(2.0 * np.pi * variances)[None, :, :]
                        + diff * diff / variances[None, :, :],
                        axis=2,
                    )
                )

                norm = logsumexp(log_prob, axis=1)
                responsibilities = np.exp(
                    log_prob - norm[:, None]
                )
                responsibilities *= wb[:, None]

                nk += np.sum(responsibilities, axis=0)
                sx += responsibilities.T @ xb
                sx2 += responsibilities.T @ (xb * xb)
                weighted_ll += float(np.sum(norm * wb))
                weight_sum += float(np.sum(wb))

            mixture = (nk + 1.0) / (np.sum(nk) + k)
            means = sx / np.maximum(nk[:, None], 1e-8)
            variances = (
                sx2 / np.maximum(nk[:, None], 1e-8)
                - means * means
            )
            variances = np.maximum(
                variances, self.variance_floor
            )

            print(
                "FINDINGS gmm_seed=%d em_iteration=%d "
                "weighted_loglik=%.6f"
                % (
                    seed,
                    iteration + 1,
                    weighted_ll / max(weight_sum, 1.0),
                ),
                flush=True,
            )

        return (
            means.astype(np.float32),
            variances.astype(np.float32),
            mixture.astype(np.float32),
        )

    def fit(self, x, y, weights, sample_per_class):
        rng = np.random.default_rng(39103)
        totals = []

        for label in (0, 1):
            indices = np.flatnonzero(y == label)
            take = min(sample_per_class, len(indices))
            selected = rng.choice(indices, size=take, replace=False)

            class_x = x[selected]
            class_w = weights[selected]
            self.params[label] = self._fit_class(
                class_x, class_w, seed=41011 + label
            )
            totals.append(float(np.sum(weights[y == label])))

            del indices, selected, class_x, class_w
            gc.collect()

        total = sum(totals)
        self.log_priors = np.log(
            np.maximum(np.asarray(totals) / total, 1e-12)
        ).astype(np.float32)
        return self

    def _class_log_density(self, x, label):
        means, variances, mixture = self.params[label]
        result = np.empty(len(x), dtype=np.float32)

        for start in range(0, len(x), 65536):
            end = min(start + 65536, len(x))
            xb = x[start:end].astype(np.float32)
            diff = xb[:, None, :] - means[None, :, :]

            log_prob = (
                np.log(mixture + 1e-12)[None, :]
                - 0.5 * np.sum(
                    np.log(2.0 * np.pi * variances)[None, :, :]
                    + diff * diff / variances[None, :, :],
                    axis=2,
                )
            )
            result[start:end] = logsumexp(
                log_prob, axis=1
            ).astype(np.float32)

        return result

    def predict(self, x):
        negative = self._class_log_density(x, 0)
        positive = self._class_log_density(x, 1)
        return (
            positive + self.log_priors[1]
            - negative - self.log_priors[0]
        ).astype(np.float32)


def register_candidate(candidate_results, name, user_ids, labels, scores):
    metrics = evaluate(user_ids, labels, scores)
    candidate_results[name] = float(metrics["primary"])
    return metrics


inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)

if not (
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent artifacts are required")

train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y)
valid_uid = np.asarray(valid.user_id, dtype=np.int64)

tables = fit_target_tables(train, train_y)

train_x, history_names = make_features(
    train, "train", tables, labels=train_y
)
valid_x, _ = make_features(
    valid,
    "valid",
    tables,
    labels=None,
    expected_history_names=history_names,
)

feature_mean, feature_std = fit_scaler(train_x)
train_x = scale_inplace(train_x, feature_mean, feature_std)
valid_x = scale_inplace(valid_x, feature_mean, feature_std)

train_dates = np.asarray(train.date, dtype=np.int32)
latest_date = int(np.max(train_dates))
age_days = (latest_date - train_dates).astype(np.float32)

train_weights = np.exp(
    -np.log(2.0) * age_days / HALF_LIFE_DAYS
).astype(np.float32)
train_weights /= np.mean(train_weights)

models = {
    "histogram_likelihood": HistogramLikelihoodRatio(
        n_bins=N_HIST_BINS,
        smoothing=30.0,
    ),
    "full_covariance_qda": RegularizedQDA(shrinkage=0.18),
    "gaussian_mixture": DiagonalGaussianMixtureClassifier(
        n_components=GMM_COMPONENTS,
        iterations=GMM_EM_ITERATIONS,
        variance_floor=0.08,
    ),
}

print(
    "FINDINGS fitting=histogram_likelihood rows=%d dim=%d"
    % (len(train_x), train_x.shape[1]),
    flush=True,
)
models["histogram_likelihood"].fit(
    train_x, train_y, train_weights
)

print(
    "FINDINGS fitting=full_covariance_qda rows=%d dim=%d"
    % (len(train_x), train_x.shape[1]),
    flush=True,
)
models["full_covariance_qda"].fit(
    train_x, train_y, train_weights
)

print(
    "FINDINGS fitting=gaussian_mixture components=%d "
    "sample_per_class=%d"
    % (GMM_COMPONENTS, GMM_SAMPLE_PER_CLASS),
    flush=True,
)
models["gaussian_mixture"].fit(
    train_x,
    train_y,
    train_weights,
    sample_per_class=GMM_SAMPLE_PER_CLASS,
)

inc_valid_raw = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_valid_rank = within_user_rank(valid_uid, inc_valid_raw)
inc_valid_copula = copula_score(inc_valid_rank)

candidate_results = {}
inc_metrics = register_candidate(
    candidate_results,
    "trusted_incumbent",
    valid_uid,
    valid_y,
    inc_valid_rank,
)

valid_ranks = {}
standalone_scores = {}

for name, model in models.items():
    raw = model.predict(valid_x)
    rank = within_user_rank(valid_uid, raw)
    metrics = register_candidate(
        candidate_results,
        name + "_standalone",
        valid_uid,
        valid_y,
        rank,
    )

    valid_ranks[name] = rank
    standalone_scores[name] = float(metrics["primary"])

    correlation = float(np.corrcoef(rank, inc_valid_rank)[0, 1])
    disagreement = float(np.mean(np.abs(rank - inc_valid_rank)))

    print(
        "FINDINGS family=%s standalone_primary=%.6f "
        "gauc=%.6f ndcg5=%.6f rank_corr=%.6f "
        "mean_abs_disagreement=%.6f"
        % (
            name,
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
            correlation,
            disagreement,
        ),
        flush=True,
    )

best_scores = inc_valid_rank.copy()
best_primary = float(inc_metrics["primary"])
best_family = None
best_transform = "rank"
best_alpha = 0.0
best_gamma = 1.0
best_own_rank = None

alphas = [0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.32]
gammas = [1.0, 2.0, 4.0]

for family, rank in valid_ranks.items():
    for gamma in gammas:
        shaped_rank = np.power(
            np.clip(rank, 0.0, 1.0), gamma
        ).astype(np.float32)

        transform_options = [
            ("rank", inc_valid_rank, shaped_rank),
            (
                "copula",
                inc_valid_copula,
                copula_score(shaped_rank),
            ),
        ]

        for transform, incumbent_base, family_base in transform_options:
            for alpha in alphas:
                blended = (
                    (1.0 - alpha) * incumbent_base
                    + alpha * family_base
                ).astype(np.float32)

                name = "%s_%s_gamma%.1f_alpha%.2f" % (
                    family, transform, gamma, alpha
                )
                metrics = register_candidate(
                    candidate_results,
                    name,
                    valid_uid,
                    valid_y,
                    blended,
                )

                if float(metrics["primary"]) > best_primary:
                    best_primary = float(metrics["primary"])
                    best_scores = blended.copy()
                    best_family = family
                    best_transform = transform
                    best_alpha = float(alpha)
                    best_gamma = float(gamma)
                    best_own_rank = rank.copy()

# A cross-family generative consensus is another formation rule: only
# agreements shared across different density assumptions receive full weight.
family_mean_rank = np.mean(
    np.column_stack(
        [valid_ranks[name] for name in sorted(valid_ranks)]
    ),
    axis=1,
).astype(np.float32)

mean_metrics = register_candidate(
    candidate_results,
    "generative_consensus_standalone",
    valid_uid,
    valid_y,
    family_mean_rank,
)

for gamma in gammas:
    shaped = np.power(
        np.clip(family_mean_rank, 0.0, 1.0), gamma
    ).astype(np.float32)

    for transform, incumbent_base, family_base in [
        ("rank", inc_valid_rank, shaped),
        ("copula", inc_valid_copula, copula_score(shaped)),
    ]:
        for alpha in alphas:
            blended = (
                (1.0 - alpha) * incumbent_base
                + alpha * family_base
            ).astype(np.float32)

            name = (
                "generative_consensus_%s_gamma%.1f_alpha%.2f"
                % (transform, gamma, alpha)
            )
            metrics = register_candidate(
                candidate_results,
                name,
                valid_uid,
                valid_y,
                blended,
            )

            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_scores = blended.copy()
                best_family = "generative_consensus"
                best_transform = transform
                best_alpha = float(alpha)
                best_gamma = float(gamma)
                best_own_rank = family_mean_rank.copy()

final_metrics = evaluate(valid_uid, valid_y, best_scores)
best_standalone_family = max(
    standalone_scores, key=standalone_scores.get
)

print(
    "FINDINGS feature_dim=%d history_dim=%d half_life_days=%.1f"
    % (
        train_x.shape[1],
        len(history_names),
        HALF_LIFE_DAYS,
    ),
    flush=True,
)
print(
    "FINDINGS winner=%s transform=%s alpha=%.2f gamma=%.1f "
    "incumbent_primary=%.6f final_primary=%.6f"
    % (
        best_family if best_family is not None else "trusted_incumbent",
        best_transform,
        best_alpha,
        best_gamma,
        float(inc_metrics["primary"]),
        float(final_metrics["primary"]),
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_results, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(OUT, "scores_valid_raw.npy"),
        np.asarray(
            best_own_rank
            if best_own_rank is not None
            else valid_ranks[best_standalone_family],
            dtype=np.float64,
        ),
    )

del valid_x
del inc_valid_raw
gc.collect()

test = load("test")
test_uid = np.asarray(test.user_id, dtype=np.int64)

inc_test_raw = np.asarray(
    np.load(inc_test_path, mmap_mode="r"),
    dtype=np.float32,
)
inc_test_rank = within_user_rank(test_uid, inc_test_raw)

if best_family is None:
    test_scores = inc_test_rank
else:
    test_x, _ = make_features(
        test,
        "test",
        tables,
        labels=None,
        expected_history_names=history_names,
    )
    test_x = scale_inplace(test_x, feature_mean, feature_std)

    if best_family == "generative_consensus":
        test_family_ranks = []
        for name in sorted(models):
            raw = models[name].predict(test_x)
            test_family_ranks.append(
                within_user_rank(test_uid, raw)
            )
        selected_test_rank = np.mean(
            np.column_stack(test_family_ranks),
            axis=1,
        ).astype(np.float32)
    else:
        selected_test_raw = models[best_family].predict(test_x)
        selected_test_rank = within_user_rank(
            test_uid, selected_test_raw
        )

    shaped_test_rank = np.power(
        np.clip(selected_test_rank, 0.0, 1.0),
        best_gamma,
    ).astype(np.float32)

    if best_transform == "copula":
        incumbent_base = copula_score(inc_test_rank)
        selected_base = copula_score(shaped_test_rank)
    else:
        incumbent_base = inc_test_rank
        selected_base = shaped_test_rank

    test_scores = (
        (1.0 - best_alpha) * incumbent_base
        + best_alpha * selected_base
    ).astype(np.float32)

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)

print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)