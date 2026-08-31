import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
THREADS = min(8, os.cpu_count() or 8)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)

ntr = len(ytr)
nva = len(uva)
nte = len(test.user_id)

# Random row folds let every OOF prediction use other interactions from known
# users and entities, while ensuring that the current label was never used by
# the base model producing its stacking feature.
NFOLDS = 5
rng = np.random.default_rng(SEED)
fold_id = np.empty(ntr, dtype=np.int8)
fold_id[rng.permutation(ntr)] = (
    np.arange(ntr, dtype=np.int64) % NFOLDS
).astype(np.int8)

# Recency weighting is fitted entirely from train dates.
dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
day_index = np.searchsorted(unique_dates, dates)
age = (len(unique_dates) - 1 - day_index).astype(np.float32)
sample_weight = np.exp2(-age / 4.0).astype(np.float32)
sample_weight /= sample_weight.mean()


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(scores), dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    positions = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranks_sorted = positions / denominators

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = ranks_sorted
    return result


def global_standardize(train_values, other_values):
    center = float(np.median(train_values))
    q25, q75 = np.quantile(train_values, [0.25, 0.75])
    scale = max(float(q75 - q25), 1.0e-4)
    return (
        np.clip((train_values - center) / scale, -10.0, 10.0),
        np.clip((other_values - center) / scale, -10.0, 10.0),
    )


# -------------------------------------------------------------------------
# Family 1: categorical boosted trees.
# -------------------------------------------------------------------------
TREE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "music_type",
    "video_type",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]


def categorical_matrix(sample, fields):
    return np.column_stack([
        np.asarray(sample.X[f], dtype=np.int32) for f in fields
    ]).astype(np.int32, copy=False)


tree_xtr = categorical_matrix(train, TREE_FIELDS)
tree_xva = categorical_matrix(valid, TREE_FIELDS)
tree_xte = categorical_matrix(test, TREE_FIELDS)

tree_oof = np.empty(ntr, dtype=np.float64)
tree_valid = np.zeros(nva, dtype=np.float64)
tree_test = np.zeros(nte, dtype=np.float64)

tree_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.055,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 900,
    "lambda_l2": 5.0,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 30.0,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
}

for fold in range(NFOLDS):
    fit_rows = np.flatnonzero(fold_id != fold)
    hold_rows = np.flatnonzero(fold_id == fold)

    dset = lgb.Dataset(
        tree_xtr[fit_rows],
        label=ytr[fit_rows],
        weight=sample_weight[fit_rows],
        categorical_feature=list(range(len(TREE_FIELDS))),
        free_raw_data=True,
    )
    model = lgb.train(tree_params, dset, num_boost_round=115)

    tree_oof[hold_rows] = model.predict(tree_xtr[hold_rows])
    tree_valid += model.predict(tree_xva) / NFOLDS
    tree_test += model.predict(tree_xte) / NFOLDS

    del model, dset, fit_rows, hold_rows
    gc.collect()

print("FINDINGS completed_crossfit_family=boosted_tree folds=5")


# -------------------------------------------------------------------------
# Family 2: empirical-Bayes class-conditional evidence.
#
# Each field and cross contributes a smoothed log-odds estimate. This forms a
# prediction additively from population evidence rather than tree partitions.
# -------------------------------------------------------------------------
NB_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "music_type",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
NB_CROSSES = [
    ("tab", "tag"),
    ("duration_bucket", "tag"),
    ("author_id", "tab"),
    ("upload_type", "duration_bucket"),
    ("onehot_feat3", "tag"),
]


def nb_arrays(sample):
    arrays = []
    cards = []

    for field in NB_FIELDS:
        arrays.append(np.asarray(sample.X[field], dtype=np.int64))
        cards.append(int(FEATURE_CARDINALITIES[field]))

    for left, right in NB_CROSSES:
        rc = int(FEATURE_CARDINALITIES[right])
        arrays.append(
            np.asarray(sample.X[left], dtype=np.int64) * rc
            + np.asarray(sample.X[right], dtype=np.int64)
        )
        cards.append(int(FEATURE_CARDINALITIES[left]) * rc)

    return arrays, cards


nb_tr, nb_cards = nb_arrays(train)
nb_va, _ = nb_arrays(valid)
nb_te, _ = nb_arrays(test)

nb_oof = np.empty(ntr, dtype=np.float64)
nb_valid = np.zeros(nva, dtype=np.float64)
nb_test = np.zeros(nte, dtype=np.float64)

for fold in range(NFOLDS):
    fit_mask = fold_id != fold
    hold_rows = np.flatnonzero(fold_id == fold)

    wf = sample_weight[fit_mask].astype(np.float64)
    yf = ytr[fit_mask].astype(np.float64)
    prior = float(np.sum(wf * yf) / np.sum(wf))
    prior_logit = np.log((prior + 1e-6) / (1.0 - prior + 1e-6))

    fold_hold_score = np.zeros(len(hold_rows), dtype=np.float64)
    fold_valid_score = np.zeros(nva, dtype=np.float64)
    fold_test_score = np.zeros(nte, dtype=np.float64)

    for j, card in enumerate(nb_cards):
        ids_fit = nb_tr[j][fit_mask]
        total = np.bincount(
            ids_fit, weights=wf, minlength=card
        ).astype(np.float64)
        positive = np.bincount(
            ids_fit, weights=wf * yf, minlength=card
        ).astype(np.float64)

        # Larger smoothing for high-cardinality identities, smaller smoothing
        # for stable low-cardinality side information.
        smoothing = 45.0 if card > 5000 else (30.0 if card > 500 else 18.0)
        rate = (positive + smoothing * prior) / (total + smoothing)
        evidence = (
            np.log((rate + 1e-5) / (1.0 - rate + 1e-5))
            - prior_logit
        )
        reliability = total / (total + smoothing)
        evidence *= reliability

        fold_hold_score += evidence[nb_tr[j][hold_rows]]
        fold_valid_score += evidence[nb_va[j]]
        fold_test_score += evidence[nb_te[j]]

    scale = 1.0 / np.sqrt(len(nb_cards))
    nb_oof[hold_rows] = fold_hold_score * scale
    nb_valid += fold_valid_score * scale / NFOLDS
    nb_test += fold_test_score * scale / NFOLDS

    del fit_mask, hold_rows, wf, yf
    gc.collect()

print("FINDINGS completed_crossfit_family=empirical_bayes folds=5")


# -------------------------------------------------------------------------
# Family 3: factorization machine.
#
# Its global latent pairwise interactions provide a complementary geometry to
# both the piecewise tree model and additive empirical-Bayes evidence.
# -------------------------------------------------------------------------
FM_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
]

offsets = []
running = 0
for field in FM_FIELDS:
    offsets.append(running)
    running += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)
total_cardinality = int(running)


def fm_matrix(sample):
    matrix = np.column_stack([
        np.asarray(sample.X[f], dtype=np.int64) for f in FM_FIELDS
    ])
    matrix += offsets[None, :]
    return matrix


fm_xtr = fm_matrix(train)
fm_xva = fm_matrix(valid)
fm_xte = fm_matrix(test)


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, dimension=12):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, dimension)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.018)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        emb = self.embedding(x)
        summed = emb.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - emb.square().sum(dim=(1, 2))
        )
        return self.bias + linear + interaction


def fm_predict(model, matrix):
    result = np.empty(len(matrix), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(matrix), 32768):
            end = min(begin + 32768, len(matrix))
            xb = torch.from_numpy(matrix[begin:end])
            result[begin:end] = model(xb).cpu().numpy()
    return result


fm_oof = np.empty(ntr, dtype=np.float64)
fm_valid = np.zeros(nva, dtype=np.float64)
fm_test = np.zeros(nte, dtype=np.float64)

for fold in range(NFOLDS):
    torch.manual_seed(SEED + 100 + fold)
    model = FactorizationMachine(total_cardinality, dimension=12)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-6
    )

    fit_rows = np.flatnonzero(fold_id != fold)
    hold_rows = np.flatnonzero(fold_id == fold)
    local_rng = np.random.default_rng(SEED + 200 + fold)

    model.train()
    for epoch in range(2):
        local_rng.shuffle(fit_rows)
        for begin in range(0, len(fit_rows), 16384):
            rows = fit_rows[begin:begin + 16384]
            xb = torch.from_numpy(fm_xtr[rows])
            yb = torch.from_numpy(ytr[rows])
            wb = torch.from_numpy(sample_weight[rows])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (losses * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    fm_oof[hold_rows] = fm_predict(model, fm_xtr[hold_rows])
    fm_valid += fm_predict(model, fm_xva) / NFOLDS
    fm_test += fm_predict(model, fm_xte) / NFOLDS

    del model, optimizer, fit_rows, hold_rows
    gc.collect()

print("FINDINGS completed_crossfit_family=factorization folds=5")

del fm_xtr, fm_xva, fm_xte
gc.collect()


# -------------------------------------------------------------------------
# Cross-fitted residual/disagreement stacker.
#
# The stacker sees only OOF base predictions on train. Pairwise differences
# explicitly expose regions where structurally different families disagree.
# Stable categorical context allows conditional reliability rather than one
# global blend weight.
# -------------------------------------------------------------------------
STACK_CONTEXT = [
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]


def stack_matrix(sample, tree_score, nb_score, fm_score):
    continuous = np.column_stack([
        tree_score,
        nb_score,
        fm_score,
        tree_score - nb_score,
        tree_score - fm_score,
        nb_score - fm_score,
        np.abs(tree_score - nb_score),
        np.abs(tree_score - fm_score),
        np.abs(nb_score - fm_score),
    ]).astype(np.float32)

    context = np.column_stack([
        np.asarray(sample.X[f], dtype=np.float32)
        for f in STACK_CONTEXT
    ])
    return np.column_stack([continuous, context]).astype(
        np.float32, copy=False
    )


stack_xtr = stack_matrix(train, tree_oof, nb_oof, fm_oof)
stack_xva = stack_matrix(valid, tree_valid, nb_valid, fm_valid)
stack_xte = stack_matrix(test, tree_test, nb_test, fm_test)

n_stack_cont = 9
stack_categorical = list(
    range(n_stack_cont, n_stack_cont + len(STACK_CONTEXT))
)

stack_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 23,
    "max_depth": 7,
    "min_data_in_leaf": 1400,
    "lambda_l2": 8.0,
    "feature_fraction": 0.88,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 24,
    "cat_smooth": 35.0,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED + 300,
    "feature_fraction_seed": SEED + 301,
    "bagging_seed": SEED + 302,
}

stack_dset = lgb.Dataset(
    stack_xtr,
    label=ytr,
    weight=sample_weight,
    categorical_feature=stack_categorical,
    free_raw_data=True,
)
stack_model = lgb.train(stack_params, stack_dset, num_boost_round=150)
stack_valid = stack_model.predict(stack_xva).astype(np.float64)
stack_test = stack_model.predict(stack_xte).astype(np.float64)

del stack_model, stack_dset, stack_xtr, stack_xva, stack_xte
gc.collect()

corr = np.corrcoef(
    np.column_stack([tree_valid, nb_valid, fm_valid]).T
)
print(
    "FINDINGS validation_base_correlations="
    + json.dumps({
        "tree_nb": float(corr[0, 1]),
        "tree_fm": float(corr[0, 2]),
        "nb_fm": float(corr[1, 2]),
    }, sort_keys=True)
)


# -------------------------------------------------------------------------
# Compare every standalone family and every family/incumbent rank blend.
# The trusted-incumbent contract explicitly permits choosing a blend weight
# on validation and applying the same fixed weight to test.
# -------------------------------------------------------------------------
shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

families_valid = {
    "boosted_tree": tree_valid,
    "empirical_bayes": nb_valid,
    "factorization": fm_valid,
    "crossfit_disagreement_stack": stack_valid,
}
families_test = {
    "boosted_tree": tree_test,
    "empirical_bayes": nb_test,
    "factorization": fm_test,
    "crossfit_disagreement_stack": stack_test,
}

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_scores = {}
candidate_arrays = {}
candidate_tests = {}
candidate_raw = {}

# Include the trusted incumbent as a safety reference.
inc_metrics = evaluate(uva, yva, inc_valid_rank)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_arrays["incumbent"] = inc_valid_rank
candidate_tests["incumbent"] = inc_test_rank
candidate_raw["incumbent"] = stack_valid

alphas = [0.15, 0.30, 0.50, 0.70, 1.00]

for family_name in families_valid:
    own_valid = families_valid[family_name]
    own_test = families_test[family_name]

    own_valid_rank = within_user_rank(valid.user_id, own_valid)
    own_test_rank = within_user_rank(test.user_id, own_test)

    for alpha in alphas:
        if alpha == 1.0:
            name = family_name + "_standalone"
            blended_valid = own_valid_rank
            blended_test = own_test_rank
        else:
            name = "%s_blend_%.2f" % (family_name, alpha)
            blended_valid = (
                alpha * own_valid_rank
                + (1.0 - alpha) * inc_valid_rank
            )
            blended_test = (
                alpha * own_test_rank
                + (1.0 - alpha) * inc_test_rank
            )

        metrics = evaluate(uva, yva, blended_valid)
        candidate_scores[name] = float(metrics["primary"])
        candidate_arrays[name] = blended_valid
        candidate_tests[name] = blended_test
        candidate_raw[name] = own_valid

best_name = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_arrays[best_name]
test_scores = candidate_tests[best_name]
final_metrics = evaluate(uva, yva, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected_candidate=%s selected_primary=%.8f"
    % (best_name, float(final_metrics["primary"]))
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    # Always preserve the selected candidate's own unblended model score.
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(candidate_raw[best_name], dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)