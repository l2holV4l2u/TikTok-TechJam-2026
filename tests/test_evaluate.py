"""Run: python -m tests.test_evaluate. Plain asserts, no pytest, no data files."""
import math

import numpy as np

import pipeline.evaluate as ev
from pipeline.evaluate import evaluate

LOG2_2 = math.log2(2.0)
LOG2_3 = math.log2(3.0)
LOG2_4 = math.log2(4.0)
LOG2_5 = math.log2(5.0)


def test_combined_hand_computed_A_B_C():
    # user A: negs=[0.1,0.4], pos=[0.3,0.6] -> AUC 0.75 (same numbers as archive/aliccp AUC test)
    # user B: all-positive (2/2) -> excluded from GAUC, still counted for nDCG
    # user C: zero-positive -> excluded from GAUC, contributes 0 to nDCG average
    user_ids = ["A", "A", "A", "A", "B", "B", "C", "C"]
    scores =   [0.1, 0.4, 0.3, 0.6, 0.2, 0.5, 0.3, 0.7]
    labels =   [0,   0,   1,   1,   1,   1,   0,   0]

    out = evaluate(user_ids, labels, scores)

    # GAUC: only A is valid (0 < positives < impressions); weight doesn't matter with 1 valid user
    assert math.isclose(out["gauc"], 0.75, rel_tol=1e-9)

    # nDCG@5 for A: predicted order desc by score -> labels [1,0,1,0] at ranks 1..4
    dcg_a = 1.0 / LOG2_2 + 0.0 / LOG2_3 + 1.0 / LOG2_4 + 0.0 / LOG2_5
    idcg_a = 1.0 / LOG2_2 + 1.0 / LOG2_3 + 0.0 / LOG2_4 + 0.0 / LOG2_5
    ndcg_a = dcg_a / idcg_a

    # nDCG@5 for B: order desc -> labels [1,1] both positive -> perfect ranking -> 1.0
    ndcg_b = 1.0

    # nDCG@5 for C: zero positives -> IDCG=0 -> defined as 0
    ndcg_c = 0.0

    expected_ndcg5 = (ndcg_a + ndcg_b + ndcg_c) / 3.0
    assert math.isclose(out["ndcg@5"], expected_ndcg5, rel_tol=1e-9)
    assert math.isclose(out["primary"], (out["gauc"] + out["ndcg@5"]) / 2.0, rel_tol=1e-9)


def test_gauc_all_positive_user_excluded_but_ndcg_counts_it():
    user_ids = ["B", "B"]
    scores = [0.5, 0.2]
    labels = [1, 1]
    out = evaluate(user_ids, labels, scores)
    assert math.isnan(out["gauc"])  # only user has positives == impressions -> no valid users
    assert math.isclose(out["ndcg@5"], 1.0, rel_tol=1e-9)  # perfect ranking, all positive


def test_gauc_zero_positive_user_excluded_ndcg_contributes_zero():
    user_ids = ["C", "C"]
    scores = [0.7, 0.3]
    labels = [0, 0]
    out = evaluate(user_ids, labels, scores)
    assert math.isnan(out["gauc"])
    assert out["ndcg@5"] == 0.0


def test_gauc_weighting_changes_result():
    # user A: 1 pos vs 1 neg, perfect ranking -> AUC 1.0
    # user B: 3 pos vs 1 neg, fully reversed -> AUC 0.0
    # simple mean would be 0.5; positive-count-weighted mean is 0.25
    user_ids = ["A", "A", "B", "B", "B", "B"]
    scores = [0.1, 0.9, 0.9, 0.1, 0.2, 0.3]
    labels = [0, 1, 0, 1, 1, 1]
    out = evaluate(user_ids, labels, scores)
    assert math.isclose(out["gauc"], 0.25, rel_tol=1e-9)
    assert not math.isclose(out["gauc"], 0.5, rel_tol=1e-9)


def test_exponential_gain_not_linear_gain():
    # true relevances: item1=2, item2=1, item3=0 (graded, exercises 2^rel-1 vs linear rel)
    # predicted order swaps the top two vs ideal order
    user_ids = ["X", "X", "X"]
    scores = [0.5, 0.9, 0.1]  # item1=0.5(rel2) rank2, item2=0.9(rel1) rank1, item3=0.1(rel0) rank3
    labels = [2, 1, 0]
    out = evaluate(user_ids, labels, scores)

    dcg_exp = (2.0 ** 1 - 1) / LOG2_2 + (2.0 ** 2 - 1) / LOG2_3 + (2.0 ** 0 - 1) / LOG2_4
    idcg_exp = (2.0 ** 2 - 1) / LOG2_2 + (2.0 ** 1 - 1) / LOG2_3 + (2.0 ** 0 - 1) / LOG2_4
    expected_exp = dcg_exp / idcg_exp

    dcg_lin = 1.0 / LOG2_2 + 2.0 / LOG2_3 + 0.0 / LOG2_4
    idcg_lin = 2.0 / LOG2_2 + 1.0 / LOG2_3 + 0.0 / LOG2_4
    expected_lin = dcg_lin / idcg_lin

    assert not math.isclose(expected_exp, expected_lin, rel_tol=1e-9)  # sanity: the two gain rules disagree here
    assert math.isclose(out["ndcg@5"], expected_exp, rel_tol=1e-9)
    assert not math.isclose(out["ndcg@5"], expected_lin, rel_tol=1e-6)


def test_score_tie_break_is_original_row_order():
    # tied scores: row 0 must outrank row 1 (documented tiebreak = original row order)
    user_ids = ["Y", "Y"]
    scores = [0.5, 0.5]
    labels = [0, 1]  # negative listed first, positive listed second
    out = evaluate(user_ids, labels, scores)
    # row0(label0) ranks 1st, row1(label1) ranks 2nd under the documented tiebreak
    dcg = 0.0 / LOG2_2 + 1.0 / LOG2_3
    idcg = 1.0 / LOG2_2 + 0.0 / LOG2_3
    assert math.isclose(out["ndcg@5"], dcg / idcg, rel_tol=1e-9)
    assert not math.isclose(out["ndcg@5"], 1.0, rel_tol=1e-9)  # would be 1.0 under the opposite tiebreak


def test_perfect_ranking_gives_one_reversed_gives_zero_auc():
    user_ids = ["P", "P", "P", "P"]
    scores = [0.9, 0.8, 0.2, 0.1]  # positives strictly above negatives
    labels = [1, 1, 0, 0]
    out = evaluate(user_ids, labels, scores)
    assert math.isclose(out["gauc"], 1.0, rel_tol=1e-9)
    assert math.isclose(out["ndcg@5"], 1.0, rel_tol=1e-9)

    reversed_scores = [0.1, 0.2, 0.8, 0.9]  # negatives strictly above positives
    out_rev = evaluate(user_ids, labels, reversed_scores)
    assert math.isclose(out_rev["gauc"], 0.0, rel_tol=1e-9)


def test_ndcg10_matches_ndcg5_when_group_smaller_than_five():
    user_ids = ["Z", "Z"]
    scores = [0.9, 0.1]
    labels = [1, 0]
    out = evaluate(user_ids, labels, scores)
    assert math.isclose(out["ndcg@5"], out["ndcg@10"], rel_tol=1e-9)
    assert math.isclose(out["ndcg@5"], 1.0, rel_tol=1e-9)


def test_recall_at_50_excludes_zero_positive_users_from_average():
    user_ids = ["P", "P", "P", "Z", "Z"]
    scores = [0.9, 0.5, 0.1, 0.6, 0.4]
    labels = [1, 0, 0, 0, 0]  # P has 1 positive ranked first (in top-50); Z has none
    out = evaluate(user_ids, labels, scores)
    assert math.isclose(out["recall@50"], 1.0, rel_tol=1e-9)  # only P counts, not averaged down by Z


def test_recall_at_50_only_counts_top_50():
    # 51 rows for one user: the single positive sits at rank 51 (score is the lowest) -> excluded from top-50
    n = 51
    scores = list(range(n, 0, -1))  # strictly decreasing scores -> rank == position + 1
    labels = [0] * (n - 1) + [1]  # positive is the lowest-scored row -> rank 51
    user_ids = ["U"] * n
    out = evaluate(user_ids, labels, scores)
    assert out["recall@50"] == 0.0

    labels_in_window = [0] * (n - 2) + [1, 0]  # move the positive to rank 50 -> included
    out2 = evaluate(user_ids, labels_in_window, scores)
    assert out2["recall@50"] == 1.0


def test_empty_input_returns_nan_without_crashing():
    out = evaluate([], [], [])
    assert math.isnan(out["gauc"])
    assert math.isnan(out["ndcg@5"])
    assert math.isnan(out["primary"])
    assert math.isnan(out["ndcg@10"])
    assert math.isnan(out["recall@50"])


def test_grouping_does_not_assume_sorted_or_contiguous_users():
    # interleaved, unsorted user_ids must still group correctly
    user_ids = ["B", "A", "B", "A", "C"]
    scores = [0.2, 0.1, 0.5, 0.9, 0.3]
    labels = [1, 0, 1, 1, 0]
    out_shuffled = evaluate(user_ids, labels, scores)

    user_ids2 = ["A", "A", "B", "B", "C"]
    scores2 = [0.1, 0.9, 0.2, 0.5, 0.3]
    labels2 = [0, 1, 1, 1, 0]
    out_sorted = evaluate(user_ids2, labels2, scores2)

    assert math.isclose(out_shuffled["gauc"], out_sorted["gauc"], rel_tol=1e-9)
    assert math.isclose(out_shuffled["ndcg@5"], out_sorted["ndcg@5"], rel_tol=1e-9)




def test_per_user_arrays_reconstruct_the_published_aggregates():
    """The breakdown must be the same numbers, not a second implementation of the metric.

    Its purpose is diagnosis: the aggregate cannot say WHERE a model loses, and reimplementing
    per-user nDCG by hand is exactly how the zero-positive convention or the AUC tie handling
    gets subtly wrong. So it has to agree with the scored aggregate to the last bit.
    """
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(300), rng.integers(1, 20, size=300))
    labels = rng.integers(0, 2, size=users.size)
    scores = rng.normal(size=users.size)

    agg = evaluate(users, labels, scores)
    det = evaluate(users, labels, scores, per_user=True)
    for k in ("gauc", "ndcg@5", "primary", "ndcg@10", "recall@50"):
        assert abs(agg[k] - det[k]) < 1e-12, k

    pu = det["per_user"]
    n, p, auc, nd = pu["n_impressions"], pu["n_positives"], pu["auc"], pu["ndcg@5"]
    assert len(n) == len(p) == len(auc) == len(nd) == len(pu["user_id"]) == 300
    assert (n == np.bincount(users)).all()
    assert (p == np.bincount(users, weights=labels)).all()

    m = ~np.isnan(auc)
    assert abs(np.average(auc[m], weights=p[m]) - agg["gauc"]) < 1e-12, \
        "GAUC is the positive-count-weighted mean of per-user AUC"
    assert abs(nd.mean() - agg["ndcg@5"]) < 1e-12, \
        "nDCG@5 is the plain mean over ALL users, zero-positive ones included as 0"
    excluded = (p == 0) | (p == n)
    assert np.isnan(auc[excluded]).all(), "users with no positive or no negative are not in GAUC"
    assert not np.isnan(auc[~excluded]).any()
    assert (nd[p == 0] == 0).all(), "zero-positive users score nDCG 0, per the spec"


def test_per_user_is_off_by_default_so_the_scored_path_is_unchanged():
    users = np.array([1, 1, 2, 2])
    out = evaluate(users, np.array([1, 0, 0, 1]), np.array([0.9, 0.1, 0.2, 0.8]))
    assert "per_user" not in out


def test_closed_form_binary_idcg_matches_the_sorting_path():
    """The 0/1 shortcut must agree with the general sort-based IDCG, or nDCG silently drifts.

    evaluate() skips a full sort of the labels when every label is 0 or 1, computing IDCG@k
    from each user's positive count instead. That is only safe while the two agree exactly,
    including users with more positives than k and users with none.
    """
    rng = np.random.default_rng(7)
    for n_users, n_per in ((40, 3), (25, 12), (10, 60)):
        users = np.repeat(np.arange(n_users), n_per)
        labels = rng.integers(0, 2, users.size)
        scores = rng.random(users.size)
        idx = np.arange(users.size)

        order, gid, _rank1, n_groups = ev._group_by_user(users, scores, idx)
        n_pos = np.bincount(gid, weights=labels[order].astype(np.float64), minlength=n_groups)

        o_ideal, gid_ideal, rank_ideal, _ng = ev._group_by_user(users, labels.astype(np.float64), idx)
        for k in (5, 10):
            by_sort = ev._dcg_sum(labels[o_ideal].astype(np.float64), rank_ideal, gid_ideal, n_groups, k)
            closed = ev._idcg_binary(n_pos, k)
            assert np.allclose(by_sort, closed, rtol=0, atol=0), (n_per, k, by_sort[:5], closed[:5])

    # a graded label set must fall back to the sorting path, not the shortcut
    assert ev._is_binary(np.array([0.0, 1.0, 1.0]))
    assert not ev._is_binary(np.array([0.0, 1.0, 2.0]))


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(tests)} tests passed")
