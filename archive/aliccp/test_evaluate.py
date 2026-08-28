"""Run: python -m tests.test_evaluate. Plain asserts, no pytest, no torch, no data."""
import math
import warnings

from pipeline.evaluate import ctr_cvr_auc


def test_ctr_auc_hand_computed_no_ties():
    # negs=[0.1,0.4], pos=[0.3,0.6] -> concordant pairs 3/4 -> AUC 0.75 (verified by hand)
    y_click = [0, 0, 1, 1]
    p_ctr = [0.1, 0.4, 0.3, 0.6]
    y_conv = [0, 0, 0, 0]
    p_cvr = [0.1, 0.2, 0.3, 0.4]
    out = ctr_cvr_auc(y_click, y_conv, p_ctr, p_cvr)
    assert math.isclose(out["ctr_auc"], 0.75, rel_tol=1e-9)


def test_ties_all_tied_scores_give_half():
    y_click = [0, 1, 0, 1]
    p_ctr = [0.5, 0.5, 0.5, 0.5]
    y_conv = [0, 0, 0, 0]
    p_cvr = [0.5, 0.5, 0.5, 0.5]
    out = ctr_cvr_auc(y_click, y_conv, p_ctr, p_cvr)
    assert math.isclose(out["ctr_auc"], 0.5, rel_tol=1e-9)


def test_ties_partial_ties_hand_computed():
    # negs=[0.3,0.5], pos=[0.5,0.7]; the 0.5/0.5 pair counts as half a concordant pair -> 3.5/4 = 0.875
    y_click = [0, 0, 1, 1]
    p_ctr = [0.3, 0.5, 0.5, 0.7]
    y_conv = [0, 0, 0, 0]
    p_cvr = [0.1, 0.2, 0.3, 0.4]
    out = ctr_cvr_auc(y_click, y_conv, p_ctr, p_cvr)
    assert math.isclose(out["ctr_auc"], 0.875, rel_tol=1e-9)


def test_cvr_auc_ignores_unclicked_rows():
    # rows 2 and 4 are unclicked with deliberately poisonous labels/scores that would
    # wreck the AUC if mistakenly included; true CVR AUC uses only rows 0,1,3 -> 1.0
    y_click = [1, 1, 0, 1, 0]
    y_conv = [0, 1, 0, 1, 1]
    p_ctr = [0.1, 0.2, 0.3, 0.4, 0.5]
    p_cvr = [0.2, 0.9, 0.99, 0.7, 0.01]
    out = ctr_cvr_auc(y_click, y_conv, p_ctr, p_cvr)
    assert math.isclose(out["cvr_auc"], 1.0, rel_tol=1e-9)


def test_zero_positive_ctr_returns_nan():
    y_click = [0, 0, 0]
    y_conv = [0, 0, 0]
    p_ctr = [0.1, 0.2, 0.3]
    p_cvr = [0.1, 0.2, 0.3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ctr_cvr_auc(y_click, y_conv, p_ctr, p_cvr)
    assert math.isnan(out["ctr_auc"])
    assert math.isnan(out["cvr_auc"])  # zero clicks -> empty clicked subset -> also undefined
    assert out["mean_delta_ready"] is False


def test_zero_positive_cvr_only_returns_nan_while_ctr_stays_valid():
    # 2 clicks, both non-converting -> clicked subset has zero positives -> cvr nan, ctr unaffected
    y_click = [1, 1, 0]
    y_conv = [0, 0, 0]
    p_ctr = [0.1, 0.6, 0.2]
    p_cvr = [0.3, 0.4, 0.5]
    out = ctr_cvr_auc(y_click, y_conv, p_ctr, p_cvr)
    assert not math.isnan(out["ctr_auc"])
    assert math.isnan(out["cvr_auc"])
    assert out["mean_delta_ready"] is False


def test_zero_negative_class_returns_nan_not_plausible_default():
    # all positives -> no negatives to rank against -> must be nan, not a silently-plausible 0.5 or 1.0
    y_click = [1, 1, 1]
    y_conv = [1, 1, 1]
    p_ctr = [0.1, 0.2, 0.3]
    p_cvr = [0.1, 0.2, 0.3]
    out = ctr_cvr_auc(y_click, y_conv, p_ctr, p_cvr)
    assert math.isnan(out["ctr_auc"])
    assert math.isnan(out["cvr_auc"])


def test_mean_delta_ready_true_when_both_aucs_valid():
    y_click = [0, 0, 1, 1]
    y_conv = [0, 1, 0, 1]
    p_ctr = [0.1, 0.4, 0.3, 0.6]
    p_cvr = [0.2, 0.9, 0.99, 0.7]
    out = ctr_cvr_auc(y_click, y_conv, p_ctr, p_cvr)
    assert out["mean_delta_ready"] is True


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(tests)} tests passed")
