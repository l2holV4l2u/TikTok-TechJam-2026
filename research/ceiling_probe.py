"""How much of this task is learnable at all?

    python -m research.ceiling_probe

Human analysis, NOT on the submission path -- see research/README.md.

The question this answers: every method anyone tried on KuaiRand-Pure lands within about
+-0.005 of the official baseline. Is that because the models are too small, because the
features carry little signal, or because whatever signal exists does not survive the date
split? Those have very different implications, and the answer decides whether chasing a large
delta is worth anyone's time.

Method: fit a deliberately over-powered gradient-boosted model on all 37 categorical fields and
score it BOTH in-sample and on validation. In-sample performance upper-bounds what the feature
set can express when memorisation is allowed. The gap to validation is what fails to transfer.
"""
import time

import lightgbm as lgb
import numpy as np

from pipeline.data import FEATURE_CARDINALITIES as FC
from pipeline.data import load
from pipeline.evaluate import evaluate

BASELINE_VALID = {"gauc": 0.6674, "ndcg@5": 0.5357, "primary": 0.6016}


def _raw(split, fields):
    return np.stack([np.minimum(split.X[f], FC[f] - 1) for f in fields], 1)


def main(num_leaves: int = 255, rounds: int = 250, lr: float = 0.2) -> None:
    t0 = time.perf_counter()
    tr, va = load("train"), load("valid")
    fields = list(FC)
    Xtr, Xva = _raw(tr, fields), _raw(va, fields)

    model = lgb.train(
        dict(objective="binary", num_leaves=num_leaves, learning_rate=lr,
             min_data_in_leaf=5, verbose=-1),
        lgb.Dataset(Xtr, tr.y, categorical_feature=list(range(len(fields)))),
        num_boost_round=rounds,
    )

    ins = evaluate(tr.user_id, tr.y, model.predict(Xtr))
    oos = evaluate(va.user_id, va.y, model.predict(Xva))
    oracle = evaluate(va.user_id, va.y, va.y.astype(float))

    rows = [
        ("high-capacity, IN-SAMPLE (train)", ins),
        ("same model, validation", oos),
        ("official baseline, validation", BASELINE_VALID),
        ("oracle (true labels), validation", oracle),
    ]
    print(f"{'':34}{'GAUC':>9}{'nDCG@5':>9}{'primary':>9}")
    for name, r in rows:
        print(f"{name:34}{r['gauc']:>9.4f}{r['ndcg@5']:>9.4f}{r['primary']:>9.4f}")

    gap = ins["primary"] - oos["primary"]
    print(f"\ngeneralisation gap (in-sample - validation): {gap:.4f}")
    print(f"validation vs official baseline:             {oos['primary'] - 0.6016:+.4f}")
    print(f"elapsed {time.perf_counter() - t0:.0f}s  "
          f"(num_leaves={num_leaves}, rounds={rounds}, lr={lr})")

    # the finding, asserted so this stops being an anecdote if the data ever changes
    assert ins["primary"] > 0.85, "expected near-memorisation in-sample"
    assert oos["primary"] < 0.6016, "expected the over-powered model to LOSE to the baseline"
    print("\nCapacity is not the binding constraint; transfer across the date split is.")


if __name__ == "__main__":
    main()
