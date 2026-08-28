"""Dataset facts for the task brief, measured from the cache rather than written by hand.

The brief used to state KuaiRand-Pure's row counts, metric ceiling and self-check rungs as
literals. That is fine until the harness is pointed at another variant, at which point the
agent is reasoning from false premises. Measuring them also removes one more block of
human-authored numbers from the prompt.

  python -m agent.facts --baseline research/reference_pure.json --out runs/facts_pure.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.data import load
from pipeline.evaluate import evaluate


def measure(baseline: dict, seed: int = 0) -> dict:
    """Row counts, date windows, the perfect-ranking ceiling and the random/item-pop rungs."""
    tr, va, te = load("train"), load("valid"), load("test")
    f: dict = {}
    for name, sp in (("train", tr), ("valid", va), ("test", te)):
        d = np.asarray(sp.date)
        f[f"{name}_rows"] = len(sp.y)
        f[f"{name}_lo"], f[f"{name}_hi"] = int(d.min()), int(d.max())
        f[f"{name}_days"] = int(len(np.unique(d)))

    te_u, te_y = np.asarray(te.user_id), np.asarray(te.y).astype(np.float64)
    # scoring the label itself is a perfect ranking: the ceiling primary can reach here
    f["ceiling"] = evaluate(te_u, te.y, te_y)["primary"]
    # users with no positive score 0 on nDCG and are excluded from GAUC -- they are why the
    # ceiling is not 1.0, so the agent needs the number to judge headroom
    order = np.argsort(te_u, kind="stable")
    u_sorted, y_sorted = te_u[order], te_y[order]
    _, starts = np.unique(u_sorted, return_index=True)
    pos_per_user = np.add.reduceat(y_sorted, starts)
    f["zero_pos_user_pct"] = 100.0 * float((pos_per_user == 0).mean())

    rng = np.random.default_rng(seed)
    f["random_primary"] = evaluate(te_u, te.y, rng.random(len(te.y)))["primary"]

    tr_v, tr_y = np.asarray(tr.video_id), np.asarray(tr.y).astype(np.float64)
    n_vid = int(max(tr_v.max(), np.asarray(te.video_id).max())) + 1
    pos = np.bincount(tr_v, weights=tr_y, minlength=n_vid)
    cnt = np.maximum(np.bincount(tr_v, minlength=n_vid), 1)
    f["itempop_primary"] = evaluate(te_u, te.y, (pos / cnt)[np.asarray(te.video_id)])["primary"]

    # '@' is not a legal str.format key, so ndcg@5 -> ndcg5
    f.update({f"baseline_{k}".replace("@", ""): v
              for k, v in baseline.items() if isinstance(v, (int, float))})
    f["baseline_source"] = baseline.get("source", "our reproduction of the organizers' recipe")
    std = baseline.get("seed_std")
    f["baseline_noise_note"] = f"  (mean of 5 seeds, std {std})" if std else ""
    from pipeline.data import NUMERIC_FEATURES
    present = sorted(tr.num) if tr.num else []
    f["n_numeric"] = len(present)
    f["numeric_names"] = ", ".join(present) if present else "(none cached)"
    try:
        import torch as _t
        f["torch_version"] = _t.__version__
        f["gpu"] = (f"{_t.cuda.get_device_properties(0).name}, "
                    f"{_t.cuda.get_device_properties(0).total_memory/1e9:.1f} GB"
                    if _t.cuda.is_available() else "none")
    except Exception:
        f["torch_version"], f["gpu"] = "unavailable", "none"
    from pipeline.data import FEATURE_CARDINALITIES
    cats = sorted(FEATURE_CARDINALITIES)
    f["n_categorical"] = len(cats)
    f["categorical_names"] = ", ".join(cats)
    f["train_rows_m"] = f["train_rows"] / 1e6
    return f


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="JSON from research.baseline_reference")
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="KuaiRand-Pure")
    args = p.parse_args()
    facts = measure(json.loads(Path(args.baseline).read_text(encoding="utf-8")))
    facts["variant"] = args.variant
    Path(args.out).write_text(json.dumps(facts, indent=2), encoding="utf-8")
    for k, v in facts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
