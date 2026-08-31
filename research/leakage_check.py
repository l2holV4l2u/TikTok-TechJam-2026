"""Would training on validation explain our scores? Run this yourself; trust the output, not us.

Fits one model two ways and scores both splits. If the agent were training on validation, its
reported validation number would sit near the train+valid row, not the train-only row.
"""
import numpy as np, lightgbm as lgb
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

tr, va, te = load("train"), load("valid"), load("test")
F = list(FEATURE_CARDINALITIES)[:8]
X = lambda s: np.column_stack([np.asarray(s.X[f]) for f in F]).astype(np.float32)
Xtr, Xva, Xte = X(tr), X(va), X(te)
ytr, yva, yte = np.asarray(tr.y), np.asarray(va.y), np.asarray(te.y)
p = dict(objective="binary", num_leaves=63, learning_rate=0.1, verbose=-1, num_threads=4)
cf = list(range(len(F)))

fits = {
    "train only ": lgb.train(p, lgb.Dataset(Xtr, ytr, categorical_feature=cf), num_boost_round=120),
    "train+valid": lgb.train(p, lgb.Dataset(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]),
                                            categorical_feature=cf), num_boost_round=120),
}
print(f"{'fitted on':<12} {'valid':>9} {'test':>9} {'test delta':>11}")
for name, m in fits.items():
    v = evaluate(va.user_id, yva, m.predict(Xva))["primary"]
    t = evaluate(te.user_id, yte, m.predict(Xte))["primary"]
    print(f"{name:<12} {v:>9.5f} {t:>9.5f} {t - 0.5946:>+11.5f}")
print("\nOur submitted run reports validation 0.605309. Compare it to the two rows above.")
