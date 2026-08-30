"""Does a late-window validation selector rank iterations more like the hidden test than full validation?"""
import json, sys
from pathlib import Path
import numpy as np
from pipeline.data import load
from pipeline.evaluate import evaluate

va, te = load('valid'), load('test')
vdate = np.asarray(va.date)
days = np.unique(vdate)

def prim(sp, s, mask=None):
    if mask is None:
        return evaluate(sp.user_id, sp.y, s)['primary']
    return evaluate(np.asarray(sp.user_id)[mask], np.asarray(sp.y)[mask],
                    np.asarray(s, np.float64)[mask])['primary']

rows = []
for d in sorted(Path('runs').glob('r*/scripts/*_out')):
    v, t = d / 'scores_valid.npy', d / 'scores_test.npy'
    if not (v.exists() and t.exists()):
        continue
    sv, st = np.load(v), np.load(t)
    if sv.shape[0] != len(va) or st.shape[0] != len(te):
        continue
    if not (np.isfinite(sv).all() and np.isfinite(st).all()):
        continue
    r = {'run': d.parts[1], 'iter': d.name, 'full': prim(va, sv), 'test': prim(te, st)}
    for k in (2, 3, 4):
        r[f'last{k}'] = prim(va, sv, np.isin(vdate, days[-k:]))
    rows.append(r)

print(f'{len(rows)} scored iterations with both splits\n')
sel = ['full', 'last2', 'last3', 'last4']

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

test = np.array([r['test'] for r in rows])
print(f"{'selector':>9} {'spearman vs test':>17} {'test of its pick':>17} {'regret':>9}")
best_test = test.max()
for s in sel:
    x = np.array([r[s] for r in rows])
    pick = test[int(np.argmax(x))]
    print(f'{s:>9} {spearman(x, test):>17.4f} {pick:>17.6f} {best_test-pick:>9.6f}')
print(f"\noracle (best achievable test) {best_test:.6f}")

# per-run: the selector only ever chooses WITHIN one run
print('\nper-run pick (which iteration each selector would submit)')
print(f"{'run':>6} {'n':>3} " + ' '.join(f'{s:>9}' for s in sel) + f"{'oracle':>10}")
tot = {s: 0.0 for s in sel}; n_runs = 0
for run in sorted({r['run'] for r in rows}):
    g = [r for r in rows if r['run'] == run]
    if len(g) < 3:
        continue
    n_runs += 1
    tt = np.array([r['test'] for r in g])
    line = f'{run:>6} {len(g):>3} '
    for s in sel:
        p = tt[int(np.argmax([r[s] for r in g]))]
        tot[s] += p
        line += f'{p:>9.5f} '
    print(line + f'{tt.max():>10.5f}')
print('\nmean test of the submitted model, over', n_runs, 'runs')
for s in sel:
    print(f'  {s:>6}  {tot[s]/n_runs:.6f}')
