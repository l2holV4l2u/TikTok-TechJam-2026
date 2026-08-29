# Bugs / design gaps found during review

Running log. One `##` section per finding, newest at bottom.

## Tree backtrack is dead code for scored misses

`agent/tree.py`'s docstring claims node retirement gives the search backtracking a linear walk
can't do: a node that keeps producing non-improving children is retired, and `select()` falls
back to the next-best live node.

For genuine scored misses, this never fires. `Tree.max_misses` and `run_loop`'s `patience` are
both `3`, checked against the same `epsilon` (0.002), on the same reigning-leader score. A
node's 3rd non-improving *scored* child is, by construction, also the run's 3rd consecutive
non-improvement of the global best -- `agent/loop.py:converged()` is checked at the *top* of
the next iteration, before `tree.select()` runs for it. So the run ends one step before any
fallback selection could take effect.

**The only path where backtrack actually executes:** crashes. A crashed/timed-out iteration
hits `continue` in `agent/loop.py` before `stale` is touched, so it never counts against
`patience`, only against the node's separate crash counter (`Tree.record_failure`, threshold
`2 * max_misses = 6`). A node can die from repeated crashes without spending any of the run's
convergence budget, and only then does `select()` actually return a different node.

**Fix, if the general-purpose backtrack is wanted:** set `max_misses < patience` so a node
retires before the run converges. As shipped, the mechanism as documented only rescues
crash-heavy branches, not genuinely-tried-and-failed ones.
**Resolved 29 Aug 2026.** `--max-misses` and `Tree.max_misses` are now `2` against a
`patience` of `3`, so a node retires one iteration before the run can converge and `select()`
reaches the fallback. Verified by `python -m agent.tree`.

## Cross-iteration search cannot pay for itself under this convergence rule

The fix above makes backtracking reachable, but it does not make it valuable. With
eps = 0.002 and N = 3 fixed by the organizers, a run gets roughly three to six iterations.
Tree search, node retirement and backtracking all need more samples than that to earn their
complexity; they are the right structure for a budget an order of magnitude larger.

What the runs actually show is that the search already lives somewhere else. The convergence
rule charges per ITERATION, not per model, so comparing candidates inside one script is free
and comparing them across scripts costs a life:

| run | iterations | candidates compared inside them |
|---|---|---|
| r59 | 6 | 161 |
| r70 | 6 | 39 |

A 4x spread means that search was incidental, not designed. The brief now states the
economics explicitly. The cross-iteration tree remains as an audit trail and as the mechanism
that survives crash-heavy branches, which is what it is actually good for here.

## Parent code truncation silently corrupts the reference script

`agent/proposer.py` sends the selected parent node's source in full, hard-truncated at
`MAX_CODE_CHARS = 14000` (`parent.code[:MAX_CODE_CHARS]`, a plain slice with no boundary
awareness). Measured across `runs/`: 35 of 442 generated scripts (8%) exceed 14,000 characters.

When one of those is selected as parent, the proposer is told "here is the parent script, make
a targeted edit" and shown a slice that ends mid-statement. Confirmed on
`runs/r39/scripts/iter_6.py` (22,889 chars): the cut lands inside a numpy slice expression,
`1:])`, mid-token. The model cannot see whatever runs after that point -- which may include the
`scores_test.npy` save, the leakage-safe aggregate, or the metrics line -- while being asked to
edit it as if it saw the whole thing.

Later, more-evolved scripts (which accumulate feature engineering across iterations) are the
ones most likely to cross the cap, so this disproportionately hits exactly the highest-value
parents. Measured prompt size overall already reaches ~12,000 tokens uncapped
(`agent/proposer.py:27-29`), so this is one symptom of a broader context-budget problem, not
an isolated one.

**Not yet fixed.** A smarter truncation (cut at a line/def boundary, or summarize the tail
instead of dropping it) would at least make the loss legible instead of silent.
