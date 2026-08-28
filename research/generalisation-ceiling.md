
## Selection is not the bottleneck

A natural response to a noisy one-week validation split is to build a better selector -- for
example scoring candidates on a train-only chronological fold as well. We measured the ceiling
on that idea before building it, by asking what a *perfect* selector would have been worth: for
each run, compare the test score of the validation-best iteration against the test score of the
best iteration available in that run.

| run | validation-best | its test | best test in run | cost of selecting on validation |
|---|---|---|---|---|
| r33 | 0.6044 | 0.5981 | 0.5981 | 0.0000 |
| r34 | 0.6043 | 0.5979 | 0.5984 | 0.0005 |
| r35 | 0.6049 | 0.5985 | 0.5988 | 0.0003 |
| r36 | 0.6037 | 0.5982 | 0.5982 | 0.0000 |
| r37 | 0.6037 | 0.5987 | 0.5987 | 0.0000 |
| r39 | 0.6053 | 0.5997 | 0.5997 | 0.0000 |
| r40 | 0.6036 | 0.5976 | 0.5976 | 0.0000 |
| r41 | 0.6059 | 0.5996 | 0.5998 | 0.0002 |

Validation picks the test-best iteration in **5 of 8 runs**, and an oracle with access to the
hidden test set would gain **+0.0001 primary on average** -- about 2% of our +0.0050 delta.

The problem is not choosing among the iterations a run produces. It is that the best iteration
a run produces is not better. Effort spent on the selection signal is therefore capped at
roughly a fiftieth of the effort spent on what gets proposed.

One thing this does **not** rule out: a temporal fold could still change *which experiments the
agent proposes*, by putting drift evidence into `FINDINGS` and the belief set. That is a
different mechanism from selection and this measurement says nothing about it.
