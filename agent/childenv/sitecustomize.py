"""Loaded automatically in every generated script's interpreter, before its own imports.

torch ships its own OpenMP runtime (`torch/lib/libomp.dylib`) and LightGBM links a different
one. Two OpenMP runtimes in one process do not coexist: on macOS the second one either
segfaults the process or deadlocks it. Whichever library loads first wins, so the fix is
entirely about import order --

    import lightgbm; import torch    -> works
    import torch; import lightgbm    -> SIGSEGV, or a hang inside Dataset construction

-- and `KMP_DUPLICATE_LIB_OK=TRUE` does not help.

Generated scripts import torch at the top, so they lose that race by default. r83 lost three
of its fifteen improve iterations to it: #2 and #3 hung inside `lgb.Dataset.__init_from_np2d`
and were killed at the 900s timeout, burning 1,800 of the run's 2,806 script-seconds, and #14
segfaulted (1.6s, empty stdout, empty stderr, non-zero exit -- a failure with no diagnosis in
it). The same LightGBM stage runs in 22s in a process that never imported torch.

Doing it here rather than in the task brief keeps it out of the prompt: the agent cannot
forget an import it never has to make, it costs no prompt budget, and it expresses no opinion
about which methods to try.

This module is NOT on the controller's own path -- `agent/childenv/` is added to PYTHONPATH by
`executor.run_script` for the child only, so the harness does not pay to import LightGBM.
"""

try:  # never let a missing or broken optional dependency stop a script from starting
    import lightgbm  # noqa: F401
except Exception:
    pass
