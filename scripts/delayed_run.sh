#!/usr/bin/env bash
# Sleep, then start a full-size run. It does NOT poll the quota: probing all three keys costs
# 3 daily requests per check while they only regenerate ~2 in the same window, so the watcher
# that checked every 20 minutes drove the balance from 9 down to 0 without ever launching.
# If the quota is still short the run stops itself on llm_daily_limit, having at least
# reproduced the baseline.
set -u
ENV_FILE="$1"; RUN_DIR="$2"; WAIT_S="${3:-9000}"
set -a; . "$ENV_FILE"; set +a
export PYTHONPATH=. PYTHONIOENCODING=utf-8 LLM_MIN_INTERVAL_S=25
unset LLM_PROMPT_CHAR_BUDGET
echo "$(date +%H:%M) sleeping ${WAIT_S}s so the daily allowance can rebuild"
sleep "$WAIT_S"
mkdir -p "$RUN_DIR"
echo "$(date +%H:%M) launching full-size run in $RUN_DIR"
exec python -u run_agent.py --run-dir "$RUN_DIR" --timeout 1800
