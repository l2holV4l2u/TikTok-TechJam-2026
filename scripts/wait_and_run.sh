#!/usr/bin/env bash
# Wait until the friend keys hold enough sol quota for a full run, then start one.
# Full-size prompt on purpose: oversized calls are served whenever the rolling TPM window is
# quiet -- r35, r43, r53 and r59 all converged with calls over 10k -- and the 429 retry budget
# now stops a rate limit from eating the daily allowance.
set -u
ENV_FILE="$1"; RUN_DIR="$2"; NEED="${3:-13}"
set -a; . "$ENV_FILE"; set +a
export PYTHONPATH=. PYTHONIOENCODING=utf-8 LLM_MIN_INTERVAL_S=25
unset LLM_PROMPT_CHAR_BUDGET

read_quota() {
  python - <<'PY'
import os, json, urllib.request, urllib.error
tot = 0
for k in [k.strip() for k in os.environ["OPENAI_API_KEYS"].split(",")]:
    body = json.dumps({"model": "gpt-5.6-sol",
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_completion_tokens": 16}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=60)
        tot += int(r.headers.get("x-ratelimit-remaining-requests") or 0)
    except urllib.error.HTTPError as e:
        tot += int(e.headers.get("x-ratelimit-remaining-requests") or 0)
    except Exception:
        pass
print(tot)
PY
}

for _ in $(seq 1 48); do
  left="$(read_quota)"
  echo "$(date +%H:%M) sol quota=${left:-?} (need $NEED)"
  if [ "${left:-0}" -ge "$NEED" ]; then
    mkdir -p "$RUN_DIR"
    echo "$(date +%H:%M) launching full-size run in $RUN_DIR"
    exec python -u run_agent.py --run-dir "$RUN_DIR" --timeout 1800
  fi
  sleep 1200
done
echo "gave up waiting for quota"
