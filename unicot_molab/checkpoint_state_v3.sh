#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/zebra_Cot-BAGEL-UniMMMU-test"
cd "$ROOT"

OUTPUT_NAME="${OUTPUT_NAME:-unicot_native_v02_state_v3}"
GIT_BRANCH="${GIT_BRANCH:-unicot-eval-molab}"

mkdir -p "runs/$OUTPUT_NAME" logs
touch "logs/${OUTPUT_NAME}.log"

exec 9>"$ROOT/.git/unicot_state_v3_checkpoint.lock"
flock -w 120 9

git add -A -- "runs/$OUTPUT_NAME"

git add -f --   "logs/${OUTPUT_NAME}.log"   2>/dev/null || true

if ! git diff --cached --quiet; then
    git commit -m       "checkpoint: ${OUTPUT_NAME} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

git push origin "HEAD:${GIT_BRANCH}"
