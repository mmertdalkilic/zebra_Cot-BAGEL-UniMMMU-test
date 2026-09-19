#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
cd "$ROOT"

BRANCH="${UNICOT_BRANCH:-unicot-eval-molab}"
REMOTE="${GIT_REMOTE:-origin}"

current="$(git branch --show-current)"
if [[ "$current" != "$BRANCH" ]]; then
  echo "[git-sync] ERROR: on '$current', expected '$BRANCH'" >&2
  exit 1
fi

mkdir -p runs/unicot_native_v02 logs

git add -A runs/unicot_native_v02 logs/unicot_native_v02.log 2>/dev/null || true

if ! git diff --cached --quiet; then
  git commit -m "checkpoint: UniCoT Uni-MMMU $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

# Push even if there was no new commit, so a previous transient push failure can recover.
if git push "$REMOTE" "HEAD:$BRANCH"; then
  echo "[git-sync] $(date -u +%FT%TZ) pushed $BRANCH"
else
  echo "[git-sync] WARNING: push failed; local commit retained for next retry" >&2
  exit 0
fi
