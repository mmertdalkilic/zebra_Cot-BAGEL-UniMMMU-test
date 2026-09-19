#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
INTERVAL="${1:-${GIT_SYNC_INTERVAL:-900}}"

echo "[git-sync] watcher started; interval=${INTERVAL}s"
while true; do
  sleep "$INTERVAL"
  bash "$HERE/checkpoint_once.sh" || true
done
