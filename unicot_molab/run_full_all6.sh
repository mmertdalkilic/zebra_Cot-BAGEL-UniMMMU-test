#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "[full] all six tasks; resume markers from smoke/previous runs are preserved"
TASKS=all MAX_SAMPLES=0 bash "$HERE/run_tasks.sh"
