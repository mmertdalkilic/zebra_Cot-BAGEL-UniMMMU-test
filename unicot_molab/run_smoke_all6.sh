#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "[smoke] exactly 1 example from each of all six Uni-MMMU tasks"
TASKS=all MAX_SAMPLES=1 bash "$HERE/run_tasks.sh"
