#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
[[ -f "$HERE/config.env" ]] && source "$HERE/config.env"

SHARED="${SHARED_ROOT:-/workspace/unicot_unimmmu_shared}"
VENV="$SHARED/venv"
OUTPUT_NAME="${OUTPUT_NAME:-unicot_native_v02}"
TASKS="${TASKS:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_REFLECTIONS="${MAX_REFLECTIONS:-20}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-90GiB}"
INTERVAL="${GIT_SYNC_INTERVAL:-900}"
BRANCH="${UNICOT_BRANCH:-unicot-eval-molab}"

if [[ "$(git -C "$ROOT" branch --show-current)" != "$BRANCH" ]]; then
  echo "[ERROR] Must run on branch '$BRANCH'." >&2
  exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/runs/$OUTPUT_NAME"

LOG="$ROOT/logs/unicot_native_v02.log"
SYNC_LOG="$ROOT/logs/unicot_native_v02_git_sync.log"

# Start periodic GitHub checkpointing.
bash "$HERE/git_sync_loop.sh" "$INTERVAL" >>"$SYNC_LOG" 2>&1 &
SYNC_PID=$!

stop_sync_watcher() {
  pkill -TERM -P "$SYNC_PID" 2>/dev/null || true
  kill -TERM "$SYNC_PID" 2>/dev/null || true
  wait "$SYNC_PID" 2>/dev/null || true
}

cleanup() {
  code=$?
  trap - EXIT INT TERM
  stop_sync_watcher
  echo "[run] final GitHub checkpoint"
  bash "$HERE/checkpoint_once.sh" || true
  exit "$code"
}
trap cleanup EXIT INT TERM

echo "============================================================" | tee -a "$LOG"
echo "UniCoT-v0.2 native × Uni-MMMU" | tee -a "$LOG"
echo "tasks=$TASKS max_samples=$MAX_SAMPLES" | tee -a "$LOG"
echo "GitHub sync every ${INTERVAL}s" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$SHARED/hf_cache}"

"$VENV/bin/python" "$HERE/run_unicot_native_v02_unimmmu.py" \
  --root "$SHARED" \
  --tasks "$TASKS" \
  --output-name "$OUTPUT_NAME" \
  --max-samples "$MAX_SAMPLES" \
  --max-reflections "$MAX_REFLECTIONS" \
  --max-gpu-memory "$MAX_GPU_MEMORY" \
  2>&1 | tee -a "$LOG"

echo "[run] generation completed" | tee -a "$LOG"
