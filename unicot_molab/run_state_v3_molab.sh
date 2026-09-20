#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/zebra_Cot-BAGEL-UniMMMU-test"
SHARED="/workspace/unicot_unimmmu_shared"

PY="$SHARED/venv/bin/python"

DRIVER="$ROOT/unicot_molab/run_unicot_native_v02_unimmmu.py"

CHECKPOINT="$ROOT/unicot_molab/checkpoint_state_v3.sh"

OUTPUT_NAME="${OUTPUT_NAME:-unicot_native_v02_state_v3}"

TASKS="${TASKS:-science,math_geo,jigsaw,maze,sliding,code_rendering}"

MAX_SAMPLES="${MAX_SAMPLES:-}"

MAX_REFLECTIONS="${MAX_REFLECTIONS:-20}"

MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-90GiB}"

GIT_SYNC_INTERVAL="${GIT_SYNC_INTERVAL:-900}"

RUN_TAG="${RUN_TAG:-manual}"

LOG="$ROOT/logs/${OUTPUT_NAME}.log"

SYNC_LOG="$ROOT/logs/${OUTPUT_NAME}_git_sync.log"

STATUS="$ROOT/runs/${OUTPUT_NAME}/RUN_STATUS_${RUN_TAG}.txt"

mkdir -p   "$ROOT/logs"   "$ROOT/runs/$OUTPUT_NAME"

touch "$LOG" "$SYNC_LOG"

rm -f "$STATUS"

sync_loop() {
    while true; do
        sleep "$GIT_SYNC_INTERVAL"

        echo           "[git-sync] $(date -u +%Y-%m-%dT%H:%M:%SZ)"           >> "$SYNC_LOG"

        "$CHECKPOINT" >> "$SYNC_LOG" 2>&1 ||           echo "[git-sync] checkpoint failed" >> "$SYNC_LOG"
    done
}

sync_loop &
SYNC_PID=$!

cleanup() {
    rc=$?

    trap - EXIT INT TERM

    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true

    {
        echo "run_tag=$RUN_TAG"
        echo "exit_code=$rc"
        echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$STATUS"

    if ! "$CHECKPOINT" >> "$SYNC_LOG" 2>&1; then
        echo           "[git-sync] final checkpoint failed"           >> "$SYNC_LOG"
    fi

    exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"

for TASK in "${TASK_ARRAY[@]}"; do

    echo | tee -a "$LOG"

    echo       "[run] starting task=$TASK"       | tee -a "$LOG"

    ARGS=(
      --root "$SHARED"
      --task "$TASK"
      --output-name "$OUTPUT_NAME"
      --max-reflections "$MAX_REFLECTIONS"
      --max-gpu-memory "$MAX_GPU_MEMORY"
    )

    if [[ -n "$MAX_SAMPLES" ]]; then
        ARGS+=(--max-samples "$MAX_SAMPLES")
    fi

    set +e

    UNIMMMU_TASK="$TASK"     PYTHONUNBUFFERED=1       "$PY" "$DRIVER" "${ARGS[@]}"       2>&1 | tee -a "$LOG"

    TASK_RC=${PIPESTATUS[0]}

    set -e

    if [[ "$TASK_RC" -ne 0 ]]; then

        echo           "[run] task=$TASK failed rc=$TASK_RC"           | tee -a "$LOG"

        exit "$TASK_RC"
    fi

    echo       "[run] completed task=$TASK"       | tee -a "$LOG"

done

echo   "[run] all requested tasks completed"   | tee -a "$LOG"
