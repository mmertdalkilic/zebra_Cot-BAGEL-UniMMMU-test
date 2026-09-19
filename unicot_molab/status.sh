#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$ROOT/runs/${OUTPUT_NAME:-unicot_native_v02}"

echo "branch: $(git -C "$ROOT" branch --show-current)"
echo "latest commit: $(git -C "$ROOT" log -1 --oneline)"
echo

for task in science math jigsaw maze sliding code; do
  d="$OUT/$task"
  if [[ -d "$d" ]]; then
    done_count="$(find "$d" -type f -name _done.ok 2>/dev/null | wc -l | tr -d ' ')"
    result_count="$(find "$d" -type f -name result.json 2>/dev/null | wc -l | tr -d ' ')"
    echo "$(printf '%-8s' "$task") done=$done_count result_json=$result_count"
  else
    echo "$(printf '%-8s' "$task") not started"
  fi
done

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo
echo "=== last generation log lines ==="
tail -25 "$ROOT/logs/unicot_native_v02.log" 2>/dev/null || true

echo
echo "=== last Git sync lines ==="
tail -12 "$ROOT/logs/unicot_native_v02_git_sync.log" 2>/dev/null || true
