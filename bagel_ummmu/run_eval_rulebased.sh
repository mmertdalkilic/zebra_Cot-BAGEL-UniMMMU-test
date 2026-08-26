#!/usr/bin/env bash
# Finish Uni-MMMU evaluation WITHOUT loading Qwen2.5-VL or Qwen3.
#
# Use this after the VL/Qwen3 judge session already wrote science, SVG/code,
# and geometry per-item JSON to the GitHub `outputs` branch. This launcher:
#   1. restores that branch (source of truth for VL artifacts)
#   2. flips geometry status overlay_done -> ok when text_ok exists, and
#      rewrites math/eval_summary.json from those item files
#   3. runs jigsaw (DreamSim) + maze + sliding only
#   4. writes all_tasks_summary_*.xlsx
#   5. pushes ONLY the allowlisted eval paths (never git add -A, never --force)
#
# Do NOT run run_eval.sh from the same session: that reloads the 72B VL judge
# and can rewrite science/code/math item JSON.
#
#   bash run_eval_rulebased.sh
#
# Molab cell (same shape as the full eval cell):
#
#   eval_btn = mo.ui.run_button(label="EVALUATION (final session, rule-based leftover)")
#   eval_btn
#   mo.stop(not eval_btn.value, mo.md("Run only after VL judge results are on the GitHub outputs branch. Do not use run_eval.sh."))
#   _r = subprocess.run(["bash", f"{SCRIPTS}/outputs_git.sh", "restore"], text=True)
#   print("restore exit:", _r.returncode)
#   _r = subprocess.run(["bash", f"{SCRIPTS}/run_eval_rulebased.sh"], text=True)
#   print("eval exit code:", _r.returncode)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"
UMMMU_ROOT="${UMMMU_ROOT:-$WORK_DIR/Uni-MMMU}"
MODEL_NAME="${MODEL_NAME:-bagel-zebra-cot}"
SKIP_RESTORE="${SKIP_RESTORE:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

cd "$UMMMU_ROOT"

# Light deps only. Do NOT install autoawq / upgrade transformers / pull 72B.
pip install dreamsim pandas xlsxwriter openpyxl tqdm

SYNC_ON=0
if [[ -n "${GITHUB_TOKEN:-}" && -n "${OUTPUTS_REPO:-}" ]]; then
  SYNC_ON=1
  echo "== GitHub output sync enabled ($OUTPUTS_REPO) =="
  if [[ "$SKIP_RESTORE" != "1" ]]; then
    bash "$SCRIPT_DIR/outputs_git.sh" restore
  else
    echo "== SKIP_RESTORE=1 — not checking out origin/outputs =="
  fi
else
  echo "== GitHub output sync DISABLED (set GITHUB_TOKEN + OUTPUTS_REPO to enable) =="
fi

SYNCED_EVAL="$UMMMU_ROOT/outputs/_eval/$MODEL_NAME"
if [[ ! -f "$SYNCED_EVAL/science/eval_summary.json" || ! -f "$SYNCED_EVAL/code/eval_summary.json" ]]; then
  echo "ERROR: VL summaries missing under $SYNCED_EVAL" >&2
  echo "Restore the GitHub outputs branch first. Refusing to run so we cannot overwrite VL judges." >&2
  exit 1
fi
if [[ ! -d "$SYNCED_EVAL/math/items" ]] || ! ls "$SYNCED_EVAL/math/items"/*.json >/dev/null 2>&1; then
  echo "ERROR: geometry per-item JSON missing under $SYNCED_EVAL/math/items" >&2
  exit 1
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export DREAMSIM_CACHE="${DREAMSIM_CACHE:-$WORK_DIR/.cache/dreamsim}"
mkdir -p "$DREAMSIM_CACHE"

LOG="$WORK_DIR/eval_rulebased_$(date +%Y%m%d_%H%M%S).log"
echo "Rule-based leftover eval; logging to $LOG"
echo "Eval artifacts -> $SYNCED_EVAL"
echo "Will NOT load Qwen2.5-VL or Qwen3."

python -u "$SCRIPT_DIR/finalize_eval.py" \
  --ummmu-root "$UMMMU_ROOT" \
  --model-name "$MODEL_NAME" \
  --work-dir "$WORK_DIR" \
  2>&1 | tee "$LOG"

if [[ "$SYNC_ON" = "1" && "$SKIP_PUSH" != "1" ]]; then
  bash "$SCRIPT_DIR/outputs_git.sh" push-rulebased
elif [[ "$SKIP_PUSH" = "1" ]]; then
  echo "== SKIP_PUSH=1 — not pushing =="
fi

echo ""
echo "Results: $SYNCED_EVAL/all_tasks_summary_${MODEL_NAME}.xlsx"
echo "VL science/code/math judgments were not re-run and were not force-pushed."
