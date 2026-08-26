#!/usr/bin/env bash
# Re-score tag-fixed texts only. Copies image/overlay metrics from
# _eval/bagel-zebra-cot. Writes _eval/bagel-zebra-cot-tagfix.
# Loads Qwen2.5-VL for science + SVG TEXT judges only (not image judges).
#
#   bash run_eval_tagfix.sh
#
# Molab:
#   eval_tag_btn = mo.ui.run_button(label="EVAL TAGFIX TEXT (keep original images)")
#   eval_tag_btn
#   mo.stop(not eval_tag_btn.value, mo.md("Run after TAG FIX. Does not rewrite _eval/bagel-zebra-cot. Loads VL for science/SVG text only."))
#   _r = subprocess.run(["bash", f"{SCRIPTS}/outputs_git.sh", "restore"], text=True)
#   print("restore exit:", _r.returncode)
#   _r = subprocess.run(["bash", f"{SCRIPTS}/run_eval_tagfix.sh"], text=True)
#   print("eval tagfix exit code:", _r.returncode)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"
UMMMU_ROOT="${UMMMU_ROOT:-$WORK_DIR/Uni-MMMU}"
SRC_MODEL="${SRC_MODEL:-bagel-zebra-cot}"
DST_MODEL="${DST_MODEL:-bagel-zebra-cot-tagfix}"
SKIP_RESTORE="${SKIP_RESTORE:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"
USE_AWQ_JUDGES="${USE_AWQ_JUDGES:-1}"

cd "$UMMMU_ROOT"

pip install "transformers==4.51.3"
pip install qwen-vl-utils cairosvg pandas xlsxwriter openpyxl tqdm autoawq

SYNC_ON=0
if [[ -n "${GITHUB_TOKEN:-}" && -n "${OUTPUTS_REPO:-}" ]]; then
  SYNC_ON=1
  echo "== GitHub output sync enabled ($OUTPUTS_REPO) =="
  if [[ "$SKIP_RESTORE" != "1" ]]; then
    bash "$SCRIPT_DIR/outputs_git.sh" restore
  else
    echo "== SKIP_RESTORE=1 — not checking out origin/outputs =="
  fi
  export EVAL_SYNC_CMD="bash '$SCRIPT_DIR/outputs_git.sh' push-tagfix"
  export EVAL_SYNC_INTERVAL_MIN="${SYNC_INTERVAL_MIN:-15}"
else
  echo "== GitHub output sync DISABLED =="
fi

if [[ ! -d "$UMMMU_ROOT/outputs/$DST_MODEL" ]]; then
  echo "ERROR: missing $UMMMU_ROOT/outputs/$DST_MODEL — run run_tagfix.sh first" >&2
  exit 1
fi
if [[ ! -d "$UMMMU_ROOT/outputs/_eval/$SRC_MODEL" ]]; then
  echo "ERROR: missing original eval $UMMMU_ROOT/outputs/_eval/$SRC_MODEL" >&2
  exit 1
fi

cp eval_ummmu.py eval_ummmu_patched.py
sed -i "s#\"path-to-UMMMU-home\"#\"$UMMMU_ROOT\"#" eval_ummmu_patched.py
if [ "$USE_AWQ_JUDGES" = "1" ]; then
  sed -i 's#"Qwen/Qwen3-32B"#"Qwen/Qwen3-32B-AWQ"#' eval_ummmu_patched.py
  sed -i 's#"Qwen/Qwen2.5-VL-72B-Instruct"#"Qwen/Qwen2.5-VL-72B-Instruct-AWQ"#' eval_ummmu_patched.py
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export EVAL_TIME_BUDGET_HOURS="${EVAL_TIME_BUDGET_HOURS:-10.5}"
python "$SCRIPT_DIR/apply_eval_resume.py" eval_ummmu_patched.py
python "$SCRIPT_DIR/apply_eval_sequential.py" eval_ummmu_patched.py

LOG="$WORK_DIR/eval_tagfix_$(date +%Y%m%d_%H%M%S).log"
echo "Tagfix text eval $DST_MODEL ; log $LOG"
echo "Eval artifacts -> $UMMMU_ROOT/outputs/_eval/$DST_MODEL  (time budget ${EVAL_TIME_BUDGET_HOURS}h)"
echo "Will NOT rewrite _eval/$SRC_MODEL. Images/overlay copied from there."

python -u "$SCRIPT_DIR/eval_tagfix_text.py" \
  --ummmu-root "$UMMMU_ROOT" \
  --src-model "$SRC_MODEL" \
  --dst-model "$DST_MODEL" \
  --patched-eval "$UMMMU_ROOT/eval_ummmu_patched.py" \
  2>&1 | tee "$LOG"

if [[ "$SYNC_ON" = "1" && "$SKIP_PUSH" != "1" ]]; then
  bash "$SCRIPT_DIR/outputs_git.sh" push-tagfix
fi

echo ""
echo "Paper table: $UMMMU_ROOT/outputs/_eval/${DST_MODEL}/ummmu_table2_${DST_MODEL}.xlsx"
echo "Original _eval/${SRC_MODEL} was not modified."
