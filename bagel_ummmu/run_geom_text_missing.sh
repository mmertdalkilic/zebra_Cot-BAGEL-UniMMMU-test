#!/usr/bin/env bash
# Score only geometry items that already have overlay_ok but lack text_ok.
# Loads Qwen3-32B-AWQ. Does NOT load Qwen2.5-VL-72B.
#
#   bash run_geom_text_missing.sh
#
# Molab:
#   eval_btn = mo.ui.run_button(label="EVALUATION (Qwen3 text, 26 geometry cases)")
#   eval_btn
#   mo.stop(not eval_btn.value, mo.md("Only scores geometry items that have overlay but no text_ok. Does not load Qwen2.5-VL-72B. Do not use run_eval.sh."))
#   _r = subprocess.run(["bash", f"{SCRIPTS}/outputs_git.sh", "restore"], text=True)
#   print("restore exit:", _r.returncode)
#   _r = subprocess.run(["bash", f"{SCRIPTS}/run_geom_text_missing.sh"], text=True)
#   print("eval exit code:", _r.returncode)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"
UMMMU_ROOT="${UMMMU_ROOT:-$WORK_DIR/Uni-MMMU}"
MODEL_NAME="${MODEL_NAME:-bagel-zebra-cot}"
SKIP_RESTORE="${SKIP_RESTORE:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"
USE_AWQ_JUDGES="${USE_AWQ_JUDGES:-1}"

cd "$UMMMU_ROOT"

# Qwen3-AWQ needs the same transformers pin as the previous eval session.
# Do not install a VL checkpoint; we never construct LocalVL.
pip install "transformers==4.51.3"
pip install pandas xlsxwriter openpyxl tqdm autoawq

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
  echo "== GitHub output sync DISABLED =="
fi

SYNCED_EVAL="$UMMMU_ROOT/outputs/_eval/$MODEL_NAME"
if [[ ! -f "$SYNCED_EVAL/science/eval_summary.json" || ! -f "$SYNCED_EVAL/code/eval_summary.json" ]]; then
  echo "ERROR: VL summaries missing under $SYNCED_EVAL" >&2
  exit 1
fi
if [[ ! -d "$SYNCED_EVAL/math/items" ]] || ! ls "$SYNCED_EVAL/math/items"/*.json >/dev/null 2>&1; then
  echo "ERROR: geometry per-item JSON missing under $SYNCED_EVAL/math/items" >&2
  exit 1
fi

cp eval_ummmu.py eval_ummmu_patched.py
sed -i "s#\"path-to-UMMMU-home\"#\"$UMMMU_ROOT\"#" eval_ummmu_patched.py
if [ "$USE_AWQ_JUDGES" = "1" ]; then
  sed -i 's#"Qwen/Qwen3-32B"#"Qwen/Qwen3-32B-AWQ"#' eval_ummmu_patched.py
  # Keep the VL string patched so an accidental import cannot pull the 72B bf16 id.
  sed -i 's#"Qwen/Qwen2.5-VL-72B-Instruct"#"Qwen/Qwen2.5-VL-72B-Instruct-AWQ"#' eval_ummmu_patched.py
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
python "$SCRIPT_DIR/apply_eval_resume.py" eval_ummmu_patched.py
python "$SCRIPT_DIR/apply_eval_sequential.py" eval_ummmu_patched.py

LOG="$WORK_DIR/eval_geom_text_$(date +%Y%m%d_%H%M%S).log"
echo "Geometry text-only eval; logging to $LOG"
echo "Will NOT load Qwen2.5-VL."

python -u "$SCRIPT_DIR/eval_geom_text_missing.py" \
  --ummmu-root "$UMMMU_ROOT" \
  --model-name "$MODEL_NAME" \
  --patched-eval "$UMMMU_ROOT/eval_ummmu_patched.py" \
  2>&1 | tee "$LOG"

if [[ "$SYNC_ON" = "1" && "$SKIP_PUSH" != "1" ]]; then
  bash "$SCRIPT_DIR/outputs_git.sh" push-rulebased
elif [[ "$SKIP_PUSH" = "1" ]]; then
  echo "== SKIP_PUSH=1 — not pushing =="
fi

echo ""
echo "Paper table: $SYNCED_EVAL/ummmu_table2_${MODEL_NAME}.xlsx"
echo "Science/code/overlay JSON were not re-run and were not force-pushed."
