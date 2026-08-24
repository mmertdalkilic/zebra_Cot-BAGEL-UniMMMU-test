#!/usr/bin/env bash
# Run the official Uni-MMMU evaluation (eval_ummmu.py) on the Bagel-Zebra-CoT outputs.
#
# IMPORTANT: the official judges are Qwen2.5-VL-72B-Instruct + Qwen3-32B in bf16,
# which need ~210 GB VRAM combined and do NOT fit on a single 96 GB GPU.
# On molab we substitute the official AWQ-quantized variants
# (~41 GB + ~19 GB, both fit together in 96 GB). Scores may differ slightly
# from a full-precision judge; note this when comparing against the paper.
#
# Resumable: per-item judgements are written under
#   $UMMMU_ROOT/outputs/_eval/<model>/
# which is on the GitHub `outputs` branch (same as sampling). Re-running skips
# finished items. Use EVAL_TIME_BUDGET_HOURS to stop cleanly before molab's 12h.
#
#   bash run_eval.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"
UMMMU_ROOT="${UMMMU_ROOT:-$WORK_DIR/Uni-MMMU}"
MODEL_NAME="${MODEL_NAME:-bagel-zebra-cot}"

# Set USE_AWQ_JUDGES=0 on a machine with >=210GB VRAM (e.g. 3x A100/H100)
# to use the official full-precision judges.
USE_AWQ_JUDGES="${USE_AWQ_JUDGES:-1}"

cd "$UMMMU_ROOT"

# Evaluation dependencies (judges, metrics).
# BAGEL's sampling env pins transformers==4.49 (no Qwen3). AutoAWQ's last
# tested pair is transformers 4.51.3 — newer 4.57+ removed PytorchGELUTanh
# and AutoAWQ crashes on import. Pin exactly that version.
pip install "transformers==4.51.3"
pip install qwen-vl-utils dreamsim cairosvg pandas xlsxwriter openpyxl autoawq

# Restore sampling outputs AND any previous eval items from GitHub.
if [[ -n "${GITHUB_TOKEN:-}" && -n "${OUTPUTS_REPO:-}" ]]; then
  echo "== GitHub output sync enabled ($OUTPUTS_REPO) =="
  bash "$SCRIPT_DIR/outputs_git.sh" restore
  export EVAL_SYNC_CMD="bash '$SCRIPT_DIR/outputs_git.sh' push"
  export EVAL_SYNC_INTERVAL_MIN="${SYNC_INTERVAL_MIN:-15}"
else
  echo "== GitHub output sync DISABLED (set GITHUB_TOKEN + OUTPUTS_REPO to enable) =="
fi

# If an older non-resumable run wrote to eval/<model>, copy it into the synced
# tree so those judgements can be exploded into per-item resume files.
OFFICIAL_EVAL="$UMMMU_ROOT/eval/$MODEL_NAME"
SYNCED_EVAL="$UMMMU_ROOT/outputs/_eval/$MODEL_NAME"
if [ -d "$OFFICIAL_EVAL" ]; then
  mkdir -p "$SYNCED_EVAL"
  cp -an "$OFFICIAL_EVAL/." "$SYNCED_EVAL/" 2>/dev/null || true
  echo "== Migrated existing eval/ -> outputs/_eval/ =="
fi

export EVAL_TIME_BUDGET_HOURS="${EVAL_TIME_BUDGET_HOURS:-10.5}"

# eval_ummmu.py hardcodes base_path as a placeholder and the judge model names
# as dataclass defaults; patch a copy instead of editing the original.
cp eval_ummmu.py eval_ummmu_patched.py
sed -i "s#\"path-to-UMMMU-home\"#\"$UMMMU_ROOT\"#" eval_ummmu_patched.py
if [ "$USE_AWQ_JUDGES" = "1" ]; then
  sed -i 's#"Qwen/Qwen3-32B"#"Qwen/Qwen3-32B-AWQ"#' eval_ummmu_patched.py
  sed -i 's#"Qwen/Qwen2.5-VL-72B-Instruct"#"Qwen/Qwen2.5-VL-72B-Instruct-AWQ"#' eval_ummmu_patched.py
  echo "Patched eval_ummmu_patched.py to use AWQ judge models."
fi
python "$SCRIPT_DIR/apply_eval_resume.py" eval_ummmu_patched.py
python "$SCRIPT_DIR/apply_eval_sequential.py" eval_ummmu_patched.py

LOG="$WORK_DIR/eval_$(date +%Y%m%d_%H%M%S).log"
echo "Evaluating outputs/$MODEL_NAME ; logging to $LOG"
echo "Eval artifacts -> $SYNCED_EVAL  (time budget ${EVAL_TIME_BUDGET_HOURS}h)"

python -u eval_ummmu_patched.py --model_name "$MODEL_NAME" 2>&1 | tee "$LOG"

if [[ -n "${EVAL_SYNC_CMD:-}" ]]; then
  bash "$SCRIPT_DIR/outputs_git.sh" push || true
fi

echo ""
echo "Results: $SYNCED_EVAL/all_tasks_summary_${MODEL_NAME}.xlsx"
echo "(also copied under GitHub branch 'outputs' as _eval/${MODEL_NAME}/)"
