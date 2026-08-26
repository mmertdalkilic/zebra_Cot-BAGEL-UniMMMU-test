#!/usr/bin/env bash
# Copy bagel-zebra-cot sampling into bagel-zebra-cot-tagfix and repair
# Uni-MMMU tags in model_text.txt (images hardlinked/copied unchanged).
#
#   bash run_tagfix.sh
#
# Molab:
#   tag_btn = mo.ui.run_button(label="TAG FIX (copy outputs, wrap missing tags)")
#   tag_btn
#   mo.stop(not tag_btn.value, mo.md("Writes bagel-zebra-cot-tagfix/ only. Does not change original sampling or _eval/bagel-zebra-cot."))
#   _r = subprocess.run(["bash", f"{SCRIPTS}/outputs_git.sh", "restore"], text=True)
#   print("restore exit:", _r.returncode)
#   _r = subprocess.run(["bash", f"{SCRIPTS}/run_tagfix.sh"], text=True)
#   print("tagfix exit code:", _r.returncode)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"
UMMMU_ROOT="${UMMMU_ROOT:-$WORK_DIR/Uni-MMMU}"
SRC_MODEL="${SRC_MODEL:-bagel-zebra-cot}"
DST_MODEL="${DST_MODEL:-bagel-zebra-cot-tagfix}"
SKIP_RESTORE="${SKIP_RESTORE:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

cd "$UMMMU_ROOT"

SYNC_ON=0
if [[ -n "${GITHUB_TOKEN:-}" && -n "${OUTPUTS_REPO:-}" ]]; then
  SYNC_ON=1
  echo "== GitHub output sync enabled ($OUTPUTS_REPO) =="
  if [[ "$SKIP_RESTORE" != "1" ]]; then
    bash "$SCRIPT_DIR/outputs_git.sh" restore
  else
    echo "== SKIP_RESTORE=1 — not checking out origin/outputs =="
  fi
  export TAGFIX_SYNC_CMD="bash '$SCRIPT_DIR/outputs_git.sh' push-tagfix"
  export TAGFIX_SYNC_INTERVAL_MIN="${SYNC_INTERVAL_MIN:-15}"
else
  echo "== GitHub output sync DISABLED =="
fi

SRC="$UMMMU_ROOT/outputs/$SRC_MODEL"
if [[ ! -d "$SRC" ]]; then
  echo "ERROR: missing $SRC — restore outputs first" >&2
  exit 1
fi

LOG="$WORK_DIR/tagfix_$(date +%Y%m%d_%H%M%S).log"
echo "Tag-fix copy $SRC_MODEL -> $DST_MODEL ; log $LOG"

python -u "$SCRIPT_DIR/tag_fix.py" \
  --ummmu-root "$UMMMU_ROOT" \
  --src-model "$SRC_MODEL" \
  --dst-model "$DST_MODEL" \
  2>&1 | tee "$LOG"

if [[ "$SYNC_ON" = "1" && "$SKIP_PUSH" != "1" ]]; then
  bash "$SCRIPT_DIR/outputs_git.sh" push-tagfix
fi

echo ""
echo "Tag-fixed sampling: $UMMMU_ROOT/outputs/$DST_MODEL"
echo "Original $SRC_MODEL was not modified."
