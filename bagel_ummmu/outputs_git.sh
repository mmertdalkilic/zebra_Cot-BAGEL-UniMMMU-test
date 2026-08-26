#!/usr/bin/env bash
# Persist Uni-MMMU sampling outputs to a GitHub branch, so that molab session
# restarts (which wipe .git dirs, images and other binaries) lose nothing.
#
# The outputs directory ($WORK_DIR/Uni-MMMU/outputs) is made a standalone git
# repo whose 'outputs' branch lives in your GitHub repo, separate from main.
#
# Usage:
#   bash outputs_git.sh restore          # session start: pull the outputs branch back
#   bash outputs_git.sh push             # commit current outputs and push (periodic; may force-push)
#   bash outputs_git.sh push-rulebased   # leftover eval only; never force-push, never add science/code
#   bash outputs_git.sh push-tagfix      # bagel-zebra-cot-tagfix trees only; never force-push
#
# Required environment:
#   GITHUB_TOKEN   PAT with write access to the repo (never commit this!)
#   OUTPUTS_REPO   owner/repo, e.g. MrTractorWheel/zebra_Cot-BAGEL-UniMMMU-test
# Optional:
#   OUTPUTS_BRANCH (default: outputs)
#   WORK_DIR       (default: $HOME/bagel_ummmu)
set -euo pipefail

CMD="${1:?usage: outputs_git.sh restore|push|push-rulebased|push-tagfix}"
WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/Uni-MMMU/outputs}"
BRANCH="${OUTPUTS_BRANCH:-outputs}"

: "${GITHUB_TOKEN:?set GITHUB_TOKEN (a PAT with write access to the repo)}"
: "${OUTPUTS_REPO:?set OUTPUTS_REPO, e.g. youruser/zebra_Cot-BAGEL-UniMMMU-test}"
URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${OUTPUTS_REPO}.git"

g() { git -C "$OUT_DIR" "$@"; }

ensure_repo() {
  mkdir -p "$OUT_DIR"
  if [ ! -d "$OUT_DIR/.git" ]; then
    g init --initial-branch "$BRANCH" >/dev/null 2>&1 \
      || { g init >/dev/null; g checkout -b "$BRANCH" >/dev/null 2>&1 || true; }
  fi
  g remote set-url origin "$URL" 2>/dev/null || g remote add origin "$URL"
  g config user.email "molab-runner@localhost"
  g config user.name "molab runner"
}

restore() {
  ensure_repo
  if g fetch origin "$BRANCH" 2>/dev/null; then
    # Remote is the source of truth at session start (local state after a
    # restart is a partial skeleton). -f overwrites conflicting local files.
    g checkout -f -B "$BRANCH" FETCH_HEAD
    echo "[outputs_git] restored outputs from origin/$BRANCH ($(g rev-parse --short HEAD))"
  else
    echo "[outputs_git] no '$BRANCH' branch on the remote yet - starting fresh"
  fi
}

push() {
  ensure_repo
  g add -A
  # 'diff --cached --quiet' exits non-zero both when there are staged changes
  # and on an unborn HEAD - in either case we want to commit.
  if g diff --cached --quiet 2>/dev/null; then
    echo "[outputs_git] nothing new to push"
    return 0
  fi
  g commit -q -m "outputs sync $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if g push -q -u origin "$BRANCH" 2>/dev/null; then
    echo "[outputs_git] pushed $(g rev-parse --short HEAD) to origin/$BRANCH"
  else
    # The branch holds generated artifacts only; local (freshly restored +
    # extended) content is authoritative, so force-push on divergence.
    echo "[outputs_git] normal push rejected - force-pushing"
    g push -q -u --force origin "$BRANCH"
    echo "[outputs_git] force-pushed $(g rev-parse --short HEAD) to origin/$BRANCH"
  fi
}

# Commit only leftover rule-based eval paths. Never `git add -A`, never --force.
# Keeps already-pushed VL science/code/math judgments intact on origin/outputs.
push_rulebased() {
  MODEL_NAME="${MODEL_NAME:-bagel-zebra-cot}"
  ensure_repo
  prefix="_eval/${MODEL_NAME}"
  add_if() {
    local p="$1"
    if [ -e "$OUT_DIR/$p" ]; then
      g add -- "$p"
    fi
  }
  add_if "${prefix}/jigsaw"
  add_if "${prefix}/maze"
  add_if "${prefix}/sliding"
  add_if "${prefix}/math/eval_summary.json"
  add_if "${prefix}/math/eval_details.json"
  add_if "${prefix}/math/items"
  add_if "${prefix}/all_tasks_summary_${MODEL_NAME}.xlsx"
  add_if "${prefix}/ummmu_table2_${MODEL_NAME}.xlsx"
  add_if "${prefix}/ummmu_table2_${MODEL_NAME}.csv"
  add_if "${prefix}/ummmu_table2_${MODEL_NAME}.md"

  if g diff --cached --quiet 2>/dev/null; then
    echo "[outputs_git] push-rulebased: nothing new to push"
    return 0
  fi

  # Refuse if the index also contains science/code (should be impossible with
  # the allowlist above; belt-and-suspenders against a bad add).
  if g diff --cached --name-only | grep -E "/(science|code)/" >/dev/null; then
    echo "[outputs_git] push-rulebased ABORT: staged files under science/ or code/" >&2
    g reset -q
    return 1
  fi

  g commit -q -m "rule-based leftover eval $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if g push -q -u origin "$BRANCH" 2>/dev/null; then
    echo "[outputs_git] pushed $(g rev-parse --short HEAD) to origin/$BRANCH (allowlist only, no force)"
  else
    echo "[outputs_git] allowlist push rejected. Refusing to force-push (would risk VL judge artifacts)." >&2
    echo "[outputs_git] Inspect: git -C $OUT_DIR status && git -C $OUT_DIR log --oneline -5" >&2
    return 1
  fi
}

# Commit only tag-fixed sampling + tag-fixed eval. Never `git add -A`, never --force.
# Original bagel-zebra-cot/ and _eval/bagel-zebra-cot/ stay exactly as restored.
push_tagfix() {
  DST_MODEL="${DST_MODEL:-bagel-zebra-cot-tagfix}"
  ensure_repo
  add_if() {
    local p="$1"
    if [ -e "$OUT_DIR/$p" ]; then
      g add -- "$p"
    fi
  }
  add_if "${DST_MODEL}"
  add_if "_eval/${DST_MODEL}"

  if g diff --cached --quiet 2>/dev/null; then
    echo "[outputs_git] push-tagfix: nothing new to push"
    return 0
  fi

  # Allow only bagel-zebra-cot-tagfix/ and _eval/bagel-zebra-cot-tagfix/.
  # Trailing slash so bagel-zebra-cot/ (original) cannot match *-tagfix.
  leak=$(g diff --cached --name-only | grep -vE "^(${DST_MODEL}/|_eval/${DST_MODEL}/)" || true)
  if [ -n "$leak" ]; then
    echo "[outputs_git] push-tagfix ABORT: staged paths outside ${DST_MODEL}:" >&2
    echo "$leak" >&2
    g reset -q
    return 1
  fi
  if g diff --cached --name-only | grep -E "^(bagel-zebra-cot/|_eval/bagel-zebra-cot/)" >/dev/null; then
    echo "[outputs_git] push-tagfix ABORT: original bagel-zebra-cot tree is staged" >&2
    g reset -q
    return 1
  fi

  g commit -q -m "tagfix outputs ${DST_MODEL} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if g push -q -u origin "$BRANCH" 2>/dev/null; then
    echo "[outputs_git] pushed $(g rev-parse --short HEAD) to origin/$BRANCH (tagfix allowlist, no force)"
  else
    echo "[outputs_git] tagfix push rejected. Refusing to force-push (would risk original VL artifacts)." >&2
    echo "[outputs_git] Inspect: git -C $OUT_DIR status && git -C $OUT_DIR log --oneline -5" >&2
    return 1
  fi
}

case "$CMD" in
  restore) restore ;;
  push)    push ;;
  push-rulebased) push_rulebased ;;
  push-tagfix) push_tagfix ;;
  *) echo "usage: outputs_git.sh restore|push|push-rulebased|push-tagfix" >&2; exit 2 ;;
esac
