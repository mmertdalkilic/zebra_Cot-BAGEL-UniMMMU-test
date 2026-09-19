#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "[ERROR] Run this inside the cloned target repository." >&2
  exit 1
fi
cd "$ROOT"

BRANCH="${UNICOT_BRANCH:-unicot-eval-molab}"
REMOTE="${GIT_REMOTE:-origin}"

echo "[branch] repository=$ROOT"
echo "[branch] target=$BRANCH"

git fetch "$REMOTE" --prune

if git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
  echo "[branch] remote branch already exists; switching to it"
  git switch -C "$BRANCH" "$REMOTE/$BRANCH"
else
  echo "[branch] creating $BRANCH from $REMOTE/main"
  git switch -C "$BRANCH" "$REMOTE/main"
fi

git lfs install
git lfs track \
  "runs/unicot_native_v02/**/*.png" \
  "runs/unicot_native_v02/**/*.jpg" \
  "runs/unicot_native_v02/**/*.jpeg" \
  "runs/unicot_native_v02/**/*.webp" >/dev/null

touch .gitignore
if ! grep -q '^# UniCoT MoLab local-only$' .gitignore; then
  cat >> .gitignore <<'EOF'

# UniCoT MoLab local-only
unicot_molab/config.env
*.part
__pycache__/
EOF
fi

mkdir -p runs/unicot_native_v02 logs

git config user.name "${GIT_USER_NAME:-UniCoT MoLab}"
git config user.email "${GIT_USER_EMAIL:-unicot-molab@local}"

git add .gitattributes .gitignore unicot_molab
if ! git diff --cached --quiet; then
  git commit -m "add native UniCoT v0.2 Uni-MMMU MoLab runner"
fi

echo "[branch] pushing $BRANCH"
git push -u "$REMOTE" "HEAD:$BRANCH"

echo
echo "[OK] branch is ready:"
git branch --show-current
git log -1 --oneline
