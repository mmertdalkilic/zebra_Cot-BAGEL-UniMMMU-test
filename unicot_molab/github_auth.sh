#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "[ERROR] Run this inside the cloned GitHub repository." >&2
  exit 1
fi
cd "$ROOT"

REMOTE="${GIT_REMOTE:-origin}"
URL="$(git remote get-url "$REMOTE")"
echo "[auth] remote=$REMOTE"
echo "[auth] url=$URL"

if [[ "$URL" == git@github.com:* || "$URL" == ssh://git@github.com/* ]]; then
  echo "[auth] SSH remote detected."
  ssh -o BatchMode=yes -T git@github.com 2>&1 || true
  echo "[auth] SSH configured; prepare_branch.sh will verify push access."
  exit 0
fi

if [[ "$URL" != https://github.com/* ]]; then
  echo "[ERROR] Expected GitHub HTTPS or SSH remote; got: $URL" >&2
  exit 1
fi

git config --global credential.helper store

GH_USER="${GITHUB_USER:-mmertdalkilic}"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  TOKEN="$GITHUB_TOKEN"
  echo "[auth] using GITHUB_TOKEN from environment (not writing it to the repo)"
else
  read -r -p "GitHub username [$GH_USER]: " entered
  GH_USER="${entered:-$GH_USER}"
  read -r -s -p "GitHub PAT (repo Contents: Read and write): " TOKEN
  echo
fi

if [[ -z "${TOKEN:-}" ]]; then
  echo "[ERROR] Empty GitHub token." >&2
  exit 1
fi

printf 'protocol=https\nhost=github.com\nusername=%s\npassword=%s\n\n' \
  "$GH_USER" "$TOKEN" | git credential approve

unset TOKEN
echo "[auth] credential stored in this MoLab environment only."
echo "[auth] prepare_branch.sh will perform the real push test."
