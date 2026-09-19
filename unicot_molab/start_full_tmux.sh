#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
SESSION="${1:-unicot_eval}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux] session '$SESSION' already exists"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash '$HERE/run_full_all6.sh'; rc=\$?; echo; echo '[tmux] run exited rc='\$rc; exec bash"

echo "[tmux] started '$SESSION'"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b then d"
