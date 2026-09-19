#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
[[ -f "$HERE/config.env" ]] && source "$HERE/config.env"
SHARED="${SHARED_ROOT:-/workspace/unicot_unimmmu_shared}"
VENV="$SHARED/venv"
OUT="${OUTPUT_NAME:-unicot_native_v02}"

echo "=== branch ==="
git -C "$ROOT" branch --show-current
git -C "$ROOT" status --short

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "=== environment ==="
"$VENV/bin/python" - <<'PY'
import torch, flash_attn
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("flash_attn:", flash_attn.__version__)
PY

echo
echo "=== required files ==="
for p in \
  "$SHARED/repos/UniCoT/inference_unicot_v0.2.py" \
  "$SHARED/models/UniCoT-7B-MoT-v0.2/ema.safetensors" \
  "$SHARED/models/UniCoT-7B-MoT-v0.2/ae.safetensors" \
  "$SHARED/repos/Uni-MMMU/data/math_data/filtered.json" \
  "$SHARED/repos/Uni-MMMU/data/jigsaw_dataset_2x2ref/metadata.json" \
  "$SHARED/repos/Uni-MMMU/data/science/dim_all.json" \
  "$SHARED/repos/Uni-MMMU/data/svg/metadata.json"; do
  [[ -e "$p" ]] && echo "[OK] $p" || { echo "[MISSING] $p"; exit 1; }
done

echo
echo "=== output link ==="
ls -ld "$SHARED/outputs/$OUT"
readlink -f "$SHARED/outputs/$OUT"

echo
echo "=== Git push target ==="
git -C "$ROOT" remote -v
echo "[OK] checks passed"
