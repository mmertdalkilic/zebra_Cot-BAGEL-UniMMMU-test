#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
if [[ -f "$HERE/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$HERE/config.env"
fi

SHARED="${SHARED_ROOT:-/workspace/unicot_unimmmu_shared}"
VENV="$SHARED/venv"
UNICOT="$SHARED/repos/UniCoT"
UMMMU="$SHARED/repos/Uni-MMMU"
MODEL="$SHARED/models/UniCoT-7B-MoT-v0.2"
DATA_SNAPSHOT="$SHARED/downloads/Uni-MMMU-Eval"
DATA_EXTRACT="$SHARED/dataset"
OUTPUT_NAME="${OUTPUT_NAME:-unicot_native_v02}"

mkdir -p "$SHARED"/{repos,models,downloads,dataset,hf_cache,setup_markers,outputs}

echo "[setup] repo=$ROOT"
echo "[setup] shared=$SHARED"

if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq git git-lfs rsync tmux build-essential ninja-build || true
  else
    apt-get update -qq || true
    apt-get install -y -qq git git-lfs rsync tmux build-essential ninja-build || true
  fi
fi

for cmd in git git-lfs; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "[ERROR] missing command: $cmd" >&2
    exit 1
  }
done
git lfs install

python3 -m pip install -q -U uv huggingface_hub hf_transfer
export HF_HOME="${HF_HOME:-$SHARED/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1

clone_or_update() {
  local url="$1" dst="$2"
  if [[ -d "$dst/.git" ]]; then
    echo "[setup] updating $dst"
    git -C "$dst" fetch --all --prune
    branch="$(git -C "$dst" symbolic-ref --short HEAD 2>/dev/null || true)"
    if [[ -n "$branch" ]]; then
      git -C "$dst" pull --ff-only || true
    fi
  else
    echo "[setup] cloning $url"
    git clone "$url" "$dst"
  fi
}

clone_or_update https://github.com/Fr0zenCrane/UniCoT.git "$UNICOT"
clone_or_update https://github.com/Vchitect/Uni-MMMU.git "$UMMMU"

# Blackwell requires a newer torch/CUDA stack than UniCoT's historical torch 2.5.1 pin.
ENV_MARKER="$SHARED/setup_markers/unicot-py310-torch280-cu128-fa283post1.ok"
if [[ ! -x "$VENV/bin/python" || ! -f "$ENV_MARKER" ]]; then
  echo "[setup] creating Python 3.10 UniCoT environment"
  rm -rf "$VENV"
  uv venv --python 3.10 --seed "$VENV"
  "$VENV/bin/python" -m pip install -U "pip<26" "setuptools<82" wheel packaging ninja

  "$VENV/bin/python" -m pip install \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

  tmpreq="$(mktemp)"
  grep -Ev '^[[:space:]]*(torch==|torchvision==|triton([[:space:];=]|$)|flash_attn|flash-attn|#.*flash_attn)' \
    "$UNICOT/requirements.txt" > "$tmpreq"
  "$VENV/bin/python" -m pip install -r "$tmpreq"
  rm -f "$tmpreq"

  "$VENV/bin/python" -m pip install \
    huggingface_hub hf_transfer gradio_client tqdm pillow accelerate safetensors

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc is required for an SM120 FlashAttention build." >&2
    echo "        Start a MoLab image/runtime with CUDA 12.8+ developer toolkit and rerun setup." >&2
    exit 1
  fi

  echo "[setup] nvcc: $(nvcc --version | tail -n 1)"
  echo "[setup] building FlashAttention 2.8.3.post1 for Blackwell SM120"
  env \
    FLASH_ATTN_CUDA_ARCHS=120 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    MAX_JOBS="${MAX_JOBS:-4}" \
    NVCC_THREADS="${NVCC_THREADS:-2}" \
    "$VENV/bin/python" -m pip install -v \
      --no-build-isolation --no-deps --no-cache-dir --no-binary=flash-attn \
      "flash-attn==2.8.3.post1"

  touch "$ENV_MARKER"
else
  echo "[setup] reusing existing venv"
fi

"$VENV/bin/python" - <<'PY'
import torch, flash_attn
print("torch:", torch.__version__)
print("cuda :", torch.version.cuda)
print("gpu  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
print("cap  :", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("arch :", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
print("flash_attn:", flash_attn.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.get_device_capability(0) != (12, 0):
    print("[WARN] Expected RTX PRO 6000 Blackwell / SM120 for this MoLab setup")
if "sm_120" not in torch.cuda.get_arch_list():
    raise SystemExit("PyTorch build does not include sm_120")
PY

echo "[setup] downloading UniCoT-v0.2 checkpoint and Uni-MMMU dataset"
"$VENV/bin/python" - "$MODEL" "$DATA_SNAPSHOT" <<'PY'
import os, sys
from huggingface_hub import snapshot_download

model_dir, data_dir = sys.argv[1], sys.argv[2]
token = os.environ.get("HF_TOKEN") or None

snapshot_download(
    repo_id="Fr0zencr4nE/UniCoT-7B-MoT-v0.2",
    local_dir=model_dir,
    token=token,
)
snapshot_download(
    repo_id="Vchitect/Uni-MMMU-Eval",
    repo_type="dataset",
    local_dir=data_dir,
    allow_patterns=["data.tar"],
    token=token,
)
PY

DATA_TAR="$DATA_SNAPSHOT/data.tar"
if [[ ! -f "$DATA_TAR" ]]; then
  echo "[ERROR] missing $DATA_TAR" >&2
  exit 1
fi

if [[ ! -f "$DATA_EXTRACT/.unimmmu_extracted" ]]; then
  echo "[setup] extracting Uni-MMMU data"
  rm -rf "$DATA_EXTRACT/data"
  tar -xf "$DATA_TAR" -C "$DATA_EXTRACT"
  touch "$DATA_EXTRACT/.unimmmu_extracted"
fi

if [[ ! -d "$DATA_EXTRACT/data" ]]; then
  echo "[ERROR] expected $DATA_EXTRACT/data after extraction" >&2
  exit 1
fi

rm -rf "$UMMMU/data"
ln -s "$DATA_EXTRACT/data" "$UMMMU/data"

# Run outputs live inside the Git repo while the native runner sees them through SHARED/outputs.
mkdir -p "$ROOT/runs/$OUTPUT_NAME" "$ROOT/logs" "$SHARED/outputs"
rm -rf "$SHARED/outputs/$OUTPUT_NAME"
ln -s "$ROOT/runs/$OUTPUT_NAME" "$SHARED/outputs/$OUTPUT_NAME"

required=(
  "$UNICOT/inference_unicot_v0.2.py"
  "$MODEL/llm_config.json"
  "$MODEL/vit_config.json"
  "$MODEL/ae.safetensors"
  "$MODEL/ema.safetensors"
  "$UMMMU/data/math_data/filtered.json"
  "$UMMMU/data/jigsaw_dataset_2x2ref/metadata.json"
  "$UMMMU/data/science/dim_all.json"
  "$UMMMU/data/svg/metadata.json"
  "$UMMMU/data/sliding/summary_steps_le_8.json"
)
for p in "${required[@]}"; do
  [[ -e "$p" ]] || { echo "[ERROR] missing expected path: $p" >&2; exit 1; }
done

{
  echo "created=$(date -Is)"
  echo "target_repo_branch=$(git -C "$ROOT" branch --show-current)"
  echo "unicot_commit=$(git -C "$UNICOT" rev-parse HEAD)"
  echo "unimmmu_commit=$(git -C "$UMMMU" rev-parse HEAD)"
  echo "checkpoint=Fr0zencr4nE/UniCoT-7B-MoT-v0.2"
  echo "torch=$("$VENV/bin/python" -c 'import torch; print(torch.__version__)')"
  echo "flash_attn=$("$VENV/bin/python" -c 'import flash_attn; print(flash_attn.__version__)')"
} > "$ROOT/runs/$OUTPUT_NAME/PROVENANCE.txt"

echo
echo "[OK] setup complete"
echo "[OK] venv=$VENV"
echo "[OK] model=$MODEL"
echo "[OK] Uni-MMMU=$UMMMU"
echo "[OK] outputs=$ROOT/runs/$OUTPUT_NAME"
