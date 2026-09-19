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
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      git git-lfs rsync tmux build-essential ninja-build unzip || true
  else
    DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      git git-lfs rsync tmux build-essential ninja-build unzip || true
  fi
fi

for cmd in git git-lfs; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "[ERROR] missing command: $cmd" >&2
    exit 1
  }
done
git lfs install

# uv is only used to create the Python 3.10 venv. Do not upgrade the host
# marimo environment's huggingface_hub; UniCoT pins HF Hub <1.0.
python3 -m pip install -q -U uv
export HF_HOME="${HF_HOME:-$SHARED/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1

clone_or_update() {
  local url="$1" dst="$2"
  if [[ -d "$dst/.git" ]]; then
    echo "[setup] updating $dst"
    git -C "$dst" fetch --all --prune
    local branch
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

# MoLab's RTX PRO 6000 runtime has the CUDA driver but not nvcc. Use the
# official FlashAttention v2.8.3 prebuilt wheel for PyTorch 2.8 / CUDA 12 /
# Python 3.10 instead of compiling from source.
ENV_MARKER="$SHARED/setup_markers/unicot-py310-torch280-cu128-fa283wheel-v2.ok"

if [[ ! -x "$VENV/bin/python" || ! -f "$ENV_MARKER" ]]; then
  echo "[setup] creating Python 3.10 UniCoT environment"
  rm -rf "$VENV"
  rm -f "$SHARED/setup_markers"/unicot-py310-*.ok

  uv venv --python 3.10 --seed "$VENV"
  "$VENV/bin/python" -m pip install -U \
    "pip<26" "setuptools<82" wheel packaging ninja

  "$VENV/bin/python" -m pip install \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

  # Keep the inference-relevant upstream requirements. These four packages are
  # UI/training/reporting extras and cause expensive resolver backtracking but
  # are not imported by inference_unicot_v0.2.py.
  tmpreq="$(mktemp)"
  grep -Eiv '^[[:space:]]*(torch==|torchvision==|triton([[:space:];=]|$)|flash_attn|flash-attn|gradio([<=>[:space:]]|$)|wandb([<=>[:space:]]|$)|bitsandbytes([<=>[:space:]]|$)|xlsxwriter([<=>[:space:]]|$)|#.*flash_attn)' \
    "$UNICOT/requirements.txt" > "$tmpreq"

  "$VENV/bin/python" -m pip install -r "$tmpreq"
  rm -f "$tmpreq"

  # UniCoT requires transformers==4.49.0, whose compatible HF Hub range is
  # <1.0. Preserve the upstream 0.29.1 pin instead of upgrading to 1.x.
  "$VENV/bin/python" -m pip install \
    "huggingface_hub==0.29.1" \
    "hf_transfer==0.1.9" \
    "gradio_client==1.11.0" \
    tqdm pillow "accelerate>=0.34.0" safetensors

  # Select the official FA wheel that matches the ABI of the installed PyTorch.
  ABI="$("$VENV/bin/python" - <<'PY'
import torch
print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")
PY
)"
  FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"

  echo "[setup] PyTorch CXX11 ABI=$ABI"
  echo "[setup] installing official FlashAttention wheel:"
  echo "        $FA_WHEEL"

  "$VENV/bin/python" -m pip install --no-deps "$FA_WHEEL"

  # Validate imports, versions, Blackwell support, and run an actual FA kernel.
  "$VENV/bin/python" - <<'PY'
import torch
import transformers
import huggingface_hub
import flash_attn
from flash_attn import flash_attn_func

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("flash_attn:", flash_attn.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("arch list:", torch.cuda.get_arch_list() if torch.cuda.is_available() else [])

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.get_device_capability(0) != (12, 0):
    print("[WARN] Expected RTX PRO 6000 Blackwell / SM120")
if "sm_120" not in torch.cuda.get_arch_list():
    raise SystemExit("PyTorch build does not include sm_120")
if transformers.__version__ != "4.49.0":
    raise SystemExit(f"Unexpected transformers version {transformers.__version__}")
if not huggingface_hub.__version__.startswith("0.29."):
    raise SystemExit(f"Unexpected huggingface_hub version {huggingface_hub.__version__}")

# Real forward kernel launch, not merely an import.
q = torch.randn((1, 64, 4, 64), device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
print("flash_attn SM120 forward kernel OK:", tuple(out.shape), out.dtype)
PY

  touch "$ENV_MARKER"
else
  echo "[setup] reusing existing venv"
fi

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
  echo "transformers=$("$VENV/bin/python" -c 'import transformers; print(transformers.__version__)')"
  echo "huggingface_hub=$("$VENV/bin/python" -c 'import huggingface_hub; print(huggingface_hub.__version__)')"
  echo "flash_attn=$("$VENV/bin/python" -c 'import flash_attn; print(flash_attn.__version__)')"
  echo "flash_attn_install=official-v2.8.3-prebuilt-cu12-torch2.8-cp310"
} > "$ROOT/runs/$OUTPUT_NAME/PROVENANCE.txt"

echo
echo "[OK] setup complete"
echo "[OK] venv=$VENV"
echo "[OK] model=$MODEL"
echo "[OK] Uni-MMMU=$UMMMU"
echo "[OK] outputs=$ROOT/runs/$OUTPUT_NAME"
