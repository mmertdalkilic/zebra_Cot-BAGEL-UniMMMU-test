#!/usr/bin/env bash
set -euo pipefail
: "${MOLAB_CHECKOUT:?}"
: "${MOLAB_RUNTIME:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$MOLAB_RUNTIME"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user 'uv==0.8.22'
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_PYTHON_INSTALL_DIR="$MOLAB_RUNTIME/python"
export UV_CACHE_DIR="$MOLAB_RUNTIME/uv-cache"
uv python install 3.11
if [[ ! -x "$MOLAB_RUNTIME/venv/bin/python" ]]; then
  uv venv --python 3.11 "$MOLAB_RUNTIME/venv"
fi
PY="$MOLAB_RUNTIME/venv/bin/python"
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 +  'torch==2.8.0' 'torchvision==0.23.0'
uv pip install --python "$PY" -r "$SCRIPT_DIR/requirements_eval.txt" +  --constraint "$SCRIPT_DIR/torch_constraints.txt"
uv pip check --python "$PY"
if [[ ! -d "$MOLAB_RUNTIME/Uni-MMMU/.git" ]]; then
  git clone https://github.com/Vchitect/Uni-MMMU.git "$MOLAB_RUNTIME/Uni-MMMU"
fi
git -C "$MOLAB_RUNTIME/Uni-MMMU" fetch origin cd1675292e5a7a713c5d6026d234109521003124
git -C "$MOLAB_RUNTIME/Uni-MMMU" checkout --detach cd1675292e5a7a713c5d6026d234109521003124
if [[ ! -f "$MOLAB_RUNTIME/Uni-MMMU/data/science/dim_all.json" ]]; then
  "$PY" - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
root=os.environ["MOLAB_RUNTIME"]
p=hf_hub_download("Vchitect/Uni-MMMU-Eval","data.tar",repo_type="dataset",
  revision="f6cabc460cbf6592acbe421e6e54e804fbb11fe7")
with tarfile.open(p) as t: t.extractall(root+"/Uni-MMMU",filter="data")
PY
fi
"$PY" - <<'PY'
import torch,transformers
assert torch.cuda.is_available(),"Attach the RTX PRO 6000 GPU in MoLab"
p=torch.cuda.get_device_properties(0)
assert p.total_memory>=80*1024**3,"Select the 96 GB GPU"
print("GPU",p.name,"VRAM GiB",round(p.total_memory/1024**3,1))
print("torch",torch.__version__,"CUDA",torch.version.cuda,"transformers",transformers.__version__)
PY
