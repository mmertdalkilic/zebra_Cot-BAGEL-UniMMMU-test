#!/usr/bin/env bash
set -euo pipefail
: "${MOLAB_CHECKOUT:?}"
: "${MOLAB_RUNTIME:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$MOLAB_RUNTIME/Uni-MMMU"
PY="$MOLAB_RUNTIME/venv/bin/python"
MODEL="bagel-zebra-cot-lora"
cd "$ROOT"
rm -rf outputs
ln -s "$MOLAB_CHECKOUT/outputs" outputs
cp eval_ummmu.py eval_ummmu_molab.py
sed -i "s#\"path-to-UMMMU-home\"#\"$ROOT\"#" eval_ummmu_molab.py
sed -i 's#"Qwen/Qwen3-32B"#"Qwen/Qwen3-32B-AWQ"#' eval_ummmu_molab.py
sed -i 's#"Qwen/Qwen2.5-VL-72B-Instruct"#"Qwen/Qwen2.5-VL-72B-Instruct-AWQ"#' eval_ummmu_molab.py
sed -i "s#/mnt/petrelfs/zoukai/.cache#$MOLAB_RUNTIME/dreamsim-cache#" eval_ummmu_molab.py
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export EVAL_TIME_BUDGET_HOURS="${EVAL_TIME_BUDGET_HOURS:-9.5}"
export QWEN3_REV="0499c3ac83fdef8810b907a23894ba91e95eddd8"
export QWEN25VL_REV="c8b87d4b81f34b6a147577a310d7e75f0698f6c2"
unset EVAL_SYNC_CMD
"$PY" "$SCRIPT_DIR/apply_eval_resume.py" eval_ummmu_molab.py
"$PY" "$SCRIPT_DIR/apply_eval_sequential.py" eval_ummmu_molab.py
sed -i 's#AutoTokenizer.from_pretrained(model_name)#AutoTokenizer.from_pretrained(model_name, revision=os.environ["QWEN3_REV"])#' eval_ummmu_molab.py
sed -i 's#model_name, torch_dtype="auto", device_map="auto"#model_name, revision=os.environ["QWEN3_REV"], torch_dtype="auto", device_map="auto"#' eval_ummmu_molab.py
sed -i 's#from_pretrained(model_name, \*\*kwargs)#from_pretrained(model_name, revision=os.environ["QWEN25VL_REV"], **kwargs)#' eval_ummmu_molab.py
sed -i 's#AutoProcessor.from_pretrained(model_name)#AutoProcessor.from_pretrained(model_name, revision=os.environ["QWEN25VL_REV"])#' eval_ummmu_molab.py
"$PY" - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path(os.environ["MOLAB_CHECKOUT"])
scripts=root/"molab_eval_transfer"
protocol={
 "official_scorer_commit":"cd1675292e5a7a713c5d6026d234109521003124",
 "dataset_revision":"f6cabc460cbf6592acbe421e6e54e804fbb11fe7",
 "vision_judge":{"model":"Qwen/Qwen2.5-VL-72B-Instruct-AWQ","revision":os.environ["QWEN25VL_REV"]},
 "text_judge":{"model":"Qwen/Qwen3-32B-AWQ","revision":os.environ["QWEN3_REV"],"enable_thinking":True},
 "attention":"sdpa","sampling":"do_sample=False","model_output":"original Leonardo generations",
 "workflow_sha256":{}
}
for name in ("apply_eval_resume.py","apply_eval_sequential.py","patch_awq_triton.py","run_official_eval.sh"):
 p=scripts/name; protocol["workflow_sha256"][name]=hashlib.sha256(p.read_bytes()).hexdigest()
out=root/"outputs/_eval/bagel-zebra-cot-lora/protocol.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(protocol,indent=2)+"\n")
PY
"$PY" -u eval_ummmu_molab.py --model_name "$MODEL"
