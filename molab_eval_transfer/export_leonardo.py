#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from common import BRANCH, MODEL, Git, atomic_json, portable, sha256, token

TASKS=("math","science","code","jigsaw","maze","sliding")
KEEP={".json",".txt",".png",".jpg",".jpeg",".webp",".ok"}
EXPECTED={"math":140,"science":157,"code":200,"jigsaw":150,"maze":149,"sliding":84}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",type=Path,default=Path("/leonardo_work/EUHPC_B38_013/mdalkili/bagel_zebra_leonardo"))
    ap.add_argument("--checkout",type=Path,required=True)
    ap.add_argument("--branch",default=BRANCH); ap.add_argument("--model",default=MODEL)
    args=ap.parse_args()
    benchmark=(args.project/"Uni-MMMU").resolve()
    source=benchmark/"outputs"/args.model
    if not source.is_dir(): raise SystemExit(f"Missing generation directory: {source}")
    inventory={}; counts={}
    for task in TASKS:
        case_dirs=sorted(p.parent for p in (source/task).glob("*/result.json"))
        if not case_dirs: raise SystemExit(f"No generated cases found for {task}")
        counts[task]=len(case_dirs)
        if counts[task] != EXPECTED[task]:
            raise SystemExit(f"{task}: found {counts[task]} cases; expected {EXPECTED[task]}")
        for case in case_dirs:
            rec=json.loads((case/"result.json").read_text())
            if rec.get("status") not in ("ok","success","completed") or not (case/"_done.ok").exists():
                raise SystemExit(f"Incomplete generation: {case} status={rec.get('status')}")
            if not (case/"model_text.txt").exists(): raise SystemExit(f"Missing text: {case}")
            for p in case.rglob("*"):
                if p.is_file() and p.suffix.lower() in KEEP:
                    rel=str(p.relative_to(source)); inventory[rel]={"bytes":p.stat().st_size,"sha256_source":sha256(p)}
                    if p.stat().st_size>=90*1024**2: raise SystemExit(f"File too large for regular Git: {p}")
    print("Validated cases:",counts,"files:",len(inventory),flush=True)
    git=Git(token())
    try:
        checkout=git.clone(args.checkout,args.branch,create=True)
        out=checkout/"outputs"/args.model
        for rel,meta in inventory.items():
            src=source/rel; dst=out/rel; dst.parent.mkdir(parents=True,exist_ok=True)
            if src.name=="result.json":
                atomic_json(dst,portable(json.loads(src.read_text()),benchmark))
            else: shutil.copy2(src,dst)
            meta["sha256_export"]=sha256(dst)
        # Official jigsaw scorer enumerates summary.per_item. Rebuild it from
        # verified case folders so resumed/sharded generation cannot omit cases.
        jids=sorted(p.name for p in (out/"jigsaw").iterdir() if p.is_dir())
        atomic_json(out/"jigsaw/summary.json",{"count_total":len(jids),"per_item":[{"id":x} for x in jids]})
        scripts=checkout/"molab_eval_transfer"; scripts.mkdir(exist_ok=True)
        for p in Path(__file__).parent.iterdir():
            if p.is_file() and p.suffix in (".py",".sh",".txt",".md"): shutil.copy2(p,scripts/p.name)
        manifest={"format":1,"complete":False,"model":args.model,"source":str(source),
          "source_benchmark":str(benchmark),"counts":counts,"files":inventory}
        mp=checkout/"generation_manifest.json"; atomic_json(mp,manifest)
        bootstrap=[str(p.relative_to(checkout)) for p in scripts.iterdir() if p.is_file()]+["generation_manifest.json"]
        git.push(checkout,bootstrap,"Add Leonardo to MoLab evaluation workflow",args.branch)
        batch=[]; size=0
        for rel,meta in inventory.items():
            batch.append(str(Path("outputs")/args.model/rel)); size+=meta["bytes"]
            if size>=150*1024**2:
                git.push(checkout,batch,"Upload Leonardo generation batch",args.branch); batch=[]; size=0
        batch.append(str(Path("outputs")/args.model/"jigsaw/summary.json"))
        git.push(checkout,batch,"Upload final Leonardo generation batch",args.branch)
        manifest["complete"]=True; atomic_json(mp,manifest)
        git.push(checkout,["generation_manifest.json"],"Mark Leonardo generation transfer complete",args.branch)
        print("TRANSFER COMPLETE:",args.branch,flush=True)
    finally: git.close()
if __name__=="__main__": main()
