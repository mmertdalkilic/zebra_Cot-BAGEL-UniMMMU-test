#!/usr/bin/env python3
from __future__ import annotations
import csv, io, json
from pathlib import Path
from common import MODEL, atomic_json

def flatten(v,p=""):
    if isinstance(v,dict):
        for k,x in v.items(): yield from flatten(x,f"{p}.{k}" if p else k)
    elif isinstance(v,list):
        for i,x in enumerate(v): yield from flatten(x,f"{p}[{i}]")
    elif isinstance(v,(int,float)) and not isinstance(v,bool): yield p,v

def usable(obj):
    blob=json.dumps(obj)
    return not any(x in blob for x in ("API Error","IncompatibleTypeError","rshift_cuda"))

def complete_record(task,obj):
    if not usable(obj): return False
    if task=="math":
        return obj.get("status")=="ok" and "overlay_ok" in obj and "text_ok" in obj
    if task=="science":
        return "text_eval" in obj and "image_eval" in obj
    if task=="code":
        return bool(obj.get("image_evals")) and "text_eval" in obj
    return True

def build(checkout):
    checkout=Path(checkout); root=checkout/"outputs/_eval"/MODEL; root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((checkout/"generation_manifest.json").read_text()); counts={}; rows=[]
    for task in ("math","science","code","jigsaw"):
        files=sorted((root/task/"items").glob("*.json"))
        completed=0
        for f in files:
            try: obj=json.loads(f.read_text())
            except Exception: continue
            if usable(obj):
                for metric,value in flatten(obj): rows.append((task,f.stem,metric,value))
            if complete_record(task,obj): completed+=1
        counts[task]={"saved":completed,"files":len(files),"expected":manifest["counts"][task]}
    for task in ("maze","sliding"):
        summary=root/task/"summary.json"; done=summary.exists()
        counts[task]={"saved":manifest["counts"][task] if done else 0,"expected":manifest["counts"][task]}
        if done:
            for metric,value in flatten(json.loads(summary.read_text())): rows.append((task,"SUMMARY",metric,value))
    complete=all(x["saved"]==x["expected"] for x in counts.values())
    atomic_json(root/"progress.json",{"complete":complete,"tasks":counts})
    output=io.StringIO(); w=csv.writer(output); w.writerow(["task","case_id","metric","value"]); w.writerows(rows)
    (root/"scores.csv").write_text(output.getvalue())
    lines=["# BAGEL Zebra-CoT LoRA evaluation progress","","| Task | Saved | Expected |","|---|---:|---:|"]
    lines += [f"| {k} | {v['saved']} | {v['expected']} |" for k,v in counts.items()]
    lines += ["",f"Complete: **{complete}**","","scores.csv contains every numeric field saved so far.","Partial task averages are not final benchmark results.",""]
    (root/"PROGRESS.md").write_text("\n".join(lines))
    return complete

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("checkout",type=Path)
    print("complete=",build(p.parse_args().checkout))
