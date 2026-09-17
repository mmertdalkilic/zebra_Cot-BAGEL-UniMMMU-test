#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, threading, time
from pathlib import Path
from common import BRANCH, MODEL, Git, atomic_json, token
from build_progress import build

def eval_paths(checkout):
    root=Path("outputs")/"_eval"/MODEL; result=[]
    for p in (checkout/root).rglob("*"):
        if p.is_file() and not p.name.startswith(".") and p.suffix in (".json",".csv",".md",".xlsx"):
            result.append(str(p.relative_to(checkout)))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkout",type=Path,required=True)
    ap.add_argument("--runtime",type=Path,required=True)
    ap.add_argument("--branch",default=BRANCH)
    ap.add_argument("--hours",type=float,default=10.0)
    args=ap.parse_args()
    if not 0<args.hours<=10.5: raise SystemExit("--hours must be >0 and <=10.5")
    checkout=args.checkout.resolve(); runtime=args.runtime.resolve(); runtime.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((checkout/"generation_manifest.json").read_text())
    if not manifest.get("complete"): raise SystemExit("Leonardo transfer is not complete")
    git=Git(token()); stop=threading.Event(); backup_failed=threading.Event(); sync_failures=[0]; child=[None]
    status=checkout/"outputs/_eval"/MODEL/"session_status.json"
    def sync():
        try:
            build(checkout); paths=eval_paths(checkout)
            if paths: git.push(checkout,paths,"Save resumable Uni-MMMU evaluation progress",args.branch)
            sync_failures[0]=0
        except Exception as e:
            sync_failures[0]+=1; print(f"[sync failed {sync_failures[0]}] {e}",flush=True)
            if sync_failures[0]>=3:
                backup_failed.set()
                stop.set()
                if child[0] and child[0].poll() is None: child[0].terminate()
    def loop():
        while not stop.wait(900): sync()
    def handler(*_):
        stop.set()
        if child[0] and child[0].poll() is None: child[0].terminate()
    signal.signal(signal.SIGINT,handler); signal.signal(signal.SIGTERM,handler)
    code=1
    try:
        git.clone(checkout,args.branch)
        atomic_json(status,{"state":"starting","time":time.strftime("%FT%TZ",time.gmtime())}); sync()
        threading.Thread(target=loop,daemon=True).start()
        env=dict(os.environ,MOLAB_CHECKOUT=str(checkout),MOLAB_RUNTIME=str(runtime),
          HF_HOME=str(runtime/"hf"),TORCH_HOME=str(runtime/"torch"),PYTHONUNBUFFERED="1",
          HF_HUB_ETAG_TIMEOUT="60",HF_HUB_DOWNLOAD_TIMEOUT="60",HF_HUB_DISABLE_XET="1")
        for k in ("GITHUB_TOKEN","GIT_ASKPASS","GIT_TRACE","GIT_TRACE_CURL","GIT_CURL_VERBOSE"): env.pop(k,None)
        deadline=time.time()+args.hours*3600
        commands=[("setup",["bash",str(checkout/"molab_eval_transfer/setup_eval.sh")]),
          ("evaluation",["bash",str(checkout/"molab_eval_transfer/run_official_eval.sh")])]
        code=0
        for phase,cmd in commands:
            if stop.is_set() or time.time()>=deadline: code=75; break
            atomic_json(status,{"state":"running","phase":phase,"time":time.strftime("%FT%TZ",time.gmtime())}); sync()
            log=runtime/(phase+".log")
            with log.open("a",buffering=1) as f:
                child[0]=subprocess.Popen(cmd,cwd=checkout,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                while child[0].poll() is None:
                    if stop.is_set() or time.time()>=deadline:
                        child[0].terminate()
                        try: child[0].wait(30)
                        except subprocess.TimeoutExpired: child[0].kill()
                        code=75; break
                    time.sleep(2)
                if not code: code=child[0].returncode
            sync()
            if code: break
        complete=build(checkout)
        state="complete" if code==0 and complete else ("paused" if code==75 else "failed")
        if code==0 and not complete: code=1; state="failed_incomplete"
        atomic_json(status,{"state":state,"exit_code":code,"time":time.strftime("%FT%TZ",time.gmtime())})
    finally:
        stop.set()
        for _ in range(3):
            sync()
            if sync_failures[0]==0: break
            time.sleep(5)
        git.close()
    return 1 if backup_failed.is_set() else code
if __name__=="__main__": sys.exit(main())
