from __future__ import annotations
import getpass, hashlib, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

REPO = "https://github.com/mmertdalkilic/zebra_Cot-BAGEL-UniMMMU-test.git"
BRANCH = "leonardo-lora-0015270-molab-eval"
MODEL = "bagel-zebra-cot-lora"
SCORER_REV = "cd1675292e5a7a713c5d6026d234109521003124"
DATA_REV = "f6cabc460cbf6592acbe421e6e54e804fbb11fe7"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".writing-",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(obj,f,ensure_ascii=False,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def token():
    return os.environ.get("GITHUB_TOKEN") or getpass.getpass("GitHub token (hidden, not stored): ")

class Git:
    def __init__(self,pat):
        if not pat: raise ValueError("GitHub token is required")
        self.tmp=tempfile.TemporaryDirectory(prefix="bagel-git-auth-")
        helper=Path(self.tmp.name)/"askpass.py"
        helper.write_text("#!"+sys.executable+"\nimport os,sys\nprint('x-access-token' if 'username' in sys.argv[1].lower() else os.environ['GITHUB_TOKEN'])\n")
        helper.chmod(0o700)
        self.env=dict(os.environ,GITHUB_TOKEN=pat,GIT_ASKPASS=str(helper),GIT_TERMINAL_PROMPT="0")
        for k in ("GIT_TRACE","GIT_TRACE_CURL","GIT_CURL_VERBOSE"): self.env.pop(k,None)
    def run(self,*args,cwd=None,check=True,capture=True):
        return subprocess.run(["git","-c","credential.helper=","-c","core.hooksPath=/dev/null",
          "-c","pack.threads=2","-c","pack.windowMemory=64m","-c","gc.auto=0",*map(str,args)],
          cwd=cwd,env=self.env,check=check,text=True,
          stdout=subprocess.PIPE if capture else None,stderr=subprocess.PIPE if capture else None,timeout=900)
    def clone(self,dest,branch=BRANCH,create=False):
        dest=Path(dest).resolve(); self.run("check-ref-format","--branch",branch)
        if not (dest/".git").is_dir():
            exists=bool(self.run("ls-remote","--heads",REPO,branch).stdout.strip())
            if not exists and not create: raise RuntimeError("Branch does not exist; complete the Leonardo export first")
            args=["clone","--single-branch"]
            if exists: args += ["--branch",branch]
            self.run(*args,REPO,dest,capture=False)
            if not exists: self.run("switch","-c",branch,cwd=dest)
        if self.run("remote","get-url","origin",cwd=dest).stdout.strip()!=REPO: raise RuntimeError("Unexpected Git remote")
        if self.run("branch","--show-current",cwd=dest).stdout.strip()!=branch: raise RuntimeError("Unexpected Git branch")
        self.run("config","user.name","BAGEL evaluation",cwd=dest)
        self.run("config","user.email","mmertdalkilic@users.noreply.github.com",cwd=dest)
        return dest
    def push(self,checkout,paths,message,branch=BRANCH):
        paths=[str(x) for x in paths if (Path(checkout)/x).exists()]
        if not paths: return
        for i in range(0,len(paths),200): self.run("add","--",*paths[i:i+200],cwd=checkout)
        if self.run("diff","--cached","--quiet",cwd=checkout,check=False).returncode==1:
            self.run("commit","-m",message,cwd=checkout)
        self.run("push","--set-upstream","origin",f"HEAD:refs/heads/{branch}",cwd=checkout,capture=False)
        print("[sync] pushed",self.run("rev-parse","--short","HEAD",cwd=checkout).stdout.strip(),time.strftime("%FT%TZ",time.gmtime()),flush=True)
    def close(self): self.tmp.cleanup()

def portable(value,old_root):
    if isinstance(value,dict): return {k:portable(v,old_root) for k,v in value.items()}
    if isinstance(value,list): return [portable(v,old_root) for v in value]
    if isinstance(value,str):
        prefix=str(old_root)
        if value==prefix or value.startswith(prefix+"/"): return "."+value[len(prefix):]
    return value
