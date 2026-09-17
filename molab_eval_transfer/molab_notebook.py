import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import os, subprocess, sys, tempfile, time, uuid
    from pathlib import Path
    return Path, mo, os, subprocess, sys, tempfile, time, uuid

@app.cell
def _(mo):
    mo.md("""
    # BAGEL Zebra-CoT LoRA — Uni-MMMU evaluation

    First attach the **NVIDIA RTX PRO 6000 96 GB GPU** in MoLab's notebook
    specifications. The Leonardo transfer must have printed TRANSFER COMPLETE.
    Enter the token only in the password field below.
    """)
    return

@app.cell
def _(mo):
    github_token = mo.ui.text(kind="password",label="GitHub token").form(
        submit_button_label="Use token for this session")
    hours = mo.ui.number(start=0.5,stop=10.5,step=0.5,value=10.0,label="Run budget (hours)")
    start = mo.ui.run_button(label="Start or resume evaluation")
    mo.vstack([github_token,hours,start])
    return github_token, hours, start

@app.cell
def _(Path, github_token, hours, mo, os, start, subprocess, sys, tempfile, time, uuid):
    mo.stop(not start.value,mo.md("Submit the token, then click Start."))
    mo.stop(not github_token.value,mo.md("The token field is empty."))
    branch="leonardo-lora-0015270-molab-eval"
    repo="https://github.com/mmertdalkilic/zebra_Cot-BAGEL-UniMMMU-test.git"
    session_dir=Path.home()/"bagel_molab_sessions"/(time.strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6])
    checkout=session_dir/"repo"; runtime=session_dir/"runtime"; session_dir.mkdir(parents=True)
    env=dict(os.environ,GITHUB_TOKEN=github_token.value,GIT_TERMINAL_PROMPT="0")
    with tempfile.TemporaryDirectory(prefix="bagel-auth-") as auth:
        helper=Path(auth)/"askpass.py"
        helper.write_text("#!"+sys.executable+"\nimport os,sys\nprint('x-access-token' if 'username' in sys.argv[1].lower() else os.environ['GITHUB_TOKEN'])\n")
        helper.chmod(0o700); env["GIT_ASKPASS"]=str(helper)
        subprocess.run(["git","-c","credential.helper=","clone","--single-branch","--branch",branch,repo,str(checkout)],
                       env=env,check=True)
    env.pop("GIT_ASKPASS",None)
    supervisor_log=session_dir/"supervisor.log"
    log_handle=supervisor_log.open("w")
    evaluation_process=subprocess.Popen(
        [sys.executable,"-u",str(checkout/"molab_eval_transfer/run_molab.py"),
         "--checkout",str(checkout),"--runtime",str(runtime),"--hours",str(hours.value)],
        env=env,stdout=log_handle,stderr=subprocess.STDOUT,start_new_session=True)
    del env
    mo.md(f"Started PID **{evaluation_process.pid}**. Session directory: {session_dir}")
    return checkout, evaluation_process, log_handle, runtime, session_dir, supervisor_log

@app.cell
def _(checkout, evaluation_process, mo, runtime, supervisor_log, time):
    def recent(path):
        if not path.exists(): return ""
        with path.open("rb") as f:
            f.seek(max(0,path.stat().st_size-16384))
            return "\n".join(f.read().decode(errors="replace").splitlines()[-15:])
    while evaluation_process.poll() is None:
        progress=checkout/"outputs/_eval/bagel-zebra-cot-lora/PROGRESS.md"
        phase_logs=sorted(runtime.glob("*.log"),key=lambda p:p.stat().st_mtime)
        mo.output.replace(mo.vstack([
            mo.md(progress.read_text() if progress.exists() else "Setup is running."),
            mo.plain_text(recent(supervisor_log)),
            mo.plain_text(recent(phase_logs[-1]) if phase_logs else "")]))
        time.sleep(10)
    mo.output.append(mo.md(f"Finished with exit code **{evaluation_process.returncode}**."))
    mo.output.append(mo.plain_text(recent(supervisor_log)))
    return

@app.cell
def _(evaluation_process, mo):
    stop = mo.ui.run_button(label="Pause and push now")
    if stop.value and evaluation_process.poll() is None:
        evaluation_process.terminate()
    stop
    return

if __name__ == "__main__":
    app.run()
