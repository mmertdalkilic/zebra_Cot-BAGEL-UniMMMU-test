# MoLab cell contents

The recommended route is to import molab_notebook.py from the new GitHub
branch. If you prefer a blank MoLab notebook, use these Python cells.

## Cell 1 — secure inputs

    import marimo as mo
    github_token = mo.ui.text(kind="password", label="GitHub token").form(
        submit_button_label="Use token for this session")
    hours = mo.ui.number(start=0.5, stop=10.5, step=0.5, value=10.0,
                         label="Run budget (hours)")
    start = mo.ui.run_button(label="Start or resume evaluation")
    mo.vstack([github_token, hours, start])

## Cell 2 — clone and start

    import os, subprocess, sys, tempfile, time, uuid
    from pathlib import Path
    mo.stop(not start.value)
    mo.stop(not github_token.value)
    branch = "leonardo-lora-0015270-molab-eval"
    repo = "https://github.com/mmertdalkilic/zebra_Cot-BAGEL-UniMMMU-test.git"
    session_dir = Path.home() / "bagel_molab_sessions" / (
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    checkout, runtime = session_dir / "repo", session_dir / "runtime"
    session_dir.mkdir(parents=True)
    env = dict(os.environ, GITHUB_TOKEN=github_token.value,
               GIT_TERMINAL_PROMPT="0")
    with tempfile.TemporaryDirectory(prefix="bagel-auth-") as auth:
        helper = Path(auth) / "askpass.py"
        helper.write_text("#!" + sys.executable +
            "\nimport os,sys\nprint('x-access-token' if 'username' in sys.argv[1].lower() else os.environ['GITHUB_TOKEN'])\n")
        helper.chmod(0o700)
        env["GIT_ASKPASS"] = str(helper)
        subprocess.run(["git", "-c", "credential.helper=", "clone",
            "--single-branch", "--branch", branch, repo, str(checkout)],
            env=env, check=True)
    env.pop("GIT_ASKPASS", None)
    supervisor_log = session_dir / "supervisor.log"
    log_handle = supervisor_log.open("w")
    evaluation_process = subprocess.Popen(
        [sys.executable, "-u",
         str(checkout / "molab_eval_transfer/run_molab.py"),
         "--checkout", str(checkout), "--runtime", str(runtime),
         "--hours", str(hours.value)],
        env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True)
    del env
    (evaluation_process.pid, session_dir)

## Cell 3 — monitor

    def recent(path):
        if not path.exists():
            return ""
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - 16384))
            return "\n".join(
                f.read().decode(errors="replace").splitlines()[-15:])

    while evaluation_process.poll() is None:
        progress = checkout / (
            "outputs/_eval/bagel-zebra-cot-lora/PROGRESS.md")
        logs = sorted(runtime.glob("*.log"),
                      key=lambda p: p.stat().st_mtime)
        mo.output.replace(mo.vstack([
            mo.md(progress.read_text()
                  if progress.exists() else "Setup is running."),
            mo.plain_text(recent(supervisor_log)),
            mo.plain_text(recent(logs[-1]) if logs else "")]))
        time.sleep(10)
    evaluation_process.returncode

To stop, interrupt the monitoring cell and run:

    evaluation_process.terminate()
    evaluation_process.wait()

Wait for the final [sync] pushed line in supervisor.log.
