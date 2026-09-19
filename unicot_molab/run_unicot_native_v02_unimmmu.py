#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TASKS = ["science", "math_geo", "jigsaw", "maze", "sliding", "code_rendering"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--output-name", default="unicot_native_v02")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-reflections", type=int, default=20)
    ap.add_argument("--max-gpu-memory", default="90GiB")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    here = Path(__file__).resolve().parent

    if args.tasks.strip().lower() == "all":
        tasks = TASKS
    else:
        tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
        bad = [x for x in tasks if x not in TASKS]
        if bad:
            raise SystemExit(f"Unknown tasks: {bad}; choices={TASKS}")

    for task in tasks:
        cmd = [
            sys.executable,
            str(here / "run_official_unimmmu_task_native_v02.py"),
            "--root", str(root),
            "--task", task,
            "--output-name", args.output_name,
            "--max-samples", str(args.max_samples),
            "--max-reflections", str(args.max_reflections),
            "--max-gpu-memory", args.max_gpu_memory,
        ]
        print("\n[native-v0.2] RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print("\n[native-v0.2] requested tasks finished", flush=True)


if __name__ == "__main__":
    main()
