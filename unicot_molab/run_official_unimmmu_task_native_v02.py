#!/usr/bin/env python3
"""
Run the official Uni-MMMU task script while replacing only its model-provider
functions with native UniCoT-v0.2 calls.

No TorchUMM inference class/adapter is imported.

Upstream UniCoT does not currently ship a Uni-MMMU understanding runner, so
this file is benchmark glue around the released native UniCoT-v0.2 primitives.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from native_unicot_v02 import (
    ContextItem,
    NativeUniCoTV02,
    add_image_path,
    add_text,
)


class GeneratedImageResult(os.PathLike):
    """Works both as a path and as `(path, metadata)` when unpacked."""
    def __init__(self, path, metadata=None):
        self.path = str(path)
        self.metadata = metadata or {}

    def __fspath__(self):
        return self.path

    def __str__(self):
        return self.path

    def __iter__(self):
        yield self.path
        yield self.metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--task",
        required=True,
        choices=["science", "math_geo", "jigsaw", "maze", "sliding", "code_rendering"],
    )
    ap.add_argument("--output-name", default="unicot_native_v02")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-reflections", type=int, default=20)
    ap.add_argument("--max-gpu-memory", default="90GiB")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    unicot_repo = root / "repos" / "UniCoT"
    ummmu_repo = root / "repos" / "Uni-MMMU"
    ckpt = root / "models" / "UniCoT-7B-MoT-v0.2"

    script = ummmu_repo / "sample_code_example" / "gpt" / f"{args.task}.py"
    if not script.is_file():
        raise SystemExit(f"Official Uni-MMMU task script missing: {script}")

    task_out_name = (
        "code" if args.task == "code_rendering"
        else ("math" if args.task == "math_geo" else args.task)
    )
    out_root = root / "outputs" / args.output_name / task_out_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[native-v0.2] task={args.task}", flush=True)
    print(f"[native-v0.2] official task script={script}", flush=True)
    print(f"[native-v0.2] output={out_root}", flush=True)

    backend = NativeUniCoTV02(
        unicot_repo,
        ckpt,
        max_gpu_memory=args.max_gpu_memory,
        max_reflections=args.max_reflections,
    )

    def generate_text_from_context(ctx, prompt_suffix="", *a, **kw):
        return backend.generate_text_from_context(ctx, prompt_suffix=prompt_suffix)

    def generate_image_from_context(ctx, out_path=None, prompt_suffix="", *a, **kw):
        if out_path is None:
            out_path = kw.get("output_path") or kw.get("save_path")
        if out_path is None:
            raise TypeError("Native bridge needs out_path/output_path/save_path")
        p = backend.generate_image_from_context(ctx, out_path, prompt_suffix=prompt_suffix)
        return GeneratedImageResult(p, {"backend": "UniCoT-v0.2-native"})

    injected = {
        "ContextItem": ContextItem,
        "add_text": add_text,
        "add_image_path": add_image_path,
        "generate_text_from_context": generate_text_from_context,
        "generate_image_from_context": generate_image_from_context,
        "PROJECT_ID": "local-native-unicot-v02",
        "LOCATION": "local",
    }

    ns = {
        "__name__": "unimmmu_native_task",
        "__file__": str(script),
        "__package__": None,
        **injected,
    }

    old_cwd = os.getcwd()
    os.chdir(ummmu_repo)
    try:
        source = script.read_text(encoding="utf-8")
        exec(compile(source, str(script), "exec"), ns, ns)

        # Force native provider functions even if an official example happens
        # to define provider-specific names itself.
        ns.update(injected)

        # Redirect task outputs into the Git-tracked run tree (via the shared-root symlink).
        for name in (
            "RUN_ROOT", "OUT_ROOT", "OUTPUT_ROOT", "OUT_DIR",
            "SAVE_ROOT", "RESULT_ROOT",
        ):
            if name in ns:
                ns[name] = out_root
        for name in ("SUMMARY_FN", "SUMMARY_PATH"):
            if name in ns:
                ns[name] = out_root / "summary.json"

        # Current official sliding.py ships data_dir="path_to_ummmu".
        # Point it to the actual Uni-MMMU checkout.
        if "data_dir" in ns and args.task == "sliding":
            ns["data_dir"] = str(ummmu_repo)

        # Resume rather than overwrite.
        for name, value in (
            ("RESUME_SKIP_DONE", True),
            ("FORCE_RERUN", False),
            ("REGENERATE_IF_NO_IMAGE", True),
        ):
            if name in ns:
                ns[name] = value

        # Current official Uni-MMMU task scripts use different sample-limit names:
        # Geometry TEST_LIMIT, Jigsaw MAX_TESTS, Maze/Sliding/Science NUM_SAMPLES,
        # Code NUM_SAMPLES_TOTAL. Keep extra legacy variants for robustness.
        limit = None if args.max_samples <= 0 else args.max_samples
        touched = []
        for name in (
            "TEST_LIMIT",
            "MAX_TESTS",
            "NUM_SAMPLES",
            "NUM_SAMPLES_TOTAL",
            "MAX_ITEMS",
            "MAX_CASES",
            "NUM_CASES",
            "TEST_NUM",
            "SAMPLE_NUM",
        ):
            if name in ns:
                ns[name] = limit
                touched.append(name)

        print(
            f"[native-v0.2] sample limit={args.max_samples or 'ALL'} "
            f"controls={touched or ['NONE']}",
            flush=True,
        )
        if args.max_samples > 0 and not touched:
            raise RuntimeError(
                f"Smoke-test limit requested but no sample-limit control was found in {script}"
            )

        main_fn = ns.get("main")
        if not callable(main_fn):
            raise RuntimeError(f"{script} did not define main()")
        main_fn()
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
