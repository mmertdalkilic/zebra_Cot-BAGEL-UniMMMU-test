#!/usr/bin/env python3
"""
Run official Uni-MMMU task scripts while replacing their provider calls
with native UniCoT-v0.2 generation.

No TorchUMM inference class or TorchUMM model adapter is instantiated.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

from native_unicot_v02 import (
    ContextItem,
    NativeUniCoTV02,
    add_image_path,
    add_text,
)


class GeneratedImageResult(os.PathLike):
    """Works as a path and as `(path, metadata)` when unpacked."""

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
        choices=[
            "science",
            "math_geo",
            "jigsaw",
            "maze",
            "sliding",
            "code_rendering",
        ],
    )

    ap.add_argument(
        "--output-name",
        default="unicot_native_v02",
    )

    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--max-reflections",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--max-gpu-memory",
        default="90GiB",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()

    unicot_repo = root / "repos" / "UniCoT"
    ummmu_repo = root / "repos" / "Uni-MMMU"

    ckpt = root / "models" / "UniCoT-7B-MoT-v0.2"

    script = (
        ummmu_repo
        / "sample_code_example"
        / "gpt"
        / f"{args.task}.py"
    )

    if not script.is_file():
        raise SystemExit(
            f"Official Uni-MMMU task script missing: {script}"
        )

    task_out_name = (
        "code"
        if args.task == "code_rendering"
        else (
            "math"
            if args.task == "math_geo"
            else args.task
        )
    )

    out_root = (
        root
        / "outputs"
        / args.output_name
        / task_out_name
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[native-v0.2] task={args.task}",
        flush=True,
    )

    print(
        f"[native-v0.2] official task script={script}",
        flush=True,
    )

    print(
        f"[native-v0.2] output={out_root}",
        flush=True,
    )

    backend = NativeUniCoTV02(
        unicot_repo,
        ckpt,
        max_gpu_memory=args.max_gpu_memory,
        max_reflections=args.max_reflections,
    )

    def generate_text_from_context(
        ctx,
        prompt_suffix="",
        *a,
        **kw,
    ):
        return backend.generate_text_from_context(
            ctx,
            prompt_suffix=prompt_suffix,
        )

    def generate_image_from_context(
        ctx,
        out_path=None,
        prompt_suffix="",
        *a,
        **kw,
    ):
        if out_path is None:
            out_path = (
                kw.get("output_path")
                or kw.get("save_path")
            )

        if out_path is None:
            raise TypeError(
                "Native bridge needs "
                "out_path/output_path/save_path"
            )

        generated = backend.generate_image_from_context(
            ctx,
            out_path,
            prompt_suffix=prompt_suffix,
        )

        return GeneratedImageResult(
            generated,
            {
                "backend":
                "UniCoT-v0.2-native"
            },
        )

    injected = {
        "ContextItem": ContextItem,
        "add_text": add_text,
        "add_image_path": add_image_path,
        "generate_text_from_context":
            generate_text_from_context,
        "generate_image_from_context":
            generate_image_from_context,
        "PROJECT_ID":
            "local-native-unicot-v02",
        "LOCATION":
            "local",
    }

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # @dataclass looks up cls.__module__ through sys.modules.
    # A bare exec(namespace) is therefore insufficient for scripts
    # such as the official Maze runner.
    # ------------------------------------------------------------

    module_name = (
        f"_unimmmu_native_{args.task}"
    )

    task_module = types.ModuleType(
        module_name
    )

    task_module.__file__ = str(script)
    task_module.__package__ = None

    task_module.__dict__.update(
        injected
    )

    sys.modules[module_name] = task_module

    ns = task_module.__dict__

    old_cwd = os.getcwd()

    os.chdir(ummmu_repo)

    try:
        source = script.read_text(
            encoding="utf-8"
        )

        exec(
            compile(
                source,
                str(script),
                "exec",
            ),
            ns,
            ns,
        )

        # Official example scripts may define provider-specific
        # symbols themselves. Always restore the native backend.
        ns.update(injected)

        # --------------------------------------------------------
        # JIGSAW PATH FIX
        #
        # Current metadata may already contain:
        #
        # ./data/jigsaw_dataset_2x2ref/ex_x/ref_2x2.png
        #
        # while the example's resolver prepends DATASET_DIR.
        # Prefer paths relative to the Uni-MMMU repository when
        # they already resolve correctly.
        # --------------------------------------------------------

        if args.task == "jigsaw":

            raw_dataset_dir = Path(
                ns.get(
                    "DATASET_DIR",
                    "./data/jigsaw_dataset_2x2ref",
                )
            )

            if raw_dataset_dir.is_absolute():
                dataset_base = raw_dataset_dir
            else:
                dataset_base = (
                    ummmu_repo
                    / raw_dataset_dir
                ).resolve()

            def resolve_under_dataset_fixed(p):
                if not p:
                    return None

                pth = Path(p)

                if pth.is_absolute():
                    return pth

                # Metadata already contains data/...:
                direct = (
                    ummmu_repo
                    / pth
                ).resolve()

                if direct.exists():
                    return direct

                # Metadata contains only ex_x/...:
                nested = (
                    dataset_base
                    / pth
                ).resolve()

                return nested

            ns[
                "resolve_under_dataset"
            ] = resolve_under_dataset_fixed

            print(
                "[native-v0.2] "
                "Jigsaw path resolver patched",
                flush=True,
            )

        # --------------------------------------------------------
        # Redirect official outputs into the Git-tracked run tree.
        # --------------------------------------------------------

        for name in (
            "RUN_ROOT",
            "OUT_ROOT",
            "OUTPUT_ROOT",
            "OUT_DIR",
            "SAVE_ROOT",
            "RESULT_ROOT",
        ):
            if name in ns:
                ns[name] = out_root

        for name in (
            "SUMMARY_FN",
            "SUMMARY_PATH",
        ):
            if name in ns:
                ns[name] = (
                    out_root
                    / "summary.json"
                )

        # Sliding's released example contains a placeholder root.
        if (
            args.task == "sliding"
            and "data_dir" in ns
        ):
            ns["data_dir"] = str(
                ummmu_repo
            )

        # Resume completed examples.
        for name, value in (
            (
                "RESUME_SKIP_DONE",
                True,
            ),
            (
                "FORCE_RERUN",
                False,
            ),
            (
                "REGENERATE_IF_NO_IMAGE",
                True,
            ),
        ):
            if name in ns:
                ns[name] = value

        # --------------------------------------------------------
        # Official task scripts currently use different limit names.
        # --------------------------------------------------------

        limit = (
            None
            if args.max_samples <= 0
            else args.max_samples
        )

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
            "[native-v0.2] "
            f"sample limit="
            f"{args.max_samples or 'ALL'} "
            f"controls={touched or ['NONE']}",
            flush=True,
        )

        if (
            args.max_samples > 0
            and not touched
        ):
            raise RuntimeError(
                "Smoke-test limit requested "
                "but no sample-limit control "
                f"was found in {script}"
            )

        main_fn = ns.get("main")

        if not callable(main_fn):
            raise RuntimeError(
                f"{script} did not define main()"
            )

        main_fn()

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
