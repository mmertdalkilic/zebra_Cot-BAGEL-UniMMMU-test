# -*- coding: utf-8 -*-
"""Score geometry text_ok only for items that already have overlay_ok.

Loads Qwen3-32B-AWQ. Never instantiates Qwen2.5-VL. Does not rewrite
science/ or code/ judge files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from finalize_eval import (
    VL_PROTECTED_REL,
    _collect_protected_files,
    _fingerprint_files,
    fix_geometry_status,
)
from report_ummmu_table import write_paper_table


def _sha256_json_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _overlay_fingerprint(items_dir: Path) -> str:
    parts: List[str] = []
    for p in sorted(items_dir.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        parts.append(
            _sha256_json_obj(
                {
                    "id": rec.get("id") or p.stem,
                    "overlay_ok": rec.get("overlay_ok"),
                    "overlay_reason": rec.get("overlay_reason"),
                }
            )
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _missing_text_ids(items_dir: Path) -> List[str]:
    missing = []
    for p in sorted(items_dir.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        has_overlay = rec.get("overlay_ok") is not None
        has_text = "text_ok" in rec and rec.get("text_ok") is not None
        if has_overlay and not has_text:
            missing.append(str(rec.get("id") or p.stem))
    return missing


def _wire_eval_path(configs: Any, ummmu_root: Path, model_name: str) -> Path:
    eval_path = ummmu_root / "outputs" / "_eval" / model_name
    configs.eval_path = str(eval_path)
    configs.geometry["out_eval_dir"] = str(eval_path / "math")
    configs.jigsaw["eval_dir"] = str(eval_path / "jigsaw")
    configs.science["eval_root"] = str(eval_path / "science")
    configs.code_svg["eval_out_dir"] = str(eval_path / "code")
    configs.maze["out_root"] = str(eval_path / "maze")
    configs.sliding_puzzle["out_root"] = str(eval_path / "sliding")
    return eval_path


def _forbid_vl(eu) -> None:
    def _raise(*_a, **_k):
        raise RuntimeError(
            "eval_geom_text_missing.py refused to load Qwen2.5-VL. "
            "This script only runs the geometry text judge."
        )

    if hasattr(eu, "LocalVL"):
        eu.LocalVL.__init__ = _raise  # type: ignore[method-assign]
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        Qwen2_5_VLForConditionalGeneration.from_pretrained = _raise
    except Exception:
        pass


def _patch_qwen3_awq_load(eu) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _init(self, model_name: str) -> None:  # noqa: ANN001
        from patch_awq_triton import prepare_awq_model

        self.model_name = model_name
        print(f"[lm] loading {model_name} (preserve int32 qweight)", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto"
        )
        prepare_awq_model(self.model, label="qwen3")

    eu.LocalTextLM.__init__ = _init  # type: ignore[method-assign]


def _install_write_guard(eu, eval_root: Path) -> None:
    allowed_dirs = {(eval_root / "math").resolve()}
    orig = eu.UtilityHelpers.write_json

    @staticmethod
    def guarded(obj: Any, path) -> None:
        p = Path(path).resolve()
        for root in allowed_dirs:
            try:
                p.relative_to(root)
                return orig(obj, path)
            except ValueError:
                continue
        raise RuntimeError(f"blocked write outside math eval dir: {p}")

    eu.UtilityHelpers.write_json = guarded  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ummmu-root", required=True)
    parser.add_argument("--model-name", default="bagel-zebra-cot")
    parser.add_argument("--patched-eval", default="eval_ummmu_patched.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ummmu_root = Path(args.ummmu_root).resolve()
    patched = Path(args.patched_eval)
    if not patched.is_file():
        patched = ummmu_root / args.patched_eval
    if not patched.is_file():
        raise SystemExit(f"patched eval file not found: {patched}")

    sys.path.insert(0, str(ummmu_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    try:
        from patch_awq_triton import apply_awq_triton_patch

        apply_awq_triton_patch()
    except Exception as e:
        print(f"[awq] patch failed (non-fatal if Qwen3 still loads): {e}", flush=True)

    # Register in sys.modules before exec: @dataclass looks up cls.__module__.
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_ummmu_patched", patched)
    eu = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["eval_ummmu_patched"] = eu
    spec.loader.exec_module(eu)
    _forbid_vl(eu)

    configs = eu.Configurations(
        model_name=args.model_name,
        base_path=str(ummmu_root),
        qwen3_model_name=os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-32B-AWQ"),
    )
    eval_root = _wire_eval_path(configs, ummmu_root, args.model_name)
    items_dir = eval_root / "math" / "items"
    if not items_dir.is_dir():
        raise SystemExit(f"missing {items_dir} — restore the outputs branch first")

    missing = _missing_text_ids(items_dir)
    print(
        f"=== geometry text-only judge: {len(missing)} items lack text_ok ===",
        flush=True,
    )
    for i, mid in enumerate(missing, 1):
        print(f"  {i:02d}. {mid}", flush=True)

    for rel in VL_PROTECTED_REL:
        if not (eval_root / rel).is_file():
            raise SystemExit(f"VL artifact missing: {eval_root / rel}")

    protected = _collect_protected_files(eval_root)
    fp_before = _fingerprint_files(protected)
    overlay_before = _overlay_fingerprint(items_dir)

    if args.dry_run:
        print("[dry-run] not loading Qwen3", flush=True)
        write_paper_table(eval_root, args.model_name)
        return

    if not missing:
        print("Nothing to score; rewriting summaries from existing items.", flush=True)
        fix_geometry_status(eval_root / "math")
        write_paper_table(eval_root, args.model_name)
        return

    _install_write_guard(eu, eval_root)
    _patch_qwen3_awq_load(eu)

    print("=" * 20 + " Loading text judge (Qwen3) " + "=" * 20, flush=True)
    lm = eu.LocalTextLM(configs.qwen3_model_name)
    configs.geometry["geometry_phase"] = "text"
    geometry_evaluator = eu.GeometryEvaluator(configs.geometry, lm=lm, vl=None)
    geometry_evaluator.evaluate()
    geometry_evaluator.summarize()
    del lm

    still = _missing_text_ids(items_dir)
    print(f"[geometry] still missing text_ok after judge: {len(still)}", flush=True)
    if still:
        print("  " + ", ".join(still), flush=True)

    fix_geometry_status(eval_root / "math")

    fp_after = _fingerprint_files(protected)
    if fp_before != fp_after:
        changed = [
            p
            for p in sorted(set(fp_before) | set(fp_after))
            if fp_before.get(p) != fp_after.get(p)
        ]
        raise SystemExit("science/code VL artifacts changed:\n  " + "\n  ".join(changed))
    overlay_after = _overlay_fingerprint(items_dir)
    if overlay_before != overlay_after:
        raise SystemExit("geometry overlay judgments changed; aborting")

    write_paper_table(eval_root, args.model_name)
    print("Geometry text-only eval completed. Overlay/science/code unchanged.", flush=True)


if __name__ == "__main__":
    main()
