# -*- coding: utf-8 -*-
"""Finish Uni-MMMU eval without loading Qwen2.5-VL or Qwen3.

1. Fix geometry/math item status (overlay_done -> ok when text_ok exists)
   and rewrite math/eval_summary.json from the existing per-item JSON.
2. Run rule-based judges only: jigsaw (DreamSim), maze, sliding.
3. Write all_tasks_summary_*.xlsx from the already-finished VL summaries
   plus the new rule-based summaries.

Never instantiates LocalVL / LocalTextLM / GeometryEvaluator /
ScienceEvaluator / CodeSVGEvaluator. Science and SVG/code artifacts are
read-only; a write guard + fingerprint check abort if they change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple


JUDGMENT_KEYS = (
    "overlay_ok",
    "overlay_reason",
    "text_ok",
    "text_reason",
    "reasoning_rigorous",
    "conclusion_correct",
)

VL_PROTECTED_REL = (
    "science/eval_summary.json",
    "science/eval_details.json",
    "code/eval_summary.json",
    "code/eval_details.json",
)


def _stub_qwen_vl_utils() -> None:
    if "qwen_vl_utils" in sys.modules:
        return
    try:
        import qwen_vl_utils  # noqa: F401
    except Exception:
        stub = ModuleType("qwen_vl_utils")
        stub.process_vision_info = lambda *a, **k: (None, None)  # type: ignore[attr-defined]
        sys.modules["qwen_vl_utils"] = stub


def _forbid_pretrained(name: str):
    def _raise(*_a, **_k):
        raise RuntimeError(
            f"finalize_eval.py refused to load {name}. "
            "This script must not instantiate Qwen2.5-VL or Qwen3."
        )

    return _raise


def _import_eval_ummmu(ummmu_root: Path):
    _stub_qwen_vl_utils()
    sys.path.insert(0, str(ummmu_root))
    import eval_ummmu as eu  # type: ignore

    try:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    except Exception:
        AutoModelForCausalLM = None
        AutoModelForImageTextToText = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except Exception:
        Qwen2_5_VLForConditionalGeneration = None

    if AutoModelForCausalLM is not None:
        AutoModelForCausalLM.from_pretrained = _forbid_pretrained("AutoModelForCausalLM")
    if AutoModelForImageTextToText is not None:
        AutoModelForImageTextToText.from_pretrained = _forbid_pretrained(
            "AutoModelForImageTextToText"
        )
    if Qwen2_5_VLForConditionalGeneration is not None:
        Qwen2_5_VLForConditionalGeneration.from_pretrained = _forbid_pretrained(
            "Qwen2.5-VL"
        )
    if hasattr(eu, "LocalVL"):
        eu.LocalVL.__init__ = _forbid_pretrained("LocalVL")  # type: ignore[method-assign]
    if hasattr(eu, "LocalTextLM"):
        eu.LocalTextLM.__init__ = _forbid_pretrained("LocalTextLM")  # type: ignore[method-assign]
    for cls_name in ("GeometryEvaluator", "ScienceEvaluator", "CodeSVGEvaluator"):
        cls = getattr(eu, cls_name, None)
        if cls is not None:
            cls.__init__ = _forbid_pretrained(cls_name)
    return eu


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_json_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_protected_files(eval_root: Path) -> List[Path]:
    files: List[Path] = []
    for rel in VL_PROTECTED_REL:
        p = eval_root / rel
        if p.is_file():
            files.append(p)
    for sub in ("science", "code"):
        items = eval_root / sub / "items"
        if items.is_dir():
            files.extend(sorted(items.glob("*.json")))
    return files


def _fingerprint_files(paths: Iterable[Path]) -> Dict[str, str]:
    return {str(p): _sha256_file(p) for p in paths}


def _math_judgment_fingerprint(items_dir: Path) -> str:
    parts: List[str] = []
    for p in sorted(items_dir.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        parts.append(
            _sha256_json_obj(
                {"id": rec.get("id") or p.stem, **{k: rec.get(k) for k in JUDGMENT_KEYS}}
            )
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _geom_scored(r: Dict[str, Any]) -> bool:
    if r.get("status") == "ok":
        return True
    return r.get("overlay_ok") is not None and "text_ok" in r


def _text_looks_like_api_error(rec: Dict[str, Any]) -> bool:
    reason = str(rec.get("text_reason") or "")
    return (
        reason.startswith("API Error")
        or "IncompatibleTypeError" in reason
        or "rshift_cuda" in reason
    )


def fix_geometry_status(eval_math_dir: Path) -> Tuple[List[Dict[str, Any]], int]:
    items_dir = eval_math_dir / "items"
    if not items_dir.is_dir():
        raise SystemExit(f"geometry items directory missing: {items_dir}")

    results: List[Dict[str, Any]] = []
    n_fixed = 0
    n_complete = 0
    for path in sorted(items_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        complete = rec.get("overlay_ok") is not None and "text_ok" in rec
        if complete:
            n_complete += 1
            if rec.get("status") != "ok":
                rec["status"] = "ok"
                path.write_text(
                    json.dumps(rec, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                n_fixed += 1
        results.append(rec)

    if not results:
        raise SystemExit(f"no geometry item JSON under {items_dir}")

    scored = [r for r in results if _geom_scored(r)]
    summary = {
        "total_items": len(results),
        "scored_items": len(scored),
        "ok_overlay": sum(int(r.get("overlay_ok") or 0) for r in results),
        "ok_text": sum(int(r.get("text_ok") or 0) for r in results),
        "status_flipped_to_ok": n_fixed,
        "note": "Rewritten from per-item JSON without re-running VL/Qwen3 judges.",
    }
    summary["overlay_acc"] = (
        summary["ok_overlay"] / summary["scored_items"] if summary["scored_items"] else 0.0
    )
    summary["text_acc"] = (
        summary["ok_text"] / summary["scored_items"] if summary["scored_items"] else 0.0
    )
    eval_math_dir.mkdir(parents=True, exist_ok=True)
    (eval_math_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (eval_math_dir / "eval_details.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    n_api = sum(1 for r in results if _text_looks_like_api_error(r))
    print("\n=== Geometry status repair (no judges) ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(
        f"[geometry] items={len(results)} complete={n_complete} "
        f"status_flipped_to_ok={n_fixed} text_api_error_rows={n_api}",
        flush=True,
    )
    return results, n_fixed


def _install_write_guard(eu, eval_root: Path) -> None:
    allowed_dirs = {
        (eval_root / "jigsaw").resolve(),
        (eval_root / "maze").resolve(),
        (eval_root / "sliding").resolve(),
    }
    allowed_files = {
        (eval_root / "math" / "eval_summary.json").resolve(),
        (eval_root / "math" / "eval_details.json").resolve(),
    }
    orig = eu.UtilityHelpers.write_json

    @staticmethod
    def guarded(obj: Any, path) -> None:
        p = Path(path).resolve()
        if p in allowed_files:
            return orig(obj, path)
        for root in allowed_dirs:
            try:
                p.relative_to(root)
                return orig(obj, path)
            except ValueError:
                continue
        raise RuntimeError(
            f"finalize_eval.py blocked a write outside the rule-based allowlist: {p}"
        )

    eu.UtilityHelpers.write_json = guarded  # type: ignore[method-assign]


def _jigsaw_run_ids(run_summary: Dict[str, Any], out_root: Path, item_index: Dict[str, Any]) -> List[str]:
    per_item = run_summary.get("per_item")
    run_ids: List[str] = []
    if isinstance(per_item, dict):
        run_ids = [str(k) for k in per_item.keys()]
    elif isinstance(per_item, list):
        for x in per_item:
            if isinstance(x, dict) and x.get("id"):
                run_ids.append(str(x["id"]))
            elif isinstance(x, str):
                run_ids.append(x)
    if not run_ids and out_root.is_dir():
        run_ids = sorted(
            p.name
            for p in out_root.iterdir()
            if p.is_dir() and (p / "model_text.txt").is_file()
        )
    if not run_ids:
        run_ids = list(item_index.keys())
    return run_ids


def _resolve_existing(raw: Any, *roots: Path) -> Path:
    path = Path(str(raw))
    if path.is_file():
        return path
    for root in roots:
        cand = root / path
        if cand.is_file():
            return cand
        if not path.is_absolute():
            cand2 = root / str(raw)
            if cand2.is_file():
                return cand2
    return path


def _patch_jigsaw_evaluator(eu, base_path: Path) -> None:
    def evaluate(self) -> None:  # noqa: ANN001
        import glob
        from PIL import Image
        from tqdm import tqdm

        meta = self.utils.read_json(Path(self.config["dataset_dir"]) / "metadata.json")
        item_index = {item["id"]: item for item in meta.get("items", [])}
        out_root = Path(self.config["out_root"])
        eval_dir = Path(self.config["eval_dir"])
        eval_dir.mkdir(parents=True, exist_ok=True)
        run_summary = self.utils.read_json(out_root / "summary.json")
        run_ids = _jigsaw_run_ids(run_summary, out_root, item_index)
        items_to_eval = [item_index[i] for i in run_ids if i in item_index]

        max_items = self.config.get("max_items_per_task")
        if max_items is not None and max_items > 0:
            print(f"[INFO] Limiting to the first {max_items} jigsaw items.")
            items_to_eval = items_to_eval[:max_items]

        dataset_dir = Path(self.config["dataset_dir"])
        n_skip = 0
        for item in tqdm(items_to_eval, desc="Evaluating Jigsaw"):
            item_id = item["id"]
            cached_path = eval_dir / "items" / f"{item_id}.json"
            if cached_path.is_file():
                try:
                    cached = json.loads(cached_path.read_text(encoding="utf-8"))
                    if cached.get("id") == item_id:
                        self.results.append(cached)
                        n_skip += 1
                        continue
                except Exception:
                    pass

            case_dir = out_root / item_id
            record: Dict[str, Any] = {"id": item_id, "label": item["label"]}

            choice, text_status = self._parse_choice_from_text(case_dir / "model_text.txt")
            record["text_status"] = text_status
            record["choice"] = choice
            record["text_correct"] = int(choice == item["label"]) if choice is not None else 0

            gt_ok_path = _resolve_existing(
                item["gt_completed_2x2_path"], dataset_dir, base_path
            )
            gt_bad_path = _resolve_existing(
                item["gt_wrong_2x2_path"], dataset_dir, base_path
            )
            gt_c0_path, gt_c1_path = (
                (gt_ok_path, gt_bad_path) if item["label"] == 0 else (gt_bad_path, gt_ok_path)
            )

            pred_imgs = sorted(glob.glob(str(case_dir / "model_image_*.*")))
            if len(pred_imgs) == 2:
                try:
                    im0 = Image.open(pred_imgs[0])
                    im1 = Image.open(pred_imgs[1])
                    gt0 = Image.open(gt_c0_path)
                    gt1 = Image.open(gt_c1_path)
                    d0 = self._dreamsim_distance(im0, gt0)
                    d1 = self._dreamsim_distance(im1, gt1)
                    record.update(
                        {
                            "image_status": "ok",
                            "d0": d0,
                            "d1": d1,
                            "d_mean": (d0 + d1) / 2.0,
                        }
                    )
                except Exception as e:
                    record.update(
                        {
                            "image_status": f"read_error: {e}",
                            "d_mean": self.config["penalty_distance"],
                        }
                    )
            else:
                record.update(
                    {
                        "image_status": f"invalid_count_{len(pred_imgs)}",
                        "d_mean": self.config["penalty_distance"],
                    }
                )
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            cached_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            self.results.append(record)
        print(f"[jigsaw] resumed {n_skip} cached items; scored {len(self.results)} total", flush=True)

    eu.JigsawEvaluator.evaluate = evaluate  # type: ignore[method-assign]


def _require_vl_artifacts(eval_root: Path) -> None:
    missing = []
    for rel in ("science/eval_summary.json", "code/eval_summary.json"):
        if not (eval_root / rel).is_file():
            missing.append(str(eval_root / rel))
    items = eval_root / "math" / "items"
    if not items.is_dir() or not any(items.glob("*.json")):
        missing.append(str(items))
    if missing:
        raise SystemExit(
            "VL-judge artifacts are missing. Restore the GitHub `outputs` branch "
            "before this script, and do not run run_eval.sh.\n  "
            + "\n  ".join(missing)
        )


def _wire_eval_path(configs: Any, ummmu_root: Path, model_name: str, dreamsim_cache: Path) -> Path:
    eval_path = ummmu_root / "outputs" / "_eval" / model_name
    configs.eval_path = str(eval_path)
    configs.geometry["out_eval_dir"] = str(eval_path / "math")
    configs.jigsaw["eval_dir"] = str(eval_path / "jigsaw")
    configs.jigsaw["dreamsim_cache"] = str(dreamsim_cache)
    configs.science["eval_root"] = str(eval_path / "science")
    configs.code_svg["eval_out_dir"] = str(eval_path / "code")
    configs.maze["out_root"] = str(eval_path / "maze")
    configs.sliding_puzzle["out_root"] = str(eval_path / "sliding")
    return eval_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix geometry status and run jigsaw/maze/sliding (no VL/Qwen3)."
    )
    parser.add_argument("--ummmu-root", required=True)
    parser.add_argument("--model-name", default="bagel-zebra-cot")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    ummmu_root = Path(args.ummmu_root).resolve()
    work_dir = Path(args.work_dir or os.environ.get("WORK_DIR") or ummmu_root.parent).resolve()
    dreamsim_cache = Path(
        os.environ.get("DREAMSIM_CACHE") or (work_dir / ".cache" / "dreamsim")
    )
    dreamsim_cache.mkdir(parents=True, exist_ok=True)

    print("=== finalize_eval.py (rule-based leftover; no Qwen2.5-VL / Qwen3) ===", flush=True)
    print(f"UMMMU_ROOT={ummmu_root}", flush=True)
    print(f"model={args.model_name}", flush=True)
    print(f"dreamsim_cache={dreamsim_cache}", flush=True)

    eu = _import_eval_ummmu(ummmu_root)
    configs = eu.Configurations(
        model_name=args.model_name,
        base_path=str(ummmu_root),
        max_items_per_task=args.max_items,
    )
    eval_root = _wire_eval_path(configs, ummmu_root, args.model_name, dreamsim_cache)
    print(f"eval artifacts -> {eval_root}", flush=True)

    _require_vl_artifacts(eval_root)
    protected = _collect_protected_files(eval_root)
    fp_before = _fingerprint_files(protected)
    math_fp_before = _math_judgment_fingerprint(eval_root / "math" / "items")
    print(
        f"[guard] fingerprint {len(protected)} science/code files; "
        f"math judgment hash={math_fp_before[:12]}",
        flush=True,
    )

    _install_write_guard(eu, eval_root)
    _patch_jigsaw_evaluator(eu, ummmu_root)

    fix_geometry_status(eval_root / "math")

    jigsaw_output_path = Path(configs.jigsaw["out_root"])
    print("\n" + "=" * 20 + " Starting Jigsaw Evaluation " + "=" * 20, flush=True)
    if jigsaw_output_path.is_dir():
        jigsaw_evaluator = eu.JigsawEvaluator(configs.jigsaw)
        jigsaw_evaluator.evaluate()
        jigsaw_evaluator.summarize()
    else:
        print(f"Skipping Jigsaw: Output directory not found at {jigsaw_output_path}", flush=True)

    maze_output_path = Path(configs.maze["run_root"])
    print("\n" + "=" * 20 + " Starting Maze Evaluation " + "=" * 20, flush=True)
    if maze_output_path.is_dir():
        maze_evaluator = eu.MazeEvaluator(configs.maze)
        maze_evaluator.evaluate()
        maze_evaluator.summarize()
    else:
        print(f"Skipping Maze: Output directory not found at {maze_output_path}", flush=True)

    sliding_output_path = Path(configs.sliding_puzzle["run_root"])
    print("\n" + "=" * 20 + " Starting Sliding Puzzle Evaluation " + "=" * 20, flush=True)
    if sliding_output_path.is_dir():
        sliding_evaluator = eu.SlidingPuzzleEvaluator(configs.sliding_puzzle)
        sliding_evaluator.evaluate()
        sliding_evaluator.summarize()
    else:
        print(
            f"Skipping Sliding Puzzle: Output directory not found at {sliding_output_path}",
            flush=True,
        )

    eu.summarize_all_tasks(configs)

    fp_after = _fingerprint_files(protected)
    if fp_before != fp_after:
        changed = [p for p in sorted(set(fp_before) | set(fp_after)) if fp_before.get(p) != fp_after.get(p)]
        raise SystemExit(
            "Refusing to continue: science/code VL artifacts changed.\n  " + "\n  ".join(changed)
        )
    math_fp_after = _math_judgment_fingerprint(eval_root / "math" / "items")
    if math_fp_before != math_fp_after:
        raise SystemExit(
            "Refusing to continue: geometry overlay/text judgments changed. "
            "This script may only flip status and rewrite summaries."
        )

    xlsx = eval_root / f"all_tasks_summary_{args.model_name}.xlsx"
    print("\nRule-based leftover eval completed.", flush=True)
    print(f"Excel: {xlsx}", flush=True)
    print("VL science/code fingerprints unchanged; geometry judgments unchanged.", flush=True)


if __name__ == "__main__":
    main()
