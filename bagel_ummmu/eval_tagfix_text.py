# -*- coding: utf-8 -*-
"""Re-score tag-fixed texts only. Copy image/overlay metrics from the original
eval folder. Writes under outputs/_eval/<dst-model>/. Never writes the original
_eval/bagel-zebra-cot tree.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from report_ummmu_table import write_paper_table

RE_RENDER = re.compile(r"<RENDER_SUMMARY>\s*(.*?)\s*</RENDER_SUMMARY>", re.I | re.S)
RE_FINAL = re.compile(r"<FINAL_ANSWER_JSON>\s*(\{.*?\})\s*</FINAL_ANSWER_JSON>", re.I | re.S)
RE_ANSWER = re.compile(r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>", re.I | re.S)


def _sync(eu, force: bool = False) -> None:
    fn = getattr(eu.UtilityHelpers, "eval_sync_maybe", None)
    if callable(fn):
        fn(force=force)


def _out_of_time(eu) -> bool:
    fn = getattr(eu.UtilityHelpers, "eval_out_of_time", None)
    return bool(callable(fn) and fn())


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_patched(path: Path):
    spec = importlib.util.spec_from_file_location("eval_ummmu_patched", path)
    eu = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["eval_ummmu_patched"] = eu
    spec.loader.exec_module(eu)
    return eu


def _wire(configs: Any, ummmu_root: Path, model_name: str) -> Path:
    eval_path = ummmu_root / "outputs" / "_eval" / model_name
    configs.eval_path = str(eval_path)
    configs.outputs_path = str(ummmu_root / "outputs" / model_name)
    configs.geometry["out_dir"] = f"{configs.outputs_path}/math"
    configs.geometry["out_eval_dir"] = str(eval_path / "math")
    configs.jigsaw["out_root"] = f"{configs.outputs_path}/jigsaw"
    configs.jigsaw["eval_dir"] = str(eval_path / "jigsaw")
    configs.science["run_root"] = f"{configs.outputs_path}/science"
    configs.science["eval_root"] = str(eval_path / "science")
    configs.code_svg["sample_root"] = f"{configs.outputs_path}/code"
    configs.code_svg["eval_out_dir"] = str(eval_path / "code")
    configs.maze["run_root"] = f"{configs.outputs_path}/maze"
    configs.maze["out_root"] = str(eval_path / "maze")
    configs.sliding_puzzle["run_root"] = f"{configs.outputs_path}/sliding"
    configs.sliding_puzzle["out_root"] = str(eval_path / "sliding")
    return eval_path


def _parse_jigsaw_choice(text: str):
    m = RE_FINAL.search(text or "")
    if not m:
        return None, "no_final_json_block"
    try:
        data = json.loads(m.group(1))
    except Exception:
        return None, "bad_json"
    choice = data.get("choice")
    if choice in (0, 1):
        return int(choice), "ok"
    return None, "invalid_choice_value"


def _parse_moves(text: str) -> List[str]:
    matches = list(RE_ANSWER.finditer(text or ""))
    if not matches:
        return []
    try:
        arr = json.loads(matches[-1].group(1))
        return [str(x).strip().lower() for x in arr]
    except Exception:
        return []


def _text_move_scores(pred: List[str], gt: List[str]) -> Dict[str, float]:
    if not gt:
        return {"text_exact": 0, "text_frame_acc": 0.0}
    match_count = sum(1 for p, g in zip(pred, gt) if p == g)
    return {
        "text_exact": 1 if pred == gt else 0,
        "text_frame_acc": match_count / len(gt),
    }


def _copy_tree_resume(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        q = dst / p.relative_to(src)
        if q.is_file():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)


def _copy_math(orig_eval: Path, new_eval: Path) -> None:
    src = orig_eval / "math"
    dst = new_eval / "math"
    if not src.is_dir():
        print("[math] original eval missing; skipping copy", flush=True)
        return
    _copy_tree_resume(src, dst)
    print(f"[math] copied overlay+text eval from {src}", flush=True)


def _resolve_gt(path_str: Optional[str], ummmu_root: Path) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(str(path_str))
    if p.is_file():
        return p
    s = str(path_str).replace("\\", "/")
    marker = "Uni-MMMU/"
    if marker in s:
        q = ummmu_root / s.split(marker, 1)[1]
        if q.is_file():
            return q
    return None


def eval_jigsaw(eu, configs, orig_eval: Path, new_eval: Path) -> None:
    out_root = Path(configs.jigsaw["out_root"])
    eval_dir = Path(configs.jigsaw["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    orig_items = orig_eval / "jigsaw" / "items"
    meta = eu.UtilityHelpers.read_json(Path(configs.jigsaw["dataset_dir"]) / "metadata.json")
    item_index = {item["id"]: item for item in meta.get("items", [])}
    ids = sorted(p.stem for p in orig_items.glob("*.json")) if orig_items.is_dir() else []
    if not ids and out_root.is_dir():
        ids = sorted(p.name for p in out_root.iterdir() if p.is_dir())
    results = []
    n_skip = 0
    for item_id in ids:
        cached = eval_dir / "items" / f"{item_id}.json"
        if cached.is_file():
            rec = _read_json(cached)
            if rec:
                results.append(rec)
                n_skip += 1
                continue
        orig = _read_json(orig_items / f"{item_id}.json") if orig_items.is_dir() else {}
        item = item_index.get(item_id) or {}
        text_path = out_root / item_id / "model_text.txt"
        text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.is_file() else ""
        choice, text_status = _parse_jigsaw_choice(text)
        label = item.get("label", orig.get("label"))
        rec = dict(orig) if orig else {"id": item_id, "label": label}
        rec["id"] = item_id
        rec["text_status"] = text_status
        rec["choice"] = choice
        rec["text_correct"] = int(choice == label) if choice is not None and label is not None else 0
        rec["image_from"] = "copied_from_original_eval"
        rec["text_from"] = "tagfix"
        _write_json(rec, cached)
        results.append(rec)
        _sync(eu)
    print(f"[jigsaw] scored {len(results)} (resumed {n_skip})", flush=True)
    if not results:
        return
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(eval_dir / "per_item.csv", index=False)
    if "d_mean" in df.columns:
        penalized = [float(x) for x in df["d_mean"].tolist() if pd.notna(x)]
    else:
        penalized = []
    summary = {
        "counts": {
            "total_items": len(df),
            "image_ok": int((df["image_status"] == "ok").sum()) if "image_status" in df.columns else 0,
            "text_parsed": int((df["text_status"] == "ok").sum()),
        },
        "metrics": {
            "mean_distance_penalized": float(np.mean(penalized)) if penalized else None,
            "image_score_penalized": float(1.0 - np.mean(penalized)) if penalized else None,
            "text_accuracy": float(df["text_correct"].mean()) if len(df) else 0.0,
        },
        "note": "Image distances copied from bagel-zebra-cot; text re-parsed on tag-fixed files.",
    }
    _write_json(summary, eval_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


def eval_maze_like(task: str, eu, configs, orig_eval: Path, ummmu_root: Path) -> None:
    cfg = configs.maze if task == "maze" else configs.sliding_puzzle
    run_root = Path(cfg["run_root"])
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    orig_csv = orig_eval / ("maze" if task == "maze" else "sliding") / "manifest_eval.csv"
    if not orig_csv.is_file():
        print(f"[{task}] original manifest missing at {orig_csv}", flush=True)
        return
    with orig_csv.open(encoding="utf-8") as f:
        orig_rows = list(csv.DictReader(f))
    evaluator = eu.MazeEvaluator(cfg) if task == "maze" else eu.SlidingPuzzleEvaluator(cfg)
    results = []
    for row in orig_rows:
        rec = dict(row)
        case_id = rec.get("case_id")
        cand_idx = int(float(rec.get("cand_idx") or 1))
        case_dir = run_root / str(case_id)
        raw = ""
        text_path = case_dir / "model_text.txt"
        if text_path.is_file():
            parts = text_path.read_text(encoding="utf-8", errors="replace").split("\n\n\n-----\n\n\n")
            raw = parts[cand_idx - 1] if cand_idx - 1 < len(parts) else parts[-1]
        pred = _parse_moves(raw)
        orig_json = eu.UtilityHelpers.read_json(case_dir / "result.json")
        gt_moves: List[str] = []
        if task == "maze":
            step0 = orig_json.get("step0")
            gt_path = evaluator._derive_gt_json_path(step0) if step0 else None
            gt_file = _resolve_gt(str(gt_path) if gt_path else None, ummmu_root)
            if gt_file:
                gt_moves = [str(m).lower() for m in eu.UtilityHelpers.read_json(gt_file).get("steps_long", [])]
        else:
            step0 = orig_json.get("init_png")
            gt_json, _ = evaluator._derive_gt_files(step0) if step0 else (None, None)
            gt_file = _resolve_gt(str(gt_json) if gt_json else None, ummmu_root)
            if gt_file:
                gt_moves = [str(m).lower() for m in eu.UtilityHelpers.read_json(gt_file).get("steps_words", [])]
        scores = _text_move_scores(pred, gt_moves)
        rec["text_exact"] = scores["text_exact"]
        rec["text_frame_acc"] = scores["text_frame_acc"]
        rec["text_from"] = "tagfix"
        rec["image_from"] = "copied_from_original_eval"
        results.append(rec)
    import pandas as pd

    df = pd.DataFrame(results)
    summary = {
        "total_candidates": len(df),
        "text_accuracy_exact": float(df["text_exact"].astype(float).mean()) if len(df) else 0.0,
        "text_accuracy_frame_macro": float(df["text_frame_acc"].astype(float).mean()) if len(df) else 0.0,
        "note": "Image columns copied from original eval; text recomputed on tag-fixed files.",
    }
    if "img_exact" in df.columns:
        summary["img_accuracy_exact"] = float(df["img_exact"].astype(float).mean())
        summary["img_accuracy_frame_macro"] = float(df["img_frame_acc"].astype(float).mean())
    if task == "maze" and "parse_all_ok" in df.columns:
        summary["parse_accuracy_all_ok"] = float(df["parse_all_ok"].astype(float).mean())
        summary["parse_accuracy_frame_macro"] = float(df["parse_frame_success"].astype(float).mean())
    _write_json(summary, out_root / "summary.json")
    df.to_csv(out_root / "manifest_eval.csv", index=False)
    print(f"[{task}] {json.dumps(summary, indent=2)}", flush=True)


def eval_science_text(eu, configs, orig_eval: Path, vl) -> bool:
    eval_root = Path(configs.science["eval_root"])
    run_root = Path(configs.science["run_root"])
    orig_items = orig_eval / "science" / "items"
    evaluator = eu.ScienceEvaluator(configs.science, vl=vl)
    cases = eu.UtilityHelpers.read_json(configs.science["data_json"])
    flat = [sample for block in cases for sample in block.get("samples", [])]
    results = []
    n_skip = 0
    timed_out = False
    for idx, case in enumerate(flat, 1):
        if _out_of_time(eu):
            timed_out = True
            print(f"[science] time budget hit at {idx}/{len(flat)}; resume next session", flush=True)
            break
        cid = f"case_{idx:02d}"
        cached = eval_root / "items" / f"{cid}.json"
        if cached.is_file():
            rec = _read_json(cached)
            blob = json.dumps(rec)
            if rec and "API Error" not in blob:
                results.append(rec)
                n_skip += 1
                continue
        orig = _read_json(orig_items / f"{cid}.json")
        pred_path = run_root / cid / "model_text.txt"
        pred_text = pred_path.read_text(encoding="utf-8", errors="replace") if pred_path.is_file() else ""
        gt_text = case.get("output_prompt") or ""
        condition = case.get("input_prompt")
        text_eval = evaluator._evaluate_text("", condition, gt_text, pred_text)
        rec = {
            "case_id": cid,
            "text_eval": text_eval,
            "image_eval": orig.get("image_eval") or {"image_correct": 0},
            "image_from": "copied_from_original_eval",
            "text_from": "tagfix",
        }
        _write_json(rec, cached)
        results.append(rec)
        _sync(eu)
        if idx % 10 == 0:
            print(f"[science] {idx}/{len(flat)}", flush=True)
    print(f"[science] scored {len(results)} (resumed {n_skip})", flush=True)
    evaluator.results = results
    if results:
        evaluator.summarize()
    return not timed_out and len(results) == len(flat)


def eval_svg_text(eu, configs, orig_eval: Path, vl) -> bool:
    eval_dir = Path(configs.code_svg["eval_out_dir"])
    sample_root = Path(configs.code_svg["sample_root"])
    orig_items = orig_eval / "code" / "items"
    evaluator = eu.CodeSVGEvaluator(configs.code_svg, vl=vl)
    case_dirs = sorted(sample_root.glob("case_*_*"))
    results = []
    n_skip = 0
    timed_out = False
    for case_dir in case_dirs:
        if _out_of_time(eu):
            timed_out = True
            print(f"[svg] time budget hit at {len(results)}/{len(case_dirs)}; resume next session", flush=True)
            break
        cid = case_dir.name
        cached = eval_dir / "items" / f"{cid}.json"
        if cached.is_file():
            rec = _read_json(cached)
            blob = json.dumps(rec)
            if rec and "API Error" not in blob:
                results.append(rec)
                n_skip += 1
                continue
        orig = _read_json(orig_items / f"{cid}.json")
        full_path = case_dir / "model_text.txt"
        full_text = full_path.read_text(encoding="utf-8", errors="replace") if full_path.is_file() else ""
        m = RE_RENDER.search(full_text)
        render_summary = m.group(1).strip() if m else ""
        ref_img = evaluator._get_reference_image(case_dir)
        image_evals = orig.get("image_evals") or []
        if ref_img:
            text_eval = evaluator._eval_text(render_summary, ref_img)
        else:
            text_eval = {"text_semantic_match": 0, "explanation": "no_reference_image"}
        rec = {
            "case_dir": str(case_dir),
            "image_evals": image_evals,
            "text_eval": text_eval,
            "image_from": "copied_from_original_eval",
            "text_from": "tagfix",
        }
        _write_json(rec, cached)
        results.append(rec)
        _sync(eu)
        if len(results) % 10 == 0:
            print(f"[svg] {len(results)}/{len(case_dirs)}", flush=True)
    print(f"[svg] scored {len(results)} (resumed {n_skip})", flush=True)
    evaluator.results = results
    if results:
        evaluator.summarize()
    return not timed_out and len(results) == len(case_dirs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ummmu-root", required=True)
    parser.add_argument("--src-model", default="bagel-zebra-cot")
    parser.add_argument("--dst-model", default="bagel-zebra-cot-tagfix")
    parser.add_argument("--patched-eval", default="eval_ummmu_patched.py")
    parser.add_argument("--skip-vl", action="store_true", help="jigsaw/maze/sliding/math only")
    args = parser.parse_args()

    ummmu_root = Path(args.ummmu_root).resolve()
    orig_eval = ummmu_root / "outputs" / "_eval" / args.src_model
    tagfix_out = ummmu_root / "outputs" / args.dst_model
    if not tagfix_out.is_dir():
        raise SystemExit(f"tag-fixed sampling missing: {tagfix_out} (run run_tagfix.sh first)")
    if not orig_eval.is_dir():
        raise SystemExit(f"original eval missing: {orig_eval}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(ummmu_root))
    try:
        from patch_awq_triton import apply_awq_triton_patch

        apply_awq_triton_patch()
    except Exception as e:
        print(f"[awq] {e}", flush=True)

    patched = Path(args.patched_eval)
    if not patched.is_file():
        patched = ummmu_root / args.patched_eval
    eu = _load_patched(patched)
    configs = eu.Configurations(model_name=args.dst_model, base_path=str(ummmu_root))
    new_eval = _wire(configs, ummmu_root, args.dst_model)
    new_eval.mkdir(parents=True, exist_ok=True)

    print(f"=== tagfix text eval -> {new_eval} ===", flush=True)
    print("Image/overlay scores copied from original eval. Original _eval tree is not written.", flush=True)

    _copy_math(orig_eval, new_eval)
    eval_jigsaw(eu, configs, orig_eval, new_eval)
    eval_maze_like("maze", eu, configs, orig_eval, ummmu_root)
    eval_maze_like("sliding", eu, configs, orig_eval, ummmu_root)

    complete = True
    if args.skip_vl:
        print("[skip-vl] not loading Qwen2.5-VL; science/SVG text not rescored", flush=True)
        complete = False
    else:
        print("=" * 20 + " Loading VL judge for science/SVG TEXT only " + "=" * 20, flush=True)
        vl = eu.LocalVL(
            os.environ.get("QWEN_VL_MODEL", configs.qwen2_5_vl_model_name),
            attn_implementation=configs.vl_attn_impl or "sdpa",
        )
        complete = eval_science_text(eu, configs, orig_eval, vl) and complete
        if complete:
            complete = eval_svg_text(eu, configs, orig_eval, vl) and complete
        else:
            print("[svg] skipped this session because science did not finish", flush=True)
        del vl

    try:
        eu.summarize_all_tasks(configs)
    except Exception as e:
        print(f"[warn] summarize_all_tasks: {e}", flush=True)
    write_paper_table(new_eval, args.dst_model)
    _sync(eu, force=True)
    if complete:
        print("Tag-fixed text eval completed.", flush=True)
    else:
        print("Tag-fixed text eval partial; re-run run_eval_tagfix.sh to resume.", flush=True)


if __name__ == "__main__":
    main()
