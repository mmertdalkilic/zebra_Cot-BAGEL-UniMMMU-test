# -*- coding: utf-8 -*-
"""Second-pass patch: keep the official VL judge (Qwen2.5-VL-72B-Instruct-AWQ)
but never hold it in VRAM together with Qwen3.

Also force fp16 on the VL AWQ load — `dtype=auto` picks bf16 on Blackwell and
Triton AWQ then crashes, which was recorded as overlay_ok=0.
"""
from __future__ import annotations

import sys
from pathlib import Path


def must_replace(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        raise SystemExit(f"sequential patch {label}: pattern not found")
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"sequential patch {label}: expected 1 occurrence, found {n}")
    return src.replace(old, new, 1)


VL_INIT_OLD = '''        kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
'''

VL_INIT_NEW = '''        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        print(f"[vl] loading {model_name} dtype=float16 attn={attn_implementation!r}", flush=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
'''

# Post-resume geometry loop (apply_eval_resume.py already rewrote the for-loop head).
GEOM_LOOP_HEAD_OLD = '''        self.utils.explode_details_into_items(self.config['out_eval_dir'], ["id"])
        for big_k, small_k, item in tqdm(items, desc="Evaluating Geometry"):
            case_id = f"{self.utils.sanitize_filename(big_k)}__{self.utils.sanitize_filename(small_k)}"
            cached = self.utils.try_load_item(self.config['out_eval_dir'], case_id)
            if cached is not None:
                self.results.append(cached)
                continue
            if self.utils.eval_out_of_time():
                print("[eval] time budget hit during geometry — remaining items resume next session", flush=True)
                break
            case_dir = Path(self.config['out_dir']) / case_id
            record = {"id": case_id, "type": item.get("type"), "status": "init"}
'''

GEOM_LOOP_HEAD_NEW = '''        phase = self.config.get("geometry_phase", "both")
        self.utils.explode_details_into_items(self.config['out_eval_dir'], ["id"])
        for big_k, small_k, item in tqdm(items, desc=f"Evaluating Geometry ({phase})"):
            case_id = f"{self.utils.sanitize_filename(big_k)}__{self.utils.sanitize_filename(small_k)}"
            cached = self.utils.try_load_item(self.config['out_eval_dir'], case_id)
            reason = str((cached or {}).get("overlay_reason") or "")
            overlay_bad = reason.startswith("API Error") or "IncompatibleTypeError" in reason
            overlay_done = bool(cached) and cached.get("overlay_ok") is not None and not overlay_bad
            text_done = bool(cached) and cached.get("status") == "ok" and "text_ok" in cached and not overlay_bad
            if phase == "overlay" and overlay_done:
                self.results.append(cached)
                continue
            if phase == "text" and text_done:
                self.results.append(cached)
                continue
            if phase == "both" and text_done:
                self.results.append(cached)
                continue
            if self.utils.eval_out_of_time():
                print("[eval] time budget hit during geometry — remaining items resume next session", flush=True)
                break
            case_dir = Path(self.config['out_dir']) / case_id
            record = dict(cached) if cached else {"id": case_id, "type": item.get("type"), "status": "init"}
            record["id"] = case_id
'''

GEOM_OVERLAY_TEXT_OLD = '''            pred_imgs = sorted(case_dir.glob("model_image_*.*"))
            cand_text = (case_dir / "model_text.txt").read_text(encoding="utf-8") if (case_dir / "model_text.txt").exists() else ""

            if pred_imgs:
                overlay_eval = self._call_overlay_judge(orig_img, aux_img, pred_imgs[0], item.get("auxiliary_text_en", ""))
                record.update(overlay_eval)
            else:
                record["overlay_ok"] = 0
                record["overlay_reason"] = "No predicted image found."

            problem_text = item.get("problem_text_en") or item.get("problem_text", "")
            gt_text = item.get("solution_en") or item.get("solution", "")
            task_type = "CALCULATION" if (item.get("type") or "").lower().startswith("calc") else "PROVING"
            text_eval = self._call_text_judge(task_type, problem_text, gt_text, cand_text)

            record.update(text_eval)
            record["status"] = "ok"
            self.utils.save_resume_item(self.config['out_eval_dir'], case_id, record)
            self.results.append(record)
'''

GEOM_OVERLAY_TEXT_NEW = '''            if phase in ("overlay", "both") and (self.vl is not None) and (not overlay_done):
                pred_imgs = sorted(case_dir.glob("model_image_*.*"))
                if pred_imgs:
                    overlay_eval = self._call_overlay_judge(orig_img, aux_img, pred_imgs[0], item.get("auxiliary_text_en", ""))
                    record.update(overlay_eval)
                else:
                    record["overlay_ok"] = 0
                    record["overlay_reason"] = "No predicted image found."
                record["status"] = "overlay_done"

            if phase in ("text", "both") and (self.lm is not None):
                cand_text = (case_dir / "model_text.txt").read_text(encoding="utf-8") if (case_dir / "model_text.txt").exists() else ""
                problem_text = item.get("problem_text_en") or item.get("problem_text", "")
                gt_text = item.get("solution_en") or item.get("solution", "")
                task_type = "CALCULATION" if (item.get("type") or "").lower().startswith("calc") else "PROVING"
                text_eval = self._call_text_judge(task_type, problem_text, gt_text, cand_text)
                record.update(text_eval)
                record["status"] = "ok"

            self.utils.save_resume_item(self.config['out_eval_dir'], case_id, record)
            self.results.append(record)
'''

MAIN_OLD = '''    print("="*20 + " Loading Local Models " + "="*20)
    lm = LocalTextLM(configs.qwen3_model_name)
    vl = LocalVL(configs.qwen2_5_vl_model_name, attn_implementation=configs.vl_attn_impl)

    # 1) Geometry
    print("\\n" + "="*20 + " Starting Geometry Evaluation " + "="*20)
    geometry_output_path = Path(configs.geometry['out_dir'])
    if geometry_output_path.is_dir():
        geometry_evaluator = GeometryEvaluator(configs.geometry, lm=lm, vl=vl)
        geometry_evaluator.evaluate()
        geometry_evaluator.summarize()
    else:
        print(f"Skipping Geometry: Output directory not found at {geometry_output_path}")

    # 2) Jigsaw
    print("\\n" + "="*20 + " Starting Jigsaw Evaluation " + "="*20)
    jigsaw_output_path = Path(configs.jigsaw['out_root'])
    if jigsaw_output_path.is_dir():
        jigsaw_evaluator = JigsawEvaluator(configs.jigsaw)
        jigsaw_evaluator.evaluate()
        jigsaw_evaluator.summarize()
    else:
        print(f"Skipping Jigsaw: Output directory not found at {jigsaw_output_path}")

    # 3) Science
    print("\\n" + "="*20 + " Starting Science Evaluation " + "="*20)
    science_output_path = Path(configs.science['run_root'])
    if science_output_path.is_dir():
        science_evaluator = ScienceEvaluator(configs.science, vl=vl)
        science_evaluator.evaluate()
        science_evaluator.summarize()
    else:
        print(f"Skipping Science: Output directory not found at {science_output_path}")

    # 4) SVG
    print("\\n" + "="*20 + " Starting SVG Code Evaluation " + "="*20)
    svg_output_path = Path(configs.code_svg['sample_root'])
    if svg_output_path.is_dir():
        svg_evaluator = CodeSVGEvaluator(configs.code_svg, vl=vl)
        svg_evaluator.evaluate()
        svg_evaluator.summarize()
    else:
        print(f"Skipping SVG: Output directory not found at {svg_output_path}")

    # 5) Maze
    print("\\n" + "="*20 + " Starting Maze Evaluation " + "="*20)
    maze_output_path = Path(configs.maze['run_root'])
    if maze_output_path.is_dir():
        maze_evaluator = MazeEvaluator(configs.maze)
        maze_evaluator.evaluate()
        maze_evaluator.summarize()
    else:
        print(f"Skipping Maze: Output directory not found at {maze_output_path}")

    # 6) Sliding
    print("\\n" + "="*20 + " Starting Sliding Puzzle Evaluation " + "="*20)
    sliding_output_path = Path(configs.sliding_puzzle['run_root'])
    if sliding_output_path.is_dir():
        sliding_puzzle_evaluator = SlidingPuzzleEvaluator(configs.sliding_puzzle)
        sliding_puzzle_evaluator.evaluate()
        sliding_puzzle_evaluator.summarize()
    else:
        print(f"Skipping Sliding Puzzle: Output directory not found at {sliding_output_path}")
'''

MAIN_NEW = '''    def _gpu_flush(*names):
        import gc
        for n in names:
            if n in locals() or n in globals():
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[eval] GPU flushed", flush=True)

    # Official judges, one at a time: VL for overlay/science/code, then Qwen3
    # for geometry text. Same model IDs as the benchmark (AWQ only so 72B fits).
    print("="*20 + " Loading VL judge (Qwen2.5-VL-72B) " + "="*20)
    vl = LocalVL(configs.qwen2_5_vl_model_name, attn_implementation=configs.vl_attn_impl or "sdpa")

    print("\\n" + "="*20 + " Starting Geometry Evaluation (overlay / VL) " + "="*20)
    geometry_output_path = Path(configs.geometry['out_dir'])
    if geometry_output_path.is_dir():
        configs.geometry["geometry_phase"] = "overlay"
        GeometryEvaluator(configs.geometry, lm=None, vl=vl).evaluate()
    else:
        print(f"Skipping Geometry overlay: Output directory not found at {geometry_output_path}")

    print("\\n" + "="*20 + " Starting Science Evaluation " + "="*20)
    science_output_path = Path(configs.science['run_root'])
    if science_output_path.is_dir():
        science_evaluator = ScienceEvaluator(configs.science, vl=vl)
        science_evaluator.evaluate()
        science_evaluator.summarize()
    else:
        print(f"Skipping Science: Output directory not found at {science_output_path}")

    print("\\n" + "="*20 + " Starting SVG Code Evaluation " + "="*20)
    svg_output_path = Path(configs.code_svg['sample_root'])
    if svg_output_path.is_dir():
        svg_evaluator = CodeSVGEvaluator(configs.code_svg, vl=vl)
        svg_evaluator.evaluate()
        svg_evaluator.summarize()
    else:
        print(f"Skipping SVG: Output directory not found at {svg_output_path}")

    del vl
    _gpu_flush()

    print("="*20 + " Loading text judge (Qwen3-32B) " + "="*20)
    lm = LocalTextLM(configs.qwen3_model_name)

    print("\\n" + "="*20 + " Starting Geometry Evaluation (text / Qwen3) " + "="*20)
    if geometry_output_path.is_dir():
        configs.geometry["geometry_phase"] = "text"
        geometry_evaluator = GeometryEvaluator(configs.geometry, lm=lm, vl=None)
        geometry_evaluator.evaluate()
        geometry_evaluator.summarize()
    else:
        print(f"Skipping Geometry text: Output directory not found at {geometry_output_path}")

    del lm
    _gpu_flush()

    print("\\n" + "="*20 + " Starting Jigsaw Evaluation " + "="*20)
    jigsaw_output_path = Path(configs.jigsaw['out_root'])
    if jigsaw_output_path.is_dir():
        jigsaw_evaluator = JigsawEvaluator(configs.jigsaw)
        jigsaw_evaluator.evaluate()
        jigsaw_evaluator.summarize()
    else:
        print(f"Skipping Jigsaw: Output directory not found at {jigsaw_output_path}")

    print("\\n" + "="*20 + " Starting Maze Evaluation " + "="*20)
    maze_output_path = Path(configs.maze['run_root'])
    if maze_output_path.is_dir():
        maze_evaluator = MazeEvaluator(configs.maze)
        maze_evaluator.evaluate()
        maze_evaluator.summarize()
    else:
        print(f"Skipping Maze: Output directory not found at {maze_output_path}")

    print("\\n" + "="*20 + " Starting Sliding Puzzle Evaluation " + "="*20)
    sliding_output_path = Path(configs.sliding_puzzle['run_root'])
    if sliding_output_path.is_dir():
        sliding_puzzle_evaluator = SlidingPuzzleEvaluator(configs.sliding_puzzle)
        sliding_puzzle_evaluator.evaluate()
        sliding_puzzle_evaluator.summarize()
    else:
        print(f"Skipping Sliding Puzzle: Output directory not found at {sliding_output_path}")
'''


def patch(src: str) -> str:
    src = must_replace(src, VL_INIT_OLD, VL_INIT_NEW, "vl_fp16")
    src = must_replace(src, GEOM_LOOP_HEAD_OLD, GEOM_LOOP_HEAD_NEW, "geom_phase_head")
    src = must_replace(src, GEOM_OVERLAY_TEXT_OLD, GEOM_OVERLAY_TEXT_NEW, "geom_phase_body")
    src = must_replace(src, MAIN_OLD, MAIN_NEW, "main_sequential")
    return src


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_eval_sequential.py <eval_ummmu_patched.py>")
    path = Path(sys.argv[1])
    path.write_text(patch(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"[sequential] patched {path} (one judge in VRAM; VL=fp16 AWQ 72B)")


if __name__ == "__main__":
    main()
