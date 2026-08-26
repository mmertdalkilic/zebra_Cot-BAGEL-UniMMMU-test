# -*- coding: utf-8 -*-
"""Write Uni-MMMU Table 2–style scores from per-task eval JSON.

Official eval_ummmu.py writes all_tasks_summary_*.xlsx in raw judge units:
  rates in [0, 1], maze/sliding as a single frame-macro float, SVG 0–5.
The paper (Table 2) reports everything on [0, 100], maze/sliding as
step/sample (a/b), and SVG shape/position as (score / 5) * 100.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pandas as pd
except Exception:
    pd = None


PAPER_COLUMNS = [
    "Model",
    "Jig. I",
    "Jig. T",
    "Maze I",
    "Maze T",
    "Slid. I",
    "Slid. T",
    "Geo I",
    "Geo T",
    "Sci. R",
    "Sci. T",
    "Sci. I",
    "C. T",
    "C. S",
    "C. P",
    "Avg.",
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pct(rate: Optional[float], nd: int = 1) -> Optional[float]:
    if rate is None:
        return None
    return round(float(rate) * 100.0, nd)


def _pct_from_5(score: Optional[float], nd: int = 1) -> Optional[float]:
    """Paper normalizes SVG 0–5 judges to [0, 100]."""
    if score is None:
        return None
    return round(float(score) / 5.0 * 100.0, nd)


def _ab(step: Optional[float], sample: Optional[float]) -> Optional[str]:
    if step is None and sample is None:
        return None
    a = _pct(step) if step is not None else 0.0
    b = _pct(sample) if sample is not None else 0.0
    return f"{a:.1f}/{b:.1f}"


def _ab_mid(step: Optional[float], sample: Optional[float]) -> Optional[float]:
    if step is None and sample is None:
        return None
    a = _pct(step) if step is not None else 0.0
    b = _pct(sample) if sample is not None else 0.0
    return round((a + b) / 2.0, 1)


def build_paper_row(eval_root: Path, model_name: str) -> Dict[str, Any]:
    jigsaw = _read_json(eval_root / "jigsaw" / "summary.json")
    maze = _read_json(eval_root / "maze" / "summary.json")
    sliding = _read_json(eval_root / "sliding" / "summary.json")
    math = _read_json(eval_root / "math" / "eval_summary.json")
    science = _read_json(eval_root / "science" / "eval_summary.json")
    code = _read_json(eval_root / "code" / "eval_summary.json")
    jm = jigsaw.get("metrics") or {}

    jig_i = _pct(jm.get("image_score_penalized"))
    jig_t = _pct(jm.get("text_accuracy"))
    maze_i = _ab(maze.get("img_accuracy_frame_macro"), maze.get("img_accuracy_exact"))
    maze_t = _ab(maze.get("text_accuracy_frame_macro"), maze.get("text_accuracy_exact"))
    slid_i = _ab(sliding.get("img_accuracy_frame_macro"), sliding.get("img_accuracy_exact"))
    slid_t = _ab(sliding.get("text_accuracy_frame_macro"), sliding.get("text_accuracy_exact"))
    geo_i = _pct(math.get("overlay_acc"))
    geo_t = _pct(math.get("text_acc"))
    sci_r = _pct(science.get("text_reasoning_acc"))
    sci_t = _pct(science.get("text_result_acc"))
    sci_i = _pct(science.get("image_acc"))
    c_t = _pct(code.get("text_semantic_match_rate"))
    c_s = _pct_from_5(code.get("avg_shape_color_accuracy"))
    c_p = _pct_from_5(code.get("avg_position_accuracy"))

    avg_parts = [
        jig_i,
        jig_t,
        _ab_mid(maze.get("img_accuracy_frame_macro"), maze.get("img_accuracy_exact")),
        _ab_mid(maze.get("text_accuracy_frame_macro"), maze.get("text_accuracy_exact")),
        _ab_mid(sliding.get("img_accuracy_frame_macro"), sliding.get("img_accuracy_exact")),
        _ab_mid(sliding.get("text_accuracy_frame_macro"), sliding.get("text_accuracy_exact")),
        geo_i,
        geo_t,
        sci_r,
        sci_t,
        sci_i,
        c_t,
        c_s,
        c_p,
    ]
    present = [x for x in avg_parts if x is not None]
    avg = round(sum(present) / len(present), 1) if present else None

    return {
        "Model": model_name,
        "Jig. I": jig_i,
        "Jig. T": jig_t,
        "Maze I": maze_i,
        "Maze T": maze_t,
        "Slid. I": slid_i,
        "Slid. T": slid_t,
        "Geo I": geo_i,
        "Geo T": geo_t,
        "Sci. R": sci_r,
        "Sci. T": sci_t,
        "Sci. I": sci_i,
        "C. T": c_t,
        "C. S": c_s,
        "C. P": c_p,
        "Avg.": avg,
        "_avg_parts": avg_parts,
        "_math_scored": math.get("scored_items"),
        "_math_total": math.get("total_items"),
    }


def format_markdown(row: Dict[str, Any]) -> str:
    cells = [str(row.get(c) if row.get(c) is not None else "-") for c in PAPER_COLUMNS]
    header = "| " + " | ".join(PAPER_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in PAPER_COLUMNS) + " |"
    body = "| " + " | ".join(cells) + " |"
    note = (
        "Scores on [0, 100] as in Uni-MMMU Table 2. "
        "Maze/Sliding are step-level / sample-level. "
        "C. S / C. P are (0–5 judge) × 20. "
        "Avg. is the unweighted mean of the 14 metrics, using (step+sample)/2 "
        "for each Maze/Sliding I and T cell."
    )
    extra = ""
    if row.get("_math_scored") and row.get("_math_total"):
        extra = f" Geometry scored {row['_math_scored']}/{row['_math_total']} items."
    return header + "\n" + sep + "\n" + body + "\n\n" + note + extra + "\n"


def write_paper_table(eval_root: Path, model_name: str) -> Dict[str, Any]:
    row = build_paper_row(eval_root, model_name)
    out_xlsx = eval_root / f"ummmu_table2_{model_name}.xlsx"
    out_csv = eval_root / f"ummmu_table2_{model_name}.csv"
    export = {k: row[k] for k in PAPER_COLUMNS}
    if pd is not None:
        df = pd.DataFrame([export], columns=PAPER_COLUMNS)
        try:
            df.to_excel(out_xlsx, index=False)
        except Exception as e:
            print(f"[report] xlsx write failed ({e}); csv still written", flush=True)
        df.to_csv(out_csv, index=False)
    else:
        out_csv.write_text(
            ",".join(PAPER_COLUMNS)
            + "\n"
            + ",".join("" if export[c] is None else str(export[c]) for c in PAPER_COLUMNS)
            + "\n",
            encoding="utf-8",
        )
    md = format_markdown(row)
    (eval_root / f"ummmu_table2_{model_name}.md").write_text(md, encoding="utf-8")
    print("\n=== Uni-MMMU Table 2 format ===", flush=True)
    print(md, flush=True)
    print(f"[report] {out_csv}", flush=True)
    if out_xlsx.is_file():
        print(f"[report] {out_xlsx}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit Uni-MMMU Table 2–style scores.")
    parser.add_argument("--eval-dir", required=True, help="outputs/_eval/<model>")
    parser.add_argument("--model-name", default="bagel-zebra-cot")
    args = parser.parse_args()
    write_paper_table(Path(args.eval_dir), args.model_name)


if __name__ == "__main__":
    main()
