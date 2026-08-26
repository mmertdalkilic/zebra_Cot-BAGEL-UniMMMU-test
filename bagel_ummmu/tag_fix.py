# -*- coding: utf-8 -*-
"""Copy bagel-zebra-cot sampling into bagel-zebra-cot-tagfix and repair
Uni-MMMU output tags in model_text.txt only.

Does not regenerate images or call a model. Recovers tags around text the
model already wrote (think-blocks, bare JSON). Does not invent a jigsaw
choice or a move list that is not already in the file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TASKS = ("science", "math", "code", "jigsaw", "maze", "sliding")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

RE_CHAT = re.compile(r"<\|im_start\|>.*?<\|im_end\|>", re.S)
RE_THINK = re.compile(r"<think>(.*?)</think>", re.I | re.S)
RE_RENDER = re.compile(r"<RENDER_SUMMARY>\s*(.*?)\s*</RENDER_SUMMARY>", re.I | re.S)
RE_OUTPUT = re.compile(r"<OUTPUT_PROMPT>\s*(.*?)\s*</OUTPUT_PROMPT>", re.I | re.S)
RE_FINAL = re.compile(r"<FINAL_ANSWER_JSON>\s*(\{.*?\})\s*</FINAL_ANSWER_JSON>", re.I | re.S)
RE_ANSWER = re.compile(r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>", re.I | re.S)
RE_CHOICE_JSON = re.compile(r"\{\s*\"choice\"\s*:\s*([01])\b", re.I)
RE_CHOICE_LOOSE = re.compile(r"\bchoice\b[\"'\s:=]+([01])\b", re.I)
RE_DIR_ARRAY = re.compile(
    r"\[\s*\"(?:up|down|left|right)\"(?:\s*,\s*\"(?:up|down|left|right)\")*\s*\]",
    re.I,
)
MOVES = {"up", "down", "left", "right"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _strip_chat(text: str) -> str:
    return RE_CHAT.sub(" ", text).strip()


def _think_or_body(text: str) -> str:
    m = RE_THINK.search(text)
    if m:
        return m.group(1).strip()
    return _strip_chat(text).strip()


def _valid_choice_block(text: str) -> bool:
    m = RE_FINAL.search(text)
    if not m:
        return False
    try:
        data = json.loads(m.group(1))
    except Exception:
        return False
    return data.get("choice") in (0, 1)


def _valid_moves_block(text: str) -> bool:
    matches = list(RE_ANSWER.finditer(text))
    if not matches:
        return False
    try:
        arr = json.loads(matches[-1].group(1))
    except Exception:
        return False
    return isinstance(arr, list) and all(str(x).strip().lower() in MOVES for x in arr)


def fix_science(text: str) -> Tuple[str, str]:
    if RE_OUTPUT.search(text):
        return text, "already_tagged"
    body = _think_or_body(text) or text.strip()
    if not body:
        return text, "empty"
    return f"<OUTPUT_PROMPT>\n{body}\n</OUTPUT_PROMPT>\n", "wrapped_output_prompt"


def fix_code(text: str) -> Tuple[str, str]:
    if RE_RENDER.search(text):
        return text, "already_tagged"
    body = _think_or_body(text) or text.strip()
    if not body:
        return text, "empty"
    return f"<RENDER_SUMMARY>\n{body}\n</RENDER_SUMMARY>\n", "wrapped_render_summary"


def fix_jigsaw(text: str) -> Tuple[str, str]:
    if _valid_choice_block(text):
        return text, "already_tagged"
    choice = None
    m = RE_CHOICE_JSON.search(text) or RE_CHOICE_LOOSE.search(text)
    if m:
        choice = int(m.group(1))
    if choice not in (0, 1):
        return text, "no_choice_found"
    rationale = _think_or_body(text)
    rationale = re.sub(r"\s+", " ", rationale)[:180].strip() or "Recovered choice from model text."
    block = (
        "<FINAL_ANSWER_JSON>\n"
        + json.dumps({"choice": choice, "rationale": rationale}, ensure_ascii=False)
        + "\n</FINAL_ANSWER_JSON>\n"
    )
    return block, "recovered_choice"


def _fix_moves_segment(seg: str) -> Tuple[str, str]:
    if _valid_moves_block(seg):
        return seg, "already_tagged"
    m = RE_DIR_ARRAY.search(seg)
    if m:
        try:
            arr = json.loads(m.group(0))
            arr = [str(x).strip().lower() for x in arr]
            if arr and all(x in MOVES for x in arr):
                return f"<ANSWER_JSON>{json.dumps(arr)}</ANSWER_JSON>\n", "wrapped_dir_array"
        except Exception:
            pass
    return seg, "no_moves_found"


def fix_maze_or_sliding(text: str) -> Tuple[str, str]:
    parts = text.split("\n\n\n-----\n\n\n")
    reasons: List[str] = []
    out: List[str] = []
    for part in parts:
        fixed, reason = _fix_moves_segment(part)
        out.append(fixed)
        reasons.append(reason)
    if len(out) == 1:
        return out[0], reasons[0]
    return "\n\n\n-----\n\n\n".join(out), ",".join(reasons)


def fix_math(text: str) -> Tuple[str, str]:
    return text, "unchanged_no_required_tag"


def fix_text(task: str, text: str) -> Tuple[str, str]:
    if task == "science":
        return fix_science(text)
    if task == "code":
        return fix_code(text)
    if task == "jigsaw":
        return fix_jigsaw(text)
    if task in ("maze", "sliding"):
        return fix_maze_or_sliding(text)
    if task == "math":
        return fix_math(text)
    return text, "unknown_task"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_case_assets(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if not p.is_file():
            if p.is_dir():
                nested = dst / p.name
                nested.mkdir(exist_ok=True)
                for q in p.rglob("*"):
                    if q.is_file():
                        link_or_copy(q, nested / q.relative_to(p))
            continue
        if p.name in {"model_text.txt", "_tagfix.ok", "_tagfix.json"}:
            continue
        link_or_copy(p, dst / p.name)


def maybe_sync(force: bool = False) -> None:
    cmd = os.environ.get("TAGFIX_SYNC_CMD")
    if not cmd:
        return
    interval = float(os.environ.get("TAGFIX_SYNC_INTERVAL_MIN", "15")) * 60.0
    now = time.time()
    last = float(getattr(maybe_sync, "_last", 0.0))
    if not force and now - last < interval:
        return
    maybe_sync._last = now  # type: ignore[attr-defined]
    print(f"[tagfix-sync] {cmd}", flush=True)
    try:
        subprocess.run(cmd, shell=True, check=False, timeout=1800)
    except Exception as e:
        print(f"[tagfix-sync] failed (non-fatal): {e}", flush=True)


def process_case(task: str, src: Path, dst: Path) -> Dict[str, Any]:
    marker = dst / "_tagfix.ok"
    if marker.is_file():
        return {"id": src.name, "status": "skipped"}
    copy_case_assets(src, dst)
    original = _read(src / "model_text.txt")
    if original and not (dst / "model_text.original.txt").is_file():
        _write(dst / "model_text.original.txt", original)
    fixed, reason = fix_text(task, original)
    _write(dst / "model_text.txt", fixed)
    changed = fixed != original
    rec = {
        "id": src.name,
        "task": task,
        "status": "ok",
        "changed": changed,
        "reason": reason,
        "src": str(src),
        "dst": str(dst),
    }
    _write(dst / "_tagfix.json", json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
    marker.write_text("ok\n", encoding="utf-8")
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy sampling outputs and repair Uni-MMMU tags.")
    parser.add_argument("--ummmu-root", required=True)
    parser.add_argument("--src-model", default="bagel-zebra-cot")
    parser.add_argument("--dst-model", default="bagel-zebra-cot-tagfix")
    parser.add_argument("--tasks", default="all")
    args = parser.parse_args()

    src_root = Path(args.ummmu_root).resolve() / "outputs" / args.src_model
    dst_root = Path(args.ummmu_root).resolve() / "outputs" / args.dst_model
    if not src_root.is_dir():
        raise SystemExit(f"missing source outputs: {src_root}")

    tasks = TASKS if args.tasks == "all" else tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    dst_root.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {"src": str(src_root), "dst": str(dst_root), "tasks": {}}

    print(f"=== tag fix {args.src_model} -> {args.dst_model} ===", flush=True)
    for extra in src_root.iterdir():
        if extra.is_file():
            link_or_copy(extra, dst_root / extra.name)
    for task in tasks:
        src_task = src_root / task
        dst_task = dst_root / task
        if not src_task.is_dir():
            print(f"[skip] no {src_task}", flush=True)
            continue
        dst_task.mkdir(parents=True, exist_ok=True)
        for extra in src_task.iterdir():
            if extra.is_file():
                link_or_copy(extra, dst_task / extra.name)
        cases = sorted(p for p in src_task.iterdir() if p.is_dir())
        n_skip = n_change = n_ok = 0
        reasons: Dict[str, int] = {}
        for case in cases:
            rec = process_case(task, case, dst_task / case.name)
            if rec.get("status") == "skipped":
                n_skip += 1
            else:
                n_ok += 1
                if rec.get("changed"):
                    n_change += 1
                reasons[rec.get("reason") or "?"] = reasons.get(rec.get("reason") or "?", 0) + 1
            maybe_sync()
        stats["tasks"][task] = {
            "cases": len(cases),
            "skipped": n_skip,
            "written": n_ok,
            "changed": n_change,
            "reasons": reasons,
        }
        print(
            f"[{task}] cases={len(cases)} skipped={n_skip} written={n_ok} changed={n_change} {reasons}",
            flush=True,
        )

    _write(dst_root / "_tagfix_manifest.json", json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    maybe_sync(force=True)
    print(f"Tag-fixed sampling tree: {dst_root}", flush=True)


if __name__ == "__main__":
    main()
