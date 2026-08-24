# -*- coding: utf-8 -*-
"""Patch a copy of Uni-MMMU eval_ummmu.py so evaluation is resumable and
lives inside the GitHub-synced outputs tree.

Writes per-item JSON under outputs/_eval/<model>/<task>/items/. Re-running
skips those files. Eval artifacts therefore survive molab restarts the same
way sampling outputs do.
"""
from __future__ import annotations

import sys
from pathlib import Path

HELPERS = r'''
    @staticmethod
    def eval_sync_maybe(force: bool = False) -> None:
        """Periodic GitHub push of outputs/_eval (non-fatal)."""
        import os, subprocess, time
        cmd = os.environ.get("EVAL_SYNC_CMD")
        if not cmd:
            return
        interval = float(os.environ.get("EVAL_SYNC_INTERVAL_MIN", "15")) * 60.0
        now = time.time()
        last = float(getattr(UtilityHelpers, "_last_eval_sync", 0.0))
        if not force and now - last < interval:
            return
        UtilityHelpers._last_eval_sync = now
        print(f"[eval-sync] {cmd}", flush=True)
        try:
            subprocess.run(cmd, shell=True, check=False, timeout=1800)
        except Exception as e:
            print(f"[eval-sync] failed (non-fatal): {e}", flush=True)

    @staticmethod
    def eval_out_of_time() -> bool:
        import os, time
        raw = os.environ.get("EVAL_TIME_BUDGET_HOURS")
        if not raw:
            return False
        start = float(getattr(UtilityHelpers, "_eval_t0", 0.0) or 0.0)
        if start == 0.0:
            UtilityHelpers._eval_t0 = time.time()
            start = UtilityHelpers._eval_t0
        return time.time() >= start + float(raw) * 3600.0

    @staticmethod
    def resume_item_path(eval_dir, name: str):
        return Path(eval_dir) / "items" / f"{name}.json"

    @staticmethod
    def try_load_item(eval_dir, name: str):
        p = UtilityHelpers.resume_item_path(eval_dir, name)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def save_resume_item(eval_dir, name: str, record) -> None:
        p = UtilityHelpers.resume_item_path(eval_dir, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        UtilityHelpers.eval_sync_maybe()

    @staticmethod
    def explode_details_into_items(eval_dir, name_keys) -> None:
        """If a previous non-resumable run wrote eval_details.json, split it
        into per-item files so those judgements are not redone."""
        d = Path(eval_dir)
        details = d / "eval_details.json"
        items_dir = d / "items"
        if not details.is_file():
            return
        if items_dir.is_dir() and any(items_dir.glob("*.json")):
            return
        try:
            data = json.loads(details.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, list) or not data:
            return
        items_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for rec in data:
            name = None
            for k in name_keys:
                if rec.get(k):
                    name = str(rec[k])
                    break
            if not name and rec.get("case_dir"):
                name = Path(rec["case_dir"]).name
            if not name:
                continue
            name = name.replace("/", "_")
            (items_dir / f"{name}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            n += 1
        print(f"[resume] exploded {n} records from {details} -> {items_dir}", flush=True)
'''

GEOM_LOOP_OLD = '''        for big_k, small_k, item in tqdm(items, desc="Evaluating Geometry"):
            case_id = f"{self.utils.sanitize_filename(big_k)}__{self.utils.sanitize_filename(small_k)}"
            case_dir = Path(self.config['out_dir']) / case_id
            record = {"id": case_id, "type": item.get("type"), "status": "init"}
'''

GEOM_LOOP_NEW = '''        self.utils.explode_details_into_items(self.config['out_eval_dir'], ["id"])
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

GEOM_APPEND_OK_OLD = '''            record.update(text_eval)
            record["status"] = "ok"
            self.results.append(record)
'''

GEOM_APPEND_OK_NEW = '''            record.update(text_eval)
            record["status"] = "ok"
            self.utils.save_resume_item(self.config['out_eval_dir'], case_id, record)
            self.results.append(record)
'''

SCI_LOOP_OLD = '''        for idx, case in enumerate(tqdm(flat_cases, desc="Evaluating Science"), 1):
            case_dir = Path(self.config['run_root']) / f"case_{idx:02d}"
            record = {"case_id": f"case_{idx:02d}"}
'''

SCI_LOOP_NEW = '''        self.utils.explode_details_into_items(self.config['eval_root'], ["case_id"])
        for idx, case in enumerate(tqdm(flat_cases, desc="Evaluating Science"), 1):
            cid = f"case_{idx:02d}"
            cached = self.utils.try_load_item(self.config['eval_root'], cid)
            if cached is not None:
                blob = json.dumps(cached)
                if "API Error" not in blob and "IncompatibleTypeError" not in blob:
                    self.results.append(cached)
                    continue
            if self.utils.eval_out_of_time():
                print("[eval] time budget hit during science — remaining items resume next session", flush=True)
                break
            case_dir = Path(self.config['run_root']) / cid
            record = {"case_id": cid}
'''

SCI_APPEND_OLD = '''            self.results.append(record)
'''

# Science evaluate() has a single append at the end of the happy path; the
# method is unique enough if we include the image_eval lines before it.
SCI_APPEND_CTX_OLD = '''            if pred_imgs and initial_image:
                record["image_eval"] = self._evaluate_image(pred_imgs[0], Path(initial_image), condition, gt_text)
            else:
                record["image_eval"] = {"image_correct": 0}

            self.results.append(record)
'''

SCI_APPEND_CTX_NEW = '''            if pred_imgs and initial_image:
                record["image_eval"] = self._evaluate_image(pred_imgs[0], Path(initial_image), condition, gt_text)
            else:
                record["image_eval"] = {"image_correct": 0}

            self.utils.save_resume_item(self.config['eval_root'], cid, record)
            self.results.append(record)
'''

SVG_LOOP_OLD = '''        for case_dir in tqdm(case_dirs, desc="Evaluating SVG"):
            ref_img = self._get_reference_image(case_dir)
'''

SVG_LOOP_NEW = '''        self.utils.explode_details_into_items(self.config['eval_out_dir'], ["case_dir"])
        for case_dir in tqdm(case_dirs, desc="Evaluating SVG"):
            cid = case_dir.name
            cached = self.utils.try_load_item(self.config['eval_out_dir'], cid)
            if cached is not None:
                blob = json.dumps(cached)
                if "API Error" not in blob and "IncompatibleTypeError" not in blob:
                    self.results.append(cached)
                    continue
            if self.utils.eval_out_of_time():
                print("[eval] time budget hit during SVG — remaining items resume next session", flush=True)
                break
            ref_img = self._get_reference_image(case_dir)
'''

SVG_APPEND_OLD = '''            self.results.append({
                "case_dir": str(case_dir),
                "image_evals": image_evals,
                "text_eval": text_eval,
            })
'''

SVG_APPEND_NEW = '''            rec = {
                "case_dir": str(case_dir),
                "image_evals": image_evals,
                "text_eval": text_eval,
            }
            self.utils.save_resume_item(self.config['eval_out_dir'], cid, rec)
            self.results.append(rec)
'''

JIG_LOOP_OLD = '''        for item in tqdm(items_to_eval, desc="Evaluating Jigsaw"):
            item_id = item['id']
            case_dir = Path(self.config['out_root']) / item_id
            record = {"id": item_id, "label": item["label"]}
'''

JIG_LOOP_NEW = '''        self.utils.explode_details_into_items(self.config['eval_dir'], ["id"])
        for item in tqdm(items_to_eval, desc="Evaluating Jigsaw"):
            item_id = item['id']
            cached = self.utils.try_load_item(self.config['eval_dir'], item_id)
            if cached is not None:
                self.results.append(cached)
                continue
            case_dir = Path(self.config['out_root']) / item_id
            record = {"id": item_id, "label": item["label"]}
'''

# Jigsaw appends in several branches; wrap the final append in evaluate by
# replacing `self.results.append(record)` inside JigsawEvaluator.evaluate only
# via a unique trailing update. Read the end of that loop.

EVAL_PATH_OLD = '        self.eval_path = f"{self.base_path}/eval/{self.model_name}"'
EVAL_PATH_NEW = '        self.eval_path = f"{self.base_path}/outputs/_eval/{self.model_name}"'

MAIN_END_OLD = '''    # Final Aggregation Step
    summarize_all_tasks(configs)

    print("\\nAll available evaluations and final summary generation completed.")
'''

MAIN_END_NEW = '''    # Final Aggregation Step
    summarize_all_tasks(configs)
    UtilityHelpers.eval_sync_maybe(force=True)

    print("\\nAll available evaluations and final summary generation completed.")
'''


def must_replace(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        raise SystemExit(f"patch {label}: pattern not found")
    if src.count(old) != 1:
        raise SystemExit(f"patch {label}: expected 1 occurrence, found {src.count(old)}")
    return src.replace(old, new, 1)


def patch(src: str) -> str:
    # Insert helper methods after write_json.
    marker = '''        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
'''
    if marker not in src:
        raise SystemExit("could not find write_json body to insert helpers")
    src = src.replace(marker, marker + "\n" + HELPERS, 1)

    src = must_replace(src, EVAL_PATH_OLD, EVAL_PATH_NEW, "eval_path")
    src = must_replace(src, GEOM_LOOP_OLD, GEOM_LOOP_NEW, "geom_loop")
    src = must_replace(src, GEOM_APPEND_OK_OLD, GEOM_APPEND_OK_NEW, "geom_append")
    src = must_replace(src, SCI_LOOP_OLD, SCI_LOOP_NEW, "sci_loop")
    src = must_replace(src, SCI_APPEND_CTX_OLD, SCI_APPEND_CTX_NEW, "sci_append")
    src = must_replace(src, SVG_LOOP_OLD, SVG_LOOP_NEW, "svg_loop")
    src = must_replace(src, SVG_APPEND_OLD, SVG_APPEND_NEW, "svg_append")
    jig_ids_old = '''        run_ids = [item['id'] for item in run_summary.get("per_item", [])]
        items_to_eval = [item_index[i] for i in run_ids if i in item_index]
'''
    jig_ids_new = '''        run_ids = [item['id'] for item in run_summary.get("per_item", []) if item.get("id")]
        # Last sampling session may rewrite summary.json with an empty per_item
        # (everything skipped). Fall back to output folders / full metadata.
        if not run_ids:
            out_root = Path(self.config['out_root'])
            run_ids = [p.name for p in out_root.iterdir() if p.is_dir()] if out_root.is_dir() else list(item_index)
        items_to_eval = [item_index[i] for i in run_ids if i in item_index]
'''
    src = must_replace(src, jig_ids_old, jig_ids_new, "jig_ids")
    src = must_replace(src, JIG_LOOP_OLD, JIG_LOOP_NEW, "jig_loop")

    # Persist every jigsaw record (the loop has several append(record) sites;
    # wrap via a tiny helper assignment would be nicer, but saving on the
    # common append is enough if we replace all in JigsawEvaluator.evaluate).
    jig_append_old = "            self.results.append(record)\n"
    # Too common. Save jigsaw at summarize time from results is already there;
    # per-item save: add after each record is fully built. The jigsaw loop
    # always ends at `self.results.append(record)` once per item. Count in
    # evaluate() only: there is exactly one such append in JigsawEvaluator.evaluate
    # (line 725). We'll unique it with the preceding d_mean lines.

    jig_save_old = '''            else:
                record.update({"image_status": f"invalid_count_{len(pred_imgs)}", "d_mean": self.config['penalty_distance']})
            self.results.append(record)
'''
    jig_save_new = '''            else:
                record.update({"image_status": f"invalid_count_{len(pred_imgs)}", "d_mean": self.config['penalty_distance']})
            self.utils.save_resume_item(self.config['eval_dir'], item_id, record)
            self.results.append(record)
'''
    src = must_replace(src, jig_save_old, jig_save_new, "jig_append")

    src = must_replace(src, MAIN_END_OLD, MAIN_END_NEW, "main_end")
    return src


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_eval_resume.py <eval_ummmu_patched.py>")
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    path.write_text(patch(text), encoding="utf-8")
    print(f"[resume] patched {path} (eval artifacts -> outputs/_eval, per-item resume)")


if __name__ == "__main__":
    main()
