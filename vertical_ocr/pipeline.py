from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .cache import ContentCache, array_region_fingerprint, derive_key
from .codes import (
    FIELD_TRUNCATED,
    MATCH_BUDGET_PENDING,
    OK,
    SNAPSHOT_STALE,
    STAGE_EXCEPTION,
)
from .config import canonical_hash, stage_parameter_fingerprints
from .errors import BusinessError
from .glyphs import template_variants
from .exchange import (
    downgrade_v3_to_v1,
    parse_v1,
    parse_v2,
    parse_v3,
    to_v2,
    to_v3,
    upgrade_v1_to_v3,
)
from .ledger import JsonlLedger, SidecarStore, atomic_write_json
from .matching import block_point_cost, match_block
from .model import MISSING, ExchangeRecord, is_missing
from .reading import associate_truth_block, build_reading_sequence
from .scheduler import BudgetScheduler
from .snapshot import SnapshotStore
from .validation import load_dataset
from .vision import build_projection_candidates, detect_columns, process_candidate


class Pipeline:
    def __init__(self, root: str | Path, config: dict[str, Any]):
        self.root = Path(root)
        self.config = config
        self.config_hash = canonical_hash(config)
        self.param_fingerprints = stage_parameter_fingerprints(config)
        self.cache = ContentCache(self.root / "ledger" / "cache_store.json")
        self.cache_ledger = JsonlLedger(self.root / "ledger" / "cache_ledger.jsonl")
        self.conversion_ledger = JsonlLedger(self.root / "ledger" / "conversion_ledger.jsonl")
        self.sidecars = SidecarStore(self.root / "ledger" / "sidecar_store.json", self.conversion_ledger)
        self.snapshots = SnapshotStore(self.root)
        self.templates = template_variants(config["template_version"])

    def _global_version(self, dataset: dict[str, Any]) -> Any:
        plan_path = self.root / "input" / "sample_plan.json"
        if plan_path.exists():
            import json

            return json.loads(plan_path.read_text(encoding="utf-8")).get("global_version", 1)
        return "ds-" + canonical_hash({
            "images": {name: img["fingerprint"] for name, img in dataset["images"].items()},
            "keypoints": dataset["keypoint_fingerprint"],
        })[:12]

    def run(
        self,
        *,
        batch_budget: int | None = None,
        kill_before_commit: bool = False,
        continue_pending: bool = True,
    ) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        code_counts: dict[str, int] = {}
        failures: list[dict[str, Any]] = []

        def count(code: str) -> None:
            code_counts[code] = code_counts.get(code, 0) + 1

        dataset = load_dataset(self.root, self.config)
        for failure in dataset["failures"]:
            failures.append(failure)
            count(failure.get("code", "E_INPUT_INVALID"))

        global_version = self._global_version(dataset)
        fresh, snapshot_reason, current_snapshot = self.snapshots.is_fresh(global_version, self.config_hash)
        snapshot_status = {
            "fresh": fresh,
            "reason": snapshot_reason,
            "global_version": str(global_version),
            "config_hash": self.config_hash,
            "recovery": self.snapshots.recovery_summary(),
        }
        if not fresh:
            count(SNAPSHOT_STALE)
            self.cache_ledger.append(SNAPSHOT_STALE.lower(), {"reason": snapshot_reason, "global_version": str(global_version)})

        image_names = sorted(dataset["images"].keys())
        column_products: dict[str, list[dict[str, Any]]] = {}
        stage_errors: dict[str, dict[str, Any]] = {}

        def column_worker(name: str) -> tuple[str, list[dict[str, Any]] | None, dict[str, Any] | None]:
            try:
                gray = dataset["images"][name]["array"]
                image_fp = dataset["images"][name]["fingerprint"]
                key = derive_key("column_detection", image_fp, self.param_fingerprints["column"])
                scope = f"columns:{name}"
                self.cache.check_scope_invalidation(scope, key, "image_or_column_parameter_changed", {
                    "stage": "column_detection",
                    "image": name,
                    "image_fingerprint": image_fp,
                    "parameter_fingerprint": self.param_fingerprints["column"],
                })
                product = self.cache.get(key)
                if product is not None:
                    self.cache.hit("column_detection", scope, key, {"image_fingerprint": image_fp})
                    return name, product, None
                product = detect_columns(gray, self.config)
                self.cache.put("column_detection", scope, key, product, basis={"image_fingerprint": image_fp})
                return name, product, None
            except Exception as exc:  # isolation boundary
                code = exc.code if isinstance(exc, BusinessError) else STAGE_EXCEPTION
                error = {"image": name, "stage": "column_detection", "code": code, "message": str(exc)}
                return name, None, error

        workers = min(int(self.config["max_workers"]), max(1, len(image_names)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name, columns, error in pool.map(column_worker, image_names):
                if error:
                    stage_errors[name] = error
                    failures.append(error)
                    count(error["code"])
                else:
                    column_products[name] = columns or []

        prepared_images: dict[str, dict[str, Any]] = {}
        work_items: list[dict[str, Any]] = []
        all_match_results: list[dict[str, Any]] = []

        for name in image_names:
            if name in stage_errors:
                continue
            try:
                gray = dataset["images"][name]["array"]
                image_fp = dataset["images"][name]["fingerprint"]
                columns = column_products.get(name, [])
                blocks: list[dict[str, Any]] = []
                dropped: list[dict[str, Any]] = []
                candidate_products: list[dict[str, Any]] = []
                for column in columns:
                    region_fp = array_region_fingerprint(gray, column["x1"], 0, column["x2"], gray.shape[0])
                    pkey = derive_key("projection_segmentation", region_fp,
                                      self.param_fingerprints["projection"])
                    pscope = f"projection:{name}:{column['interval_key']}"
                    self.cache.check_scope_invalidation(pscope, pkey, "projection_input_changed", {"stage": "projection_segmentation"})
                    projection = self.cache.get(pkey)
                    if projection is not None:
                        self.cache.hit("projection_segmentation", pscope, pkey)
                    else:
                        projection = build_projection_candidates(gray, column, self.config)
                        self.cache.put("projection_segmentation", pscope, pkey, projection)
                    candidate_products.append(projection)

                    for candidate in projection["candidates"]:
                        requires_split = int(candidate["height"]) > int(self.config["char_max_height"])
                        ckey = derive_key(
                            "candidate_regression",
                            region_fp,
                            candidate["y1"],
                            candidate["y2"],
                            self.param_fingerprints["projection"],
                            self.param_fingerprints["split"] if requires_split else "no_dp_split_required",
                            self.param_fingerprints["regression"],
                        )
                        cscope = f"candidate:{name}:{column['interval_key']}:{candidate['y1']}-{candidate['y2']}"
                        self.cache.check_scope_invalidation(cscope, ckey, "candidate_or_regression_changed", {"stage": "border_regression"})
                        processed = self.cache.get(ckey)
                        if processed is not None:
                            self.cache.hit("border_regression", cscope, ckey)
                        else:
                            processed = process_candidate(gray, column, candidate, self.config)
                            self.cache.put("border_regression", cscope, ckey, processed)
                        blocks.extend(processed["blocks"])
                        dropped.extend(processed["dropped"])

                blocks.sort(key=lambda b: (int(b["rtl_index"]), int(b["bbox"][1])))
                for char_index, block in enumerate(blocks):
                    block["detected_char_index"] = char_index
                    block["stable_char_key"] = f"img{image_names.index(name) + 1}-{column['column_key']}-ch{char_index}"
                    truth_blocks = [v for key, v in dataset["block_index"].items() if key[0] == name]
                    truth = associate_truth_block(block["bbox"], truth_blocks)
                    if truth:
                        block["truth_block_id"] = truth["block_id"]
                        block["truth_reading_no"] = truth.get("reading_no")
                        block["truth_column_id"] = truth.get("column_id")
                    else:
                        block["truth_block_id"] = None
                        block["truth_reading_no"] = None
                        block["truth_column_id"] = None
                    work_id = f"{name}:{block['truth_block_id'] or block['stable_char_key']}"
                    work_items.append({
                        "work_id": work_id,
                        "image": name,
                        "rtl_index": block["rtl_index"],
                        "char_index": char_index,
                        "global_detected_index": char_index,
                        "character_id": block["truth_block_id"] or block["stable_char_key"],
                        "truth_block_id": block["truth_block_id"],
                        "truth_reading_no": block["truth_reading_no"],
                        "truth_column_id": block["truth_column_id"],
                        "stable_char_key": block["stable_char_key"],
                        "bbox": block["bbox"],
                        "point_cost": block_point_cost(
                            block, len(next(iter(self.templates.values())))
                        ),
                        "block": block,
                    })
                prepared_images[name] = {
                    "columns": columns,
                    "candidate_products": candidate_products,
                    "blocks": blocks,
                    "dropped": dropped,
                }
            except Exception as exc:
                code = exc.code if isinstance(exc, BusinessError) else STAGE_EXCEPTION
                error = {"image": name, "stage": "projection_or_regression", "code": code, "message": str(exc)}
                stage_errors[name] = error
                failures.append(error)
                count(error["code"])

        queue_path = self.root / "ledger" / "budget_queue.json"
        scheduler = BudgetScheduler(
            batch_budget if batch_budget is not None else int(self.config["match_point_budget"]),
            queue_path=queue_path if continue_pending else None,
            max_workers=int(self.config["max_workers"]),
        )

        results_by_id: dict[str, dict[str, Any]] = {}
        match_key_by_id: dict[str, str] = {}
        to_compute: list[dict[str, Any]] = []
        budget_items: list[dict[str, Any]] = []
        for item in work_items:
            name = item["image"]
            gray = dataset["images"][name]["array"]
            block = item["block"]
            crop_fp = array_region_fingerprint(gray, *block["bbox"])
            mkey = derive_key(
                "template_match",
                crop_fp,
                self.param_fingerprints["matching"],
                self.config["template_version"],
                self.config["fixed_point"]["quantization_level"],
            )
            match_key_by_id[item["work_id"]] = mkey
            scope = f"match:{item['work_id']}"
            self.cache.check_scope_invalidation(scope, mkey, "character_pixels_or_matching_parameters_changed", {
                "stage": "template_matching",
                "character_id": item["character_id"],
            })
            product = self.cache.get(mkey)
            if product is not None:
                self.cache.hit("template_matching", scope, mkey, {"character_id": item["character_id"]})
                result = dict(product)
                result.update({
                    "truth_block_id": item.get("truth_block_id"),
                    "truth_reading_no": item.get("truth_reading_no"),
                    "truth_column_id": item.get("truth_column_id"),
                    "stable_char_key": item.get("stable_char_key"),
                    "match_cache_key": mkey,
                    "match_status": OK,
                })
                results_by_id[item["work_id"]] = result
                all_match_results.append(result)
            else:
                budget_items.append(item)

        admitted, pending_markers = scheduler.admit(budget_items)
        admitted_ids = {item["work_id"] for item in admitted}
        pending_ids = {p["work_id"] for p in pending_markers}
        admitted_order = [item["work_id"] for item in admitted]
        for marker in pending_markers:
            count(MATCH_BUDGET_PENDING)
        for item in admitted:
            to_compute.append({"item": item, "key": match_key_by_id[item["work_id"]]})

        def match_worker(job: dict[str, Any]) -> dict[str, Any]:
            item = job["item"]
            try:
                gray = dataset["images"][item["image"]]["array"]
                result = match_block(gray, item["block"], self.config, self.templates)
                result["match_status"] = OK
                return {"work_id": item["work_id"], "key": job["key"], "result": result, "error": None}
            except Exception as exc:
                code = exc.code if isinstance(exc, BusinessError) else STAGE_EXCEPTION
                return {"work_id": item["work_id"], "key": job["key"], "result": None,
                        "error": {"image": item["image"], "stage": "character_matching", "code": code, "message": str(exc)}}

        for outcome in scheduler.run_pool(to_compute, match_worker):
            if outcome["error"]:
                failures.append(outcome["error"])
                count(outcome["error"]["code"])
                continue
            item = next(job["item"] for job in to_compute if job["key"] == outcome["key"])
            result = outcome["result"]
            self.cache.put("template_matching", f"match:{outcome['work_id']}", outcome["key"], result)
            result.update({
                "truth_block_id": item.get("truth_block_id"),
                "truth_reading_no": item.get("truth_reading_no"),
                "truth_column_id": item.get("truth_column_id"),
                "stable_char_key": item.get("stable_char_key"),
                "match_cache_key": outcome["key"],
            })
            results_by_id[outcome["work_id"]] = result
            all_match_results.append(result)

        readings: dict[str, dict[str, Any]] = {}
        for name in image_names:
            if name in stage_errors:
                continue
            try:
                image_items = [item for item in work_items if item["image"] == name]
                character_records: list[dict[str, Any]] = []
                for item in image_items:
                    if item["work_id"] in results_by_id:
                        character_records.append(results_by_id[item["work_id"]])
                    else:
                        block = item["block"]
                        character_records.append({
                            **{k: v for k, v in block.items() if k != "regression"},
                            "truth_block_id": item.get("truth_block_id"),
                            "truth_reading_no": item.get("truth_reading_no"),
                            "truth_column_id": item.get("truth_column_id"),
                            "stable_char_key": item.get("stable_char_key"),
                            "match_status": MATCH_BUDGET_PENDING,
                            "recognized_char": None,
                            "confidence": None,
                        })
                dependency = canonical_hash([
                    {
                        "work_id": item["work_id"],
                        "match_key": match_key_by_id.get(item["work_id"]),
                        "pending": item["work_id"] in pending_ids,
                        "truth_reading_no": item.get("truth_reading_no"),
                    }
                    for item in image_items
                ])
                rkey = derive_key("reading_assembly", dependency, dataset["keypoint_fingerprint"],
                                  self.param_fingerprints["reading"])
                rscope = f"reading:{name}"
                self.cache.check_scope_invalidation(rscope, rkey, "keypoint_or_character_dependency_changed", {
                    "stage": "reading_assembly",
                    "keypoint_fingerprint": dataset["keypoint_fingerprint"],
                })
                reading = self.cache.get(rkey)
                if reading is not None:
                    self.cache.hit("reading_assembly", rscope, rkey)
                else:
                    reading = build_reading_sequence(name, character_records, dataset["keypoint_submissions"])
                    self.cache.put("reading_assembly", rscope, rkey, reading)
                readings[name] = reading
            except Exception as exc:
                code = exc.code if isinstance(exc, BusinessError) else STAGE_EXCEPTION
                error = {"image": name, "stage": "reading_assembly", "code": code, "message": str(exc)}
                stage_errors[name] = error
                failures.append(error)
                count(error["code"])

        image_no_by_name = {
            entry["image"]: entry["image_no"]
            for entry in (dataset["truth"] or {}).get("images", [])
            if isinstance(entry, dict) and "image" in entry
        }
        exchange_records = self._build_exchange_records(readings, image_no_by_name)
        receipts = self._write_exchange(exchange_records)
        for code in receipts.get("parse_error_codes", []):
            count(code)

        conflict_totals = {"cross_column": 0, "wrong_order": 0, "missing_number": 0}
        images_report = []
        for name in image_names:
            reading = readings.get(name)
            if not reading:
                continue
            for key in conflict_totals:
                conflict_totals[key] += reading["conflict_counts"].get(key, 0)
            images_report.append({
                "image": name,
                "image_no": image_no_by_name.get(name),
                "columns": len(column_products.get(name, [])),
                "characters": len(reading["entries"]),
                "reading_sequence": [
                    {
                        "reading_no": e["reading_no"],
                        "character_id": e["character_id"],
                        "truth_reading_no": e.get("truth_reading_no"),
                        "char": e.get("recognized_char"),
                        "status": e.get("match_status", OK),
                        "keypoint": e.get("keypoint_coordinates"),
                        "keypoint_version": e.get("keypoint_version"),
                        "annotation_version": e.get("annotation_version"),
                        "in_conflict": e.get("in_conflict", False),
                    }
                    for e in reading["entries"]
                ],
                "conflict_counts": reading["conflict_counts"],
                "conflicts": reading["conflicts"],
                "dropped_candidates": len(prepared_images.get(name, {}).get("dropped", [])),
            })

        cache_counts = self.cache.counts()
        for event in self.cache.events:
            self.cache_ledger.append(f"cache_{event['status']}", {
                "scope": event.get("scope"),
                "stage": event.get("stage"),
                "cache_key": event.get("cache_key"),
                "reason": event.get("reason"),
            })

        stage_products = {
            "01_load_validation": {
                "failures": failures,
                "image_fingerprints": {n: img["fingerprint"] for n, img in dataset["images"].items()},
                "keypoint_fingerprint": dataset["keypoint_fingerprint"],
            },
            "02_column_detection": column_products,
            "03_character_matching": {"results": all_match_results, "pending": pending_markers},
            "04_reading_assembly": readings,
            "budget_queue": {
                "batch_budget": scheduler.batch_budget,
                "batch_consumed": scheduler.batch_consumed,
                "remaining_budget": scheduler.remaining,
                "total_consumed": scheduler.total_consumed,
                "pending": pending_markers,
            },
        }
        manifest = self.snapshots.commit(
            global_version=global_version,
            config_hash=self.config_hash,
            template_version=self.config["template_version"],
            keypoint_fingerprint=dataset["keypoint_fingerprint"],
            completed_images=[name for name in image_names if name not in stage_errors],
            consumed_budget=scheduler.total_consumed,
            pending=pending_markers,
            stage_products=stage_products,
            kill_before_switch=kill_before_commit,
        )

        elapsed_ns = time.monotonic_ns() - started_ns
        report = {
            "schema": "vertical-ocr-report/v1",
            "elapsed_ms": round(elapsed_ns / 1_000_000, 3),
            "global_version": str(global_version),
            "config_hash": self.config_hash,
            "snapshot": {**snapshot_status, "committed_manifest": manifest},
            "receipt_counts": receipts,
            "images": images_report,
            "conflict_counts": conflict_totals,
            "cache_counts": cache_counts,
            "cache_events": self.cache.events,
            "pending": pending_markers,
            "budget": {
                "batch_budget": scheduler.batch_budget,
                "batch_consumed": scheduler.batch_consumed,
                "remaining_budget": scheduler.remaining,
                "total_consumed": scheduler.total_consumed,
                "pending_count": len(pending_markers),
                "admitted_order": admitted_order,
            },
            "business_code_counts": code_counts,
            "failures": failures,
            "snapshot_manifest": {k: v for k, v in manifest.items() if k != "pending"},
        }
        atomic_write_json(self.root / "reports" / "last_report.json", report)
        return report

    def _build_exchange_records(
        self,
        readings: dict[str, dict[str, Any]],
        image_no_by_name: dict[str, int],
    ) -> list[ExchangeRecord]:
        records: list[ExchangeRecord] = []
        for name in sorted(readings):
            reading = readings[name]
            image_no = image_no_by_name.get(name, MISSING)
            for entry in reading["entries"]:
                pending = entry.get("match_status") == MATCH_BUDGET_PENDING
                rec = ExchangeRecord(source_version="pipeline")
                rec.image_no = image_no
                rec.char_no = entry["reading_no"]
                rec.bbox = [float(v) for v in entry["bbox"]]
                rec.reading_order = entry["reading_order_name"]
                rec.keypoint_x = entry["authoritative_keypoint"]["x"] if entry.get("authoritative_keypoint") else MISSING
                rec.keypoint_y = entry["authoritative_keypoint"]["y"] if entry.get("authoritative_keypoint") else MISSING
                rec.keypoint_version = entry.get("keypoint_version", MISSING)
                rec.annotation_version = entry.get("annotation_version", MISSING)
                rec.present_fields = {"image_no", "char_no", "bbox", "reading_order"}
                rec.precision_labels = ["predicted_integer_pixel_bbox", "pipeline_float_confidence"]
                if pending or entry.get("confidence") is None:
                    rec.confidence = MISSING
                    rec.char = MISSING
                    rec.mark_missing("confidence")
                    rec.mark_missing("char")
                else:
                    rec.confidence = float(entry["confidence"])
                    rec.char = entry.get("recognized_char") or MISSING
                    rec.present_fields.update({"confidence", "char"})
                if is_missing(rec.keypoint_x):
                    rec.mark_missing("keypoint_x")
                    rec.mark_missing("keypoint_y")
                else:
                    rec.present_fields.update({"keypoint_x", "keypoint_y"})
                records.append(rec)
        return records

    def _write_exchange(self, records: list[ExchangeRecord]) -> dict[str, Any]:
        out_dir = self.root / "exchange"
        out_dir.mkdir(parents=True, exist_ok=True)
        v1_text, sidecar_entries = downgrade_v3_to_v1(records, self.sidecars, self.conversion_ledger)
        (out_dir / "results.v1.txt").write_text(v1_text, encoding="utf-8")
        v2_text = to_v2(records)
        (out_dir / "results.v2.json").write_text(v2_text, encoding="utf-8")
        v3_text = to_v3(records)
        (out_dir / "results.v3.txt").write_text(v3_text, encoding="utf-8")

        parsed_v1 = parse_v1(v1_text)
        parsed_v2 = parse_v2(v2_text)
        parsed_v3 = parse_v3(v3_text)
        restored = upgrade_v1_to_v3(parsed_v1, self.sidecars, self.conversion_ledger)
        restored_v3 = to_v3(restored)
        (out_dir / "restored_from_v1_sidecar.v3.txt").write_text(restored_v3, encoding="utf-8")
        reparsed_restored = parse_v3(restored_v3)
        return {
            "generated_records": len(records),
            "v1_receipts": len(parsed_v1),
            "v2_receipts": len(parsed_v2),
            "v3_receipts": len(parsed_v3),
            "restored_sidecar_receipts": len(reparsed_restored),
            "sidecar_entries": len(sidecar_entries),
            "parse_error_codes": [],
        }
