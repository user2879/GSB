"""Deterministic orchestration, caching, scheduling, and diagnostics."""

from __future__ import annotations

import copy
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibration import recover_axis_mapping, restore_data_points
from .config import load_config, semantic_config
from .loader import PageRecord, load_sample_directory
from .snapshot import LoadedSnapshot, SnapshotStore
from .util import atomic_write_json, sha256_json, stable_round
from .vision import StageError, detect_graphic_candidates, detect_ticks


ENGINE_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = 1
TICK_ALGORITHM_VERSION = 1
MAPPING_ALGORITHM_VERSION = 1
DATA_ALGORITHM_VERSION = 1
STAGE_ORDER = {"ticks": 0, "mapping": 1, "data": 2}


class Workbench:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        serial: bool = False,
        stop_event: Optional[threading.Event] = None,
    ):
        self.config = config
        self.serial = serial
        self.stop_event = stop_event or threading.Event()
        self.sample_dir = Path(config["_sample_dir"])
        self.output_dir = Path(config["_output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        self.snapshot_store = SnapshotStore(self.output_dir)
        self._tick_params = {
            "algorithm_version": TICK_ALGORITHM_VERSION,
            "tick_min_length": int(config["tick_min_length"]),
            "line_width_tolerance": int(config["line_width_tolerance"]),
            "axis_detection_threshold": float(config["axis_detection_threshold"]),
        }
        self._mapping_params = {
            "algorithm_version": MAPPING_ALGORITHM_VERSION,
            "inverse_mode": str(config["inverse_mode"]),
            "log_axis_detection_threshold": float(config["log_axis_detection_threshold"]),
            "broken_axis_detection_threshold": float(config["broken_axis_detection_threshold"]),
            "label_occlusion_tolerance": float(config["label_occlusion_tolerance"]),
        }
        self._data_params = {
            "algorithm_version": DATA_ALGORITHM_VERSION,
            "line_width_tolerance": int(config["line_width_tolerance"]),
            "page_data_point_quota": int(config["page_data_point_quota"]),
            "quantization_step": float(config["quantization_step"]),
            "fixed_decimal_places": int(config["fixed_decimal_places"]),
        }

    def request_stop(self) -> None:
        self.stop_event.set()

    def _keys(self, record: PageRecord) -> Tuple[Dict[str, str], Dict[str, Any]]:
        fingerprints = {
            "ticks": self._tick_params,
            "mapping": self._mapping_params,
            "data": self._data_params,
        }
        tick_key = sha256_json(
            {"stage": "ticks", "image_sha256": record.image_sha256, "parameters": self._tick_params}
        )
        mapping_key = sha256_json(
            {"stage": "mapping", "tick_key": tick_key, "parameters": self._mapping_params}
        )
        data_key = sha256_json({"stage": "data", "mapping_key": mapping_key, "parameters": self._data_params})
        return {"ticks": tick_key, "mapping": mapping_key, "data": data_key}, fingerprints

    def _tick_worker(
        self,
        record: PageRecord,
        tick_key: str,
        cached_entry: Optional[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        if cached_entry is not None and cached_entry.get("key") == tick_key:
            return "reuse", cached_entry
        try:
            data = detect_ticks(record.gray, self.config)
            data["ok"] = True
        except StageError as exc:
            data = {"ok": False, "error": {"code": exc.code, "message": exc.message}}
        except Exception as exc:  # Keep one page's failure isolated from the batch.
            data = {"ok": False, "error": {"code": "E_PIPELINE_ERROR", "message": str(exc)}}
        return "recompute", {"key": tick_key, "data": data}

    def _mapping_and_data_worker(
        self,
        record: PageRecord,
        tick_entry: Dict[str, Any],
        keys: Dict[str, str],
        cached_mapping: Optional[Dict[str, Any]],
        cached_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tick_data = tick_entry["data"]
        if not tick_data.get("ok"):
            return {
                "mapping_action": "skipped",
                "mapping_entry": None,
                "data_action": "skipped",
                "data_entry": None,
            }

        mapping_action = "reuse"
        mapping_entry = cached_mapping
        if mapping_entry is None or mapping_entry.get("key") != keys["mapping"]:
            mapping_action = "recompute"
            try:
                axes = {
                    "x": recover_axis_mapping("x", tick_data["axes"]["x"], self.config),
                    "y": recover_axis_mapping("y", tick_data["axes"]["y"], self.config),
                }
                mapping_data = {"ok": True, "axes": axes}
            except StageError as exc:
                mapping_data = {"ok": False, "error": {"code": exc.code, "message": exc.message}}
            except Exception as exc:
                mapping_data = {
                    "ok": False,
                    "error": {"code": "E_PIPELINE_ERROR", "message": str(exc)},
                }
            mapping_entry = {"key": keys["mapping"], "data": mapping_data}

        if not mapping_entry["data"].get("ok"):
            return {
                "mapping_action": mapping_action,
                "mapping_entry": mapping_entry,
                "data_action": "skipped",
                "data_entry": None,
            }

        data_action = "reuse"
        data_entry = cached_data
        if data_entry is None or data_entry.get("key") != keys["data"]:
            data_action = "recompute"
            try:
                plot = tick_data["plot_box"]
                graphic = detect_graphic_candidates(
                    record.gray,
                    plot,
                    int(self.config["line_width_tolerance"]),
                )
                restored = restore_data_points(
                    graphic["candidates"],
                    graphic["chart_type"],
                    mapping_entry["data"]["axes"]["y"],
                    self.config,
                )
                data = {"ok": True, **restored}
            except StageError as exc:
                data = {"ok": False, "error": {"code": exc.code, "message": exc.message}, "data_points": []}
            except Exception as exc:
                data = {
                    "ok": False,
                    "error": {"code": "E_PIPELINE_ERROR", "message": str(exc)},
                    "data_points": [],
                }
            data_entry = {"key": keys["data"], "data": data}

        return {
            "mapping_action": mapping_action,
            "mapping_entry": mapping_entry,
            "data_action": data_action,
            "data_entry": data_entry,
        }

    def _prior_page(self, previous: Optional[LoadedSnapshot], page: str) -> Optional[Dict[str, Any]]:
        if previous is None:
            return None
        metadata = previous.state.get("pages", {}).get(page)
        return metadata if isinstance(metadata, dict) else None

    def _miss_reason(
        self,
        stage: str,
        page: str,
        keys: Dict[str, str],
        fingerprints: Dict[str, Any],
        prior_meta: Optional[Dict[str, Any]],
        image_changed: bool,
        page_added: bool,
    ) -> str:
        if page_added or prior_meta is None:
            return "page_added"
        if image_changed:
            return "image_changed"
        prior_fingerprints = prior_meta.get("fingerprints", {})
        prior_keys = prior_meta.get("keys", {})
        if stage == "ticks":
            if prior_fingerprints.get("ticks") != fingerprints["ticks"]:
                return "tick_parameters_changed"
        elif stage == "mapping":
            if prior_keys.get("ticks") != keys["ticks"]:
                return "tick_dependency_changed"
            if prior_fingerprints.get("mapping") != fingerprints["mapping"]:
                return "mapping_parameters_changed"
        else:
            if prior_keys.get("mapping") != keys["mapping"]:
                return "mapping_dependency_changed"
            if prior_fingerprints.get("data") != fingerprints["data"]:
                return "data_parameters_changed"
        return "cache_missing"

    def _tick_summary(self, tick_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not tick_data or not tick_data.get("ok"):
            error = (tick_data or {}).get("error", {})
            return {"ok": False, "error": error, "x": {"count": 0}, "y": {"count": 0}}
        result: Dict[str, Any] = {
            "ok": True,
            "plot_box": tick_data["plot_box"],
            "axis_detection": tick_data.get("axis_detection", {}),
        }
        for axis in ("x", "y"):
            axis_data = tick_data["axes"][axis]
            association_by_tick = {
                association["tick_id"]: association for association in axis_data["associations"]
            }
            ticks = []
            for tick in axis_data["ticks"]:
                association = association_by_tick.get(tick["id"])
                ticks.append(
                    {
                        "id": tick["id"],
                        "position": int(tick["position"]),
                        "length": int(tick["length"]),
                        "thickness": int(tick["thickness"]),
                        "sources": list(tick["sources"]),
                        "label_id": association["label_id"] if association else None,
                        "label_text": association["text"] if association else None,
                        "label_value": association["value"] if association else None,
                        "association_distance": association["distance"] if association else None,
                    }
                )
            result[axis] = {
                "count": len(ticks),
                "label_count": len(axis_data["labels"]),
                "association_count": len(axis_data["associations"]),
                "ticks": ticks,
                "labels": [
                    {
                        "id": label["id"],
                        "text": label["text"],
                        "value": label["value"],
                        "bbox": label["bbox"],
                        "anchor": label["anchor"],
                    }
                    for label in axis_data["labels"]
                ],
            }
        return result

    def _build_page_result(
        self,
        record: PageRecord,
        tick_entry: Dict[str, Any],
        mapping_entry: Optional[Dict[str, Any]],
        data_entry: Optional[Dict[str, Any]],
        stage_cache: Dict[str, Any],
        budget_points: int,
    ) -> Dict[str, Any]:
        tick_data = tick_entry["data"]
        errors: List[Dict[str, Any]] = []
        status = "ok"
        mappings: Dict[str, Any] = {}
        data_points: List[Dict[str, Any]] = []
        chart_type = None
        detected_count = 0

        if not tick_data.get("ok"):
            error = dict(tick_data.get("error", {"code": "E_PIPELINE_ERROR", "message": "tick stage failed"}))
            error["stage"] = "ticks"
            errors.append(error)
            status = "failed"
        elif mapping_entry is None:
            errors.append(
                {
                    "code": "E_CALIB_BUDGET_PENDING",
                    "message": "page has not completed axis mapping",
                    "stage": "mapping",
                }
            )
            status = "pending"
        elif not mapping_entry["data"].get("ok"):
            error = dict(
                mapping_entry["data"].get(
                    "error", {"code": "E_AXIS_MAPPING_FAILED", "message": "mapping stage failed"}
                )
            )
            error["stage"] = "mapping"
            errors.append(error)
            status = "failed"
        else:
            mappings = mapping_entry["data"]["axes"]
            if data_entry is None:
                errors.append(
                    {
                        "code": "E_CALIB_BUDGET_PENDING",
                        "message": "page has not completed data restoration",
                        "stage": "data",
                    }
                )
                status = "pending"
            elif not data_entry["data"].get("ok"):
                error = dict(
                    data_entry["data"].get(
                        "error", {"code": "E_DATA_POINTS_NOT_FOUND", "message": "data stage failed"}
                    )
                )
                error["stage"] = "data"
                errors.append(error)
                status = "failed"
            else:
                data_data = data_entry["data"]
                chart_type = data_data.get("chart_type")
                data_points = data_data.get("data_points", [])
                detected_count = data_data.get("detected_count", len(data_points))
                for error in data_data.get("errors", []):
                    staged = dict(error)
                    staged["stage"] = "data"
                    errors.append(staged)
                if errors and data_points:
                    status = "partial"
                elif errors and not data_points:
                    status = "failed"

        result = {
            "page": record.page,
            "status": status,
            "image": {
                "sha256": record.image_sha256,
                "width": record.width,
                "height": record.height,
                "mode": record.source_mode,
                "truth_sha256": record.truth_sha256,
            },
            "chart_type": chart_type,
            "ticks": self._tick_summary(tick_data),
            "axis_mappings": mappings,
            "data_points": data_points,
            "data_point_detected_count": detected_count,
            "tick_comparison_points": int(budget_points),
            "errors": errors,
            "stage_cache": stage_cache,
        }
        result["verification"] = self._verify(record, result)
        return result

    def _verify(self, record: PageRecord, page_result: Dict[str, Any]) -> Dict[str, Any]:
        truth = record.truth
        tick_result = page_result.get("ticks", {})
        position_errors: List[int] = []
        value_mismatches = 0
        matched_ticks = 0
        if tick_result.get("ok"):
            for axis in ("x", "y"):
                truth_axis = truth["axes"][axis]
                detected = list(tick_result[axis]["ticks"])
                used: set[str] = set()
                for truth_position, truth_value in zip(
                    truth_axis["pixel_positions"], truth_axis["values"]
                ):
                    candidates = [
                        tick
                        for tick in detected
                        if tick["id"] not in used
                    ]
                    if not candidates:
                        continue
                    candidates.sort(key=lambda tick: (abs(tick["position"] - truth_position), tick["position"], tick["id"]))
                    best = candidates[0]
                    distance = abs(int(best["position"]) - int(truth_position))
                    if distance > 2:
                        continue
                    used.add(best["id"])
                    matched_ticks += 1
                    position_errors.append(distance)
                    detected_value = best.get("label_value")
                    if detected_value is None:
                        continue
                    if abs(float(detected_value) - float(truth_value)) > 1e-9:
                        value_mismatches += 1

        truth_points = truth["data_points"]
        recovered_points = page_result.get("data_points", [])
        pixel_errors: List[float] = []
        value_abs_errors: List[float] = []
        value_rel_errors: List[float] = []
        mappings = page_result.get("axis_mappings", {})
        for truth_point, recovered in zip(truth_points, recovered_points):
            pixel_errors.append(
                math.hypot(
                    truth_point["pixel"][0] - recovered["pixel"][0],
                    truth_point["pixel"][1] - recovered["pixel"][1],
                )
            )
            absolute = abs(float(recovered["value"]) - float(truth_point["value"]))
            value_abs_errors.append(absolute)
            denominator = max(1.0, abs(float(truth_point["value"])))
            value_rel_errors.append(absolute / denominator)

        expected_mapping = truth.get("expected_mapping", {})
        mapping_matches = {
            axis: bool(mappings.get(axis) and mappings[axis].get("mapping_type") == expected_mapping.get(axis))
            for axis in ("x", "y")
        }
        max_value_abs = max(value_abs_errors) if value_abs_errors else 0.0
        max_value_rel = max(value_rel_errors) if value_rel_errors else 0.0
        tolerance_step = float(self.config["quantization_step"])
        y_type = mappings.get("y", {}).get("mapping_type") if mappings else None
        if y_type == "log10":
            values_within = bool(value_rel_errors) and max_value_rel <= 0.02
        else:
            values_within = bool(value_abs_errors) and max_value_abs <= max(1.0, tolerance_step * 2.0)
        within_tolerance = bool(
            len(recovered_points) == len(truth_points)
            and all(mapping_matches.values())
            and (not position_errors or max(position_errors) <= 1)
            and value_mismatches == 0
            and (not pixel_errors or max(pixel_errors) <= 1.0)
            and values_within
        )
        return {
            "truth_tick_count": sum(len(truth["axes"][axis]["pixel_positions"]) for axis in ("x", "y")),
            "matched_tick_count": matched_ticks,
            "tick_position_max_abs_error": max(position_errors) if position_errors else None,
            "tick_value_mismatch_count": value_mismatches,
            "truth_data_point_count": len(truth_points),
            "recovered_data_point_count": len(recovered_points),
            "data_pixel_max_distance": stable_round(max(pixel_errors), 9) if pixel_errors else None,
            "data_value_max_abs_error": stable_round(max_value_abs, 9) if value_abs_errors else None,
            "data_value_max_relative_error": stable_round(max_value_rel, 9) if value_rel_errors else None,
            "expected_mapping": expected_mapping,
            "actual_mapping": {axis: mappings.get(axis, {}).get("mapping_type") for axis in ("x", "y")},
            "mapping_matches": mapping_matches,
            "within_tolerance": within_tolerance,
        }

    def _schedule_budget(self, costs: Dict[str, int]) -> Dict[str, Any]:
        budget = int(self.config["tick_comparison_budget"])
        remaining_pages = list(costs.keys())
        batches: List[Dict[str, Any]] = []
        pending_events: List[Dict[str, Any]] = []
        oversized: List[str] = []
        batch_number = 0
        while remaining_pages:
            batch_number += 1
            available = budget
            charges: List[Dict[str, Any]] = []
            while remaining_pages:
                page = remaining_pages[0]
                cost = int(costs[page])
                if cost > budget:
                    oversized.append(page)
                    pending_events.append(
                        {
                            "code": "E_CALIB_BUDGET_PENDING",
                            "batch": batch_number,
                            "page": page,
                            "remaining": available,
                            "required": cost,
                            "reason": "page_cost_exceeds_batch_budget",
                        }
                    )
                    remaining_pages.pop(0)
                    continue
                if cost <= available:
                    charges.append({"page": page, "points": cost})
                    available -= cost
                    remaining_pages.pop(0)
                    continue
                pending_events.append(
                    {
                        "code": "E_CALIB_BUDGET_PENDING",
                        "batch": batch_number,
                        "page": page,
                        "remaining": available,
                        "required": cost,
                        "reason": "batch_budget_exhausted",
                    }
                )
                break
            if charges:
                batches.append(
                    {
                        "batch": batch_number,
                        "limit": budget,
                        "remaining_start": budget,
                        "remaining_end": available,
                        "charges": charges,
                    }
                )
            elif not remaining_pages:
                break
        return {
            "limit_per_batch": budget,
            "batches": batches,
            "pending_events": pending_events,
            "oversized_pages": oversized,
            "planned_total": sum(int(cost) for page, cost in costs.items() if page not in oversized),
            "consumed_total": 0,
        }

    def _state_payload(
        self,
        page_order: Sequence[str],
        completed_pages: Sequence[str],
        page_meta: Dict[str, Any],
        consumed_budget: int,
        budget_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "engine_version": ENGINE_VERSION,
            "config_hash": self.config["_config_hash"],
            "page_order": list(page_order),
            "completed_pages": list(completed_pages),
            "consumed_budget": int(consumed_budget),
            "budget": copy.deepcopy(budget_report),
            "pages": copy.deepcopy(page_meta),
        }

    def _prune_artifacts(
        self,
        page_order: Sequence[str],
        page_meta: Dict[str, Any],
        cache: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for page in page_order:
            keys = page_meta.get(page, {}).get("keys", {})
            entries: Dict[str, Dict[str, Any]] = {}
            for stage in ("ticks", "mapping", "data"):
                entry = cache.get(page, {}).get(stage)
                if entry is not None and entry.get("key") == keys.get(stage):
                    entries[stage] = entry
            if entries:
                result[page] = entries
        return result

    def _commit(
        self,
        page_order: Sequence[str],
        completed_pages: Sequence[str],
        page_meta: Dict[str, Any],
        cache: Dict[str, Dict[str, Dict[str, Any]]],
        budget_report: Dict[str, Any],
        costs: Dict[str, int],
    ) -> Dict[str, Any]:
        consumed = sum(int(costs.get(page, 0)) for page in completed_pages)
        budget_report["consumed_total"] = consumed
        state = self._state_payload(page_order, completed_pages, page_meta, consumed, budget_report)
        artifacts = self._prune_artifacts(page_order, page_meta, cache)
        return self.snapshot_store.commit(state, artifacts)

    def _load_previous(self) -> Optional[LoadedSnapshot]:
        try:
            return self.snapshot_store.load()
        except Exception:
            return None

    def run(self, *, stop_after_pages: Optional[int] = None) -> Dict[str, Any]:
        loaded = load_sample_directory(self.sample_dir)
        valid_pages = list(loaded.pages)
        valid_names = [page.page for page in valid_pages]
        valid_set = set(valid_names)
        loader_failures = list(loaded.failures)
        discovered_pages = sorted(valid_set | {failure.page for failure in loader_failures})
        previous = self._load_previous()
        config_changed = bool(previous and previous.config_hash != self.config["_config_hash"])

        previous_pages = set(previous.state.get("pages", {})) if previous else set()
        added_pages = sorted(valid_set - previous_pages)
        removed_pages = sorted(previous_pages - valid_set)

        cache: Dict[str, Dict[str, Dict[str, Any]]] = {
            page: copy.deepcopy(previous.artifacts.get(page, {}))
            for page in valid_names
        } if previous else {page: {} for page in valid_names}
        cache_actions: Dict[str, Dict[str, Dict[str, str]]] = {
            page: {} for page in valid_names
        }
        page_meta: Dict[str, Dict[str, Any]] = {}
        keys_by_page: Dict[str, Dict[str, str]] = {}
        fingerprints_by_page: Dict[str, Dict[str, Any]] = {}
        prior_meta_by_page: Dict[str, Optional[Dict[str, Any]]] = {}
        image_changed_by_page: Dict[str, bool] = {}

        for record in valid_pages:
            keys, fingerprints = self._keys(record)
            keys_by_page[record.page] = keys
            fingerprints_by_page[record.page] = fingerprints
            prior_meta = self._prior_page(previous, record.page)
            prior_meta_by_page[record.page] = prior_meta
            image_changed = bool(
                prior_meta is not None and prior_meta.get("image_sha256") not in {None, record.image_sha256}
            )
            image_changed_by_page[record.page] = image_changed
            page_meta[record.page] = {
                "image_sha256": record.image_sha256,
                "truth_sha256": record.truth_sha256,
                "fingerprints": fingerprints,
                "keys": keys,
            }

        workers = 1 if self.serial else max(1, int(self.config["thread_pool_concurrency"]))
        tick_entries: Dict[str, Dict[str, Any]] = {}
        needs_commit = False
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(valid_pages)))) as executor:
            futures = {
                executor.submit(
                    self._tick_worker,
                    record,
                    keys_by_page[record.page]["ticks"],
                    cache.get(record.page, {}).get("ticks"),
                ): record
                for record in valid_pages
            }
            tick_results = []
            for future in as_completed(futures):
                record = futures[future]
                action, entry = future.result()
                tick_results.append((record, action, entry))
        for record, action, entry in sorted(tick_results, key=lambda item: item[0].page):
            tick_entries[record.page] = entry
            cache.setdefault(record.page, {})["ticks"] = entry
            reason = None
            if action != "reuse":
                reason = self._miss_reason(
                    "ticks",
                    record.page,
                    keys_by_page[record.page],
                    fingerprints_by_page[record.page],
                    prior_meta_by_page[record.page],
                    image_changed_by_page[record.page],
                    record.page in added_pages,
                )
                needs_commit = True
            cache_actions[record.page]["ticks"] = {"action": action}
            if reason:
                cache_actions[record.page]["ticks"]["reason"] = reason

        costs: Dict[str, int] = {}
        for record in valid_pages:
            tick_data = tick_entries[record.page]["data"]
            if tick_data.get("ok"):
                costs[record.page] = sum(
                    len(tick_data["axes"][axis]["associations"]) for axis in ("x", "y")
                )
            else:
                costs[record.page] = 0
        budget_report = self._schedule_budget(costs)

        page_results: Dict[str, Dict[str, Any]] = {}
        previous_completed = set(previous.state.get("completed_pages", [])) if previous else set()
        precompleted: set[str] = set()
        for page in valid_names:
            if page not in previous_completed:
                continue
            tick_cached = cache.get(page, {}).get("ticks")
            if tick_cached is None or tick_cached.get("key") != keys_by_page[page]["ticks"]:
                continue
            tick_data = tick_cached.get("data", {})
            if not tick_data.get("ok"):
                precompleted.add(page)
                cache_actions[page]["mapping"] = {"action": "skipped", "reason": "tick_stage_failed"}
                cache_actions[page]["data"] = {"action": "skipped", "reason": "tick_stage_failed"}
                continue
            mapping_cached = cache.get(page, {}).get("mapping")
            if mapping_cached is None or mapping_cached.get("key") != keys_by_page[page]["mapping"]:
                continue
            if not mapping_cached.get("data", {}).get("ok"):
                precompleted.add(page)
                cache_actions[page]["mapping"] = {"action": "reuse"}
                cache_actions[page]["data"] = {"action": "skipped", "reason": "mapping_stage_failed"}
                continue
            data_cached = cache.get(page, {}).get("data")
            if data_cached is not None and data_cached.get("key") == keys_by_page[page]["data"]:
                precompleted.add(page)
                cache_actions[page]["mapping"] = {"action": "reuse"}
                cache_actions[page]["data"] = {"action": "reuse"}
        completed_set: set[str] = set(precompleted)
        processed_this_run = 0
        last_commit_count = len([page for page in valid_names if page in precompleted])
        final_state_payload: Optional[Dict[str, Any]] = copy.deepcopy(previous.state) if previous else None
        tick_recomputed = any(
            cache_actions.get(page, {}).get("ticks", {}).get("action") == "recompute"
            for page in valid_names
        )
        truth_changed = any(
            prior_meta_by_page.get(record.page) is not None
            and prior_meta_by_page[record.page].get("truth_sha256") not in {None, record.truth_sha256}
            for record in valid_pages
        )
        needs_commit = (
            bool(previous and previous.config_hash != self.config["_config_hash"])
            or bool(removed_pages)
            or tick_recomputed
            or truth_changed
        )
        stopped = self.stop_event.is_set()

        if not stopped:
            for batch in budget_report["batches"]:
                if self.stop_event.is_set():
                    stopped = True
                    break
                batch_pages = [charge["page"] for charge in batch["charges"]]
                with ThreadPoolExecutor(max_workers=min(workers, max(1, len(batch_pages)))) as executor:
                    futures = {
                        executor.submit(
                            self._mapping_and_data_worker,
                            next(record for record in valid_pages if record.page == page),
                            tick_entries[page],
                            keys_by_page[page],
                            cache.get(page, {}).get("mapping"),
                            cache.get(page, {}).get("data"),
                        ): page
                        for page in batch_pages
                    }
                    batch_results = []
                    for future in as_completed(futures):
                        page = futures[future]
                        batch_results.append((page, future.result()))
                for page, result in sorted(batch_results, key=lambda item: item[0]):
                    mapping_entry = result["mapping_entry"]
                    data_entry = result["data_entry"]
                    if mapping_entry is not None:
                        cache.setdefault(page, {})["mapping"] = mapping_entry
                    if data_entry is not None:
                        cache.setdefault(page, {})["data"] = data_entry
                    mapping_action = result["mapping_action"]
                    data_action = result["data_action"]
                    if mapping_action != "reuse":
                        reason = self._miss_reason(
                            "mapping",
                            page,
                            keys_by_page[page],
                            fingerprints_by_page[page],
                            prior_meta_by_page[page],
                            image_changed_by_page[page],
                            page in added_pages,
                        )
                        cache_actions[page]["mapping"] = {"action": mapping_action, "reason": reason}
                    else:
                        cache_actions[page]["mapping"] = {"action": mapping_action}
                    if data_action != "reuse":
                        reason = self._miss_reason(
                            "data",
                            page,
                            keys_by_page[page],
                            fingerprints_by_page[page],
                            prior_meta_by_page[page],
                            image_changed_by_page[page],
                            page in added_pages,
                        )
                        cache_actions[page]["data"] = {"action": data_action, "reason": reason}
                    else:
                        cache_actions[page]["data"] = {"action": data_action}
                    stage_cache = {
                        stage: {"key": keys_by_page[page][stage]}
                        for stage in ("ticks", "mapping", "data")
                    }
                    record = next(record for record in valid_pages if record.page == page)
                    page_results[page] = self._build_page_result(
                        record,
                        tick_entries[page],
                        mapping_entry,
                        data_entry,
                        stage_cache,
                        costs.get(page, 0),
                    )
                    newly_completed = page not in completed_set
                    completed_set.add(page)
                    if newly_completed and page not in precompleted:
                        processed_this_run += 1
                    if (
                        newly_completed
                        or mapping_action == "recompute"
                        or data_action == "recompute"
                    ):
                        needs_commit = True
                    page_meta[page]["status"] = page_results[page]["status"]
                    page_meta[page]["completed"] = True
                completed_order = [page for page in valid_names if page in completed_set]
                if len(completed_order) - last_commit_count >= int(
                    self.config["snapshot_commit_interval_pages"]
                ):
                    final_state_payload = self._commit(
                        valid_names, completed_order, page_meta, cache, budget_report, costs
                    )
                    last_commit_count = len(completed_order)
                    needs_commit = False
                if stop_after_pages is not None and processed_this_run >= int(stop_after_pages):
                    stopped = True
                    break

        if self.stop_event.is_set() or stop_after_pages is not None and processed_this_run >= int(stop_after_pages or 0):
            stopped = True

        completed_order = [page for page in valid_names if page in completed_set]
        pending_pages = [page for page in valid_names if page not in completed_set]
        if needs_commit or final_state_payload is None:
            final_state_payload = self._commit(
                valid_names, completed_order, page_meta, cache, budget_report, costs
            )
        else:
            # Re-commit only when completion state advanced since the last periodic commit.
            if len(completed_order) != last_commit_count:
                final_state_payload = self._commit(
                    valid_names, completed_order, page_meta, cache, budget_report, costs
                )

        if pending_pages:
            existing_events = {event["page"] for event in budget_report["pending_events"] if event.get("page") in pending_pages}
            for page in pending_pages:
                if page not in existing_events:
                    budget_report["pending_events"].append(
                        {
                            "code": "E_CALIB_BUDGET_PENDING",
                            "page": page,
                            "reason": "stopped_before_page_completion",
                        }
                    )
                    needs_commit = True
            if needs_commit or len(completed_order) != last_commit_count:
                final_state_payload = self._commit(
                    valid_names, completed_order, page_meta, cache, budget_report, costs
                )
                last_commit_count = len(completed_order)

        pages_output: List[Dict[str, Any]] = []
        failure_records: List[Dict[str, Any]] = []
        loader_failure_map = {failure.page: failure for failure in loader_failures}
        for page in discovered_pages:
            if page in page_results:
                pages_output.append(page_results[page])
            elif page in completed_set:
                record = next(record for record in valid_pages if record.page == page)
                stage_cache = {
                    stage: {"key": keys_by_page[page][stage]}
                    for stage in ("ticks", "mapping", "data")
                }
                pages_output.append(
                    self._build_page_result(
                        record,
                        tick_entries[page],
                        cache.get(page, {}).get("mapping"),
                        cache.get(page, {}).get("data"),
                        stage_cache,
                        costs.get(page, 0),
                    )
                )
            elif page in valid_set:
                record = next(record for record in valid_pages if record.page == page)
                stage_cache = {
                    stage: {"key": keys_by_page[page][stage]}
                    for stage in ("ticks", "mapping", "data")
                }
                pending_result = self._build_page_result(
                    record,
                    tick_entries[page],
                    None,
                    None,
                    stage_cache,
                    costs.get(page, 0),
                )
                pending_result["status"] = "pending"
                pages_output.append(pending_result)
            else:
                loader_failure = loader_failure_map.get(page)
                pages_output.append(
                    {
                        "page": page,
                        "status": "invalid",
                        "image": None,
                        "chart_type": None,
                        "ticks": {"ok": False, "x": {"count": 0}, "y": {"count": 0}},
                        "axis_mappings": {},
                        "data_points": [],
                        "data_point_detected_count": 0,
                        "tick_comparison_points": 0,
                        "errors": [
                            {
                                "stage": "load",
                                "code": "E_CHART_INVALID",
                                "message": loader_failure.message if loader_failure else "page failed input validation",
                            }
                        ],
                        "stage_cache": {},
                        "verification": {"within_tolerance": False},
                    }
                )

        for result in pages_output:
            for error in result.get("errors", []):
                stage = error.get("stage", "unknown")
                if error.get("code") == "E_CALIB_BUDGET_PENDING":
                    stage = "scheduler"
                failure_records.append(
                    {
                        "page": result["page"],
                        "code": error.get("code", "E_PIPELINE_ERROR"),
                        "message": error.get("message", ""),
                        "stage": stage,
                    }
                )
        deduplicated_failures: Dict[Tuple[str, str], Dict[str, Any]] = {}
        stage_priority = {"load": 0, "ticks": 1, "mapping": 2, "data": 3, "scheduler": 4}
        for record in failure_records:
            key = (record["page"], record["code"])
            current = deduplicated_failures.get(key)
            if current is None or stage_priority.get(record["stage"], 9) < stage_priority.get(current["stage"], 9):
                deduplicated_failures[key] = record
        failure_records = list(deduplicated_failures.values())
        failure_records.sort(key=lambda item: (item["page"], {"load": 0, "ticks": 1, "mapping": 2, "data": 3, "scheduler": 4}.get(item["stage"], 9), item["code"]))

        reuse_records: List[Dict[str, str]] = []
        recompute_records: List[Dict[str, Any]] = []
        skipped_records: List[Dict[str, str]] = []
        config_changed_pages: set[str] = set()
        for page in valid_names:
            for stage in ("ticks", "mapping", "data"):
                action = cache_actions.get(page, {}).get(stage)
                if action is None:
                    continue
                if action["action"] == "reuse":
                    reuse_records.append({"page": page, "stage": stage})
                elif action["action"] == "recompute":
                    record = {"page": page, "stage": stage, "reason": action.get("reason", "cache_missing")}
                    recompute_records.append(record)
                    if image_changed_by_page.get(page) is False and record["reason"] in {
                        "tick_parameters_changed",
                        "mapping_parameters_changed",
                        "data_parameters_changed",
                        "tick_dependency_changed",
                        "mapping_dependency_changed",
                    }:
                        config_changed_pages.add(page)
                elif action["action"] == "skipped":
                    skipped_records.append({"page": page, "stage": stage})
        reuse_records.sort(key=lambda item: (item["page"], STAGE_ORDER[item["stage"]]))
        recompute_records.sort(key=lambda item: (item["page"], STAGE_ORDER[item["stage"]]))
        skipped_records.sort(key=lambda item: (item["page"], STAGE_ORDER[item["stage"]]))

        cache_stats = {
            "reused": len(reuse_records),
            "recomputed": len(recompute_records),
            "skipped": len(skipped_records),
            "ticks_reused": sum(1 for item in reuse_records if item["stage"] == "ticks"),
            "ticks_recomputed": sum(1 for item in recompute_records if item["stage"] == "ticks"),
            "mapping_reused": sum(1 for item in reuse_records if item["stage"] == "mapping"),
            "mapping_recomputed": sum(1 for item in recompute_records if item["stage"] == "mapping"),
            "data_reused": sum(1 for item in reuse_records if item["stage"] == "data"),
            "data_recomputed": sum(1 for item in recompute_records if item["stage"] == "data"),
        }

        consumed_budget = sum(int(costs.get(page, 0)) for page in completed_order)
        budget_report["consumed_total"] = consumed_budget
        summary = {
            "pages_discovered": len(discovered_pages),
            "pages_valid": len(valid_pages),
            "pages_ok": sum(1 for result in pages_output if result["status"] == "ok"),
            "pages_partial": sum(1 for result in pages_output if result["status"] == "partial"),
            "pages_failed": sum(1 for result in pages_output if result["status"] in {"failed", "invalid"}),
            "pages_pending": sum(1 for result in pages_output if result["status"] == "pending"),
            "pages_completed": len(completed_order),
            "data_points": sum(len(result.get("data_points", [])) for result in pages_output),
            "tick_comparison_points": sum(int(costs.get(page, 0)) for page in valid_names),
            "budget_consumed_total": consumed_budget,
            "budget_batch_count": len(budget_report["batches"]),
            "failures": len(failure_records),
            "verification_passed_pages": sum(
                1 for result in pages_output if result.get("verification", {}).get("within_tolerance")
            ),
        }

        result_core = {
            "pages": [
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"stage_cache"}
                }
                for result in pages_output
            ],
            "failures": failure_records,
            "budget": budget_report,
        }
        result_digest = sha256_json(result_core)

        artifact_versions: Dict[str, Dict[str, str]] = {}
        if final_state_payload is not None:
            for page, metadata in final_state_payload.get("pages", {}).items():
                versions = {
                    stage: info.get("version")
                    for stage, info in metadata.get("artifacts", {}).items()
                }
                if versions:
                    artifact_versions[page] = versions

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "config_hash": self.config["_config_hash"],
            "config": semantic_config(self.config),
            "result_digest": result_digest,
            "summary": summary,
            "pages": pages_output,
            "failures": failure_records,
            "cache": {
                "previous_version": previous.version if previous else 0,
                "config_changed": config_changed,
                "added_pages": added_pages,
                "removed_pages": removed_pages,
                "config_changed_pages": sorted(config_changed_pages),
                "reused": reuse_records,
                "recomputed": recompute_records,
                "recompute": recompute_records,
                "skipped": skipped_records,
                "stats": cache_stats,
            },
            "budget": budget_report,
            "snapshot": {
                "version": final_state_payload.get("version") if final_state_payload else 0,
                "config_hash": self.config["_config_hash"],
                "completed_pages": completed_order,
                "pending_pages": pending_pages,
                "consumed_budget": consumed_budget,
                "artifact_versions": artifact_versions,
            },
            "stopped": stopped,
        }
        self._write_report(report)
        return report

    def _write_report(self, report: Dict[str, Any]) -> None:
        atomic_write_json(self.output_dir / "report.json", report)
        atomic_write_json(self.output_dir / "reports" / "latest.json", report)
        digest_prefix = str(report["config_hash"])[:12]
        atomic_write_json(self.output_dir / "reports" / f"run-{digest_prefix}.json", report)


def run_workbench(
    config_path: Path,
    *,
    output_override: Optional[Path] = None,
    serial: bool = False,
    stop_after_pages: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    config = load_config(config_path, output_override=output_override)
    workbench = Workbench(config, serial=serial, stop_event=stop_event)
    return workbench.run(stop_after_pages=stop_after_pages)
