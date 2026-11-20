"""Deterministic concurrent page pipeline with incremental stage reuse."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .columns import COLUMNS_STAGE_VERSION, infer_columns
from .config import config_hash, stage_parameters
from .detector import DETECT_STAGE_VERSION, detect_blocks, validate_block_bounds
from .errors import (
    E_BLOCK_BOUNDS,
    E_IMAGE_INVALID,
    E_INTERNAL,
    E_ORDER_DUPLICATE,
    ConfigError,
    WorkbenchError,
)
from .faults import FaultRegistry
from .fingerprint import fingerprint, sha256_bytes, stage_cache_key
from .image_loader import IMAGE_STAGE_VERSION, gray_from_artifact, load_image_artifact
from .models import BBox, Block
from .ordering import (
    ORDER_STAGE_VERSION,
    sort_reading_order,
    validate_reading_order,
)
from .roles import ROLE_STAGE_VERSION, assign_roles
from .store import SnapshotStore
from .truth import compare_truth, load_truth


RESULT_SCHEMA_VERSION = 1


@dataclass
class PageStageOutcome:
    cache_key: str
    object_id: str | None
    status: str
    sort_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "object_id": self.object_id,
            "status": self.status,
            "stable_task_sort_key": self.sort_key,
        }


def _block_from_dict(value: dict[str, Any]) -> Block:
    return Block(
        block_id=str(value["block_id"]),
        bbox=BBox.from_any(value["bbox"]),
        ink_density=float(value.get("ink_density", 0.0)),
        projection_periodicity=float(value.get("projection_periodicity", 0.0)),
        role=value.get("role"),
        role_evidence=copy.deepcopy(value.get("role_evidence", {})),
        column_id=value.get("column_id"),
        is_spanning=bool(value.get("is_spanning", False)),
        reading_order=value.get("reading_order"),
        reading_order_rule=value.get("reading_order_rule"),
    )


def _blocks_from_payload(payload: dict[str, Any]) -> list[Block]:
    return [_block_from_dict(item) for item in payload.get("blocks", [])]


def _blocks_payload(blocks: list[Block]) -> list[dict[str, Any]]:
    return [block.to_dict() for block in blocks]


def _attach_fault_context(
    error: WorkbenchError, items: list[dict[str, Any]], stage: str
) -> WorkbenchError:
    if not items:
        return error
    injection_ids = list(error.details.get("injection_ids", []))
    for item in items:
        if item["id"] not in injection_ids:
            injection_ids.append(item["id"])
    error.details["injection_ids"] = sorted(injection_ids)
    if error.injection_id is None:
        error.injection_id = sorted(injection_ids)[0]
    error.stage = stage
    return error


def _apply_detect_faults(
    blocks: list[Block],
    fault_items: list[dict[str, Any]],
    width: int,
    height: int,
) -> None:
    if not fault_items:
        return
    for item in sorted(fault_items, key=lambda value: value["id"]):
        if item["kind"] != "block_out_of_bounds":
            continue
        index = int(item.get("parameters", {}).get("block_index", 0))
        if blocks:
            index = min(max(index, 0), len(blocks) - 1)
            block = blocks[index]
            block.bbox = BBox(block.bbox.x0, block.bbox.y0, width + 32, block.bbox.y1)
        else:
            blocks.append(
                Block(
                    block_id="b9999",
                    bbox=BBox(-8, 0, width + 32, max(1, min(height, 8))),
                    ink_density=0.0,
                    projection_periodicity=0.0,
                )
            )
    try:
        validate_block_bounds(blocks, width, height, "detect")
    except WorkbenchError as exc:
        ids = sorted(item["id"] for item in fault_items)
        raise WorkbenchError(
            E_BLOCK_BOUNDS,
            exc.message,
            "detect",
            injection_id=ids[0],
            details={**exc.details, "injection_ids": ids},
        ) from exc


def _apply_order_faults(
    blocks: list[Block], fault_items: list[dict[str, Any]]
) -> None:
    if not fault_items or not blocks:
        return
    if len(blocks) >= 2:
        blocks[1].reading_order = blocks[0].reading_order
    else:
        blocks[0].reading_order = 0
    try:
        validate_reading_order(blocks)
    except WorkbenchError as exc:
        ids = sorted(item["id"] for item in fault_items)
        raise WorkbenchError(
            E_ORDER_DUPLICATE,
            exc.message,
            "order",
            injection_id=ids[0],
            details={**exc.details, "injection_ids": ids},
        ) from exc


def _read_cached_stage(
    store: SnapshotStore,
    manifest: dict[str, Any] | None,
    page: str,
    stage: str,
    key: str,
) -> tuple[dict[str, Any], str | None] | None:
    wrapper = store.read_artifact(manifest, page, stage, key)
    if wrapper is None:
        return None
    payload = wrapper.get("payload")
    if not isinstance(payload, dict):
        return None
    entry = (manifest or {}).get("pages", {}).get(page, {}).get("stages", {}).get(stage, {})
    return payload, entry.get("object_id")


def _write_cached_stage(
    store: SnapshotStore,
    stage: str,
    version: int,
    key: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    wrapper = {
        "stage": stage,
        "stage_version": version,
        "cache_key": key,
        "payload": payload,
    }
    return payload, store.write_artifact(wrapper)


def _analysis_from_payloads(
    page: str,
    image_payload: dict[str, Any],
    detect_diagnostics: dict[str, Any],
    role_diagnostics: dict[str, Any],
    column_diagnostics: dict[str, Any],
    order_diagnostics: dict[str, Any],
    blocks: list[Block],
) -> dict[str, Any]:
    reading_sequence = list(order_diagnostics.get("reading_sequence", []))
    return {
        "page": page,
        "image": {
            "width": image_payload["width"],
            "height": image_payload["height"],
            "mode": image_payload["mode"],
            "content_hash": image_payload["content_hash"],
            "gray_sha256": image_payload["gray_sha256"],
        },
        "blocks": _blocks_payload(blocks),
        "column_groups": copy.deepcopy(column_diagnostics.get("column_groups", [])),
        "reading_order": reading_sequence,
        "reading_order_rule": order_diagnostics.get("reading_order_rule"),
        "diagnostics": {
            "detection": detect_diagnostics,
            "roles": role_diagnostics,
            "columns": column_diagnostics,
            "ordering": order_diagnostics,
            "summary": {
                "block_count": len(blocks),
                "column_count": column_diagnostics.get("column_count", 0),
                "column_layout": column_diagnostics.get("column_layout"),
                "spanning_block_count": len(column_diagnostics.get("spanning_blocks", [])),
                "column_conflict_count": column_diagnostics.get("conflict_count", 0),
                "reading_order_rule": order_diagnostics.get("reading_order_rule"),
            },
        },
    }


def _process_page(
    *,
    page: str,
    path: Path,
    config: dict[str, Any],
    fault_registry: FaultRegistry,
    store: SnapshotStore,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    stage_info: dict[str, PageStageOutcome] = {}
    cache_entries: dict[str, dict[str, Any]] = {}
    current_stage = "image"
    current_key: str | None = None

    def remember_reused(stage: str, key: str, object_id: str | None) -> None:
        stage_info[stage] = PageStageOutcome(
            key, object_id, "reused", f"{page}\x00{stage}"
        )
        if object_id is not None:
            cache_entries[stage] = {
                "cache_key": key,
                "object_id": object_id,
                "implementation_version": _implementation_version(stage),
            }

    def remember_recomputed(stage: str, key: str, object_id: str) -> None:
        stage_info[stage] = PageStageOutcome(
            key, object_id, "recomputed", f"{page}\x00{stage}"
        )
        cache_entries[stage] = {
            "cache_key": key,
            "object_id": object_id,
            "implementation_version": _implementation_version(stage),
        }

    try:
        current_stage = "image"
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise WorkbenchError(
                E_IMAGE_INVALID,
                f"unable to read page bytes: {exc}",
                "image",
            ) from exc
        raw_hash = sha256_bytes(raw_bytes)
        image_faults = fault_registry.stage_fingerprint(page, "image")
        current_key = stage_cache_key(
            "image",
            IMAGE_STAGE_VERSION,
            {"content_hash": raw_hash},
            stage_parameters(config, "image"),
            image_faults,
        )
        cached = _read_cached_stage(store, manifest, page, "image", current_key)
        if cached is not None:
            image_payload, object_id = cached
            remember_reused("image", current_key, object_id)
        else:
            faulted_bytes = fault_registry.truncate_image_bytes(page, raw_bytes)
            try:
                image_payload = load_image_artifact(
                    faulted_bytes, config["image_validation"]
                )
            except WorkbenchError as exc:
                raise _attach_fault_context(
                    exc,
                    [
                        item.fingerprint_value()
                        for item in fault_registry.for_stage(page, "image")
                    ],
                    "image",
                ) from exc
            image_payload, object_id = _write_cached_stage(
                store, "image", IMAGE_STAGE_VERSION, current_key, image_payload
            )
            remember_recomputed("image", current_key, object_id)

        current_stage = "detect"
        detect_faults = fault_registry.stage_fingerprint(page, "detect")
        current_key = stage_cache_key(
            "detect",
            DETECT_STAGE_VERSION,
            {
                "image_key": stage_info["image"].cache_key,
                "gray_sha256": image_payload["gray_sha256"],
            },
            stage_parameters(config, "detect"),
            detect_faults,
        )
        cached = _read_cached_stage(store, manifest, page, "detect", current_key)
        if cached is not None:
            detect_payload, object_id = cached
            blocks = _blocks_from_payload(detect_payload)
            detect_diagnostics = copy.deepcopy(detect_payload["diagnostics"])
            remember_reused("detect", current_key, object_id)
        else:
            gray = gray_from_artifact(image_payload)
            blocks, detect_diagnostics = detect_blocks(gray, config)
            if detect_faults:
                try:
                    _apply_detect_faults(
                        blocks,
                        detect_faults,
                        int(image_payload["width"]),
                        int(image_payload["height"]),
                    )
                except WorkbenchError:
                    raise
            else:
                validate_block_bounds(
                    blocks,
                    int(image_payload["width"]),
                    int(image_payload["height"]),
                    "detect",
                )
            detect_payload = {
                "blocks": _blocks_payload(blocks),
                "diagnostics": detect_diagnostics,
            }
            detect_payload, object_id = _write_cached_stage(
                store, "detect", DETECT_STAGE_VERSION, current_key, detect_payload
            )
            remember_recomputed("detect", current_key, object_id)

        current_stage = "role"
        role_faults = fault_registry.stage_fingerprint(page, "role")
        current_key = stage_cache_key(
            "role",
            ROLE_STAGE_VERSION,
            {"detect_key": stage_info["detect"].cache_key},
            stage_parameters(config, "role"),
            role_faults,
        )
        cached = _read_cached_stage(store, manifest, page, "role", current_key)
        if cached is not None:
            role_payload, object_id = cached
            blocks = _blocks_from_payload(role_payload)
            role_diagnostics = copy.deepcopy(role_payload["diagnostics"])
            remember_reused("role", current_key, object_id)
        else:
            blocks, role_diagnostics = assign_roles(
                blocks, int(image_payload["height"]), config
            )
            role_payload = {
                "blocks": _blocks_payload(blocks),
                "diagnostics": role_diagnostics,
            }
            role_payload, object_id = _write_cached_stage(
                store, "role", ROLE_STAGE_VERSION, current_key, role_payload
            )
            remember_recomputed("role", current_key, object_id)

        current_stage = "columns"
        column_faults = fault_registry.stage_fingerprint(page, "columns")
        current_key = stage_cache_key(
            "columns",
            COLUMNS_STAGE_VERSION,
            {"role_key": stage_info["role"].cache_key},
            stage_parameters(config, "columns"),
            column_faults,
        )
        cached = _read_cached_stage(store, manifest, page, "columns", current_key)
        if cached is not None:
            column_payload, object_id = cached
            blocks = _blocks_from_payload(column_payload)
            column_diagnostics = copy.deepcopy(column_payload["diagnostics"])
            remember_reused("columns", current_key, object_id)
        else:
            effective_config = copy.deepcopy(config)
            if any(item["kind"] == "column_gap_zero" for item in column_faults):
                effective_config["column_gap_threshold"] = 0
            try:
                blocks, column_diagnostics = infer_columns(
                    blocks,
                    int(image_payload["width"]),
                    int(image_payload["height"]),
                    effective_config,
                )
            except WorkbenchError as exc:
                raise _attach_fault_context(exc, column_faults, "columns") from exc
            column_payload = {
                "blocks": _blocks_payload(blocks),
                "diagnostics": column_diagnostics,
            }
            column_payload, object_id = _write_cached_stage(
                store, "columns", COLUMNS_STAGE_VERSION, current_key, column_payload
            )
            remember_recomputed("columns", current_key, object_id)

        current_stage = "order"
        order_faults = fault_registry.stage_fingerprint(page, "order")
        current_key = stage_cache_key(
            "order",
            ORDER_STAGE_VERSION,
            {"columns_key": stage_info["columns"].cache_key},
            stage_parameters(config, "order"),
            order_faults,
        )
        cached = _read_cached_stage(store, manifest, page, "order", current_key)
        if cached is not None:
            order_payload, object_id = cached
            blocks = _blocks_from_payload(order_payload)
            order_diagnostics = copy.deepcopy(order_payload["diagnostics"])
            remember_reused("order", current_key, object_id)
        else:
            blocks, order_diagnostics = sort_reading_order(
                blocks, config, column_diagnostics
            )
            if order_faults:
                try:
                    _apply_order_faults(blocks, order_faults)
                except WorkbenchError:
                    raise
            order_payload = {
                "blocks": _blocks_payload(blocks),
                "diagnostics": order_diagnostics,
            }
            order_payload, object_id = _write_cached_stage(
                store, "order", ORDER_STAGE_VERSION, current_key, order_payload
            )
            remember_recomputed("order", current_key, object_id)

        analysis = _analysis_from_payloads(
            page,
            image_payload,
            detect_diagnostics,
            role_diagnostics,
            column_diagnostics,
            order_diagnostics,
            blocks,
        )
        return {
            "page": page,
            "status": "ok",
            "stages": {stage: outcome.to_dict() for stage, outcome in stage_info.items()},
            "cache_entries": cache_entries,
            "analysis": analysis,
            "failure": None,
        }
    except ConfigError:
        raise
    except WorkbenchError as exc:
        stage_info[current_stage] = PageStageOutcome(
            current_key or "", None, "failed", f"{page}\x00{current_stage}"
        )
        failure = {
            "page": page,
            "stage": exc.stage,
            "code": exc.code,
            "message": exc.message,
            "injection_id": exc.injection_id,
            "injection_ids": list(exc.details.get("injection_ids", [])),
            "details": copy.deepcopy(exc.details),
        }
        return {
            "page": page,
            "status": "failed",
            "stages": {stage: outcome.to_dict() for stage, outcome in stage_info.items()},
            "cache_entries": cache_entries,
            "analysis": None,
            "failure": failure,
        }
    except Exception as exc:  # noqa: BLE001 - page isolation boundary
        stage_info[current_stage] = PageStageOutcome(
            current_key or "", None, "failed", f"{page}\x00{current_stage}"
        )
        failure = {
            "page": page,
            "stage": current_stage,
            "code": E_INTERNAL,
            "message": f"unexpected {type(exc).__name__}: {exc}",
            "injection_id": None,
            "injection_ids": [],
            "details": {"traceback": traceback.format_exc(limit=4)},
        }
        return {
            "page": page,
            "status": "failed",
            "stages": {stage: outcome.to_dict() for stage, outcome in stage_info.items()},
            "cache_entries": cache_entries,
            "analysis": None,
            "failure": failure,
        }


def _implementation_version(stage: str) -> int:
    return {
        "image": IMAGE_STAGE_VERSION,
        "detect": DETECT_STAGE_VERSION,
        "role": ROLE_STAGE_VERSION,
        "columns": COLUMNS_STAGE_VERSION,
        "order": ORDER_STAGE_VERSION,
    }[stage]


class _RunState:
    def __init__(self, base_manifest: dict[str, Any] | None, current_config_hash: str) -> None:
        self.base_manifest = base_manifest
        if base_manifest is None:
            self.pages: dict[str, Any] = {}
            self.completed: set[str] = set()
            self.failures: dict[str, dict[str, Any]] = {}
            self.last_completed_page: str | None = None
            self.last_processed_page: str | None = None
        else:
            self.pages = copy.deepcopy(base_manifest.get("pages", {}))
            self.completed = set(base_manifest.get("completed_pages", []))
            self.failures = {
                item["page"]: copy.deepcopy(item)
                for item in base_manifest.get("failed_pages", [])
                if isinstance(item, dict) and "page" in item
            }
            progress = base_manifest.get("progress", {})
            self.last_completed_page = progress.get("last_completed_page")
            self.last_processed_page = progress.get("last_processed_page")
        self.config_hash = current_config_hash

    def apply(self, result: dict[str, Any]) -> None:
        page = result["page"]
        self.pages[page] = {"stages": copy.deepcopy(result.get("cache_entries", {}))}
        self.last_processed_page = page
        if result["status"] == "ok":
            self.completed.add(page)
            self.failures.pop(page, None)
            self.last_completed_page = page
        else:
            self.completed.discard(page)
            failure = result["failure"]
            self.failures[page] = {
                "page": page,
                "stage": failure["stage"],
                "code": failure["code"],
                "message": failure["message"],
                "injection_ids": failure.get("injection_ids", []),
                "stages_completed": sorted(result.get("cache_entries", {})),
            }

    def manifest_payload(
        self,
        input_pages: set[str],
        interrupted: bool,
    ) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "completed_pages": sorted(self.completed & input_pages),
            "failed_pages": [
                self.failures[page]
                for page in sorted(self.failures)
                if page in input_pages
            ],
            "pages": {
                page: self.pages[page]
                for page in sorted(self.pages)
                if page in input_pages
            },
            "progress": {
                "last_completed_page": self.last_completed_page,
                "last_processed_page": self.last_processed_page,
                "interrupted": bool(interrupted),
            },
        }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _find_truth_path(input_dir: Path, explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    for name in ("layout_truth.json", "truth.json"):
        candidate = input_dir / name
        if candidate.exists():
            return candidate
    return None


def run_pipeline(
    *,
    input_dir: str | Path,
    config: dict[str, Any],
    fault_registry: FaultRegistry,
    state_dir: str | Path,
    truth_path: str | Path | None = None,
    workers: int | None = None,
    stop_event: threading.Event | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    source_dir = Path(input_dir)
    if not source_dir.is_dir():
        raise ConfigError(f"input directory does not exist: {source_dir}")
    page_paths = sorted(
        [
            entry
            for entry in source_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() == ".png"
        ],
        key=lambda entry: entry.name,
    )
    page_names = [entry.name for entry in page_paths]
    page_name_set = set(page_names)
    fault_registry.validate_pages(page_name_set)

    effective_workers = int(workers if workers is not None else config["thread_pool_workers"])
    if effective_workers < 1 or effective_workers > 64:
        raise ConfigError("worker count must be between 1 and 64")
    batches = int(config["pages_per_batch"])
    snapshot_interval = int(config["snapshot_interval_pages"])
    execution_mode = "serial" if effective_workers == 1 else "parallel"
    stop = stop_event or threading.Event()

    store = SnapshotStore(state_dir)
    base_manifest = store.load_manifest()
    current_config_hash = config_hash(config)
    state = _RunState(base_manifest, current_config_hash)
    results: list[dict[str, Any]] = []
    commit_versions: list[int] = []
    interrupted = False

    def commit(interrupted_now: bool) -> None:
        manifest = store.commit(
            state.manifest_payload(page_name_set, interrupted=interrupted_now)
        )
        version = int(manifest["global_version"])
        if version not in commit_versions:
            commit_versions.append(version)

    if page_paths:
        with ThreadPoolExecutor(
            max_workers=effective_workers, thread_name_prefix="layout-page"
        ) as executor:
            completed_count = 0
            for batch_start in range(0, len(page_paths), batches):
                if stop.is_set():
                    interrupted = True
                    break
                batch = page_paths[batch_start : batch_start + batches]
                future_by_page = {
                    path.name: executor.submit(
                        _process_page,
                        page=path.name,
                        path=path,
                        config=config,
                        fault_registry=fault_registry,
                        store=store,
                        manifest=base_manifest,
                    )
                    for path in batch
                }
                for path in batch:
                    result = future_by_page[path.name].result()
                    results.append(result)
                    state.apply(result)
                    completed_count += 1
                    if completed_count % snapshot_interval == 0:
                        commit(interrupted_now=stop.is_set())
                if stop.is_set():
                    interrupted = True
                    break
    all_processed = len(results) == len(page_paths)
    interrupted = bool(stop.is_set() and not all_processed)
    final_manifest = store.commit(
        state.manifest_payload(page_name_set, interrupted=interrupted)
    )
    final_version = int(final_manifest["global_version"])
    if final_version not in commit_versions:
        commit_versions.append(final_version)

    successful_results = sorted(
        [result for result in results if result["status"] == "ok"],
        key=lambda result: result["page"],
    )
    failure_results = sorted(
        [result for result in results if result["status"] == "failed"],
        key=lambda result: (result["page"], result["failure"]["stage"]),
    )
    analysis_pages = [result["analysis"] for result in successful_results]

    truth_file = _find_truth_path(source_dir, truth_path)
    if truth_file is not None:
        if not truth_file.exists():
            raise ConfigError(f"truth file does not exist: {truth_file}")
        truth_comparison = compare_truth(
            analysis_pages,
            load_truth(truth_file),
            str(config["reading_order_rule"]),
        )
        truth_comparison["truth_path"] = str(truth_file)
    else:
        truth_comparison = {
            "available": False,
            "reason": "no truth JSON supplied or found",
        }

    injection_records = {
        item.item_id: item.report_record() for item in fault_registry.items
    }
    result_by_page = {result["page"]: result for result in results}
    for item in fault_registry.items:
        record = injection_records[item.item_id]
        if item.page not in page_name_set:
            record.update(
                {
                    "status": "page_missing",
                    "business_code": item.expected_code,
                    "mapping": f"{item.kind}->{item.expected_code}",
                }
            )
            continue
        page_result = result_by_page.get(item.page)
        if page_result is None:
            record.update(
                {
                    "status": "not_evaluated",
                    "business_code": item.expected_code,
                    "mapping": f"{item.kind}->{item.expected_code}",
                    "reason": "page was not processed before graceful stop",
                }
            )
            continue
        failure = page_result.get("failure") or {}
        injection_ids = set(failure.get("injection_ids", []))
        if item.item_id in injection_ids:
            record.update(
                {
                    "hit": True,
                    "status": "hit",
                    "business_code": failure.get("code", item.expected_code),
                    "mapping": f"{item.kind}->{failure.get('code', item.expected_code)}",
                    "failure_stage": failure.get("stage"),
                }
            )
        else:
            stage_status = page_result.get("stages", {}).get(item.stage, {}).get("status")
            record.update(
                {
                    "status": "not_triggered" if stage_status else "stage_not_run",
                    "business_code": item.expected_code,
                    "mapping": f"{item.kind}->{item.expected_code}",
                }
            )
    injections = [injection_records[item_id] for item_id in sorted(injection_records)]

    def stage_count(status: str) -> int:
        return sum(
            1
            for result in results
            for stage in result.get("stages", {}).values()
            if stage.get("status") == status
        )

    summary = {
        "input_page_count": len(page_names),
        "processed_page_count": len(results),
        "completed_page_count": len(analysis_pages),
        "failed_page_count": len(failure_results),
        "pending_page_count": len(page_names) - len(results),
        "reused_stage_count": stage_count("reused"),
        "recomputed_stage_count": stage_count("recomputed"),
        "failed_stage_count": stage_count("failed"),
        "injection_count": len(injections),
        "injection_hit_count": sum(1 for item in injections if item.get("hit")),
        "truth_validation_passed": bool(truth_comparison.get("passed", False)),
    }

    recovery = {
        "resumed": base_manifest is not None,
        "base_snapshot_version": (
            int(base_manifest.get("global_version", 0)) if base_manifest else 0
        ),
        "recovered_pages": (
            list(base_manifest.get("completed_pages", [])) if base_manifest else []
        ),
        "recovery_point_page": (
            base_manifest.get("progress", {}).get("last_completed_page")
            if base_manifest
            else None
        ),
        "base_was_interrupted": bool(
            base_manifest.get("progress", {}).get("interrupted", False)
            if base_manifest
            else False
        ),
    }

    report: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "workbench": "deterministic-layout-reading-workbench",
        "status": "interrupted" if interrupted else "completed",
        "config_hash": current_config_hash,
        "execution": {
            "mode": execution_mode,
            "workers": effective_workers,
            "pages_per_batch": batches,
            "snapshot_interval_pages": snapshot_interval,
            "stable_task_sort_key": "{filename}\\x00{stage}",
            "thread_pool_scope": "page pipeline (detection through ordering)",
        },
        "snapshot": {
            "base_version": recovery["base_snapshot_version"],
            "committed_version": final_version,
            "commit_versions": sorted(set(commit_versions)),
            "manifest_file": str(store.manifest_path),
        },
        "recovery": recovery,
        "summary": summary,
        "pages": analysis_pages,
        "page_cache": [
            {
                "page": result["page"],
                "status": result["status"],
                "stages": copy.deepcopy(result.get("stages", {})),
            }
            for result in sorted(results, key=lambda item: item["page"])
        ],
        "failures": [result["failure"] for result in failure_results],
        "injections": injections,
        "truth_validation": truth_comparison,
        "pending_pages": page_names[len(results) :] if interrupted else [],
    }
    portable_truth_comparison = copy.deepcopy(truth_comparison)
    portable_truth_comparison.pop("truth_path", None)
    report["result_digest"] = fingerprint(
        {
            "pages": analysis_pages,
            "failures": report["failures"],
            "injections": [
                {
                    "id": item["id"],
                    "hit": item["hit"],
                    "business_code": item["business_code"],
                }
                for item in injections
            ],
            "truth_validation": portable_truth_comparison,
        }
    )

    if report_path is not None:
        _atomic_write_json(Path(report_path), report)
    return report
