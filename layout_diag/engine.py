"""End-to-end incremental diagnostic engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .adaptive import select_effective_config
from .cache import (
    DiagnosticCache,
    component_key,
    fingerprint_bundle,
    keypoint_column_key,
    reading_key,
    reuse_entry,
    sliding_key,
)
from .components import run_component_stage
from .config import Config, EffectiveConfig
from .constants import CONFLICT_TYPES, SCOPE_ORDER, STAGE_ORDER
from .keypoints import assign_components_to_columns, regression_metadata, run_keypoint_column
from .models import Dataset, FailureLog
from .reading import index_all_keypoints, run_reading_stage
from .scheduler import build_schedule, stable_task_key
from .sliding import run_sliding_stage
from .snapshot import SnapshotStore, decide_run_mode
from .utils import file_hash, stable_hash


@dataclass
class RunResult:
    snapshot_dir: Path
    documents: dict[str, Any]
    manifest: dict[str, Any]

    def canonical_results(self, *, include_reuse: bool = True) -> dict[str, Any]:
        """Return the deterministic diagnostic payload suitable for equality checks."""

        report = deepcopy(self.documents.get("report", {}))
        if not include_reuse:
            report.pop("cache", None)
            report.pop("run_decision", None)
            report.pop("snapshot_recovery", None)
        payload = {
            "schedule": self.documents.get("schedule", {}),
            "coarse": self.documents.get("coarse", {}),
            "components": self.documents.get("components", {}),
            "keypoints": self.documents.get("keypoints", {}),
            "reading_order": self.documents.get("reading_order", {}),
            "failures": self.documents.get("failures", {}),
            "report": report,
            "manifest": {
                key: self.manifest.get(key)
                for key in (
                    "schema_version",
                    "global_version",
                    "dataset_hash",
                    "config_hash",
                    "effective_hash",
                    "tier",
                    "keypoint_version",
                    "completed_images",
                    "pending_images",
                    "failed_images",
                    "completed_columns",
                    "point_budget",
                    "completed_keypoints",
                    "pending_points",
                )
            },
        }
        if include_reuse:
            payload["reuse"] = self.documents.get("reuse", {})
        return payload


class DiagnosticEngine:
    def __init__(
        self,
        dataset: Dataset,
        config: Config,
        output_root: str | Path,
        *,
        failures: FailureLog | None = None,
        force_rebuild: bool = False,
        serial: bool = False,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.output_root = Path(output_root)
        self.failures = failures if failures is not None else FailureLog()
        self.force_rebuild = force_rebuild
        self.serial = serial

    def run(self) -> RunResult:
        validation_failures = self.failures.as_dicts()
        effective, tier_basis = select_effective_config(self.config, self.dataset.stats())
        schedule = build_schedule(self.dataset, effective, self.failures)
        store = SnapshotStore(self.output_root)
        previous_manifest = store.load_manifest()
        previous_cache_data = None if self.force_rebuild else store.load_cache()
        current_complete = _all_images_complete(self.dataset, schedule)
        decision = decide_run_mode(
            previous_manifest,
            dataset_hash=self.dataset.content_hash,
            config_hash=self.config.content_hash(),
            effective_hash=effective.params_hash(),
            all_images_complete=current_complete,
            force_rebuild=self.force_rebuild,
        )
        cache = DiagnosticCache(previous_cache_data)
        reuse: list[dict[str, Any]] = []

        image_states = self._localize_images(effective, cache, reuse)
        keypoint_tasks = self._prepare_keypoint_tasks(image_states, schedule, effective)
        keypoint_results = self._run_keypoint_tasks(image_states, keypoint_tasks, effective, cache, reuse)
        self._attach_keypoint_results(image_states, keypoint_tasks, keypoint_results, cache, reuse)
        all_keypoints, reading_documents = self._run_reading(image_states, effective, cache, reuse)

        coarse_images = [state["sliding_record"] for state in image_states.values()]
        component_images = [state["component_record"] for state in image_states.values()]
        coarse_images.sort(key=lambda item: item["image"])
        component_images.sort(key=lambda item: item["image"])
        reading_images = sorted(reading_documents, key=lambda item: item["image"])
        conflicts = [conflict for document in reading_images for conflict in document["conflicts"]]
        conflicts.sort(key=lambda item: (item["keypoint_index"], item["code"], item["keypoint_id"]))
        for index, conflict in enumerate(conflicts, start=1):
            conflict["seq"] = index

        final_failures = self.failures.as_dicts()
        reuse_document = _reuse_document(reuse)
        keypoint_document = {
            "keypoints": all_keypoints,
            "by_image": _keypoints_by_image(all_keypoints),
        }
        reading_document = {
            "images": reading_images,
            "sequence": [
                {
                    **entry,
                    "image": document["image"],
                }
                for document in reading_images
                for entry in document["sequence"]
            ],
            "conflicts": conflicts,
        }
        failures_document = {"failures": final_failures}
        report = self._build_report(
            effective=effective,
            tier_basis=tier_basis,
            schedule=schedule,
            coarse_images=coarse_images,
            component_images=component_images,
            keypoints=all_keypoints,
            conflicts=conflicts,
            failures=final_failures,
            reuse_document=reuse_document,
            decision=decision,
            recovery_state=store.recovery_state,
        )
        cache_document = cache.finish()
        completed_images, pending_images, failed_images, completed_columns, pending_points = _completion_state(
            self.dataset, schedule, image_states
        )
        global_version = int(
            stable_hash(
                [self.dataset.content_hash, self.config.content_hash(), effective.params_hash()],
                length=12,
            ),
            16,
        )
        generation = int(previous_manifest.get("generation", 0)) + 1 if isinstance(previous_manifest, dict) else 1
        manifest = {
            "schema_version": 1,
            "global_version": global_version,
            "generation": generation,
            "source_version": stable_hash(
                [self.dataset.content_hash, self.config.content_hash(), effective.params_hash()]
            ),
            "dataset_hash": self.dataset.content_hash,
            "config_hash": self.config.content_hash(),
            "effective_hash": effective.params_hash(),
            "tier": effective.tier,
            "keypoint_version": effective.keypoint_version,
            "point_budget": effective.point_budget,
            "completed_images": completed_images,
            "pending_images": pending_images,
            "failed_images": failed_images,
            "completed_columns": completed_columns,
            "pending_points": pending_points,
            "completed_keypoints": len(all_keypoints),
            "run_mode": decision["mode"],
            "recovery_state": store.recovery_state,
        }
        documents = {
            "validation": {"failures": validation_failures, "tier_basis": tier_basis},
            "schedule": schedule,
            "coarse": {"images": coarse_images},
            "components": {"images": component_images},
            "keypoints": keypoint_document,
            "reading_order": reading_document,
            "failures": failures_document,
            "reuse": reuse_document,
            "report": report,
            "cache": cache_document,
        }
        snapshot_dir = store.commit(documents, manifest)
        return RunResult(snapshot_dir=snapshot_dir, documents=documents, manifest=manifest)

    def _localize_images(
        self,
        effective: EffectiveConfig,
        cache: DiagnosticCache,
        reuse: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for entry in sorted(self.dataset.images, key=lambda item: item.name):
            truth = self.dataset.image_truths.get(entry.name)
            if truth is None:
                continue
            image_path = Path(self.dataset.root) / entry.image_path
            if not image_path.is_file():
                continue
            try:
                with Image.open(image_path) as image:
                    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
            except Exception as exc:
                self.failures.add(
                    "E_IMAGE_HEADER",
                    "image",
                    entry.name,
                    f"image could not be decoded for diagnostics: {exc}",
                    path=entry.image_path,
                )
                continue
            if grayscale.shape != (effective.resolution[1], effective.resolution[0]):
                self.failures.add(
                    "E_IMAGE_SIZE",
                    "image",
                    entry.name,
                    "decoded array shape does not match fixed resolution",
                    expected=[effective.resolution[1], effective.resolution[0]],
                    actual=list(grayscale.shape),
                )
                continue
            image_fp = entry.image_hash or file_hash(image_path)
            s_key = sliding_key(image_fp, effective)
            lookup = cache.lookup_image(entry.name, "sliding", s_key, image_fp)
            if lookup.status == "hit" and isinstance(lookup.result, dict):
                sliding_result = deepcopy(lookup.result)
            else:
                sliding_result = run_sliding_stage(entry.name, image_fp, grayscale, effective)
            cache.put_image(entry.name, image_fp, "sliding", s_key, sliding_result)
            reuse.append(
                reuse_entry(
                    image=entry.name,
                    scope="image",
                    stage="sliding",
                    product_id="candidate-set",
                    key=s_key,
                    status=lookup.status,
                    reason="stage key matched prior snapshot" if lookup.status == "hit" else "stage key absent or changed",
                )
            )

            c_key = component_key(image_fp, effective)
            lookup = cache.lookup_image(entry.name, "components", c_key, image_fp)
            if lookup.status == "hit" and isinstance(lookup.result, dict):
                component_result = deepcopy(lookup.result)
            else:
                component_result = run_component_stage(
                    entry.name,
                    grayscale,
                    effective,
                    sliding_result["candidates"],
                )
            cache.put_image(entry.name, image_fp, "components", c_key, component_result)
            reuse.append(
                reuse_entry(
                    image=entry.name,
                    scope="image",
                    stage="components",
                    product_id="component-set",
                    key=c_key,
                    status=lookup.status,
                    reason="stage key matched prior snapshot" if lookup.status == "hit" else "stage key absent or changed",
                )
            )
            fingerprint = fingerprint_bundle(image_fp, effective)
            states[entry.name] = {
                "entry": entry,
                "truth": truth,
                "image_fp": image_fp,
                "grayscale": grayscale,
                "sliding_key": s_key,
                "component_key": c_key,
                "fingerprint": fingerprint,
                "sliding_record": {
                    **deepcopy(sliding_result),
                    "cache_key": s_key,
                    "fingerprint": fingerprint,
                },
                "component_record": {
                    **deepcopy(component_result),
                    "cache_key": c_key,
                    "fingerprint": fingerprint,
                },
                "assigned_components": assign_components_to_columns(component_result["components"], truth.columns),
                "column_keys": {},
                "column_results": {},
                "processed_columns": set(),
                "truth_limits": {},
            }
        return states

    def _prepare_keypoint_tasks(
        self,
        image_states: dict[str, dict[str, Any]],
        schedule: dict[str, Any],
        effective: EffectiveConfig,
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for task in schedule["tasks"]:
            state = image_states.get(task["image"])
            if state is None:
                continue
            truth = state["truth"]
            column = next((item for item in truth.columns if item.id == task["column"]), None)
            if column is None:
                continue
            assigned = list(state["assigned_components"].get(column.id, []))
            selected = assigned[: task["allocated_points"]]
            component_ids = [component["component_id"] for component in selected]
            key = keypoint_column_key(
                state["image_fp"],
                effective,
                state["sliding_key"],
                state["component_key"],
                column.id,
                component_ids,
                task["allocated_points"],
            )
            task = dict(task)
            task["column_object"] = column
            task["selected_components"] = selected
            task["key"] = key
            tasks.append(task)
        tasks.sort(key=stable_task_key)
        return tasks

    def _run_keypoint_tasks(
        self,
        image_states: dict[str, dict[str, Any]],
        tasks: list[dict[str, Any]],
        effective: EffectiveConfig,
        cache: DiagnosticCache,
        reuse: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not tasks:
            return []
        worker_count = 1 if self.serial else min(effective.max_workers, len(tasks))
        worker_count = max(1, worker_count)

        def worker(task: dict[str, Any]) -> dict[str, Any]:
            state = image_states[task["image"]]
            lookup = cache.lookup_column(task["image"], task["column"], task["key"], state["image_fp"])
            if lookup.status == "hit" and isinstance(lookup.result, dict):
                result = deepcopy(lookup.result)
            else:
                result = run_keypoint_column(
                    task["image"],
                    state["truth"],
                    task["column_object"],
                    task["selected_components"],
                    effective,
                )
            return {"task": task, "lookup_status": lookup.status, "result": result}

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="layout-keypoint") as executor:
            results = list(executor.map(worker, tasks))
        results.sort(key=lambda item: stable_task_key(item["task"]))
        return results

    def _attach_keypoint_results(
        self,
        image_states: dict[str, dict[str, Any]],
        tasks: list[dict[str, Any]],
        results: list[dict[str, Any]],
        cache: DiagnosticCache,
        reuse: list[dict[str, Any]],
    ) -> None:
        for item in results:
            task = item["task"]
            state = image_states[task["image"]]
            result = item["result"]
            cache.put_column(task["image"], state["image_fp"], task["column"], task["key"], result)
            state["column_keys"][task["column"]] = task["key"]
            state["column_results"][task["column"]] = result
            state["processed_columns"].add(task["column"])
            state["truth_limits"][task["column"]] = task["allocated_points"]
            status = item["lookup_status"]
            reason = "column key matched prior snapshot" if status == "hit" else "column key absent or changed"
            reuse.append(
                reuse_entry(
                    image=task["image"],
                    scope="column",
                    stage="keypoints",
                    product_id=task["column"],
                    key=task["key"],
                    status=status,
                    reason=reason,
                )
            )
            for point in result.get("keypoints", []):
                point["column_order"] = task["column_object"].order
                reuse.append(
                    reuse_entry(
                        image=task["image"],
                        scope="keypoint",
                        stage="keypoints",
                        product_id=point["keypoint_id"],
                        key=task["key"],
                        status=status,
                        reason=reason,
                    )
                )

    def _run_reading(
        self,
        image_states: dict[str, dict[str, Any]],
        effective: EffectiveConfig,
        cache: DiagnosticCache,
        reuse: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raw_points: list[dict[str, Any]] = []
        for state in image_states.values():
            for column in sorted(state["truth"].columns, key=lambda item: item.order):
                result = state["column_results"].get(column.id)
                if result is not None:
                    raw_points.extend(deepcopy(result.get("keypoints", [])))
        indexed = index_all_keypoints(raw_points)
        points_by_image: dict[str, list[dict[str, Any]]] = {}
        for point in indexed:
            points_by_image.setdefault(point["image"], []).append(point)

        reading_documents: list[dict[str, Any]] = []
        for image_name in sorted(image_states):
            state = image_states[image_name]
            points = points_by_image.get(image_name, [])
            processed_columns = sorted(
                state["processed_columns"],
                key=lambda item: (
                    next((column.order for column in state["truth"].columns if column.id == item), 1_000_000),
                    item,
                ),
            )
            column_keys = sorted((column, state["column_keys"][column]) for column in processed_columns)
            r_key = reading_key(
                state["image_fp"],
                effective,
                column_keys,
                processed_columns,
                [point["keypoint_id"] for point in points],
            )
            lookup = cache.lookup_image(image_name, "reading_order", r_key, state["image_fp"])
            if lookup.status == "hit" and isinstance(lookup.result, dict):
                document = deepcopy(lookup.result)
                _remap_reading_document(document, points)
            else:
                document = run_reading_stage(
                    image_name,
                    state["truth"],
                    points,
                    processed_columns=set(processed_columns),
                    truth_limits=state["truth_limits"],
                )
            document["cache_key"] = r_key
            document["fingerprint"] = fingerprint_bundle(state["image_fp"], effective)
            cache.put_image(image_name, state["image_fp"], "reading_order", r_key, document)
            status = lookup.status
            reason = "reading key matched prior snapshot" if status == "hit" else "reading key absent or changed"
            reuse.append(
                reuse_entry(
                    image=image_name,
                    scope="reading",
                    stage="reading_order",
                    product_id="sequence",
                    key=r_key,
                    status=status,
                    reason=reason,
                )
            )
            for entry in document.get("sequence", []):
                reuse.append(
                    reuse_entry(
                        image=image_name,
                        scope="reading",
                        stage="reading_order",
                        product_id=entry["keypoint_id"],
                        key=r_key,
                        status=status,
                        reason=reason,
                    )
                )
            reading_documents.append(document)
        return indexed, reading_documents

    def _build_report(
        self,
        *,
        effective: EffectiveConfig,
        tier_basis: dict[str, Any],
        schedule: dict[str, Any],
        coarse_images: list[dict[str, Any]],
        component_images: list[dict[str, Any]],
        keypoints: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        failures: list[dict[str, Any]],
        reuse_document: dict[str, Any],
        decision: dict[str, Any],
        recovery_state: str,
    ) -> dict[str, Any]:
        conflict_counts = {name: 0 for name in CONFLICT_TYPES}
        for conflict in conflicts:
            conflict_counts[conflict["type"]] = conflict_counts.get(conflict["type"], 0) + 1
        code_counts: dict[str, int] = {}
        for failure in failures:
            code_counts[failure["code"]] = code_counts.get(failure["code"], 0) + 1
        per_image: dict[str, int] = {}
        for point in keypoints:
            per_image[point["image"]] = per_image.get(point["image"], 0) + 1
        return {
            "tier_selection": tier_basis,
            "effective_config": effective.as_dict(),
            "regression": regression_metadata(effective),
            "keypoint_stats": {
                "detected": len(keypoints),
                "budget_allocated": schedule["allocated_total"],
                "per_image": dict(sorted(per_image.items())),
                "fixed_bits": effective.precision.fixed_bits,
                "quant_step": effective.precision.quant_step,
            },
            "annotations": {
                "active_version": effective.keypoint_version,
                "images": [
                    _annotation_record(self.dataset, entry.name, effective.keypoint_version)
                    for entry in sorted(self.dataset.images, key=lambda item: item.name)
                ],
            },
            "conflict_counts": conflict_counts,
            "business_code_counts": dict(sorted(code_counts.items())),
            "cache": reuse_document["summary"],
            "scheduling": {
                "budget": schedule["budget"],
                "remaining_after_plan": schedule["remaining_after_plan"],
                "task_count": len(schedule["tasks"]),
                "pending_count": len(schedule["pending"]),
                "pending": schedule["pending"],
                "max_workers_configured": effective.max_workers,
                "stable_task_keys": [task["stable_key"] for task in schedule["tasks"]],
            },
            "stages": {
                "image_count": len(coarse_images),
                "window_count": sum(item["window_count"] for item in coarse_images),
                "candidate_count": sum(item["candidate_count"] for item in coarse_images),
                "raw_component_count": sum(item["raw_component_count"] for item in component_images),
                "component_count": sum(item["component_count"] for item in component_images),
                "merge_record_count": sum(len(item["merge_records"]) for item in component_images),
                "split_record_count": sum(len(item["split_records"]) for item in component_images),
            },
            "run_decision": decision,
            "snapshot_recovery": recovery_state,
        }


def _reuse_document(entries: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        entries,
        key=lambda item: (
            item["image"],
            SCOPE_ORDER.get(item["scope"], 99),
            STAGE_ORDER.get(item["stage"], 99),
            item["product_id"],
            item["key"],
            item["status"],
        ),
    )
    for index, entry in enumerate(ordered, start=1):
        entry["seq"] = index
    hits = sum(1 for entry in ordered if entry["status"] == "hit")
    misses = sum(1 for entry in ordered if entry["status"] == "miss")
    by_stage: dict[str, dict[str, int]] = {}
    for entry in ordered:
        bucket = by_stage.setdefault(entry["stage"], {"hit": 0, "miss": 0})
        bucket[entry["status"]] += 1
    return {
        "entries": ordered,
        "summary": {
            "entries": len(ordered),
            "hits": hits,
            "misses": misses,
            "by_stage": dict(sorted(by_stage.items())),
        },
    }


def _annotation_record(dataset: Dataset, image: str, active_version: str) -> dict[str, Any]:
    version = dataset.keypoint_versions.get(image, {}).get(active_version)
    if version is None:
        return {
            "image": image,
            "version": active_version,
            "available": False,
            "point_count": 0,
            "content_hash": None,
        }
    return {
        "image": image,
        "version": active_version,
        "available": True,
        "path": version.path,
        "point_count": len(version.keypoints),
        "content_hash": stable_hash([point.as_dict() for point in version.keypoints]),
    }


def _remap_reading_document(document: dict[str, Any], points: list[dict[str, Any]]) -> None:
    """Reapply current global keypoint indexes after an image-local reading cache hit."""

    mapping = {point["keypoint_id"]: point["keypoint_index"] for point in points}
    for entry in document.get("sequence", []):
        if entry.get("keypoint_id") in mapping:
            entry["keypoint_index"] = mapping[entry["keypoint_id"]]
    conflicts = document.get("conflicts", [])
    for conflict in conflicts:
        if conflict.get("keypoint_id") in mapping:
            conflict["keypoint_index"] = mapping[conflict["keypoint_id"]]
        detail = conflict.get("detail", {})
        previous_id = detail.get("previous_keypoint_id")
        if previous_id in mapping:
            detail["previous_keypoint_index"] = mapping[previous_id]
    conflicts.sort(key=lambda item: (item["keypoint_index"], item.get("code", ""), item["keypoint_id"]))
    for index, conflict in enumerate(conflicts, start=1):
        conflict["seq"] = index


def _keypoints_by_image(keypoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in keypoints:
        grouped.setdefault(point["image"], []).append(point)
    return [
        {
            "image": image,
            "keypoints": sorted(points, key=lambda item: item["keypoint_index"]),
        }
        for image, points in sorted(grouped.items())
    ]


def _all_images_complete(dataset: Dataset, schedule: dict[str, Any]) -> bool:
    if not dataset.images:
        return False
    entries = schedule["all_entries"]
    for entry in dataset.images:
        own = [item for item in entries if item["image"] == entry.name]
        if not own or any(item["status"] != "complete" for item in own):
            return False
        if entry.name not in dataset.image_truths:
            return False
    return True


def _completion_state(
    dataset: Dataset,
    schedule: dict[str, Any],
    image_states: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]], int]:
    completed_images: list[str] = []
    pending_images: list[str] = []
    failed_images: list[str] = []
    completed_columns: list[dict[str, Any]] = []
    for entry in sorted(dataset.images, key=lambda item: item.name):
        if entry.name not in image_states:
            failed_images.append(entry.name)
            continue
        own = [item for item in schedule["all_entries"] if item["image"] == entry.name]
        pending = [item for item in own if item["status"] != "complete"]
        complete = [item for item in own if item["status"] == "complete"]
        for item in complete:
            completed_columns.append({"image": item["image"], "column": item["column"]})
        if pending:
            pending_images.append(entry.name)
        if own and not pending:
            completed_images.append(entry.name)
    pending_points = sum(item["pending_points"] for item in schedule["pending"])
    return completed_images, pending_images, failed_images, completed_columns, pending_points


def run_from_paths(
    dataset_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    *,
    force_rebuild: bool = False,
    serial: bool = False,
) -> tuple[RunResult, FailureLog]:
    from .validation import validate_workspace

    dataset, config, failures = validate_workspace(dataset_root, config_path)
    engine = DiagnosticEngine(
        dataset,
        config,
        output_root,
        failures=failures,
        force_rebuild=force_rebuild,
        serial=serial,
    )
    return engine.run(), failures
