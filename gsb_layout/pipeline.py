"""End-to-end deterministic pipeline, incremental cache orchestration and reports."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from hashlib import sha256
import copy
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .algorithms import (
    COLUMN_ALGORITHM_VERSION,
    MATCH_ALGORITHM_VERSION,
    REGRESSION_ALGORITHM_VERSION,
    SEGMENT_ALGORITHM_VERSION,
    CandidateSegment,
    DetectedColumn,
    binary_digest,
    block_digest,
    binarize,
    confidence_details,
    detect_columns,
    match_block,
    regress_segment,
    segment_column,
)
from .cache import ArtifactCache, content_key
from .config import Config, canonical_hash, canonical_json
from .errors import BusinessError, ImageInvalidError, TruthInvalidError
from .io_layer import ImageRecord, TruthRecord, discover_pngs, load_png, load_truth, truth_to_dict
from .snapshot import SnapshotLoad, SnapshotStore
from .templates import TemplateSet, build_template_set

REPORT_SCHEMA_VERSION = 1
RUN_PIPELINE_VERSION = "gsb-layout-1.0.0"


@dataclass
class LoadedImage:
    image: ImageRecord
    truth: TruthRecord


@dataclass
class ImageWork:
    name: str
    image: ImageRecord
    truth: TruthRecord
    columns: list[dict[str, Any]] = field(default_factory=list)
    column_cache: dict[str, Any] = field(default_factory=dict)
    pending_matches: int = 0
    complete: bool = True


def _sort_name(name: str) -> tuple[str, str]:
    return name.casefold(), name


def _load_inputs(samples_dir: Path, config: Config) -> tuple[list[LoadedImage], list[dict[str, Any]], list[str]]:
    loaded: list[LoadedImage] = []
    failures: list[dict[str, Any]] = []
    discovered: list[str] = []
    try:
        png_paths = discover_pngs(samples_dir)
    except BusinessError as exc:
        failures.append({"code": exc.code, "image": None, "detail": exc.message})
        return loaded, failures, discovered
    for png_path in png_paths:
        discovered.append(png_path.name)
        try:
            image = load_png(png_path, config)
        except ImageInvalidError as exc:
            failures.append({"code": exc.code, "image": png_path.name, "detail": exc.message})
            continue
        try:
            truth = load_truth(png_path, image)
        except TruthInvalidError as exc:
            failures.append({"code": exc.code, "image": png_path.name, "detail": exc.message})
            continue
        loaded.append(LoadedImage(image=image, truth=truth))
    loaded.sort(key=lambda item: _sort_name(item.image.name))
    return loaded, failures, discovered


def _column_cache_inputs(image: ImageRecord, config: Config, template_version: str) -> dict[str, Any]:
    return {
        "image_sha256": image.sha256,
        "width": image.width,
        "height": image.height,
        "binary_image_digest": binary_digest(
            binarize(image.pixels, config.binarization_threshold)
        ),
        "row_projection_window": config.row_projection_window,
        "template_version": template_version,
    }


def _segment_cache_inputs(
    image: ImageRecord,
    column: DetectedColumn,
    binary_strip_digest: str,
    config: Config,
    template_version: str,
) -> dict[str, Any]:
    return {
        "image_sha256": image.sha256,
        "column_x0": column.x0,
        "column_x1": column.x1,
        "binary_strip_digest": binary_strip_digest,
        "row_projection_window": config.row_projection_window,
        "template_version": template_version,
    }


def _tight_ink_height(pixels: np.ndarray, column: DetectedColumn, segment: CandidateSegment, threshold: int) -> int:
    binary = pixels[segment.y0 : segment.y1, column.x0 : column.x1] < threshold
    rows = np.flatnonzero(binary.sum(axis=1) > 0)
    if rows.size == 0:
        return segment.height
    return int(rows[-1] - rows[0] + 1)


def _regression_cache_inputs(
    image: ImageRecord,
    column: DetectedColumn,
    segment: CandidateSegment,
    segment_digest: str,
    config: Config,
    template_version: str,
) -> dict[str, Any]:
    tight_height = _tight_ink_height(image.pixels, column, segment, config.binarization_threshold)
    target_height = tight_height + 2
    policy: dict[str, Any] = {
        "height_class": segment.height_class,
        "algorithm": REGRESSION_ALGORITHM_VERSION,
    }
    # Bounds matter only when the regression clamp can actually bite. This is
    # what lets an unrelated min/max edit reuse unaffected normal characters.
    if target_height < config.char_min_height or segment.height_class == "under_min":
        policy["min_height"] = config.char_min_height
    if target_height > config.char_max_height or segment.height_class == "over_max":
        policy["max_height"] = config.char_max_height
    if segment.height_class == "over_max":
        policy["min_height"] = config.char_min_height
        policy["split_penalty"] = config.split_penalty
        policy["row_projection_window"] = config.row_projection_window
    return {
        "image_sha256": image.sha256,
        "column_x0": column.x0,
        "column_x1": column.x1,
        "segment_y0": segment.y0,
        "segment_y1": segment.y1,
        "segment_digest": segment_digest,
        "policy": policy,
        "template_version": template_version,
    }


def _match_cache_inputs(
    image: ImageRecord,
    char: dict[str, Any],
    config: Config,
    templates: TemplateSet,
) -> dict[str, Any]:
    x0, y0, x1, y1 = char["bbox"]
    block = image.pixels[y0:y1, x0:x1]
    digest = block_digest(block)
    binary_block_digest = binary_digest(block < config.binarization_threshold)
    return {
        "block_digest": digest,
        "binary_block_digest": binary_block_digest,
        "metric": config.match_metric,
        "weights": list(config.template_weights),
        "fixed_bits": config.fixed_bits,
        "quant_levels": config.quant_levels,
        "template_version": templates.version,
        "template_algorithm": templates.algorithm,
    }


def _semantic_image(image: dict[str, Any]) -> dict[str, Any]:
    """Return the complete deterministic image result payload.

    The report image object contains only business output (no cache action,
    timestamp or budget-run metadata), so every field can participate in the
    incremental/full comparison.
    """
    return image


def semantic_digest(images: Sequence[dict[str, Any]]) -> str:
    payload = [_semantic_image(image) for image in images if image.get("status") != "failed"]
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class Pipeline:
    def __init__(
        self,
        samples_dir: str | Path,
        config: Config,
        state_dir: str | Path = "state",
        *,
        templates: TemplateSet | None = None,
    ) -> None:
        self.samples_dir = Path(samples_dir)
        self.config = config
        self.state_dir = Path(state_dir)
        self.templates = templates or build_template_set(config.template_weights)
        if self.templates.variant_count != len(config.template_weights):
            raise BusinessError(
                f"字模变体数 {self.templates.variant_count} 与融合权重数 {len(config.template_weights)} 不一致",
                code="E_CONFIG_INVALID",
            )
        self.cache = ArtifactCache(self.state_dir / "cache")
        self.snapshots = SnapshotStore(self.state_dir)

    def _detect_one(
        self,
        item: LoadedImage,
    ) -> tuple[ImageWork, dict[str, Any]]:
        inputs = _column_cache_inputs(item.image, self.config, self.templates.version)
        value, key, reused = self.cache.compute(
            "columns",
            inputs,
            COLUMN_ALGORITHM_VERSION,
            lambda: {
                "columns": [column.as_dict() for column in detect_columns(item.image.pixels, self.config)]
            },
        )
        work = ImageWork(item.image.name, item.image, item.truth)
        work.column_cache = {"key": key, "action": "reuse" if reused else "recompute"}
        return work, value

    def _segment_columns(
        self,
        work: ImageWork,
        column_results: list[dict[str, Any]],
        trace: dict[str, list[dict[str, Any]]],
    ) -> None:
        binary = binarize(work.image.pixels, self.config.binarization_threshold)
        for col_data in column_results:
            column = DetectedColumn(
                reading_index=int(col_data["reading_index"]),
                x0=int(col_data["x0"]),
                x1=int(col_data["x1"]),
                raw_ink=int(col_data.get("raw_ink", 0)),
                peak_density=int(col_data.get("peak_density", 0)),
            )
            strip = binary[:, column.x0 : column.x1]
            strip_digest = binary_digest(strip)
            inputs = _segment_cache_inputs(work.image, column, strip_digest, self.config, self.templates.version)
            value, key, reused = self.cache.compute(
                "segments",
                inputs,
                SEGMENT_ALGORITHM_VERSION,
                lambda: {
                    "segments": [
                        {
                            "segment_index": segment.segment_index,
                            "y0": segment.y0,
                            "y1": segment.y1,
                            "height": segment.height,
                            "raw_ink": segment.raw_ink,
                        }
                        for segment in segment_column(work.image.pixels, column, self.config)
                    ]
                },
            )
            trace["columns"].append(
                {
                    "image": work.name,
                    "column": column.reading_index,
                    "stage": "segments",
                    "action": "reuse" if reused else "recompute",
                    "key": key,
                    "task_key": f"{work.name}|segments|{column.reading_index:04d}",
                }
            )
            segments: list[CandidateSegment] = []
            for raw in value.get("segments", []):
                height = int(raw["y1"]) - int(raw["y0"])
                if height < self.config.char_min_height:
                    height_class = "under_min"
                    needs = True
                elif height > self.config.char_max_height:
                    height_class = "over_max"
                    needs = True
                else:
                    height_class = "normal"
                    needs = False
                segments.append(
                    CandidateSegment(
                        segment_index=int(raw["segment_index"]),
                        y0=int(raw["y0"]),
                        y1=int(raw["y1"]),
                        height=height,
                        raw_ink=int(raw.get("raw_ink", 0)),
                        height_class=height_class,
                        needs_regression=needs,
                    )
                )
            column_record: dict[str, Any] = {
                **column.as_dict(),
                "segments": [segment.as_dict() for segment in segments],
                "chars": [],
                "cache_key": key,
            }
            work.columns.append(column_record)

    def _regress_columns(
        self,
        work: ImageWork,
        trace: dict[str, list[dict[str, Any]]],
        boundary_candidates: list[dict[str, Any]],
    ) -> None:
        binary = binarize(work.image.pixels, self.config.binarization_threshold)
        for column_record in work.columns:
            column = DetectedColumn(
                int(column_record["reading_index"]),
                int(column_record["x0"]),
                int(column_record["x1"]),
                int(column_record.get("raw_ink", 0)),
                int(column_record.get("peak_density", 0)),
            )
            char_number = 1
            for segment_raw in column_record["segments"]:
                segment = CandidateSegment(
                    int(segment_raw["segment_index"]),
                    int(segment_raw["y0"]),
                    int(segment_raw["y1"]),
                    int(segment_raw["height"]),
                    int(segment_raw.get("raw_ink", 0)),
                    str(segment_raw["height_class"]),
                    bool(segment_raw["needs_regression"]),
                )
                segment_digest = binary_digest(binary[segment.y0 : segment.y1, column.x0 : column.x1])
                inputs = _regression_cache_inputs(work.image, column, segment, segment_digest, self.config, self.templates.version)
                value, key, reused = self.cache.compute(
                    "regression",
                    inputs,
                    REGRESSION_ALGORITHM_VERSION,
                    lambda: regress_segment(work.image.pixels, column, segment, self.config),
                )
                for box in value.get("boxes", []):
                    bbox = [int(v) for v in box["bbox"]]
                    char = {
                        "reading_index": char_number,
                        "bbox": bbox,
                        "text": None,
                        "confidence": None,
                        "confidence_q": None,
                        "low_confidence": False,
                        "low_reasons": [],
                        "regression": {
                            "tight_bbox": list(box["tight_bbox"]),
                            "residual": int(box["residual"]),
                            "residual_detail": dict(box["residual_detail"]),
                            "centroid_y_q4": int(box["centroid_y_q4"]),
                            "ink_pixels": int(box["ink_pixels"]),
                            "part_index": int(box["part_index"]),
                        },
                        "split_record": copy.deepcopy(box["split_record"]),
                        "match": None,
                        "cache_key": key,
                    }
                    column_record["chars"].append(char)
                    trace["characters"].append(
                        {
                            "image": work.name,
                            "column": column.reading_index,
                            "char": char_number,
                            "stage": "regression",
                            "action": "reuse" if reused else "recompute",
                            "key": key,
                            "task_key": f"{work.name}|{column.reading_index:04d}|{char_number:04d}|box",
                        }
                    )
                    if segment.height_class != "normal":
                        boundary_candidates.append(
                            {
                                "image": work.name,
                                "column": column.reading_index,
                                "segment_index": segment.segment_index,
                                "y0": segment.y0,
                                "y1": segment.y1,
                                "height": segment.height,
                                "kind": segment.height_class,
                                "reading_index": char_number,
                                "resolution": "split" if value.get("split", {}).get("applied") else "expanded",
                                "split_count": int(value.get("split", {}).get("split_count", 0)),
                                "split_tied": bool(value.get("split", {}).get("tied", False)),
                            }
                        )
                    char_number += 1

    def _match_all(
        self,
        work_items: list[ImageWork],
        trace: dict[str, list[dict[str, Any]]],
        *,
        prior_pending_count: int,
    ) -> dict[str, Any]:
        jobs: list[dict[str, Any]] = []
        for work in work_items:
            for column in work.columns:
                for char in column["chars"]:
                    inputs = _match_cache_inputs(work.image, char, self.config, self.templates)
                    key = content_key("match", inputs, MATCH_ALGORITHM_VERSION)
                    x0, y0, x1, y1 = char["bbox"]
                    point_cost = int(max(1, x1 - x0) * max(1, y1 - y0) * self.templates.variant_count)
                    jobs.append(
                        {
                            "work": work,
                            "column": column,
                            "char": char,
                            "key": key,
                            "point_cost": point_cost,
                            "job_id": f"{work.name}:{column['reading_index']:04d}:{char['reading_index']:04d}",
                            "task_key": f"{work.name}|{column['reading_index']:04d}|{char['reading_index']:04d}|match",
                        }
                    )
        remaining = int(self.config.match_point_budget)
        selected: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        budget_sequence: list[dict[str, Any]] = []
        deferred = False
        cache_hits = 0
        cache_misses = 0
        for job in jobs:
            found = self.cache.get("match", job["key"], record=False)
            if found.hit:
                self.cache.stats.record("match", "reuse")
                job["char"]["match"] = found.value
                cache_hits += 1
                trace["characters"].append(
                    {
                        "image": job["work"].name,
                        "column": job["column"]["reading_index"],
                        "char": job["char"]["reading_index"],
                        "stage": "template_match",
                        "action": "reuse",
                        "key": job["key"],
                        "points": job["point_cost"],
                        "task_key": job["task_key"],
                    }
                )
                continue
            cache_misses += 1
            if deferred:
                pending.append(job)
                continue
            if job["point_cost"] <= remaining:
                remaining -= job["point_cost"]
                selected.append(job)
                budget_sequence.append(
                    {
                        "sequence": len(budget_sequence) + 1,
                        "job_id": job["job_id"],
                        "task_key": job["task_key"],
                        "image": job["work"].name,
                        "column": job["column"]["reading_index"],
                        "char": job["char"]["reading_index"],
                        "points": job["point_cost"],
                        "remaining": remaining,
                    }
                )
            else:
                deferred = True
                pending.append(job)
        workers = min(self.config.thread_count, max(1, len(selected))) if selected else 0
        futures: dict[Any, dict[str, Any]] = {}
        if selected:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gsb-match") as pool:
                for job in selected:
                    futures[pool.submit(self._match_job, job)] = job
                for future in as_completed(futures):
                    job = futures[future]
                    result = future.result()
                    job["char"]["match"] = result
        for job in selected:
            self.cache.stats.record("match", "recompute")
            trace["characters"].append(
                {
                    "image": job["work"].name,
                    "column": job["column"]["reading_index"],
                    "char": job["char"]["reading_index"],
                    "stage": "template_match",
                    "action": "recompute",
                    "key": job["key"],
                    "points": job["point_cost"],
                    "task_key": job["task_key"],
                }
            )
        for job in pending:
            job["char"]["match"] = None
            job["work"].pending_matches += 1
            job["work"].complete = False
            trace["characters"].append(
                {
                    "image": job["work"].name,
                    "column": job["column"]["reading_index"],
                    "char": job["char"]["reading_index"],
                    "stage": "template_match",
                    "action": "pending",
                    "key": job["key"],
                    "points": job["point_cost"],
                    "task_key": job["task_key"],
                }
            )
        return {
            "limit": int(self.config.match_point_budget),
            "consumed_points": int(self.config.match_point_budget) - remaining,
            "remaining_points": int(remaining),
            "candidate_jobs": len(jobs),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "computed_jobs": len(selected),
            "pending_jobs": len(pending),
            "workers": workers,
            "resumed_from_prior_pending": bool(prior_pending_count),
            "prior_pending_count": int(prior_pending_count),
            "sequence": budget_sequence,
            "pending_order": [job["job_id"] for job in pending],
            "pending_tasks": [
                {
                    "job_id": job["job_id"],
                    "task_key": job["task_key"],
                    "points": job["point_cost"],
                }
                for job in pending
            ],
        }

    def _match_job(self, job: dict[str, Any]) -> dict[str, Any]:
        char = job["char"]
        result = match_block(job["work"].image.pixels, char["bbox"], self.config, self.templates)
        self.cache.put("match", job["key"], result)
        return result

    def run(self, *, full_rebuild: bool = False) -> dict[str, Any]:
        snapshot_load: SnapshotLoad = self.snapshots.load(self.config)
        previous_manifest = snapshot_load.manifest
        if full_rebuild:
            snapshot_load = SnapshotLoad(None, False, None, [])
            previous_manifest = None
        previous_version = int(previous_manifest.get("global_version", 0)) if previous_manifest else 0
        global_version = previous_version + 1
        prior_pending = int(previous_manifest.get("pending_count", 0)) if previous_manifest else 0

        loaded, load_failures, discovered = _load_inputs(self.samples_dir, self.config)
        trace: dict[str, list[dict[str, Any]]] = {"columns": [], "characters": []}
        boundary_candidates: list[dict[str, Any]] = []
        failures = list(load_failures)
        work_items: list[ImageWork] = []

        for item in loaded:
            work = ImageWork(item.image.name, item.image, item.truth)
            trace["columns"].append(
                {
                    "image": work.name,
                    "column": None,
                    "stage": "columns",
                    "action": "recompute",
                    "key": None,
                    "task_key": f"{work.name}|columns|image",
                }
            )
            work_items.append(work)

        # Image-level column detection tasks are independent and collected in stable
        # filename order, so scheduling cannot reorder the output.
        workers = min(self.config.thread_count, max(1, len(loaded))) if loaded else 0
        if loaded:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gsb-columns") as pool:
                future_map = {pool.submit(self._detect_one, item): item for item in loaded}
                detected: dict[str, tuple[ImageWork, dict[str, Any]]] = {}
                for future in as_completed(future_map):
                    item = future_map[future]
                    if isinstance(item, LoadedImage):
                        detected[item.image.name] = future.result()
            for work_index, work in enumerate(work_items, start=1):
                detected_work, column_result = detected[work.name]
                work.column_cache = detected_work.column_cache
                action = work.column_cache["action"]
                trace["columns"] = [
                    entry
                    for entry in trace["columns"]
                    if not (entry["image"] == work.name and entry["stage"] == "columns")
                ]
                columns = column_result.get("columns", [])
                for col in columns:
                    trace["columns"].append(
                        {
                            "image": work.name,
                            "column": int(col["reading_index"]),
                            "stage": "columns",
                            "action": action,
                            "key": work.column_cache["key"],
                            "task_key": f"{work.name}|columns|image",
                        }
                    )
                self._segment_columns(work, columns, trace)
                self._regress_columns(work, trace, boundary_candidates)
                if self.config.snapshot_interval > 0 and work_index % self.config.snapshot_interval == 0:
                    self.snapshots.write_checkpoint(
                        f"layout-{work_index:04d}",
                        {
                            "completed_layout_images": [w.name for w in work_items[:work_index]],
                            "image": work.name,
                            "columns": work.columns,
                        },
                    )

        budget_report = self._match_all(
            work_items, trace, prior_pending_count=prior_pending
        )

        low_reason_counts: dict[str, int] = {}
        tie_items: list[dict[str, Any]] = []
        match_tie_count = 0
        split_tie_count = 0
        images_report: list[dict[str, Any]] = []
        total_chars = 0
        low_conf_chars = 0
        for work in work_items:
            image_status = "ok"
            if work.pending_matches:
                image_status = "pending"
            image_chars: list[dict[str, Any]] = []
            for column in work.columns:
                for char in column["chars"]:
                    total_chars += 1
                    match = char.get("match")
                    if match is None:
                        failures.append(
                            {
                                "code": "E_MATCH_BUDGET_PENDING",
                                "image": work.name,
                                "column": column["reading_index"],
                                "char": char["reading_index"],
                                "detail": "模板比对点数预算不足，已保留到下一批优先处理",
                            }
                        )
                        continue
                    regression_context = dict(char["regression"])
                    regression_context["split_record"] = char.get("split_record", {})
                    confidence = confidence_details(match, regression_context, self.config)
                    char["text"] = match["text"]
                    char["confidence_q"] = confidence["confidence_q"]
                    char["confidence"] = confidence["confidence"]
                    char["low_confidence"] = confidence["low_confidence"]
                    char["low_reasons"] = confidence["low_reasons"]
                    char["confidence_detail"] = confidence
                    if confidence["low_confidence"]:
                        low_conf_chars += 1
                    for reason in confidence["low_reasons"]:
                        low_reason_counts[reason["code"]] = low_reason_counts.get(reason["code"], 0) + 1
                    if match.get("tie"):
                        match_tie_count += 1
                        tie_items.append(
                            {
                                "kind": "match",
                                "image": work.name,
                                "column": column["reading_index"],
                                "char": char["reading_index"],
                                "characters": match.get("tied_characters", []),
                                "selected": match.get("text"),
                            }
                        )
                    split_record = char.get("split_record", {})
                    if split_record.get("tied"):
                        split_tie_count += 1
                        tie_items.append(
                            {
                                "kind": "split",
                                "image": work.name,
                                "column": column["reading_index"],
                                "char": char["reading_index"],
                                "cuts": split_record.get("cuts", []),
                            }
                        )
            images_report.append(
                {
                    "name": work.name,
                    "status": image_status,
                    "width": work.image.width,
                    "height": work.image.height,
                    "mode": work.image.mode,
                    "image_sha256": work.image.sha256,
                    "truth": truth_to_dict(work.truth),
                    "columns": work.columns,
                    "pending_matches": work.pending_matches,
                }
            )
        images_report.sort(key=lambda image: _sort_name(image["name"]))
        failures.sort(
            key=lambda failure: (
                _sort_name(str(failure.get("image") or "")),
                int(failure.get("column") or 0),
                int(failure.get("char") or 0),
                str(failure.get("code") or ""),
                str(failure.get("detail") or ""),
            )
        )
        for index, failure in enumerate(failures, start=1):
            failure["failure_id"] = f"F{index:04d}"
        boundary_candidates.sort(
            key=lambda item: (
                _sort_name(item["image"]),
                int(item["column"]),
                int(item["segment_index"]),
                int(item.get("reading_index", 0)),
            )
        )
        tie_items.sort(
            key=lambda item: (
                _sort_name(item["image"]),
                int(item["column"]),
                int(item["char"]),
                str(item["kind"]),
            )
        )
        trace["columns"].sort(
            key=lambda entry: (
                _sort_name(entry.get("image", "")),
                int(entry.get("column") or 0),
                str(entry.get("stage", "")),
                str(entry.get("task_key", "")),
            )
        )
        trace["characters"].sort(
            key=lambda entry: (
                _sort_name(entry.get("image", "")),
                int(entry.get("column") or 0),
                int(entry.get("char") or 0),
                str(entry.get("stage", "")),
                str(entry.get("task_key", "")),
            )
        )
        semantic = semantic_digest(images_report)
        hard_failure_images = {
            failure.get("image")
            for failure in failures
            if failure.get("image") and failure.get("code") != "E_MATCH_BUDGET_PENDING"
        }
        pending_images = {
            failure.get("image")
            for failure in failures
            if failure.get("image") and failure.get("code") == "E_MATCH_BUDGET_PENDING"
        }
        failure_code_counts: dict[str, int] = {}
        for failure in failures:
            code = str(failure.get("code"))
            failure_code_counts[code] = failure_code_counts.get(code, 0) + 1
        completed_images = [
            image["name"]
            for image in images_report
            if image["status"] == "ok"
        ]
        image_hashes = {image["name"]: image["image_sha256"] for image in images_report}
        stage_payloads = self._stage_payloads(images_report)
        cache_report = {
            "summary": self.cache.export_stats(),
            "reuse_recompute": trace,
            "columns_reused": sum(1 for e in trace["columns"] if e["action"] == "reuse"),
            "columns_recomputed": sum(1 for e in trace["columns"] if e["action"] == "recompute"),
            "characters_reused": sum(1 for e in trace["characters"] if e["action"] == "reuse"),
            "characters_recomputed": sum(1 for e in trace["characters"] if e["action"] == "recompute"),
            "characters_pending": sum(1 for e in trace["characters"] if e["action"] == "pending"),
        }
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "pipeline_version": RUN_PIPELINE_VERSION,
            "config": self.config.canonical_dict(),
            "run": {
                "mode": "full" if full_rebuild else "incremental",
                "global_version": global_version,
                "previous_global_version": previous_version,
                "config_hash": self.config.fingerprint(),
                "template_version": self.templates.version,
                "snapshot_stale": snapshot_load.stale,
                "snapshot_stale_reason": snapshot_load.stale_reason,
                "full_rebuild": bool(full_rebuild),
            },
            "summary": {
                "discovered_images": len(discovered),
                "loaded_images": len(loaded),
                "completed_images": len(completed_images),
                "failed_images": len(hard_failure_images),
                "pending_images": len(pending_images),
                "characters": total_chars,
                "low_confidence_characters": low_conf_chars,
                "match_tie_decisions": match_tie_count,
                "split_tie_decisions": split_tie_count,
                "pending_matches": budget_report["pending_jobs"],
                "failure_code_counts": dict(sorted(failure_code_counts.items())),
            },
            "images": images_report,
            "failures": failures,
            "diagnostics": snapshot_load.diagnostics,
            "low_confidence_reason_counts": dict(sorted(low_reason_counts.items())),
            "tie_decisions": {"count": len(tie_items), "items": tie_items},
            "boundary_candidates": boundary_candidates,
            "cache": cache_report,
            "budget": budget_report,
            "threading": {
                "configured_max_workers": int(self.config.thread_count),
                "column_workers": int(workers),
                "match_workers": int(budget_report["workers"]),
                "column_task_keys": sorted(
                    {
                        entry["task_key"]
                        for entry in trace["columns"]
                        if entry.get("stage") == "columns" and entry.get("task_key")
                    },
                    key=lambda value: (value.casefold(), value),
                ),
                "selected_match_task_keys": [
                    item["task_key"] for item in budget_report["sequence"]
                ],
                "pending_match_task_keys": [
                    item["task_key"] for item in budget_report["pending_tasks"]
                ],
            },
            "semantic_digest": semantic,
        }
        report_digest = sha256(canonical_json(report).encode("utf-8")).hexdigest()
        manifest = self.snapshots.commit(
            config=self.config,
            global_version=global_version,
            completed_images=completed_images,
            image_hashes=image_hashes,
            stage_payloads=stage_payloads,
            report_digest=report_digest,
            budget=budget_report,
            pending_count=budget_report["pending_jobs"],
            previous_manifest=previous_manifest,
        )
        report["snapshot"] = {
            "global_version": manifest["global_version"],
            "stage_versions": manifest["stage_versions"],
            "pending_count": manifest["pending_count"],
            "report_digest": report_digest,
            "stale": snapshot_load.stale,
            "stale_reason": snapshot_load.stale_reason,
        }
        return report

    def _stage_payloads(self, images: Sequence[dict[str, Any]]) -> dict[str, Any]:
        columns = {"images": []}
        segments = {"images": []}
        boxes = {"images": []}
        matches = {"images": []}
        for image in images:
            column_rows = []
            segment_rows = []
            box_rows = []
            match_rows = []
            for column in image.get("columns", []):
                column_rows.append(
                    {
                        "reading_index": column["reading_index"],
                        "x0": column["x0"],
                        "x1": column["x1"],
                    }
                )
                segment_rows.append(
                    {
                        "reading_index": column["reading_index"],
                        "segments": column["segments"],
                    }
                )
                box_rows.append(
                    {
                        "reading_index": column["reading_index"],
                        "chars": [
                            {
                                "reading_index": char["reading_index"],
                                "bbox": char["bbox"],
                                "residual": char["regression"]["residual"],
                                "split_record": char["split_record"],
                            }
                            for char in column["chars"]
                        ],
                    }
                )
                match_rows.append(
                    {
                        "reading_index": column["reading_index"],
                        "chars": [
                            {
                                "reading_index": char["reading_index"],
                                "match": char.get("match"),
                                "confidence_q": char.get("confidence_q"),
                                "low_confidence": char.get("low_confidence"),
                                "low_reasons": char.get("low_reasons"),
                            }
                            for char in column["chars"]
                        ],
                    }
                )
            columns["images"].append({"name": image["name"], "columns": column_rows})
            segments["images"].append({"name": image["name"], "columns": segment_rows})
            boxes["images"].append({"name": image["name"], "columns": box_rows})
            matches["images"].append({"name": image["name"], "columns": match_rows})
        return {"columns": columns, "segments": segments, "boxes": boxes, "matches": matches}


def run_pipeline(
    samples_dir: str | Path,
    config: Config,
    state_dir: str | Path = "state",
    *,
    full_rebuild: bool = False,
) -> dict[str, Any]:
    return Pipeline(samples_dir, config, state_dir).run(full_rebuild=full_rebuild)
