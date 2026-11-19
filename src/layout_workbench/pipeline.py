from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

from .cache import PIPELINE_CACHE_VERSION, CacheStore, cache_key
from .canonical import digest
from .config import RecognitionConfig
from .detection import aggregate_line_candidates, detect_blocks
from .font import FONT_VERSION
from .loading import ImageSource, LoadError, LoadedImage, load_image_source
from .matching import match_point_cost, recognize_segment
from .segmentation import segment_line

T = TypeVar("T")


@dataclass
class StageState:
    key: str
    state: str
    duration_ms: float
    product: dict[str, Any]
    reused_items: int = 0
    recomputed_items: int = 0


def _parallel_ordered(
    tasks: list[tuple[str, Callable[[], T]]], workers: int
) -> tuple[dict[str, T], int]:
    if not tasks:
        return {}, 0
    results: dict[str, T] = {}
    active = 0
    maximum = 0
    lock = threading.Lock()

    def wrapped(key: str, fn: Callable[[], T]) -> T:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            return fn()
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="layout-worker") as executor:
        futures = {
            executor.submit(wrapped, key, fn): key
            for key, fn in sorted(tasks, key=lambda item: item[0])
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
    return results, maximum


class Pipeline:
    def __init__(
        self,
        config: RecognitionConfig,
        cache: CacheStore,
        *,
        respect_faults: bool = False,
    ) -> None:
        self.config = config
        self.cache = cache
        self.respect_faults = respect_faults

    def run(
        self,
        sources: Iterable[ImageSource],
        *,
        mode: str = "incremental",
        snapshot_version: int = 0,
        command: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        ordered_sources = sorted(list(sources), key=lambda source: source.name)
        failures: list[dict[str, Any]] = []
        loaded: list[LoadedImage] = []
        for source in ordered_sources:
            try:
                loaded.append(load_image_source(source, self.config))
            except LoadError as exc:
                failures.append(
                    {
                        "image": source.name,
                        "code": exc.code,
                        "message": exc.message,
                    }
                )
        failures.sort(key=lambda item: (item["image"], item["code"], item["message"]))

        timings: dict[str, float] = {
            "load": 0.0,
            "detection": 0.0,
            "line_aggregation": 0.0,
            "segmentation": 0.0,
            "recognition": 0.0,
        }
        fault_hits: list[dict[str, Any]] = []
        detection_by_image: dict[str, StageState] = {}
        lines_by_image: dict[str, StageState] = {}
        segmentation_by_image: dict[str, StageState] = {}
        recognition_by_image: dict[str, StageState] = {}
        observed_workers = 0

        detection_started = time.perf_counter()
        detection_tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = []
        for image in loaded:
            key = cache_key(
                "detection",
                {
                    "image_fingerprint": image.content_fingerprint,
                    "image_size": [image.width, image.height],
                    "detection_params": self.config.detection_params,
                    "font_version": FONT_VERSION,
                },
            )
            artifact = self.cache.get(key)
            if artifact is not None:
                detection_by_image[image.name] = StageState(
                    key=key,
                    state="reused",
                    duration_ms=0.0,
                    product=artifact["payload"],
                    reused_items=1,
                )
            else:
                detection_tasks.append(
                    (
                        f"{image.name}|detection",
                        lambda image=image: detect_blocks(image.gray, self.config),
                    )
                )
        task_results, workers = _parallel_ordered(detection_tasks, self.config.thread_pool_workers)
        observed_workers = max(observed_workers, workers)
        for image in loaded:
            if image.name in detection_by_image:
                continue
            key = cache_key(
                "detection",
                {
                    "image_fingerprint": image.content_fingerprint,
                    "image_size": [image.width, image.height],
                    "detection_params": self.config.detection_params,
                    "font_version": FONT_VERSION,
                },
            )
            task_started = time.perf_counter()
            product = task_results[f"{image.name}|detection"]
            duration = (time.perf_counter() - task_started) * 1000.0
            self.cache.put(
                stage="detection",
                key=key,
                image=image.name,
                dependencies={
                    "image_fingerprint": image.content_fingerprint,
                    "detection_params": self.config.detection_params,
                    "font_version": FONT_VERSION,
                },
                payload=product,
                font_version=FONT_VERSION,
            )
            detection_by_image[image.name] = StageState(
                key=key,
                state="recomputed",
                duration_ms=duration,
                product=product,
                recomputed_items=1,
            )
        timings["detection"] = (time.perf_counter() - detection_started) * 1000.0

        line_started = time.perf_counter()
        line_tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = []
        for image in loaded:
            detection = detection_by_image[image.name]
            key = cache_key(
                "line_aggregation",
                {
                    "image_fingerprint": image.content_fingerprint,
                    "detection_key": detection.key,
                    "detection_params": self.config.detection_params,
                    "line_params": self.config.line_params,
                    "font_version": FONT_VERSION,
                },
            )
            artifact = self.cache.get(key)
            if artifact is not None:
                lines_by_image[image.name] = StageState(
                    key=key,
                    state="reused",
                    duration_ms=0.0,
                    product=artifact["payload"],
                    reused_items=1,
                )
            else:
                line_tasks.append(
                    (
                        f"{image.name}|line_aggregation",
                        lambda image=image, detection=detection: aggregate_line_candidates(
                            image.gray, detection.product, self.config
                        ),
                    )
                )
        task_results, workers = _parallel_ordered(line_tasks, self.config.thread_pool_workers)
        observed_workers = max(observed_workers, workers)
        for image in loaded:
            if image.name in lines_by_image:
                continue
            detection = detection_by_image[image.name]
            key = cache_key(
                "line_aggregation",
                {
                    "image_fingerprint": image.content_fingerprint,
                    "detection_key": detection.key,
                    "detection_params": self.config.detection_params,
                    "line_params": self.config.line_params,
                    "font_version": FONT_VERSION,
                },
            )
            task_started = time.perf_counter()
            product = task_results[f"{image.name}|line_aggregation"]
            duration = (time.perf_counter() - task_started) * 1000.0
            self.cache.put(
                stage="line_aggregation",
                key=key,
                image=image.name,
                dependencies={
                    "detection_key": detection.key,
                    "line_params": self.config.line_params,
                    "font_version": FONT_VERSION,
                },
                payload=product,
                font_version=FONT_VERSION,
            )
            lines_by_image[image.name] = StageState(
                key=key,
                state="recomputed",
                duration_ms=duration,
                product=product,
                recomputed_items=1,
            )
        timings["line_aggregation"] = (time.perf_counter() - line_started) * 1000.0

        injection_types = self._fault_types()
        if "drop_last_line" in injection_types:
            target = self._fault_target("drop_last_line")
            target_image = next((img for img in loaded if not target or img.name == target), None)
            if target_image is not None:
                product = copy.deepcopy(lines_by_image[target_image.name].product)
                if product.get("candidate_lines"):
                    dropped = product["candidate_lines"].pop()
                    product["suppressed_lines"].append(
                        {**dropped, "reason": "injected_drop_last_line"}
                    )
                    lines_by_image[target_image.name].product = product
                fault_hits.append(
                    {
                        "id": self._fault_id("drop_last_line"),
                        "type": "drop_last_line",
                        "stage": "line_aggregation",
                        "image": target_image.name,
                    }
                )

        segmentation_started = time.perf_counter()
        segmentation_tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = []
        for image in loaded:
            line_state = lines_by_image[image.name]
            key = cache_key(
                "segmentation",
                {
                    "image_fingerprint": image.content_fingerprint,
                    "line_key": line_state.key,
                    "line_product_digest": digest(line_state.product),
                    "segmentation_params": self.config.segmentation_params,
                    "font_version": FONT_VERSION,
                },
            )
            artifact = self.cache.get(key)
            if artifact is not None:
                segmentation_by_image[image.name] = StageState(
                    key=key,
                    state="reused",
                    duration_ms=0.0,
                    product=artifact["payload"],
                    reused_items=1,
                )
            else:
                segmentation_tasks.append(
                    (
                        f"{image.name}|segmentation",
                        lambda image=image, line_state=line_state: {
                            "stage": "segmentation",
                            "lines": [
                                segment_line(image.gray, line, self.config)
                                for line in line_state.product.get("candidate_lines", [])
                            ],
                        },
                    )
                )
        task_results, workers = _parallel_ordered(segmentation_tasks, self.config.thread_pool_workers)
        observed_workers = max(observed_workers, workers)
        for image in loaded:
            if image.name in segmentation_by_image:
                continue
            line_state = lines_by_image[image.name]
            key = cache_key(
                "segmentation",
                {
                    "image_fingerprint": image.content_fingerprint,
                    "line_key": line_state.key,
                    "line_product_digest": digest(line_state.product),
                    "segmentation_params": self.config.segmentation_params,
                    "font_version": FONT_VERSION,
                },
            )
            task_started = time.perf_counter()
            product = task_results[f"{image.name}|segmentation"]
            duration = (time.perf_counter() - task_started) * 1000.0
            self.cache.put(
                stage="segmentation",
                key=key,
                image=image.name,
                dependencies={
                    "line_key": line_state.key,
                    "segmentation_params": self.config.segmentation_params,
                    "font_version": FONT_VERSION,
                },
                payload=product,
                font_version=FONT_VERSION,
            )
            segmentation_by_image[image.name] = StageState(
                key=key,
                state="recomputed",
                duration_ms=duration,
                product=product,
                recomputed_items=1,
            )
        timings["segmentation"] = (time.perf_counter() - segmentation_started) * 1000.0

        recognition_started = time.perf_counter()
        point_cost = match_point_cost()
        scheduled_points = 0
        recomputed_points = 0
        reused_points = 0
        pending_started = False
        pending_queue: list[dict[str, Any]] = []
        recognition_refs: list[dict[str, Any]] = []
        recognition_tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = []
        char_key_by_stage: dict[str, dict[str, str]] = {}
        for image in loaded:
            segmentation = segmentation_by_image[image.name]
            line_products = segmentation.product.get("lines", [])
            image_keys: dict[str, str] = {}
            for line_index, line_seg in enumerate(line_products):
                for segment in line_seg.get("segments", []):
                    key = cache_key(
                        "recognition",
                        {
                            "image_fingerprint": image.content_fingerprint,
                            "segmentation_key": segmentation.key,
                            "line_index": line_index,
                            "segment_index": segment["segment_index"],
                            "segment_bbox": segment["bbox"],
                            "recognition_params": self.config.recognition_params,
                            "font_version": FONT_VERSION,
                        },
                    )
                    image_keys[f"{line_index}:{segment['segment_index']}"] = key
                    ref = {
                        "key": key,
                        "image": image,
                        "line_index": line_index,
                        "segment": segment,
                    }
                    recognition_refs.append(ref)
                    if pending_started:
                        ref["state"] = "pending"
                        pending_queue.append(
                            {
                                "image": image.name,
                                "line_index": line_index,
                                "segment_index": segment["segment_index"],
                                "bbox": segment["bbox"],
                                "cache_key": key,
                                "code": "E_POINT_BUDGET_PENDING",
                            }
                        )
                        continue
                    if scheduled_points + point_cost > self.config.char_match_point_budget:
                        pending_started = True
                        ref["state"] = "pending"
                        pending_queue.append(
                            {
                                "image": image.name,
                                "line_index": line_index,
                                "segment_index": segment["segment_index"],
                                "bbox": segment["bbox"],
                                "cache_key": key,
                                "code": "E_POINT_BUDGET_PENDING",
                            }
                        )
                        continue
                    scheduled_points += point_cost
                    artifact = self.cache.get(key)
                    if artifact is not None:
                        ref["state"] = "reused"
                        ref["product"] = artifact["payload"]
                        reused_points += point_cost
                    else:
                        ref["state"] = "recomputed"
                        recognition_tasks.append(
                            (
                                f"{image.name}|recognition|{line_index:06d}|{segment['segment_index']:06d}",
                                lambda image=image, segment=segment: recognize_segment(
                                    image.gray, segment["bbox"], self.config
                                ),
                            )
                        )
            char_key_by_stage[image.name] = image_keys
        task_results, workers = _parallel_ordered(recognition_tasks, self.config.thread_pool_workers)
        observed_workers = max(observed_workers, workers)
        for ref in recognition_refs:
            if ref["state"] != "recomputed":
                continue
            image: LoadedImage = ref["image"]
            segment = ref["segment"]
            task_key = (
                f"{image.name}|recognition|{ref['line_index']:06d}|{segment['segment_index']:06d}"
            )
            task_started = time.perf_counter()
            product = task_results[task_key]
            duration = (time.perf_counter() - task_started) * 1000.0
            self.cache.put(
                stage="recognition",
                key=ref["key"],
                image=image.name,
                dependencies={
                    "recognition_params": self.config.recognition_params,
                    "segment_bbox": segment["bbox"],
                    "font_version": FONT_VERSION,
                },
                payload=product,
                font_version=FONT_VERSION,
            )
            ref["product"] = product
            ref["duration_ms"] = duration
            recomputed_points += point_cost
        timings["recognition"] = (time.perf_counter() - recognition_started) * 1000.0

        if "force_l1_metric" in injection_types:
            for ref in recognition_refs:
                if ref.get("state") in {"reused", "recomputed"} and ref.get("product"):
                    image = ref["image"]
                    ref["product"] = recognize_segment(
                        image.gray, ref["segment"]["bbox"], self.config, metric_override="l1"
                    )
            fault_hits.append(
                {
                    "id": self._fault_id("force_l1_metric"),
                    "type": "force_l1_metric",
                    "stage": "recognition",
                    "image": "*",
                }
            )

        by_line: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for ref in recognition_refs:
            by_line.setdefault((ref["image"].name, ref["line_index"]), []).append(ref)

        image_reports: list[dict[str, Any]] = []
        pending_images: set[str] = set()
        for image in loaded:
            detection = detection_by_image[image.name]
            line_state = lines_by_image[image.name]
            segmentation = segmentation_by_image[image.name]
            image_pending = False
            lines_report: list[dict[str, Any]] = []
            line_seg_products = segmentation.product.get("lines", [])
            candidate_lines = line_state.product.get("candidate_lines", [])
            for line_index, line_candidate in enumerate(candidate_lines):
                line_seg = line_seg_products[line_index] if line_index < len(line_seg_products) else {"segments": [], "records": []}
                refs = sorted(
                    by_line.get((image.name, line_index), []),
                    key=lambda ref: ref["segment"]["segment_index"],
                )
                chars: list[dict[str, Any]] = []
                for ref in refs:
                    segment = ref["segment"]
                    if ref["state"] == "pending":
                        image_pending = True
                        chars.append(
                            {
                                "index": len(chars),
                                "segment_index": segment["segment_index"],
                                "bbox": segment["bbox"],
                                "text": "",
                                "score": None,
                                "metric": self.config.match_metric,
                                "status": "pending",
                                "cache_state": "pending",
                                "cache_key": ref["key"],
                                "pending_code": "E_POINT_BUDGET_PENDING",
                                "candidates": [],
                            }
                        )
                        continue
                    product = ref["product"]
                    chars.append(
                        {
                            "index": len(chars),
                            "segment_index": segment["segment_index"],
                            "bbox": product["bbox"],
                            "text": product["text"],
                            "score": product["score"],
                            "metric": product["metric"],
                            "status": "recognized",
                            "cache_state": ref["state"],
                            "cache_key": ref["key"],
                            "tie_break": product["tie_break"],
                            "candidates": product["candidates"],
                            "candidate_count": product["candidate_count"],
                            "eliminated_beyond_top_k": product["eliminated_beyond_top_k"],
                        }
                    )
                if "reverse_character_order" in injection_types and chars:
                    target = self._fault_target("reverse_character_order")
                    if not target or target == image.name:
                        chars.reverse()
                        for index, char in enumerate(chars):
                            char["index"] = index
                        fault_hits.append(
                            {
                                "id": self._fault_id("reverse_character_order"),
                                "type": "reverse_character_order",
                                "stage": "recognition",
                                "image": image.name,
                                "line_index": line_index,
                            }
                        )
                line_text = "".join(char["text"] for char in chars if char["status"] == "recognized")
                lines_report.append(
                    {
                        "line_index": line_index,
                        "bbox": line_candidate["bbox"],
                        "score": line_candidate["score"],
                        "scale": line_candidate["scale"],
                        "blocks": line_candidate.get("blocks", []),
                        "segments": line_seg.get("segments", []),
                        "segmentation_records": line_seg.get("records", []),
                        "initial_fragments": line_seg.get("initial_fragments", []),
                        "text": line_text,
                        "characters": chars,
                        "pending_count": sum(1 for c in chars if c["status"] == "pending"),
                    }
                )
            if image_pending:
                pending_images.add(image.name)
            recognized_chars = [
                char
                for line in lines_report
                for char in line["characters"]
                if char["status"] == "recognized"
            ]
            reused_chars = sum(1 for char in recognized_chars if char["cache_state"] == "reused")
            recomputed_chars = sum(
                1 for char in recognized_chars if char["cache_state"] == "recomputed"
            )
            if image_pending and not recognized_chars:
                recognition_state = "pending"
            elif reused_chars and not recomputed_chars and not image_pending:
                recognition_state = "reused"
            elif recomputed_chars:
                recognition_state = "recomputed"
            elif reused_chars:
                recognition_state = "reused_pending"
            else:
                recognition_state = "pending"
            image_keys = char_key_by_stage.get(image.name, {})
            recognition_stage_key = digest(
                {
                    "stage": "recognition",
                    "image": image.name,
                    "segmentation_key": segmentation.key,
                    "keys": [image_keys[key] for key in sorted(image_keys)],
                    "recognition_params": self.config.recognition_params,
                    "font_version": FONT_VERSION,
                    "pipeline_cache_version": PIPELINE_CACHE_VERSION,
                }
            )
            image_reports.append(
                {
                    "name": image.name,
                    "status": "ok",
                    "image_hash": image.content_fingerprint,
                    "size": [image.width, image.height],
                    "mode": image.mode,
                    "truth": {
                        "valid": True,
                        "line_count": len(image.truth["lines"]),
                        "char_count": sum(len(line["chars"]) for line in image.truth["lines"]),
                    },
                    "stages": {
                        "detection": self._stage_dict(detection, "blocks"),
                        "line_aggregation": self._stage_dict(line_state, "candidate_lines"),
                        "segmentation": self._stage_dict(segmentation, "lines"),
                        "recognition": {
                            "state": recognition_state,
                            "cache_key": recognition_stage_key,
                            "duration_ms": round(
                                sum(
                                    ref.get("duration_ms", 0.0)
                                    for ref in recognition_refs
                                    if ref["image"].name == image.name
                                ),
                                3,
                            ),
                            "reused_characters": reused_chars,
                            "recomputed_characters": recomputed_chars,
                            "pending_characters": sum(
                                1 for line in lines_report for char in line["characters"] if char["status"] == "pending"
                            ),
                            "char_cache_keys": [image_keys[key] for key in sorted(image_keys)],
                        },
                    },
                    "lines": lines_report,
                    "suppressed_lines": [
                        {
                            "bbox": line["bbox"],
                            "score": line["score"],
                            "scale": line["scale"],
                            "reason": line.get("reason", "candidate_score_threshold"),
                            "block_count": len(line.get("blocks", [])),
                        }
                        for line in line_state.product.get("suppressed_lines", [])
                    ],
                    "pending_match_queue": [
                        item for item in pending_queue if item["image"] == image.name
                    ],
                }
            )

        cache_per_image: list[dict[str, Any]] = []
        reused_list: list[dict[str, Any]] = []
        recomputed_list: list[dict[str, Any]] = []
        for report in image_reports:
            stage_map: dict[str, Any] = {}
            for stage_name, stage in report["stages"].items():
                if stage_name == "recognition":
                    state = stage["state"]
                else:
                    state = stage["state"]
                stage_map[stage_name] = {
                    "state": state,
                    "cache_key": stage["cache_key"],
                }
                item = {
                    "image": report["name"],
                    "stage": stage_name,
                    "cache_key": stage["cache_key"],
                }
                if state == "reused":
                    reused_list.append(item)
                elif state.startswith("reused_pending"):
                    reused_list.append(item)
                else:
                    recomputed_list.append(item)
            cache_per_image.append({"image": report["name"], "stages": stage_map})
        reused_list.sort(key=lambda item: (item["image"], item["stage"]))
        recomputed_list.sort(key=lambda item: (item["image"], item["stage"]))

        total_elapsed = (time.perf_counter() - started) * 1000.0
        timings["total"] = total_elapsed
        report = {
            "schema": "layout-run-report-v1",
            "mode": mode,
            "command": command,
            "config_hash": self.config.config_hash,
            "font_version": FONT_VERSION,
            "pipeline_cache_version": PIPELINE_CACHE_VERSION,
            "match_metric": self.config.match_metric,
            "snapshot_version": snapshot_version,
            "images": image_reports,
            "failures": failures,
            "pending_match_queue": pending_queue,
            "budget": {
                "limit": self.config.char_match_point_budget,
                "point_cost_per_character": point_cost,
                "consumed": scheduled_points,
                "recomputed": recomputed_points,
                "reused": reused_points,
                "remaining": max(0, self.config.char_match_point_budget - scheduled_points),
                "pending_count": len(pending_queue),
            },
            "cache": {
                "per_image": cache_per_image,
                "reused": reused_list,
                "recomputed": recomputed_list,
                "stats": self.cache.stats.as_dict(),
            },
            "fault_hits": fault_hits,
            "timings_ms": {key: round(value, 3) for key, value in timings.items()},
            "concurrency": {
                "max_workers_configured": self.config.thread_pool_workers,
                "max_workers_observed": observed_workers,
                "never_exceeded_configured_limit": observed_workers
                <= self.config.thread_pool_workers,
                "stable_sort_key": "image_name|line_index|segment_index|stage",
                "stable_order": True,
                "conclusion": "stable",
            },
            "diagnostics": {
                "failure_count": len(failures),
                "suppressed_line_count": sum(
                    len(image["suppressed_lines"]) for image in image_reports
                ),
                "pending_count": len(pending_queue),
                "fault_hit_count": len(fault_hits),
                "cache_hit_count": self.cache.stats.hits,
                "cache_miss_count": self.cache.stats.misses,
            },
        }
        return report

    @staticmethod
    def _stage_dict(state: StageState, product_key: str) -> dict[str, Any]:
        product = state.product
        if product_key == "blocks":
            count = len(product.get("blocks", []))
        elif product_key == "candidate_lines":
            count = len(product.get("candidate_lines", []))
        else:
            count = len(product.get("lines", []))
        return {
            "state": state.state,
            "cache_key": state.key,
            "duration_ms": round(state.duration_ms, 3),
            "product_count": count,
            "reused_items": state.reused_items,
            "recomputed_items": state.recomputed_items,
        }

    def _fault_types(self) -> set[str]:
        if not self.respect_faults:
            return set()
        return {str(item["type"]) for item in self.config.fault_injections}

    def _fault_spec(self, fault_type: str) -> dict[str, Any]:
        return next(
            (item for item in self.config.fault_injections if item.get("type") == fault_type),
            {},
        )

    def _fault_target(self, fault_type: str) -> str | None:
        target = self._fault_spec(fault_type).get("image")
        return str(target) if target else None

    def _fault_id(self, fault_type: str) -> str:
        item = self._fault_spec(fault_type)
        return str(item.get("id", fault_type))
