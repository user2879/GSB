"""Incremental, fixed-point vertical-layout recognition engine."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import __version__
from .config import config_hash, load_config, normalize_config, stage_param_fingerprint
from .errors import (
    AppError,
    E_CONFIG_INVALID,
    E_MA_BUDGET_PENDING,
    E_TEMPLATE_MISSING,
)
from .imaging import binarize
from .injection import InjectionPlan, load_injection_plan, record_injection_observations
from .loader import Dataset, load_dataset
from .matching import compare_points, match_character, prepare_templates
from .pipeline import make_border_artifact, make_columns_artifact, make_split_artifact
from .runtime import validate_runtime
from .snapshot import SnapshotHandle, SnapshotStore
from .templates import (
    build_templates,
    clone_templates,
    template_variant_count,
    template_version,
)
from .util import pretty_json, sha256_bytes, sha256_json


ENGINE_VERSION = f"engine-{__version__}"


@dataclass
class RunResult:
    recognition: dict[str, Any]
    report: dict[str, Any]
    diagnostics: dict[str, Any]
    snapshot_manifest: dict[str, Any]
    snapshot_path: Path
    ledger_events: list[dict[str, Any]]

    @property
    def pending(self) -> list[dict[str, Any]]:
        return list(self.recognition.get("pending", []))


class RecognitionEngine:
    def __init__(
        self,
        samples_dir: str | Path,
        *,
        config_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        state_dir: str | Path = ".ancient-layout-state",
        injection_mode: bool = False,
        force_rebuild: bool = False,
    ):
        self.samples_dir = Path(samples_dir)
        self.config_path = Path(config_path) if config_path is not None else self.samples_dir / "config.json"
        self.state_dir = Path(state_dir)
        self.injection_mode = bool(injection_mode)
        self.force_rebuild = bool(force_rebuild)
        self.templates = build_templates()
        self.template_version = template_version(self.templates)
        self.variant_count = template_variant_count()
        if config is None:
            self.config = load_config(self.config_path, variant_count=self.variant_count)
        else:
            self.config = normalize_config(copy.deepcopy(config), variant_count=self.variant_count)
        self.config_hash = config_hash(self.config)
        self.runtime_report = validate_runtime()
        injection_path = self.samples_dir / self.config["injection_manifest"]
        self.injection_plan: InjectionPlan = load_injection_plan(injection_path, self.injection_mode)
        self.store = SnapshotStore(self.state_dir)
        self.current = None if force_rebuild else self.store.load_current()
        self._peak_lock = threading.Lock()
        self._active_tasks = 0
        self._peak_tasks = 0
        self._reset_peak()

    def _reset_peak(self) -> None:
        with self._peak_lock:
            self._active_tasks = 0
            self._peak_tasks = 0

    def _enter_worker(self) -> None:
        with self._peak_lock:
            self._active_tasks += 1
            self._peak_tasks = max(self._peak_tasks, self._active_tasks)

    def _leave_worker(self) -> None:
        with self._peak_lock:
            self._active_tasks -= 1

    def _parallel(self, tasks: list[dict[str, Any]], worker: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        if not tasks:
            return {}
        workers = min(int(self.config["thread_pool_workers"]), len(tasks))
        results: dict[str, Any] = {}

        def wrapped(task: dict[str, Any]) -> Any:
            self._enter_worker()
            try:
                return worker(task)
            finally:
                self._leave_worker()

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ancient-layout") as pool:
            futures = {pool.submit(wrapped, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                results[task["stable_key"]] = future.result()
        return results

    def _reset_injection_items(self) -> None:
        for item in self.injection_plan.items:
            item.applied = False
            item.observed_codes = []

    def _effective_image_configs(
        self, png_names: Iterable[str]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
        valid: dict[str, dict[str, Any]] = {}
        errors: dict[str, dict[str, Any]] = {}
        injection_observations: list[dict[str, Any]] = []
        for name in sorted(set(png_names)):
            effective, touched = self.injection_plan.effective_config(self.config, name)
            try:
                normalized = normalize_config(effective, variant_count=self.variant_count)
                valid[name] = normalized
            except AppError as exc:
                errors[name] = exc.as_dict()
                for item in touched:
                    injection_observations.append(
                        {"target": name, "code": E_CONFIG_INVALID, "injection_id": item.id}
                    )
        return valid, errors, injection_observations

    def _prior_artifact(self, stage: str, key: str) -> tuple[dict[str, Any] | None, str | None]:
        if self.current is None or self.force_rebuild:
            return None, None
        files = self.current.manifest.get("stage_files", {}).get(stage, {})
        relative = files.get(key)
        if not relative:
            return None, None
        try:
            artifact = self.current.read_stage_json(relative)
            if artifact.get("kind") != stage or artifact.get("key") != key:
                raise ValueError("artifact kind/key mismatch")
            return artifact, relative
        except (OSError, ValueError, TypeError):
            return None, relative

    def _invalidation_reasons(self, stage: str, product: str, old_key: str | None) -> list[str]:
        if self.force_rebuild:
            return ["forced_full_rebuild"]
        if self.current is None:
            return ["no_previous_snapshot"]
        old_products = self.current.manifest.get("products", {})
        if product not in old_products:
            return ["new_product"]
        old_entry = old_products[product]
        if old_entry.get("key") == old_key:
            return ["cache_artifact_missing_or_corrupt"]
        reasons = ["stage_parameter_changed"]
        prior_config = self.current.manifest.get("config", {})
        changed = sorted(
            key
            for key in self.config
            if prior_config.get(key) != self.config.get(key)
        )
        if changed:
            reasons.append("config_fields:" + ",".join(changed))
        prior_images = self.current.manifest.get("image_hashes", {})
        if stage in {"columns", "split", "border", "match"}:
            reasons.append("upstream_or_content_dependency_changed")
        if stage == "columns" and prior_images:
            reasons.append("image_content_or_parameter_changed")
        if stage == "match" and self.current.manifest.get("template_version") != self.template_version:
            reasons.append("template_version_changed")
        return reasons

    def _record_stage(
        self,
        stage: str,
        product: str,
        key: str,
        artifact: dict[str, Any] | None,
        relative: str | None,
        *,
        hit: bool,
        reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        if artifact is None:
            raise ValueError("stage artifact cannot be None")
        if hit:
            self.concurrent_hits.append({"stage": stage, "product": product, "cache_key": key})
        else:
            self.concurrent_recomputed.append({"stage": stage, "product": product, "cache_key": key})
            if reasons:
                self.concurrent_invalidated.append(
                    {"stage": stage, "product": product, "cache_key": key, "reasons": sorted(set(reasons))}
                )
        self.product_map[product] = {
            "stage": stage,
            "key": key,
            "path": f"stages/{stage}/{key}.json",
        }
        self.used_artifacts[stage][key] = artifact
        return artifact

    def _get_or_compute(
        self,
        stage: str,
        key: str,
        product: str,
        compute: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        artifact, relative = self._prior_artifact(stage, key)
        if artifact is not None:
            return self._record_stage(stage, product, key, artifact, relative, hit=True)
        reasons = self._invalidation_reasons(stage, product, key)
        result = compute()
        return self._record_stage(stage, product, key, result, None, hit=False, reasons=reasons)

    def _stage_file_map(self) -> dict[str, dict[str, str]]:
        return {
            stage: {key: f"stages/{stage}/{key}.json" for key in sorted(artifacts)}
            for stage, artifacts in sorted(self.used_artifacts.items())
        }

    def _stage_versions(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for stage, artifacts in sorted(self.used_artifacts.items()):
            values = [
                [key, sha256_bytes(pretty_json(artifact).encode("utf-8"))]
                for key, artifact in sorted(artifacts.items())
            ]
            result[stage] = sha256_json(values)
        return result

    def _build_match_tasks(
        self,
        descriptors: list[dict[str, Any]],
        templates_by_image: dict[str, dict[str, list[np.ndarray]]],
        available_versions: dict[str, str],
        configs_by_image: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for descriptor in descriptors:
            image_name = descriptor["image_name"]
            config = configs_by_image[image_name]
            match_key = sha256_json(
                {
                    "stage": "match-v1",
                    "engine_version": ENGINE_VERSION,
                    "image_fingerprint": descriptor["image_fingerprint"],
                    "border_key": descriptor["border_key"],
                    "segment_fingerprint": descriptor["segment_fingerprint"],
                    "bbox": descriptor["bbox"],
                    "parameters_fingerprint": stage_param_fingerprint("match", config),
                    "match_metric": config["match_metric"],
                    "fusion_weights": config["fusion_weights"],
                    "fixed_point_bits": config["fixed_point_bits"],
                    "confidence_quantization_steps": config["confidence_quantization_steps"],
                    "template_version": self.template_version,
                    "available_template_version": available_versions[image_name],
                }
            )
            task = dict(descriptor)
            task.update(
                {
                    "match_key": match_key,
                    "product": descriptor["match_product"],
                    "stable_key": f"{image_name}:c{descriptor['column_index']:04d}:h{descriptor['char_index']:04d}:match",
                    "points": compare_points(descriptor["bbox"], self.variant_count),
                    "templates": templates_by_image[image_name],
                    "available_version": available_versions[image_name],
                }
            )
            tasks.append(task)
        tasks.sort(key=lambda item: item["stable_key"])
        return tasks

    def _match_worker(
        self,
        task: dict[str, Any],
        image_arrays: dict[str, np.ndarray],
        prepared_by_image: dict[str, dict[str, list[dict[str, Any]]]],
    ) -> dict[str, Any]:
        return match_character(
            image_arrays[task["image_name"]],
            task["bbox"],
            task["effective_config"],
            task["templates"],
            task["image_name"],
            task["column_index"],
            task["char_index"],
            task["piece_id"],
            task["segment_id"],
            task["parent_segment_id"],
            task["column_range"],
            self.template_version,
            task["match_key"],
            prepared_by_image[task["image_name"]],
        )

    def _assemble_recognition(
        self,
        dataset: Dataset,
        columns_by_image: dict[str, dict[str, Any]],
        char_descriptors: list[dict[str, Any]],
        match_results: dict[str, dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pending_by_task = {entry["task_key"]: entry for entry in pending}
        by_image: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for descriptor in char_descriptors:
            entry = match_results.get(descriptor["match_product"])
            if entry is None:
                task = pending_by_task.get(descriptor["match_product"])
                entry = {
                    "bbox": descriptor["bbox"],
                    "recognized_char": None,
                    "status": "pending",
                    "errors": [E_MA_BUDGET_PENDING],
                    "confidence_value": None,
                    "confidence_fixed": None,
                    "fixed_point_bits": descriptor["effective_config"]["fixed_point_bits"],
                    "points_required": task["points"] if task else descriptor["points"],
                }
            else:
                entry = {
                    "bbox": entry["bbox"],
                    "recognized_char": entry["recognized_char"],
                    "status": "ok",
                    "errors": [],
                    "confidence_value": entry["confidence_value"],
                    "confidence_fixed": entry["confidence_fixed"],
                    "confidence_raw_fixed": entry["confidence_raw_fixed"],
                    "confidence_digits": entry["confidence_digits"],
                    "confidence_quantization_steps": entry["confidence_quantization_steps"],
                    "confidence_quantum": entry["confidence_quantum"],
                    "confidence_level": entry["confidence_level"],
                    "confidence_levels": entry["confidence_levels"],
                    "fixed_point_bits": entry["fixed_point_bits"],
                    "fixed_point_scale": entry["fixed_point_scale"],
                    "top_score_fixed": entry["top_score_fixed"],
                    "second_score_fixed": entry["second_score_fixed"],
                    "score_margin_fixed": entry["score_margin_fixed"],
                    "coverage_fixed": entry["coverage_fixed"],
                    "match_metric": entry["match_metric"],
                    "top_tied_chars": entry["top_tied_chars"],
                    "tie_break": entry["tie_break"],
                }
            entry.update(
                {
                    "char_index": descriptor["char_index"],
                    "reading_index": descriptor["reading_index"],
                    "column_ref": descriptor["column_range"],
                    "column_index": descriptor["column_index"],
                    "candidate_segment": descriptor["segment_id"],
                    "segment_fingerprint": descriptor["segment_fingerprint"],
                    "parent_block_id": descriptor["parent_segment_id"],
                    "piece_id": descriptor["piece_id"],
                }
            )
            by_image.setdefault(descriptor["image_name"], {}).setdefault(
                descriptor["column_index"], []
            ).append(entry)

        images: list[dict[str, Any]] = []
        for image_name in sorted(dataset.images):
            artifact = columns_by_image[image_name]
            image_chars = by_image.get(image_name, {})
            columns = []
            for column in sorted(artifact["columns"], key=lambda item: item["column_index"]):
                chars = sorted(
                    image_chars.get(column["column_index"], []),
                    key=lambda item: (item["char_index"], item["bbox"][1], item["bbox"][0]),
                )
                columns.append(
                    {
                        "column_index": int(column["column_index"]),
                        "x0": int(column["x0"]),
                        "x1": int(column["x1"]),
                        "chars": chars,
                    }
                )
            images.append(
                {
                    "file": image_name,
                    "content_fingerprint": dataset.images[image_name].content_fingerprint,
                    "columns": columns,
                }
            )
        return {
            "schema_version": 1,
            "engine_version": ENGINE_VERSION,
            "config_hash": self.config_hash,
            "template_version": self.template_version,
            "images": images,
            "failures": sorted(dataset.failures, key=lambda item: item["file"]),
            "pending": sorted(pending, key=lambda item: item["stable_key"]),
        }

    def run(self) -> RunResult:
        self._reset_peak()
        self._reset_injection_items()
        self.concurrent_hits: list[dict[str, Any]] = []
        self.concurrent_recomputed: list[dict[str, Any]] = []
        self.concurrent_invalidated: list[dict[str, Any]] = []
        self.product_map: dict[str, Any] = {}
        self.used_artifacts: dict[str, dict[str, dict[str, Any]]] = {
            "columns": {},
            "split": {},
            "border": {},
            "match": {},
        }

        png_names = sorted(
            path.name for path in self.samples_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
        )
        configs_by_image, config_errors, config_injection_observations = self._effective_image_configs(
            png_names
        )
        raw_overrides = self.injection_plan.raw_overrides(self.samples_dir)
        truth_mutators = self.injection_plan.truth_mutators()
        dataset = load_dataset(
            self.samples_dir,
            self.config,
            raw_overrides=raw_overrides,
            truth_mutators=truth_mutators,
            external_errors=config_errors,
        )
        image_set_hash = dataset.image_set_hash()
        injection_hash = self.injection_plan.manifest_hash
        environment_hash = sha256_json(
            {
                "engine_version": ENGINE_VERSION,
                "config_hash": self.config_hash,
                "template_version": self.template_version,
                "image_set_hash": image_set_hash,
                "injection_mode": self.injection_mode,
                "injection_hash": injection_hash,
            }
        )

        columns_by_image: dict[str, dict[str, Any]] = {}
        column_compute_tasks: list[dict[str, Any]] = []
        for image_name in sorted(dataset.images):
            image = dataset.images[image_name]
            key = sha256_json(
                {
                    "stage": "columns-v1",
                    "engine_version": ENGINE_VERSION,
                    "image_fingerprint": image.content_fingerprint,
                    "parameters_fingerprint": stage_param_fingerprint("columns", self.config),
                }
            )
            product = f"columns:{image_name}"
            artifact, relative = self._prior_artifact("columns", key)
            if artifact is not None:
                columns_by_image[image_name] = self._record_stage(
                    "columns", product, key, artifact, relative, hit=True
                )
            else:
                column_compute_tasks.append(
                    {
                        "image_name": image_name,
                        "key": key,
                        "product": product,
                        "relative": relative,
                        "stable_key": f"{image_name}:columns",
                    }
                )

        def compute_columns(task: dict[str, Any]) -> dict[str, Any]:
            image = dataset.images[task["image_name"]]
            return make_columns_artifact(image.array, self.config, image.content_fingerprint, task["key"])

        computed_columns = self._parallel(column_compute_tasks, compute_columns)
        for task in column_compute_tasks:
            reasons = self._invalidation_reasons("columns", task["product"], task["key"])
            columns_by_image[task["image_name"]] = self._record_stage(
                "columns",
                task["product"],
                task["key"],
                computed_columns[task["stable_key"]],
                None,
                hit=False,
                reasons=reasons,
            )

        split_artifacts: dict[str, dict[str, Any]] = {}
        border_artifacts: dict[str, dict[str, Any]] = {}
        char_descriptors: list[dict[str, Any]] = []
        segment_diagnostics: list[dict[str, Any]] = []
        split_diagnostics: list[dict[str, Any]] = []
        for image_name in sorted(dataset.images):
            image = dataset.images[image_name]
            ink = binarize(image.array, self.config["binarization_threshold"])
            reading_index = 0
            for column in columns_by_image[image_name]["columns"]:
                split_key = sha256_json(
                    {
                        "stage": "split-v1",
                        "engine_version": ENGINE_VERSION,
                        "image_fingerprint": image.content_fingerprint,
                        "column_range": [column["x0"], column["x1"]],
                        "column_key": columns_by_image[image_name]["key"],
                        "parameters_fingerprint": stage_param_fingerprint("split", self.config),
                    }
                )
                split_product = f"split:{image_name}:c{column['column_index']:04d}"

                def compute_split(
                    column: dict[str, Any] = column,
                    split_key: str = split_key,
                ) -> dict[str, Any]:
                    return make_split_artifact(
                        ink,
                        self.config,
                        image_name,
                        image.content_fingerprint,
                        columns_by_image[image_name]["key"],
                        split_key,
                        column,
                    )

                split = self._get_or_compute(
                    "split", split_key, split_product, compute_split
                )
                split_artifacts[split_product] = split
                for discarded in split["discarded"]:
                    segment_diagnostics.append(
                        {
                            "file": image_name,
                            "column_index": column["column_index"],
                            **discarded,
                        }
                    )
                char_index = 0
                for segment in split["candidates"]:
                    border_key = sha256_json(
                        {
                            "stage": "border-v1",
                            "engine_version": ENGINE_VERSION,
                            "image_fingerprint": image.content_fingerprint,
                            "segment_fingerprint": segment["segment_fingerprint"],
                            "split_key": split_key,
                            "parameters_fingerprint": stage_param_fingerprint("border", self.config),
                        }
                    )
                    border_product = f"border:{segment['segment_id']}"

                    def compute_border(
                        segment: dict[str, Any] = segment,
                        border_key: str = border_key,
                    ) -> dict[str, Any]:
                        return make_border_artifact(
                            ink,
                            self.config,
                            border_key,
                            segment,
                            split_key,
                            split_key,
                        )

                    border = self._get_or_compute(
                        "border", border_key, border_product, compute_border
                    )
                    border_artifacts[border_product] = border
                    split_diagnostics.append(
                        {
                            "file": image_name,
                            "column_index": column["column_index"],
                            "segment_id": segment["segment_id"],
                            "parent_segment_id": segment["segment_id"],
                            "segment_fingerprint": segment["segment_fingerprint"],
                            "split": {
                                key: value
                                for key, value in border["split"].items()
                                if key != "infeasible_cuts"
                            },
                            "infeasible_cuts": border["split"].get("infeasible_cuts", []),
                        }
                    )
                    for piece in border["pieces"]:
                        effective_config = configs_by_image[image_name]
                        match_product = f"match:{image_name}:c{column['column_index']:04d}:h{char_index:04d}"
                        char_descriptors.append(
                            {
                                "image_name": image_name,
                                "image_fingerprint": image.content_fingerprint,
                                "column_index": int(column["column_index"]),
                                "column_range": [int(column["x0"]), int(column["x1"])],
                                "char_index": int(char_index),
                                "reading_index": int(reading_index),
                                "segment_id": segment["segment_id"],
                                "parent_segment_id": piece["parent_segment_id"],
                                "segment_fingerprint": segment["segment_fingerprint"],
                                "piece_id": piece["piece_id"],
                                "bbox": list(piece["bbox"]),
                                "border_key": border_key,
                                "match_product": match_product,
                                "effective_config": effective_config,
                                "points": compare_points(piece["bbox"], self.variant_count),
                            }
                        )
                        char_index += 1
                        reading_index += 1

        templates_by_image: dict[str, dict[str, list[np.ndarray]]] = {}
        prepared_by_image: dict[str, dict[str, list[dict[str, Any]]]] = {}
        available_versions: dict[str, str] = {}
        missing_template_diagnostics: list[dict[str, Any]] = []
        for image_name in sorted(dataset.images):
            missing, touched = self.injection_plan.missing_templates(image_name)
            available = clone_templates(self.templates)
            for char in sorted(missing):
                if char in available:
                    del available[char]
                missing_template_diagnostics.append(
                    {
                        "target": image_name,
                        "file": image_name,
                        "code": E_TEMPLATE_MISSING,
                        "template_id": char,
                        "injection_ids": sorted(item.id for item in touched),
                    }
                )
            templates_by_image[image_name] = available
            prepared_by_image[image_name] = prepare_templates(
                available, 1 << int(configs_by_image[image_name]["fixed_point_bits"])
            )
            available_versions[image_name] = sha256_json(
                [self.template_version, sorted(available)]
            )

        match_tasks = self._build_match_tasks(
            char_descriptors,
            templates_by_image,
            available_versions,
            configs_by_image,
        )
        match_results: dict[str, dict[str, Any]] = {}
        pending_tasks: list[dict[str, Any]] = []
        pending_priority: set[str] = set()
        prior_pending = self.current.manifest.get("pending", []) if self.current else []
        for entry in prior_pending:
            pending_priority.add(str(entry.get("task_key")))

        to_compute: list[dict[str, Any]] = []
        for task in match_tasks:
            artifact, relative = self._prior_artifact("match", task["match_key"])
            if artifact is not None:
                match_results[task["product"]] = self._record_stage(
                    "match", task["product"], task["match_key"], artifact, relative, hit=True
                )
            else:
                task["relative"] = relative
                to_compute.append(task)

        to_compute.sort(
            key=lambda task: (
                0 if task["product"] in pending_priority else 1,
                task["stable_key"],
            )
        )
        prior_ledger = self.current.manifest.get("budget_ledger", {}) if self.current else {}
        ledger_events: list[dict[str, Any]] = list(prior_ledger.get("events", []))
        batch_number = int(prior_ledger.get("batch_count", 0)) + 1
        current_events: list[dict[str, Any]] = []
        remaining = int(self.config["ma_point_budget"])
        reserved: list[dict[str, Any]] = []
        for task in to_compute:
            cost = int(task["points"])
            if cost > remaining:
                continue
            before = remaining
            remaining -= cost
            event = {
                "sequence": len(ledger_events) + len(current_events) + 1,
                "batch": batch_number,
                "task_key": task["product"],
                "stable_key": task["stable_key"],
                "points": cost,
                "remaining_before": before,
                "remaining_after": remaining,
                "match_key": task["match_key"],
            }
            current_events.append(event)
            reserved.append(task)

        reserved_set = {task["product"] for task in reserved}

        image_arrays = {name: image.array for name, image in dataset.images.items()}

        def compute_match(task: dict[str, Any]) -> dict[str, Any]:
            return self._match_worker(task, image_arrays, prepared_by_image)

        computed_matches = self._parallel(reserved, compute_match)
        for task in reserved:
            artifact = computed_matches[task["stable_key"]]
            reasons = self._invalidation_reasons("match", task["product"], task["match_key"])
            match_results[task["product"]] = self._record_stage(
                "match",
                task["product"],
                task["match_key"],
                artifact,
                None,
                hit=False,
                reasons=reasons,
            )
        for task in to_compute:
            if task["product"] in reserved_set:
                continue
            pending_tasks.append(
                {
                    "task_key": task["product"],
                    "stable_key": task["stable_key"],
                    "match_key": task["match_key"],
                    "file": task["image_name"],
                    "column_index": task["column_index"],
                    "char_index": task["char_index"],
                    "bbox": task["bbox"],
                    "points": task["points"],
                    "code": E_MA_BUDGET_PENDING,
                }
            )

        ledger_events.extend(current_events)
        budget_ledger = {
            "events": ledger_events,
            "batch_count": batch_number,
            "last_batch": batch_number,
        }
        recognition = self._assemble_recognition(
            dataset,
            columns_by_image,
            char_descriptors,
            match_results,
            pending_tasks,
        )

        observed: list[dict[str, Any]] = list(config_injection_observations)
        observed.extend(
            {"file": failure["file"], "code": failure["code"]} for failure in dataset.failures
        )
        observed.extend(missing_template_diagnostics)
        record_injection_observations(self.injection_plan.items, observed)
        injection_report = {
            "mode": "fault" if self.injection_mode else "normal",
            "manifest_hash": injection_hash,
            "items": [item.as_dict() for item in sorted(self.injection_plan.items, key=lambda item: item.id)],
            "diagnostics": self.injection_plan.diagnostics,
        }

        prior_environment = self.current.manifest.get("environment_hash") if self.current else None
        prior_config_hash = self.current.manifest.get("config_hash") if self.current else None
        prior_pointer_recovered = self.current.pointer_recovered if self.current else False
        prior_pending_records = prior_pending
        state_changed = (
            self.current is None
            or prior_environment != environment_hash
            or bool(self.concurrent_recomputed)
            or bool(current_events)
            or pending_tasks != prior_pending_records
        )

        stage_files = self._stage_file_map()
        stage_versions = self._stage_versions()
        image_hashes = {entry["file"]: entry.get("image_sha256") for entry in dataset.entries}
        manifest_fields = {
            "engine_version": ENGINE_VERSION,
            "environment_hash": environment_hash,
            "config_hash": self.config_hash,
            "template_version": self.template_version,
            "image_set_hash": image_set_hash,
            "image_hashes": image_hashes,
            "injection_mode": self.injection_mode,
            "injection_hash": injection_hash,
            "config": self.config,
            "stage_files": stage_files,
            "stage_versions": stage_versions,
            "products": self.product_map,
            "pending": recognition["pending"],
            "budget_ledger": budget_ledger,
            "runtime": self.runtime_report,
            "snapshot_interval": int(self.config["snapshot_interval"]),
        }

        if state_changed:
            files: dict[str, Any] = {"results.json": recognition}
            for stage, artifacts in self.used_artifacts.items():
                for key, artifact in artifacts.items():
                    files[f"stages/{stage}/{key}.json"] = artifact
            committed = self.store.commit(manifest_fields, files, self.current)
            self.current = committed

        assert self.current is not None
        cache_report = {
            "hits": sorted(self.concurrent_hits, key=lambda item: (item["product"], item["stage"])),
            "recomputed": sorted(
                self.concurrent_recomputed, key=lambda item: (item["product"], item["stage"])
            ),
            "invalidated": sorted(
                self.concurrent_invalidated, key=lambda item: (item["product"], item["stage"])
            ),
            "hit_count": len(self.concurrent_hits),
            "recomputed_count": len(self.concurrent_recomputed),
        }
        budget_report = {
            "batch": batch_number,
            "ma_point_budget": int(self.config["ma_point_budget"]),
            "remaining_after_batch": remaining,
            "current_events": current_events,
            "cumulative_event_count": len(ledger_events),
            "pending_count": len(pending_tasks),
        }
        concurrency_report = {
            "configured_max_workers": int(self.config["thread_pool_workers"]),
            "observed_peak_tasks": int(self._peak_tasks),
            "column_task_order": sorted(task["stable_key"] for task in column_compute_tasks),
            "match_task_order": sorted(task["stable_key"] for task in match_tasks),
        }
        diagnostics = {
            "schema_version": 1,
            "runtime": self.runtime_report,
            "failures": recognition["failures"],
            "discarded_segments": sorted(
                segment_diagnostics,
                key=lambda item: (item["file"], item["column_index"], item["y0"], item["y1"]),
            ),
            "split_diagnostics": sorted(
                split_diagnostics,
                key=lambda item: (item["file"], item["column_index"], item["segment_id"]),
            ),
            "missing_templates": sorted(
                missing_template_diagnostics, key=lambda item: (item["file"], item["template_id"])
            ),
            "pending": recognition["pending"],
            "injections": injection_report,
        }
        snapshot_state = {
            "generation": self.current.generation,
            "global_version": self.current.manifest.get("global_version"),
            "config_hash": self.current.manifest.get("config_hash"),
            "template_version": self.current.manifest.get("template_version"),
            "environment_hash": self.current.manifest.get("environment_hash"),
            "stage_versions": self.current.manifest.get("stage_versions", {}),
            "snapshot_interval": self.current.manifest.get(
                "snapshot_interval", int(self.config["snapshot_interval"])
            ),
            "commit_scope": "whole_generation_after_batch",
            "resumed": prior_environment == environment_hash,
            "config_changed": prior_config_hash is not None and prior_config_hash != self.config_hash,
            "version_changed": prior_environment is not None and prior_environment != environment_hash,
            "pointer_recovered": bool(prior_pointer_recovered),
        }
        report = {
            "schema_version": 1,
            "batch": batch_number,
            "snapshot": snapshot_state,
            "cache": cache_report,
            "budget": budget_report,
            "concurrency": concurrency_report,
            "injections": injection_report,
            "runtime_ok": self.runtime_report["ok"],
            "pending": recognition["pending"],
        }
        return RunResult(
            recognition=recognition,
            report=report,
            diagnostics=diagnostics,
            snapshot_manifest=self.current.manifest,
            snapshot_path=self.current.path,
            ledger_events=current_events,
        )

    def settle(self, max_batches: int = 100) -> RunResult:
        last: RunResult | None = None
        batches = 0
        previous_pending: list[dict[str, Any]] | None = None
        while batches < max_batches:
            last = self.run()
            batches += 1
            if not last.pending:
                break
            if previous_pending == last.pending and not last.ledger_events:
                break
            previous_pending = last.pending
        assert last is not None
        last.report["settle"] = {"batches": batches, "pending": len(last.pending)}
        return last
