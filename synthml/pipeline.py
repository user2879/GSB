"""Staged incremental execution, scoring, evaluation and reporting."""

from __future__ import annotations

import copy
import json
import math
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import yaml

from . import SCHEMA_VERSION, __version__
from .classifiers import ClassifierModel, TrainingInstance, decide_labels, train_model
from .config import (
    ExperimentConfig,
    InjectionItem,
    InjectionSet,
    config_with_threshold,
    load_injections,
    load_plan,
    load_yaml_mapping,
    parse_config,
)
from .errors import (
    ConfigValidationError,
    Failure,
    FeatureValidationError,
    ImageDecodeError,
    PipelineAbort,
    SampleStageError,
    SnapshotMismatchError,
)
from .imaging import FEATURE_ALGORITHM_VERSION, decode_png, extract_features
from .store import CurrentSnapshot, SnapshotSession, SnapshotStore
from .strategy import select_strategy
from .util import atomic_write_json, rounded, sha256_bytes, stable_hash


@dataclass
class RunOptions:
    samples_dir: Path
    config_path: Path
    plan_path: Path
    state_dir: Path
    report_path: Path
    mode: str = "incremental"
    tiers: list[str] | None = None
    injections_path: Path | None = None
    injection_ids: list[str] | None = None
    force_strategy: str | None = None
    force_feature_method: str | None = None
    force_color_bins: int | None = None
    force_hog_bins: int | None = None
    force_classifier: str | None = None
    force_knn_k: int | None = None
    force_threshold: float | None = None
    precision: str | None = None
    concurrency: int | None = None
    snapshot_interval: int | None = None


@dataclass
class RunResult:
    exit_code: int
    report: dict[str, Any]
    report_path: Path


@dataclass
class LoadedSample:
    name: str
    image_array: np.ndarray
    labels: tuple[str, ...]
    boxes: dict[str, list[int]]
    width: int
    height: int
    pixel_sha256: str
    file_sha256: str
    load_key: str


class InjectionTracker:
    def __init__(self, injections: InjectionSet) -> None:
        self.injections = injections
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        for item in injections.items:
            self._records[item.id] = {
                "id": item.id,
                "kind": item.kind,
                "sample": item.sample,
                "stage": item.stage,
                "enabled": item.enabled,
                "hit": False,
                "occurrences": 0,
                "business_code": None,
                "exception_type": None,
                "message": None,
            }

    def find(self, kind: str, sample: str | None = None) -> InjectionItem | None:
        return self.injections.by_kind(kind, sample)

    def inject(self, item: InjectionItem | None, code: str, message: str, exception_type: str) -> None:
        if item is None:
            return
        with self._lock:
            record = self._records[item.id]
            record["hit"] = True
            record["occurrences"] = int(record["occurrences"]) + 1
            if record["business_code"] is None:
                record["business_code"] = code
                record["exception_type"] = exception_type
                record["message"] = message

    def hits(self) -> list[dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(self._records[item_id]) for item_id in sorted(self._records)]


class TimingAccumulator:
    def __init__(self) -> None:
        self.values = {stage: 0.0 for stage in ("load", "feature", "score", "decision", "metric")}
        self.started = time.perf_counter()

    def add(self, stage: str, seconds: float) -> None:
        self.values[stage] = self.values.get(stage, 0.0) + float(seconds)

    def report(self) -> dict[str, Any]:
        total = time.perf_counter() - self.started
        result = {f"{stage}_seconds": rounded(value, 6) for stage, value in self.values.items()}
        result["total_seconds"] = rounded(total, 6)
        return result


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "pyyaml": yaml.__version__,
        "pipeline_schema_version": SCHEMA_VERSION,
        "pipeline_version": __version__,
    }


def _discover(samples_dir: Path) -> tuple[list[tuple[Path, Path]], list[Failure]]:
    if not samples_dir.is_dir():
        return (
            [],
            [
                Failure(
                    sample="<samples>",
                    stage="load",
                    code="E_SAMPLE_INVALID",
                    exception_type="MissingSampleDirectoryError",
                    message=f"sample directory does not exist: {samples_dir}",
                )
            ],
        )
    pngs = sorted(path for path in samples_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    jsons = sorted(path for path in samples_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")
    pairs: list[tuple[Path, Path]] = []
    failures: list[Failure] = []
    json_names = {path.name for path in jsons}
    paired_json: set[str] = set()
    for png in pngs:
        annotation = png.with_suffix(".json")
        if not annotation.is_file():
            failures.append(
                Failure(
                    sample=png.name,
                    stage="load",
                    code="E_SAMPLE_INVALID",
                    exception_type="MissingAnnotationError",
                    message=f"missing annotation JSON: {annotation.name}",
                )
            )
            continue
        pairs.append((png, annotation))
        paired_json.add(annotation.name)
    for json_path in jsons:
        if json_path.name not in paired_json:
            failures.append(
                Failure(
                    sample=json_path.name,
                    stage="load",
                    code="E_SAMPLE_INVALID",
                    exception_type="OrphanAnnotationError",
                    message="annotation JSON has no matching PNG",
                )
            )
    if not pairs and not failures:
        failures.append(
            Failure(
                sample="<samples>",
                stage="load",
                code="E_SAMPLE_INVALID",
                exception_type="EmptyDatasetError",
                message="no PNG/JSON sample pairs found",
            )
        )
    return pairs, failures


def _validate_annotation(
    annotation: Any,
    image_name: str,
    config: ExperimentConfig,
) -> tuple[tuple[str, ...], dict[str, list[int]]]:
    if not isinstance(annotation, dict):
        raise SampleStageError("E_SAMPLE_INVALID", "annotation root must be an object")
    if annotation.get("image") != image_name:
        raise SampleStageError(
            "E_SAMPLE_INVALID",
            f"annotation image {annotation.get('image')!r} does not match {image_name!r}",
        )
    labels_raw = annotation.get("labels")
    boxes_raw = annotation.get("boxes")
    if not isinstance(labels_raw, list):
        raise SampleStageError("E_SAMPLE_INVALID", "annotation labels must be a list")
    if not isinstance(boxes_raw, dict):
        raise SampleStageError("E_SAMPLE_INVALID", "annotation boxes must be an object")
    if not all(isinstance(label, str) and label for label in labels_raw):
        raise SampleStageError("E_SAMPLE_INVALID", "annotation labels must be non-empty strings")
    if len(set(labels_raw)) != len(labels_raw):
        raise SampleStageError("E_SAMPLE_INVALID", "annotation labels must not contain duplicates")
    labels = tuple(sorted(labels_raw))
    unknown = sorted(set(labels) - set(config.labels))
    if unknown:
        raise SampleStageError(
            "E_SAMPLE_INVALID", "annotation contains labels outside configured label set: " + ", ".join(unknown)
        )
    if set(boxes_raw) != set(labels):
        raise SampleStageError(
            "E_SAMPLE_INVALID", "annotation boxes keys must exactly match annotation labels"
        )
    boxes: dict[str, list[int]] = {}
    width, height = config.resolution
    for label in labels:
        box = boxes_raw[label]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in box)
        ):
            raise SampleStageError("E_SAMPLE_INVALID", f"bbox for {label} must be four integers")
        x1, y1, x2, y2 = (int(value) for value in box)
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            raise SampleStageError(
                "E_SAMPLE_INVALID",
                f"bbox for {label} is outside image bounds or degenerate: {box}",
            )
        boxes[label] = [x1, y1, x2, y2]
    return labels, {label: boxes[label] for label in sorted(boxes)}


def _injection_token_for_load(tracker: InjectionTracker, sample: str) -> str | None:
    ids: list[str] = []
    for kind in ("truncate_png", "unknown_label"):
        item = tracker.find(kind, sample)
        if item is not None:
            ids.append(item.id)
    return "+".join(sorted(ids)) if ids else None


def _load_one(
    png_path: Path,
    annotation_path: Path,
    config: ExperimentConfig,
    session: SnapshotSession,
    tracker: InjectionTracker,
) -> tuple[LoadedSample | None, Failure | None, str, str]:
    name = png_path.name
    try:
        data = png_path.read_bytes()
    except OSError as exc:
        return None, Failure(name, "load", "E_SAMPLE_INVALID", type(exc).__name__, str(exc)), "", "file_read_failed"
    truncate_item = tracker.find("truncate_png", name)
    if truncate_item is not None:
        tracker.inject(
            truncate_item,
            "E_SAMPLE_INVALID",
            f"injected truncated PNG bytes for {name}",
            "InjectedTruncatedPNG",
        )
        data = data[: max(8, len(data) // 3)]
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, Failure(name, "load", "E_SAMPLE_INVALID", type(exc).__name__, f"invalid annotation JSON: {exc}"), "", "annotation_read_failed"
    unknown_item = tracker.find("unknown_label", name)
    if unknown_item is not None:
        tracker.inject(
            unknown_item,
            "E_SAMPLE_INVALID",
            f"injected unknown label into {name}",
            "InjectedUnknownLabel",
        )
        if isinstance(annotation, dict):
            labels = annotation.get("labels")
            boxes = annotation.get("boxes")
            if not isinstance(labels, list):
                labels = []
            if not isinstance(boxes, dict):
                boxes = {}
            injected_label = "__INJECTED_UNKNOWN__"
            labels = list(labels) + [injected_label]
            boxes = dict(boxes)
            boxes[injected_label] = [0, 0, 1, 1]
            annotation["labels"] = labels
            annotation["boxes"] = boxes
    file_sha = sha256_bytes(data)
    injection_token = _injection_token_for_load(tracker, name)
    load_key = stable_hash(
        {
            "kind": "load",
            "schema_version": SCHEMA_VERSION,
            "load_algorithm_version": 1,
            "file_sha256": file_sha,
            "annotation": annotation,
            "labels_schema": config.labels,
            "resolution": list(config.resolution),
            "injection_token": injection_token,
        }
    )
    if session.has_artifact(load_key, require_sidecar=True):
        try:
            artifact = session.read_artifact(load_key)
            payload = artifact["payload"]
            sidecar = session.read_sidecar(load_key)
            width = int(payload["width"])
            height = int(payload["height"])
            array = np.frombuffer(sidecar, dtype=np.uint8).reshape((height, width, 3)).copy()
            if sha256_bytes(array.tobytes(order="C")) != payload["pixel_sha256"]:
                raise ValueError("cached load sidecar fingerprint mismatch")
            labels, boxes = _validate_annotation(payload, name, config)
            sample = LoadedSample(
                name=name,
                image_array=array,
                labels=labels,
                boxes=boxes,
                width=width,
                height=height,
                pixel_sha256=str(payload["pixel_sha256"]),
                file_sha256=str(payload["file_sha256"]),
                load_key=load_key,
            )
            return sample, None, load_key, "reused"
        except Exception:
            session.drop_artifact(load_key)
    try:
        labels, boxes = _validate_annotation(annotation, name, config)
        decoded = decode_png(data, config.resolution)
        array = decoded["array"]
        payload = {
            "image": name,
            "name": name,
            "width": int(decoded["width"]),
            "height": int(decoded["height"]),
            "labels": list(labels),
            "boxes": boxes,
            "pixel_sha256": decoded["pixel_sha256"],
            "file_sha256": file_sha,
        }
        metadata = {
            "decoder": "Pillow RGB uint8",
            "config_resolution": list(config.resolution),
            "annotation_sha256": stable_hash(annotation),
        }
        session.write_artifact(
            stage="load",
            key=load_key,
            tier=None,
            sample=name,
            classifier=None,
            payload=payload,
            metadata=metadata,
            input_refs=[file_sha, stable_hash(annotation)],
            sidecar=array.tobytes(order="C"),
        )
        return (
            LoadedSample(
                name=name,
                image_array=array,
                labels=labels,
                boxes=boxes,
                width=int(decoded["width"]),
                height=int(decoded["height"]),
                pixel_sha256=str(decoded["pixel_sha256"]),
                file_sha256=file_sha,
                load_key=load_key,
            ),
            None,
            load_key,
            "recomputed",
        )
    except SampleStageError as exc:
        return None, Failure(name, "load", exc.code, type(exc).__name__, exc.message, detail=exc.detail), load_key, "failed"
    except Exception as exc:
        return None, Failure(name, "load", "E_SAMPLE_INVALID", type(exc).__name__, str(exc)), load_key, "failed"


def _feature_key(
    sample: LoadedSample,
    tier_id: str,
    params: dict[str, Any],
    injection_id: str | None,
) -> str:
    return stable_hash(
        {
            "kind": "feature",
            "schema_version": SCHEMA_VERSION,
            "feature_algorithm_version": FEATURE_ALGORITHM_VERSION,
            "tier": tier_id,
            "pixel_sha256": sample.pixel_sha256,
            "image_shape": [sample.height, sample.width, 3],
            "extraction_parameters": params,
            "injection_id": injection_id,
        }
    )


def _feature_worker(
    sample: LoadedSample,
    params: dict[str, Any],
    injection_item: InjectionItem | None,
    tracker: InjectionTracker,
) -> tuple[list[float], dict[str, Any]]:
    vector, metadata = extract_features(sample.image_array, params)
    if injection_item is not None:
        vector = vector.copy()
        vector[0] = np.inf
        message = f"injected non-finite feature for {sample.name}"
        tracker.inject(injection_item, "E_FEATURE_INVALID", message, "InjectedNonFiniteFeature")
        raise FeatureValidationError(
            message,
            detail={"injection_id": injection_item.id, "component": 0, "value": "inf"},
        )
    if not np.all(np.isfinite(vector)):
        raise FeatureValidationError("feature vector contains non-finite values")
    return [rounded(float(value), int(params["rounding_decimals"])) for value in vector.tolist()], metadata


def _training_fingerprint(
    tier_id: str,
    labels: list[str],
    samples: list[tuple[LoadedSample, str, list[float]]],
) -> dict[str, Any]:
    return {
        "kind": "training_set",
        "tier": tier_id,
        "labels": list(labels),
        "samples": [
            {
                "name": sample.name,
                "pixel_sha256": sample.pixel_sha256,
                "labels": [label for label in labels if label in sample.labels],
                "feature_key": feature_key,
            }
            for sample, feature_key, _vector in sorted(samples, key=lambda item: item[0].name)
        ],
    }


def _metric_values(
    labels: list[str],
    predictions: list[dict[str, Any]],
    truth_by_sample: dict[str, tuple[str, ...]],
    metric_decimals: int,
) -> dict[str, Any]:
    per_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        true_positive = false_positive = false_negative = support = 0
        for item in predictions:
            truth = set(truth_by_sample[item["sample"]])
            predicted = set(item["predicted_labels"])
            truth_positive = label in truth
            predicted_positive = label in predicted
            if truth_positive:
                support += 1
            if truth_positive and predicted_positive:
                true_positive += 1
            elif not truth_positive and predicted_positive:
                false_positive += 1
            elif truth_positive and not predicted_positive:
                false_negative += 1
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": rounded(precision, metric_decimals),
            "recall": rounded(recall, metric_decimals),
            "f1": rounded(f1, metric_decimals),
            "support": support,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        }
    macro_precision = sum(per_label[label]["precision"] for label in labels) / max(1, len(labels))
    macro_recall = sum(per_label[label]["recall"] for label in labels) / max(1, len(labels))
    macro_f1 = sum(per_label[label]["f1"] for label in labels) / max(1, len(labels))
    return {
        "per_label": per_label,
        "macro_precision": rounded(macro_precision, metric_decimals),
        "macro_recall": rounded(macro_recall, metric_decimals),
        "macro_f1": rounded(macro_f1, metric_decimals),
        "samples_evaluated": len(predictions),
    }


def _reason_for_miss(
    *,
    stage: str,
    tier: str | None,
    sample: str | None,
    classifier: str | None,
    prior_index: dict[tuple[Any, ...], str],
    reuse: bool,
    mode: str,
) -> str:
    if mode == "rebuild":
        return "rebuild_forces_recompute"
    if not reuse:
        return "no_prior_snapshot"
    if (stage, tier, sample, classifier) in prior_index:
        return "cache_key_changed_by_upstream_or_parameters"
    return "new_artifact"


def _write_report(report: dict[str, Any], path: Path) -> None:
    atomic_write_json(path, report)


def _base_report(options: RunOptions) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": __version__,
        "exit_code": 0,
        "run": {
            "mode": options.mode,
            "samples_dir": str(options.samples_dir),
            "config_path": str(options.config_path),
            "plan_path": str(options.plan_path),
            "state_dir": str(options.state_dir),
            "report_path": str(options.report_path),
            "overrides": {
                "strategy": options.force_strategy,
                "feature_method": options.force_feature_method,
                "color_bins": options.force_color_bins,
                "hog_bins": options.force_hog_bins,
                "classifier": options.force_classifier,
                "knn_k": options.force_knn_k,
                "threshold": options.force_threshold,
                "precision": options.precision,
                "concurrency": options.concurrency,
                "snapshot_interval": options.snapshot_interval,
                "tiers": options.tiers,
            },
        },
        "environment": _environment(),
        "snapshot": None,
        "scope": {},
        "tiers": [],
        "failures": [],
        "injection_hits": [],
        "cache_audit": [],
        "reuse": [],
        "invalidations": [],
        "cache_hit_counts": {"load": 0, "feature": 0, "score": 0, "decision": 0, "metric": 0},
        "timings": {},
        "diagnosis": [],
        "error": None,
    }


def _abort_result(
    options: RunOptions,
    failure: Failure,
    tracker: InjectionTracker,
    timings: TimingAccumulator,
    code: str,
) -> RunResult:
    report = _base_report(options)
    report["exit_code"] = 2 if code == "E_CONFIG_INVALID" else 3 if code == "E_CONFIG_MISMATCH" else 4
    report["error"] = failure.to_dict()
    report["failures"] = [failure.to_dict()]
    report["injection_hits"] = tracker.hits()
    report["timings"] = timings.report()
    report["diagnosis"] = [f"run aborted with {code}: {failure.message}"]
    _write_report(report, options.report_path)
    return RunResult(exit_code=report["exit_code"], report=report, report_path=options.report_path)


def _validate_overrides(config: ExperimentConfig, options: RunOptions) -> None:
    errors: list[str] = []
    if options.mode not in {"incremental", "rebuild", "resume"}:
        errors.append("run mode must be incremental, rebuild or resume")
    if options.force_strategy is not None and options.force_strategy not in {
        strategy["id"] for strategy in config.strategies
    }:
        errors.append(f"unknown forced strategy: {options.force_strategy}")
    if options.force_feature_method is not None and options.force_feature_method not in config.features["supported"]:
        errors.append(f"unsupported forced feature method: {options.force_feature_method}")
    if options.force_color_bins is not None and options.force_color_bins not in config.features["color_bins_candidates"]:
        errors.append("forced color bins must be in features.color_bins_candidates")
    if options.force_hog_bins is not None and options.force_hog_bins not in config.features["hog_bins_candidates"]:
        errors.append("forced HOG bins must be in features.hog_bins_candidates")
    if options.force_color_bins is not None or options.force_hog_bins is not None:
        color_bins = int(options.force_color_bins or config.features["color_bins"])
        hog_bins = int(options.force_hog_bins or config.features["hog_bins"])
        for method in config.features["supported"]:
            if (method, color_bins, hog_bins) not in config.expected_dimensions:
                errors.append(
                    "forced bins have no declared expected dimension for feature method "
                    f"{method}: color={color_bins} hog={hog_bins}"
                )
    if options.force_classifier is not None and options.force_classifier not in config.classifiers["supported"]:
        errors.append(f"unsupported forced classifier: {options.force_classifier}")
    if options.force_knn_k is not None and options.force_knn_k not in config.classifiers["knn"]["k_candidates"]:
        errors.append("forced kNN k must be in classifiers.knn.k_candidates")
    if options.precision is not None and options.precision not in config.precision_levels:
        errors.append(f"unknown precision level: {options.precision}")
    if options.force_threshold is not None:
        value = options.force_threshold
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            errors.append("forced threshold must be finite")
        elif float(value) < 0.0 or float(value) > 1.0:
            errors.append("forced threshold must be in [0, 1]")
    if options.concurrency is not None:
        if options.concurrency <= 0:
            errors.append("concurrency override must be positive")
        elif options.concurrency > config.max_concurrency:
            errors.append("concurrency override exceeds max_concurrency")
    if options.snapshot_interval is not None and options.snapshot_interval <= 0:
        errors.append("snapshot interval override must be positive")
    if errors:
        raise ConfigValidationError(errors)


def _run_identity(
    config: ExperimentConfig,
    plan: dict[str, Any],
    options: RunOptions,
    injections: InjectionSet,
) -> str:
    return stable_hash(
        {
            "config_hash": config.config_hash,
            "plan_hash": plan["plan_hash"],
            "tiers": options.tiers,
            "overrides": {
                "strategy": options.force_strategy,
                "feature_method": options.force_feature_method,
                "color_bins": options.force_color_bins,
                "hog_bins": options.force_hog_bins,
                "classifier": options.force_classifier,
                "knn_k": options.force_knn_k,
                "threshold": options.force_threshold,
                "precision": options.precision,
                "concurrency": options.concurrency,
                "snapshot_interval": options.snapshot_interval,
            },
            "injections": [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "sample": item.sample,
                    "stage": item.stage,
                    "value": item.value,
                    "enabled": item.enabled,
                }
                for item in injections.items
            ],
        }
    )


def _execute(
    options: RunOptions,
    config: ExperimentConfig,
    plan: dict[str, Any],
    store: SnapshotStore,
    prior: CurrentSnapshot | None,
    reuse: bool,
    higher_version: int,
    tracker: InjectionTracker,
    timings: TimingAccumulator,
    session: SnapshotSession,
    run_identity: str,
) -> RunResult:
    concurrency = int(options.concurrency or config.concurrency)
    snapshot_interval = int(options.snapshot_interval or config.snapshot_interval)
    selected_tiers = list(plan["tiers"])
    if options.tiers:
        wanted = set(options.tiers)
        known = {tier["id"] for tier in plan["tiers"]}
        missing = sorted(wanted - known)
        if missing:
            raise ConfigValidationError(["unknown tier ids: " + ", ".join(missing)])
        selected_tiers = [tier for tier in plan["tiers"] if tier["id"] in wanted]
    failures: list[Failure] = []
    completed_cursor = {stage: "" for stage in ("load", "feature", "score", "decision", "metric")}
    active_keys: set[str] = set()
    prior_index = {
        (record.get("stage"), record.get("tier"), record.get("sample"), record.get("classifier")): key
        for key, record in session.artifacts.items()
    }

    def checkpoint() -> None:
        session.commit(
            config_hash=config.config_hash,
            plan_hash=plan["plan_hash"],
            completed_cursor=completed_cursor,
            tiers=selected_tiers,
            in_progress=True,
            active_keys=active_keys,
            extra={
                "run_mode": options.mode,
                "run_identity": run_identity,
                "tiers": [tier["id"] for tier in selected_tiers],
            },
        )

    # Load and validate every sample, including samples outside the largest tier.
    stage_start = time.perf_counter()
    pairs, discovery_failures = _discover(options.samples_dir)
    failures.extend(discovery_failures)
    loaded: list[LoadedSample] = []
    for index, (png_path, annotation_path) in enumerate(pairs, start=1):
        sample, failure, load_key, status = _load_one(
            png_path, annotation_path, config, session, tracker
        )
        if load_key:
            active_keys.add(load_key)
            reason = (
                "cache_key_present_in_prior_snapshot"
                if status == "reused"
                else _reason_for_miss(
                    stage="load",
                    tier=None,
                    sample=png_path.name,
                    classifier=None,
                    prior_index=prior_index,
                    reuse=reuse,
                    mode=options.mode,
                )
            )
            session.audit(
                stage="load",
                key=load_key,
                status=status,
                tier=None,
                sample=png_path.name,
                classifier=None,
                reason=reason,
                input_refs=[],
            )
        if failure is not None:
            failures.append(failure)
            if load_key and not session.has_artifact(load_key):
                # A failed sample has no successful artifact; keep its key in the audit only.
                active_keys.discard(load_key)
        else:
            assert sample is not None
            loaded.append(sample)
        completed_cursor["load"] = png_path.name
        if index < len(pairs) and index % snapshot_interval == 0:
            checkpoint()
    loaded.sort(key=lambda item: item.name)
    timings.add("load", time.perf_counter() - stage_start)

    tier_reports: list[dict[str, Any]] = []
    for tier in selected_tiers:
        tier_id = str(tier["id"])
        labels = config.labels[: int(tier["max_labels"])]
        selected = loaded[: int(tier["max_samples"])]
        sample_label_sets = [
            tuple(label for label in labels if label in sample.labels) for sample in selected
        ]
        decision_record = select_strategy(
            config,
            tier,
            labels,
            sample_label_sets,
            force_strategy=options.force_strategy,
            force_feature_method=options.force_feature_method,
            force_color_bins=options.force_color_bins,
            force_hog_bins=options.force_hog_bins,
            force_classifier=options.force_classifier,
            force_knn_k=options.force_knn_k,
            force_threshold=options.force_threshold,
            precision=options.precision,
        )
        feature_params = decision_record["selected"]["feature_parameters"]
        classifier_name = str(decision_record["selected"]["classifier"])
        classifier_params = decision_record["selected"]["classifier_hyperparameters"]
        threshold = float(decision_record["threshold"]["final"])

        # Feature stage.
        stage_start = time.perf_counter()
        feature_results: dict[str, dict[str, Any]] = {}
        tasks: list[tuple[LoadedSample, str, str | None]] = []
        for sample in selected:
            item = tracker.find("nonfinite_feature", sample.name)
            key = _feature_key(sample, tier_id, feature_params, item.id if item else None)
            tasks.append((sample, key, item.id if item else None))
        for chunk_start in range(0, len(tasks), snapshot_interval):
            chunk = tasks[chunk_start : chunk_start + snapshot_interval]
            pending: list[tuple[LoadedSample, str, InjectionItem | None]] = []
            for sample, key, _item_id in chunk:
                active_keys.add(key)
                if session.has_artifact(key):
                    try:
                        artifact = session.read_artifact(key)
                        vector = np.asarray(artifact["payload"]["feature_vector"], dtype=np.float64)
                        if vector.shape != (int(feature_params["feature_dimension"]),):
                            raise ValueError("cached feature dimension mismatch")
                        if not np.all(np.isfinite(vector)):
                            raise ValueError("cached feature is non-finite")
                        feature_results[sample.name] = {
                            "key": key,
                            "vector": vector,
                            "metadata": artifact["metadata"],
                        }
                        session.audit(
                            stage="feature",
                            key=key,
                            status="reused",
                            tier=tier_id,
                            sample=sample.name,
                            classifier=None,
                            reason="cache_key_present_in_prior_snapshot",
                            input_refs=[sample.pixel_sha256],
                        )
                        continue
                    except Exception:
                        session.drop_artifact(key)
                injection_item = tracker.find("nonfinite_feature", sample.name)
                pending.append((sample, key, injection_item))
            worker_results: dict[str, tuple[list[float], dict[str, Any]] | Exception] = {}
            if pending:
                with ThreadPoolExecutor(max_workers=min(concurrency, len(pending))) as executor:
                    futures = {
                        sample.name: executor.submit(
                            _feature_worker, sample, feature_params, injection_item, tracker
                        )
                        for sample, _key, injection_item in pending
                    }
                    for sample, _key, _item in sorted(pending, key=lambda item: item[0].name):
                        try:
                            worker_results[sample.name] = futures[sample.name].result()
                        except Exception as exc:  # noqa: BLE001 - isolate per-sample failures
                            worker_results[sample.name] = exc
            for sample, key, _item_id in sorted(pending, key=lambda item: item[0].name):
                result = worker_results[sample.name]
                reason = _reason_for_miss(
                    stage="feature",
                    tier=tier_id,
                    sample=sample.name,
                    classifier=None,
                    prior_index=prior_index,
                    reuse=reuse,
                    mode=options.mode,
                )
                if isinstance(result, Exception):
                    code = result.code if isinstance(result, SampleStageError) else "E_FEATURE_INVALID"
                    detail = result.detail if isinstance(result, SampleStageError) else {}
                    failures.append(
                        Failure(
                            sample.name,
                            "feature",
                            code,
                            type(result).__name__,
                            str(result),
                            tier=tier_id,
                            detail=detail,
                        )
                    )
                    session.audit(
                        stage="feature",
                        key=key,
                        status="failed",
                        tier=tier_id,
                        sample=sample.name,
                        classifier=None,
                        reason=reason,
                        input_refs=[sample.pixel_sha256],
                    )
                    active_keys.discard(key)
                    continue
                vector_list, metadata = result
                vector = np.asarray(vector_list, dtype=np.float64)
                session.write_artifact(
                    stage="feature",
                    key=key,
                    tier=tier_id,
                    sample=sample.name,
                    classifier=None,
                    payload={
                        "sample": sample.name,
                        "tier": tier_id,
                        "pixel_sha256": sample.pixel_sha256,
                        "feature_vector": vector_list,
                    },
                    metadata=metadata,
                    input_refs=[sample.pixel_sha256, stable_hash(feature_params)],
                )
                feature_results[sample.name] = {"key": key, "vector": vector, "metadata": metadata}
                session.audit(
                    stage="feature",
                    key=key,
                    status="recomputed",
                    tier=tier_id,
                    sample=sample.name,
                    classifier=None,
                    reason=reason,
                    input_refs=[sample.pixel_sha256],
                )
            if chunk:
                completed_cursor["feature"] = f"{tier_id}:{chunk[-1][0].name}"
                checkpoint()
        timings.add("feature", time.perf_counter() - stage_start)

        # Scoring stage: train one deterministic model, then score samples in stable key order.
        stage_start = time.perf_counter()
        training_pairs = [
            (sample, feature_results[sample.name]["key"], feature_results[sample.name]["vector"].tolist())
            for sample in selected
            if sample.name in feature_results
        ]
        training_fingerprint = _training_fingerprint(tier_id, labels, training_pairs)
        training_fingerprint_hash = stable_hash(training_fingerprint)
        scoring_signature = {
            "classifier": classifier_name,
            "classifier_parameters": classifier_params,
            "feature_parameters": feature_params,
            "probability_decimals": int(config.evaluation["probability_decimals"]),
        }
        model_key = stable_hash(
            {
                "kind": "classifier_model",
                "schema_version": SCHEMA_VERSION,
                "classifier_algorithm_version": 1,
                "tier": tier_id,
                "training_fingerprint": training_fingerprint_hash,
                "scoring_signature": scoring_signature,
            }
        )
        model_payload: dict[str, Any] | None = None
        model_status = "failed"
        model_training_error: Exception | None = None
        if session.has_artifact(model_key):
            try:
                artifact = session.read_artifact(model_key)
                payload = artifact.get("payload")
                if not isinstance(payload, dict) or payload.get("classifier") != classifier_name:
                    raise ValueError("cached classifier model is malformed")
                model_payload = payload
                model_status = "reused"
                active_keys.add(model_key)
                session.audit(
                    stage="score",
                    key=model_key,
                    status="reused",
                    tier=tier_id,
                    sample="<model>",
                    classifier=classifier_name,
                    reason="cache_key_present_in_prior_snapshot",
                    input_refs=[training_fingerprint_hash],
                )
            except Exception:
                session.drop_artifact(model_key)
        if model_payload is None:
            if training_pairs:
                instances = [
                    TrainingInstance(
                        name=sample.name,
                        features=vector,
                        labels=tuple(label for label in labels if label in sample.labels),
                    )
                    for sample, _key, values in training_pairs
                    for vector in [np.asarray(values, dtype=np.float64)]
                ]
                try:
                    model_payload = train_model(classifier_name, classifier_params, labels, instances)
                    session.write_artifact(
                        stage="score",
                        key=model_key,
                        tier=tier_id,
                        sample="<model>",
                        classifier=classifier_name,
                        payload=model_payload,
                        metadata={
                            "training_fingerprint": training_fingerprint_hash,
                            "training_order": [instance.name for instance in instances],
                        },
                        input_refs=[training_fingerprint_hash],
                    )
                    model_status = "recomputed"
                    active_keys.add(model_key)
                    session.audit(
                        stage="score",
                        key=model_key,
                        status="recomputed",
                        tier=tier_id,
                        sample="<model>",
                        classifier=classifier_name,
                        reason=_reason_for_miss(
                            stage="score",
                            tier=tier_id,
                            sample="<model>",
                            classifier=classifier_name,
                            prior_index=prior_index,
                            reuse=reuse,
                            mode=options.mode,
                        ),
                        input_refs=[training_fingerprint_hash],
                    )
                except Exception as exc:
                    model_payload = None
                    model_training_error = exc
        score_results: dict[str, dict[str, Any]] = {}
        if model_payload is None:
            for sample in selected:
                if sample.name in feature_results:
                    failures.append(
                        Failure(
                            sample.name,
                            "score",
                            "E_MODEL_TRAIN",
                            type(model_training_error).__name__ if model_training_error else "NoModelError",
                            str(model_training_error) if model_training_error else "classifier model could not be trained",
                            tier=tier_id,
                        )
                    )
        else:
            predictor = ClassifierModel(model_payload, int(config.evaluation["probability_decimals"]))
            score_tasks: list[tuple[LoadedSample, str]] = []
            for sample in selected:
                if sample.name not in feature_results:
                    continue
                feature_key = feature_results[sample.name]["key"]
                score_key = stable_hash(
                    {
                        "kind": "score",
                        "schema_version": SCHEMA_VERSION,
                        "score_algorithm_version": 1,
                        "tier": tier_id,
                        "sample": sample.name,
                        "model_key": model_key,
                        "feature_key": feature_key,
                        "classifier": classifier_name,
                        "scoring_signature": scoring_signature,
                    }
                )
                score_tasks.append((sample, score_key))
            for chunk_start in range(0, len(score_tasks), snapshot_interval):
                chunk = score_tasks[chunk_start : chunk_start + snapshot_interval]
                pending_scores: list[tuple[LoadedSample, str]] = []
                for sample, score_key in chunk:
                    active_keys.add(score_key)
                    if session.has_artifact(score_key):
                        try:
                            artifact = session.read_artifact(score_key)
                            payload = artifact["payload"]
                            if sorted(payload.get("probabilities", {})) != labels:
                                raise ValueError("cached score labels mismatch")
                            score_results[sample.name] = {"key": score_key, "payload": payload}
                            session.audit(
                                stage="score",
                                key=score_key,
                                status="reused",
                                tier=tier_id,
                                sample=sample.name,
                                classifier=classifier_name,
                                reason="cache_key_present_in_prior_snapshot",
                                input_refs=[feature_results[sample.name]["key"], model_key],
                            )
                            continue
                        except Exception:
                            session.drop_artifact(score_key)
                    pending_scores.append((sample, score_key))
                worker_scores: dict[str, dict[str, Any] | Exception] = {}
                if pending_scores:
                    with ThreadPoolExecutor(max_workers=min(concurrency, len(pending_scores))) as executor:
                        futures = {
                            sample.name: executor.submit(
                                predictor.score,
                                sample.name,
                                feature_results[sample.name]["vector"],
                            )
                            for sample, _score_key in pending_scores
                        }
                        for sample, _score_key in sorted(pending_scores, key=lambda item: item[0].name):
                            try:
                                worker_scores[sample.name] = futures[sample.name].result()
                            except Exception as exc:  # noqa: BLE001 - isolate per-sample failures
                                worker_scores[sample.name] = exc
                for sample, score_key in sorted(pending_scores, key=lambda item: item[0].name):
                    result = worker_scores[sample.name]
                    reason = _reason_for_miss(
                        stage="score",
                        tier=tier_id,
                        sample=sample.name,
                        classifier=classifier_name,
                        prior_index=prior_index,
                        reuse=reuse,
                        mode=options.mode,
                    )
                    if isinstance(result, Exception):
                        failures.append(
                            Failure(
                                sample.name,
                                "score",
                                "E_SCORE_INVALID",
                                type(result).__name__,
                                str(result),
                                tier=tier_id,
                            )
                        )
                        session.audit(
                            stage="score",
                            key=score_key,
                            status="failed",
                            tier=tier_id,
                            sample=sample.name,
                            classifier=classifier_name,
                            reason=reason,
                            input_refs=[feature_results[sample.name]["key"], model_key],
                        )
                        active_keys.discard(score_key)
                        continue
                    payload = result
                    session.write_artifact(
                        stage="score",
                        key=score_key,
                        tier=tier_id,
                        sample=sample.name,
                        classifier=classifier_name,
                        payload=payload,
                        metadata={
                            "model_key": model_key,
                            "feature_key": feature_results[sample.name]["key"],
                            "raw_and_probability_score": True,
                        },
                        input_refs=[feature_results[sample.name]["key"], model_key],
                    )
                    score_results[sample.name] = {"key": score_key, "payload": payload}
                    session.audit(
                        stage="score",
                        key=score_key,
                        status="recomputed",
                        tier=tier_id,
                        sample=sample.name,
                        classifier=classifier_name,
                        reason=reason,
                        input_refs=[feature_results[sample.name]["key"], model_key],
                    )
                if chunk:
                    completed_cursor["score"] = f"{tier_id}:{chunk[-1][0].name}"
                    checkpoint()
        timings.add("score", time.perf_counter() - stage_start)

        # Threshold decision stage.
        stage_start = time.perf_counter()
        decision_results: dict[str, dict[str, Any]] = {}
        decision_tasks: list[tuple[str, str]] = []
        for sample_name in sorted(score_results):
            decision_key = stable_hash(
                {
                    "kind": "decision",
                    "schema_version": SCHEMA_VERSION,
                    "decision_algorithm_version": 1,
                    "tier": tier_id,
                    "sample": sample_name,
                    "score_key": score_results[sample_name]["key"],
                    "threshold": threshold,
                    "boundary_policy": config.evaluation["boundary_policy"],
                }
            )
            decision_tasks.append((sample_name, decision_key))
        for chunk_start in range(0, len(decision_tasks), snapshot_interval):
            chunk = decision_tasks[chunk_start : chunk_start + snapshot_interval]
            for sample_name, decision_key in chunk:
                active_keys.add(decision_key)
                if session.has_artifact(decision_key):
                    try:
                        artifact = session.read_artifact(decision_key)
                        payload = artifact["payload"]
                        if payload.get("threshold") != threshold:
                            raise ValueError("cached decision threshold mismatch")
                        decision_results[sample_name] = {"key": decision_key, "payload": payload}
                        session.audit(
                            stage="decision",
                            key=decision_key,
                            status="reused",
                            tier=tier_id,
                            sample=sample_name,
                            classifier=classifier_name,
                            reason="cache_key_present_in_prior_snapshot",
                            input_refs=[score_results[sample_name]["key"]],
                        )
                        continue
                    except Exception:
                        session.drop_artifact(decision_key)
                score_payload = score_results[sample_name]["payload"]
                decided = decide_labels(
                    score_payload["probabilities"],
                    threshold,
                    str(config.evaluation["boundary_policy"]),
                )
                payload = {
                    "sample": sample_name,
                    "tier": tier_id,
                    "classifier": classifier_name,
                    "score_key": score_results[sample_name]["key"],
                    "threshold": threshold,
                    **decided,
                }
                session.write_artifact(
                    stage="decision",
                    key=decision_key,
                    tier=tier_id,
                    sample=sample_name,
                    classifier=classifier_name,
                    payload=payload,
                    metadata={
                        "threshold": threshold,
                        "boundary_policy": config.evaluation["boundary_policy"],
                    },
                    input_refs=[score_results[sample_name]["key"]],
                )
                decision_results[sample_name] = {"key": decision_key, "payload": payload}
                session.audit(
                    stage="decision",
                    key=decision_key,
                    status="recomputed",
                    tier=tier_id,
                    sample=sample_name,
                    classifier=classifier_name,
                    reason=_reason_for_miss(
                        stage="decision",
                        tier=tier_id,
                        sample=sample_name,
                        classifier=classifier_name,
                        prior_index=prior_index,
                        reuse=reuse,
                        mode=options.mode,
                    ),
                    input_refs=[score_results[sample_name]["key"]],
                )
            if chunk:
                completed_cursor["decision"] = f"{tier_id}:{chunk[-1][0]}"
                checkpoint()
        timings.add("decision", time.perf_counter() - stage_start)

        # Metric stage.
        stage_start = time.perf_counter()
        truth_by_sample = {
            sample.name: tuple(label for label in labels if label in sample.labels) for sample in selected
        }
        predictions = [
            {
                "sample": sample_name,
                "predicted_labels": decision_results[sample_name]["payload"]["predicted_labels"],
                "decision_key": decision_results[sample_name]["key"],
            }
            for sample_name in sorted(decision_results)
        ]
        excluded = sorted(
            sample.name for sample in selected if sample.name not in decision_results
        )
        metric_key = stable_hash(
            {
                "kind": "metrics",
                "schema_version": SCHEMA_VERSION,
                "metric_algorithm_version": 1,
                "tier": tier_id,
                "labels": labels,
                "classifier": classifier_name,
                "threshold": threshold,
                "metric_decimals": int(config.evaluation["metric_decimals"]),
                "predictions": predictions,
                "truth": {name: list(truth_by_sample[name]) for name in sorted(truth_by_sample)},
                "excluded": excluded,
            }
        )
        active_keys.add(metric_key)
        if session.has_artifact(metric_key):
            try:
                artifact = session.read_artifact(metric_key)
                metrics = artifact["payload"]
                session.audit(
                    stage="metric",
                    key=metric_key,
                    status="reused",
                    tier=tier_id,
                    sample="<metrics>",
                    classifier=classifier_name,
                    reason="cache_key_present_in_prior_snapshot",
                    input_refs=[item["decision_key"] for item in predictions],
                )
            except Exception:
                session.drop_artifact(metric_key)
                metrics = None
        else:
            metrics = None
        if metrics is None:
            metrics = _metric_values(
                labels, predictions, truth_by_sample, int(config.evaluation["metric_decimals"])
            )
            metrics["excluded_samples"] = excluded
            session.write_artifact(
                stage="metric",
                key=metric_key,
                tier=tier_id,
                sample="<metrics>",
                classifier=classifier_name,
                payload=metrics,
                metadata={"threshold": threshold, "metric_decimals": int(config.evaluation["metric_decimals"])},
                input_refs=[item["decision_key"] for item in predictions],
            )
            session.audit(
                stage="metric",
                key=metric_key,
                status="recomputed",
                tier=tier_id,
                sample="<metrics>",
                classifier=classifier_name,
                reason=_reason_for_miss(
                    stage="metric",
                    tier=tier_id,
                    sample="<metrics>",
                    classifier=classifier_name,
                    prior_index=prior_index,
                    reuse=reuse,
                    mode=options.mode,
                ),
                input_refs=[item["decision_key"] for item in predictions],
            )
        completed_cursor["metric"] = tier_id
        checkpoint()
        timings.add("metric", time.perf_counter() - stage_start)

        reference_sets: dict[str, set[str]] = {label: set() for label in labels}
        sample_reports: list[dict[str, Any]] = []
        for sample_name in sorted(decision_results):
            score_payload = score_results[sample_name]["payload"]
            decision_payload = decision_results[sample_name]["payload"]
            for label in labels:
                reference_sets[label].update(score_payload.get("references", {}).get(label, []))
            sample_reports.append(
                {
                    "sample": sample_name,
                    "truth_labels": list(truth_by_sample[sample_name]),
                    "raw_scores": score_payload["raw_scores"],
                    "probabilities": score_payload["probabilities"],
                    "predicted_labels": decision_payload["predicted_labels"],
                    "boundary_labels": decision_payload["boundary_labels"],
                    "boundary": decision_payload["boundary"],
                    "boundary_policy": decision_payload["boundary_policy"],
                    "decisions": decision_payload["decisions"],
                    "score_references": score_payload.get("references", {}),
                    "cache_keys": {
                        "feature": feature_results[sample_name]["key"],
                        "score": score_results[sample_name]["key"],
                        "decision": decision_results[sample_name]["key"],
                    },
                }
            )
        classifier_diagnostics: dict[str, Any] = {
            "model_key": model_key,
            "training_sample_count": len(training_pairs),
            "training_fingerprint": training_fingerprint_hash,
            "probability_normalization": (
                "sigmoid(logit)"
                if classifier_name == "logistic_regression"
                else "sigmoid(4*normalized_centroid_margin)"
                if classifier_name == "nearest_centroid"
                else "laplace_smoothed_positive_vote_fraction"
            ),
        }
        if model_payload is not None:
            classifier_diagnostics.update(
                {
                    "effective_k": model_payload.get("effective_k"),
                    "loss_history": model_payload.get("loss_history"),
                    "label_model_references": model_payload.get("references"),
                }
            )
        tier_reports.append(
            {
                "tier_id": tier_id,
                "labels": labels,
                "selected_strategy": decision_record,
                "feature_metadata": feature_params,
                "classifier_diagnostics": classifier_diagnostics,
                "metrics": metrics,
                "samples": sample_reports,
                "excluded_samples": excluded,
                "label_score_references": {
                    label: sorted(reference_sets[label]) for label in labels
                },
            }
        )

    failures_sorted = sorted(failures, key=lambda item: item.sort_key())
    audit = session.audit_sorted()
    reuse_entries = [entry for entry in audit if entry["status"] == "reused"]
    invalidations = [entry for entry in audit if entry["status"] in {"recomputed", "failed"}]
    hit_counts = session.cache_hit_counts()
    stage_versions = session.stage_versions(active_keys)
    expected_version = session.next_version
    global_version = f"v{expected_version:06d}-{config.config_hash[:12]}"
    report = _base_report(options)
    report.update(
        {
            "exit_code": 0,
            "scope": {
                "all_labels": config.labels,
                "selected_tiers": selected_tiers,
                "tier_ids": [tier["id"] for tier in selected_tiers],
                "resolution": list(config.resolution),
                "precision_level": options.precision or config.precision,
                "concurrency": concurrency,
                "snapshot_interval": snapshot_interval,
            },
            "tiers": tier_reports,
            "failures": [failure.to_dict() for failure in failures_sorted],
            "injection_hits": tracker.hits(),
            "cache_audit": audit,
            "reuse": reuse_entries,
            "invalidations": invalidations,
            "cache_hit_counts": hit_counts,
            "timings": {},
            "snapshot": {
                "snapshot_version": expected_version,
                "snapshot_name": f"v{expected_version:06d}",
                "global_version": global_version,
                "config_hash": config.config_hash,
                "plan_hash": plan["plan_hash"],
                "stage_versions": stage_versions,
                "previous_snapshot": prior.name if prior else None,
                "run_identity": run_identity,
                "in_progress": False,
            },
        }
    )
    diagnosis: list[str] = []
    for tier_report in tier_reports:
        strategy = tier_report["selected_strategy"]
        diagnosis.append(
            f"{tier_report['tier_id']}: samples={strategy['input_metrics']['sample_count']} "
            f"labels={strategy['input_metrics']['label_count']} "
            f"imbalance={strategy['input_metrics']['imbalance_ratio']} -> "
            f"{strategy['selected']['feature_method']}+{strategy['selected']['classifier']} "
            f"macro_f1={tier_report['metrics']['macro_f1']}"
        )
    diagnosis.append(f"failures={len(failures_sorted)}; active_injections={len(tracker.hits())}")
    report["diagnosis"] = diagnosis
    report["timings"] = timings.report()
    report_sha = session.write_report(report)
    committed = session.commit(
        config_hash=config.config_hash,
        plan_hash=plan["plan_hash"],
        completed_cursor=completed_cursor,
        tiers=selected_tiers,
        in_progress=False,
        active_keys=active_keys,
        report_sha256=report_sha,
        extra={
            "run_mode": options.mode,
            "run_identity": run_identity,
            "tiers": [tier["id"] for tier in selected_tiers],
        },
    )
    if int(committed["snapshot_version"]) != expected_version:
        raise PipelineAbort("E_INTERNAL", "snapshot version changed during final commit", stage="state")
    store.prune()
    _write_report(report, options.report_path)
    return RunResult(exit_code=0, report=report, report_path=options.report_path)


def run_pipeline(options: RunOptions) -> RunResult:
    timings = TimingAccumulator()
    tracker = InjectionTracker(InjectionSet(items=[], path=Path("<none>")))
    session: SnapshotSession | None = None
    try:
        try:
            injections = load_injections(options.injections_path, options.injection_ids)
        except ConfigValidationError as exc:
            return _abort_result(
                options,
                Failure(
                    "<injections>",
                    "config",
                    "E_CONFIG_INVALID",
                    type(exc).__name__,
                    exc.message,
                    detail={"errors": exc.errors},
                ),
                tracker,
                timings,
                "E_CONFIG_INVALID",
            )
        tracker = InjectionTracker(injections)
        raw = load_yaml_mapping(options.config_path)
        threshold_injection = injections.by_kind("threshold_out_of_range")
        if threshold_injection is not None:
            tracker.inject(
                threshold_injection,
                "E_CONFIG_INVALID",
                f"injected threshold {threshold_injection.value!r}",
                "InjectedThresholdOutOfRange",
            )
            raw = config_with_threshold(raw, threshold_injection.value)
        try:
            config = parse_config(raw, options.config_path)
        except ConfigValidationError as exc:
            return _abort_result(
                options,
                Failure(
                    "<config>",
                    "config",
                    "E_CONFIG_INVALID",
                    type(exc).__name__,
                    exc.message,
                    detail={"errors": exc.errors},
                ),
                tracker,
                timings,
                "E_CONFIG_INVALID",
            )
        try:
            _validate_overrides(config, options)
            plan = load_plan(options.plan_path, config.labels)
        except ConfigValidationError as exc:
            return _abort_result(
                options,
                Failure(
                    "<config>",
                    "config",
                    "E_CONFIG_INVALID",
                    type(exc).__name__,
                    exc.message,
                    detail={"errors": exc.errors},
                ),
                tracker,
                timings,
                "E_CONFIG_INVALID",
            )
        run_identity = _run_identity(config, plan, options, injections)
        store = SnapshotStore(options.state_dir, SCHEMA_VERSION)
        prior: CurrentSnapshot | None = None
        reuse = False
        try:
            if options.mode == "rebuild":
                prior = None
                reuse = False
            elif options.mode == "resume":
                prior = store.load_current()
                if prior is None:
                    raise SnapshotMismatchError("resume requested but no current snapshot exists")
                if prior.manifest.get("config_hash") != config.config_hash:
                    raise SnapshotMismatchError(
                        "resume configuration hash does not match the snapshot",
                        {
                            "snapshot_config_hash": prior.manifest.get("config_hash"),
                            "run_config_hash": config.config_hash,
                        },
                    )
                if prior.manifest.get("plan_hash") != plan["plan_hash"]:
                    raise SnapshotMismatchError(
                        "resume scale-plan hash does not match the snapshot",
                        {
                            "snapshot_plan_hash": prior.manifest.get("plan_hash"),
                            "run_plan_hash": plan["plan_hash"],
                        },
                    )
                prior_identity = prior.manifest.get("extra", {}).get("run_identity")
                if prior_identity != run_identity:
                    raise SnapshotMismatchError(
                        "resume run identity does not match the snapshot (overrides or injections differ)",
                        {
                            "snapshot_run_identity": prior_identity,
                            "run_run_identity": run_identity,
                        },
                    )
                reuse = True
            else:
                prior = store.load_current()
                reuse = prior is not None
        except SnapshotMismatchError as exc:
            return _abort_result(
                options,
                Failure(
                    "<state>",
                    "state",
                    "E_CONFIG_MISMATCH",
                    type(exc).__name__,
                    exc.message,
                    detail=exc.detail,
                ),
                tracker,
                timings,
                "E_CONFIG_MISMATCH",
            )
        session = store.begin(
            prior=prior,
            reuse=reuse,
            higher_version=store.max_version(),
        )
        return _execute(
            options,
            config,
            plan,
            store,
            prior,
            reuse,
            store.max_version(),
            tracker,
            timings,
            session,
            run_identity,
        )
    except PipelineAbort as exc:
        return _abort_result(
            options,
            Failure(
                "<pipeline>",
                exc.stage,
                exc.code,
                type(exc).__name__,
                exc.message,
                detail=exc.detail,
            ),
            tracker,
            timings,
            exc.code,
        )
    except Exception as exc:  # noqa: BLE001 - report unexpected run-level failures
        return _abort_result(
            options,
            Failure(
                "<pipeline>",
                "state",
                "E_INTERNAL",
                type(exc).__name__,
                str(exc),
            ),
            tracker,
            timings,
            "E_INTERNAL",
        )
    finally:
        if session is not None:
            session.close()
