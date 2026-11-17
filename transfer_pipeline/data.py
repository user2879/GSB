"""Input parsing and validation for tasks, samples, encoders, and adapter configs."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .canonical import array_fingerprint, ensure_finite_number, fingerprint
from .errors import (
    AdapterConfigError,
    EncoderError,
    FeatureLibraryError,
    PipelineConfigError,
    TaskDescriptionError,
)


def read_structured_file(path: str | Path) -> Any:
    """Read JSON, or YAML for .yaml/.yml files, without implicit coercion."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return json.load(handle)


def _mapping(value: Any, label: str, error_type: type[Exception]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{label} must be an object")
    return value


def _string_list(value: Any, label: str, error_type: type[Exception]) -> list[str]:
    if not isinstance(value, list):
        raise error_type(f"{label} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise error_type(f"{label}[{index}] must be a non-empty string")
        result.append(item)
    return result


def _canonical_label_value(value: Any, label: str, error_type: type[Exception]) -> str:
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, bool) and isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return format(value, ".17g")
    raise error_type(f"{label} must be a non-empty string or numeric class label")


@dataclass(frozen=True)
class FeatureSample:
    sample_id: str
    features: tuple[float, ...]
    label: str

    def as_array(self) -> np.ndarray:
        return np.asarray(self.features, dtype=np.float64)

    def feature_fingerprint(self) -> str:
        return fingerprint(
            {"sample_id": self.sample_id, "features": list(self.features)},
            domain="sample-features",
        )

    def content_fingerprint(self) -> str:
        return fingerprint(
            {
                "sample_id": self.sample_id,
                "features": list(self.features),
                "label": self.label,
            },
            domain="sample-content",
        )


@dataclass(frozen=True)
class FeatureLibrary:
    samples: Mapping[str, FeatureSample]
    feature_dim: int
    seed: int | None

    def __post_init__(self) -> None:
        if not self.samples:
            raise FeatureLibraryError("feature library must contain at least one sample")
        if self.feature_dim <= 0:
            raise FeatureLibraryError("feature_dim must be a positive integer")

    def sorted_ids(self) -> list[str]:
        return sorted(self.samples)

    def fingerprint(self) -> str:
        return fingerprint(
            {
                "feature_dim": self.feature_dim,
                "samples": {
                    sample_id: {
                        "features": list(self.samples[sample_id].features),
                        "label": self.samples[sample_id].label,
                    }
                    for sample_id in self.sorted_ids()
                },
            },
            domain="feature-library",
        )


@dataclass(frozen=True)
class EncoderPackage:
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(self.weights.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.weights.shape[0])

    def fingerprint(self) -> str:
        return fingerprint(
            {
                "weights": array_fingerprint(self.weights),
                "bias": array_fingerprint(self.bias),
                "mean": array_fingerprint(self.mean),
                "std": array_fingerprint(self.std),
            },
            domain="shared-encoder",
        )


@dataclass(frozen=True)
class TaskDescription:
    name: str
    classes: tuple[str, ...]
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    shared_feature_dim: int

    @property
    def stable_sort_key(self) -> str:
        # The prefix is fixed and therefore lexicographic order remains name order.
        return f"task:{self.name}"

    def fingerprint(self) -> str:
        return fingerprint(
            {
                "name": self.name,
                "classes": list(self.classes),
                "train_ids": list(self.train_ids),
                "val_ids": list(self.val_ids),
                "shared_feature_dim": self.shared_feature_dim,
            },
            domain="task-description",
        )


@dataclass(frozen=True)
class EffectiveAdapterConfig:
    adapter_type: str
    rank: int
    learning_rate: float
    epochs: int
    batch_size: int
    validation_threshold: float
    convergence_tolerance: float
    lora_scale: float

    def training_material(self) -> dict[str, Any]:
        """Only fields that can alter trained parameters (not metric threshold)."""

        return {
            "adapter_type": self.adapter_type,
            "rank": self.rank,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "convergence_tolerance": self.convergence_tolerance,
            "lora_scale": self.lora_scale,
        }

    def public(self) -> dict[str, Any]:
        return {
            "adapter_type": self.adapter_type,
            "rank": self.rank,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_threshold": self.validation_threshold,
            "convergence_tolerance": self.convergence_tolerance,
            "lora_scale": self.lora_scale,
        }


_ADAPTER_KEYS = {
    "adapter_type",
    "rank",
    "learning_rate",
    "lr",
    "epochs",
    "batch_size",
    "validation_threshold",
    "convergence_tolerance",
    "lora_scale",
}

_ADAPTER_ALIASES = {
    "type": "adapter_type",
    "adapter_rank": "rank",
    "validation_metric_threshold": "validation_threshold",
    "val_threshold": "validation_threshold",
    "num_epochs": "epochs",
    "max_epochs": "epochs",
    "batch": "batch_size",
    "convergence_eps": "convergence_tolerance",
}


def _normalize_adapter_keys(mapping: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        canonical = _ADAPTER_ALIASES.get(str(key), str(key))
        normalized[canonical] = copy.deepcopy(value)
    return normalized


@dataclass(frozen=True)
class AdapterConfigDocument:
    defaults: Mapping[str, Any]
    task_overrides: Mapping[str, Mapping[str, Any]]
    concurrency: int
    snapshot_commit_interval: int
    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "AdapterConfigDocument":
        document = _mapping(value, "adapter config", PipelineConfigError)
        normalized = _normalize_adapter_keys(document)
        concurrency = normalized.pop("max_concurrency", normalized.get("concurrency", 1))
        commit_interval = normalized.pop(
            "snapshot_interval",
            normalized.get("snapshot_commit_interval", 1),
        )
        raw = copy.deepcopy(normalized)
        raw["concurrency"] = concurrency
        raw["snapshot_commit_interval"] = commit_interval
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
            raise PipelineConfigError("concurrency must be a positive integer")
        if (
            isinstance(commit_interval, bool)
            or not isinstance(commit_interval, int)
            or commit_interval <= 0
        ):
            raise PipelineConfigError("snapshot_commit_interval must be a positive integer")
        overrides_raw = raw.get("task_overrides", {})
        if not isinstance(overrides_raw, Mapping):
            raise PipelineConfigError("task_overrides must be an object")
        overrides: dict[str, Mapping[str, Any]] = {}
        for task_name, changes in overrides_raw.items():
            if not isinstance(task_name, str) or not task_name:
                raise PipelineConfigError("task_overrides keys must be non-empty strings")
            if not isinstance(changes, Mapping):
                raise PipelineConfigError(f"task_overrides[{task_name!r}] must be an object")
            unknown = set(changes) - _ADAPTER_KEYS
            if unknown:
                raise PipelineConfigError(
                    f"task_overrides[{task_name!r}] has unknown keys: {sorted(unknown)}"
                )
            overrides[task_name] = _normalize_adapter_keys(changes)
        defaults = {
            key: copy.deepcopy(default)
            for key, default in raw.items()
            if key in _ADAPTER_KEYS
        }
        unknown = set(raw) - _ADAPTER_KEYS - {
            "task_overrides",
            "concurrency",
            "snapshot_commit_interval",
        }
        if unknown:
            raise PipelineConfigError(f"adapter config has unknown keys: {sorted(unknown)}")
        return cls(
            defaults=defaults,
            task_overrides=overrides,
            concurrency=concurrency,
            snapshot_commit_interval=commit_interval,
            raw=raw,
        )

    def with_overrides(self, extra: Mapping[str, Mapping[str, Any]]) -> "AdapterConfigDocument":
        merged_raw = copy.deepcopy(dict(self.raw))
        merged_overrides = copy.deepcopy(dict(merged_raw.get("task_overrides", {})))
        for task_name, changes in extra.items():
            existing = merged_overrides.setdefault(task_name, {})
            existing.update(copy.deepcopy(dict(changes)))
        merged_raw["task_overrides"] = merged_overrides
        return AdapterConfigDocument.from_mapping(merged_raw)

    def resolve(self, task_name: str) -> EffectiveAdapterConfig:
        merged = copy.deepcopy(dict(self.defaults))
        merged.update(copy.deepcopy(dict(self.task_overrides.get(task_name, {}))))
        unknown = set(merged) - _ADAPTER_KEYS
        if unknown:
            raise AdapterConfigError(f"unknown adapter keys: {sorted(unknown)}")
        if "learning_rate" not in merged and "lr" in merged:
            merged["learning_rate"] = merged.pop("lr")
        elif "learning_rate" in merged:
            merged.pop("lr", None)
        adapter_type_raw = merged.get("adapter_type")
        if not isinstance(adapter_type_raw, str):
            raise AdapterConfigError("adapter_type must be a string")
        normalized_type = adapter_type_raw.strip().lower().replace("-", "_")
        aliases = {
            "linear": "linear_probe",
            "linearprobe": "linear_probe",
            "probe": "linear_probe",
            "linear_probe": "linear_probe",
            "lora": "low_rank",
            "lowrank": "low_rank",
            "low_rank": "low_rank",
        }
        adapter_type = aliases.get(normalized_type)
        if adapter_type is None:
            raise AdapterConfigError(
                "adapter_type must be 'linear_probe' or 'low_rank'"
            )
        rank = merged.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise AdapterConfigError("rank must be a positive integer")
        learning_rate = ensure_finite_number(merged.get("learning_rate"), "learning_rate")
        if learning_rate <= 0:
            raise AdapterConfigError("learning_rate must be greater than zero")
        epochs = merged.get("epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise AdapterConfigError("epochs must be a positive integer")
        batch_size = merged.get("batch_size")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise AdapterConfigError("batch_size must be a positive integer")
        threshold = ensure_finite_number(
            merged.get("validation_threshold"), "validation_threshold"
        )
        if threshold < 0.0 or threshold > 1.0:
            raise AdapterConfigError("validation_threshold must be in [0, 1]")
        tolerance = ensure_finite_number(
            merged.get("convergence_tolerance", 1e-10), "convergence_tolerance"
        )
        if tolerance < 0.0:
            raise AdapterConfigError("convergence_tolerance must be non-negative")
        lora_scale = ensure_finite_number(merged.get("lora_scale", 0.1), "lora_scale")
        if lora_scale <= 0.0:
            raise AdapterConfigError("lora_scale must be greater than zero")
        return EffectiveAdapterConfig(
            adapter_type=adapter_type,
            rank=rank,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            validation_threshold=threshold,
            convergence_tolerance=tolerance,
            lora_scale=lora_scale,
        )


def parse_feature_library(value: Any) -> FeatureLibrary:
    document = _mapping(value, "feature library", FeatureLibraryError)
    samples_raw = document.get("samples")
    if isinstance(samples_raw, Mapping):
        sample_items = list(samples_raw.items())
    elif isinstance(samples_raw, list):
        sample_items = []
        for index, item in enumerate(samples_raw):
            sample_doc = _mapping(item, f"feature library.samples[{index}]", FeatureLibraryError)
            sample_id = sample_doc.get("sample_id", sample_doc.get("id"))
            if not isinstance(sample_id, str) or not sample_id:
                raise FeatureLibraryError(
                    f"feature library.samples[{index}] needs a non-empty sample_id"
                )
            sample_items.append((sample_id, sample_doc))
    else:
        raise FeatureLibraryError("feature library.samples must be an object or list")
    parsed: dict[str, FeatureSample] = {}
    inferred_dim: int | None = None
    for sample_id, sample_raw in sample_items:
        if not isinstance(sample_id, str) or not sample_id:
            raise FeatureLibraryError("sample IDs must be non-empty strings")
        if sample_id in parsed:
            raise FeatureLibraryError(f"duplicate sample ID {sample_id!r}")
        sample_doc = _mapping(sample_raw, f"sample {sample_id!r}", FeatureLibraryError)
        features_raw = sample_doc.get("features", sample_doc.get("vector"))
        if not isinstance(features_raw, list) or not features_raw:
            raise FeatureLibraryError(f"sample {sample_id!r} features must be a non-empty list")
        features: list[float] = []
        for index, raw_value in enumerate(features_raw):
            try:
                features.append(ensure_finite_number(raw_value, f"sample {sample_id} feature {index}"))
            except ValueError as exc:
                raise FeatureLibraryError(str(exc)) from exc
        label_value = sample_doc.get("label", sample_doc.get("class"))
        label = _canonical_label_value(
            label_value, f"sample {sample_id!r} label", FeatureLibraryError
        )
        if inferred_dim is None:
            inferred_dim = len(features)
        elif inferred_dim != len(features):
            raise FeatureLibraryError("all samples must have the same feature dimension")
        parsed[sample_id] = FeatureSample(
            sample_id=sample_id,
            features=tuple(features),
            label=label,
        )
    feature_dim = document.get(
        "feature_dim", document.get("shared_feature_dim", inferred_dim)
    )
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
        raise FeatureLibraryError("feature_dim must be a positive integer")
    if inferred_dim != feature_dim:
        raise FeatureLibraryError("feature_dim does not match sample vectors")
    seed = document.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise FeatureLibraryError("seed must be an integer when present")
    return FeatureLibrary(samples=parsed, feature_dim=feature_dim, seed=seed)


def _matrix(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EncoderError(f"{label} must be a numeric matrix") from exc
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise EncoderError(f"{label} must be a non-empty 2-D matrix")
    if not np.isfinite(array).all():
        raise EncoderError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _vector(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EncoderError(f"{label} must be a numeric vector") from exc
    if array.ndim != 1 or array.shape[0] <= 0:
        raise EncoderError(f"{label} must be a non-empty 1-D vector")
    if not np.isfinite(array).all():
        raise EncoderError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def parse_encoder(value: Any) -> EncoderPackage:
    document = _mapping(value, "encoder package", EncoderError)
    weights_raw = document.get(
        "weights",
        document.get("weight", document.get("weight_matrix", document.get("W"))),
    )
    bias_raw = document.get("bias")
    if weights_raw is None:
        raise EncoderError("encoder package.weights is required")
    if bias_raw is None:
        raise EncoderError("encoder package.bias is required")
    weights = _matrix(weights_raw, "weights")
    declared_input = document.get("input_dim")
    declared_output = document.get("output_dim")
    if (
        isinstance(declared_input, int)
        and not isinstance(declared_input, bool)
        and isinstance(declared_output, int)
        and not isinstance(declared_output, bool)
        and weights.shape == (declared_input, declared_output)
        and weights.shape != (declared_output, declared_input)
    ):
        # Accept the common [input_dim, output_dim] convention when dimensions
        # are explicitly declared; internal math always uses [out, in].
        weights = np.ascontiguousarray(weights.T, dtype=np.float64)
    bias = _vector(bias_raw, "bias")
    input_dim = weights.shape[1]
    output_dim = weights.shape[0]
    if bias.shape[0] != output_dim:
        raise EncoderError("bias length must equal the encoder output dimension")
    normalization_raw = document.get(
        "normalization", document.get("norm", document.get("normalization_stats"))
    )
    if normalization_raw is None:
        normalization = {
            "mean": document.get("mean", document.get("norm_mean")),
            "std": document.get("std", document.get("norm_std")),
        }
    else:
        normalization = _mapping(normalization_raw, "normalization", EncoderError)
    mean_raw = normalization.get("mean", normalization.get("norm_mean"))
    std_raw = normalization.get("std", normalization.get("norm_std"))
    if mean_raw is None or std_raw is None:
        raise EncoderError("normalization.mean and normalization.std are required")
    mean = _vector(mean_raw, "normalization.mean")
    std = _vector(std_raw, "normalization.std")
    if mean.shape[0] != input_dim or std.shape[0] != input_dim:
        raise EncoderError("normalization statistics must match the encoder input dimension")
    if np.any(std <= 0.0):
        raise EncoderError("normalization.std values must be greater than zero")
    declared_input = document.get("input_dim", input_dim)
    declared_output = document.get("output_dim", output_dim)
    if declared_input != input_dim or declared_output != output_dim:
        raise EncoderError("declared encoder dimensions do not match parameter shapes")
    return EncoderPackage(weights=weights, bias=bias, mean=mean, std=std)


def failure_record(
    task_name: str | None,
    stage: str,
    code: str,
    exception_type: str,
    message: str,
    injection_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "stage": stage,
        "code": code,
        "business_code": code,
        "exception_type": exception_type,
        "message": message,
        "injection_ids": sorted(set(injection_ids)),
    }


def failure_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("task_name") or "",
        str(record.get("stage", "")),
        str(record.get("code", "")),
        str(record.get("exception_type", "")),
        tuple(record.get("injection_ids", [])),
        str(record.get("message", "")),
    )


def _task_raw_list(tasks_document: Any) -> tuple[list[Any], Any]:
    if isinstance(tasks_document, list):
        return list(tasks_document), None
    document = _mapping(tasks_document, "task description document", TaskDescriptionError)
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise TaskDescriptionError("task description document.tasks must be a list")
    return list(tasks), document.get(
        "shared_feature_dim", document.get("feature_dim")
    )


def validate_tasks(
    tasks_document: Any,
    features: FeatureLibrary,
    encoder: EncoderPackage,
    injection_ids_by_task: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[TaskDescription], list[dict[str, Any]]]:
    """Validate task descriptions independently and retain all valid tasks."""

    injections = injection_ids_by_task or {}
    failures: list[dict[str, Any]] = []
    try:
        raw_tasks, top_level_dim = _task_raw_list(tasks_document)
    except TaskDescriptionError as exc:
        return [], [
            failure_record(None, "task_description", exc.code, type(exc).__name__, str(exc))
        ]
    names: list[str] = []
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, Mapping):
            failures.append(
                failure_record(
                    None,
                    "task_description",
                    TaskDescriptionError.code,
                    TaskDescriptionError.__name__,
                    f"tasks[{index}] must be an object",
                )
            )
            continue
        name = raw_task.get("name", raw_task.get("task_name"))
        if isinstance(name, str) and name:
            names.append(name)
    duplicate_names = {name for name in names if names.count(name) > 1}
    for name in sorted(duplicate_names):
        failures.append(
            failure_record(
                name,
                "task_description",
                TaskDescriptionError.code,
                TaskDescriptionError.__name__,
                "task name must be unique",
                injections.get(name, ()),
            )
        )

    accepted: list[TaskDescription] = []
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, Mapping):
            continue
        name_value = raw_task.get("name", raw_task.get("task_name"))
        task_name = name_value if isinstance(name_value, str) and name_value else None
        if task_name in duplicate_names:
            continue
        injection_ids = injections.get(task_name or "", ())
        try:
            if not isinstance(name_value, str) or not name_value:
                raise TaskDescriptionError("task.name must be a non-empty string")
            classes_raw = raw_task.get(
                "classes", raw_task.get("class_set", raw_task.get("categories"))
            )
            if not isinstance(classes_raw, list) or not classes_raw:
                raise TaskDescriptionError("task.classes must be a non-empty list")
            classes: list[str] = []
            for class_index, class_value in enumerate(classes_raw):
                classes.append(
                    _canonical_label_value(
                        class_value,
                        f"task.classes[{class_index}]",
                        TaskDescriptionError,
                    )
                )
            if len(set(classes)) != len(classes):
                raise TaskDescriptionError("task.classes must not contain duplicates")
            train_ids_raw = raw_task.get(
                "train_ids", raw_task.get("training_sample_ids")
            )
            val_ids_raw = raw_task.get(
                "val_ids", raw_task.get("validation_sample_ids")
            )
            if not isinstance(train_ids_raw, list) or not train_ids_raw:
                raise TaskDescriptionError("task.train_ids must be a non-empty list")
            if not isinstance(val_ids_raw, list) or not val_ids_raw:
                raise TaskDescriptionError("task.val_ids must be a non-empty list")
            train_ids = _string_list(train_ids_raw, "task.train_ids", TaskDescriptionError)
            val_ids = _string_list(val_ids_raw, "task.val_ids", TaskDescriptionError)
            if len(set(train_ids)) != len(train_ids):
                raise TaskDescriptionError("task.train_ids must not contain duplicates")
            if len(set(val_ids)) != len(val_ids):
                raise TaskDescriptionError("task.val_ids must not contain duplicates")
            shared_dim = raw_task.get(
                "shared_feature_dim", raw_task.get("feature_dim", top_level_dim)
            )
            if (
                isinstance(shared_dim, bool)
                or not isinstance(shared_dim, int)
                or shared_dim <= 0
            ):
                raise TaskDescriptionError("shared_feature_dim must be a positive integer")
            if shared_dim != features.feature_dim:
                raise TaskDescriptionError(
                    "shared_feature_dim does not match the feature library dimension"
                )
            if shared_dim != encoder.input_dim:
                raise TaskDescriptionError(
                    "shared_feature_dim does not match the encoder weight dimension"
                )
            for sample_id in train_ids + val_ids:
                if sample_id not in features.samples:
                    raise TaskDescriptionError(
                        f"sample ID {sample_id!r} does not exist in the feature library"
                    )
            class_set = set(classes)
            for sample_id in train_ids + val_ids:
                label = features.samples[sample_id].label
                if label not in class_set:
                    raise TaskDescriptionError(
                        f"sample {sample_id!r} label {label!r} is outside task classes"
                    )
            accepted.append(
                TaskDescription(
                    name=name_value,
                    classes=tuple(sorted(classes)),
                    train_ids=tuple(sorted(train_ids)),
                    val_ids=tuple(sorted(val_ids)),
                    shared_feature_dim=shared_dim,
                )
            )
        except (TaskDescriptionError, ValueError) as exc:
            code = exc.code if isinstance(exc, TaskDescriptionError) else TaskDescriptionError.code
            failures.append(
                failure_record(
                    task_name,
                    "task_description",
                    code,
                    type(exc).__name__,
                    str(exc),
                    injection_ids,
                )
            )
    accepted.sort(key=lambda task: task.name)
    failures.sort(key=failure_sort_key)
    return accepted, failures


def deep_copy_json(value: Any) -> Any:
    """Copy JSON-like input so interactive edits cannot mutate parser state."""

    return copy.deepcopy(value)


def normalized_run_material(
    tasks_document: Any,
    features: FeatureLibrary,
    encoder: EncoderPackage,
    adapter_config: AdapterConfigDocument,
    injections: Any,
) -> dict[str, Any]:
    """Build the complete run identity, independent of task list input order."""

    if isinstance(tasks_document, list):
        raw_tasks = list(tasks_document)
        shared_dim = None
    else:
        raw_tasks = list(tasks_document.get("tasks", []))
        shared_dim = tasks_document.get("shared_feature_dim")

    def public_task(item: Any) -> Any:
        if not isinstance(item, Mapping):
            return item
        return {key: value for key, value in item.items() if not str(key).startswith("_")}

    sorted_tasks = [
        public_task(item)
        for item in sorted(
            raw_tasks,
            key=lambda item: (
                item.get("name", item.get("task_name", ""))
                if isinstance(item, Mapping)
                else ""
            ),
        )
    ]
    return {
        "tasks": sorted_tasks,
        "shared_feature_dim": shared_dim,
        "features": {
            sample_id: {
                "features": list(features.samples[sample_id].features),
                "label": features.samples[sample_id].label,
            }
            for sample_id in features.sorted_ids()
        },
        "encoder": encoder.fingerprint(),
        "adapter_config": adapter_config.raw,
        "injections": injections,
    }


def finite_float(value: Any, label: str) -> float:
    result = ensure_finite_number(value, label)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result
