"""Input loading, validation, deterministic fingerprints, and edit operations."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .canonical import (
    atomic_write_text,
    canonical_json,
    pretty_json,
    sha256_json,
    stable_task_sort_key,
    to_jsonable,
)
from .errors import ConfigError, EncoderValidationError, TaskValidationError


REQUIRED_INPUT_FILES = (
    "tasks.json",
    "features.jsonl",
    "encoder.json",
    "config.yaml",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(value)
    return records


def _as_float_list(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field} contains a non-finite number")
        result.append(number)
    return result


def _as_matrix(value: Any, *, field: str) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty matrix")
    rows = [_as_float_list(row, field=f"{field}[{index}]") for index, row in enumerate(value)]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{field} must be rectangular and non-empty")
    return np.asarray(rows, dtype=np.float64)


def _get_first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml must contain an object")

    adapter = raw.get("adapter") if isinstance(raw.get("adapter"), dict) else {}
    training = raw.get("training") if isinstance(raw.get("training"), dict) else {}
    validation = raw.get("validation") if isinstance(raw.get("validation"), dict) else {}
    execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}

    adapter_type = _get_first(
        raw,
        "adapter_type",
        default=_get_first(adapter, "type", default="linear_probe"),
    )
    adapter_aliases = {
        "linear": "linear_probe",
        "linear_probe": "linear_probe",
        "probe": "linear_probe",
        "low_rank": "low_rank",
        "low-rank": "low_rank",
        "lora": "low_rank",
    }
    normalized_type = adapter_aliases.get(str(adapter_type).strip().lower())
    if normalized_type is None:
        raise ConfigError(f"unsupported adapter type: {adapter_type!r}")

    rank = _get_first(raw, "rank", default=adapter.get("rank", 2))
    scale = _get_first(raw, "scale", default=adapter.get("scale", adapter.get("scaling", 1.0)))
    learning_rate = _get_first(
        raw,
        "learning_rate",
        "lr",
        default=training.get("learning_rate", 0.05),
    )
    epochs = _get_first(raw, "epochs", default=training.get("epochs", 10))
    batch_size = _get_first(raw, "batch_size", default=training.get("batch_size", 4))
    metric = _get_first(
        raw,
        "validation_metric",
        "metric",
        default=validation.get("metric", "macro_f1"),
    )
    threshold = _get_first(
        raw,
        "validation_threshold",
        "validation_metric_threshold",
        "threshold",
        default=validation.get(
            "threshold", validation.get("metric_threshold", 0.5)
        ),
    )
    concurrency = _get_first(
        raw,
        "max_concurrency",
        "concurrency",
        default=execution.get("max_concurrency", execution.get("concurrency", 1)),
    )
    snapshot_interval = _get_first(
        raw,
        "snapshot_interval",
        "snapshot_commit_interval",
        default=execution.get(
            "snapshot_interval", execution.get("snapshot_commit_interval", 1)
        ),
    )
    activation = _get_first(
        raw,
        "activation",
        default=execution.get("activation", "tanh"),
    )
    seed_offset = _get_first(raw, "seed_offset", default=raw.get("random_seed_offset", 0))

    metric_aliases = {
        "f1": "macro_f1",
        "macro_f1": "macro_f1",
        "macro-f1": "macro_f1",
        "accuracy": "accuracy",
        "acc": "accuracy",
    }
    normalized_metric = metric_aliases.get(str(metric).strip().lower())
    if normalized_metric is None:
        raise ConfigError(f"unsupported validation metric: {metric!r}")
    activation_aliases = {
        "identity": "identity",
        "linear": "identity",
        "relu": "relu",
        "tanh": "tanh",
    }
    normalized_activation = activation_aliases.get(str(activation).strip().lower())
    if normalized_activation is None:
        raise ConfigError(f"unsupported activation: {activation!r}")

    try:
        rank_value = int(rank)
        epochs_value = int(epochs)
        batch_value = int(batch_size)
        concurrency_value = int(concurrency)
        interval_value = int(snapshot_interval)
        seed_offset_value = int(seed_offset)
        scale_value = float(scale)
        learning_rate_value = float(learning_rate)
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config contains a non-numeric value: {exc}") from exc

    if rank_value < 0:
        raise ConfigError("rank must be >= 0; zero is executed as E_RANK_INVALID")
    if not math.isfinite(scale_value):
        raise ConfigError("scale must be finite")
    if not math.isfinite(learning_rate_value) or learning_rate_value <= 0:
        raise ConfigError("learning_rate must be a positive finite number")
    if epochs_value < 1:
        raise ConfigError("epochs must be >= 1")
    if batch_value < 1:
        raise ConfigError("batch_size must be >= 1")
    if concurrency_value < 1:
        raise ConfigError("max_concurrency must be >= 1")
    if interval_value < 1:
        raise ConfigError("snapshot_interval must be >= 1")
    if not math.isfinite(threshold_value):
        raise ConfigError("validation threshold must be finite")

    normalized = {
        "activation": normalized_activation,
        "adapter_type": normalized_type,
        "rank": rank_value,
        "scale": scale_value,
        "learning_rate": learning_rate_value,
        "epochs": epochs_value,
        "batch_size": batch_value,
        "validation_metric": normalized_metric,
        "validation_threshold": threshold_value,
        "max_concurrency": concurrency_value,
        "snapshot_interval": interval_value,
        "seed_offset": seed_offset_value,
    }

    # Preserve additional declarative configuration in the hash. Behavioural
    # values are normalized through aliases above.
    extras = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "adapter",
            "training",
            "validation",
            "execution",
            "adapter_type",
            "rank",
            "scale",
            "learning_rate",
            "lr",
            "epochs",
            "batch_size",
            "validation_metric",
            "metric",
            "validation_threshold",
            "validation_metric_threshold",
            "threshold",
            "max_concurrency",
            "concurrency",
            "snapshot_interval",
            "snapshot_commit_interval",
            "activation",
            "seed_offset",
            "random_seed_offset",
        }
    }
    if extras:
        normalized["extras"] = to_jsonable(extras)
    return normalized


def config_hash(config: dict[str, Any]) -> str:
    return sha256_json(config)


@dataclass(frozen=True)
class EncoderLayer:
    name: str
    version: str
    weights: np.ndarray
    bias: np.ndarray
    fingerprint: str

    @property
    def input_dim(self) -> int:
        return int(self.weights.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.weights.shape[1])


@dataclass(frozen=True)
class Encoder:
    shared_feature_dim: int
    mean: np.ndarray
    std: np.ndarray
    layers: tuple[EncoderLayer, ...]
    fingerprint: str
    normalization_fingerprint: str

    def layer_index(self, name: str) -> int | None:
        for index, layer in enumerate(self.layers):
            if layer.name == name:
                return index
        return None

    def layer(self, name: str) -> EncoderLayer:
        index = self.layer_index(name)
        if index is None:
            raise KeyError(name)
        return self.layers[index]

    def layer_manifests(self) -> list[dict[str, Any]]:
        return [
            {
                "name": layer.name,
                "version": layer.version,
                "fingerprint": layer.fingerprint,
                "input_dim": layer.input_dim,
                "output_dim": layer.output_dim,
            }
            for layer in self.layers
        ]


@dataclass(frozen=True)
class TaskSpec:
    index: int
    name: str
    classes: tuple[str, ...]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    target_layer: str
    target_layer_whitelist: tuple[str, ...]
    adapter_rank_override: int | None
    injection_id: str | None
    sort_key: str
    raw: dict[str, Any]

    @property
    def effective_classes(self) -> tuple[str, ...]:
        return tuple(sorted(self.classes))


@dataclass
class TaskLoadResult:
    tasks: list[TaskSpec]
    failures: list[dict[str, Any]]
    injection_hits: list[dict[str, Any]]
    encoder_failure: dict[str, Any] | None = None


def parse_encoder(raw: dict[str, Any], expected_shared_dim: int | None = None) -> Encoder:
    if not isinstance(raw, dict):
        raise EncoderValidationError("encoder.json must contain an object")
    try:
        shared_dim = int(raw["shared_feature_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EncoderValidationError("encoder shared_feature_dim is missing or invalid") from exc
    if expected_shared_dim is not None and shared_dim != expected_shared_dim:
        raise EncoderValidationError(
            f"encoder shared_feature_dim {shared_dim} != tasks shared_feature_dim {expected_shared_dim}"
        )
    if shared_dim < 1:
        raise EncoderValidationError("encoder shared_feature_dim must be >= 1")

    normalization = raw.get("normalization")
    if not isinstance(normalization, dict):
        raise EncoderValidationError("encoder normalization object is required")
    try:
        mean = np.asarray(
            _as_float_list(normalization.get("mean"), field="normalization.mean"),
            dtype=np.float64,
        )
        std = np.asarray(
            _as_float_list(normalization.get("std"), field="normalization.std"),
            dtype=np.float64,
        )
    except ValueError as exc:
        raise EncoderValidationError(str(exc)) from exc
    if mean.shape != (shared_dim,) or std.shape != (shared_dim,):
        raise EncoderValidationError("normalization mean/std dimensions must equal shared_feature_dim")
    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise EncoderValidationError("normalization std values must be positive and finite")

    raw_layers = raw.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise EncoderValidationError("encoder layers must be a non-empty list")
    layers: list[EncoderLayer] = []
    names: set[str] = set()
    next_input_dim = shared_dim
    for index, raw_layer in enumerate(raw_layers):
        if not isinstance(raw_layer, dict):
            raise EncoderValidationError(f"encoder layer[{index}] must be an object")
        name = raw_layer.get("name", raw_layer.get("layer_name"))
        if not isinstance(name, str) or not name:
            raise EncoderValidationError(f"encoder layer[{index}] has no valid name")
        if name in names:
            raise EncoderValidationError(f"duplicate encoder layer name: {name}")
        names.add(name)
        version_value = raw_layer.get("version", raw_layer.get("layer_version"))
        if version_value is None:
            raise EncoderValidationError(f"encoder layer {name} has no version")
        version = str(version_value)
        try:
            weights = _as_matrix(raw_layer.get("weights", raw_layer.get("weight")), field=f"{name}.weights")
            bias = np.asarray(
                _as_float_list(raw_layer.get("bias"), field=f"{name}.bias"),
                dtype=np.float64,
            )
        except ValueError as exc:
            raise EncoderValidationError(str(exc)) from exc
        if weights.shape[0] != next_input_dim:
            raise EncoderValidationError(
                f"layer {name} input dimension {weights.shape[0]} != expected {next_input_dim}"
            )
        if bias.shape != (weights.shape[1],):
            raise EncoderValidationError(
                f"layer {name} bias shape {bias.shape} != ({weights.shape[1]},)"
            )
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise EncoderValidationError(f"layer {name} contains non-finite parameters")
        layer_fingerprint = sha256_json(
            {
                "name": name,
                "version": version,
                "weights": weights.tolist(),
                "bias": bias.tolist(),
            }
        )
        weights = np.array(weights, dtype=np.float64, copy=True)
        bias = np.array(bias, dtype=np.float64, copy=True)
        weights.flags.writeable = False
        bias.flags.writeable = False
        layers.append(
            EncoderLayer(
                name=name,
                version=version,
                weights=weights,
                bias=bias,
                fingerprint=layer_fingerprint,
            )
        )
        next_input_dim = int(weights.shape[1])

    mean.flags.writeable = False
    std.flags.writeable = False
    normalization_fingerprint = sha256_json(
        {"mean": mean.tolist(), "std": std.tolist()}
    )
    fingerprint = sha256_json(
        {
            "shared_feature_dim": shared_dim,
            "normalization": normalization_fingerprint,
            "layers": [
                {
                    "name": layer.name,
                    "version": layer.version,
                    "fingerprint": layer.fingerprint,
                }
                for layer in layers
            ],
        }
    )
    return Encoder(
        shared_feature_dim=shared_dim,
        mean=mean,
        std=std,
        layers=tuple(layers),
        fingerprint=fingerprint,
        normalization_fingerprint=normalization_fingerprint,
    )


def _failure(
    *,
    task_name: str,
    target_layer: str,
    phase: str,
    code: str,
    message: str,
    exception_type: str | None = None,
    injection_id: str | None = None,
    task_index: int | None = None,
    reason_code: str | None = None,
) -> dict[str, Any]:
    return {
        "sort_key": stable_task_sort_key(task_name, target_layer),
        "task_name": task_name,
        "task_index": task_index,
        "target_layer": target_layer,
        "phase": phase,
        "business_code": code,
        "reason_code": reason_code,
        "message": message,
        "exception_type": exception_type,
        "injection_id": injection_id,
    }


def load_tasks(
    raw: dict[str, Any],
    features: dict[str, dict[str, Any]],
    encoder: Encoder | None,
    *,
    encoder_error: str | None = None,
) -> TaskLoadResult:
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), list):
        failure = _failure(
            task_name="__tasks__",
            target_layer="",
            phase="load",
            code="E_TASK_INVALID",
            message="tasks.json must contain a tasks list",
        )
        return TaskLoadResult(tasks=[], failures=[failure], injection_hits=[])

    root_whitelist_raw = raw.get("target_layer_whitelist", [])
    root_whitelist = tuple(str(item) for item in root_whitelist_raw) if isinstance(root_whitelist_raw, list) else ()
    expected_dim_raw = raw.get("shared_feature_dim")
    try:
        expected_dim = int(expected_dim_raw) if expected_dim_raw is not None else None
    except (TypeError, ValueError):
        expected_dim = -1
    if expected_dim is None and encoder is not None:
        expected_dim = encoder.shared_feature_dim

    name_counts: dict[str, int] = {}
    for raw_task in raw["tasks"]:
        if isinstance(raw_task, dict):
            name = str(raw_task.get("name", raw_task.get("task_name", "")))
            name_counts[name] = name_counts.get(name, 0) + 1

    valid_tasks: list[TaskSpec] = []
    failures: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for index, raw_task in enumerate(raw["tasks"]):
        name = ""
        target_layer = ""
        injection_id: str | None = None
        try:
            if not isinstance(raw_task, dict):
                raise TaskValidationError(f"task[{index}] is not an object")
            name = str(raw_task.get("name", raw_task.get("task_name", ""))).strip()
            injection_value = raw_task.get("_injection_id")
            injection_id = str(injection_value) if injection_value is not None else None
            target_layer = str(raw_task.get("target_layer", ""))
            if not name:
                raise TaskValidationError("task name is empty")
            if name_counts[name] > 1:
                raise TaskValidationError(f"duplicate task name: {name}")
            classes_raw = raw_task.get("classes", raw_task.get("class_set"))
            if not isinstance(classes_raw, list) or not classes_raw:
                raise TaskValidationError("class set must be a non-empty list")
            if any(not isinstance(item, str) or not item for item in classes_raw):
                raise TaskValidationError("class set entries must be non-empty strings")
            classes = tuple(classes_raw)
            if len(set(classes)) != len(classes):
                raise TaskValidationError("class set contains duplicate classes")
            train_raw = raw_task.get("train_ids", raw_task.get("training_ids"))
            validation_raw = raw_task.get("validation_ids", raw_task.get("val_ids"))
            if not isinstance(train_raw, list) or not train_raw:
                raise TaskValidationError("training sample ID list must be non-empty")
            if not isinstance(validation_raw, list):
                raise TaskValidationError("validation sample ID list must be a list")
            if not validation_raw:
                raise TaskValidationError("validation sample ID list is empty")
            if any(not isinstance(item, str) or not item for item in train_raw + validation_raw):
                raise TaskValidationError("sample IDs must be non-empty strings")
            train_ids = tuple(train_raw)
            validation_ids = tuple(validation_raw)
            if len(set(train_ids)) != len(train_ids):
                raise TaskValidationError("training sample ID list contains duplicates")
            if len(set(validation_ids)) != len(validation_ids):
                raise TaskValidationError("validation sample ID list contains duplicates")
            if set(train_ids) & set(validation_ids):
                raise TaskValidationError("training and validation sample IDs overlap")
            if not target_layer:
                raise TaskValidationError("target_layer is empty")
            whitelist_raw = raw_task.get("target_layer_whitelist", root_whitelist)
            if not isinstance(whitelist_raw, (list, tuple)) or not whitelist_raw:
                raise TaskValidationError("per-task target layer whitelist is empty")
            whitelist = tuple(str(item) for item in whitelist_raw)
            if target_layer not in whitelist:
                raise TaskValidationError(
                    f"target layer {target_layer!r} is not in the task whitelist"
                )
            rank_override_raw = raw_task.get("adapter_rank", raw_task.get("rank_override"))
            rank_override: int | None = None
            if rank_override_raw is not None:
                try:
                    rank_override = int(rank_override_raw)
                except (TypeError, ValueError) as exc:
                    raise TaskValidationError("adapter_rank override is not an integer") from exc
                if rank_override < 0:
                    raise TaskValidationError("adapter_rank override must be >= 0")
            if expected_dim is not None and expected_dim < 1:
                raise TaskValidationError("tasks shared_feature_dim is invalid")
            for sample_id in train_ids + validation_ids:
                if sample_id not in features:
                    raise TaskValidationError(f"sample ID {sample_id!r} is missing from features.jsonl")
                record = features[sample_id]
                vector = record.get("features")
                if not isinstance(vector, list) or len(vector) != expected_dim:
                    raise TaskValidationError(
                        f"sample {sample_id} feature dimension does not match shared_feature_dim"
                    )
                try:
                    parsed_vector = _as_float_list(vector, field=f"{sample_id}.features")
                except ValueError as exc:
                    raise TaskValidationError(str(exc)) from exc
                if len(parsed_vector) != len(vector):
                    raise TaskValidationError(f"sample {sample_id} has invalid features")
                label = record.get("label", record.get("class"))
                if label not in classes:
                    raise TaskValidationError(
                        f"sample {sample_id} label {label!r} is not in the task class set"
                    )
            if encoder_error is not None or encoder is None:
                raise TaskValidationError(encoder_error or "encoder is unavailable")
            if encoder.layer_index(target_layer) is None:
                raise TaskValidationError(f"target layer {target_layer!r} does not exist in encoder")
            for layer_name in whitelist:
                if encoder.layer_index(layer_name) is None:
                    raise TaskValidationError(
                        f"whitelist layer {layer_name!r} does not exist in encoder"
                    )

            sort_key = stable_task_sort_key(name, target_layer)
            spec = TaskSpec(
                index=index,
                name=name,
                classes=classes,
                train_ids=train_ids,
                validation_ids=validation_ids,
                target_layer=target_layer,
                target_layer_whitelist=whitelist,
                adapter_rank_override=rank_override,
                injection_id=injection_id,
                sort_key=sort_key,
                raw=copy.deepcopy(raw_task),
            )
            valid_tasks.append(spec)
            if injection_id:
                hits.append(
                    {
                        "injection_id": injection_id,
                        "task_name": name,
                        "kind": "load_failure",
                        "phase": "load",
                        "business_code": "E_TASK_INVALID",
                        "hit": False,
                        "detail": "task passed validation",
                    }
                )
        except TaskValidationError as exc:
            code = "E_TASK_INVALID"
            message = str(exc)
            reason_code = None
            if message == "validation sample ID list is empty":
                reason_code = "E_VALIDATION_EMPTY"
            failure = _failure(
                task_name=name or f"<task-index-{index}>",
                target_layer=target_layer,
                phase="load",
                code=code,
                message=message,
                injection_id=injection_id,
                task_index=index,
                reason_code=reason_code,
            )
            failures.append(failure)
            if injection_id:
                hits.append(
                    {
                        "injection_id": injection_id,
                        "task_name": name,
                        "kind": "load_failure",
                        "phase": "load",
                        "business_code": code,
                        "reason_code": reason_code,
                        "hit": True,
                        "detail": message,
                    }
                )

    failures.sort(
        key=lambda item: (
            item.get("sort_key", ""),
            item.get("task_name", ""),
            item.get("phase", ""),
            item.get("business_code", ""),
            item.get("message", ""),
        )
    )
    valid_tasks.sort(key=lambda task: (task.sort_key, task.name, task.target_layer, task.index))
    hits.sort(key=lambda item: (item["injection_id"], item["task_name"], item["phase"]))
    return TaskLoadResult(tasks=valid_tasks, failures=failures, injection_hits=hits)


def sample_content_fingerprint(
    features: dict[str, dict[str, Any]], sample_ids: list[str] | tuple[str, ...]
) -> str:
    payload = []
    for sample_id in sorted(sample_ids):
        record = features[sample_id]
        payload.append(
            {
                "sample_id": sample_id,
                "features": [float(value) for value in record["features"]],
                "label": record.get("label", record.get("class")),
            }
        )
    return sha256_json(payload)


def feature_library_fingerprint(features: dict[str, dict[str, Any]]) -> str:
    payload = []
    for sample_id in sorted(features):
        record = features[sample_id]
        payload.append(
            {
                "sample_id": sample_id,
                "features": [float(value) for value in record["features"]],
                "label": record.get("label", record.get("class")),
            }
        )
    return sha256_json(payload)


class InputState:
    """Mutable, deterministic view over one local sample directory."""

    def __init__(
        self,
        *,
        tasks: dict[str, Any],
        features: dict[str, dict[str, Any]],
        encoder: dict[str, Any],
        config: dict[str, Any],
        injections: dict[str, Any],
        edit_plan: dict[str, Any] | None = None,
    ) -> None:
        self.tasks = copy.deepcopy(tasks)
        self.features = copy.deepcopy(features)
        self.encoder = copy.deepcopy(encoder)
        self.config = copy.deepcopy(config)
        self.injections = copy.deepcopy(injections)
        self.edit_plan = copy.deepcopy(edit_plan) if edit_plan else {"batches": []}

    @classmethod
    def from_directory(cls, directory: str | Path, *, injections_path: str | Path | None = None) -> "InputState":
        directory = Path(directory)
        missing = [name for name in REQUIRED_INPUT_FILES if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing input files in {directory}: {', '.join(missing)}")
        tasks = _read_json(directory / "tasks.json")
        records = _read_jsonl(directory / "features.jsonl")
        features: dict[str, dict[str, Any]] = {}
        for record in records:
            sample_id = record.get("sample_id", record.get("id"))
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("every feature record needs a non-empty sample_id")
            if sample_id in features:
                raise ValueError(f"duplicate sample ID in features.jsonl: {sample_id}")
            features[sample_id] = copy.deepcopy(record)
        encoder = _read_json(directory / "encoder.json")
        with (directory / "config.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError("config.yaml must contain an object")
        injection_file = Path(injections_path) if injections_path else directory / "injections.json"
        injections: dict[str, Any] = {"injections": []}
        if injection_file.is_file():
            loaded = _read_json(injection_file)
            if isinstance(loaded, dict):
                injections = loaded
            elif isinstance(loaded, list):
                injections = {"injections": loaded}
        edit_file = directory / "edit_plan.json"
        edit_plan = _read_json(edit_file) if edit_file.is_file() else {"batches": []}
        return cls(
            tasks=tasks,
            features=features,
            encoder=encoder,
            config=config,
            injections=injections,
            edit_plan=edit_plan,
        )

    @property
    def edit_batches(self) -> list[dict[str, Any]]:
        batches = self.edit_plan.get("batches", []) if isinstance(self.edit_plan, dict) else []
        return batches if isinstance(batches, list) else []

    def canonical_state(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "features": [
                self.features[sample_id] for sample_id in sorted(self.features)
            ],
            "encoder": self.encoder,
            "config": self.config,
            "injections": self.injections,
        }

    @property
    def input_state_hash(self) -> str:
        return sha256_json(self.canonical_state())

    @property
    def normalized_config(self) -> dict[str, Any]:
        return normalize_config(self.config)

    @property
    def config_hash(self) -> str:
        return config_hash(self.normalized_config)

    def save_to(self, directory: str | Path, *, include_edit_plan: bool = True) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(directory / "tasks.json", pretty_json(self.tasks))
        feature_lines = "".join(
            canonical_json(self.features[sample_id]) + "\n"
            for sample_id in sorted(self.features)
        )
        atomic_write_text(directory / "features.jsonl", feature_lines)
        atomic_write_text(directory / "encoder.json", pretty_json(self.encoder))
        atomic_write_text(
            directory / "config.yaml",
            yaml.safe_dump(self.config, sort_keys=True, allow_unicode=True),
        )
        atomic_write_text(directory / "injections.json", pretty_json(self.injections))
        if include_edit_plan:
            atomic_write_text(directory / "edit_plan.json", pretty_json(self.edit_plan))

    def apply_batch(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        operations = batch.get("operations", []) if isinstance(batch, dict) else []
        if not isinstance(operations, list):
            raise ValueError("edit batch operations must be a list")
        applied: list[dict[str, Any]] = []
        for operation in operations:
            self.apply_operation(operation)
            applied.append(copy.deepcopy(operation))
        return applied

    def apply_operation(self, operation: dict[str, Any]) -> None:
        if not isinstance(operation, dict):
            raise ValueError("edit operation must be an object")
        op = str(operation.get("op", operation.get("type", ""))).strip().lower()
        if op in {"add_task", "add-task", "create_task"}:
            task = operation.get("task")
            if not isinstance(task, dict):
                raise ValueError("add_task requires a task object")
            name = task.get("name", task.get("task_name"))
            if not isinstance(name, str) or not name:
                raise ValueError("add_task requires a non-empty task name")
            existing = {
                item.get("name", item.get("task_name"))
                for item in self.tasks.get("tasks", [])
                if isinstance(item, dict)
            }
            if name in existing:
                raise ValueError(f"add_task cannot create duplicate task name {name!r}")
            self.tasks.setdefault("tasks", []).append(copy.deepcopy(task))
            return
        if op in {"delete_task", "delete-task", "remove_task"}:
            name = str(operation.get("name", operation.get("task_name", "")))
            if not name:
                raise ValueError("delete_task requires a task name")
            self.tasks["tasks"] = [
                task
                for task in self.tasks.get("tasks", [])
                if not isinstance(task, dict)
                or str(task.get("name", task.get("task_name", ""))) != name
            ]
            return
        if op in {"set_classes", "set-classes", "replace_classes"}:
            name = str(operation.get("name", operation.get("task_name", "")))
            classes = operation.get("classes")
            if not name or not isinstance(classes, list):
                raise ValueError("set_classes requires name and classes")
            found = False
            for task in self.tasks.get("tasks", []):
                if isinstance(task, dict) and str(task.get("name", task.get("task_name", ""))) == name:
                    task["classes"] = copy.deepcopy(classes)
                    found = True
            if not found:
                raise ValueError(f"set_classes task not found: {name}")
            return
        if op in {"replace_sample_features", "replace_sample", "replace_features"}:
            sample_id = str(operation.get("sample_id", operation.get("id", "")))
            if not sample_id:
                raise ValueError("replace_sample_features requires sample_id")
            existing = self.features.get(sample_id)
            features = operation.get("features", existing.get("features") if existing else None)
            label = operation.get("label", existing.get("label", existing.get("class")) if existing else None)
            if not isinstance(features, list) or label is None:
                raise ValueError("replace_sample_features requires features and label")
            record = copy.deepcopy(existing) if existing else {"sample_id": sample_id}
            record["sample_id"] = sample_id
            record["features"] = copy.deepcopy(features)
            record["label"] = label
            record.pop("class", None)
            self.features[sample_id] = record
            return
        if op in {"update_adapter_config", "update_config", "set_config"}:
            changes = operation.get("changes")
            if not isinstance(changes, dict) or not changes:
                raise ValueError("update_adapter_config requires a changes object")
            for key, value in changes.items():
                self._set_dotted(self.config, key, copy.deepcopy(value))
            return
        if op in {"replace_base_layer", "replace_layer", "replace_layer_weights"}:
            layer_name = str(operation.get("layer", operation.get("layer_name", "")))
            if not layer_name:
                raise ValueError("replace_base_layer requires layer")
            layers = self.encoder.get("layers", [])
            found = False
            for layer in layers:
                if not isinstance(layer, dict):
                    continue
                if str(layer.get("name", layer.get("layer_name", ""))) == layer_name:
                    if "weights" in operation:
                        layer["weights"] = copy.deepcopy(operation["weights"])
                    if "bias" in operation:
                        layer["bias"] = copy.deepcopy(operation["bias"])
                    if "version" in operation:
                        layer["version"] = str(operation["version"])
                    elif "layer_version" in operation:
                        layer["version"] = str(operation["layer_version"])
                    found = True
            if not found:
                raise ValueError(f"replace_base_layer layer not found: {layer_name}")
            return
        raise ValueError(f"unsupported edit operation: {op!r}")

    @staticmethod
    def _set_dotted(mapping: dict[str, Any], key: str, value: Any) -> None:
        parts = key.split(".")
        cursor = mapping
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value


def runtime_injections(injections: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    entries = injections.get("injections", []) if isinstance(injections, dict) else []
    if not isinstance(entries, list):
        return result
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "runtime_exception":
            continue
        task_name = str(entry.get("task", ""))
        phase = str(entry.get("phase", "training"))
        result[(task_name, phase)] = entry
    return result


def injection_ids(injections: dict[str, Any]) -> list[str]:
    entries = injections.get("injections", []) if isinstance(injections, dict) else []
    if not isinstance(entries, list):
        return []
    return sorted(
        str(entry.get("id"))
        for entry in entries
        if isinstance(entry, dict) and entry.get("id") is not None
    )
