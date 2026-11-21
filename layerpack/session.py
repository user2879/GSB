"""Mutable interactive session state and the eight supported edit commands."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import ALLOWED_BITS, ALLOWED_SCHEMES
from .fingerprint import canonical_json, sha256_json, to_builtin
from .loader import (
    load_calibration_set,
    load_weight_package,
    make_layer,
    write_calibration_set,
    write_weight_package,
)
from .model import CalibrationSample, CalibrationSet, Layer


COMMAND_TYPES = {
    "set_bits",
    "set_layer_bits",
    "set_scheme",
    "switch_scheme",
    "set_quantization_scheme",
    "switch_quantization_scheme",
    "set_layer_scheme",
    "set_group_size",
    "set_group_granularity",
    "set_quant_group_size",
    "set_per_channel",
    "set_per_channel_scaling",
    "toggle_per_channel",
    "set_temperature",
    "set_distillation_temperature",
    "set_soft_label_weight",
    "replace_layer_weights",
    "replace_weights",
    "replace_calibration_samples",
    "replace_calibration_sample",
}


class CommandError(ValueError):
    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.detail = detail


def _ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CommandError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise CommandError(f"{name} must be >= {minimum}")
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CommandError(f"{name} must be finite")
    return number


def _numeric_vector(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise CommandError(f"{name} must be a non-empty numeric vector")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CommandError(f"{name} must be a non-empty numeric vector")
        number = float(item)
        if not math.isfinite(number):
            raise CommandError(f"{name} must contain finite values")
        result.append(number)
    return tuple(result)


def _parse_calibration_records(records: Any) -> CalibrationSet:
    if not isinstance(records, list) or not records:
        raise CommandError("calibration samples must be a non-empty list")
    samples: list[CalibrationSample] = []
    identifiers: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise CommandError(f"calibration sample {index} must be an object")
        sample_id = record.get("id", record.get("sample_id"))
        if not isinstance(sample_id, str) or not sample_id:
            raise CommandError(f"calibration sample {index} has no id")
        if sample_id in identifiers:
            raise CommandError(f"duplicate calibration sample id: {sample_id}")
        inputs = _numeric_vector(record.get("input", record.get("inputs")), "input")
        reference = _numeric_vector(
            record.get("reference_output", record.get("reference", record.get("output"))),
            "reference_output",
        )
        layers_value = record.get("layers", [])
        if layers_value in (None, "all"):
            layers_value = [] if layers_value is None else ["all"]
        if not isinstance(layers_value, list) or any(
            not isinstance(item, str) or not item for item in layers_value
        ):
            raise CommandError(f"calibration sample {sample_id} layers must be strings")
        samples.append(
            CalibrationSample(
                sample_id=sample_id,
                inputs=inputs,
                reference_output=reference,
                layers=tuple(layers_value),
                source_index=index,
            )
        )
        identifiers.add(sample_id)
    samples.sort(key=lambda sample: sample.sample_id)
    fingerprint = sha256_json(
        {
            "kind": "calibration-set-v1",
            "samples": [sample.to_input_dict() for sample in samples],
        }
    )
    return CalibrationSet(samples=samples, fingerprint=fingerprint)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class SessionState:
    config: dict[str, Any]
    layers: dict[str, Layer]
    layer_order: list[str]
    calibration: CalibrationSet
    generation: int = 0

    @classmethod
    def from_paths(
        cls,
        weights_path: str | Path,
        calibration_path: str | Path,
        config_path: str | Path,
    ) -> "SessionState":
        layers, issues = load_weight_package(weights_path)
        if issues:
            first = issues[0]
            raise CommandError(
                f"cannot initialize session from weights: {first.message}",
                detail=first.to_dict(),
            )
        calibration = load_calibration_set(calibration_path)
        if calibration.issues:
            first = calibration.issues[0]
            raise CommandError(
                f"cannot initialize session from calibration: {first.message}",
                detail=first.to_dict(),
            )
        try:
            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise CommandError(f"cannot initialize session from config: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise CommandError("configuration root must be a mapping")
        return cls(
            config=raw,
            layers=layers,
            layer_order=list(layers),
            calibration=calibration,
        )

    def clone(self) -> "SessionState":
        return copy.deepcopy(self)

    def content_hash(self) -> str:
        return sha256_json(
            {
                "config": to_builtin(self.config),
                "layers": [
                    {
                        "name": self.layers[name].name,
                        "content_fingerprint": self.layers[name].content_fingerprint,
                        "content_version": self.layers[name].content_version,
                    }
                    for name in self.layer_order
                ],
                "calibration_fingerprint": self.calibration.fingerprint,
            }
        )

    def export_to(self, directory: str | Path) -> dict[str, str]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / "config.yaml"
        weights_path = root / "weights.jsonl"
        calibration_path = root / "calibration.json"
        _atomic_text(
            config_path,
            yaml.safe_dump(
                to_builtin(self.config),
                sort_keys=False,
                allow_unicode=True,
            ),
        )
        weighted = sorted(
            (self.layers[name] for name in self.layer_order),
            key=lambda layer: (layer.content_version, layer.name),
        )
        temporary_weights = root / f".weights.{os.getpid()}.tmp"
        try:
            write_weight_package(temporary_weights, weighted)
            os.replace(temporary_weights, weights_path)
        finally:
            if temporary_weights.exists():
                temporary_weights.unlink()
        temporary_calibration = root / f".calibration.{os.getpid()}.tmp"
        try:
            write_calibration_set(temporary_calibration, self.calibration)
            os.replace(temporary_calibration, calibration_path)
        finally:
            if temporary_calibration.exists():
                temporary_calibration.unlink()
        return {
            "weights": str(weights_path),
            "calibration": str(calibration_path),
            "config": str(config_path),
        }

    def _layer_override(self, layer: str) -> dict[str, Any]:
        if layer not in self.layers:
            raise CommandError(f"unknown layer: {layer}")
        root = _ensure_mapping(self.config, "layers")
        value = root.get(layer)
        if isinstance(value, int) and not isinstance(value, bool):
            value = {"bits": value}
        if not isinstance(value, dict):
            value = {}
            root[layer] = value
        return value

    def expected_affected_layers(self, command: dict[str, Any]) -> list[str]:
        all_layers = sorted(self.layers)
        command_type = str(command.get("type", ""))
        layer = command.get("layer")
        if command_type in {
            "replace_calibration_samples",
            "replace_calibration_sample",
        }:
            records: Any
            if command_type == "replace_calibration_sample":
                records = [command.get("sample", {})]
            else:
                records = command.get("samples", [])
            affected: set[str] = set()
            if isinstance(records, list):
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    targets = record.get("layers", [])
                    if not targets or "all" in targets:
                        affected.update(all_layers)
                    elif isinstance(targets, list):
                        affected.update(
                            name for name in targets if name in self.layers
                        )
            return sorted(affected) if affected else all_layers
        if command_type in {
            "set_temperature",
            "set_distillation_temperature",
            "set_soft_label_weight",
        }:
            return all_layers
        if isinstance(layer, str) and layer:
            return [layer]
        return all_layers

    def apply_command(self, command: dict[str, Any]) -> list[str]:
        if not isinstance(command, dict):
            raise CommandError("command must be an object")
        command_type = str(command.get("type", ""))
        if command_type not in COMMAND_TYPES:
            raise CommandError(f"unsupported command type: {command_type}")
        affected = self.expected_affected_layers(command)
        quantization = _ensure_mapping(self.config, "quantization")

        if command_type in {"set_bits", "set_layer_bits"}:
            bits = _require_int(command.get("bits", command.get("value")), "bits")
            if bits not in ALLOWED_BITS:
                raise CommandError(f"bits must be one of {list(ALLOWED_BITS)}")
            layer = command.get("layer")
            if layer is None:
                quantization["default_bits"] = bits
                for name in self.layers:
                    override = self._layer_override(name)
                    override.pop("bits", None)
            else:
                if not isinstance(layer, str) or layer not in self.layers:
                    raise CommandError(f"unknown layer: {layer}")
                self._layer_override(layer)["bits"] = bits

        elif command_type in {
            "set_scheme",
            "switch_scheme",
            "set_quantization_scheme",
            "switch_quantization_scheme",
            "set_layer_scheme",
        }:
            scheme = str(command.get("scheme", command.get("value", ""))).lower()
            if scheme not in ALLOWED_SCHEMES:
                raise CommandError(
                    f"scheme must be one of {list(ALLOWED_SCHEMES)}"
                )
            layer = command.get("layer")
            if layer is None:
                quantization["scheme"] = scheme
                for name in self.layers:
                    self._layer_override(name).pop("scheme", None)
            else:
                if not isinstance(layer, str) or layer not in self.layers:
                    raise CommandError(f"unknown layer: {layer}")
                self._layer_override(layer)["scheme"] = scheme

        elif command_type in {
            "set_group_size",
            "set_group_granularity",
            "set_quant_group_size",
        }:
            group_size = _require_int(
                command.get(
                    "group_size",
                    command.get("granularity", command.get("value")),
                ),
                "group_size",
                minimum=1,
            )
            layer = command.get("layer")
            if layer is None:
                quantization["group_size"] = group_size
                for name in self.layers:
                    self._layer_override(name).pop("group_size", None)
            else:
                if not isinstance(layer, str) or layer not in self.layers:
                    raise CommandError(f"unknown layer: {layer}")
                self._layer_override(layer)["group_size"] = group_size

        elif command_type in {
            "set_per_channel",
            "set_per_channel_scaling",
            "toggle_per_channel",
        }:
            enabled = command.get(
                "enabled",
                command.get(
                    "value",
                    command.get(
                        "per_channel",
                        command.get("per_channel_scaling"),
                    ),
                ),
            )
            if enabled is None and command_type == "toggle_per_channel":
                # Toggling without an explicit value is relative to the current
                # global setting and is therefore still deterministic.
                enabled = not bool(quantization.get("per_channel_scaling", False))
            if not isinstance(enabled, bool):
                raise CommandError("enabled must be boolean")
            layer = command.get("layer")
            if layer is None:
                quantization["per_channel_scaling"] = enabled
                for name in self.layers:
                    self._layer_override(name).pop("per_channel", None)
            else:
                if not isinstance(layer, str) or layer not in self.layers:
                    raise CommandError(f"unknown layer: {layer}")
                self._layer_override(layer)["per_channel"] = enabled

        elif command_type in {
            "set_temperature",
            "set_distillation_temperature",
        }:
            temperature = _require_number(
                command.get("value", command.get("temperature")),
                "temperature",
            )
            if temperature <= 0:
                raise CommandError("temperature must be positive")
            _ensure_mapping(self.config, "distillation")[
                "temperature"
            ] = temperature

        elif command_type == "set_soft_label_weight":
            weight = _require_number(
                command.get("value", command.get("weight")), "soft_label_weight"
            )
            if not 0.0 <= weight <= 1.0:
                raise CommandError("soft_label_weight must be in [0, 1]")
            _ensure_mapping(self.config, "distillation")[
                "soft_label_weight"
            ] = weight

        elif command_type in {"replace_layer_weights", "replace_weights"}:
            layer_name = command.get("layer")
            if not isinstance(layer_name, str) or layer_name not in self.layers:
                raise CommandError(f"unknown layer: {layer_name}")
            shape = command.get(
                "shape",
                command.get("new_shape", list(self.layers[layer_name].shape)),
            )
            if not isinstance(shape, list):
                raise CommandError("shape must be a list")
            weights = command.get(
                "weights", command.get("weight", command.get("new_weights"))
            )
            content_version = command.get("content_version", command.get("version"))
            if content_version is None:
                content_version = max(
                    layer.content_version for layer in self.layers.values()
                ) + 1
            else:
                content_version = _require_int(
                    content_version, "content_version", minimum=0
                )
            try:
                replacement = make_layer(
                    layer_name,
                    shape,
                    weights,
                    content_version,
                    source_index=self.layers[layer_name].source_index,
                )
            except (TypeError, ValueError) as exc:
                raise CommandError(str(exc)) from exc
            self.layers[layer_name] = replacement

        else:  # calibration replacement
            if command_type == "replace_calibration_sample":
                sample = command.get("sample")
                if not isinstance(sample, dict):
                    raise CommandError("sample must be an object")
                records = [item.to_input_dict() for item in self.calibration.samples]
                sample_id = sample.get("id", sample.get("sample_id"))
                replaced = False
                for index, record in enumerate(records):
                    if record["id"] == sample_id:
                        records[index] = sample
                        replaced = True
                        break
                if not replaced:
                    records.append(sample)
                self.calibration = _parse_calibration_records(records)
            else:
                calibration_block = command.get("calibration")
                records = command.get(
                    "samples",
                    command.get(
                        "calibration_samples",
                        calibration_block.get("samples")
                        if isinstance(calibration_block, dict)
                        else None,
                    ),
                )
                self.calibration = _parse_calibration_records(records)

        self.generation += 1
        return affected


class SessionStore:
    """Content-addressed saved session state; the pointer is atomically replaced."""

    def __init__(self, state_root: str | Path) -> None:
        self.root = Path(state_root) / "session"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state: SessionState) -> str:
        content_hash = state.content_hash()
        target = self.root / f"state-{content_hash[:20]}"
        if not target.is_dir():
            temporary = Path(tempfile.mkdtemp(prefix=".state-", dir=self.root))
            try:
                state.export_to(temporary)
                if target.exists():
                    shutil.rmtree(temporary)
                else:
                    os.replace(temporary, target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
        pointer_payload = {"state": target.name, "content_hash": content_hash}
        _atomic_text(
            self.root / "current.json",
            canonical_json(pointer_payload) + "\n",
        )
        return target.as_posix()

    def load(self) -> SessionState | None:
        pointer = self.root / "current.json"
        candidates: list[Path] = []
        if pointer.is_file():
            try:
                data = json.loads(pointer.read_text(encoding="utf-8"))
                candidate = self.root / str(data.get("state", ""))
                if candidate.is_dir():
                    candidates.append(candidate)
            except (OSError, json.JSONDecodeError):
                pass
        candidates.extend(
            sorted(
                [path for path in self.root.glob("state-*") if path.is_dir()],
                reverse=True,
            )
        )
        for candidate in candidates:
            try:
                state = SessionState.from_paths(
                    candidate / "weights.jsonl",
                    candidate / "calibration.json",
                    candidate / "config.yaml",
                )
                return state
            except (CommandError, OSError):
                continue
        return None

    def load_or_create(
        self,
        weights_path: str | Path,
        calibration_path: str | Path,
        config_path: str | Path,
        *,
        reset: bool = False,
    ) -> SessionState:
        if not reset:
            existing = self.load()
            if existing is not None:
                return existing
        state = SessionState.from_paths(weights_path, calibration_path, config_path)
        self.save(state)
        return state
