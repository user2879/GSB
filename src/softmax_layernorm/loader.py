from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BusinessError, BusinessFailure
from .jsonutil import sha256_bytes, sha256_json
from .models import CallSpec, CommandSpec, FORMAT_VERSION, LoadedRun, RunConfig, TensorSpec


def _validation_failure(code: str, message: str, details: dict[str, Any] | None = None) -> BusinessFailure:
    return BusinessFailure(
        code=code,
        message=message,
        stage="input",
        details=details or {},
    )


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {token}")


def read_json_file(path: Path, *, stage: str = "input") -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BusinessError(
            _validation_failure(
                "E_CONFIG_INVALID",
                f"cannot parse JSON file {path}: {exc}",
                {"path": str(path), "stage": stage},
            )
        ) from exc


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_config(raw: Any) -> RunConfig:
    problems: list[str] = []
    if not isinstance(raw, dict):
        problems.append("config root must be an object")
        raw = {}

    budget = raw.get("interaction_budget_ms", 50.0)
    if not _is_finite_number(budget) or float(budget) < 0.0:
        problems.append("interaction_budget_ms must be a finite non-negative number")

    workers = raw.get("thread_pool_workers", 4)
    if not _is_int(workers) or workers <= 0:
        problems.append("thread_pool_workers must be a positive integer")

    interval = raw.get("snapshot_interval_commands", 3)
    if not _is_int(interval) or interval <= 0:
        problems.append("snapshot_interval_commands must be a positive integer")

    scale = raw.get("layernorm_scale", 1.0)
    if not _is_finite_number(scale) or float(scale) <= 0.0:
        problems.append("layernorm_scale must be a finite positive number")

    resume = raw.get("resume", True)
    if not isinstance(resume, bool):
        problems.append("resume must be a boolean")

    for key, default in (
        ("snapshot_root", "output"),
        ("failure_dir", "output/failures"),
        ("report_dir", "output/reports"),
    ):
        value = raw.get(key, default)
        if not isinstance(value, str) or not value:
            problems.append(f"{key} must be a non-empty string")

    if problems:
        raise BusinessError(
            _validation_failure(
                "E_CONFIG_INVALID",
                "run configuration is invalid",
                {"problems": problems},
            )
        )

    return RunConfig(
        interaction_budget_ms=float(budget),
        thread_pool_workers=int(workers),
        snapshot_interval_commands=int(interval),
        layernorm_scale=float(scale),
        snapshot_root=str(raw.get("snapshot_root", "output")),
        failure_dir=str(raw.get("failure_dir", "output/failures")),
        report_dir=str(raw.get("report_dir", "output/reports")),
        resume=bool(resume),
    )


def coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric tensor value")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"nan", "+nan", "-nan"}:
            return math.nan
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return math.inf
        if lowered in {"-inf", "-infinity"}:
            return -math.inf
        return float(value)
    raise ValueError(f"not a numeric tensor value: {value!r}")


def _parse_shape(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    first, second = value
    if not _is_int(first) or not _is_int(second) or first <= 0 or second <= 0:
        return None
    return int(first), int(second)


def parse_tensor_spec(raw: Any) -> TensorSpec:
    if not isinstance(raw, dict):
        raise BusinessError(
            _validation_failure("E_CONFIG_INVALID", "tensor description root must be an object")
        )
    if isinstance(raw.get("tensor"), dict):
        raw = raw["tensor"]

    problems: list[str] = []
    shape = _parse_shape(raw.get("shape"))
    if shape is None:
        problems.append("shape must contain two positive integers")

    seed = raw.get("seed")
    if not _is_int(seed):
        problems.append("seed must be an integer")

    distribution_raw = raw.get("distribution", "normal")
    distribution_params: dict[str, Any] = {}
    if isinstance(distribution_raw, str):
        distribution = distribution_raw
        distribution_params = dict(raw.get("distribution_params", {}))
    elif isinstance(distribution_raw, dict):
        distribution = distribution_raw.get("kind", distribution_raw.get("type"))
        distribution_params = {
            key: value for key, value in distribution_raw.items() if key not in {"kind", "type"}
        }
    else:
        distribution = None

    if distribution not in {"normal", "uniform", "constant"}:
        problems.append("distribution must be normal, uniform, or constant")

    if distribution == "normal":
        loc = distribution_params.get("loc", 0.0)
        scale = distribution_params.get("scale", 1.0)
        if not _is_finite_number(loc):
            problems.append("normal distribution loc must be finite")
        if not _is_finite_number(scale) or float(scale) <= 0.0:
            problems.append("normal distribution scale must be finite and positive")
    elif distribution == "uniform":
        low = distribution_params.get("low", -1.0)
        high = distribution_params.get("high", 1.0)
        if not _is_finite_number(low) or not _is_finite_number(high) or float(low) >= float(high):
            problems.append("uniform distribution requires finite low < high")
    elif distribution == "constant":
        value = distribution_params.get("value", 0.0)
        try:
            coerce_float(value)
        except (TypeError, ValueError):
            problems.append("constant distribution value must be numeric")

    special_values_raw = raw.get("special_values", [])
    special_values: list[dict[str, Any]] = []
    if not isinstance(special_values_raw, list):
        problems.append("special_values must be a list")
    elif shape is not None:
        for index, item in enumerate(special_values_raw):
            if not isinstance(item, dict):
                problems.append(f"special_values[{index}] must be an object")
                continue
            row = item.get("row")
            col = item.get("col", item.get("column"))
            if not _is_int(row) or not _is_int(col):
                problems.append(f"special_values[{index}] row/col must be integers")
                continue
            if not (0 <= row < shape[0] and 0 <= col < shape[1]):
                raise BusinessError(
                    _validation_failure(
                        "E_SHAPE_MISMATCH",
                        f"special value position ({row}, {col}) is outside tensor shape {shape}",
                        {"index": index, "shape": list(shape)},
                    )
                )
            try:
                numeric_value = coerce_float(item.get("value"))
            except (TypeError, ValueError) as exc:
                problems.append(f"special_values[{index}].value is invalid: {exc}")
                continue
            special_values.append({"row": int(row), "col": int(col), "value": numeric_value})

    declared_dtype = raw.get("dtype", "float32")
    if declared_dtype not in {"float32", "f32", None}:
        problems.append("dtype must be float32")
    if problems:
        raise BusinessError(
            _validation_failure(
                "E_CONFIG_INVALID",
                "tensor description is invalid",
                {"problems": problems},
            )
        )

    return TensorSpec(
        shape=shape,
        seed=int(seed),
        distribution=str(distribution),
        distribution_params=distribution_params,
        special_values=tuple(special_values),
    )


def generate_tensor(spec: TensorSpec) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(spec.seed))
    if spec.distribution == "normal":
        loc = np.float32(spec.distribution_params.get("loc", 0.0))
        scale = np.float32(spec.distribution_params.get("scale", 1.0))
        tensor = (
            rng.standard_normal(spec.shape, dtype=np.float32) * scale + loc
        ).astype(np.float32, copy=False)
    elif spec.distribution == "uniform":
        low = np.float32(spec.distribution_params.get("low", -1.0))
        high = np.float32(spec.distribution_params.get("high", 1.0))
        tensor = rng.uniform(low, high, size=spec.shape).astype(np.float32, copy=False)
    else:
        value = np.float32(coerce_float(spec.distribution_params.get("value", 0.0)))
        tensor = np.full(spec.shape, value, dtype=np.float32)

    for item in spec.special_values:
        tensor[int(item["row"]), int(item["col"])] = np.float32(item["value"])
    return np.ascontiguousarray(tensor, dtype=np.float32)


def tensor_fingerprint(tensor: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(tensor, dtype="<f4")
    header = struct.pack("<2I", contiguous.shape[0], contiguous.shape[1])
    digest = hashlib.sha256()
    digest.update(b"float32-le-2d")
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _normalize_axis(value: Any) -> int | None:
    if not _is_int(value) or value < -2 or value > 1:
        return None
    return int(value) % 2


def parse_calls(raw: Any, tensor_shape: tuple[int, int], config: RunConfig) -> list[CallSpec]:
    if isinstance(raw, dict) and "calls" in raw:
        raw_calls = raw["calls"]
    else:
        raw_calls = raw
    if not isinstance(raw_calls, list) or not raw_calls:
        raise BusinessError(
            _validation_failure("E_CONFIG_INVALID", "calls must be a non-empty list")
        )

    failures: list[BusinessFailure] = []
    calls: list[CallSpec] = []
    seen_ids: set[str] = set()
    seen_sequences: set[int] = set()
    for index, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            failures.append(
                _validation_failure("E_CONFIG_INVALID", f"calls[{index}] must be an object")
            )
            continue

        call_id = item.get("call_id", item.get("id", f"call-{index + 1}"))
        if not isinstance(call_id, str) or not call_id:
            failures.append(
                _validation_failure("E_CONFIG_INVALID", f"calls[{index}] call_id must be a string")
            )
            call_id = f"call-{index + 1}"
        if call_id in seen_ids:
            failures.append(
                _validation_failure("E_CONFIG_INVALID", f"duplicate call_id: {call_id}")
            )
        seen_ids.add(call_id)

        sequence = item.get("sequence", index + 1)
        if not _is_int(sequence) or sequence <= 0:
            failures.append(
                _validation_failure(
                    "E_CONFIG_INVALID", f"calls[{index}] sequence must be a positive integer"
                )
            )
            sequence = index + 1
        if sequence in seen_sequences:
            failures.append(
                _validation_failure("E_CONFIG_INVALID", f"duplicate call sequence: {sequence}")
            )
        seen_sequences.add(int(sequence))

        rows = item.get("rows")
        cols = item.get("cols", item.get("columns"))
        if not _is_int(rows) or not _is_int(cols) or rows <= 0 or cols <= 0:
            failures.append(
                _validation_failure(
                    "E_SHAPE_MISMATCH",
                    f"calls[{index}] rows/cols must be positive integers",
                    {"call_id": call_id},
                )
            )
            continue
        if (int(rows), int(cols)) != tensor_shape:
            failures.append(
                BusinessFailure(
                    code="E_SHAPE_MISMATCH",
                    message=(
                        f"call {call_id} shape {(int(rows), int(cols))} does not match "
                        f"tensor shape {tensor_shape}"
                    ),
                    stage="input",
                    call_id=call_id,
                    call_sequence=int(sequence),
                    details={"call_shape": [int(rows), int(cols)], "tensor_shape": list(tensor_shape)},
                )
            )
            continue

        softmax_axis = _normalize_axis(item.get("softmax_axis"))
        if softmax_axis is None:
            failures.append(
                BusinessFailure(
                    code="E_AXIS_OUT_OF_RANGE",
                    message=f"call {call_id} softmax_axis must be in [-2, 1] for a 2-D tensor",
                    stage="input",
                    call_id=call_id,
                    call_sequence=int(sequence),
                    details={"softmax_axis": item.get("softmax_axis")},
                )
            )
            softmax_axis = 1

        layernorm_axis = _normalize_axis(item.get("layernorm_axis"))
        if layernorm_axis is None:
            failures.append(
                BusinessFailure(
                    code="E_AXIS_OUT_OF_RANGE",
                    message=f"call {call_id} layernorm_axis must be in [-2, 1] for a 2-D tensor",
                    stage="input",
                    call_id=call_id,
                    call_sequence=int(sequence),
                    details={"layernorm_axis": item.get("layernorm_axis")},
                )
            )
            layernorm_axis = 1

        epsilon = item.get("epsilon")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
        ):
            failures.append(
                BusinessFailure(
                    code="E_EPS_INVALID",
                    message=f"call {call_id} epsilon must be finite and positive",
                    stage="input",
                    call_id=call_id,
                    call_sequence=int(sequence),
                    details={"epsilon": epsilon},
                )
            )
            epsilon = 1.0

        stabilize = item.get("stabilize", item.get("stabilize_softmax", True))
        if not isinstance(stabilize, bool):
            failures.append(
                _validation_failure(
                    "E_CONFIG_INVALID", f"call {call_id} stabilize must be a boolean"
                )
            )
            stabilize = True

        scale = item.get("layernorm_scale", config.layernorm_scale)
        if not _is_finite_number(scale) or float(scale) <= 0.0:
            failures.append(
                _validation_failure(
                    "E_CONFIG_INVALID", f"call {call_id} layernorm_scale must be finite and positive"
                )
            )
            scale = config.layernorm_scale

        reference_precision = item.get("reference_precision")
        precision_aliases = {
            "float32": "float32",
            "f32": "float32",
            "standard": "float32",
            "float64": "float64",
            "f64": "float64",
            "high": "float64",
            "high_precision": "float64",
        }
        if reference_precision is not None and reference_precision not in precision_aliases:
            failures.append(
                _validation_failure(
                    "E_CONFIG_INVALID",
                    f"call {call_id} reference_precision must be null, float32, or float64",
                )
            )
            reference_precision = None
        elif reference_precision is not None:
            reference_precision = precision_aliases[reference_precision]

        calls.append(
            CallSpec(
                call_id=call_id,
                sequence=int(sequence),
                rows=int(rows),
                cols=int(cols),
                softmax_axis=int(softmax_axis),
                layernorm_axis=int(layernorm_axis),
                epsilon=float(epsilon),
                stabilize=bool(stabilize),
                layernorm_scale=float(scale),
                reference_precision=reference_precision,
            )
        )

    if failures:
        raise BusinessError(failures)
    calls.sort(key=lambda call: call.sequence)
    return calls


def build_config_hash(config: RunConfig, calls: list[CallSpec], tensor_spec: TensorSpec) -> str:
    return sha256_json(
        {
            "format_version": FORMAT_VERSION,
            "config": config.to_dict(),
            "calls": [call.to_dict() for call in calls],
            "tensor_spec": tensor_spec.to_dict(),
        }
    )


def load_run(calls_path: str | Path, tensor_path: str | Path | None, config_path: str | Path) -> LoadedRun:
    calls_path = Path(calls_path)
    config_path = Path(config_path)
    calls_document = read_json_file(calls_path, stage="calls")
    config = validate_config(read_json_file(config_path, stage="config"))

    if tensor_path is None:
        if not isinstance(calls_document, dict) or "tensor" not in calls_document:
            raise BusinessError(
                _validation_failure(
                    "E_CONFIG_INVALID", "tensor description is absent; pass --tensor or embed tensor in calls JSON"
                )
            )
        tensor_document = calls_document["tensor"]
        tensor_source_path = calls_path
    else:
        tensor_path = Path(tensor_path)
        tensor_document = read_json_file(tensor_path, stage="tensor")
        tensor_source_path = tensor_path

    tensor_spec = parse_tensor_spec(tensor_document)
    calls = parse_calls(calls_document, tensor_spec.shape, config)
    tensor = generate_tensor(tensor_spec)
    fingerprint = tensor_fingerprint(tensor)
    config_hash = build_config_hash(config, calls, tensor_spec)
    return LoadedRun(
        config=config,
        tensor_spec=tensor_spec,
        tensor=tensor,
        calls=calls,
        config_hash=config_hash,
        source_input_fingerprint=fingerprint,
        current_input_fingerprint=fingerprint,
        call_description_hash=sha256_file_or_json(calls_path, calls_document),
        tensor_description_hash=sha256_file_or_json(tensor_source_path, tensor_document),
    )


def sha256_file_or_json(path: Path, fallback: Any) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return sha256_json(fallback)


def load_commands_file(path: str | Path | None) -> tuple[list[dict[str, Any]], list[CommandSpec], str]:
    if path is None:
        return [], [], sha256_json([])
    document = read_json_file(Path(path), stage="commands")
    raw_commands = document.get("commands") if isinstance(document, dict) else document
    if raw_commands is None:
        raw_commands = []
    if not isinstance(raw_commands, list):
        raise BusinessError(
            _validation_failure("E_CONFIG_INVALID", "commands must be a list")
        )

    commands: list[CommandSpec] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_commands):
        if not isinstance(raw, dict):
            raise BusinessError(
                _validation_failure(
                    "E_CONFIG_INVALID", f"commands[{index}] must be an object"
                )
            )
        command_id = raw.get("command_id", raw.get("id", f"cmd-{index + 1:04d}"))
        if not isinstance(command_id, str) or not command_id:
            raise BusinessError(
                _validation_failure(
                    "E_CONFIG_INVALID", f"commands[{index}].command_id must be a non-empty string"
                )
            )
        if command_id in seen_ids:
            raise BusinessError(
                _validation_failure("E_CONFIG_INVALID", f"duplicate command_id: {command_id}")
            )
        seen_ids.add(command_id)
        commands.append(CommandSpec(index=index, command_id=command_id, raw=dict(raw)))
    return [dict(raw) for raw in raw_commands], commands, sha256_json(raw_commands)
