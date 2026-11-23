"""Configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import (
    E_BANDWIDTH,
    E_CELL_BUDGET,
    E_CONFIG,
    E_MAX_PATH_LENGTH,
    E_PATH_POLICY,
    E_QUANT_LEVEL,
    E_SNAPSHOT_INTERVAL,
    E_STEP_MODE,
    E_WORKERS,
    Failure,
)
from .fingerprints import fingerprint


STEP_MODES = ("symmetric", "asymmetric")
PATH_LENGTH_POLICIES = ("reject", "truncate")


@dataclass(frozen=True)
class Config:
    reference_sequence_id: str | None
    step_mode: str
    bandwidth_permille: int
    distance_coefficient: int
    fraction_bits: int
    quantization_step: int
    missing_marker: Any
    missing_penalty: int
    max_path_length: int
    path_length_policy: str
    thread_pool_workers: int
    cell_recompute_budget: int
    snapshot_interval_batches: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def alignment_dict(self) -> dict[str, Any]:
        return {
            "step_mode": self.step_mode,
            "distance_coefficient": self.distance_coefficient,
            "fraction_bits": self.fraction_bits,
            "quantization_step": self.quantization_step,
            "missing_marker": self.missing_marker,
            "missing_penalty": self.missing_penalty,
        }

    def alignment_fingerprint(self) -> str:
        return fingerprint(self.alignment_dict())

    def quant_fingerprint(self) -> str:
        return fingerprint(
            {
                "fraction_bits": self.fraction_bits,
                "quantization_step": self.quantization_step,
                "missing_marker": self.missing_marker,
            }
        )

    def replace_step_mode(self, step_mode: str) -> "Config":
        return Config(**{**self.to_dict(), "step_mode": step_mode})

    def replace_bandwidth(self, bandwidth_permille: int) -> "Config":
        return Config(**{**self.to_dict(), "bandwidth_permille": bandwidth_permille})


@dataclass
class ConfigLoadResult:
    config: Config | None
    failures: list[Failure]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_field(
    raw: dict[str, Any],
    key: str,
    default: Any,
    code: str,
    stage: str,
    failures: list[Failure],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    value = raw.get(key, default)
    if not _is_int(value):
        failures.append(
            Failure(code, stage, f"{key} must be an integer", detail={"value": value})
        )
        return None
    if minimum is not None and value < minimum:
        failures.append(
            Failure(
                code,
                stage,
                f"{key} must be >= {minimum}",
                detail={"value": value, "minimum": minimum},
            )
        )
        return None
    if maximum is not None and value > maximum:
        failures.append(
            Failure(
                code,
                stage,
                f"{key} must be <= {maximum}",
                detail={"value": value, "maximum": maximum},
            )
        )
        return None
    return value


def parse_config(raw: Any, *, stage: str = "config") -> ConfigLoadResult:
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        return ConfigLoadResult(
            None,
            [Failure(E_CONFIG, stage, "configuration root must be a mapping")],
        )

    step_mode = raw.get("step_mode")
    if step_mode not in STEP_MODES:
        failures.append(
            Failure(
                E_STEP_MODE,
                stage,
                f"unknown step_mode {step_mode!r}",
                detail={"allowed": list(STEP_MODES), "value": step_mode},
            )
        )

    bandwidth = _int_field(
        raw,
        "bandwidth_permille",
        None,
        E_BANDWIDTH,
        stage,
        failures,
        minimum=0,
        maximum=1000,
    )
    coefficient = _int_field(
        raw,
        "distance_coefficient",
        1,
        E_CONFIG,
        stage,
        failures,
        minimum=1,
    )

    fixed = raw.get("fixed_point", {})
    if not isinstance(fixed, dict):
        failures.append(
            Failure(E_CONFIG, stage, "fixed_point must be a mapping")
        )
        fixed = {}

    fraction_bits = _int_field(
        fixed,
        "fraction_bits",
        fixed.get("bits", raw.get("fraction_bits")),
        E_QUANT_LEVEL,
        stage,
        failures,
        minimum=1,
        maximum=30,
    )
    quantization_step = _int_field(
        fixed,
        "quantization_step",
        fixed.get("step", raw.get("quantization_step")),
        E_QUANT_LEVEL,
        stage,
        failures,
        minimum=1,
    )
    if (
        fraction_bits is not None
        and quantization_step is not None
        and quantization_step > (1 << fraction_bits)
    ):
        failures.append(
            Failure(
                E_QUANT_LEVEL,
                stage,
                "quantization_step must not exceed 2**fraction_bits",
                detail={
                    "quantization_step": quantization_step,
                    "fraction_bits": fraction_bits,
                },
            )
        )

    missing_penalty = _int_field(
        raw,
        "missing_penalty",
        0,
        E_CONFIG,
        stage,
        failures,
        minimum=0,
    )
    max_path_length = _int_field(
        raw,
        "max_path_length",
        None,
        E_MAX_PATH_LENGTH,
        stage,
        failures,
        minimum=1,
    )

    path_length_policy = raw.get("path_length_policy", "reject")
    if path_length_policy not in PATH_LENGTH_POLICIES:
        failures.append(
            Failure(
                E_PATH_POLICY,
                stage,
                f"unknown path_length_policy {path_length_policy!r}",
                detail={"allowed": list(PATH_LENGTH_POLICIES)},
            )
        )

    workers = _int_field(
        raw,
        "thread_pool_workers",
        1,
        E_WORKERS,
        stage,
        failures,
        minimum=1,
    )
    budget = _int_field(
        raw,
        "cell_recompute_budget",
        0,
        E_CELL_BUDGET,
        stage,
        failures,
        minimum=0,
    )
    interval = _int_field(
        raw,
        "snapshot_interval_batches",
        1,
        E_SNAPSHOT_INTERVAL,
        stage,
        failures,
        minimum=1,
    )

    missing_marker = raw.get("missing_marker", None)
    if missing_marker is not None and (
        isinstance(missing_marker, bool)
        or not isinstance(missing_marker, (str, int, float))
        or (
            isinstance(missing_marker, float)
            and not math.isfinite(missing_marker)
        )
    ):
        failures.append(
            Failure(
                E_CONFIG,
                stage,
                "missing_marker must be null, a string, or a finite number",
                detail={"value": repr(missing_marker)},
            )
        )

    reference_sequence_id = raw.get(
        "reference_sequence_id", raw.get("reference_id")
    )
    if reference_sequence_id is not None and (
        not isinstance(reference_sequence_id, str) or not reference_sequence_id
    ):
        failures.append(
            Failure(
                E_CONFIG,
                stage,
                "reference_sequence_id must be a non-empty string or null",
                detail={"value": repr(reference_sequence_id)},
            )
        )

    if failures:
        return ConfigLoadResult(None, failures)

    config = Config(
        reference_sequence_id=reference_sequence_id,
        step_mode=step_mode,
        bandwidth_permille=bandwidth,
        distance_coefficient=coefficient,
        fraction_bits=fraction_bits,
        quantization_step=quantization_step,
        missing_marker=missing_marker,
        missing_penalty=missing_penalty,
        max_path_length=max_path_length,
        path_length_policy=path_length_policy,
        thread_pool_workers=workers,
        cell_recompute_budget=budget,
        snapshot_interval_batches=interval,
    )
    return ConfigLoadResult(config, [])


def load_config(path: str | Path) -> ConfigLoadResult:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except Exception as exc:  # pragma: no cover - exact parser messages vary
        return ConfigLoadResult(
            None,
            [Failure(E_LOAD, "config", f"failed to load YAML: {exc}")],
        )
    return parse_config(raw)
