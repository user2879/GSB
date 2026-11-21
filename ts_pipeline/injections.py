"""Independent fault-injection manifest loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import file_sha256, fingerprint, to_builtin
from .config import Config
from .errors import E_CONFIG_INVALID, Failure
from .samples import SequenceSample
from .yamlutil import load_yaml_file


INJECTION_KINDS = {
    "non_finite_observation",
    "irregular_timestamp",
    "window_too_long",
    "quantization_out_of_range",
}

INJECTION_CODES = {
    "non_finite_observation": "E_SAMPLE_INVALID",
    "irregular_timestamp": "E_SAMPLE_INVALID",
    "window_too_long": "E_CONFIG_INVALID",
    "quantization_out_of_range": "E_CONFIG_INVALID",
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    kind: str
    sequence_id: str
    window_index: int
    raw: dict[str, Any]

    @property
    def business_code(self) -> str:
        return INJECTION_CODES[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.injection_id,
            "injection_id": self.injection_id,
            "kind": self.kind,
            "sequence_id": self.sequence_id,
            "window_index": self.window_index,
            "business_code": self.business_code,
            "parameters": to_builtin(self.raw),
        }


@dataclass
class LoadedInjections:
    by_window: dict[tuple[str, int], tuple[Injection, ...]]
    items: tuple[Injection, ...]
    failures: list[Failure]
    manifest_hash: str
    fatal: bool

    def for_window(self, sequence_id: str, window_index: int) -> tuple[Injection, ...]:
        return self.by_window.get((sequence_id, window_index), ())


def window_count(length: int, window_length: int, step: int, forecast_horizon: int) -> int:
    required = window_length + forecast_horizon
    if length < required:
        return 0
    return (length - required) // step + 1


def _injection_failure(message: str, injection_id: str | None = None) -> Failure:
    return Failure(
        code=E_CONFIG_INVALID,
        stage="injection_validation",
        message=message,
        injection_id=injection_id,
    )


def load_injections(
    path: Path,
    config: Config,
    samples: dict[str, SequenceSample],
) -> LoadedInjections:
    failures: list[Failure] = []
    if not path.exists():
        failures.append(_injection_failure(f"injection manifest not found: {path}"))
        return LoadedInjections({}, (), failures, "", True)
    manifest_hash = file_sha256(path)
    try:
        raw = load_yaml_file(path)
    except Exception as exc:
        failures.append(_injection_failure(f"cannot parse injection manifest: {exc}"))
        return LoadedInjections({}, (), failures, manifest_hash, True)
    items_raw: Any
    if isinstance(raw, dict):
        items_raw = raw.get("injections", [])
    elif isinstance(raw, list):
        items_raw = raw
    else:
        failures.append(_injection_failure("injection manifest root must be a mapping or list"))
        return LoadedInjections({}, (), failures, manifest_hash, True)
    if not isinstance(items_raw, list):
        failures.append(_injection_failure("injections must be a list"))
        return LoadedInjections({}, (), failures, manifest_hash, True)

    valid: list[Injection] = []
    seen_ids: set[str] = set()
    count_cache: dict[str, int] = {
        sequence_id: window_count(
            sample.length,
            config.backtest.window_length,
            config.backtest.step,
            config.backtest.forecast_horizon,
        )
        for sequence_id, sample in samples.items()
    }
    for index, item in enumerate(items_raw):
        if not isinstance(item, dict):
            failures.append(_injection_failure(f"injection item #{index} must be a mapping"))
            continue
        injection_id = item.get("id")
        if not isinstance(injection_id, str) or not injection_id.strip():
            failures.append(_injection_failure(f"injection item #{index} has no non-empty id"))
            continue
        if injection_id in seen_ids:
            failures.append(_injection_failure("duplicate injection id", injection_id))
            continue
        seen_ids.add(injection_id)
        kind = item.get("kind")
        if kind not in INJECTION_KINDS:
            failures.append(_injection_failure(f"unknown injection kind: {kind}", injection_id))
            continue
        sequence_id = item.get("sequence_id")
        if not isinstance(sequence_id, str) or sequence_id not in samples:
            failures.append(_injection_failure(f"injection references unknown sequence: {sequence_id}", injection_id))
            continue
        window_index = item.get("window_index")
        if isinstance(window_index, bool) or not isinstance(window_index, int) or window_index < 0:
            failures.append(_injection_failure("window_index must be a non-negative integer", injection_id))
            continue
        if window_index >= count_cache[sequence_id]:
            failures.append(
                _injection_failure(
                    f"injection references non-existent window {window_index} for {sequence_id}",
                    injection_id,
                )
            )
            continue
        if kind == "non_finite_observation":
            position = item.get("position", 0)
            if isinstance(position, bool) or not isinstance(position, int) or not (0 <= position < config.backtest.window_length):
                failures.append(_injection_failure("non_finite_observation.position must be inside the train window", injection_id))
                continue
            marker = item.get("value", "NaN")
            if marker not in {"NaN", "Infinity", "-Infinity"}:
                failures.append(_injection_failure("non_finite_observation.value must be NaN, Infinity or -Infinity", injection_id))
                continue
        elif kind == "irregular_timestamp":
            position = item.get("position", config.backtest.window_length - 1)
            max_position = config.backtest.window_length + config.backtest.forecast_horizon
            if isinstance(position, bool) or not isinstance(position, int) or not (0 <= position < max_position):
                failures.append(_injection_failure("irregular_timestamp.position is outside the window", injection_id))
                continue
            offset = item.get("offset_ms", 1)
            if isinstance(offset, bool) or not isinstance(offset, int) or offset == 0:
                failures.append(_injection_failure("irregular_timestamp.offset_ms must be a non-zero integer", injection_id))
                continue
        elif kind == "window_too_long":
            requested = item.get("window_length")
            if isinstance(requested, bool) or not isinstance(requested, int) or requested <= config.backtest.window_length:
                failures.append(
                    _injection_failure(
                        "window_too_long.window_length must be greater than configured window_length",
                        injection_id,
                    )
                )
                continue
        elif kind == "quantization_out_of_range":
            level = item.get("quantization_level")
            if isinstance(level, bool) or not isinstance(level, int):
                failures.append(_injection_failure("quantization_out_of_range.quantization_level must be an integer", injection_id))
                continue
            model_name = item.get("model")
            if model_name is not None and model_name not in config.models.enabled:
                failures.append(_injection_failure("quantization injection references unknown model", injection_id))
                continue
        valid.append(
            Injection(
                injection_id=injection_id,
                kind=kind,
                sequence_id=sequence_id,
                window_index=window_index,
                raw=item,
            )
        )

    grouped: dict[tuple[str, int], list[Injection]] = {}
    for injection in sorted(valid, key=lambda item: item.injection_id):
        grouped.setdefault((injection.sequence_id, injection.window_index), []).append(injection)
    frozen = {key: tuple(value) for key, value in grouped.items()}
    return LoadedInjections(
        by_window=frozen,
        items=tuple(sorted(valid, key=lambda item: item.injection_id)),
        failures=failures,
        manifest_hash=manifest_hash,
        fatal=False,
    )


def window_fault_fingerprint(injections: tuple[Injection, ...]) -> str:
    return fingerprint(
        {
            "schema": "window-faults-v1",
            "injections": [injection.to_dict() for injection in injections],
        }
    )
