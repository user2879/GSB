"""JSONL command-stream validation and immutable state edits."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import E_COMMAND_INVALID, Issue
from .jsonutil import loads_strict
from .loader import parse_observation_value, validate_series_object
from .model import (
    ALLOWED_FIXED_POINT_BITS,
    ALLOWED_QUANTIZATION_SHIFTS,
    MAX_HARMONIC_ORDER,
    MAX_REGULARIZATION,
    SCALE_PRESETS,
    SCALE_PROFILES,
    FitConfig,
    SeriesRecord,
)


@dataclass(frozen=True)
class CommandState:
    series: tuple[SeriesRecord, ...]
    config: FitConfig


class CommandProcessor:
    def __init__(
        self,
        series: Iterable[SeriesRecord],
        config: FitConfig,
        *,
        source: str = "<commands>",
    ) -> None:
        self._series = {record.series_id: record for record in series}
        self._config = config
        self.source = source
        self.issues: list[Issue] = []

    @property
    def state(self) -> CommandState:
        return CommandState(
            series=tuple(self._series[key] for key in sorted(self._series)),
            config=self._config,
        )

    def _invalid(self, line: int | None, message: str, series_id: str | None = None) -> None:
        self.issues.append(
            Issue(
                code=E_COMMAND_INVALID,
                message=message,
                source=self.source,
                line=line,
                series_id=series_id,
                stage="command",
            )
        )

    def _next_config(self, **changes: Any) -> FitConfig:
        return self._config.with_changes(version=self._config.version + 1, **changes)

    def apply(self, command: Mapping[str, Any], *, line: int | None = None) -> bool:
        op = command.get("op")
        if not isinstance(op, str):
            self._invalid(line, "command.op must be a string")
            return False
        try:
            if op == "run":
                return True
            if op == "set_harmonic_order":
                value = command.get("value", command.get("harmonic_order"))
                if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= MAX_HARMONIC_ORDER):
                    self._invalid(line, f"harmonic order must be an integer in [0, {MAX_HARMONIC_ORDER}]")
                    return False
                self._config = self._next_config(harmonic_order=value)
                return True
            if op == "set_regularization_strength":
                value = command.get("value", command.get("regularization_strength"))
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not (0.0 <= float(value) <= MAX_REGULARIZATION)
                ):
                    self._invalid(line, f"regularization_strength must be in [0, {MAX_REGULARIZATION}]")
                    return False
                self._config = self._next_config(regularization_strength=float(value))
                return True
            if op == "set_fixed_point":
                bits = command.get("bits", command.get("fixed_point_bits"))
                shift = command.get("quantization_shift", command.get("quantization_tier_shift"))
                if isinstance(bits, bool) or not isinstance(bits, int) or bits not in ALLOWED_FIXED_POINT_BITS:
                    self._invalid(line, f"fixed-point width must be one of {list(ALLOWED_FIXED_POINT_BITS)}")
                    return False
                if (
                    isinstance(shift, bool)
                    or not isinstance(shift, int)
                    or shift not in ALLOWED_QUANTIZATION_SHIFTS
                    or shift > bits - 8
                ):
                    self._invalid(line, "quantization shift is not legal for the selected fixed-point width")
                    return False
                if self._config.pivot_threshold > (1 << (bits - 1)) - 1:
                    self._invalid(line, "pivot threshold does not fit the selected fixed-point width")
                    return False
                self._config = self._next_config(
                    fixed_point_bits=bits, quantization_shift=shift
                )
                return True
            if op == "set_scale_profile":
                profile = command.get("profile", command.get("value"))
                if not isinstance(profile, str) or profile not in SCALE_PROFILES:
                    self._invalid(line, f"scale profile must be one of {list(SCALE_PROFILES)}")
                    return False
                preset = SCALE_PRESETS[profile]
                self._config = self._next_config(
                    scale_profile=profile,
                    global_fit_points=preset["global_fit_points"],
                    per_series_quota=preset["per_series_quota"],
                    batch_observation_points=preset["batch_observation_points"],
                )
                return True
            if op == "set_budget":
                changes: dict[str, Any] = {}
                for key in (
                    "global_fit_points",
                    "per_series_quota",
                    "batch_observation_points",
                ):
                    if key not in command:
                        continue
                    value = command[key]
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        self._invalid(line, f"{key} must be a positive integer")
                        return False
                    changes[key] = value
                if not changes:
                    self._invalid(line, "set_budget requires at least one budget field")
                    return False
                self._config = self._next_config(**changes)
                return True
            if op == "set_thread_pool_workers":
                value = command.get("value", command.get("workers"))
                cpu_count = os.cpu_count() or 1
                if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= cpu_count):
                    self._invalid(line, f"thread pool workers must be in [1, {cpu_count}]")
                    return False
                self._config = self._next_config(thread_pool_workers=value)
                return True
            if op == "replace_observation":
                series_id = command.get("series_id")
                index = command.get("index")
                if not isinstance(series_id, str) or series_id not in self._series:
                    self._invalid(line, "replace_observation targets an unknown series_id")
                    return False
                if isinstance(index, bool) or not isinstance(index, int):
                    self._invalid(line, "replace_observation.index must be an integer", series_id)
                    return False
                record = self._series[series_id]
                if not (0 <= index < record.length):
                    self._invalid(
                        line,
                        f"replace_observation.index {index} outside [0, {record.length - 1}]",
                        series_id,
                    )
                    return False
                if "value" not in command:
                    self._invalid(line, "replace_observation requires value", series_id)
                    return False
                value, error = parse_observation_value(command["value"])
                if error is not None:
                    self._invalid(line, error, series_id)
                    return False
                updated = record.with_value(index, value)
                if not updated.observed_indices:
                    self._invalid(line, "replacement would remove the last observed value", series_id)
                    return False
                self._series[series_id] = updated
                self._config = self._next_config()
                return True
            if op == "append_series":
                payload = command.get("series", command)
                record, issues = validate_series_object(
                    payload, source=self.source, line=line
                )
                if issues or record is None:
                    for issue in issues:
                        self._invalid(line, issue.message, issue.series_id)
                    return False
                if record.series_id in self._series:
                    self._invalid(line, f"duplicate series_id {record.series_id!r}", record.series_id)
                    return False
                self._series[record.series_id] = replace(
                    record, source=self.source, source_line=line or 0
                )
                self._config = self._next_config()
                return True
            if op == "delete_series":
                series_id = command.get("series_id")
                if not isinstance(series_id, str) or series_id not in self._series:
                    self._invalid(line, "delete_series targets an unknown series_id")
                    return False
                del self._series[series_id]
                self._config = self._next_config()
                return True
            self._invalid(line, f"unknown command op {op!r}")
            return False
        except Exception as exc:  # defensive: command parser never aborts the stream
            self._invalid(line, f"command application failed: {type(exc).__name__}: {exc}")
            return False


def load_and_apply_commands(
    path: str | Path,
    processor: CommandProcessor,
) -> list[dict[str, Any]]:
    """Apply JSONL commands sequentially and return a deterministic trace."""

    trace: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        processor._invalid(None, f"cannot read command file: {exc}")  # noqa: SLF001
        return trace
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            command = loads_strict(line)
        except (ValueError, TypeError) as exc:
            processor._invalid(line_number, f"invalid JSON: {exc}")
            trace.append(
                {
                    "line": line_number,
                    "op": "<invalid-json>",
                    "applied": False,
                    "version_after": processor.state.config.version,
                }
            )
            continue
        if not isinstance(command, dict):
            processor._invalid(line_number, "command line must be a JSON object")
            trace.append(
                {
                    "line": line_number,
                    "op": "<invalid-shape>",
                    "applied": False,
                    "version_after": processor.state.config.version,
                }
            )
            continue
        applied = processor.apply(command, line=line_number)
        trace.append(
            {
                "line": line_number,
                "op": command.get("op"),
                "applied": applied,
                "version_after": processor.state.config.version,
            }
        )
    return trace
