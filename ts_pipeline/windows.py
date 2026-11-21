"""Deterministic backtest window construction and within-window imputation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .canonical import fingerprint, to_builtin
from .config import Config
from .errors import E_CONFIG_INVALID, E_SAMPLE_INVALID, Failure
from .injections import Injection, window_fault_fingerprint
from .samples import SequenceSample


@dataclass
class WindowBuildError(RuntimeError):
    code: str
    message: str
    injection_id: str | None = None
    detail: dict[str, Any] | None = None

    def as_failure(self, sequence_id: str, window_index: int) -> Failure:
        return Failure(
            code=self.code,
            stage="window_build",
            message=self.message,
            sequence_id=sequence_id,
            window_index=window_index,
            injection_id=self.injection_id,
            detail=self.detail or {},
        )


def _is_equally_spaced(timestamps: list[int]) -> bool:
    if len(timestamps) < 2:
        return True
    interval = timestamps[1] - timestamps[0]
    if interval <= 0:
        return False
    for index in range(2, len(timestamps)):
        if timestamps[index] - timestamps[index - 1] != interval:
            return False
    return True


def _impute(values: list[float | None], window_index: int, start_index: int) -> tuple[list[float], list[dict]]:
    result: list[float | None] = list(values)
    records: list[dict] = []
    observed = [index for index, value in enumerate(result) if value is not None]
    if not observed:
        raise WindowBuildError(
            E_SAMPLE_INVALID,
            "training window has no observed value available for imputation",
        )
    for position, value in enumerate(result):
        if value is not None:
            continue
        before = [index for index in observed if index < position]
        after = [index for index in observed if index > position]
        left = before[-1] if before else None
        right = after[0] if after else None
        if left is not None and right is not None:
            left_value = float(result[left])  # type: ignore[arg-type]
            right_value = float(result[right])  # type: ignore[arg-type]
            fraction = (position - left) / (right - left)
            filled = left_value + (right_value - left_value) * fraction
            method = "linear_interpolation"
            sources = [left, right]
        elif left is not None:
            filled = float(result[left])  # type: ignore[arg-type]
            method = "carry_forward"
            sources = [left]
        else:
            filled = float(result[right])  # type: ignore[arg-type]
            method = "carry_backward"
            sources = [right]
        result[position] = filled
        records.append(
            {
                "position_in_window": position,
                "global_index": start_index + position,
                "method": method,
                "source_positions_in_window": sources,
                "source_global_indices": [start_index + source for source in sources],
                "source_window": {
                    "series_id": None,
                    "window_index": window_index,
                    "start_index": start_index,
                    "end_index_exclusive": start_index + len(values),
                },
                "filled_value": filled,
            }
        )
    return [float(value) for value in result], records  # type: ignore[arg-type]


def window_cache_key(
    sample: SequenceSample,
    window_index: int,
    config: Config,
    injections: tuple[Injection, ...],
) -> str:
    return fingerprint(
        {
            "schema": "window-artifact-v1",
            "series_id": sample.sequence_id,
            "series_fingerprint": sample.content_fingerprint,
            "window_index": window_index,
            "window_spec": {
                "window_length": config.backtest.window_length,
                "step": config.backtest.step,
                "forecast_horizon": config.backtest.forecast_horizon,
            },
            "fault_fingerprint": window_fault_fingerprint(injections),
        }
    )


def build_window(
    sample: SequenceSample,
    window_index: int,
    config: Config,
    injections: tuple[Injection, ...],
) -> dict[str, Any]:
    backtest = config.backtest
    start = window_index * backtest.step
    effective_window_length = backtest.window_length
    active_injections = list(injections)
    fault_fingerprint = window_fault_fingerprint(injections)

    for injection in injections:
        if injection.kind == "window_too_long":
            requested = int(injection.raw["window_length"])
            effective_window_length = max(effective_window_length, requested)
            raise WindowBuildError(
                E_CONFIG_INVALID,
                f"injected window length {requested} exceeds available sequence support",
                injection_id=injection.injection_id,
                detail={"requested_window_length": requested, "sequence_length": sample.length},
            )

    required_end = start + effective_window_length + backtest.forecast_horizon
    if required_end > sample.length:
        raise WindowBuildError(
            E_CONFIG_INVALID,
            "window plus forecast horizon exceeds available sequence length",
            detail={
                "start_index": start,
                "required_end_index_exclusive": required_end,
                "sequence_length": sample.length,
            },
        )

    timestamps = list(sample.timestamps_ms[start:required_end])
    values: list[float | None] = list(sample.values[start:required_end])
    missing = list(sample.missing[start:required_end])

    for injection in injections:
        if injection.kind == "non_finite_observation":
            position = int(injection.raw["position"])
            marker = injection.raw.get("value", "NaN")
            values[position] = {
                "NaN": float("nan"),
                "Infinity": float("inf"),
                "-Infinity": float("-inf"),
            }[marker]
            missing[position] = False
        elif injection.kind == "irregular_timestamp":
            position = int(injection.raw["position"])
            timestamps[position] += int(injection.raw["offset_ms"])

    if not _is_equally_spaced(timestamps):
        bad = next(
            (
                injection
                for injection in injections
                if injection.kind == "irregular_timestamp"
            ),
            None,
        )
        raise WindowBuildError(
            E_SAMPLE_INVALID,
            "window timestamps are not strictly equally spaced",
            injection_id=bad.injection_id if bad else None,
        )
    for position, value in enumerate(values):
        if missing[position]:
            continue
        if value is None or not math.isfinite(float(value)):
            bad = next(
                (
                    injection
                    for injection in injections
                    if injection.kind == "non_finite_observation"
                ),
                None,
            )
            raise WindowBuildError(
                E_SAMPLE_INVALID,
                f"window value at position {position} is not finite",
                injection_id=bad.injection_id if bad else None,
                detail={"position_in_window": position, "global_index": start + position},
            )

    raw_training = values[:effective_window_length]
    train_values, imputations = _impute(raw_training, window_index, start)
    for record in imputations:
        record["source_window"]["series_id"] = sample.sequence_id
    forecast_start = start + effective_window_length
    actual_values = values[effective_window_length : effective_window_length + backtest.forecast_horizon]
    actual_missing = missing[effective_window_length : effective_window_length + backtest.forecast_horizon]
    forecast_timestamps = timestamps[effective_window_length : effective_window_length + backtest.forecast_horizon]

    window_spec = {
        "window_length": backtest.window_length,
        "step": backtest.step,
        "forecast_horizon": backtest.forecast_horizon,
    }
    window_key = fingerprint(
        {
            "schema": "window-artifact-v1",
            "series_id": sample.sequence_id,
            "series_fingerprint": sample.content_fingerprint,
            "window_index": window_index,
            "window_spec": window_spec,
            "fault_fingerprint": fault_fingerprint,
        }
    )
    actual_payload = [
        None if missing_value else float(value)
        for value, missing_value in zip(actual_values, actual_missing)
    ]
    artifact = {
        "schema": "window-artifact-v1",
        "window_key": window_key,
        "series_id": sample.sequence_id,
        "series_fingerprint": sample.content_fingerprint,
        "window_index": window_index,
        "start_index": start,
        "train_start_index": start,
        "train_end_index_exclusive": start + effective_window_length,
        "train_timestamps_ms": timestamps[:effective_window_length],
        "train_values": train_values,
        "original_train_values": [to_builtin(value) for value in raw_training],
        "imputation_records": imputations,
        "imputed_positions": [record["position_in_window"] for record in imputations],
        "forecast_start_index": forecast_start,
        "forecast_timestamps_ms": forecast_timestamps,
        "actual_values": actual_payload,
        "actual_missing": actual_missing,
        "actual_fingerprint": fingerprint(
            {
                "schema": "actual-values-v1",
                "values": actual_payload,
                "missing": actual_missing,
            }
        ),
        "window_spec": window_spec,
        "effective_window_length": effective_window_length,
        "fault_fingerprint": fault_fingerprint,
        "injection_ids": [injection.injection_id for injection in active_injections],
        "source_window": {
            "series_id": sample.sequence_id,
            "window_index": window_index,
            "start_index": start,
            "end_index_exclusive": start + effective_window_length,
        },
    }
    return artifact
