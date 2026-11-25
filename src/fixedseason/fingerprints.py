"""Content fingerprints and stage cache keys."""

from __future__ import annotations

from typing import Any, Iterable

from .jsonutil import fingerprint
from .model import FitConfig, SeriesRecord

CACHE_STAGES = (
    "interpolation",
    "basis",
    "normal_equations",
    "coefficients",
)


def series_content_fingerprint(series: SeriesRecord) -> str:
    return fingerprint(
        {
            "series_id": series.series_id,
            "timestamps": list(series.timestamps),
            "values": [
                "<MISSING>" if value is None else value for value in series.values
            ],
        }
    )


def config_computation_fingerprint(config: FitConfig) -> str:
    return fingerprint(config.computational_dict())


def config_full_fingerprint(config: FitConfig) -> str:
    return fingerprint(config.full_dict())


def series_library_fingerprint(series: Iterable[SeriesRecord]) -> dict[str, str]:
    return {
        item.series_id: series_content_fingerprint(item)
        for item in sorted(series, key=lambda record: record.series_id)
    }


def library_fingerprint(series: Iterable[SeriesRecord]) -> str:
    return fingerprint(series_library_fingerprint(series))


def stage_cache_key(
    stage: str,
    *,
    series_fingerprint: str,
    config_fingerprint: str,
    bit_tier: str,
) -> str:
    if stage not in CACHE_STAGES:
        raise ValueError(f"unknown cache stage {stage!r}")
    return fingerprint(
        {
            "schema": 1,
            "stage": stage,
            "series_content": series_fingerprint,
            "config": config_fingerprint,
            "bit_width_tier": bit_tier,
        }
    )


def all_stage_keys(
    *,
    series_fingerprint: str,
    config: FitConfig,
    config_fingerprint: str | None = None,
) -> dict[str, str]:
    config_fp = config_fingerprint or config_computation_fingerprint(config)
    return {
        stage: stage_cache_key(
            stage,
            series_fingerprint=series_fingerprint,
            config_fingerprint=config_fp,
            bit_tier=config.bit_tier,
        )
        for stage in CACHE_STAGES
    }


def referenced_observations(series: SeriesRecord) -> dict[str, Any]:
    return {
        "series_id": series.series_id,
        "observed_indices": list(series.observed_indices),
        "missing_indices": list(series.missing_indices),
        "time_index_range": [0, series.length - 1],
        "window": "entire_series",
    }
