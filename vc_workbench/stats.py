"""Speaker and segment-local acoustic statistic aggregation."""

from __future__ import annotations

from typing import Any, Iterable

from .config import WorkbenchConfig
from .dsp import fixed_stats, vector_stats
from .util import sha256_value


def _speech_frames(frames: Iterable[dict[str, Any]], cfg: WorkbenchConfig) -> list[dict[str, Any]]:
    threshold = cfg.energy_threshold_q
    return [frame for frame in frames if int(frame.get("energy_db_q", -10**18)) >= threshold]


def aggregate_stats(
    frames: list[dict[str, Any]],
    cfg: WorkbenchConfig,
    *,
    cache_key: str | None = None,
    source_keys: list[str] | None = None,
) -> dict[str, Any]:
    speech = _speech_frames(frames, cfg)
    energies = [int(frame["energy_q"]) for frame in speech]
    energy_db = [int(frame["energy_db_q"]) for frame in speech]
    f0_units = [int(frame["f0_unit"]) for frame in speech if int(frame["f0_unit"]) >= 0]
    cepstra = [[int(value) for value in frame["cepstrum_q"]] for frame in speech]
    logmels = [[int(value) for value in frame["mel_energy_q"]] for frame in speech]
    payload: dict[str, Any] = {
        "count": len(speech),
        "speech_frame_count": len(speech),
        "energy": fixed_stats(energies, cfg.quantiles, cfg.quantile_interpolation),
        "energy_db": fixed_stats(energy_db, cfg.quantiles, cfg.quantile_interpolation),
        "f0_unit": fixed_stats(f0_units, cfg.quantiles, cfg.quantile_interpolation),
        "cepstrum": vector_stats(cepstra, cfg.quantiles, cfg.quantile_interpolation),
        "logmel": vector_stats(logmels, cfg.quantiles, cfg.quantile_interpolation),
        "quantile_interpolation": cfg.quantile_interpolation,
    }
    if source_keys is not None:
        payload["source_keys"] = list(source_keys)
    stats_fingerprint = sha256_value(payload)
    payload["fingerprint"] = stats_fingerprint
    payload["cache_key"] = cache_key or stats_fingerprint
    return payload


def speaker_stats_from_analyses(
    analyses: list[dict[str, Any]],
    cfg: WorkbenchConfig,
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    source_keys: list[str] = []
    for analysis in analyses:
        source_keys.append(str(analysis["analysis_key"]))
        frames.extend(analysis.get("frames", []))
    cache_key = sha256_value(
        {
            "stage": "speaker-stats",
            "analysis_keys": sorted(source_keys),
            "stats_config": cfg.stats_fingerprint(),
        }
    )
    return aggregate_stats(frames, cfg, cache_key=cache_key, source_keys=sorted(source_keys))


def segment_stats_from_analysis(analysis: dict[str, Any], cfg: WorkbenchConfig) -> dict[str, Any]:
    cache_key = sha256_value(
        {
            "stage": "segment-stats",
            "analysis_key": analysis["analysis_key"],
            "stats_config": cfg.stats_fingerprint(),
        }
    )
    return aggregate_stats(analysis.get("frames", []), cfg, cache_key=cache_key, source_keys=[analysis["analysis_key"]])
