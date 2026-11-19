"""Offline full-scan analysis and reproducible report assembly."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .bs1770 import GatedResult, StreamingLoudnessMeter
from .compressor import LogDomainCompressor
from .config import (
    ABSOLUTE_GATE_LKFS,
    ENVELOPE_TOLERANCE_DB,
    LOUDNESS_TOLERANCE_LU,
    RELATIVE_GATE_OFFSET_LU,
    SAMPLE_RATE,
    TRUE_PEAK_TOLERANCE_DB,
    TRUE_PEAK_UPSAMPLE,
    CompressorConfig,
)
from .errors import E_REPORT_MISMATCH, WorkbenchError


def run_offline_analysis(
    signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    absolute_gate: float = ABSOLUTE_GATE_LKFS,
    relative_offset: float = RELATIVE_GATE_OFFSET_LU,
) -> dict:
    """Scan the whole programme in one meter instance."""
    meter = StreamingLoudnessMeter(sample_rate=sample_rate)
    meter.process(signal)
    gated: GatedResult = meter.finalize(absolute_gate, relative_offset)
    return {
        "convention": {
            "standard": "ITU-R BS.1770-4",
            "sample_rate": sample_rate,
            "integration_window_ms": 400,
            "window_hop_ms": 100,
            "absolute_gate_lkfs": absolute_gate,
            "relative_gate_offset_lu": relative_offset,
            "true_peak_oversampling": TRUE_PEAK_UPSAMPLE,
        },
        "integrated_lkfs": gated.integrated_lkfs,
        "relative_gate_lkfs": gated.relative_gate_lkfs,
        "loudness_range_lu": gated.lra_lu,
        "true_peak_dbtp": meter.true_peak.peak_dbtp(),
        "blocks_total": gated.blocks_total,
        "blocks_below_absolute": gated.blocks_below_absolute,
        "blocks_below_relative": gated.blocks_below_relative,
        "blocks_kept": gated.blocks_kept,
        "short_term_series": [
            {
                "index": b.index,
                "start_s": round(b.start_s, 6),
                "end_s": round(b.end_s, 6),
                "loudness_lkfs": _safe_float(b.loudness_lkfs),
            }
            for b in gated.short_term
        ],
        "meter_state": meter.get_state(),
    }


def run_offline_compression(
    signal: np.ndarray, config: CompressorConfig, sample_rate: int = SAMPLE_RATE
) -> tuple[np.ndarray, LogDomainCompressor]:
    compressor = LogDomainCompressor(config, sample_rate=sample_rate)
    processed = compressor.process(signal)
    return processed, compressor


def envelope_hash(envelope: np.ndarray) -> str:
    """Stable SHA-256 over the raw float64 envelope bytes."""
    return hashlib.sha256(np.asarray(envelope, dtype=np.float64).tobytes()).hexdigest()


def assert_conclusions_match(
    streaming: dict,
    offline: dict,
    stream_envelope: np.ndarray,
    offline_envelope: np.ndarray,
    loudness_tol: float = LOUDNESS_TOLERANCE_LU,
    tp_tol: float = TRUE_PEAK_TOLERANCE_DB,
    env_tol: float = ENVELOPE_TOLERANCE_DB,
) -> dict:
    """Explicit pass/fail comparison between the two processing paths."""
    l_diff = abs(streaming["integrated_lkfs"] - offline["integrated_lkfs"])
    tp_diff = abs(streaming["true_peak_dbtp"] - offline["true_peak_dbtp"])
    env_diff = float(np.max(np.abs(stream_envelope - offline_envelope)))
    verdict = {
        "loudness_abs_diff_lu": l_diff,
        "true_peak_abs_diff_db": tp_diff,
        "envelope_max_abs_diff_db": env_diff,
        "loudness_tolerance_lu": loudness_tol,
        "true_peak_tolerance_db": tp_tol,
        "envelope_tolerance_db": env_tol,
        "loudness_pass": bool(l_diff <= loudness_tol),
        "true_peak_pass": bool(tp_diff <= tp_tol),
        "envelope_pass": bool(env_diff <= env_tol),
    }
    verdict["all_pass"] = bool(
        verdict["loudness_pass"] and verdict["true_peak_pass"] and verdict["envelope_pass"]
    )
    if not verdict["all_pass"]:
        raise WorkbenchError(
            E_REPORT_MISMATCH,
            f"streaming/offline mismatch: {json.dumps(verdict, sort_keys=True)}",
        )
    return verdict


def _safe_float(value: float) -> float | None:
    if np.isneginf(value) or np.isnan(value):
        return None
    return float(value)


def write_report(path: str, report: dict) -> None:
    """Deterministic JSON: sorted keys, fixed separators, trailing newline."""
    def sanitize(node):
        if isinstance(node, dict):
            return {key: sanitize(value) for key, value in node.items()}
        if isinstance(node, (list, tuple)):
            return [sanitize(value) for value in node]
        if isinstance(node, float) and (np.isnan(node) or np.isinf(node)):
            return None
        if isinstance(node, np.ndarray):
            return sanitize(node.tolist())
        if isinstance(node, (np.floating, np.integer)):
            return node.item()
        return node

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            sanitize(report),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    raise TypeError(f"not JSON serializable: {type(obj)!r}")
