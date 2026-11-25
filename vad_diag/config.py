"""Typed configuration loading and validation.

The single JSON configuration file describes:

* frame / analysis-window geometry,
* dual-threshold hysteresis VAD parameters,
* the fixed-point quantization format,
* the interaction latency budget and the thread-pool concurrency cap,
* storage locations (created by the program itself),
* an optional pointer to an *injection manifest* used to stage real faults.

All values are validated eagerly so an invalid run never silently proceeds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .errors import VADValidationError, E_FIELD_INVALID
from .fixedpoint import ALLOWED_FRAC_BITS, ALLOWED_QUANT_MODES

# Re-exported from the package root for convenience.
__all__ = [
    "Config",
    "InjectionRule",
    "load_config",
    "load_injection_manifest",
    "ALLOWED_FRAC_BITS",
    "ALLOWED_QUANT_MODES",
]

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2

# Injection fault vocabulary (see injection.py for semantics).
INJECTION_FAULTS = (
    "truncate_data",        # shorten the RIFF data chunk / sample payload
    "non_finite_sample",    # force a NaN/Inf into the sample timeline
    "range_overflow",       # reference a sample interval outside the timeline
    "quant_mode_invalid",   # present an unsupported quantization mode
    "feature_dim_mismatch", # produce a window feature vector of wrong width
)


def _require_int(name: str, value: Any, *, minimum: Optional[int] = None,
                 positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VADValidationError(
            f"{name} must be an integer, got {type(value).__name__}",
            E_FIELD_INVALID, detail={"field": name, "value": value})
    if positive and value <= 0:
        raise VADValidationError(
            f"{name} must be a positive integer", E_FIELD_INVALID,
            detail={"field": name, "value": value})
    if minimum is not None and value < minimum:
        raise VADValidationError(
            f"{name} must be >= {minimum}", E_FIELD_INVALID,
            detail={"field": name, "value": value})
    return value


def _require_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VADValidationError(
            f"{name} must be a number", E_FIELD_INVALID,
            detail={"field": name, "value": value})
    fv = float(value)
    if fv != fv or fv in (float("inf"), float("-inf")):
        raise VADValidationError(
            f"{name} must be finite", E_FIELD_INVALID,
            detail={"field": name, "value": value})
    return fv


@dataclass(frozen=True)
class InjectionRule:
    """One deterministic fault injection entry."""

    injection_id: str
    fault: str
    stage: str
    target: str                 # segment id or window id
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "injection_id": self.injection_id,
            "fault": self.fault,
            "stage": self.stage,
            "target": self.target,
        }
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True)
class Config:
    # Frame geometry (in samples).
    frame_length: int
    frame_shift: int
    # Analysis window geometry (in samples).
    window_length: int
    window_shift: int
    # Dual-threshold hysteresis.
    energy_low: float
    energy_high: float
    flatness_low: float
    flatness_high: float
    min_speech_frames: int
    min_silence_frames: int
    # Fixed point.
    frac_bits: int
    quant_mode: str
    # Posterior / decision.
    speech_threshold: float     # de-quantized threshold in [0, 1]
    # Interaction semantics.
    latency_budget_ms: int
    segment_cost_ms: int
    window_cost_ms: int
    max_workers: int
    # Storage.
    snapshot_dir: str
    ledger_dir: str
    sample_rate: int = DEFAULT_SAMPLE_RATE
    # Optional injection manifest path (resolved relative to config file).
    injection_manifest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def material_params(self) -> Dict[str, Any]:
        """Parameters that feed the VAD/window parameter fingerprint."""
        return {
            "frame_length": self.frame_length,
            "frame_shift": self.frame_shift,
            "window_length": self.window_length,
            "window_shift": self.window_shift,
            "energy_low": self.energy_low,
            "energy_high": self.energy_high,
            "flatness_low": self.flatness_low,
            "flatness_high": self.flatness_high,
            "min_speech_frames": self.min_speech_frames,
            "min_silence_frames": self.min_silence_frames,
            "speech_threshold": self.speech_threshold,
        }

    def quant_params(self) -> Dict[str, Any]:
        return {"frac_bits": self.frac_bits, "quant_mode": self.quant_mode}


def _check_geometry(cfg: Dict[str, Any]) -> None:
    frame_length = _require_int("frame_length", cfg.get("frame_length"),
                                positive=True)
    frame_shift = _require_int("frame_shift", cfg.get("frame_shift"),
                               positive=True)
    if frame_shift > frame_length:
        raise VADValidationError(
            "frame_shift must be <= frame_length", E_FIELD_INVALID,
            detail={"frame_shift": frame_shift,
                    "frame_length": frame_length})
    window_length = _require_int("window_length", cfg.get("window_length"),
                                 positive=True)
    window_shift = _require_int("window_shift", cfg.get("window_shift"),
                                positive=True)
    if window_shift > window_length:
        raise VADValidationError(
            "window_shift must be <= window_length", E_FIELD_INVALID,
            detail={"window_shift": window_shift,
                    "window_length": window_length})
    _require_int("min_speech_frames", cfg.get("min_speech_frames"),
                 minimum=1)
    _require_int("min_silence_frames", cfg.get("min_silence_frames"),
                 minimum=1)


def _check_thresholds(cfg: Dict[str, Any]) -> None:
    energy_low = _require_number("energy_low", cfg.get("energy_low"))
    energy_high = _require_number("energy_high", cfg.get("energy_high"))
    if energy_low > energy_high:
        raise VADValidationError(
            "energy_low must be <= energy_high", E_FIELD_INVALID,
            detail={"low": energy_low, "high": energy_high})
    # Flatness uses inverted polarity (low flatness => tonal speech); both
    # thresholds are independently validated as finite, bounded values.
    flatness_low = _require_number("flatness_low", cfg.get("flatness_low"))
    flatness_high = _require_number("flatness_high",
                                    cfg.get("flatness_high"))
    for name, value in (("flatness_low", flatness_low),
                        ("flatness_high", flatness_high)):
        if not 0.0 <= value <= 1.0:
            raise VADValidationError(
                f"{name} must be within [0, 1]", E_FIELD_INVALID,
                detail={name: value})
    threshold = _require_number("speech_threshold",
                                cfg.get("speech_threshold"))
    if not 0.0 <= threshold <= 1.0:
        raise VADValidationError(
            "speech_threshold must be within [0, 1]", E_FIELD_INVALID,
            detail={"speech_threshold": threshold})


def build_config(raw: Dict[str, Any], *, base_dir: Optional[str] = None
                 ) -> Config:
    if not isinstance(raw, dict):
        raise VADValidationError(
            "configuration root must be a JSON object", E_FIELD_INVALID)
    _check_geometry(raw)
    _check_thresholds(raw)

    frac_bits = _require_int("frac_bits", raw.get("frac_bits"))
    if frac_bits not in ALLOWED_FRAC_BITS:
        raise VADValidationError(
            f"frac_bits {frac_bits} not in {list(ALLOWED_FRAC_BITS)}",
            E_FIELD_INVALID, detail={"frac_bits": frac_bits})
    quant_mode = raw.get("quant_mode")
    if quant_mode not in ALLOWED_QUANT_MODES:
        raise VADValidationError(
            f"quant_mode {quant_mode!r} not in {list(ALLOWED_QUANT_MODES)}",
            E_FIELD_INVALID, detail={"quant_mode": quant_mode})

    latency_budget_ms = _require_int(
        "latency_budget_ms", raw.get("latency_budget_ms"), minimum=0)
    segment_cost_ms = _require_int(
        "segment_cost_ms", raw.get("segment_cost_ms", 1), minimum=0)
    window_cost_ms = _require_int(
        "window_cost_ms", raw.get("window_cost_ms", 1), minimum=0)
    max_workers = _require_int("max_workers", raw.get("max_workers", 1),
                               positive=True)

    snapshot_dir = raw.get("snapshot_dir", "snapshots")
    ledger_dir = raw.get("ledger_dir", "ledger")
    if not isinstance(snapshot_dir, str) or not snapshot_dir:
        raise VADValidationError("snapshot_dir must be a non-empty string",
                                 E_FIELD_INVALID)
    if not isinstance(ledger_dir, str) or not ledger_dir:
        raise VADValidationError("ledger_dir must be a non-empty string",
                                 E_FIELD_INVALID)
    if base_dir:
        if not os.path.isabs(snapshot_dir):
            snapshot_dir = os.path.join(base_dir, snapshot_dir)
        if not os.path.isabs(ledger_dir):
            ledger_dir = os.path.join(base_dir, ledger_dir)

    sample_rate = _require_int(
        "sample_rate", raw.get("sample_rate", DEFAULT_SAMPLE_RATE), positive=True)

    injection_manifest = raw.get("injection_manifest")
    if injection_manifest is not None:
        if not isinstance(injection_manifest, str):
            raise VADValidationError(
                "injection_manifest must be a string path", E_FIELD_INVALID)
        if base_dir and not os.path.isabs(injection_manifest):
            injection_manifest = os.path.join(base_dir, injection_manifest)

    return Config(
        frame_length=raw["frame_length"],
        frame_shift=raw["frame_shift"],
        window_length=raw["window_length"],
        window_shift=raw["window_shift"],
        energy_low=float(raw["energy_low"]),
        energy_high=float(raw["energy_high"]),
        flatness_low=float(raw["flatness_low"]),
        flatness_high=float(raw["flatness_high"]),
        min_speech_frames=raw["min_speech_frames"],
        min_silence_frames=raw["min_silence_frames"],
        frac_bits=frac_bits,
        quant_mode=quant_mode,
        speech_threshold=float(raw["speech_threshold"]),
        latency_budget_ms=latency_budget_ms,
        segment_cost_ms=segment_cost_ms,
        window_cost_ms=window_cost_ms,
        max_workers=max_workers,
        snapshot_dir=snapshot_dir,
        ledger_dir=ledger_dir,
        sample_rate=sample_rate,
        injection_manifest=injection_manifest,
    )


def load_config(path: str) -> Config:
    if not os.path.isfile(path):
        raise VADValidationError(
            f"config file not found: {path}", E_FIELD_INVALID)
    with open(path, "r", encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise VADValidationError(
                f"config is not valid JSON: {exc}", E_FIELD_INVALID) from exc
    base_dir = os.path.dirname(os.path.abspath(path))
    return build_config(raw, base_dir=base_dir)


def load_injection_manifest(path: Optional[str]) -> List[InjectionRule]:
    if not path:
        return []
    if not os.path.isfile(path):
        raise VADValidationError(
            f"injection manifest not found: {path}", E_FIELD_INVALID)
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("injections"), list):
        raise VADValidationError(
            "injection manifest must contain an 'injections' list",
            E_FIELD_INVALID)
    rules: List[InjectionRule] = []
    seen_ids = set()
    for entry in raw["injections"]:
        if not isinstance(entry, dict):
            raise VADValidationError(
                "each injection must be an object", E_FIELD_INVALID)
        inj_id = entry.get("injection_id")
        fault = entry.get("fault")
        stage = entry.get("stage")
        target = entry.get("target")
        if not inj_id or not isinstance(inj_id, str):
            raise VADValidationError(
                "injection.injection_id is required", E_FIELD_INVALID)
        if inj_id in seen_ids:
            raise VADValidationError(
                f"duplicate injection_id {inj_id}", E_FIELD_INVALID)
        seen_ids.add(inj_id)
        if fault not in INJECTION_FAULTS:
            raise VADValidationError(
                f"unknown injection fault {fault!r}, allowed "
                f"{list(INJECTION_FAULTS)}", E_FIELD_INVALID,
                detail={"injection_id": inj_id})
        if not isinstance(target, str) or not target:
            raise VADValidationError(
                "injection.target is required", E_FIELD_INVALID,
                detail={"injection_id": inj_id})
        detail = entry.get("detail", {})
        if not isinstance(detail, dict):
            raise VADValidationError(
                "injection.detail must be an object", E_FIELD_INVALID,
                detail={"injection_id": inj_id})
        rules.append(InjectionRule(inj_id, fault, str(stage), target, detail))
    # Deterministic ordering by declared id regardless of file ordering.
    rules.sort(key=lambda r: r.injection_id)
    return rules
