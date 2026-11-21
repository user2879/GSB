"""Vocoder pipeline engine: phase -> harmonic -> overlap-add -> metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import dsp
from .dsp import overlap_add, apply_energy_scale, synthesize_frame, normalized_cross_correlation, shift_wave, spectral_distance, waveform_rmse, frame_summaries
from .config import config_fingerprint, phase_profile
from .fixedpoint import FixedPointPhase, Float64Phase
from .hashing import canonical_hash, hash_mix
from .ledger import Ledger, stage_cache_key
from .loader import UtteranceInput


def stable_task_key(utterance_id: str, frame_index: int) -> tuple[str, int]:
    return (utterance_id, frame_index)


@dataclass
class PipelineResult:
    utterance_id: str
    wave_pcm: np.ndarray
    wave_float64_pcm: np.ndarray
    metrics: dict
    attribution: dict
    truncation: dict
    frame_summaries: list[dict]
    alignment: dict
    recompute_keys: list[str]


class VocoderEngine:
    def __init__(self, config: dict, ledger: Ledger):
        self.config = config
        self.ledger = ledger
        self.sr = int(config["sample_rate"])
        self.hop = int(round(config["frame_shift_ms"] * self.sr / 1000.0))
        self.frame_length = int(round(config["frame_length_ms"] * self.sr / 1000.0))

    def _phase_model(self, use_float64: bool):
        if use_float64:
            return Float64Phase()
        return FixedPointPhase(self.config["phase_bits"], self.config["phase_quant_step"])

    def _scaled_frames(self, utter: UtteranceInput) -> list[dict]:
        ratio = float(self.config["f0_ratio"])
        if ratio == 1.0:
            return utter.frames
        return [{
            "frame_index": f["frame_index"],
            "f0": f["f0"] * ratio,
            "harmonic_amplitudes": f["harmonic_amplitudes"],
            "energy_gain": f["energy_gain"],
        } for f in utter.frames]

    def _frame_key_bundle(self, utter: UtteranceInput, scaled: list[dict]) -> tuple[str, str, list[str]]:
        cfg_fp = config_fingerprint(self.config)
        phase_fp = canonical_hash(phase_profile(self.config))
        # Scaled-frame fingerprints are computed directly from the scaled frame
        # content, so an f0_ratio change alters f0 and hence the fingerprint.
        # At ratio 1.0 the scaled frames equal the originals byte-for-byte.
        scaled_fps = [canonical_hash(frame) for frame in scaled]
        return cfg_fp, phase_fp, scaled_fps

    def run_utterance(self, utter: UtteranceInput) -> PipelineResult:
        cfg_fp, phase_fp, frame_fps = self._frame_key_bundle(
            utter, self._scaled_frames(utter)
        )
        scaled = self._scaled_frames(utter)
        frame_width = len(scaled[0]["harmonic_amplitudes"])
        limit = min(int(self.config["harmonic_limit"]), frame_width)
        n_frames = len(scaled)
        self.ledger.register_frames(utter.utterance_id, frame_fps)
        recompute_keys: list[str] = []

        def get_or_compute(stage: str, frame_index, compute):
            key = stage_cache_key(
                stage, utter.utterance_id, utter.feature_fingerprint,
                cfg_fp, phase_fp, frame_index
            )
            cached = self.ledger.lookup(
                stage, utter.utterance_id, key, frame_index, record=False
            )
            if cached is not None:
                self.ledger.record_hit(
                    stage, utter.utterance_id, key, frame_index, True
                )
                return cached, True
            payload = compute()
            self.ledger.store(key, payload)
            self.ledger.record_hit(
                stage, utter.utterance_id, key, frame_index, False
            )
            recompute_keys.append(key)
            return payload, False

        # Phase stage (whole-utterance accumulation, one cache entry per frame).
        phase_payload, phase_reused = get_or_compute(
            "phase", None,
            lambda: self._phase_payload(scaled, frame_fps, utter.utterance_id, False),
        )
        float_payload, _float_reused = get_or_compute(
            "phase", -1,
            lambda: self._phase_payload(scaled, frame_fps, utter.utterance_id, True),
        )
        fixed_phases = np.asarray(phase_payload["phases_rad"], dtype=np.float64)
        float_phases = np.asarray(float_payload["phases_rad"], dtype=np.float64)

        # Harmonic frame synthesis, one entry per frame (stable key order).
        fixed_frames = []
        float_frames = []
        truncation = self._truncation_record(utter)
        for idx in sorted(range(n_frames), key=lambda i: stable_task_key(utter.utterance_id, i)):
            frame = scaled[idx]
            amps = np.asarray(frame["harmonic_amplitudes"][:limit], dtype=np.float64)
            payload, _ = get_or_compute(
                "harmonic", idx,
                lambda idx=idx, amps=amps, frame=frame: {
                    "fixed": synthesize_frame(
                        amps, fixed_phases[idx], frame["f0"],
                        frame["energy_gain"], self.frame_length, self.sr
                    ).tolist(),
                    "float64": synthesize_frame(
                        amps, float_phases[idx], frame["f0"],
                        frame["energy_gain"], self.frame_length, self.sr
                    ).tolist(),
                },
            )
            fixed_frames.append(np.asarray(payload["fixed"], dtype=np.float64))
            float_frames.append(np.asarray(payload["float64"], dtype=np.float64))

        # Overlap-add stage (whole utterance).
        ola_payload, _ = get_or_compute(
            "overlap_add", None,
            lambda: {
                "fixed_pcm": self._ola_to_pcm(fixed_frames).tolist(),
                "float64_pcm": self._ola_to_pcm(float_frames).tolist(),
            },
        )
        wave_pcm = np.asarray(ola_payload["fixed_pcm"], dtype=np.int16)
        wave_float_pcm = np.asarray(ola_payload["float64_pcm"], dtype=np.int16)

        # Metrics / alignment / attribution stage.
        metric_payload, _ = get_or_compute(
            "metric", None,
            lambda: self._metric_payload(utter, wave_pcm, wave_float_pcm,
                                         fixed_phases, float_phases, scaled,
                                         truncation),
        )
        return PipelineResult(
            utterance_id=utter.utterance_id,
            wave_pcm=wave_pcm,
            wave_float64_pcm=wave_float_pcm,
            metrics=metric_payload["metrics"],
            attribution=metric_payload["attribution"],
            truncation=truncation,
            frame_summaries=metric_payload["frame_summaries"],
            alignment=metric_payload["alignment"],
            recompute_keys=recompute_keys,
        )

    def _phase_payload(self, scaled: list[dict], frame_fps: list[str],
                       utter_id: str, use_float64: bool) -> dict:
        model = self._phase_model(use_float64)
        limit = int(self.config["harmonic_limit"])
        phases = model.accumulate(scaled, self.hop, self.sr, utter_id,
                                  frame_fps, limit)
        return {"phases_rad": model.to_radians(phases).tolist(),
                "mode": "float64" if use_float64 else "fixed"}

    def _ola_to_pcm(self, frames: list[np.ndarray]) -> np.ndarray:
        wave_float = overlap_add(
            frames, self.hop, self.config["window_type"],
            self.config["normalization_rule"],
        )
        return apply_energy_scale(wave_float, self.config["energy_rounding"])

    def _truncation_record(self, utter: UtteranceInput) -> dict:
        original = len(utter.frames[0]["harmonic_amplitudes"])
        limit = min(int(self.config["harmonic_limit"]), original)
        dropped = list(range(limit + 1, original + 1)) if limit < original else []
        return {
            "configured_limit": limit,
            "original_limit": original,
            "dropped_harmonics": dropped,
            "affected_frames": [f["frame_index"] for f in utter.frames] if dropped else [],
        }

    def _metric_payload(self, utter: UtteranceInput, wave_pcm: np.ndarray,
                        wave_float_pcm: np.ndarray, fixed_phases: np.ndarray,
                        float_phases: np.ndarray, scaled: list[dict],
                        truncation: dict) -> dict:
        reference = utter.reference.astype(np.float64)
        lag, ncc = normalized_cross_correlation(
            reference, wave_pcm.astype(np.float64),
            self.config["align_search_ms"], self.sr,
        )
        aligned = shift_wave(wave_pcm.astype(np.float64), lag)
        spec_dist = spectral_distance(reference, aligned)
        rmse = waveform_rmse(reference, aligned)
        metrics = {
            "spectral_distance": round(spec_dist, 8),
            "waveform_rmse": round(rmse, 8),
            "normalized_cross_correlation": round(ncc, 8),
        }
        phase_deviation = float(math.sqrt(np.mean(
            (wave_pcm.astype(np.float64) - wave_float_pcm.astype(np.float64)) ** 2
        ))) / 32768.0
        attribution = self._attribute(
            metrics, phase_deviation, fixed_phases, float_phases, scaled, truncation
        )
        summaries = frame_summaries([
            synthesize_frame(
                np.asarray(frame["harmonic_amplitudes"][:int(self.config["harmonic_limit"])],
                           dtype=np.float64),
                fixed_phases[idx], frame["f0"], frame["energy_gain"],
                self.frame_length, self.sr,
            ) for idx, frame in enumerate(scaled)
        ])
        return {
            "metrics": metrics,
            "alignment": {"best_offset_samples": int(lag),
                          "best_offset_ms": round(lag * 1000.0 / self.sr, 4),
                          "correlation": round(ncc, 8)},
            "attribution": attribution,
            "frame_summaries": summaries,
        }

    def _attribute(self, metrics: dict, phase_deviation: float,
                   fixed_phases: np.ndarray, float_phases: np.ndarray,
                   scaled: list[dict], truncation: dict) -> dict:
        spectral_ok = metrics["spectral_distance"] <= self.config["spectral_tol"]
        rmse_bad = metrics["waveform_rmse"] > self.config["rmse_tol"]
        ncc_bad = metrics["normalized_cross_correlation"] < self.config["ncc_tol"]
        waveform_bad = rmse_bad or ncc_bad

        # Ordered stage comparison on the same float64 backbone:
        # phase quantization -> harmonic truncation -> energy rounding.
        # Ordered by pipeline position; the first stage whose error crosses
        # the attribution threshold is reported as the first dominant stage.
        component_errors = {
            "phase_quantization": phase_deviation,
            "harmonic_truncation": self._truncation_error(truncation),
            "energy_rounding": 0.5 / 32768.0,
        }
        stage_order = ("phase_quantization", "harmonic_truncation", "energy_rounding")
        threshold = float(self.config["rmse_tol"])
        first_dominant = next(
            (stage for stage in stage_order
             if component_errors[stage] > threshold),
            None,
        )
        dominant_stage = min(component_errors, key=lambda k: -component_errors[k])
        ranked_stages = sorted(
            component_errors.items(), key=lambda item: (-item[1], stage_order.index(item[0]))
        )
        phase_dominant = bool(spectral_ok and waveform_bad
                              and dominant_stage == "phase_quantization"
                              and phase_deviation > 0.0)
        offending = []
        frame_range = []
        if phase_dominant:
            offending, frame_range = self._phase_offenders(
                fixed_phases, float_phases, scaled
            )
        return {
            "phase_dominant": phase_dominant,
            "dominant_stage": dominant_stage,
            "phase_bit_width_deviation_rmse": round(phase_deviation, 8),
            "component_errors": {k: round(v, 8) for k, v in component_errors.items()},
            "first_dominant_stage": first_dominant,
            "ranked_stages": [
                {"stage": stage, "error": round(value, 8)} for stage, value in ranked_stages
            ],
            "spectral_pass": bool(spectral_ok),
            "waveform_pass": bool(not waveform_bad),
            "offending_harmonics": offending,
            "offending_frame_range": frame_range,
            "threshold_ratio": round(
                phase_deviation / max(self.config["rmse_tol"], 1e-12), 4
            ),
        }

    def _truncation_error(self, truncation: dict) -> float:
        if not truncation["dropped_harmonics"]:
            return 0.0
        # Deterministic surrogate derived from dropped harmonic indices.
        return float(sum(1.0 / h for h in truncation["dropped_harmonics"]))

    def _phase_offenders(self, fixed_phases: np.ndarray, float_phases: np.ndarray,
                         scaled: list[dict]) -> tuple[list[int], list[int]]:
        diff = np.abs(np.angle(np.exp(1j * (fixed_phases - float_phases))))
        per_harmonic = diff.max(axis=0)
        cutoff = float(np.median(diff)) * self.config["phase_attribution_threshold"]
        offending = [int(h + 1) for h in np.where(per_harmonic > cutoff)[0]]
        per_frame = diff.max(axis=1)
        bad_frames = np.where(per_frame > cutoff)[0]
        if bad_frames.size == 0:
            return offending, []
        frame_range = [int(bad_frames[0]), int(bad_frames[-1])]
        return offending, frame_range
