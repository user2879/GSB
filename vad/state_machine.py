"""Four-state dual-threshold hysteresis state machine."""

from __future__ import annotations

from typing import Any

import numpy as np

from .util import PRODUCT_VERSIONS, quantile, round_float


SILENCE = "SILENCE"
SUSPECT = "SUSPECT"
SPEECH = "SPEECH"
TRAIL = "TRAIL"
STATES = (SILENCE, SUSPECT, SPEECH, TRAIL)
ENDPOINT_STATES = frozenset((SUSPECT, SPEECH, TRAIL))


def estimate_noise_floor(
    features: dict[str, Any], policy: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    frames = features["frames"]
    effective = policy["effective_strategy"]
    window = min(int(effective["noise_window_frames"]), len(frames))
    energies = np.asarray(
        [float(frame["rms_log"]) for frame in frames[:window]], dtype=np.float64
    )
    method = str(effective["threshold_estimation"])
    if energies.size == 0:
        reference = 0.0
    elif method == "quantile":
        reference = quantile(energies, float(effective["low_quantile"]), "linear")
    elif method == "fixed_ratio":
        ordered = np.sort(energies)
        lower_count = max(1, (ordered.size + 1) // 2)
        reference = float(np.mean(ordered[:lower_count], dtype=np.float64))
    else:  # pragma: no cover - validated by config
        raise ValueError(f"unknown threshold estimation method: {method}")
    low_margin = float(config["low_margin"])
    hysteresis = float(config["hysteresis"])
    low_threshold = reference + low_margin
    configured_high = float(config["high_threshold"])
    high_threshold = max(configured_high, low_threshold + hysteresis)
    return {
        "estimation_method": method,
        "window_frames_requested": int(effective["noise_window_frames"]),
        "window_frames_effective": int(window),
        "low_quantile": float(effective.get("low_quantile", 0.0)),
        "reference_energy_ln": round_float(reference, 10),
        "window_mean_energy_ln": round_float(
            float(np.mean(energies)) if energies.size else 0.0, 10
        ),
        "low_margin_ln": low_margin,
        "hysteresis_ln": hysteresis,
        "configured_high_threshold_ln": configured_high,
        "low_threshold_ln": round_float(low_threshold, 10),
        "high_threshold_ln": round_float(high_threshold, 10),
        "equal_boundary_policy": "silence_first",
    }


def run_state_machine(
    features: dict[str, Any], noise: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    low = float(noise["low_threshold_ln"])
    high = float(noise["high_threshold_ln"])
    zcr_threshold = float(config["zcr_threshold"])
    flatness_threshold = float(config["flatness_threshold"])
    confirm_frames = int(config["confirm_frames"])
    release_frames = int(config["release_frames"])
    state = SILENCE
    active_run = 0
    release_run = 0
    states_out: list[dict[str, Any]] = []
    counts = {name: 0 for name in STATES}
    transitions = 0

    for frame in features["frames"]:
        before = state
        energy = float(frame["rms_log"])
        zcr = float(frame["zcr"])
        flatness = float(frame["flatness"])
        high_energy = energy > high
        low_energy = energy > low
        zcr_pass = zcr < zcr_threshold
        flatness_pass = flatness < flatness_threshold
        high_evidence = bool(high_energy and zcr_pass and flatness_pass)
        low_evidence = bool(low_energy and zcr_pass and flatness_pass)

        if state == SILENCE:
            if high_evidence:
                state = SUSPECT
                active_run = 1
                criterion = "silence_to_suspect:strict_high_gate_and_spectral_gates"
            else:
                active_run = 0
                criterion = "stay_silence:no_strict_high_evidence"
        elif state == SUSPECT:
            if high_evidence:
                active_run += 1
                if active_run >= confirm_frames:
                    state = SPEECH
                    release_run = 0
                    criterion = (
                        f"suspect_to_speech:consecutive_active_frames={active_run}"
                    )
                else:
                    criterion = (
                        f"stay_suspect:awaiting_consecutive_frames={active_run}"
                    )
            else:
                state = SILENCE
                active_run = 0
                criterion = "suspect_to_silence:consecutive_evidence_broken"
        elif state == SPEECH:
            active_run += 1
            if low_evidence:
                release_run = 0
                criterion = "stay_speech:strict_low_gate_and_spectral_gates"
            else:
                release_run = 1
                if release_run >= release_frames:
                    state = SILENCE
                    release_run = 0
                    active_run = 0
                    criterion = (
                        f"speech_to_silence:release_frames_exhausted={release_frames}"
                    )
                else:
                    state = TRAIL
                    criterion = "speech_to_trail:low_evidence_missing"
        elif state == TRAIL:
            if low_evidence:
                state = SPEECH
                release_run = 0
                criterion = "trail_to_speech:low_evidence_reacquired"
            else:
                release_run += 1
                if release_run >= release_frames:
                    state = SILENCE
                    release_run = 0
                    active_run = 0
                    criterion = (
                        f"trail_to_silence:release_frames_exhausted={release_frames}"
                    )
                else:
                    criterion = (
                        f"stay_trail:release_frames_pending={release_run}"
                    )
        else:  # pragma: no cover - guarded by state transitions
            raise AssertionError(f"unreachable state: {state}")

        if state != before:
            transitions += 1
        counts[state] += 1
        states_out.append(
            {
                "state_before": before,
                "state": state,
                "transition": f"{before}->{state}",
                "transition_criterion": criterion,
                "high_energy_pass": bool(high_energy),
                "low_energy_pass": bool(low_energy),
                "zcr_pass": bool(zcr_pass),
                "flatness_pass": bool(flatness_pass),
                "high_evidence": high_evidence,
                "low_evidence": low_evidence,
                "active_run": int(active_run),
                "release_run": int(release_run),
            }
        )

    return {
        "schema_version": "1.0",
        "product_version": PRODUCT_VERSIONS["state_machine"],
        "noise_floor": noise,
        "frame_states": states_out,
        "state_counts": counts,
        "transition_count": transitions,
        "confirm_frames": confirm_frames,
        "release_frames": release_frames,
    }
