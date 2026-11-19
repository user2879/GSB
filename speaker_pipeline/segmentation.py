"""Dual-threshold hysteresis VAD, segment cleanup and overlap marking."""

import numpy as np

from .audioio import SAMPLE_RATE
from .errors import SegmentationError
from .fingerprint import round6


def _vad_states(frames, config):
    energy_on = float(config.data["energy_threshold"])
    flat_on = float(config.data["spectral_flatness_threshold"])
    # Hysteresis: exit thresholds are 20% looser than entry thresholds.
    energy_off = energy_on * 0.8
    flat_off = min(0.95, flat_on + (1.0 - flat_on) * 0.2)
    active = False
    states = []
    for frame in frames:
        speech_like = (
            frame["energy"] >= energy_on
            and frame["spectral_flatness"] <= flat_on
        )
        if not active and speech_like:
            active = True
        elif active:
            still_speech = (
                frame["energy"] >= energy_off
                and frame["spectral_flatness"] <= flat_off
            )
            if not still_speech:
                active = False
        states.append(active)
    return states


def _runs(states):
    runs = []
    start = None
    for index, state in enumerate(states + [False]):
        if state and start is None:
            start = index
        elif not state and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def _ms_from_frame(frame_index, boundary, frame_shift):
    sample = boundary * frame_shift
    return round6(sample * 1000.0 / SAMPLE_RATE)


def segment_file(feature_product, config):
    """Turn frame records into ordered speech/silence interval records."""
    frames = feature_product["frames"]
    frame_shift = feature_product["frame_shift"]
    frame_len = feature_product["frame_len"]
    states = _vad_states(frames, config)
    speech_runs = _runs(states)

    min_speech_frames = max(
        1,
        int(round(config.data["min_speech_ms"] / config.data["frame_shift_ms"])),
    )
    min_silence_frames = int(
        round(config.data["min_silence_ms"] / config.data["frame_shift_ms"])
    )

    # Drop speech runs shorter than the minimum speech duration.
    kept = [
        run
        for run in speech_runs
        if (run[1] - run[0] + 1) >= min_speech_frames
    ]

    # Merge neighboring runs separated by a short silence.
    merged = []
    for run in kept:
        if merged:
            gap = run[0] - merged[-1][1] - 1
            if gap <= min_silence_frames:
                merged[-1] = (merged[-1][0], run[1])
                continue
        merged.append(list(run))

    intervals = []
    failures = []
    for order, (lo, hi) in enumerate(merged):
        try:
            segment_frames = frames[lo : hi + 1]
            voiceprints = np.asarray(
                [frame["voiceprint"] for frame in segment_frames],
                dtype=np.float64,
            )
            mean_voiceprint = np.mean(voiceprints, axis=0)
            if not np.all(np.isfinite(mean_voiceprint)):
                raise SegmentationError(
                    f"non-finite mean voiceprint at interval {order}",
                    stage="segmentation",
                    interval_index=order,
                )
            intervals.append(
                {
                    "interval_index": order,
                    "start_frame": int(lo),
                    "end_frame": int(hi),
                    "start_ms": _ms_from_frame(lo, lo, frame_shift),
                    "end_ms": round6(
                        (hi * frame_shift + frame_len) * 1000.0 / SAMPLE_RATE
                    ),
                    "overlap": False,
                    "mean_voiceprint": [round6(v) for v in mean_voiceprint],
                }
            )
        except SegmentationError as exc:
            failures.append(exc.to_record())

    failures.extend(_mark_overlaps(intervals, config))
    intervals.sort(key=lambda item: (item["start_ms"], item["end_frame"]))
    for new_index, item in enumerate(intervals):
        item["interval_index"] = new_index
    return {
        "intervals": intervals,
        "failures": failures,
        "frame_shift": frame_shift,
        "frame_len": frame_len,
    }


def _mark_overlaps(intervals, config):
    """Mark suspected overlaps; isolate a failure on one interval pair."""
    failures = []
    threshold = float(config.data["overlap_similarity_threshold"])
    for left, right in zip(intervals, intervals[1:]):
        try:
            sim = cosine_similarity(
                np.asarray(left["mean_voiceprint"], dtype=np.float64),
                np.asarray(right["mean_voiceprint"], dtype=np.float64),
            )
            if not np.isfinite(sim):
                raise SegmentationError(
                    f"non-finite overlap similarity at interval "
                    f"{right['interval_index']}",
                    stage="segmentation",
                    interval_index=right["interval_index"],
                )
        except SegmentationError as exc:
            failures.append(exc.to_record())
            continue
        if sim >= threshold:
            left["overlap"] = True
            right["overlap"] = True
            pair = [left["interval_index"], right["interval_index"]]
            left.setdefault("overlap_pairs", []).append(pair)
            right.setdefault("overlap_pairs", []).append(pair)
    return failures


def cosine_similarity(vec_a, vec_b):
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)
