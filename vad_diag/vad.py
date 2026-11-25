"""Dual-threshold hysteresis Voice Activity Detection.

Each frame gets an energy gate and a spectral-flatness gate.  A speech region
*starts* only when both metrics reach their **high** thresholds, and is
*maintained* while both stay above their **low** thresholds.  The resulting raw
runs then pass through deterministic minimum-speech / minimum-silence
post-processing with explicit, documented tie-breaking.

Every produced region records:

* the exact set of frame indices it covers,
* the set of segment ids it references (frames may straddle boundaries),
* the high/low gate decisions and the boundary tie metadata,

so a boundary move after a late/replacement segment can be traced in the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from . import dsp
from .config import Config
from .timeline import Timeline


@dataclass(frozen=True)
class FrameRecord:
    index: int
    start: int
    end: int
    energy: float
    flatness: float
    high_gate: bool
    low_gate: bool
    segment_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "energy": self.energy,
            "flatness": self.flatness,
            "high_gate": self.high_gate,
            "low_gate": self.low_gate,
            "segment_ids": list(self.segment_ids),
        }


@dataclass(frozen=True)
class SpeechRegion:
    region_id: str
    frame_start: int
    frame_end: int                 # exclusive
    sample_start: int
    sample_end: int                # exclusive
    frame_indices: Tuple[int, ...]
    segment_ids: Tuple[str, ...]
    gate_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "sample_start": self.sample_start,
            "sample_end": self.sample_end,
            "frame_indices": list(self.frame_indices),
            "segment_ids": sorted(self.segment_ids),
            "gate_meta": self.gate_meta,
        }


def build_frames(timeline: Timeline, config: Config
                 ) -> Tuple[List[FrameRecord], List[int]]:
    """Compute frame records; return frames plus the dropped trailing samples.
    """
    total = timeline.total_samples
    intervals = dsp.frame_intervals(
        total, config.frame_length, config.frame_shift)
    flat = None
    frames: List[FrameRecord] = []
    for index, (start, end) in enumerate(intervals):
        if flat is None:
            flat = timeline.samples()
        window = flat[start:end]
        energy = dsp.frame_energy(window)
        flatness = dsp.spectral_flatness(window)
        # Speech-like = energetic OR tonal.  Tonal speech has a *low* spectral
        # flatness (concentrated spectrum), while noise is flat (high value),
        # so the flatness gate fires when flatness is at/below its threshold.
        high_gate = (energy >= config.energy_high
                     or flatness <= config.flatness_high)
        low_gate = (energy >= config.energy_low
                    or flatness <= config.flatness_low)
        seg_ids, _ = timeline.segments_for_interval(start, end)
        frames.append(FrameRecord(
            index=index, start=start, end=end, energy=energy,
            flatness=flatness, high_gate=high_gate, low_gate=low_gate,
            segment_ids=tuple(seg_ids)))
    last_end = intervals[-1][1] if intervals else 0
    dropped = list(range(last_end, total))
    return frames, dropped


def hysteresis_runs(frames: Sequence[FrameRecord]) -> List[Tuple[int, int]]:
    """High-threshold start, low-threshold maintenance (raw speech runs)."""
    runs: List[Tuple[int, int]] = []
    active = False
    start = 0
    for frame in frames:
        if not active:
            if frame.high_gate:
                active = True
                start = frame.index
        else:
            if not frame.low_gate:
                runs.append((start, frame.index))  # exclusive end
                active = False
    if active:
        runs.append((start, frames[-1].index + 1))
    return runs


def postprocess(runs: Sequence[Tuple[int, int]], min_speech_frames: int,
                min_silence_frames: int) -> List[Tuple[int, int]]:
    """Deterministic minimum-length filtering and gap merging.

    Order of operations (each step is total and order-preserving):

    1. Drop speech runs shorter than ``min_speech_frames``.
    2. Merge adjacent surviving runs whose separating silence is shorter than
       ``min_silence_frames``; ties on the exact boundary frame are resolved by
       keeping the earlier frame (lowest-index) inside the merged region.
    """
    kept = [r for r in runs if (r[1] - r[0]) >= min_speech_frames]
    if not kept:
        return []
    merged: List[Tuple[int, int]] = [kept[0]]
    for start, end in kept[1:]:
        prev_start, prev_end = merged[-1]
        silence = start - prev_end
        if silence < min_silence_frames:
            # Merge; boundary frame at prev_end stays attributed to the merged
            # region (lowest-index tie-break) and the gap frames are absorbed.
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def run_vad(timeline: Timeline, config: Config
            ) -> Tuple[List[SpeechRegion], List[FrameRecord], Dict[str, Any]]:
    frames, dropped = build_frames(timeline, config)
    raw_runs = hysteresis_runs(frames)
    final_runs = postprocess(
        raw_runs, config.min_speech_frames, config.min_silence_frames)

    regions: List[SpeechRegion] = []
    for ordinal, (fstart, fend) in enumerate(final_runs):
        covered = tuple(range(fstart, fend))
        sample_start = frames[fstart].start
        sample_end = frames[fend - 1].end
        seg_set = sorted({
            sid for fi in covered for sid in frames[fi].segment_ids})
        # Record the exact gate that opened and closed the region.
        first = frames[fstart]
        last = frames[fend - 1]
        gate_meta = {
            "tie_rule": "lowest_index_wins",
            "start_high_gate": first.high_gate,
            "end_low_gate": last.low_gate,
            "raw_run_count": len(raw_runs),
            "merged": len(raw_runs) != len(final_runs),
        }
        regions.append(SpeechRegion(
            region_id=f"R{ordinal:04d}",
            frame_start=fstart, frame_end=fend,
            sample_start=sample_start, sample_end=sample_end,
            frame_indices=covered, segment_ids=tuple(seg_set),
            gate_meta=gate_meta))

    metadata = {
        "frame_count": len(frames),
        "dropped_trailing_samples": dropped,
        "raw_runs": [list(r) for r in raw_runs],
        "final_runs": [list(r) for r in final_runs],
        "tie_rule": "lowest_index_wins",
        "min_speech_frames": config.min_speech_frames,
        "min_silence_frames": config.min_silence_frames,
    }
    return regions, frames, metadata
