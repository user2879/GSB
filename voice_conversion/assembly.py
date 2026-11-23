"""Deterministic segment assembly with linear crossfades at the junctions."""
from __future__ import annotations

import numpy as np

from .audio import saturate_int16
from .config import PipelineConfig


def assemble_segments(segment_pcms: dict[int, np.ndarray],
                      frame_starts: dict[int, int], config: PipelineConfig
                      ) -> np.ndarray:
    """Place segment PCM on the frame-shift grid and crossfade overlaps.

    Each segment buffer was produced independently; only assembly touches more
    than one segment.  Overlapping tail/head regions use equal-power-free linear
    ramps whose length is the configured crossfade in frames.
    """
    if not segment_pcms:
        return np.zeros(0, dtype="<i2")
    fs = config.frame_shift
    fade = config.crossfade_frames * fs
    ordered = sorted(segment_pcms)
    last_end = max(
        frame_starts[idx] * fs + len(segment_pcms[idx]) for idx in ordered
    )
    total = np.zeros(last_end, dtype=np.float64)
    cursor_end = -1
    for idx in ordered:
        pcm = segment_pcms[idx].astype(np.float64)
        start = frame_starts[idx] * fs
        end = start + len(pcm)
        if cursor_end < 0 or start >= cursor_end or fade <= 0:
            total[start:end] += pcm
        else:
            overlap = min(cursor_end - start, fade, len(pcm))
            ramp_out = np.linspace(1.0, 0.0, overlap, endpoint=False)
            ramp_in = np.linspace(0.0, 1.0, overlap, endpoint=False)
            total[start:start + overlap] *= ramp_out
            total[start:start + overlap] += pcm[:overlap] * ramp_in
            total[start + overlap:end] += pcm[overlap:]
        cursor_end = max(cursor_end, end)
    return saturate_int16(total[: max(cursor_end, 0)])
