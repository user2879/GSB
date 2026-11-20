"""Stable prefix / mutable tail detection over merged frame labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .decoder import WordToken
from .streaming import ChunkResult


@dataclass(frozen=True)
class StabilityResult:
    stable_until_frame: int  # exclusive; frames [0, stable_until_frame) are stable
    boundary_runs: List[Dict[str, Any]]

    @property
    def is_fully_stable(self) -> bool:
        return self.stable_until_frame >= 0 and not self.boundary_runs


def evaluate_stability(chunk_results: Sequence[ChunkResult], n_frames: int,
                       min_consistent_frames: int) -> StabilityResult:
    """Walk chunk boundaries; a run of >= threshold equal labels fixes the prefix.

    The first boundary whose longest equal-label run is shorter than the
    threshold ends the stable prefix; if every boundary passes, the whole
    stream is stable.
    """
    ordered = sorted((result for result in chunk_results if not result.failed),
                     key=lambda item: item.index)
    runs: List[Dict[str, Any]] = []
    first_failing_frame: int | None = None
    for left, right in zip(ordered, ordered[1:]):
        boundary = right.frame_span[0]
        overlap_end = min(left.frame_span[1], right.frame_span[1])
        if boundary >= overlap_end:
            continue
        left_labels = left.labels[boundary - left.frame_span[0]:overlap_end - left.frame_span[0]]
        right_labels = right.labels[boundary - right.frame_span[0]:overlap_end - right.frame_span[0]]
        longest = 0
        current = 0
        for left_label, right_label in zip(left_labels, right_labels):
            if left_label == right_label:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        record = {
            "boundary_frame": boundary,
            "overlap_frames": overlap_end - boundary,
            "longest_consistent_run": longest,
            "stable": longest >= min_consistent_frames,
        }
        runs.append(record)
        if not record["stable"] and first_failing_frame is None:
            first_failing_frame = boundary
    if first_failing_frame is None:
        return StabilityResult(stable_until_frame=n_frames, boundary_runs=runs)
    return StabilityResult(stable_until_frame=first_failing_frame, boundary_runs=runs)


def split_tokens(tokens: List[WordToken], stable_until_frame: int
                 ) -> Tuple[List[WordToken], List[WordToken]]:
    prefix: List[WordToken] = []
    tail: List[WordToken] = []
    for token in tokens:
        if token.end_frame < stable_until_frame:
            prefix.append(token)
        else:
            tail.append(token)
    return prefix, tail
