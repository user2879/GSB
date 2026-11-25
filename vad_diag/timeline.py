"""Reconstruct a global sample timeline from independently arriving segments.

The *content* timeline is ordered by ``capture_ms`` (never by arrival time), so
late or out-of-order delivery leaves the underlying audio byte-identical.  Each
segment occupies a contiguous, non-overlapping global sample interval; windows
and frames that straddle segment boundaries record exactly which segments they
reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .fingerprint import combine
from .segments import Segment


@dataclass(frozen=True)
class SegmentSpan:
    segment_id: str
    start: int          # global inclusive start sample
    end: int            # global exclusive end sample
    capture_ms: int
    arrival_ms: int
    process_ms: int

    def contains(self, global_index: int) -> bool:
        return self.start <= global_index < self.end


class Timeline:
    """Ordered, immutable view of the accepted segments."""

    def __init__(self, segments: Sequence[Segment]):
        # Deterministic content order: capture_ms, then segment_id tiebreak,
        # then arrival_ms to keep the ordering total and reproducible.
        ordered = sorted(
            segments,
            key=lambda s: (s.capture_ms, s.segment_id, s.arrival_ms))
        self._segments: List[Segment] = ordered
        self._spans: List[SegmentSpan] = []
        self._by_id: Dict[str, Segment] = {}
        cursor = 0
        for seg in ordered:
            if seg.segment_id in self._by_id:
                raise ValueError(
                    f"duplicate segment id on timeline: {seg.segment_id}")
            start = cursor
            end = cursor + seg.num_samples
            self._spans.append(SegmentSpan(
                seg.segment_id, start, end, seg.capture_ms,
                seg.arrival_ms, seg.process_ms))
            self._by_id[seg.segment_id] = seg
            cursor = end
        self._total = cursor
        self._fingerprint = combine(
            ["timeline", [s.content_fingerprint() for s in ordered]])

    @property
    def segments(self) -> List[Segment]:
        return list(self._segments)

    @property
    def segment_ids(self) -> List[str]:
        return [s.segment_id for s in self._segments]

    @property
    def total_samples(self) -> int:
        return self._total

    @property
    def spans(self) -> List[SegmentSpan]:
        return list(self._spans)

    def fingerprint(self) -> str:
        return self._fingerprint

    def get(self, segment_id: str) -> Segment:
        return self._by_id[segment_id]

    def has(self, segment_id: str) -> bool:
        return segment_id in self._by_id

    def samples(self) -> List[int]:
        """All PCM samples in global (capture) order."""
        out: List[int] = []
        for seg in self._segments:
            out.extend(seg.samples)
        return out

    def span_of(self, segment_id: str) -> SegmentSpan:
        for span in self._spans:
            if span.segment_id == segment_id:
                return span
        raise KeyError(segment_id)

    def segments_for_interval(self, start: int, end: int
                              ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Return the segment ids overlapped by a global sample interval.

        Also returns the per-segment global ``[gstart, gend)`` intersections so
        callers can prove no out-of-window data is read.
        """
        ids: List[str] = []
        intersections: List[Tuple[int, int]] = []
        for span in self._spans:
            overlap_start = max(start, span.start)
            overlap_end = min(end, span.end)
            if overlap_start < overlap_end:
                ids.append(span.segment_id)
                intersections.append((overlap_start, overlap_end))
        return ids, intersections

    def interval_samples(self, start: int, end: int) -> List[int]:
        """Read exactly the global samples in ``[start, end)``."""
        if start < 0 or end > self._total or start > end:
            raise IndexError(
                f"sample interval [{start}, {end}) out of timeline "
                f"[0, {self._total})")
        flat = self.samples()
        return flat[start:end]

    def localize(self, global_index: int) -> Tuple[str, int]:
        """Map a global sample index to ``(segment_id, local_offset)``."""
        for span in self._spans:
            if span.contains(global_index):
                return span.segment_id, global_index - span.start
        raise IndexError(f"global sample {global_index} not on timeline")
