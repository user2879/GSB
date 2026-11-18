#!/usr/bin/env python3
"""Unit tests for candidate suppression, gap-merge chains and track confidence."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workbench.candidates import _suppress_overlaps
from workbench.config import EventParams
from workbench.segments import build_segments
from workbench.tracks import build_tracks

PARAMS = EventParams(
    change_pixel_ratio_threshold=0.01,
    color_match_tolerance=24,
    min_event_duration_ms=80,
    min_gap_ms=100,
    event_categories=("appear", "disappear", "translate", "color_switch"),
    track_max_gap_frames=2,
)


def _cand(cid, frame, bbox, color, area=10, hist_distance=0.1, category="appear"):
    return {
        "candidate_id": cid, "frame_index": frame, "category": category,
        "bbox": bbox, "area": area, "color_signature": color,
        "histogram": [1.0], "hist_distance": hist_distance,
        "suppressed_by": None, "suppress_reason": None,
    }


def test_suppression_overlap():
    keeper = _cand("k", 0, [0, 0, 5, 5], [200, 0, 0], area=20, hist_distance=0.05)
    loser = _cand("l", 0, [2, 2, 8, 8], [0, 0, 200], area=30, hist_distance=0.4)
    survivors, suppressed = _suppress_overlaps([loser, keeper], PARAMS)
    assert [c["candidate_id"] for c in survivors] == ["k"]
    assert suppressed[0]["suppressed_by"] == "k"
    assert suppressed[0]["suppress_reason"] == "overlap_lower_priority"
    print("  PASS suppression records keeper and reason")


def test_suppression_tiebreak_area_then_category():
    a = _cand("a", 0, [0, 0, 5, 5], [200, 0, 0], area=10, hist_distance=0.1,
              category="translate")
    b = _cand("b", 0, [1, 1, 6, 6], [0, 200, 0], area=30, hist_distance=0.1,
              category="appear")
    survivors, suppressed = _suppress_overlaps([a, b], PARAMS)
    assert survivors[0]["candidate_id"] == "b"  # larger area wins the tie
    print("  PASS suppression tie-break by area")


def _prim(frames, color=(200, 0, 0), y=0):
    return {"candidates": [
        _cand(f"c{f}", f, [y, 0, y + 4, 4], list(color)) for f in frames]}


def _timestamps(n, interval=40):
    return [(i * interval, "manifest") for i in range(n)]


def test_gap_merge_chain():
    # Two runs of the same primitive separated by a 40ms gap (< min_gap 100ms).
    primitives = [{"candidates":
                   _prim([0, 1, 2])["candidates"] + _prim([4, 5, 6])["candidates"]}]
    segments, dropped = build_segments("s", primitives, _timestamps(10), PARAMS)
    assert len(segments) == 1, segments
    seg = segments[0]
    assert seg["start_ms"] == 0 and seg["end_ms"] == 240
    assert seg["boundary_criterion"] == "gap_merge"
    assert seg["merge_chain"], "merge chain must record merged parts"
    print("  PASS short-gap segments merged with merge chain")


def test_min_duration_drop():
    segments, dropped = build_segments(
        "s", [_prim([0])], _timestamps(10), PARAMS)
    assert segments == [] and len(dropped) == 1
    assert "min_event_duration_ms" in dropped[0]["reason"]
    print("  PASS short events dropped with recorded reason")


def test_segment_ordering():
    prims = [_prim([0, 1, 2], color=(10, 200, 10), y=20),
             _prim([0, 1, 2], color=(200, 10, 10), y=0)]
    segments, _ = build_segments("s", prims, _timestamps(10), PARAMS)
    starts = [(s["start_ms"], s["category"]) for s in segments]
    assert starts == sorted(starts)
    print("  PASS segments ordered by start_ms then category")


def test_track_break_on_large_gap():
    near = [_prim([0, 1, 2]), _prim([4, 5, 6])]
    segments, _ = build_segments("s", near, _timestamps(20), PARAMS)
    # Force a gap beyond track_max_gap_frames by rebuilding segments manually.
    seg_a = dict(segments[0], segment_id="s:seg0000", frames=[0, 1, 2])
    seg_b = dict(segments[0], segment_id="s:seg0001", frames=[9, 10, 11],
                 start_ms=360, end_ms=440)
    tracks, decisions = build_tracks("s", [seg_a, seg_b], PARAMS)
    assert len(tracks) == 2, tracks
    assert any(d["decision"] == "break" and "gap_frames" in d["reason"]
               for d in decisions)
    print("  PASS track breaks when gap exceeds track_max_gap_frames")


def test_track_confidence_clipped_int():
    segments, _ = build_segments("s", [_prim([0, 1, 2])], _timestamps(10), PARAMS)
    tracks, _ = build_tracks("s", segments, PARAMS)
    conf = tracks[0]["confidence"]
    assert isinstance(conf, int) and 0 <= conf <= 1000
    print("  PASS track confidence is integer per-mille in [0,1000]")


if __name__ == "__main__":
    test_suppression_overlap()
    test_suppression_tiebreak_area_then_category()
    test_gap_merge_chain()
    test_min_duration_drop()
    test_segment_ordering()
    test_track_break_on_large_gap()
    test_track_confidence_clipped_int()
    print("all unit tests passed")
