"""Class-constrained IoU matching, track linking and pool eviction."""
from __future__ import annotations

from dataclasses import dataclass

from .config import MatchConfig
from .models import Candidate, FrameCandidates, Track


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchEviction:
    sequence: str
    frame: int
    evicted_box: tuple[int, int, int, int]
    evicted_score: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "frame": self.frame,
            "evicted_box": list(self.evicted_box),
            "evicted_score": self.evicted_score,
            "reason": self.reason,
        }


def _box(candidate: Candidate):
    return candidate.int_box_tuple()


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _greedy_pairs(active: list[Track], candidates: list[Candidate], threshold: float):
    options = []
    for track in active:
        predicted = (
            track.center_x + track.vx,
            track.center_y + track.vy,
        )
        predicted_box = None
        for candidate in candidates:
            if candidate.class_code != track.class_code:
                continue
            box = _box(candidate)
            score = iou(track.boxes[-1], box)
            if score >= threshold:
                center = _center(box)
                dist = abs(center[0] - predicted[0]) + abs(center[1] - predicted[1])
                options.append((score, -dist, track.track_id, candidate, track))
    options.sort(key=lambda item: (-item[0], item[1], item[2], item[3].int_box_tuple()))
    used_tracks: set[str] = set()
    used_candidates: set[tuple] = set()
    pairs: list[tuple[Track, Candidate, float]] = []
    for score, _dist, _tid, candidate, track in options:
        box = _box(candidate)
        if track.track_id in used_tracks or box in used_candidates:
            continue
        used_tracks.add(track.track_id)
        used_candidates.add(box)
        pairs.append((track, candidate, score))
    return pairs


def match_sequence(
    sequence: str,
    frames: list[FrameCandidates],
    config: MatchConfig,
    pool_capacity: int,
) -> tuple[list[Track], list[MatchEviction]]:
    """Link candidates across frames for one sequence.

    Frames must already be restricted to successful frames in ascending order;
    failed frames never reach this function and therefore never enter tracks.
    """
    active: list[Track] = []
    closed: list[Track] = []
    evictions: list[MatchEviction] = []
    counter = 0
    pooled: list[Candidate] = []

    for frame in sorted(frames, key=lambda f: f.frame):
        frame_no = frame.frame
        candidates = sorted(frame.candidates, key=lambda c: _box(c))
        pairs = _greedy_pairs(active, candidates, config.iou_threshold)
        paired_boxes = {_box(candidate) for _, candidate, _ in pairs}
        paired_tracks = {track.track_id for track, _, _ in pairs}

        for track, candidate, _ in pairs:
            box = _box(candidate)
            center = _center(box)
            if track.frames:
                dt = frame_no - track.last_seen
                if dt > 0:
                    new_vx = (center[0] - track.center_x) / dt
                    new_vy = (center[1] - track.center_y) / dt
                    track.vx = 0.7 * track.vx + 0.3 * new_vx
                    track.vy = 0.7 * track.vy + 0.3 * new_vy
            track.center_x, track.center_y = center
            track.frames.append(frame_no)
            track.boxes.append(box)
            track.scores.append(candidate.int_score_value())
            track.last_seen = frame_no

        surviving: list[Track] = []
        for track in active:
            if track.track_id in paired_tracks:
                surviving.append(track)
            elif frame_no - track.last_seen > config.max_gap_frames:
                track.closed = True
                closed.append(track)
            else:
                surviving.append(track)
        active = surviving

        for candidate in candidates:
            if _box(candidate) not in paired_boxes:
                counter += 1
                track = Track(
                    track_id=f"{sequence}-T{counter:04d}",
                    class_code=candidate.class_code,
                    sequence=sequence,
                    frames=[frame_no],
                    boxes=[_box(candidate)],
                    scores=[candidate.int_score_value()],
                    center_x=0.0,
                    center_y=0.0,
                    vx=0.0,
                    vy=0.0,
                    last_seen=frame_no,
                )
                center = _center(track.boxes[-1])
                track.center_x, track.center_y = center
                active.append(track)

        pooled = _update_pool(
            pooled, candidates, active, pool_capacity, sequence, frame_no, evictions
        )

    for track in active:
        track.closed = True
        closed.append(track)
    closed.sort(key=lambda t: t.track_id)
    return closed, evictions


def _eviction_key(candidate: Candidate):
    # Score descending; ties broken by coordinate dictionary order ascending,
    # so the key is fully deterministic and replayable.
    return (-candidate.score, candidate.x1, candidate.y1, candidate.x2, candidate.y2)


def _update_pool(pooled, candidates, active, capacity, sequence, frame_no, evictions):
    # Tracks feed back into the match pool: candidates already consumed by an
    # active track are retained preferentially at equal rank; the final
    # tie-breaker remains the mandated score/coordinate ordering, keeping the
    # result a pure deterministic function of frame candidates + track state.
    active_boxes = {box for track in active for box in track.boxes}
    merged: dict[tuple, Candidate] = {}
    for candidate in candidates:
        merged[_box(candidate)] = candidate
    for candidate in pooled:
        merged.setdefault(_box(candidate), candidate)
    entries = list(merged.values())
    if len(entries) > capacity:
        entries.sort(
            key=lambda c: (
                0 if _box(c) in active_boxes else 1,
                _eviction_key(c),
            )
        )
        removed = entries[: len(entries) - capacity]
        entries = entries[len(entries) - capacity :]
        for candidate in removed:
            evictions.append(
                MatchEviction(
                    sequence=sequence,
                    frame=frame_no,
                    evicted_box=_box(candidate),
                    evicted_score=candidate.int_score_value(),
                    reason="match_pool_score_desc_coord_lex",
                )
            )
    return entries
