from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import DetectionParams, MatchConfig
from .models import CanonicalCandidate, Track, TrackFrame, UnifiedFrame
from .util import stable_hash


MATCH_ALGORITHM_VERSION = "iou-greedy-track-pool-v1"


@dataclass
class MatchAudit:
    series_name: str
    created: list[str] = field(default_factory=list)
    extended: list[str] = field(default_factory=list)
    evicted: list[dict[str, Any]] = field(default_factory=list)
    reentered: list[dict[str, Any]] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    recomputed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_name": self.series_name,
            "created": self.created,
            "extended": self.extended,
            "evicted": self.evicted,
            "reentered": self.reentered,
            "closed": self.closed,
            "reused": self.reused,
            "recomputed": self.recomputed,
        }


def iou_q(left: CanonicalCandidate, right: CanonicalCandidate) -> float:
    intersection_width = max(0, min(left.x2_q, right.x2_q) - max(left.x1_q, right.x1_q))
    intersection_height = max(0, min(left.y2_q, right.y2_q) - max(left.y1_q, right.y1_q))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = max(0, left.x2_q - left.x1_q) * max(0, left.y2_q - left.y1_q)
    right_area = max(0, right.x2_q - right.x1_q) * max(0, right.y2_q - right.y1_q)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _candidate_token(candidate: CanonicalCandidate, frame_cache_key: str) -> str:
    return stable_hash(
        {
            "frame_cache_key": frame_cache_key,
            "candidate_identity": candidate.identity_key(),
        }
    )


def _box_iou_frame(
    track: Track, candidate: CanonicalCandidate, decimal_places: int
) -> float:
    if not track.frames:
        return 0.0
    last = track.frames[-1]
    lx1, ly1, lx2, ly2 = last.box
    cx1, cy1, cx2, cy2 = candidate.integer_box(decimal_places)
    intersection_width = max(0, min(lx2, cx2) - max(lx1, cx1))
    intersection_height = max(0, min(ly2, cy2) - max(ly1, cy1))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _new_track_id(series_name: str, candidate: CanonicalCandidate) -> str:
    suffix = stable_hash(
        {
            "series": series_name,
            "class_code": candidate.class_code,
            "frame_index": candidate.frame_index,
            "candidate_identity": candidate.identity_key(),
        }
    )[:12]
    return f"{series_name}:C{candidate.class_code:02d}:F{candidate.frame_index:04d}:{suffix}"


def _make_track_frame(
    candidate: CanonicalCandidate,
    previous: TrackFrame | None,
    *,
    decimal_places: int,
    candidate_token: str,
) -> TrackFrame:
    integer_box = list(candidate.integer_box(decimal_places))
    center_x_q = (candidate.x1_q + candidate.x2_q) // 2
    center_y_q = (candidate.y1_q + candidate.y2_q) // 2
    if previous is None:
        velocity_x_q = 0
        velocity_y_q = 0
    else:
        delta_frames = max(1, candidate.frame_index - previous.frame_index)
        velocity_x_q = int(round((center_x_q - previous.center_x_q) / delta_frames))
        velocity_y_q = int(round((center_y_q - previous.center_y_q) / delta_frames))
    return TrackFrame(
        frame_index=candidate.frame_index,
        box=integer_box,
        score_pm=candidate.score_pm,
        center_x_q=center_x_q,
        center_y_q=center_y_q,
        velocity_x_q=velocity_x_q,
        velocity_y_q=velocity_y_q,
        candidate_key=candidate_token,
    )


def _eviction_rank(track: Track) -> tuple[Any, ...]:
    last = track.frames[-1]
    return (last.score_pm, tuple(-value for value in last.box), track.track_id)


def _track_cache_key(
    track: Track,
    *,
    match_fingerprint: str,
    dependency_frame_keys: dict[int, str],
) -> str:
    return stable_hash(
        {
            "algorithm": MATCH_ALGORITHM_VERSION,
            "match_fingerprint": match_fingerprint,
            "series_name": track.series_name,
            "class_code": track.class_code,
            "frames": [frame.to_dict() for frame in track.frames],
            "dependency_frame_keys": {
                str(key): value for key, value in sorted(dependency_frame_keys.items())
            },
        }
    )


def _finalize_track(
    track: Track,
    *,
    match: MatchConfig,
    frame_keys_by_index: dict[int, str],
) -> Track:
    if not track.frames:
        return track
    dependency = {
        frame_index: frame_keys_by_index.get(frame_index, "missing-frame-key")
        for frame_index in range(track.first_frame, track.last_frame + 1)
    }
    track.dependency_frame_keys = dependency
    track.cache_key = _track_cache_key(
        track,
        match_fingerprint=match.fingerprint,
        dependency_frame_keys=dependency,
    )
    return track


def match_series(
    *,
    series_name: str,
    frames: list[UnifiedFrame],
    match: MatchConfig,
    detection: DetectionParams,
    reusable_tracks: list[Track] | None = None,
) -> tuple[list[Track], MatchAudit]:
    reusable_tracks = reusable_tracks or []
    audit = MatchAudit(series_name=series_name)
    frame_keys_by_index = {
        frame.frame_index: frame.candidate_key for frame in frames
    }
    consumed: set[tuple[int, str]] = set()
    all_tracks: dict[str, Track] = {}
    for track in reusable_tracks:
        if track.series_name != series_name:
            continue
        cloned = Track.from_dict(track.to_dict())
        cloned.reused = True
        all_tracks[cloned.track_id] = cloned
        audit.reused.append(cloned.track_id)
        for frame in cloned.frames:
            consumed.add((frame.frame_index, frame.candidate_key))

    candidates_by_class_frame: dict[int, dict[int, list[CanonicalCandidate]]] = {}
    for frame in sorted(frames, key=lambda item: item.frame_index):
        for candidate in frame.candidates:
            token = _candidate_token(candidate, frame.candidate_key)
            if (frame.frame_index, token) in consumed:
                continue
            candidates_by_class_frame.setdefault(candidate.class_code, {}).setdefault(
                frame.frame_index, []
            ).append(candidate)

    for class_code in sorted(candidates_by_class_frame):
        active: dict[str, Track] = {}
        evicted: dict[str, Track] = {}
        closed: dict[str, Track] = {}
        class_frames = candidates_by_class_frame[class_code]
        for frame_index in sorted(class_frames):
            # Tracks whose last observation is too old are closed before matching.
            for track_id in sorted(list(active)):
                track = active[track_id]
                if frame_index - track.last_frame - 1 > match.max_gap_frames:
                    closed[track_id] = active.pop(track_id)
                    audit.closed.append(track_id)
            for track_id in sorted(list(evicted)):
                track = evicted[track_id]
                if frame_index - track.last_frame - 1 > match.max_gap_frames:
                    closed[track_id] = evicted.pop(track_id)

            candidates = sorted(
                class_frames[frame_index],
                key=lambda item: (
                    -item.score_micro,
                    item.x1_q,
                    item.y1_q,
                    item.x2_q,
                    item.y2_q,
                    item.source_ordinal,
                ),
            )
            frame_by_index = next(
                item for item in frames if item.frame_index == frame_index
            )
            tokens = {
                id(candidate): _candidate_token(candidate, frame_by_index.candidate_key)
                for candidate in candidates
            }
            assignments: dict[int, str] = {}
            pair_options: list[tuple[float, str, int, str]] = []
            for candidate_index, candidate in enumerate(candidates):
                token = tokens[id(candidate)]
                for track_id, track in active.items():
                    score = _box_iou_frame(track, candidate, match.fixed_decimal_places)
                    if score >= match.iou_threshold:
                        pair_options.append((-score, track_id, candidate_index, token))
            pair_options.sort(key=lambda item: (item[0], item[1], item[2]))
            used_tracks: set[str] = set()
            for negative_iou, track_id, candidate_index, token in pair_options:
                if track_id in used_tracks or candidate_index in assignments:
                    continue
                assignments[candidate_index] = track_id
                used_tracks.add(track_id)
                track = active[track_id]
                candidate = candidates[candidate_index]
                track.frames.append(
                    _make_track_frame(
                        candidate,
                        track.frames[-1],
                        decimal_places=match.fixed_decimal_places,
                        candidate_token=token,
                    )
                )
                audit.extended.append(track_id)

            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in assignments:
                    continue
                token = tokens[id(candidate)]
                revival_options: list[tuple[float, str]] = []
                for track_id, track in evicted.items():
                    score = _box_iou_frame(track, candidate, match.fixed_decimal_places)
                    if score >= match.iou_threshold:
                        revival_options.append((-score, track_id))
                if revival_options:
                    revival_options.sort()
                    _, track_id = revival_options[0]
                    track = evicted.pop(track_id)
                    track.reentered_pool = True
                    track.frames.append(
                        _make_track_frame(
                            candidate,
                            track.frames[-1],
                            decimal_places=match.fixed_decimal_places,
                            candidate_token=token,
                        )
                    )
                    active[track_id] = track
                    audit.reentered.append(
                        {
                            "track_id": track_id,
                            "frame_index": frame_index,
                            "eviction_count": track.eviction_count,
                        }
                    )
                else:
                    track_id = _new_track_id(series_name, candidate)
                    suffix = 1
                    base_id = track_id
                    while track_id in all_tracks or track_id in active or track_id in evicted:
                        suffix += 1
                        track_id = f"{base_id}:{suffix}"
                    track = Track(
                        track_id=track_id,
                        series_name=series_name,
                        class_code=class_code,
                        class_name=candidate.class_name,
                        frames=[
                            _make_track_frame(
                                candidate,
                                None,
                                decimal_places=match.fixed_decimal_places,
                                candidate_token=token,
                            )
                        ],
                    )
                    all_tracks[track_id] = track
                    active[track_id] = track
                    audit.created.append(track_id)

            while len(active) > detection.pool_capacity:
                track_id = min(active, key=lambda item: _eviction_rank(active[item]))
                track = active.pop(track_id)
                track.eviction_count += 1
                evicted[track_id] = track
                audit.evicted.append(
                    {
                        "track_id": track_id,
                        "frame_index": frame_index,
                        "last_score_pm": track.frames[-1].score_pm,
                        "last_box": track.frames[-1].box,
                    }
                )

        for track_id, track in active.items():
            all_tracks.setdefault(track_id, track)
        for track_id, track in evicted.items():
            all_tracks.setdefault(track_id, track)
        for track_id, track in closed.items():
            all_tracks.setdefault(track_id, track)

    finalized: list[Track] = []
    for track_id in sorted(all_tracks):
        track = all_tracks[track_id]
        if track.reused:
            finalized.append(track)
            continue
        finalized.append(
            _finalize_track(
                track,
                match=match,
                frame_keys_by_index=frame_keys_by_index,
            )
        )
        audit.recomputed.append(track_id)
    finalized.sort(key=lambda item: item.sort_key())
    audit.reused.sort()
    audit.recomputed.sort()
    return finalized, audit


def track_reusable(
    track: Track,
    *,
    current_frame_keys: dict[int, str],
    match: MatchConfig,
) -> bool:
    if track.cache_key == "":
        return False
    for frame_index, expected in track.dependency_frame_keys.items():
        if current_frame_keys.get(frame_index) != expected:
            return False
    return True


def refresh_track_metadata(
    track: Track,
    *,
    match: MatchConfig,
    frame_keys_by_index: dict[int, str],
) -> Track:
    return _finalize_track(
        track,
        match=match,
        frame_keys_by_index=frame_keys_by_index,
    )


def _latest_frame_at_or_before(track: Track, frame_index: int) -> TrackFrame | None:
    selected: TrackFrame | None = None
    for frame in track.frames:
        if frame.frame_index <= frame_index:
            selected = frame
        else:
            break
    return selected


def recompute_pool_metadata(
    tracks: list[Track],
    *,
    match: MatchConfig,
    candidate_pool_capacity: int,
) -> None:
    """Replay only the deterministic active-pool lifecycle on fixed tracks.

    IoU assignment and track composition are not recomputed. This makes eviction
    counters and re-entry flags independent of whether a track came from cache.
    """

    groups: dict[tuple[str, int], list[Track]] = {}
    for track in tracks:
        groups.setdefault((track.series_name, track.class_code), []).append(track)
        track.eviction_count = 0
        track.reentered_pool = False
    for (series_name, class_code), class_tracks in sorted(groups.items()):
        observations: dict[int, list[Track]] = {}
        for track in class_tracks:
            for frame in track.frames:
                observations.setdefault(frame.frame_index, []).append(track)
        active: dict[str, Track] = {}
        evicted: dict[str, Track] = {}
        closed: set[str] = set()
        for frame_index in sorted(observations):
            for track_id in sorted(list(active)):
                track = active[track_id]
                if frame_index - track.last_frame - 1 > match.max_gap_frames:
                    closed.add(track_id)
                    active.pop(track_id)
            for track_id in sorted(list(evicted)):
                track = evicted[track_id]
                if frame_index - track.last_frame - 1 > match.max_gap_frames:
                    closed.add(track_id)
                    evicted.pop(track_id)
            for track in sorted(observations[frame_index], key=lambda item: item.track_id):
                if track.track_id in active:
                    continue
                if track.track_id in evicted:
                    track.reentered_pool = True
                    active[track.track_id] = evicted.pop(track.track_id)
                elif track.track_id not in closed:
                    active[track.track_id] = track
            while len(active) > candidate_pool_capacity:
                def rank(track_id: str) -> tuple[Any, ...]:
                    track = active[track_id]
                    latest = _latest_frame_at_or_before(track, frame_index)
                    if latest is None:
                        return (-1, (), track_id)
                    return (latest.score_pm, tuple(-value for value in latest.box), track_id)

                track_id = min(active, key=rank)
                track = active.pop(track_id)
                track.eviction_count += 1
                evicted[track_id] = track
