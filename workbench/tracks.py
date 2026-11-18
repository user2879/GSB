"""Cross-segment event trajectory reconstruction with deterministic decisions."""

from .candidates import _color_close
from .segments import _center_distance


def _link_score(seg_a, seg_b, params):
    color_dist = max(abs(a - b) for a, b in
                     zip(seg_a["color_signature"], seg_b["color_signature"]))
    spatial = _center_distance(seg_a["bbox"], seg_b["bbox"])
    color_term = max(0, 500 - 10 * int(color_dist))
    spatial_term = max(0, 500 - int(5 * spatial))
    return min(1000, color_term + spatial_term)


def build_tracks(seq_name, segments, params):
    """Link segments into trajectories by color signature and spatial continuity."""
    tracks = []
    decisions = []
    ordered = sorted(segments, key=lambda s: (s["start_ms"], s["category"],
                                              s["segment_id"]))
    current = []
    for seg in ordered:
        if not current:
            current = [seg]
            continue
        prev = current[-1]
        gap_frames = seg["frames"][0] - prev["frames"][-1] - 1
        color_ok = _color_close(prev["color_signature"], seg["color_signature"],
                                params.color_match_tolerance)
        spatial = _center_distance(prev["bbox"], seg["bbox"])
        spatial_ok = spatial <= 24.0
        gap_ok = gap_frames <= params.track_max_gap_frames
        if color_ok and spatial_ok and gap_ok and prev["category"] == seg["category"]:
            score = _link_score(prev, seg, params)
            decisions.append({
                "sequence": seq_name,
                "from_segment": prev["segment_id"],
                "to_segment": seg["segment_id"],
                "decision": "link",
                "reason": f"color_ok spatial={round(spatial, 2)} gap_frames={gap_frames}",
                "link_score": score,
            })
            current.append(seg)
        else:
            reason = []
            if not color_ok:
                reason.append("color_mismatch")
            if not spatial_ok:
                reason.append(f"spatial_jump={round(spatial, 2)}")
            if not gap_ok:
                reason.append(f"gap_frames={gap_frames}>track_max_gap_frames")
            if prev["category"] != seg["category"]:
                reason.append("category_change")
            decisions.append({
                "sequence": seq_name,
                "from_segment": prev["segment_id"],
                "to_segment": seg["segment_id"],
                "decision": "break",
                "reason": "+".join(reason),
                "link_score": 0,
            })
            tracks.append(current)
            current = [seg]
    if current:
        tracks.append(current)

    results = []
    for idx, chain in enumerate(tracks):
        link_scores = [d["link_score"] for d in decisions
                       if d["decision"] == "link"
                       and d["from_segment"] in {s["segment_id"] for s in chain}
                       and d["to_segment"] in {s["segment_id"] for s in chain}]
        confidence = (sum(link_scores) // len(link_scores)) if link_scores else 1000
        confidence = max(0, min(1000, int(confidence)))
        results.append({
            "track_id": f"{seq_name}:trk{idx:04d}",
            "category": chain[0]["category"],
            "segment_ids": [s["segment_id"] for s in chain],
            "first_ms": chain[0]["start_ms"],
            "last_ms": chain[-1]["end_ms"],
            "confidence": confidence,
        })
    results.sort(key=lambda t: (-t["confidence"], t["first_ms"], t["track_id"]))
    return results, decisions
