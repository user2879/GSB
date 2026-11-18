"""Timeline event boundary detection: merge per-frame candidates into event segments."""

from .candidates import _color_close


def _center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _center_distance(b1, b2):
    c1, c2 = _center(b1), _center(b2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def group_primitives(candidates, params):
    """Greedy deterministic grouping of candidates into per-primitive chains."""
    primitives = []
    for cand in sorted(candidates,
                       key=lambda c: (c["frame_index"], c["bbox"][0], c["bbox"][1])):
        best = None
        for prim in primitives:
            last = prim["candidates"][-1]
            if cand["frame_index"] - last["frame_index"] > params.track_max_gap_frames + 1:
                continue
            if not _color_close(cand["color_signature"], last["color_signature"],
                                params.color_match_tolerance):
                continue
            limit = max(8.0, max(last["bbox"][2] - last["bbox"][0],
                                 last["bbox"][3] - last["bbox"][1]) * 1.5)
            if _center_distance(cand["bbox"], last["bbox"]) > limit:
                continue
            dist = _center_distance(cand["bbox"], last["bbox"])
            if best is None or dist < best[0]:
                best = (dist, prim)
        if best is None:
            primitives.append({"candidates": [cand]})
        else:
            best[1]["candidates"].append(cand)
    return primitives


def build_segments(seq_name, primitives, timestamps, params):
    """Merge consecutive candidates into event segments with boundary criteria."""
    raw_segments = []
    for prim in primitives:
        run = [prim["candidates"][0]]
        for cand in prim["candidates"][1:]:
            if cand["frame_index"] - run[-1]["frame_index"] <= 1:
                run.append(cand)
            else:
                raw_segments.append(run)
                run = [cand]
        raw_segments.append(run)

    segments = []
    dropped = []
    for run in raw_segments:
        start_ms = timestamps[run[0]["frame_index"]][0]
        end_ms = timestamps[run[-1]["frame_index"]][0]
        duration = end_ms - start_ms
        if duration < params.min_event_duration_ms:
            dropped.append({
                "frames": [c["frame_index"] for c in run],
                "reason": f"duration {duration}ms < min_event_duration_ms",
            })
            continue
        strengths = [c.get("area", 0) for c in run]
        segments.append({
            "category": run[0]["category"],
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_frames": len(run),
            "evidence_strength": min(1000, int(round(
                1000.0 * sum(strengths) / max(1, len(strengths)) / 64.0))),
            "boundary_criterion": "duration_threshold",
            "timestamp_source": (
                "interpolated"
                if timestamps[run[0]["frame_index"]][1] == "interpolated"
                or timestamps[run[-1]["frame_index"]][1] == "interpolated"
                else "manifest"),
            "frames": [c["frame_index"] for c in run],
            "color_signature": run[0]["color_signature"],
            "bbox": run[-1]["bbox"],
            "merge_chain": [],
        })

    segments.sort(key=lambda s: (s["start_ms"], s["category"]))
    merged = _merge_short_gaps(segments, params)
    for i, seg in enumerate(merged):
        seg["segment_id"] = f"{seq_name}:seg{i:04d}"
    return merged, dropped


def _merge_short_gaps(segments, params):
    """Merge adjacent same-signature segments whose gap is below min_gap_ms."""
    merged = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            gap = seg["start_ms"] - prev["end_ms"]
            if (0 <= gap < params.min_gap_ms
                    and prev["category"] == seg["category"]
                    and _color_close(prev["color_signature"], seg["color_signature"],
                                     params.color_match_tolerance)):
                chain = prev["merge_chain"] or [f"base@{prev['start_ms']}"]
                prev["end_ms"] = seg["end_ms"]
                prev["duration_frames"] += seg["duration_frames"]
                prev["frames"] = prev["frames"] + seg["frames"]
                prev["evidence_strength"] = max(prev["evidence_strength"],
                                                seg["evidence_strength"])
                prev["boundary_criterion"] = "gap_merge"
                prev["merge_chain"] = list(chain) + [f"merge+{seg['start_ms']}"]
                prev["bbox"] = seg["bbox"]
                continue
        merged.append(dict(seg))
    return merged
