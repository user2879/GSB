"""Back-mapping of spectrum-position verdicts onto waveform segments."""


def map_verdicts_to_segments(all_positions, verdicts, allowed_positions=None):
    """Return an ordered segment map using each position's registered spans."""
    allowed = set(allowed_positions) if allowed_positions is not None else None
    verdict_by_position = {}
    for index, verdict in enumerate(verdicts):
        position = (allowed_positions[index] if allowed_positions is not None
                    else verdict.position)
        verdict_by_position[position] = verdict
    segment_map = {}
    for pos in all_positions:
        if allowed is not None and pos.position not in allowed:
            continue
        verdict = verdict_by_position[pos.position]
        for seg_id in pos.segments:
            bucket = segment_map.setdefault(seg_id, {
                "segment_id": seg_id,
                "positions": [],
                "anomaly_positions": [],
                "boundary_positions": [],
            })
            bucket["positions"].append(pos.position)
            if verdict.is_anomaly:
                bucket["anomaly_positions"].append(pos.position)
            if verdict.boundary:
                bucket["boundary_positions"].append(pos.position)
    result = []
    for seg_id in sorted(segment_map):
        bucket = segment_map[seg_id]
        bucket["positions"] = sorted(set(bucket["positions"]))
        bucket["anomaly_positions"] = sorted(set(bucket["anomaly_positions"]))
        bucket["boundary_positions"] = sorted(set(bucket["boundary_positions"]))
        result.append(bucket)
    return result
