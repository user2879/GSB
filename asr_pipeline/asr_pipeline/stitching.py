"""Overlap-aware stitching with last-writer-wins authority and conflict records."""


def stitch_audio(shard_results, frame_shift_samples):
    """Stitch decoded shards into one transcript keyed on global frame span.

    ``shard_results`` is ordered by shard index. Each entry carries the shard
    index, the frame offset of its first frame in the audio timeline and its
    decoded words. Later shards override earlier ones inside the overlap.
    """
    timeline = {}
    conflicts = []
    ordered = sorted(shard_results, key=lambda item: item["shard_index"])
    for result in ordered:
        offset = int(result["frame_offset"])
        source = int(result["shard_index"])
        for word in result["words"]:
            global_start = offset + word["frame_start"]
            global_end = offset + word["frame_end"]
            entry = {
                "char": word["char"],
                "frame_start": global_start,
                "frame_end": global_end,
                "emission_score": word["emission_score"],
                "lm_score": word["lm_score"],
                "total_score": word["total_score"],
                "source_shard": source,
            }
            covered = sorted(
                frame for frame in range(global_start, global_end + 1) if frame in timeline
            )
            if covered:
                owners = sorted({timeline[frame]["source_shard"] for frame in covered})
                conflicts.append({
                    "overridden_frames": covered,
                    "overridden_char": timeline[covered[0]]["char"],
                    "overridden_by_char": entry["char"],
                    "overridden_source_shards": owners,
                    "winning_source_shard": source,
                })
            for frame in range(global_start, global_end + 1):
                timeline[frame] = entry

    words = []
    seen = set()
    for frame in sorted(timeline):
        entry = timeline[frame]
        key = (entry["frame_start"], entry["frame_end"], entry["source_shard"])
        if key in seen:
            continue
        seen.add(key)
        words.append(entry)
    words.sort(key=lambda item: (item["frame_start"], item["frame_end"]))

    transcript = "".join(word["char"] for word in words)
    return {
        "transcript": transcript,
        "words": words,
        "conflicts": conflicts,
        "frame_shift_samples": int(frame_shift_samples),
    }
