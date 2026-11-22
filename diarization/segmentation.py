"""Frame-level segment reconstruction and post-processing."""


def _coalesce(runs):
    out = []
    for run in runs:
        if (out and out[-1]["label"] == run["label"]
                and out[-1]["end_frame"] == run["start_frame"]):
            out[-1]["end_frame"] = run["end_frame"]
        else:
            out.append(dict(run))
    return out


def build_segments(assignments, windows, frame_energy, n_samples, sr, config):
    """Map window cluster labels back to frame-level speaker segments.

    Each frame takes the majority label of the windows covering it
    (ties break to the lexicographically smallest cluster id). Speech
    segments shorter than min_segment_ms are dropped; same-speaker
    segments separated by less than min_silence_ms are merged; short
    silence gaps between different speakers are absorbed by the
    configured boundary rule (midpoint/left/right).
    """
    flen = sr * int(config["frame_ms"]) // 1000
    fhop = sr * int(config["frame_hop_ms"]) // 1000
    n_frames = 0 if n_samples < flen else 1 + (n_samples - flen) // fhop
    win_label = {w["window_index"]: assignments.get(w["window_index"]) for w in windows}
    labels = [None] * n_frames
    for f in range(n_frames):
        f_start = f * fhop
        f_end = f_start + flen
        votes = {}
        for w in windows:
            if w["start_sample"] <= f_start and w["end_sample"] >= f_end:
                lab = win_label[w["window_index"]]
                if lab is not None:
                    votes[lab] = votes.get(lab, 0) + 1
        best = None
        best_votes = -1
        for cid in sorted(votes):
            if votes[cid] > best_votes:
                best = cid
                best_votes = votes[cid]
        labels[f] = best

    runs = []
    f = 0
    while f < n_frames:
        g = f
        while g + 1 < n_frames and labels[g + 1] == labels[f]:
            g += 1
        runs.append({"label": labels[f], "start_frame": f, "end_frame": g + 1})
        f = g + 1

    hop_ms = 1000.0 * fhop / sr
    min_seg_frames = max(1, int(round(float(config["min_segment_ms"]) / hop_ms)))
    min_sil_frames = max(1, int(round(float(config["min_silence_ms"]) / hop_ms)))

    runs = [r for r in runs
            if r["label"] is None or (r["end_frame"] - r["start_frame"]) >= min_seg_frames]
    runs = _coalesce(runs)

    rule = config.get("boundary_rule", "midpoint")
    merged = []
    for run in runs:
        merged.append(dict(run))
        while len(merged) >= 3:
            prev, gap, nxt = merged[-3], merged[-2], merged[-1]
            if prev["label"] is None or gap["label"] is not None or nxt["label"] is None:
                break
            gap_len = gap["end_frame"] - gap["start_frame"]
            if gap_len >= min_sil_frames:
                break
            if prev["label"] == nxt["label"]:
                prev["end_frame"] = nxt["end_frame"]
                del merged[-2:]
            else:
                if rule == "left":
                    boundary = gap["start_frame"]
                elif rule == "right":
                    boundary = gap["end_frame"]
                else:
                    boundary = gap["start_frame"] + gap_len // 2
                prev["end_frame"] = boundary
                nxt["start_frame"] = boundary
                del merged[-2]
                break
    runs = _coalesce([r for r in merged if r["end_frame"] > r["start_frame"]])

    segments = []
    for run in runs:
        if run["label"] is None:
            continue
        energies = frame_energy[run["start_frame"]:run["end_frame"]]
        avg = float(sum(energies)) / len(energies) if len(energies) else 0.0
        segments.append({
            "speaker": run["label"],
            "start_ms": round(run["start_frame"] * hop_ms, 3),
            "end_ms": round(run["end_frame"] * hop_ms, 3),
            "avg_energy": round(avg, 9),
            "start_frame": run["start_frame"],
            "end_frame": run["end_frame"],
        })
    segments.sort(key=lambda s: (s["start_ms"], s["speaker"]))
    return segments
