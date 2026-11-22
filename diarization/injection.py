"""Observation injection: real perturbations applied to input sample data."""
import numpy as np


def apply_injections(samples, sr, filename, entries, config):
    """Apply manifest-declared perturbations for one audio file.

    Perturbations genuinely modify the sample stream seen by windowing
    and embedding: non-finite samples, consecutive clipped frames,
    injected silence, and data-chunk truncation. Returns the modified
    samples plus the list of hit injection ids.
    """
    flen = sr * int(config["frame_ms"]) // 1000
    fhop = sr * int(config["frame_hop_ms"]) // 1000
    data = samples
    hits = []
    for inj in sorted(entries, key=lambda e: e["id"]):
        if inj.get("file") != filename:
            continue
        start = int(inj.get("frame_start", 0)) * fhop
        end = int(inj.get("frame_end", 0)) * fhop + flen
        kind = inj.get("type")
        if kind == "truncate_data":
            data = data[:min(end, len(data))]
        elif kind in ("nonfinite", "clip", "silence"):
            data = data.copy()
            lo = min(start, len(data))
            hi = min(end, len(data))
            if kind == "nonfinite":
                data[lo:hi] = np.nan
            elif kind == "clip":
                parity = np.arange(hi - lo) % 2
                data[lo:hi] = np.where(parity == 0, 32767.0 / 32768.0, -1.0)
            else:
                data[lo:hi] = 0.0
        else:
            continue
        hits.append({
            "id": inj["id"],
            "type": kind,
            "file": filename,
            "frame_start": inj.get("frame_start"),
            "frame_end": inj.get("frame_end"),
        })
    return data, hits
