"""Anomaly injection at the input/parameter layer only.

Injections mutate the decoded *stream record* (frames list, declared params,
PCM) produced by the pipeline. They never touch pipeline/metrics/detection
code, so the same binary both with and without injections exercises identical
logic. Injection configs are plain serializable dicts recorded in the ledger.
"""
from __future__ import annotations

import numpy as np

TYPES = (
    "frame_drop",
    "timestamp_jump",
    "chroma_mislabel",
    "audio_rate_misreport",
    "audio_shift",
)


def apply_injection(stream: dict, pcm: np.ndarray, inj: dict) -> np.ndarray:
    """Apply one injection to (stream, pcm); returns the (new) pcm."""
    kind = inj["type"]
    if kind == "frame_drop":
        drop = set(inj["frames"])
        stream["frames"] = [f for f in stream["frames"] if f["src"] not in drop]
    elif kind == "timestamp_jump":
        target = inj["frame"]
        for f in stream["frames"]:
            if f["src"] == target:
                f["pts_ms"] += inj["delta_ms"]
                break
        else:
            raise ValueError(f"timestamp_jump target frame {target} not present")
    elif kind == "chroma_mislabel":
        stream["params"]["pix_fmt"] = inj["declared"]
    elif kind == "audio_rate_misreport":
        stream["audio"]["rate_declared"] = inj["declared_rate"]
    elif kind == "audio_shift":
        k = inj["offset_samples"]
        if k >= 0:
            pcm = np.concatenate([np.zeros((2, k), pcm.dtype), pcm], axis=1)[:, : pcm.shape[1]]
        else:
            pcm = np.concatenate([pcm[:, -k:], np.zeros((2, -k), pcm.dtype)], axis=1)[
                :, : pcm.shape[1]
            ]
        stream["audio"]["shifted"] = True
    else:
        raise ValueError(f"unknown injection type: {kind}")
    return pcm


def apply_injections(stream: dict, pcm: np.ndarray, injections):
    """Apply all injections targeting this encode id; returns (pcm, applied)."""
    applied = []
    for inj in injections:
        pcm = apply_injection(stream, pcm, inj)
        applied.append(dict(inj))
    return pcm, applied
