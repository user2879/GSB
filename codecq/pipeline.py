"""Encode / decode replay pipeline built on PyAV (bundled FFmpeg).

Encoding uses libx264 with threads=1 so the bitstream is deterministic for a
given parameter set. Audio is lossless pcm_s16le so audio evaluation is not
contaminated by a lossy audio codec. All ledger hashes cover decoded content
(YUV planes plus PCM), never container bytes, so muxer-level randomness (for
example the Matroska SegmentUID) cannot leak into the ledger.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from . import colors, util

AUDIO_CHUNK = 1024


def _mux_audio(container, stream, pcm: np.ndarray, rate: int) -> None:
    n = pcm.shape[1]
    pts = 0
    for start in range(0, n, AUDIO_CHUNK):
        chunk = pcm[:, start : start + AUDIO_CHUNK]
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk), format="s16p", layout="stereo")
        frame.rate = rate
        frame.pts = pts
        frame.time_base = Fraction(1, rate)
        pts += chunk.shape[1]
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)


def mux_source_reference(planes420_seq, pcm: np.ndarray, cfg, path) -> None:
    """Write a lossless (ffv1 + pcm) inspection copy of the source."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = cfg["video"]["fps"]
    container = av.open(str(path), "w", format="matroska")
    vs = container.add_stream("ffv1", rate=Fraction(fps, 1))
    vs.width = cfg["video"]["width"]
    vs.height = cfg["video"]["height"]
    vs.pix_fmt = "yuv420p"
    vs.options = {"threads": "1"}
    ars = container.add_stream("pcm_s16le", rate=cfg["audio"]["rate"])
    ars.layout = "stereo"
    for i, planes in enumerate(planes420_seq):
        frame = av.VideoFrame.from_ndarray(colors.stack_planes(planes, "yuv420p"), format="yuv420p")
        frame.pts = i
        frame.time_base = Fraction(1, fps)
        for packet in vs.encode(frame):
            container.mux(packet)
    for packet in vs.encode(None):
        container.mux(packet)
    _mux_audio(container, ars, pcm, cfg["audio"]["rate"])
    container.close()


def encode_run(planes_seq, pcm: np.ndarray, cfg, params: dict, path) -> None:
    """Encode the source with one parameter set. Raises on failure (caller isolates).

    planes_seq: pre-converted source planes for params['pix_fmt'].
    """
    pix_fmt = params["pix_fmt"]
    if pix_fmt not in colors.SUPPORTED_PIX_FMTS:
        raise ValueError(f"unsupported pix_fmt: {pix_fmt}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = cfg["video"]["fps"]
    container = av.open(str(path), "w", format="matroska")
    vs = container.add_stream("libx264", rate=Fraction(fps, 1))
    vs.width = cfg["video"]["width"]
    vs.height = cfg["video"]["height"]
    vs.pix_fmt = pix_fmt
    vs.options = {
        "crf": str(params["crf"]),
        "g": str(params["gop"]),
        "bf": str(params["bframes"]),
        "threads": "1",
    }
    ars = container.add_stream("pcm_s16le", rate=cfg["audio"]["rate"])
    ars.layout = "stereo"
    for i, planes in enumerate(planes_seq):
        frame = av.VideoFrame.from_ndarray(colors.stack_planes(planes, pix_fmt), format=pix_fmt)
        frame.pts = i
        frame.time_base = Fraction(1, fps)
        for packet in vs.encode(frame):
            container.mux(packet)
    for packet in vs.encode(None):
        container.mux(packet)
    _mux_audio(container, ars, pcm, cfg["audio"]["rate"])
    container.close()


def decode_run(path, cfg):
    """Decode one encoded file.

    Returns (stream, planes_seq, pcm, errors). The stream record carries the
    decoded frame sequence and PCM; injections later mutate only this record
    (input/parameter layer), never the pipeline code.
    """
    errors = []
    frames = []
    planes_seq = []
    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            pix_fmt = frame.format.name
            arr = frame.to_ndarray(format=pix_fmt)
            planes = colors.unstack_planes(arr, pix_fmt)
            pts_ms = int(round(frame.pts * float(frame.time_base) * 1000.0))
            frames.append(
                {
                    "dec": len(frames),
                    "src": len(frames),
                    "pts_ms": pts_ms,
                    "pix_fmt": pix_fmt,
                    "hash": util.hash_array(arr),
                }
            )
            planes_seq.append(planes)
    except Exception as exc:  # isolated: partial decode is still reported
        errors.append({"stage": "video_decode", "frame": len(frames), "error": str(exc)})
    container.close()

    pcm_parts = []
    container = av.open(str(path))
    resampler = av.AudioResampler(format="s16p", layout="stereo", rate=cfg["audio"]["rate"])
    try:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                pcm_parts.append(out.to_ndarray())
        for out in resampler.resample(None):
            pcm_parts.append(out.to_ndarray())
    except Exception as exc:
        errors.append({"stage": "audio_decode", "error": str(exc)})
    container.close()
    pcm = np.concatenate(pcm_parts, axis=1) if pcm_parts else np.zeros((2, 0), np.int16)

    stream = {
        "frames": frames,
        "actual_pix_fmt": frames[0]["pix_fmt"] if frames else None,
        "audio": {
            "rate_actual": cfg["audio"]["rate"],
            "rate_declared": cfg["audio"]["rate"],
            "samples": int(pcm.shape[1]),
            "sha256": util.hash_array(pcm),
        },
    }
    return stream, planes_seq, pcm, errors


def frame_order_hash(frames) -> str:
    """Hash of the decoded frame sequence (src mapping + pts + content)."""
    parts = [f"{f['dec']}:{f['src']}:{f['pts_ms']}:{f['hash']}" for f in frames]
    return util.sha256_hex("\n".join(parts).encode("ascii"))
