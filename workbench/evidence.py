"""Frame sequence loading, validation and per-frame change evidence extraction."""

import os
import re

import numpy as np

from .errors import BusinessError, E_DECODE_FAIL
from .pngcodec import PngError, decode_png, read_header

FRAME_RE = re.compile(r"^frame_(\d{4,})\.png$")


class SequenceError(BusinessError):
    pass


def parse_manifest(path: str) -> dict:
    """Parse 'frame_index [timestamp_ms]' lines; missing timestamps become None."""
    entries = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                frame_index = int(parts[0])
                timestamp = int(parts[1]) if len(parts) > 1 else None
            except ValueError as exc:
                raise SequenceError(E_DECODE_FAIL, f"{path}:{lineno}: bad manifest line") from exc
            if frame_index in entries:
                raise SequenceError(E_DECODE_FAIL, f"{path}:{lineno}: duplicate frame {frame_index}")
            entries[frame_index] = timestamp
    ordered = sorted(entries)
    if ordered != list(entries.keys()):
        entries = {k: entries[k] for k in ordered}
    for prev, cur in zip(ordered, ordered[1:]):
        if cur <= prev:
            raise SequenceError(E_DECODE_FAIL, f"{path}: frame indices not strictly increasing")
    return entries


def scan_sequence(seq_dir: str):
    """Validate manifest/file one-to-one mapping; return (frames, timestamps, size)."""
    manifest_path = os.path.join(seq_dir, "manifest.txt")
    if not os.path.isfile(manifest_path):
        raise SequenceError(E_DECODE_FAIL, f"{seq_dir}: manifest.txt missing")
    timestamps = parse_manifest(manifest_path)
    files = {}
    for fname in os.listdir(seq_dir):
        match = FRAME_RE.match(fname)
        if match:
            files[int(match.group(1))] = os.path.join(seq_dir, fname)
    if sorted(files) != sorted(timestamps):
        raise SequenceError(
            E_DECODE_FAIL,
            f"{seq_dir}: manifest/file mismatch "
            f"(manifest={sorted(timestamps)}, files={sorted(files)})",
        )
    size = None
    for frame_index in sorted(files):
        with open(files[frame_index], "rb") as fh:
            header = fh.read(33)
        try:
            wh = read_header(header)
        except PngError as exc:
            raise SequenceError(E_DECODE_FAIL, f"{files[frame_index]}: {exc}") from exc
        if size is None:
            size = wh
        elif wh != size:
            raise SequenceError(E_DECODE_FAIL, f"{seq_dir}: inconsistent resolution")
    frames = [files[i] for i in sorted(files)]
    stamps = [timestamps[i] for i in sorted(files)]
    return frames, stamps, size


def resolve_timestamps(timestamps, frame_interval_ms: int):
    """Fill missing timestamps by fixed-interval interpolation; tag the source."""
    resolved, sources = [], []
    first_known = next((i for i, t in enumerate(timestamps) if t is not None), None)
    base_index = first_known if first_known is not None else 0
    base_value = timestamps[base_index] if first_known is not None else 0
    for i, stamp in enumerate(timestamps):
        if stamp is None:
            resolved.append(base_value + (i - base_index) * frame_interval_ms)
            sources.append("interpolated")
        else:
            resolved.append(stamp)
            sources.append("manifest")
    return resolved, sources


def color_histogram(pixels: np.ndarray, bins: int) -> list:
    """Normalized 3-channel quantized histogram of an (N,3) uint8 array."""
    if pixels.size == 0:
        return [0.0] * (bins ** 3)
    step = 256 // bins
    idx = (pixels.astype(np.int32) // step).clip(0, bins - 1)
    flat = (idx[:, 0] * bins + idx[:, 1]) * bins + idx[:, 2]
    hist = np.bincount(flat, minlength=bins ** 3).astype(np.float64)
    hist /= hist.sum()
    return [round(float(v), 6) for v in hist]


def compute_frame_evidence(prev: np.ndarray, curr: np.ndarray, params) -> dict:
    """Change mask / ratio / color histogram between consecutive frames."""
    diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16)).max(axis=2)
    mask = diff > params.change_pixel_value_threshold
    ratio = float(mask.sum()) / float(mask.size)
    changed_pixels = [[int(y), int(x)] for y, x in zip(*np.nonzero(mask))]
    ys, xs = np.nonzero(mask)
    sel = curr[ys, xs] if mask.any() else curr.reshape(-1, 3)[:0]
    return {
        "change_ratio": round(ratio, 6),
        "changed_pixels": changed_pixels,
        "histogram": color_histogram(sel, params.hist_bins),
        "mean_color": [int(v) for v in curr.reshape(-1, 3).mean(axis=0).round()],
    }


def decode_frame(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        return decode_png(data)
    except PngError as exc:
        raise SequenceError(E_DECODE_FAIL, f"{path}: {exc}") from exc
