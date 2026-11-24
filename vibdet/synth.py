"""Deterministic synthetic rotating-machinery vibration waveforms.

All randomness comes from explicit, seed-derived ``numpy.random.Generator``
instances; repeated calls with identical descriptions return byte-identical
samples.  Runtime detector/feature code never constructs a random generator.
"""

from dataclasses import dataclass

import numpy as np
import yaml

from .errors import E_DEVICE_UNKNOWN, E_SAMPLE_NONFINITE, PipelineError
from .hashing import array_fingerprint, stable_hash


@dataclass
class SegmentMeta:
    seg_id: str
    start: int
    end: int
    fault: object = None

    def to_dict(self):
        return {"seg_id": self.seg_id, "start": self.start,
                "end": self.end, "fault": self.fault}


@dataclass
class WaveData:
    device_id: str
    values: np.ndarray
    segments: list
    missing_mask: np.ndarray
    resample_mask: np.ndarray
    impulse_mask: np.ndarray
    seed: int
    data_fingerprint: str

    def to_manifest(self):
        return {
            "device_id": self.device_id,
            "length": int(self.values.shape[0]),
            "seed": self.seed,
            "segments": [s.to_dict() for s in self.segments],
            "missing_count": int(self.missing_mask.sum()),
            "resample_count": int(self.resample_mask.sum()),
            "impulse_count": int(self.impulse_mask.sum()),
            "data_fingerprint": self.data_fingerprint,
        }


def load_descriptions(path):
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    doc.setdefault("devices", {})
    doc.setdefault("window_overrides", {})
    return doc


def _fill_missing(values, missing_mask, method):
    filled = values.copy()
    if not missing_mask.any():
        return filled
    bad = np.where(missing_mask)[0]
    good = np.where(~missing_mask)[0]
    if good.size == 0:
        filled[bad] = 0.0
        return filled
    if method == "zero":
        filled[bad] = 0.0
    elif method == "mean":
        filled[bad] = float(np.mean(filled[good]))
    else:  # linear interpolation with edge hold
        interp = np.interp(bad.astype(np.float64), good.astype(np.float64),
                           filled[good])
        filled[bad] = interp
    return filled


def _generate_segment(seg, index, sample_rate):
    length = int(seg["length"])
    seed = int(seg.get("seed", 0))
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64) / float(sample_rate)
    rpm_hz = float(seg.get("rpm_hz", 25.0))
    harmonics = list(seg.get("harmonics", [1.0, 0.5, 0.25]))
    signal = np.zeros(length, dtype=np.float64)
    for harmonic_index, amplitude in enumerate(harmonics, start=1):
        signal += float(amplitude) * np.sin(
            2.0 * np.pi * rpm_hz * harmonic_index * t)
    noise_sigma = float(seg.get("noise_sigma", 0.05))
    signal += noise_sigma * rng.standard_normal(length)

    impulse_mask = np.zeros(length, dtype=bool)
    impulse = seg.get("impulse") or {}
    imp_count = int(impulse.get("count", 0))
    if imp_count:
        imp_rng = np.random.default_rng(seed + 77)
        chosen = imp_rng.choice(length, size=min(imp_count, length),
                                replace=False)
        amp = float(impulse.get("amplitude", 3.0))
        for idx in chosen:
            impulse_mask[idx] = True
            tail = min(8, length - idx)
            decay = amp * np.exp(-np.arange(tail, dtype=np.float64) / 2.0)
            signal[idx:idx + tail] += decay

    fault = seg.get("fault")
    if fault:
        fault_rng = np.random.default_rng(seed + 131)
        amp = float(fault.get("amplitude", 1.5)) if isinstance(fault, dict) else 1.5
        signal += amp * fault_rng.standard_normal(length)

    missing_mask = np.zeros(length, dtype=bool)
    missing = seg.get("missing") or {}
    miss_count = int(missing.get("count", 0))
    if miss_count:
        miss_rng = np.random.default_rng(seed + 211)
        chosen = miss_rng.choice(length, size=min(miss_count, length),
                                 replace=False)
        missing_mask[chosen] = True
        signal[chosen] = np.nan

    resample_mask = np.zeros(length, dtype=bool)
    resample = seg.get("resample") or {}
    for idx in resample.get("indices", []) or []:
        idx = int(idx)
        if 0 <= idx < length:
            resample_mask[idx] = True
            left = signal[idx - 1] if idx > 0 and np.isfinite(signal[idx - 1]) else 0.0
            right = (signal[idx + 1] if idx < length - 1
                     and np.isfinite(signal[idx + 1]) else 0.0)
            signal[idx] = 0.5 * (left + right)

    if seg.get("force_nonfinite"):
        signal[length // 2] = np.inf
    return signal, missing_mask, resample_mask, impulse_mask, fault


def generate_device(device_id, desc, cfg):
    # Accept either the full description document (which carries top-level
    # window overrides) or a single device description.
    is_full_document = (
        isinstance(desc, dict)
        and isinstance(desc.get("devices"), dict)
        and device_id in desc["devices"])
    is_device_map = (
        isinstance(desc, dict) and not is_full_document
        and device_id in desc
        and isinstance(desc[device_id], dict)
        and "segments" in desc[device_id])
    if is_device_map:
        desc = desc[device_id]
    if is_full_document:
        if device_id not in desc["devices"]:
            raise PipelineError(
                E_DEVICE_UNKNOWN,
                f"device {device_id} is not present in waveform descriptions",
                device=device_id)
        desc = desc["devices"][device_id]
    if not isinstance(desc, dict) or not desc.get("segments"):
        raise PipelineError(
            E_DEVICE_UNKNOWN,
            f"device {device_id} has no segment descriptions",
            device=device_id)
    sample_rate = int(cfg["features"]["sample_rate"])
    pieces, miss_pieces, res_pieces, imp_pieces = [], [], [], []
    segments = []
    cursor = 0
    for index, seg in enumerate(desc.get("segments", [])):
        signal, missing_mask, resample_mask, impulse_mask, fault = \
            _generate_segment(seg, index, sample_rate)
        pieces.append(signal)
        miss_pieces.append(missing_mask)
        res_pieces.append(resample_mask)
        imp_pieces.append(impulse_mask)
        start, end = cursor, cursor + signal.shape[0]
        segments.append(SegmentMeta(seg.get("id", f"seg{index}"), start, end,
                                    fault))
        cursor = end

    raw = np.concatenate(pieces).astype(np.float64)
    missing_mask = np.concatenate(miss_pieces)
    resample_mask = np.concatenate(res_pieces)
    impulse_mask = np.concatenate(imp_pieces)
    values = _fill_missing(raw, missing_mask, cfg["missing"]["fill"])

    if not np.all(np.isfinite(values)):
        bad = int(np.where(~np.isfinite(values))[0][0])
        raise PipelineError(
            E_SAMPLE_NONFINITE,
            f"device {device_id} has non-finite sample at index {bad} "
            f"after {cfg['missing']['fill']} fill",
            device=device_id)

    data_fingerprint = stable_hash([
        "wave",
        device_id,
        array_fingerprint(values),
        missing_mask.tolist(),
        resample_mask.tolist(),
        [s.seg_id for s in segments],
    ])
    return WaveData(device_id, values, segments, missing_mask, resample_mask,
                    impulse_mask, int(desc.get("seed", 0)), data_fingerprint)
