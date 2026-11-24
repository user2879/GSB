"""Deterministic workspace edit operations used in edit-sequence validation."""

import numpy as np
import yaml

from .hashing import stable_hash


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dump_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, allow_unicode=True)


def edit_config(workspace, path, value):
    config_path = workspace / "config.yaml"
    cfg = _load_yaml(config_path)
    node = cfg
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    _dump_yaml(config_path, cfg)
    return {"op": "config", "path": path, "value": value}


def replace_window(workspace, device_id, window_index, amplitude=None):
    """Replace one window's source segment samples deterministically.

    The replacement lives inside the first segment backing the requested
    window and is regenerated from a seed derived from (device, window), so
    the same edit is reproducible.  Only the affected segment content (and
    therefore windows overlapping it) changes.
    """
    desc_path = workspace / "descriptions.yaml"
    doc = _load_yaml(desc_path)
    devices = doc["devices"]
    if device_id not in devices:
        raise KeyError(f"unknown device {device_id}")
    overrides = doc.setdefault("window_overrides", {})
    amp = 1.0 if amplitude is None else float(amplitude)
    seed = int.from_bytes(
        stable_hash([device_id, int(window_index), "window-replace"])[:8].encode(),
        "little") % (2 ** 32)
    overrides[f"{device_id}:{int(window_index)}"] = {
        "device": device_id,
        "window_index": int(window_index),
        "amplitude": amp,
        "seed": seed,
    }
    _dump_yaml(desc_path, doc)
    return {"op": "replace_window", "device": device_id,
            "window_index": int(window_index), "amplitude": amp, "seed": seed}


def delete_segment(workspace, device_id, segment_id):
    desc_path = workspace / "descriptions.yaml"
    doc = _load_yaml(desc_path)
    segments = doc["devices"][device_id]["segments"]
    before = len(segments)
    segments = [seg for seg in segments if seg.get("id") != segment_id]
    if len(segments) == before:
        raise KeyError(f"segment {segment_id} not found on {device_id}")
    doc["devices"][device_id]["segments"] = segments
    _dump_yaml(desc_path, doc)
    return {"op": "delete_segment", "device": device_id,
            "segment_id": segment_id}


def apply_window_overrides(wave_values, descriptions, device_id, window_length,
                           window_hop):
    """Apply deterministic window replacements to generated raw values."""
    values = np.array(wave_values, dtype=np.float64)
    overrides = (descriptions.get("window_overrides", {}) or {})
    for key, spec in overrides.items():
        if spec.get("device") != device_id:
            continue
        index = int(spec["window_index"])
        start = index * window_hop
        end = start + window_length
        if end > values.shape[0]:
            continue
        rng = np.random.default_rng(int(spec["seed"]))
        replacement = float(spec["amplitude"]) * rng.standard_normal(
            window_length)
        values[start:end] = replacement
    return values
