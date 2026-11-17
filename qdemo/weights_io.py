"""Deterministic local weight-pack and calibration-set generation / loading.

No real model files, no network: everything is derived from CLI seeds.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .common import (
    BusinessError,
    E_CONFIG_INVALID,
    E_WEIGHT_SHAPE_MISMATCH,
    array_fingerprint,
    fingerprint,
)

META_SUFFIX = ".meta.json"


def parse_layer_spec(spec: str):
    """Parse 'name:out:in:dist,name2:out:in:dist' where dist in normal|uniform."""
    layers = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        if len(parts) != 4:
            raise BusinessError(E_CONFIG_INVALID, f"bad layer spec item: {item!r}")
        name, out_f, in_f, dist = parts
        if dist not in ("normal", "uniform"):
            raise BusinessError(E_CONFIG_INVALID, f"bad distribution: {dist!r}")
        layers.append((name, int(out_f), int(in_f), dist))
    names = [n for n, _, _, _ in layers]
    if len(set(names)) != len(names):
        raise BusinessError(E_CONFIG_INVALID, "duplicate layer names in spec")
    return layers


def generate_weights(out_path: str, seed: int, spec: str) -> dict:
    layers = parse_layer_spec(spec)
    rng = np.random.default_rng(seed)
    arrays = {}
    meta_layers = []
    for name, out_f, in_f, dist in layers:
        if dist == "normal":
            w = rng.normal(0.0, 0.5, size=(out_f, in_f))
        else:
            w = rng.uniform(-1.0, 1.0, size=(out_f, in_f))
        w = w.astype(np.float64)
        arrays[name] = w
        meta_layers.append(
            {"name": name, "shape": [out_f, in_f], "dtype": "float64", "dist": dist}
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez(out_path, **arrays)
    meta = {"format": "qdemo-weights", "version": 1, "seed": seed, "layers": meta_layers}
    with open(out_path + META_SUFFIX, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def generate_calibration(weights_path: str, out_path: str, seed: int, samples: int) -> dict:
    pack = load_weights(weights_path)
    rng = np.random.default_rng(seed)
    in_dim = pack.arrays[pack.layer_names[0]].shape[1]
    x = rng.normal(0.0, 1.0, size=(samples, in_dim)).astype(np.float64)
    from .evaluate import forward

    y_ref = forward(pack.ordered_arrays(), x)[pack.layer_names[-1]]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez(out_path, X=x, Y_ref=y_ref)
    meta = {
        "format": "qdemo-calib",
        "version": 1,
        "seed": seed,
        "samples": samples,
        "input_dim": in_dim,
        "output_dim": int(y_ref.shape[1]),
        "weights_fingerprint": pack.fingerprint,
    }
    with open(out_path + META_SUFFIX, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


class WeightPack:
    def __init__(self, layer_names, arrays):
        self.layer_names = layer_names  # model order (generation order)
        self.arrays = arrays  # dict name -> np.ndarray
        self.layer_fingerprints = {n: array_fingerprint(arrays[n]) for n in layer_names}
        self.fingerprint = fingerprint(
            {n: {"shape": list(arrays[n].shape), "fp": self.layer_fingerprints[n]} for n in layer_names}
        )

    def ordered_arrays(self):
        return [(n, self.arrays[n]) for n in self.layer_names]


def load_weights(path: str) -> WeightPack:
    meta_path = path + META_SUFFIX
    if not os.path.exists(path) or not os.path.exists(meta_path):
        raise BusinessError(E_CONFIG_INVALID, f"weights pack or meta missing: {path}")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    layers = meta.get("layers", [])
    names = [l["name"] for l in layers]
    if len(set(names)) != len(names):
        raise BusinessError(E_CONFIG_INVALID, "duplicate layer names in weights meta")
    data = np.load(path)
    arrays = {}
    for layer in layers:
        name, shape = layer["name"], tuple(layer["shape"])
        if name not in data:
            raise BusinessError(E_WEIGHT_SHAPE_MISMATCH, f"layer {name} missing in npz")
        arr = np.asarray(data[name], dtype=np.float64)
        if arr.shape != shape:
            raise BusinessError(
                E_WEIGHT_SHAPE_MISMATCH,
                f"layer {name} shape {arr.shape} != meta {shape}",
            )
        if not np.all(np.isfinite(arr)):
            raise BusinessError(E_CONFIG_INVALID, f"layer {name} contains non-finite values")
        arrays[name] = arr
    for i in range(len(names) - 1):
        if arrays[names[i]].shape[0] != arrays[names[i + 1]].shape[1]:
            raise BusinessError(
                E_WEIGHT_SHAPE_MISMATCH,
                f"layers {names[i]}->{names[i+1]} are not chainable",
            )
    return WeightPack(names, arrays)


class CalibPack:
    def __init__(self, x, y_ref):
        self.X = x
        self.Y_ref = y_ref
        self.fingerprint = fingerprint(
            {"X": array_fingerprint(x), "Y_ref": array_fingerprint(y_ref)}
        )


def load_calibration(path: str) -> CalibPack:
    if not os.path.exists(path):
        raise BusinessError(E_CONFIG_INVALID, f"calibration pack missing: {path}")
    data = np.load(path)
    x = np.asarray(data["X"], dtype=np.float64)
    y_ref = np.asarray(data["Y_ref"], dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y_ref)):
        raise BusinessError(E_CONFIG_INVALID, "calibration pack contains non-finite values")
    return CalibPack(x, y_ref)
