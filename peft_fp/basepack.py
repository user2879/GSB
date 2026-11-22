"""Base parameter pack: binary per-layer weights + shape manifest, validation
and versioned fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .errors import E_BASE_INVALID, failure

FINGERPRINT_ALGO_VERSION = "fp-sha256-v1"
LAYERS_MANIFEST = "layers.json"


@dataclass
class Layer:
    name: str
    shape: tuple
    weight: np.ndarray  # (out, in) float64
    bias: np.ndarray  # (out,) float64


@dataclass
class BasePack:
    layers: dict = field(default_factory=dict)  # name -> Layer

    def layer_fingerprint(self, name: str) -> str:
        layer = self.layers[name]
        h = hashlib.sha256()
        h.update(FINGERPRINT_ALGO_VERSION.encode())
        h.update(name.encode())
        h.update(json.dumps(list(layer.shape)).encode())
        h.update(np.ascontiguousarray(layer.weight, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(layer.bias, dtype="<f8").tobytes())
        return h.hexdigest()

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(FINGERPRINT_ALGO_VERSION.encode())
        for name in sorted(self.layers):
            h.update(self.layer_fingerprint(name).encode())
        return h.hexdigest()


def load_base_pack(pack_dir: str):
    """Load and validate a base pack. Returns (BasePack|None, failures)."""
    failures = []
    manifest_path = os.path.join(pack_dir, LAYERS_MANIFEST)
    if not os.path.isfile(manifest_path):
        return None, [failure(E_BASE_INVALID, f"missing {LAYERS_MANIFEST}")]
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [failure(E_BASE_INVALID, f"manifest unreadable: {exc}")]
    entries = manifest.get("layers")
    if not isinstance(entries, list) or not entries:
        return None, [failure(E_BASE_INVALID, "manifest has no layers")]

    pack = BasePack()
    for entry in entries:
        name = entry.get("name")
        shape = entry.get("shape")
        if not isinstance(name, str) or not name:
            failures.append(failure(E_BASE_INVALID, "layer entry missing name"))
            continue
        if name in pack.layers:
            failures.append(failure(E_BASE_INVALID, "duplicate layer name", layer=name))
            continue
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(d, int) or d <= 0 for d in shape)
        ):
            failures.append(failure(E_BASE_INVALID, f"bad shape {shape}", layer=name))
            continue
        w, ok = _read_matrix(pack_dir, entry.get("weight_file"), tuple(shape), name, failures)
        if not ok:
            continue
        b, ok = _read_matrix(pack_dir, entry.get("bias_file"), (shape[0],), name, failures)
        if not ok:
            continue
        pack.layers[name] = Layer(name=name, shape=tuple(shape), weight=w, bias=b)

    if failures:
        return None, failures
    return pack, []


def _read_matrix(pack_dir, rel_file, shape, layer_name, failures):
    if not isinstance(rel_file, str):
        failures.append(failure(E_BASE_INVALID, "missing file entry", layer=layer_name))
        return None, False
    path = os.path.join(pack_dir, rel_file)
    try:
        raw = open(path, "rb").read()
    except OSError as exc:
        failures.append(failure(E_BASE_INVALID, f"cannot read {rel_file}: {exc}", layer=layer_name))
        return None, False
    expect = int(np.prod(shape)) * 8
    if len(raw) != expect:
        failures.append(
            failure(
                E_BASE_INVALID,
                f"shape mismatch: {rel_file} has {len(raw)} bytes, expect {expect}",
                layer=layer_name,
            )
        )
        return None, False
    arr = np.frombuffer(raw, dtype="<f8").reshape(shape).astype(np.float64)
    if not np.all(np.isfinite(arr)):
        failures.append(failure(E_BASE_INVALID, f"non-finite values in {rel_file}", layer=layer_name))
        return None, False
    return arr, True


def replace_layer(pack_dir: str, layer_name: str, weight: np.ndarray, bias: np.ndarray) -> None:
    """Atomically replace one layer's binary files (write tmp + os.replace)."""
    manifest_path = os.path.join(pack_dir, LAYERS_MANIFEST)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    for entry in manifest["layers"]:
        if entry["name"] != layer_name:
            continue
        for key, arr in (("weight_file", weight), ("bias_file", bias)):
            target = os.path.join(pack_dir, entry[key])
            tmp = target + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(np.ascontiguousarray(arr, dtype="<f8").tobytes())
            os.replace(tmp, target)
        return
    raise KeyError(f"layer not in pack: {layer_name}")
