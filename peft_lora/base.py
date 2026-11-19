from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .canonical import atomic_write_bytes, atomic_write_json, sha256_object
from .errors import E_BASE_INVALID, BusinessError


BASE_FORMAT = "peft-base-bin-v1"
FINGERPRINT_ALGORITHM = "base-fp-v1"
DTYPE_ALIASES: dict[str, np.dtype[Any]] = {
    "<f8": np.dtype("<f8"),
    "f8": np.dtype("<f8"),
    "float64": np.dtype("<f8"),
    "float64-le": np.dtype("<f8"),
    "<f4": np.dtype("<f4"),
    "f4": np.dtype("<f4"),
    "float32": np.dtype("<f4"),
    "float32-le": np.dtype("<f4"),
}


def _invalid(message: str, **details: Any) -> BusinessError:
    return BusinessError(E_BASE_INVALID, message, details=details)


def _normalize_dtype(value: Any) -> np.dtype[Any]:
    if not isinstance(value, str) or value not in DTYPE_ALIASES:
        raise _invalid(f"unsupported parameter dtype: {value!r}")
    return DTYPE_ALIASES[value]


def _shape(value: Any, parameter: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise _invalid(f"{parameter} shape must be a non-empty list")
    result: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise _invalid(f"{parameter} shape contains a non-positive dimension")
        result.append(dimension)
    return tuple(result)


def _resolve_inside(root: Path, relative: Any, parameter: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise _invalid(f"{parameter} file path is missing")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise _invalid(f"{parameter} file escapes the base package") from exc
    if not candidate.is_file():
        raise _invalid(f"{parameter} file does not exist", path=str(candidate))
    return candidate


def _read_parameter(
    root: Path,
    descriptor: Any,
    parameter: str,
    expected_ndim: int,
    expected_last_dim: int | None = None,
) -> tuple[np.ndarray[Any, Any], str, np.dtype[Any], tuple[int, ...]]:
    if not isinstance(descriptor, dict):
        raise _invalid(f"{parameter} descriptor must be a mapping")
    dtype = _normalize_dtype(descriptor.get("dtype"))
    shape = _shape(descriptor.get("shape"), parameter)
    if len(shape) != expected_ndim:
        raise _invalid(f"{parameter} must have {expected_ndim} dimensions")
    if expected_last_dim is not None and shape[-1] != expected_last_dim:
        raise _invalid(f"{parameter} shape is inconsistent with weight shape")
    path = _resolve_inside(root, descriptor.get("file"), parameter)
    expected_bytes = math.prod(shape) * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise _invalid(
            f"{parameter} byte size does not match shape",
            expected_bytes=expected_bytes,
            actual_bytes=actual_bytes,
        )
    array = np.fromfile(path, dtype=dtype)
    if array.size != math.prod(shape):
        raise _invalid(f"{parameter} element count does not match shape")
    array = array.reshape(shape).astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise _invalid(f"{parameter} contains non-finite values")
    raw = np.ascontiguousarray(array, dtype="<f8").tobytes()
    return array, hashlib.sha256(raw).hexdigest(), dtype, shape


@dataclass(frozen=True)
class LayerWeights:
    name: str
    weight: np.ndarray[Any, Any]
    bias: np.ndarray[Any, Any]
    weight_fingerprint: str
    bias_fingerprint: str
    layer_fingerprint: str


@dataclass(frozen=True)
class BasePackage:
    path: Path
    layers: Mapping[str, LayerWeights]
    layer_fingerprints: Mapping[str, str]
    fingerprint: str
    manifest: Mapping[str, Any]

    @property
    def weight_elements(self) -> int:
        return sum(int(layer.weight.size) for layer in self.layers.values())


def load_base_package(directory: str | Path) -> BasePackage:
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise _invalid("base manifest.json does not exist", path=str(manifest_path))
    try:
        import json

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise _invalid(f"base manifest cannot be read: {exc}") from exc
    if not isinstance(manifest, dict):
        raise _invalid("base manifest must be an object")
    if manifest.get("format") != BASE_FORMAT:
        raise _invalid(f"unsupported base format: {manifest.get('format')!r}")
    if manifest.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM:
        raise _invalid("unsupported base fingerprint algorithm")
    descriptors = manifest.get("layers")
    if not isinstance(descriptors, list) or not descriptors:
        raise _invalid("base manifest layers must be a non-empty list")

    layers: dict[str, LayerWeights] = {}
    layer_fingerprints: dict[str, str] = {}
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            raise _invalid(f"layer descriptor {index} must be an object")
        name = descriptor.get("name")
        if not isinstance(name, str) or not name.strip():
            raise _invalid(f"layer descriptor {index} has an invalid name")
        if name in layers:
            raise _invalid(f"duplicate layer name: {name}")
        weight, weight_hash, _, weight_shape = _read_parameter(
            root, descriptor.get("weight"), f"layer {name} weight", 2
        )
        bias, bias_hash, _, bias_shape = _read_parameter(
            root,
            descriptor.get("bias"),
            f"layer {name} bias",
            1,
            expected_last_dim=None,
        )
        if bias_shape[0] != weight_shape[0]:
            raise _invalid(f"layer {name} bias length must equal output dimension")
        layer_fingerprint = "sha256:" + sha256_object(
            {
                "algorithm": FINGERPRINT_ALGORITHM,
                "name": name,
                "weight": {"shape": list(weight_shape), "content": weight_hash},
                "bias": {"shape": list(bias_shape), "content": bias_hash},
            }
        )
        layer = LayerWeights(
            name=name,
            weight=weight,
            bias=bias,
            weight_fingerprint=weight_hash,
            bias_fingerprint=bias_hash,
            layer_fingerprint=layer_fingerprint,
        )
        layers[name] = layer
        layer_fingerprints[name] = layer_fingerprint

    fingerprint = "sha256:" + sha256_object(
        {
            "algorithm": FINGERPRINT_ALGORITHM,
            "layers": [
                {"name": name, "fingerprint": layer_fingerprints[name]}
                for name in sorted(layer_fingerprints)
            ],
        }
    )
    return BasePackage(
        path=root,
        layers=layers,
        layer_fingerprints=layer_fingerprints,
        fingerprint=fingerprint,
        manifest=manifest,
    )


def generate_base_package(
    destination: str | Path,
    layer_shapes: Mapping[str, tuple[int, int]],
    *,
    generator_seed: int = 20261120,
    changed_layers: Iterable[str] = (),
    variant_salt: str = "",
) -> BasePackage:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    changed = set(changed_layers)
    descriptors: list[dict[str, Any]] = []
    for name in sorted(layer_shapes):
        output_dim, input_dim = layer_shapes[name]
        material = f"{generator_seed}:{name}"
        if name in changed:
            material += f":upgrade:{variant_salt}"
        seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "little")
        rng = np.random.Generator(np.random.PCG64(seed))
        weight = rng.normal(0.0, 0.22, size=(output_dim, input_dim)).astype("<f8")
        bias = rng.normal(0.0, 0.04, size=(output_dim,)).astype("<f8")
        weight_name = f"{name}.weight.bin"
        bias_name = f"{name}.bias.bin"
        atomic_write_bytes(root / weight_name, weight.tobytes(order="C"))
        atomic_write_bytes(root / bias_name, bias.tobytes(order="C"))
        descriptors.append(
            {
                "name": name,
                "weight": {"file": weight_name, "dtype": "<f8", "shape": [output_dim, input_dim]},
                "bias": {"file": bias_name, "dtype": "<f8", "shape": [output_dim]},
            }
        )
    manifest = {
        "format": BASE_FORMAT,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "generator_seed": generator_seed,
        "variant_salt": variant_salt,
        "layers": descriptors,
    }
    atomic_write_json(root / "manifest.json", manifest)
    return load_base_package(root)

