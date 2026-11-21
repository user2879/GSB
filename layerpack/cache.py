"""Content-addressed persistence for reusable intermediate artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .fingerprint import sha256_array
from .model import CalibrationResult, QuantResult


CACHE_FORMAT = "layerpack-cache-v1"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class ArtifactCache:
    """A simple, inspectable JSON cache keyed by content fingerprints."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.quant_dir = self.root / "cache" / "quant"
        self.calibration_dir = self.root / "cache" / "calibration"
        self.manifest_dir = self.root / "cache" / "manifest"
        for path in (self.quant_dir, self.calibration_dir, self.manifest_dir):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _path(directory: Path, key: str) -> Path:
        if len(key) < 16 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a hexadecimal content digest")
        return directory / f"{key}.json"

    def has_quant(self, key: str) -> bool:
        return self._path(self.quant_dir, key).is_file()

    def has_calibration(self, key: str) -> bool:
        return self._path(self.calibration_dir, key).is_file()

    def has_manifest(self, key: str) -> bool:
        return self._path(self.manifest_dir, key).is_file()

    def save_quant(self, result: QuantResult) -> None:
        payload = {
            "format": CACHE_FORMAT,
            "kind": "quant",
            "key": result.key,
            "dequant_key": result.dequant_key,
            "layer": result.layer_name,
            "parameters": result.parameters,
            "codes": {
                "shape": list(result.codes.shape),
                "dtype": "<i8",
                "data": result.codes.reshape(-1, order="C").tolist(),
            },
            "dequantized": {
                "shape": list(result.dequantized.shape),
                "dtype": "<f8",
                "data": result.dequantized.reshape(-1, order="C").tolist(),
            },
            "distortion": result.distortion,
            "quantized_sha256": result.quantized_sha256,
            "dequantized_sha256": result.dequantized_sha256,
        }
        _atomic_write_json(self._path(self.quant_dir, result.key), payload)

    def load_quant(self, key: str) -> QuantResult | None:
        path = self._path(self.quant_dir, key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") != CACHE_FORMAT or payload.get("kind") != "quant":
                return None
            if payload.get("key") != key:
                return None
            code_payload = payload["codes"]
            dequant_payload = payload["dequantized"]
            codes = np.asarray(code_payload["data"], dtype=np.int64).reshape(
                tuple(code_payload["shape"])
            )
            dequantized = np.asarray(
                dequant_payload["data"], dtype=np.float64
            ).reshape(tuple(dequant_payload["shape"]))
            if not bool(payload["parameters"].get("full_precision", False)):
                if sha256_array(codes, dtype="<i8") != payload["quantized_sha256"]:
                    return None
            if sha256_array(dequantized) != payload["dequantized_sha256"]:
                return None
            return QuantResult(
                layer_name=str(payload["layer"]),
                parameters=payload["parameters"],
                dequantized=dequantized,
                codes=codes,
                distortion=payload["distortion"],
                quantized_sha256=str(payload["quantized_sha256"]),
                dequantized_sha256=str(payload["dequantized_sha256"]),
                key=key,
                dequant_key=str(payload.get("dequant_key", "")),
                reused=True,
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def save_calibration(self, result: CalibrationResult) -> None:
        payload = {
            "format": CACHE_FORMAT,
            "kind": "calibration",
            "key": result.key,
            "layer": result.layer_name,
            "samples": result.samples,
            "summary": result.summary,
        }
        _atomic_write_json(self._path(self.calibration_dir, result.key), payload)

    def load_calibration(self, key: str) -> CalibrationResult | None:
        path = self._path(self.calibration_dir, key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("format") != CACHE_FORMAT
                or payload.get("kind") != "calibration"
                or payload.get("key") != key
            ):
                return None
            return CalibrationResult(
                layer_name=str(payload["layer"]),
                samples=payload["samples"],
                summary=payload["summary"],
                key=key,
                reused=True,
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def save_manifest(self, key: str, manifest: dict[str, Any]) -> None:
        payload = {
            "format": CACHE_FORMAT,
            "kind": "deployment-manifest",
            "key": key,
            "manifest": manifest,
        }
        _atomic_write_json(self._path(self.manifest_dir, key), payload)

    def load_manifest(self, key: str) -> dict[str, Any] | None:
        path = self._path(self.manifest_dir, key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("format") != CACHE_FORMAT
                or payload.get("kind") != "deployment-manifest"
                or payload.get("key") != key
            ):
                return None
            value = payload["manifest"]
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
