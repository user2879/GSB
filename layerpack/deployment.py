"""Deployment manifest construction and atomic package export."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .config import CompressionConfig
from .fingerprint import canonical_json, sha256_json, to_builtin
from .model import CalibrationResult, Layer, QuantResult


def compute_manifest_key(
    *,
    config: CompressionConfig,
    calibration_hash: str,
    quant_results: Iterable[QuantResult],
    calibration_results: Iterable[CalibrationResult],
) -> str:
    quant_part = [
        {
            "layer": result.layer_name,
            "dequantized_sha256": result.dequantized_sha256,
            "quantized_sha256": result.quantized_sha256,
            "parameters_sha256": result.parameters.get("parameters_sha256", ""),
        }
        for result in sorted(quant_results, key=lambda item: item.layer_name)
    ]
    calibration_part = [
        {
            "layer": result.layer_name,
            "summary_sha256": sha256_json(result.summary),
        }
        for result in sorted(calibration_results, key=lambda item: item.layer_name)
    ]
    return sha256_json(
        {
            "kind": "deployment-manifest-v1",
            "config_hash": config.deployment_config_hash,
            "calibration_hash": calibration_hash,
            "package_format_version": config.deploy_format_version,
            "artifact_name": config.artifact_name,
            "quantization": quant_part,
            "calibration": calibration_part,
        }
    )


def build_manifest(
    *,
    config: CompressionConfig,
    calibration_hash: str,
    layers: dict[str, Layer],
    quant_results: dict[str, QuantResult],
    calibration_results: dict[str, CalibrationResult],
) -> dict[str, Any]:
    manifest_layers: list[dict[str, Any]] = []
    for name in sorted(quant_results):
        quant = quant_results[name]
        calibration = calibration_results.get(name)
        layer = layers[name]
        manifest_layers.append(
            {
                "name": name,
                "shape": list(layer.shape),
                "content_version": layer.content_version,
                "content_fingerprint": layer.content_fingerprint,
                "bits": quant.parameters.get("bits", 0),
                "scheme": quant.parameters.get("scheme", ""),
                "full_precision": bool(quant.parameters.get("full_precision", False)),
                "group_size": quant.parameters.get("group_size", 0),
                "group_axis": quant.parameters.get("group_axis", -1),
                "per_channel_scaling": quant.parameters.get("per_channel_scaling", False),
                "parameters_sha256": quant.parameters.get("parameters_sha256", ""),
                "quantized_sha256": quant.quantized_sha256,
                "dequantized_sha256": quant.dequantized_sha256,
                "distortion": quant.distortion,
                "calibration_summary": calibration.summary if calibration else None,
            }
        )
    total_parameters = sum(layers[name].element_count for name in quant_results)
    total_sse = 0.0
    total_elements = 0
    for item in manifest_layers:
        total_sse += float(item["distortion"].get("sum_squared_error", 0.0))
        total_elements += int(item["distortion"].get("element_count", 0))
    manifest: dict[str, Any] = {
        "format": "layerpack-deployment-manifest-v1",
        "package_format_version": config.deploy_format_version,
        "artifact_name": config.artifact_name,
        "config_hash": config.deployment_config_hash,
        "calibration_set_hash": calibration_hash,
        "layer_count": len(manifest_layers),
        "completed_layers": [item["name"] for item in manifest_layers],
        "layers": manifest_layers,
        "totals": {
            "parameters": total_parameters,
            "quantization_elements": total_elements,
            "sum_squared_error": total_sse,
            "mse": (total_sse / total_elements) if total_elements else 0.0,
        },
    }
    manifest["manifest_sha256"] = sha256_json(to_builtin(manifest))
    return manifest


def _safe_layer_filename(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__") + ".json"


def export_package(
    *,
    state_root: str | Path,
    manifest: dict[str, Any],
    quant_results: dict[str, QuantResult],
) -> str:
    """Atomically publish a versioned package and a current-package pointer."""

    root = Path(state_root)
    deploy_root = root / "deploy"
    deploy_root.mkdir(parents=True, exist_ok=True)
    package_key = str(manifest["manifest_sha256"])[:20]
    target = deploy_root / f"package-{package_key}"
    if not target.is_dir():
        temporary = Path(tempfile.mkdtemp(prefix=".package-", dir=deploy_root))
        try:
            layers_dir = temporary / "layers"
            layers_dir.mkdir()
            with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            for name in sorted(quant_results):
                quant = quant_results[name]
                payload = {
                    "layer": name,
                    "codes": {
                        "shape": list(quant.codes.shape),
                        "dtype": "<i8",
                        "data": quant.codes.reshape(-1, order="C").tolist(),
                    },
                    "parameters": quant.parameters,
                    "dequantized_sha256": quant.dequantized_sha256,
                }
                if bool(quant.parameters.get("full_precision", False)):
                    payload["full_precision_weights"] = {
                        "shape": list(quant.dequantized.shape),
                        "dtype": "<f8",
                        "data": quant.dequantized.reshape(-1, order="C").tolist(),
                    }
                with (layers_dir / _safe_layer_filename(name)).open(
                    "w", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            (temporary / "COMMITTED").write_text("complete\n", encoding="utf-8")
            if target.exists():
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    pointer = deploy_root / "current.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".current.", suffix=".tmp", dir=deploy_root
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                canonical_json(
                    {
                        "package": target.relative_to(root).as_posix(),
                        "manifest_sha256": manifest["manifest_sha256"],
                    }
                )
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(Path(temporary_name), pointer)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()
    return target.relative_to(root).as_posix()
