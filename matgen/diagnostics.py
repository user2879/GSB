"""Artifact and JSON diagnostic report generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .errors import Issue
from .fixed import CHANNELS, fixed_channel_to_uint8
from .io_utils import atomic_write_json
from .models import BuildResult, GenerationConfig, LoadedInput, Material


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _layer_report(result: BuildResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in result.layers:
        scale = float(layer.scale)
        residual = layer.residual.astype(np.float64) / scale
        rows.append(
            {
                "index": layer.index,
                "key": layer.key,
                "texture_key": layer.texture_key,
                "upstream_layer_key": layer.upstream_layer_key,
                "frequency": layer.frequency,
                "amplitude": layer.amplitude,
                "reused": layer.reused,
                "quantization_step_integer": layer.quantization_step,
                "quantization_step_normalized": layer.quantization_step / scale,
                "integer_min": int(np.min(layer.quantized)),
                "integer_max": int(np.max(layer.quantized)),
                "residual_min": float(np.min(residual)),
                "residual_max": float(np.max(residual)),
                "residual_mean_abs": float(np.mean(np.abs(residual))),
                "content_hash": layer.content_hash(),
            }
        )
    return rows


def _material_report(result: BuildResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for material_id, artifact in sorted(result.material_artifacts.items()):
        channels: dict[str, Any] = {}
        for channel, channel_artifact in sorted(artifact.channels.items()):
            channels[channel] = {
                "key": channel_artifact.key,
                "content_hash": channel_artifact.content_hash(),
                "layer_indices": list(channel_artifact.layer_indices),
                "layer_keys": list(channel_artifact.layer_keys),
                "reused": channel_artifact.reused,
                "fixed_min": int(np.min(channel_artifact.final)),
                "fixed_max": int(np.max(channel_artifact.final)),
                "normalized_min": float(np.min(channel_artifact.final)) / result.scale,
                "normalized_max": float(np.max(channel_artifact.final)) / result.scale,
                "boundary_total": channel_artifact.boundary_total,
                "boundary_samples": channel_artifact.boundary_samples,
                "contribution_layers": sorted(channel_artifact.contributions),
            }
        rows.append(
            {
                "material_id": material_id,
                "channel_summary": artifact.as_summary(result.scale),
                "channels": channels,
            }
        )
    return rows


def write_build_artifacts(
    output_dir: str | Path,
    *,
    result: BuildResult,
    error_diffs: dict[str, np.ndarray] | None = None,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_dir = destination / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    layer_arrays: dict[str, np.ndarray] = {}
    for layer in result.layers:
        layer_arrays[f"layer_{layer.index}__unquantized"] = layer.unquantized
        layer_arrays[f"layer_{layer.index}__quantized"] = layer.quantized
        layer_arrays[f"layer_{layer.index}__residual"] = layer.residual
    layer_path = artifact_dir / "layers.npz"
    np.savez_compressed(layer_path, **layer_arrays)
    paths["layers"] = str(layer_path)

    material_arrays: dict[str, np.ndarray] = {}
    final_arrays: dict[str, np.ndarray] = {}
    for material_id, artifact in result.material_artifacts.items():
        for channel, channel_artifact in artifact.channels.items():
            prefix = f"{material_id}__{channel}"
            material_arrays[f"{prefix}__mixed"] = channel_artifact.mixed
            material_arrays[f"{prefix}__final"] = channel_artifact.final
            final_arrays[prefix] = channel_artifact.final
            for layer_index, contribution in channel_artifact.contributions.items():
                material_arrays[f"{prefix}__contribution_{layer_index}"] = contribution
            preview = fixed_channel_to_uint8(channel_artifact.final, result.scale, channel)
            preview_path = artifact_dir / f"preview__{material_id}__{channel}.png"
            Image.fromarray(preview, mode="L").save(preview_path)
            paths.setdefault("previews", [])
            paths["previews"].append(str(preview_path))  # type: ignore[union-attr]
    material_path = artifact_dir / "material_channels.npz"
    np.savez_compressed(material_path, **material_arrays)
    paths["material_channels"] = str(material_path)
    final_path = artifact_dir / "final_channels.npz"
    np.savez_compressed(final_path, **final_arrays)
    paths["final_channels"] = str(final_path)

    if error_diffs:
        diff_path = artifact_dir / "error_trace_differences.npz"
        np.savez_compressed(diff_path, **error_diffs)
        paths["error_trace_differences"] = str(diff_path)
    return paths


def build_report(
    *,
    loaded: LoadedInput,
    interactive: dict[str, Any],
    error_trace: dict[str, Any] | None,
    full_rebuild: dict[str, Any] | None,
    injection_report: dict[str, Any] | None,
    snapshot_summary: dict[str, Any] | None,
    artifact_paths: dict[str, Any],
    extra_issues: list[Issue],
    input_config_hash: str,
) -> dict[str, Any]:
    result: BuildResult | None = interactive.get("final_result")
    concrete_config: GenerationConfig = interactive["config"]
    materials: tuple[Material, ...] = interactive["materials"]
    issues: list[dict[str, Any]] = [issue.as_dict() for issue in loaded.issues]
    issues.extend(issue.as_dict() for issue in extra_issues)
    if result is not None:
        issues.extend(issue.as_dict() for issue in result.issues)
    for response in interactive.get("responses", []):
        issues.extend(response.get("issues", []))
    deduped_issues: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for issue in issues:
        key = (issue["code"], issue["message"], issue["source"], issue["item"])
        if key not in seen:
            seen.add(key)
            deduped_issues.append(issue)
    response_codes = {
        code
        for response in interactive.get("responses", [])
        for code in response.get("codes", [])
    }
    codes = sorted({issue["code"] for issue in deduped_issues} | response_codes)
    complete = result is not None and all(
        len(material_artifacts.channels) == len(CHANNELS)
        for material_artifacts in result.material_artifacts.values()
    ) and len(result.material_artifacts) == len(materials)
    scheduler_events = result.scheduler_events if result is not None else []
    pending_events = [event for event in scheduler_events if event.get("code") == "E_POINT_BUDGET_PENDING"]
    report = {
        "schema_version": 1,
        "report_type": "texture_material_generation_diagnostic",
        "generated_at_unix": time.time(),
        "status": "ok" if complete and not any(code.startswith("E_") for code in codes) else "partial",
        "input": {
            "directory": loaded.input_dir,
            "input_config_hash": input_config_hash,
            "textures_loaded": sorted(loaded.textures),
            "texture_count": len(loaded.textures),
            "materials_loaded": [material.material_id for material in loaded.materials],
            "commands_loaded": len(loaded.commands),
            "resolution": concrete_config.resolution,
            "resolution_tiers": list(concrete_config.resolution_tiers),
            "noise_layer_count": concrete_config.noise_layer_count,
            "mix_mode": concrete_config.material_mix_mode,
            "fixed_point": concrete_config.fixed_point.as_dict(),
            "concurrency": concrete_config.concurrency,
            "point_budget": concrete_config.point_budget,
            "latency_budget_ms": concrete_config.latency_budget_ms,
            "snapshot_commit_interval": concrete_config.snapshot_commit_interval,
        },
        "codes": codes,
        "issues": deduped_issues,
        "layers": _layer_report(result) if result is not None else [],
        "materials": _material_report(result) if result is not None else [],
        "final": result.final_summary() if result is not None else None,
        "error_trace": error_trace,
        "full_rebuild": full_rebuild,
        "cache_ledger": {
            "entries": result.ledger_entries if result is not None else [],
            "recomputed": result.recomputed if result is not None else [],
            "reused": result.reused if result is not None else [],
            "affected_layers": result.affected_layers if result is not None else [],
            "affected_channels": result.affected_channels if result is not None else [],
        },
        "scheduler": {
            "point_budget": concrete_config.point_budget,
            "max_concurrency": concrete_config.max_concurrency,
            "used_concurrency": concrete_config.concurrency,
            "batch_count": len(scheduler_events),
            "pending_batch_count": len(pending_events),
            "pending_code": "E_POINT_BUDGET_PENDING" if pending_events else None,
            "events": scheduler_events,
        },
        "interactive": {
            "commands_total": interactive.get("commands_total"),
            "commands_started_at": interactive.get("commands_started_at"),
            "commands_processed": interactive.get("commands_processed"),
            "stale_responses": interactive.get("stale_responses"),
            "within_budget_responses": interactive.get("within_budget_responses"),
            "response_codes": sorted(response_codes),
            "responses": interactive.get("responses", []),
            "resume": interactive.get("resume"),
            "final_snapshot": interactive.get("final_snapshot"),
        },
        "injections": injection_report,
        "snapshot": snapshot_summary,
        "artifacts": artifact_paths,
    }
    return _json_safe(report)


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    atomic_write_json(path, report)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value
