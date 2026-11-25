"""End-to-end acceptance verification used by the CLI and tests."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_commands, load_injections
from .diagnostics import build_diagnostics
from .io_utils import atomic_write_json, read_json
from .material import channel_png, composite_png
from .models import MaterialConfig, RenderConfig, TextureData
from .renderer import full_render_serial
from .session import Session


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _image_pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _compare_full_render(session: Session) -> dict[str, Any]:
    if session.state.composite is None:
        return {"name": "incremental_vs_full_render", "ok": False, "reason": "incremental_composite_missing"}
    full_layers, full_channels, full_composite = full_render_serial(
        session.config, session.material, session.texture
    )
    details: list[dict[str, Any]] = []
    ok = True
    for index in sorted(full_layers):
        incremental = session.state.layers.get(index)
        reference = full_layers[index]
        checks = {
            "present": incremental is not None,
            "key": incremental is not None and incremental.key == reference.key,
            "raw": incremental is not None and _array_equal(incremental.raw, reference.raw),
            "quantized": incremental is not None and _array_equal(incremental.quantized, reference.quantized),
            "residual": incremental is not None and _array_equal(incremental.residual, reference.residual),
            "decisions": incremental is not None and _array_equal(incremental.decisions, reference.decisions),
        }
        layer_ok = all(checks.values())
        ok = ok and layer_ok
        details.append({"layer": index, "ok": layer_ok, "checks": checks})
    channel_details: list[dict[str, Any]] = []
    for name, reference in full_channels.items():
        incremental = session.state.channels.get(name)
        checks = {
            "present": incremental is not None,
            "key": incremental is not None and incremental.key == reference.key,
            "array": incremental is not None and _array_equal(incremental.array, reference.array),
            "layer_keys": incremental is not None and incremental.layer_keys == reference.layer_keys,
        }
        channel_ok = all(checks.values())
        ok = ok and channel_ok
        channel_details.append({"channel": name, "ok": channel_ok, "checks": checks})
    composite_checks = {
        "present": session.state.composite is not None,
        "key": session.state.composite is not None and session.state.composite.key == full_composite.key,
        "array": session.state.composite is not None and _array_equal(session.state.composite.array, full_composite.array),
    }
    ok = ok and all(composite_checks.values())

    incremental_image = composite_png(session.state.composite.array, session.state.composite.frac_bits)
    full_image = composite_png(full_composite.array, full_composite.frac_bits)
    png_equal = bool(np.array_equal(_image_pixels(incremental_image), _image_pixels(full_image)))
    ok = ok and png_equal
    return {
        "name": "incremental_vs_full_render",
        "ok": ok,
        "layers": details,
        "channels": channel_details,
        "composite": {"ok": all(composite_checks.values()), "checks": composite_checks},
        "png_pixels_equal": png_equal,
        "incremental_composite_key": session.state.composite.key,
        "full_composite_key": full_composite.key,
        "incremental_image_sha256": hashlib.sha256(incremental_image.tobytes()).hexdigest(),
        "full_image_sha256": hashlib.sha256(full_image.tobytes()).hexdigest(),
        "_images": (incremental_image, full_image),
    }


def _strip_private(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_private(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def verify_project(
    *,
    config_path: str | Path,
    material_path: str | Path,
    commands_path: str | Path,
    output_dir: str | Path,
    injections_path: str | Path | None = None,
    low_budget_config_path: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    commands, command_issues = load_commands(commands_path)
    config_probe = Session.open(config_path, material_path, out / "state_probe", resume=False)
    injections, injection_issues = load_injections(injections_path, len(config_probe.config.layers))
    config_probe._join_job()

    state_inc = out / "state_incremental"
    _clean_dir(state_inc)
    incremental = Session.open(config_path, material_path, state_inc, resume=False, injections=[])
    command_results: list[dict[str, Any]] = []
    for command in commands:
        result = incremental.apply_command(command)
        command_results.append(
            {
                "command": command,
                "ok": result.get("ok", False),
                "errors": result.get("errors", []),
                "preview": result.get("preview", {}),
            }
        )
    final_summary = incremental.finalize(drain=True, force_commit=True)
    comparison = _compare_full_render(incremental)
    incremental_image, full_image = comparison.pop("_images")
    incremental_image.save(out / "incremental.png", format="PNG")
    full_image.save(out / "full_rebuild.png", format="PNG")

    recovery_dir = out / "state_recovery"
    _clean_dir(recovery_dir)
    recovery_writer = Session.open(config_path, material_path, recovery_dir, resume=False)
    prefix_len = min(3, len(commands))
    for command in commands[:prefix_len]:
        recovery_writer.apply_command(command)
    recovery_writer.complete()
    recovery_writer.commit()
    recovery_writer_config = copy.deepcopy(recovery_writer.config)
    tail = commands[prefix_len : prefix_len + 2]
    for command in tail:
        recovery_writer.apply_command(command)
    recovery_writer._join_job()
    recovered = Session.open(config_path, material_path, recovery_dir, resume=True)
    recovered.complete()
    recovered.commit()
    recovery_compare = _compare_full_render(recovered)
    recovery_compare["name"] = "snapshot_recovery"
    recovered_image, recovered_full_image = recovery_compare.pop("_images")
    recovered_image.save(out / "recovered.png", format="PNG")
    recovered_full_image.save(out / "recovered_full_rebuild.png", format="PNG")
    snapshot_image_path = recovery_dir / "output.png"
    snapshot_png_matches = False
    if snapshot_image_path.exists():
        with Image.open(snapshot_image_path) as image:
            snapshot_png_matches = bool(np.array_equal(_image_pixels(image), _image_pixels(recovered_image)))
    recovery_compare["snapshot_root_png_matches_recovered"] = snapshot_png_matches
    recovery_compare["recovered_version"] = recovered.version
    recovery_compare["recovered_resume_source"] = recovered.resume_source
    recovery_compare["prefix_config_hash"] = hashlib.sha256(
        json.dumps(recovery_writer_config.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()

    partial_recovery: dict[str, Any] = {"name": "partial_snapshot_recovery", "ok": False}
    if low_budget_config_path is not None:
        state_low = out / "state_low_budget"
        _clean_dir(state_low)
        low = Session.open(low_budget_config_path, material_path, state_low, resume=False)
        low.advance(wait=True)
        partial_pending = copy.deepcopy(low.state.pending)
        low.commit()
        resumed = Session.open(low_budget_config_path, material_path, state_low, resume=True)
        resumed.complete()
        resumed.commit()
        partial_compare = _compare_full_render(resumed)
        partial_compare["name"] = "partial_snapshot_recovery"
        partial_compare.pop("_images", None)
        partial_compare["pending_at_snapshot"] = partial_pending
        partial_compare["pending_count_at_snapshot"] = len(partial_pending)
        partial_compare["resumed_version"] = resumed.version
        partial_recovery = partial_compare
    partial_recovery["ok"] = bool(partial_recovery.get("ok")) and partial_recovery.get("pending_count_at_snapshot", 0) > 0

    fault_report: dict[str, Any] = {"name": "fault_isolation", "ok": False}
    if injections:
        state_fault = out / "state_fault"
        _clean_dir(state_fault)
        fault = Session.open(config_path, material_path, state_fault, resume=False, injections=injections)
        fault.advance(wait=True)
        failure_codes = {item.get("code") for item in fault.state.failures}
        expected_codes = {item["code"] for item in injections}
        hit_codes = {item.get("code") for item in fault.state.injection_hits}
        completed_layers = sorted(fault.state.layers)
        fault_report = {
            "name": "fault_isolation",
            "ok": expected_codes.issubset(failure_codes)
            and expected_codes.issubset(hit_codes)
            and len(completed_layers) >= max(1, len(fault.config.layers) - 3),
            "expected_codes": sorted(expected_codes),
            "failure_codes": sorted(code for code in failure_codes if code),
            "injection_hit_codes": sorted(code for code in hit_codes if code),
            "completed_layers": completed_layers,
            "failures": fault.state.failures,
            "injection_hits": fault.state.injection_hits,
        }

    diagnostics_dir = out / "diagnostics"
    _clean_dir(diagnostics_dir)
    diagnostics = build_diagnostics(
        incremental.config,
        incremental.material,
        incremental.texture,
        diagnostics_dir,
        quant_steps=(4, 8),
        extra={
            "injection_hits": fault_report.get("injection_hits", []),
            "failures": fault_report.get("failures", []),
        },
    )

    checks = [comparison, recovery_compare, partial_recovery, fault_report]
    report: dict[str, Any] = {
        "ok": all(bool(item.get("ok")) for item in checks),
        "config": str(Path(config_path).resolve()),
        "material": str(Path(material_path).resolve()),
        "commands": str(Path(commands_path).resolve()),
        "injections": str(Path(injections_path).resolve()) if injections_path else None,
        "command_issues": command_issues + injection_issues + incremental.issues,
        "command_results": command_results,
        "final_summary": final_summary,
        "checks": _strip_private(checks),
        "diagnostics_file": str(diagnostics_dir / "diagnostics.json"),
        "diagnostics_summary": {
            "first_exceed_stage": diagnostics["error_propagation"].get("first_exceed_stage"),
            "first_exceed_count": diagnostics["error_propagation"].get("first_exceed_count"),
            "quant_mismatch_pixels": diagnostics.get("quant_comparison", {}).get("mismatch_pixels") if diagnostics.get("quant_comparison") else None,
            "quant_classifications": diagnostics.get("quant_comparison", {}).get("classifications") if diagnostics.get("quant_comparison") else None,
        },
    }
    atomic_write_json(out / "report.json", report)
    return report
