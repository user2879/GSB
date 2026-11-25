from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .configuration import load_config
from .diagnostics import load_injection_config, run_injection_diagnostic
from .errors import WorkbenchError
from .image_io import write_png
from .samples import generate_workspace
from .session import WorkbenchSession
from .snapshot import SnapshotStore
from .workflows import build_full_final, compare_sessions, load_bundle, open_snapshot_session


def _error_code(func) -> dict[str, Any]:
    try:
        func()
    except WorkbenchError as exc:
        return {"raised": True, **exc.to_dict()["error"]}
    return {"raised": False}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def run_demo(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    written = generate_workspace(root)
    config_path = root / "configs" / "workbench.json"
    commands_path = root / "configs" / "edit_stream.json"
    injection_path = root / "configs" / "injection.json"
    output_dir = root / "outputs"
    report_dir = root / "artifacts" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config, rectangles, pixels, image_fingerprint = load_bundle(config_path)
    from .commands import validate_command_stream
    commands = validate_command_stream(commands_path, config, rectangles)
    store = SnapshotStore(config)
    store.reset_for_demo(remove_all=True)

    budget_story: dict[str, Any] = {}
    session = WorkbenchSession.fresh(config, pixels, image_fingerprint, rectangles)
    session.apply(commands[0])
    first = session.run_batch(budget=6, max_workers=config.max_workers)
    first_manifest = session.commit(store)
    budget_story["first_batch"] = {
        "budget": 6,
        "recomputed": first["recomputed"],
        "pending_count": len(first["pending"]),
        "first_pending": first["pending"][:8],
        "status": first["status"],
        "snapshot": first_manifest["snapshot_dir"],
    }

    restored, restored_manifest = open_snapshot_session(config, pixels, image_fingerprint, rectangles, store)
    assert restored_manifest is not None
    second = restored.run_batch(budget=6, max_workers=config.max_workers)
    second_manifest = restored.commit(store)
    budget_story["restored_cursor"] = {
        "command_cursor": restored_manifest["command_cursor"],
        "applied_command_cursor": restored_manifest["applied_command_cursor"],
        "pending_count": len(restored_manifest["pending"]),
    }
    budget_story["second_batch"] = {
        "reused_count": len(second["reused"]),
        "reused_first": second["reused"][:6],
        "recomputed_count": len(second["recomputed"]),
        "pending_count": len(second["pending"]),
        "status": second["status"],
        "snapshot": second_manifest["snapshot_dir"],
    }

    drain = restored.run_batch(budget=len(restored.planned.order), max_workers=config.max_workers)
    restored.commit(store)
    budget_story["drain"] = {
        "reused_count": len(drain["reused"]),
        "recomputed_count": len(drain["recomputed"]),
        "pending_count": len(drain["pending"]),
        "status": drain["status"],
        "completed_command_cursor": drain["completed_command_cursor"],
    }

    command_summaries = []
    for command in commands[1:]:
        restored.apply(command)
        report = restored.run_batch(budget=len(restored.planned.order), max_workers=config.max_workers)
        manifest = restored.commit(store)
        command_summaries.append({
            "command_id": command.get("id"),
            "op": command["op"],
            "reused_count": len(report["reused"]),
            "recomputed_count": len(report["recomputed"]),
            "pending_count": len(report["pending"]),
            "cache_key_count": len(restored.records),
            "snapshot": manifest["snapshot_dir"],
            "commit_seq": manifest["commit_seq"],
        })

    write_png(output_dir / "incremental_final.png", restored.styled_levels[-1])
    for level in range(config.levels):
        write_png(output_dir / f"styled_L{level}.png", restored.styled_levels[level])

    full = build_full_final(config, pixels, image_fingerprint, rectangles, commands, max_workers=1)
    consistency = compare_sessions(restored, full)
    write_png(output_dir / "full_rebuild_final.png", full.styled_levels[-1])

    stale_checks = _snapshot_stale_probes(root, config, store)
    invalid_inputs = {
        "config": _error_code(lambda: load_config(root / "examples" / "invalid_config.json")),
        "mask": _error_code(lambda: _load_bad_mask(config, root / "examples" / "invalid_mask.json")),
        "tile_geometry": _tile_geometry_probe(root),
    }

    injections = load_injection_config(injection_path, config)
    injection_report = run_injection_diagnostic(config, injections)

    environment = _environment_report()
    report = {
        "ok": bool(consistency["equal"] and injection_report["consistent"]),
        "project_root": str(root),
        "generated_file_count": len(written),
        "environment": environment,
        "config": {
            "path": str(config_path),
            "global_version": config.global_version,
            "config_hash": config.config_hash,
            "levels": config.levels,
            "level_dimensions": [list(dim) for dim in config.level_dimensions],
            "tile_count": len(restored.planned.order),
            "max_workers": config.max_workers,
        },
        "budget_story": budget_story,
        "command_summaries": command_summaries,
        "consistency": consistency,
        "snapshot_stale_checks": stale_checks,
        "invalid_input_checks": invalid_inputs,
        "injection_diagnostic": injection_report,
        "outputs": {
            "incremental_final": str(output_dir / "incremental_final.png"),
            "full_rebuild_final": str(output_dir / "full_rebuild_final.png"),
            "snapshot_manifest": str(store.manifest_path),
        },
    }
    _write_json(report_dir / "verification-report.json", report)
    return report


def _load_bad_mask(config, path: Path) -> None:
    from .masks import load_mask_document
    load_mask_document(path, config.width, config.height)


def _tile_geometry_probe(root: Path) -> dict[str, Any]:
    config_path = root / "configs" / "_tile_geometry_probe.json"
    raw = json.loads((root / "configs" / "workbench.json").read_text(encoding="utf-8"))
    raw["pyramid"]["tile_sizes"] = [4, 9, 16, 32]
    config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    try:
        return _error_code(lambda: load_config(config_path))
    finally:
        config_path.unlink(missing_ok=True)


def _snapshot_stale_probes(root: Path, config, store: SnapshotStore) -> dict[str, Any]:
    current = store.load_manifest()
    compatible = {"compatible": True}
    try:
        store.check_compatible(current)
    except WorkbenchError as exc:
        compatible = {"compatible": False, **exc.to_dict()["error"]}

    version_path = root / "configs" / "_stale_version_probe.json"
    hash_path = root / "configs" / "_stale_hash_probe.json"
    raw = json.loads(config.source_path.read_text(encoding="utf-8"))
    version_raw = dict(raw)
    version_raw["global_version"] = int(raw["global_version"]) + 1
    version_path.write_text(json.dumps(version_raw, indent=2), encoding="utf-8")

    hash_raw = json.loads(json.dumps(raw))
    hash_raw["budget"] = dict(raw.get("budget", {}))
    hash_raw["budget"]["tiles_per_batch"] = int(raw["budget"]["tiles_per_batch"]) + 7
    hash_path.write_text(json.dumps(hash_raw, indent=2), encoding="utf-8")

    def check(path: Path):
        stale_config = load_config(path)
        probe_store = SnapshotStore(stale_config, root=store.root)
        manifest = probe_store.load_manifest()
        probe_store.check_compatible(manifest)

    try:
        version_mismatch = _error_code(lambda: check(version_path))
        hash_mismatch = _error_code(lambda: check(hash_path))
    finally:
        version_path.unlink(missing_ok=True)
        hash_path.unlink(missing_ok=True)
    return {"current": compatible, "version_mismatch": version_mismatch, "hash_mismatch": hash_mismatch}


def _environment_report() -> dict[str, Any]:
    import sys
    import numpy
    import PIL

    tkinter_available = False
    tkinter_error = None
    tk_version = None
    try:
        import tkinter
        tkinter_available = True
        tk_version = str(tkinter.TkVersion)
    except Exception as exc:
        tkinter_error = f"{type(exc).__name__}: {exc}"
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "tkinter_available": tkinter_available,
        "tk_version": tk_version,
        "tkinter_error": tkinter_error,
        "windows_note": "On Windows use the official python.org installer; `python -m tkinter` opens the small Tk demo if Tk is installed.",
    }
