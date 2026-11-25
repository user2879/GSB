from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .editor import apply_command
from .engine import build_reconstruction, canonical_state_blob
from .snapshot import SnapshotStore
from .synthetic import generate_project
from .validation import load_project


def tkinter_status() -> dict[str, Any]:
    try:
        import tkinter  # noqa: F401

        return {"available": True, "error": None}
    except Exception as exc:  # environment may lack libtk
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _copy_sample(root: Path, name: str) -> Path:
    destination = root / "runtime" / "acceptance" / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        root,
        destination,
        ignore=shutil.ignore_patterns("runtime", ".git", "__pycache__", "results"),
    )
    return destination


def _build(path: Path, previous: dict[str, Any] | None = None, workers: int = 1, **budget: Any):
    loaded = load_project(path)
    if not loaded.ok:
        raise AssertionError({"report": loaded.report(), "failures": [f.to_dict() for f in loaded.failures]})
    return loaded, build_reconstruction(loaded, previous_state=previous, workers=workers, **budget)


def _scenario(root: Path, name: str, command: dict[str, Any], baseline_state: dict[str, Any]) -> dict[str, Any]:
    path = _copy_sample(root, f"scenario-{name}")
    apply_command(path, command)
    loaded, incremental = _build(path, baseline_state, workers=3)
    _, fresh = _build(path, None, workers=1)
    inc_hash = canonical_state_blob(incremental["state"])
    fresh_hash = canonical_state_blob(fresh["state"])
    return {
        "name": name,
        "command_id": command.get("id"),
        "exact_match": inc_hash == fresh_hash,
        "incremental_hash": inc_hash,
        "fresh_hash": fresh_hash,
        "ledger_counts": incremental["ledger"]["counts"],
        "features_recomputed": incremental["ledger"]["features"]["recomputed"],
        "pairs_recomputed": incremental["ledger"]["pairs"]["recomputed"],
        "points_recomputed": incremental["ledger"]["points"]["recomputed"],
        "point_gate_changed": incremental["ledger"]["points"]["gate_changed"],
        "points_removed": incremental["ledger"]["points"]["removed"],
        "depth_recomputed_count": len(incremental["ledger"]["depth_pixels"]["recomputed"]),
        "voxels_recomputed_count": len(incremental["ledger"]["voxels"]["recomputed"]),
        "voxels_removed_count": len(incremental["ledger"]["voxels"]["removed"]),
    }


def _command_stream(root: Path, commands_doc: dict[str, Any], baseline_state: dict[str, Any]) -> list[dict[str, Any]]:
    path = _copy_sample(root, "command-stream")
    previous = baseline_state
    steps = []
    for index, command in enumerate(commands_doc["commands"]):
        apply_command(path, command)
        workers = 1 + (index % 4)
        _, incremental = _build(path, previous, workers=workers)
        _, fresh = _build(path, None, workers=1)
        inc_hash = canonical_state_blob(incremental["state"])
        fresh_hash = canonical_state_blob(fresh["state"])
        steps.append(
            {
                "command_id": command["id"],
                "op": command["op"],
                "workers": workers,
                "exact_match": inc_hash == fresh_hash,
                "ledger_counts": incremental["ledger"]["counts"],
            }
        )
        previous = incremental["state"]
    return steps


def _budget_checks(root: Path) -> list[dict[str, Any]]:
    cases = [
        {"global_budget": 100000, "view_quota": 100000, "workers": 1},
        {"global_budget": 100000, "view_quota": 100000, "workers": 4},
        {"global_budget": 24, "view_quota": 100000, "workers": 1},
        {"global_budget": 24, "view_quota": 100000, "workers": 4},
        {"global_budget": 12, "view_quota": 4, "workers": 1},
        {"global_budget": 12, "view_quota": 4, "workers": 4},
        {"global_budget": 0, "view_quota": 100000, "workers": 2},
    ]
    results = []
    for case in cases:
        workers = case.pop("workers")
        _, serial = _build(root, workers=1, **case)
        _, parallel = _build(root, workers=workers, **case)
        serial_hash = canonical_state_blob(serial["state"])
        parallel_hash = canonical_state_blob(parallel["state"])
        results.append(
            {
                **case,
                "workers_compared": workers,
                "exact_match": serial_hash == parallel_hash,
                "hash": serial_hash,
                "todo": serial["state"]["todo_queue"][:12],
                "todo_count": len(serial["state"]["todo_queue"]),
                "budget_ledger": serial["ledger"]["budget"],
            }
        )
    return results


def _parallel_checks(root: Path, loaded: Any) -> list[dict[str, Any]]:
    baseline = build_reconstruction(loaded, workers=1)
    expected = canonical_state_blob(baseline["state"])
    results = [{"workers": 1, "exact_match": True, "hash": expected}]
    for workers in (2, 3, 4, 8):
        result = build_reconstruction(loaded, workers=workers)
        results.append(
            {
                "workers": workers,
                "exact_match": canonical_state_blob(result["state"]) == expected,
                "hash": canonical_state_blob(result["state"]),
            }
        )
    return results


def _snapshot_check(root: Path, loaded: Any, full: dict[str, Any]) -> dict[str, Any]:
    snapshot_root = root / "runtime" / "acceptance-snapshots"
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    store = SnapshotStore(snapshot_root)
    first = store.commit(validation_report=loaded.report(), state=full["state"], ledger=full["ledger"], global_version=1)
    second = store.commit(validation_report=loaded.report(), state=full["state"], ledger=full["ledger"], global_version=2)
    healthy = store.recover()
    version_two = Path(second["snapshot"])
    target = version_two / "voxels.json"
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    recovery = store.recover()
    return {
        "commit_v1": {"status": first["status"], "version": first["manifest"]["global_version"]},
        "commit_v2": {"status": second["status"], "version": second["manifest"]["global_version"]},
        "healthy_check": healthy,
        "after_tamper_recovery": recovery,
        "recovered_to_v1": recovery.get("global_version") == 1 and recovery.get("status") == "restored",
    }


def _invalid_input_checks(root: Path) -> list[dict[str, Any]]:
    cases = []

    def record(case: str, path: Path):
        loaded = load_project(path)
        cases.append(
            {
                "case": case,
                "ok": loaded.ok,
                "code": loaded.code,
                "business_code": loaded.business_code,
                "first_failure": loaded.failures[0].to_dict() if loaded.failures else None,
            }
        )

    bad_png = _copy_sample(root, "invalid-bad-signature")
    image = bad_png / "data" / "views" / "view_00.png"
    image.write_bytes(b"NOTPNG" + image.read_bytes()[8:])
    record("bad_png_signature", bad_png)

    bad_dimensions = _copy_sample(root, "invalid-dimensions")
    from .png_codec import write_png_rgb
    import numpy as np

    write_png_rgb(bad_dimensions / "data" / "views" / "view_00.png", np.zeros((8, 8, 3), dtype=np.uint8))
    record("wrong_dimensions", bad_dimensions)

    singular = _copy_sample(root, "invalid-singular-k")
    cameras_path = singular / "data" / "cameras.json"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    cameras["views"][0]["K"] = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    cameras_path.write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    record("singular_intrinsic", singular)

    nonorthogonal = _copy_sample(root, "invalid-r")
    cameras_path = nonorthogonal / "data" / "cameras.json"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    cameras["views"][0]["R"] = [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    cameras_path.write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    record("nonorthogonal_rotation", nonorthogonal)

    duplicate = _copy_sample(root, "invalid-duplicate-index")
    cameras_path = duplicate / "data" / "cameras.json"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    cameras["views"][1]["index"] = 0
    cameras_path.write_text(json.dumps(cameras, indent=2), encoding="utf-8")
    record("duplicate_view_index", duplicate)

    bad_config = _copy_sample(root, "invalid-config")
    config_path = bad_config / "config" / "reconstruction.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["min_parallax"] = 0
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    record("non_positive_parallax", bad_config)
    return cases


def run_acceptance(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    generate_project(root)
    loaded = load_project(root)
    report: dict[str, Any] = {
        "ok": False,
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "numpy_min_required": "1.24",
        "tkinter": tkinter_status(),
        "root": str(root),
        "validation": loaded.report(),
    }
    if not loaded.ok:
        report["failures"] = [f.to_dict() for f in loaded.failures]
        return report

    full = build_reconstruction(loaded, workers=1)
    full_hash = canonical_state_blob(full["state"])
    unchanged = build_reconstruction(loaded, previous_state=full["state"], workers=4)
    unchanged_hash = canonical_state_blob(unchanged["state"])

    commands_doc = json.loads((root / "config" / "edit_commands.json").read_text(encoding="utf-8"))
    command_by_id = {item["id"]: item for item in commands_doc["commands"]}
    scenarios = [
        _scenario(root, "replace-view", command_by_id["cmd-03-replace-view"], full["state"]),
        _scenario(root, "extrinsic", command_by_id["cmd-02-extrinsic"], full["state"]),
        _scenario(root, "min-parallax", command_by_id["cmd-01-min-parallax"], full["state"]),
        _scenario(root, "voxel-size", command_by_id["cmd-04-voxel-size"], full["state"]),
        _scenario(root, "threshold", command_by_id["cmd-05-residual-threshold"], full["state"]),
    ]
    stream_steps = _command_stream(root, commands_doc, full["state"])
    budget_checks = _budget_checks(root)
    parallel_checks = _parallel_checks(root, loaded)
    snapshot_check = _snapshot_check(root, loaded, full)
    invalid_inputs = _invalid_input_checks(root)

    all_exact = (
        unchanged_hash == full_hash
        and all(item["exact_match"] for item in scenarios)
        and all(step["exact_match"] for step in stream_steps)
        and all(item["exact_match"] for item in budget_checks)
        and all(item["exact_match"] for item in parallel_checks)
    )
    invalid_ok = all(not item["ok"] and item["first_failure"] for item in invalid_inputs)
    report.update(
        {
            "ok": all_exact and invalid_ok and snapshot_check["recovered_to_v1"],
            "full_state_hash": full_hash,
            "metrics": full["state"]["metrics"],
            "full_ledger_counts": full["ledger"]["counts"],
            "unchanged_incremental": {
                "exact_match": unchanged_hash == full_hash,
                "hash": unchanged_hash,
                "ledger_counts": unchanged["ledger"]["counts"],
            },
            "parallel_checks": parallel_checks,
            "scenarios": scenarios,
            "command_stream_steps": stream_steps,
            "budget_checks": budget_checks,
            "snapshot_check": snapshot_check,
            "invalid_input_checks": invalid_inputs,
            "invalid_inputs_all_rejected": invalid_ok,
        }
    )
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    output = results_dir / "acceptance_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    summary_path = results_dir / "acceptance_summary.txt"
    summary_path.write_text(_format_summary(report), encoding="utf-8")
    return report


def _format_summary(report: dict[str, Any]) -> str:
    lines = [
        "MVRV acceptance summary",
        f"ok={report['ok']} python={report['python']} numpy={report['numpy']}",
        f"tkinter={report['tkinter']['available']} {report['tkinter']['error'] or ''}",
        f"state_hash={report.get('full_state_hash')}",
        f"metrics={json.dumps(report.get('metrics', {}), sort_keys=True)}",
        f"unchanged_exact={report.get('unchanged_incremental', {}).get('exact_match')}",
        f"scenarios_all_exact={all(x['exact_match'] for x in report.get('scenarios', []))}",
        f"stream_all_exact={all(x['exact_match'] for x in report.get('command_stream_steps', []))}",
        f"budget_all_exact={all(x['exact_match'] for x in report.get('budget_checks', []))}",
        f"parallel_all_exact={all(x['exact_match'] for x in report.get('parallel_checks', []))}",
        f"snapshot_recovery={report.get('snapshot_check', {}).get('recovered_to_v1')}",
        f"invalid_inputs_all_rejected={report.get('invalid_inputs_all_rejected')}",
    ]
    for scenario in report.get("scenarios", []):
        lines.append(
            f"scenario {scenario['name']}: exact={scenario['exact_match']} counts={scenario['ledger_counts']}"
        )
    return "\n".join(lines) + "\n"
