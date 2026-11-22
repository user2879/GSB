"""Repeatable, local-only acceptance demonstration."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .errors import DiagnosticError, E_FORMAT, E_INTEGRITY, E_STATE, E_VERSION
from .injection import InjectionManager
from .manifest import (
    FORMAT_VERSIONS,
    build_manifest_model,
    decode_manifest,
    encode_manifest_text,
)
from .store import Session, SimulatedInterruption
from .util import atomic_write_json, make_tree_writable, sha256_file
from .workflow import execute_steps, prepare_samples


class DemoFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DemoFailure(message)


def _fill(value: int) -> str:
    return f"{value & 0xFF:02x}"


def _safe_reset(path: Path, workspace: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError:
        raise DemoFailure(f"refusing to reset path outside workspace: {resolved}")
    if resolved.exists():
        make_tree_writable(resolved)
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _new_path_session(root: Path) -> Session:
    session = Session.initialize(root, block_size=512, block_count=16, threshold=0.5, reset=True)
    steps: list[dict[str, Any]] = [
        {"action": "snapshot", "snapshot_id": "s0", "parent_snapshot_id": None},
        {"action": "write", "block": 0, "fill_hex": _fill(0x11)},
        {"action": "write", "block": 5, "fill_hex": _fill(0x12)},
        {"action": "snapshot", "snapshot_id": "s1", "parent_snapshot_id": "s0"},
        {"action": "write", "block": 1, "fill_hex": _fill(0x21)},
        {"action": "write", "block": 2, "fill_hex": _fill(0x22)},
        {"action": "write", "block": 3, "fill_hex": _fill(0x23)},
        {"action": "write", "block": 4, "fill_hex": _fill(0x24)},
        {"action": "write", "block": 6, "fill_hex": _fill(0x25)},
        {"action": "write", "block": 7, "fill_hex": _fill(0x26)},
    ]
    execute_steps(session, steps)
    return session


def _new_manifest_session(root: Path) -> Session:
    session = Session.initialize(root, block_size=512, block_count=12, threshold=0.25, reset=True)
    steps = [
        {"action": "snapshot", "snapshot_id": "m0", "parent_snapshot_id": None},
        {"action": "write", "block": 1, "fill_hex": _fill(0x31)},
        {"action": "write", "block": 4, "fill_hex": _fill(0x32)},
        {"action": "snapshot", "snapshot_id": "m1", "parent_snapshot_id": "m0"},
        {"action": "write", "block": 2, "fill_hex": _fill(0x41)},
        {"action": "write", "block": 7, "fill_hex": _fill(0x42)},
        {"action": "snapshot", "snapshot_id": "m2", "parent_snapshot_id": "m1"},
    ]
    execute_steps(session, steps)
    return session


def run_demo(workspace: str | Path = ".") -> dict[str, Any]:
    started = time.time()
    workspace_path = Path(workspace).resolve()
    artifacts = workspace_path / "artifacts"
    run_root = artifacts / "demo-run"
    _safe_reset(run_root, workspace_path)
    report: dict[str, Any] = {
        "ok": False,
        "component": "snapdiag",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "workspace": str(workspace_path),
        },
        "checks": [],
    }

    def check(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        started_check = time.time()
        try:
            result = function()
            entry = {
                "name": name,
                "ok": True,
                "duration_seconds": round(time.time() - started_check, 3),
                "result": result,
            }
        except Exception as exc:
            entry = {
                "name": name,
                "ok": False,
                "duration_seconds": round(time.time() - started_check, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["checks"].append(entry)
            raise
        report["checks"].append(entry)
        return result

    # ---------------------------------------------------------- samples
    sample_results = check(
        "prepare_small_large_high_churn_samples",
        lambda: prepare_samples(workspace_path, reset=True),
    )
    report["samples"] = sample_results
    _require(
        sample_results["small"]["block_count"] == 16
        and sample_results["large"]["block_count"] == 1024
        and sample_results["high_churn"]["block_count"] == 128,
        "sample geometries are not the expected small/large/high-churn set",
    )

    manifest_session = _new_manifest_session(run_root / "manifest-replacement")
    check(
        "create_multi_snapshot_session",
        lambda: {
            "snapshots": [row["snapshot_id"] for row in manifest_session.snapshot_rows()],
            "state_version": manifest_session.state_version,
        },
    )

    # ------------------------------------------------------ normalization
    normalized_dir = run_root / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    def normalize_versions() -> dict[str, Any]:
        base_model = manifest_session.build_manifest_model()
        records: dict[str, Any] = {}
        fingerprints: dict[str, str] = {}
        for version in FORMAT_VERSIONS:
            text = encode_manifest_text(base_model, version)
            output = normalized_dir / f"{version}.json"
            output.write_text(text, encoding="utf-8")
            decoded = decode_manifest(text, resolver=manifest_session)
            fingerprints[version] = decoded.fingerprint()
            records[version] = {
                "source_format": decoded.source_format,
                "filled_fields": list(decoded.filled_fields),
                "fill_basis": dict(sorted(decoded.fill_basis.items())),
                "manifest_digest": f"sha256:{decoded.manifest_digest}",
                "snapshot_count": len(decoded.snapshots),
                "output": str(output),
            }
        _require(len(set(fingerprints.values())) == 1, "m1/m2/m3 normalized fingerprints differ")
        _require("format_version" not in records["m3"]["filled_fields"], "m3 should not need format fills")
        _require(records["m1"]["filled_fields"], "m1 normalization must report filled fields")
        return {
            "fingerprints": fingerprints,
            "all_equal": True,
            "formats": records,
        }

    report["normalization"] = check("three_version_manifest_normalization", normalize_versions)

    def replacement_test() -> dict[str, Any]:
        model = manifest_session.build_manifest_model()
        full_path = normalized_dir / "replacement-full.m3.json"
        full_path.write_text(encode_manifest_text(model, "m3"), encoding="utf-8")
        subset = build_manifest_model(
            block_size=model.block_size,
            block_count=model.block_count,
            snapshots=[snapshot for snapshot in model.snapshots if snapshot.snapshot_id != "m2"],
            source_format="m3",
        )
        subset_path = normalized_dir / "replacement-subset.m3.json"
        subset_path.write_text(encode_manifest_text(subset, "m3"), encoding="utf-8")
        remove_report = manifest_session.import_manifest(subset_path, reason="demo-remove")
        _require(remove_report["removed"] == ["m2"], "whole replacement did not list removed snapshot")
        restore_report = manifest_session.import_manifest(full_path, reason="demo-restore")
        _require(restore_report["added"] == ["m2"], "whole replacement did not list added snapshot")
        replaced_snapshots = [
            replace(snapshot, snapshot_version=snapshot.snapshot_version + 10)
            for snapshot in model.snapshots
        ]
        replaced_model = build_manifest_model(
            block_size=model.block_size,
            block_count=model.block_count,
            snapshots=replaced_snapshots,
            source_format="m3",
        )
        replaced_path = normalized_dir / "replacement-version.m3.json"
        replaced_path.write_text(encode_manifest_text(replaced_model, "m3"), encoding="utf-8")
        replaced_report = manifest_session.import_manifest(replaced_path, reason="demo-replace")
        _require(
            "m1" in replaced_report["replaced"],
            "whole replacement did not list replaced snapshot",
        )
        backups = manifest_session.list_manifest_backups()
        _require(bool(backups) and all(item["readonly"] for item in backups), "manifest backups are not read-only")
        return {
            "remove": remove_report,
            "restore": restore_report,
            "replace": replaced_report,
            "backup_count": len(backups),
            "backups_readonly": True,
        }

    report["replacement"] = check("atomic_whole_manifest_replacement", replacement_test)

    # --------------------------------------------------- path equivalence
    def path_equivalence() -> dict[str, Any]:
        sessions = {
            "bitmap": _new_path_session(run_root / "paths" / "bitmap"),
            "full": _new_path_session(run_root / "paths" / "full"),
            "auto_boundary": _new_path_session(run_root / "paths" / "auto-boundary"),
            "auto_low": _new_path_session(run_root / "paths" / "auto-low"),
        }
        plans = {
            "bitmap": sessions["bitmap"].plan_rollback("s0", path="bitmap"),
            "full": sessions["full"].plan_rollback("s0", path="full"),
            "auto_boundary": sessions["auto_boundary"].plan_rollback("s0", path="auto"),
            "auto_low": sessions["auto_low"].plan_rollback("s1", path="auto"),
        }
        _require(plans["auto_boundary"].selected_path == "full", "boundary ratio must select full path")
        _require(plans["auto_boundary"].boundary_hit, "boundary_hit was not recorded at exact threshold")
        _require(plans["auto_low"].selected_path == "bitmap", "low ratio must select bitmap path")
        results = {
            "bitmap": sessions["bitmap"].rollback("s0", path="bitmap").as_dict(),
            "full": sessions["full"].rollback("s0", path="full").as_dict(),
            "auto_boundary": sessions["auto_boundary"].rollback("s0", path="auto").as_dict(),
            "auto_low": sessions["auto_low"].rollback("s1", path="auto").as_dict(),
        }
        hashes = {
            "bitmap": sha256_file(sessions["bitmap"].geometry().image_path),
            "full": sha256_file(sessions["full"].geometry().image_path),
            "auto_boundary": sha256_file(sessions["auto_boundary"].geometry().image_path),
        }
        _require(len(set(hashes.values())) == 1, "bitmap/full/boundary rollback images differ")
        _require(
            results["bitmap"]["restored_blocks"] == results["full"]["restored_blocks"],
            "bitmap/full restored block sets differ",
        )
        _require(
            results["bitmap"]["result_sha256"] == results["full"]["result_sha256"],
            "bitmap/full result digests differ",
        )
        for name, result in results.items():
            _require(result["verification"]["ok"], f"{name} rollback block verification failed")
        return {
            "plans": {name: plan.as_dict() for name, plan in plans.items()},
            "results": results,
            "final_hashes": hashes,
            "bitmap_full_identical": True,
            "restored_blocks": results["bitmap"]["restored_blocks"],
            "boundary_value": plans["auto_boundary"].ratio,
            "boundary_selected": plans["auto_boundary"].selected_path,
        }

    report["path_equivalence"] = check("adaptive_rollback_path_equivalence", path_equivalence)

    # ------------------------------------------------------ interruption
    def interruption_test() -> dict[str, Any]:
        after_prepare = _new_path_session(run_root / "interrupt" / "after-prepare")
        try:
            after_prepare.rollback("s0", path="bitmap", failpoint="after_prepare")
            raise DemoFailure("after_prepare fault point did not interrupt")
        except SimulatedInterruption:
            pass
        recovered = after_prepare.recover_session()
        _require(recovered and recovered[0]["requires_rerun"], "interrupted rollback was not marked unknown")
        blocked_code = None
        try:
            after_prepare.rollback("s0", path="bitmap")
        except DiagnosticError as error:
            blocked_code = error.code
        _require(blocked_code == E_STATE, "incomplete rollback did not require an explicit retry")
        rerun = after_prepare.rollback("s0", path="bitmap", retry=True).as_dict()
        _require(rerun["verification"]["ok"], "retried rollback failed verification")

        after_replace = _new_path_session(run_root / "interrupt" / "after-replace")
        try:
            after_replace.rollback("s0", path="full", failpoint="after_replace")
            raise DemoFailure("after_replace fault point did not interrupt")
        except SimulatedInterruption:
            pass
        recovery_after_replace = after_replace.recover_session()
        _require(
            recovery_after_replace[0]["status"] == "completed"
            and recovery_after_replace[0]["verification"]["ok"],
            "post-replacement interruption was not recovered and verified",
        )
        return {
            "after_prepare_recovery": recovered,
            "retry_required_code": blocked_code,
            "retry_result": rerun,
            "after_replace_recovery": recovery_after_replace,
        }

    report["interruption"] = check("interrupted_rollback_recovery", interruption_test)

    # --------------------------------------------------------- injections
    def injections() -> dict[str, Any]:
        session = Session.open(workspace_path / "samples" / "generated" / "high_churn")
        manager = InjectionManager(session)
        block_injection = manager.corrupt_block_data(target="churn-base", block_index=0)
        block_diagnosis = manager.diagnose("block-corruption", target="churn-base")
        _require(block_diagnosis["observed_code"] == E_INTEGRITY, "block corruption did not report E_INTEGRITY")
        repair = manager.repair_and_verify(target="churn-base")
        _require(repair["ok"], "block corruption repair failed")
        _require(
            repair["block_verification"]["checked_blocks"] == session.block_count,
            "post-repair verification did not check every block",
        )

        truncation = manager.truncate_manifest()
        truncation_code = None
        truncation_details: dict[str, Any] = {}
        try:
            session.read_manifest(strict_version=True)
        except DiagnosticError as error:
            truncation_code = error.code
            truncation_details = error.to_dict()
        _require(truncation_code == E_FORMAT, "manifest truncation did not report E_FORMAT")
        truncation_repair = manager.restore_manifest()

        version_injection = manager.inject_version_conflict()
        version_code = None
        version_details: dict[str, Any] = {}
        try:
            session.read_manifest(strict_version=True)
        except DiagnosticError as error:
            version_code = error.code
            version_details = error.to_dict()
        _require(version_code == E_VERSION, "version conflict did not report E_VERSION")
        version_repair = manager.restore_manifest()
        consistency = session.manifest_consistency()
        _require(consistency["consistent"], "manifest repair did not restore disk consistency")
        return {
            "block": {
                "injection": block_injection,
                "diagnosis": block_diagnosis,
                "repair": repair,
            },
            "truncation": {
                "injection": truncation,
                "code": truncation_code,
                "details": truncation_details,
                "repair": truncation_repair,
            },
            "version": {
                "injection": version_injection,
                "code": version_code,
                "details": version_details,
                "repair": version_repair,
                "consistent_after_repair": consistency["consistent"],
            },
        }

    report["injections"] = check("three_anomaly_injection_and_recovery", injections)

    # ------------------------------------------------------- audit version
    def audit_versioning() -> dict[str, Any]:
        session = _new_path_session(run_root / "audit-version")
        first_audit = session.audit()
        _require(first_audit["status"] == "pass", "initial audit did not pass")
        session.write_block(15, bytes([0x5A]) * session.block_size)
        stale_code = None
        stale_details: dict[str, Any] = {}
        try:
            session.latest_audit(strict=True)
        except DiagnosticError as error:
            stale_code = error.code
            stale_details = error.to_dict()
        _require(stale_code == E_VERSION, "stale audit did not report E_VERSION")
        second_audit = session.audit()
        _require(second_audit["status"] == "pass", "audit after state change did not pass")
        history = session.audit_history()
        _require(len(history) >= 2, "audit conclusions were overwritten instead of versioned")
        return {
            "first_audit": {
                "audit_version": first_audit["audit_version"],
                "status": first_audit["status"],
            },
            "stale_code": stale_code,
            "stale_details": stale_details,
            "second_audit": {
                "audit_version": second_audit["audit_version"],
                "status": second_audit["status"],
            },
            "history": history,
        }

    report["audit_versioning"] = check("versioned_audit_and_stale_detection", audit_versioning)

    report["ok"] = True
    report["duration_seconds"] = round(time.time() - started, 3)
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_path = artifacts / "demo-report.json"
    atomic_write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the snapdiag acceptance demo")
    parser.add_argument("--workspace", default=".")
    arguments = parser.parse_args()
    try:
        result = run_demo(arguments.workspace)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "duration_seconds": result["duration_seconds"],
                "checks": [
                    {"name": item["name"], "ok": item["ok"], "duration_seconds": item["duration_seconds"]}
                    for item in result["checks"]
                ],
                "report_path": result["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
