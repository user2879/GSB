"""Repeatable end-to-end demonstration on the local deterministic sample."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .diagnostics import inject_anomaly
from .errors import DiagnosticError
from .paths import WorkspacePaths, atomic_write_json, reset_demo_workspace
from .spec import load_spec
from .store import CacheProofStore


def _sample_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "samples" / "cache_spec.json",
        Path.cwd() / "samples" / "cache_spec.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise DiagnosticError(
        "E_IO", "deterministic sample declaration was not found", field="samples"
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _log_expected_error(store: CacheProofStore, error: DiagnosticError) -> None:
    store.logger.emit(
        "business_error",
        business_code=error.code,
        segment_id=error.segment_id,
        entry_key=error.entry_key,
        field=error.field,
        details=error.to_dict(),
        monotonic_ns=error.monotonic_ns,
    )


def _run_injection_case(root: Path, spec_path: Path, kind: str) -> dict[str, Any]:
    workspace = root / "injection-cases" / kind
    report: dict[str, Any] = {"injection": kind, "workspace": str(workspace)}
    store = CacheProofStore(
        workspace,
        spec=load_spec(spec_path),
        create=True,
        auto_snapshot=False,
    )
    try:
        try:
            inject_anomaly(store, kind)
        except DiagnosticError as exc:
            _log_expected_error(store, exc)
            report.update(
                {
                    "ok": True,
                    "expected_code": exc.code,
                    "segment_id": exc.segment_id,
                    "entry_key": exc.entry_key,
                    "field": exc.field,
                    "details": exc.details,
                    "log": str(store.paths.event_log),
                }
            )
        else:
            report.update({"ok": False, "expected_code": None})
    finally:
        store.close()
    return report


def _run_crash_phase(
    workspace: Path,
    *,
    segment: str,
    key: str,
    size: int,
    seed: str,
) -> dict[str, Any]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    root = str(_project_root())
    environment["PYTHONPATH"] = (
        root if not existing_pythonpath else root + os.pathsep + existing_pythonpath
    )
    command = [
        sys.executable,
        "-m",
        "cacheproof",
        "_crash-worker",
        "--workspace",
        str(workspace),
        "--segment",
        segment,
        "--key",
        key,
        "--size",
        str(size),
        "--seed",
        seed,
        "--failpoint",
        "after_file_replace",
    ]
    completed = subprocess.run(
        command,
        cwd=str(_project_root()),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )
    store = CacheProofStore(workspace, auto_snapshot=False, auto_recover=True)
    try:
        verification = store.verify_all()
        unknown = store.list_unknown_entries()
        snapshot = store.save_snapshot()
        return {
            "ok": completed.returncode == 91,
            "worker_exit_code": completed.returncode,
            "worker_stdout": completed.stdout.strip(),
            "worker_stderr": completed.stderr.strip(),
            "recovery_kind": "post_crash_recovery",
            "exit_report_available": False,
            "recovery_basis": "committed prepare journal plus observed file",
            "unknown_entries": [
                {
                    "segment_id": item["segment_id"],
                    "entry_key": item["entry_key"],
                    "expected_digest": item["expected_digest"],
                    "observed_digest": item["observed_digest"],
                    "observed_size_bytes": item["observed_size_bytes"],
                    "reason": item["reason"],
                }
                for item in unknown
            ],
            "proofs_still_ok": verification["ok"],
            "snapshot_digest": snapshot["content_digest"],
        }
    finally:
        store.close()


def run_demo(
    workspace: str | Path,
    *,
    reset: bool = True,
    report_path: Path | None = None,
) -> dict[str, Any]:
    paths = WorkspacePaths(workspace)
    if reset:
        reset_demo_workspace(paths)
    paths.ensure()
    spec_path = _sample_path()
    spec = load_spec(spec_path)
    phases: list[dict[str, Any]] = []

    store = CacheProofStore(
        paths.root, spec=spec, create=True, auto_snapshot=False
    )
    try:
        initial = store.verify_all()
        phases.append(
            {
                "phase": "initialize",
                "code": "OK",
                "ledger": initial["ledger"],
                "segments": [
                    {
                        "segment_id": segment["segment_id"],
                        "capacity_bytes": segment["capacity_bytes"],
                        "used_bytes": segment["used_bytes"],
                        "remaining_bytes": segment["remaining_bytes"],
                        "proof_content_digest": segment["proof_content_digest"],
                    }
                    for segment in initial["segments"]
                ],
            }
        )
        proof_before = store.build_proof("seg-alpha")["content_digest"]
        try:
            store.put_entry(
                "seg-alpha", "alpha/too-large", 10_000, "oversized-demo"
            )
        except DiagnosticError as exc:
            assertion = store.build_proof("seg-alpha")["content_digest"]
            phases.append(
                {
                    "phase": "capacity_rejection",
                    "code": exc.code,
                    "segment_id": exc.segment_id,
                    "entry_key": exc.entry_key,
                    "required_bytes": exc.details.get("required_bytes"),
                    "available_bytes": exc.details.get("available_bytes"),
                    "proof_unchanged": proof_before == assertion,
                    "proof_content_digest": assertion,
                }
            )
        else:
            raise DiagnosticError(
                "E_IO", "oversized demo write was not rejected", field="capacity"
            )

        put_result = store.put_entry(
            "seg-alpha", "alpha/demo-new", 40, "alpha-demo-new-v1"
        )
        delete_result = store.delete_entry("seg-alpha", "alpha/a")
        after_mutation = store.verify_all()
        phases.append(
            {
                "phase": "write_delete",
                "code": "OK",
                "put_cursor": put_result.committed_cursor,
                "delete_cursor": delete_result.committed_cursor,
                "segments": [
                    {
                        "segment_id": segment["segment_id"],
                        "capacity_bytes": segment["capacity_bytes"],
                        "used_bytes": segment["used_bytes"],
                        "remaining_bytes": segment["remaining_bytes"],
                        "entry_keys": [
                            entry["key"] for entry in segment["entries"]
                        ],
                        "proof_content_digest": segment["proof_content_digest"],
                    }
                    for segment in after_mutation["segments"]
                ],
            }
        )

        orders = store.get_view("view-orders")
        audit = store.get_view("view-audit")
        phases.append(
            {
                "phase": "view_rebuild",
                "code": "OK",
                "views": [
                    {
                        "view_id": result["view_id"],
                        "old_view_version": result["differences"].get(
                            "old_view_version"
                        ),
                        "new_view_version": result["view_version"],
                        "rebuilt": result["rebuilt"],
                        "differences": result["differences"],
                        "snapshot_digest": result["snapshot_digest"],
                    }
                    for result in sorted(
                        (orders, audit), key=lambda item: item["view_id"]
                    )
                ],
            }
        )

        saved = store.save_snapshot()
        store.put_entry("seg-alpha", "alpha/replay", 8, "alpha-replay-v1")
        restored = store.restore_snapshot()
        phases.append(
            {
                "phase": "snapshot_replay",
                "code": "OK",
                "saved_cursor": saved["completed_cursor"],
                "replayed_operations": restored["replayed_operations"],
                "snapshot_replaced": restored["snapshot_replaced"],
                "operation_codes": restored["operation_codes"],
                "changed_fields": restored["differences"]["changes"],
                "new_snapshot_digest": restored["content_digest"],
            }
        )
    finally:
        store.close()

    crash = _run_crash_phase(
        paths.root,
        segment="seg-alpha",
        key="alpha/crash",
        size=24,
        seed="alpha-crash-v1",
    )
    crash["phase"] = "crash_recovery"
    phases.append(crash)

    injection_reports = [
        _run_injection_case(paths.root, spec_path, kind)
        for kind in ("truncate", "capacity", "ledger-digest")
    ]
    phases.append(
        {
            "phase": "structured_injections",
            "code": "OK",
            "cases": injection_reports,
        }
    )

    result = {
        "ok": all(phase.get("ok", True) for phase in phases),
        "code": "OK",
        "workspace": str(paths.root),
        "sample": str(spec_path),
        "log": str(paths.event_log),
        "phases": phases,
    }
    if report_path is not None:
        destination = report_path.expanduser().resolve()
        atomic_write_json(destination, result, paths.tmp)
        result["report_path"] = str(destination)
    return result
