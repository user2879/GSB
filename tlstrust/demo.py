"""Deterministic three-scenario demonstration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .canonical import read_json, write_canonical_json
from .errors import E_ROLLBACK, WorkbenchError
from .workspace import Workspace, capture_snapshot, prepare_workspace, run_validation


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "revision": result["revision"],
        "conclusion": result["conclusion"],
        "failure_category": result["failure_category"],
        "first_failure": result.get("first_failure"),
        "index": result["index"],
        "result_digest": result["result_digest"],
    }


def _flip_summary(flips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "from_revision",
        "to_revision",
        "kind",
        "from_conclusion",
        "to_conclusion",
        "from_failure_category",
        "to_failure_category",
    )
    return [{key: flip.get(key) for key in keys} for flip in sorted(flips, key=lambda item: (item["from_revision"], item["to_revision"], item["kind"]))]


def _family_summary(workspace: Workspace) -> dict[str, Any]:
    family = workspace.family
    return {
        "roots": [
            {
                "id": item["id"],
                "subject": item["subject"],
                "fingerprint": item["fingerprint"],
                "serial_number": item["serial_number"],
            }
            for item in family.data["roots"]
        ],
        "intermediates": [
            {
                "id": item["id"],
                "subject": item["subject"],
                "fingerprint": item["fingerprint"],
                "path_len": item["path_len"],
            }
            for item in family.data["intermediates"]
        ],
        "leaves": [
            {
                "id": item["id"],
                "profile": item["profile"],
                "fingerprint": item["fingerprint"],
                "serial_number": item["serial_number"],
                "chain_fingerprints": item["chain_fingerprints"],
            }
            for item in family.data["leaves"]
        ],
    }


def _session_ref(session: dict[str, Any], path: str | None) -> dict[str, Any]:
    return {
        "format_version": session["format_version"],
        "content_digest": session["content_digest"],
        "input_snapshot_digest": session["input_snapshot_digest"],
        "revision_chain_digest": session["revision_chain_digest"],
        "path": path,
    }


def run_demo(
    root: str | Path,
    *,
    fresh: bool = True,
    key_bits: int = 2048,
    port: int = 0,
    roots: int = 2,
    intermediate_levels: int = 2,
    leaves: int = 1,
    validity_days: int = 90,
    seed: str = "tlstrust-deterministic-v1",
    not_before_ms: int | None = None,
) -> dict[str, Any]:
    workspace = prepare_workspace(
        root,
        fresh=fresh,
        key_bits=key_bits,
        port=port,
        roots=roots,
        intermediate_levels=intermediate_levels,
        leaves=leaves,
        validity_days=validity_days,
        seed=seed,
        not_before_ms=not_before_ms,
    )
    log = workspace.event_log()
    ledger = workspace.open_ledger()
    try:
        document = ledger.current_trust_document()
        if document is None:
            raise RuntimeError("workspace preparation did not create a trust version")
        snapshot = capture_snapshot(workspace)
        input_snapshot_path = workspace.snapshot_dir / f"input-{snapshot['leaf_id']}.json"
        initial = run_validation(ledger, document, snapshot, source_path=str(workspace.trust_path))
        initial_session_file = ledger.create_session_snapshot(
            initial["run_id"], workspace.session_dir
        )
        initial_session = initial_session_file["payload"]
        view1 = ledger.build_current_view()
        log.write(
            "demo.scenario",
            scenario=1,
            trust_version=initial["trust_version"],
            run_id=initial["run_id"],
            chain_digest=snapshot["chain_digest"],
        )
        scenario1 = {
            "scenario": "initial_full_revision_validation",
            "certificate_family": _family_summary(workspace),
            "handshake": {
                "capture_method": snapshot["capture_method"],
                "tls_version": snapshot["negotiated"]["tls_version"],
                "cipher_name": snapshot["negotiated"]["cipher_name"],
                "chain_fingerprints": snapshot["chain_fingerprints"],
                "chain_serial_numbers": snapshot["chain_serial_numbers"],
                "chain_digest": snapshot["chain_digest"],
                "snapshot_digest": snapshot["snapshot_digest"],
                "input_snapshot_path": str(input_snapshot_path),
            },
            "trust_version": initial["trust_version"],
            "revision_chain_digest": document.chain_digest,
            "run_id": initial["run_id"],
            "results": [_result_summary(item) for item in initial["results"]],
            "flips": _flip_summary(initial["flips"]),
            "session_snapshot": _session_ref(initial_session, initial_session_file["path"]),
            "current_view_digest": view1["view_digest"],
        }

        rollback = ledger.rollback_to_revision(3)
        rolled_document = ledger.current_trust_document()
        if rolled_document is None:
            raise RuntimeError("rollback did not create a current trust document")
        rolled = run_validation(
            ledger,
            rolled_document,
            snapshot,
            source_path=str(workspace.trust_path),
        )
        rolled_session_file = ledger.create_session_snapshot(
            rolled["run_id"], workspace.session_dir
        )
        rolled_session = rolled_session_file["payload"]
        invalid_rollback: dict[str, Any]
        try:
            ledger.rollback_to_revision(99)
        except WorkbenchError as exc:
            if exc.code != E_ROLLBACK:
                raise
            log.error("demo.rollback_missing_target", exc)
            invalid_rollback = exc.to_dict()
        else:
            raise AssertionError("invalid rollback unexpectedly succeeded")
        view2 = ledger.build_current_view()
        all_snapshots = ledger.list_session_snapshots(current_only=False)
        scenario2 = {
            "scenario": "trust_library_rollback_and_revalidation",
            "rollback_target_revision": 3,
            "new_trust_version": rollback["version"],
            "revision_chain_digest": rolled_document.chain_digest,
            "replacement_audit": rollback["summary"],
            "results": [_result_summary(item) for item in rolled["results"]],
            "flips": _flip_summary(rolled["flips"]),
            "session_snapshot": _session_ref(rolled_session, rolled_session_file["path"]),
            "invalid_rollback": invalid_rollback,
            "stale_isolation": {
                "old_trust_versions": [
                    item
                    for item in ledger.list_trust_versions()
                    if item["stale"] == 1
                ],
                "stale_session_snapshots": [
                    item["content_digest"] for item in all_snapshots if item["stale"] == 1
                ],
                "current_view_run_ids": [run["run_id"] for run in view2["runs"]],
                "old_run_id": initial["run_id"],
            },
            "current_view_digest": view2["view_digest"],
        }
        log.write(
            "demo.scenario",
            scenario=2,
            trust_version=rollback["version"],
            run_id=rolled["run_id"],
            rollback_target=3,
        )

        # Scenario 3 starts with a real abrupt process termination.  The worker
        # commits two revisions, then calls os._exit(90) without finalization.
        ledger.close()
        marker_path = workspace.root / "snapshots" / "crash-marker.json"
        command = [
            sys.executable,
            "-m",
            "tlstrust.crash_worker",
            "--ledger",
            str(workspace.ledger_path),
            "--snapshot",
            str(input_snapshot_path),
            "--marker",
            str(marker_path),
            "--crash-after",
            "2",
        ]
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 90:
            raise RuntimeError(
                f"crash worker did not terminate as expected: code={completed.returncode} "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        marker = read_json(marker_path)
        ledger = workspace.open_ledger()
        startup_recovery = ledger.startup_recovery
        incomplete_run = ledger.get_run(marker["run_id"])
        incomplete_rows = [
            {
                "revision": row["revision"],
                "committed": row["committed"],
                "conclusion": row["conclusion"],
            }
            for row in incomplete_run["results"]
        ]
        current_document = ledger.current_trust_document()
        if current_document is None:
            raise RuntimeError("current trust revisions disappeared after crash recovery")
        rerun = run_validation(
            ledger,
            current_document,
            snapshot,
            source_path=str(workspace.trust_path),
        )
        rerun_session_file = ledger.create_session_snapshot(rerun["run_id"], workspace.session_dir)
        rerun_session = rerun_session_file["payload"]
        replay = ledger.replay_session(rerun_session, workspace.family)
        historical_replay = ledger.replay_session(initial_session, workspace.family)
        view3 = ledger.build_current_view()
        scenario3 = {
            "scenario": "forced_termination_recovery_and_snapshot_replay",
            "crash_exit_code": completed.returncode,
            "crash_marker": marker,
            "startup_recovery": startup_recovery,
            "incomplete_run": {
                "run_id": marker["run_id"],
                "status": incomplete_run["status"],
                "revision_rows": incomplete_rows,
                "unknown_revisions": [
                    row["revision"] for row in incomplete_rows if row["committed"] == 0
                ],
            },
            "rerun": {
                "run_id": rerun["run_id"],
                "results": [_result_summary(item) for item in rerun["results"]],
                "flips": _flip_summary(rerun["flips"]),
                "session_snapshot": _session_ref(rerun_session, rerun_session_file["path"]),
            },
            "replay": replay,
            "historical_stale_snapshot_replay": {
                "read_only": True,
                "result": historical_replay,
            },
            "current_view_digest": view3["view_digest"],
        }
        log.write(
            "demo.scenario",
            scenario=3,
            run_id=rerun["run_id"],
            crashed_run_id=marker["run_id"],
            unknown_revisions=scenario3["incomplete_run"]["unknown_revisions"],
        )
        summary = {
            "format_version": 1,
            "kind": "tlstrust.demo_summary",
            "workspace": str(workspace.root),
            "scenarios": [scenario1, scenario2, scenario3],
            "final": {
                "current_trust_version": view3["trust_version"],
                "revision_chain_digest": view3["revision_chain_digest"],
                "current_view_digest": view3["view_digest"],
                "current_run_ids": [run["run_id"] for run in view3["runs"]],
                "current_session_snapshots": [
                    item["content_digest"] for item in view3["session_snapshots"]
                ],
                "log_path": str(workspace.log_path),
            },
        }
        write_canonical_json(workspace.root / "demo_summary.json", summary)
        return summary
    finally:
        try:
            ledger.close()
        except Exception:
            pass
