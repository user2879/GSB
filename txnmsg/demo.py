"""Deterministic end-to-end demonstration and acceptance evidence."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json
from .diagnostics import build_diagnostics, render_text
from .errors import BusinessError
from .injections import (
    inject_corrupt_batch,
    inject_duplicate_position,
    inject_ledger_digest,
)
from .observability import StructuredLogger
from .replay_verify import replay_verify
from .sample_data import write_samples
from .snapshots import (
    create_snapshot,
    list_snapshots,
    next_snapshot_path,
    restore_snapshot,
)
from .store import Ledger
from .ui import tkinter_available
from .validation import load_partitions, load_transactions


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CRASH_EXIT_CODE = 74


def reset_workspace(workdir: str | os.PathLike[str]) -> None:
    root = Path(workdir)
    for name in (
        "ledger.sqlite",
        "ledger.sqlite-journal",
        "ledger.sqlite-wal",
        "ledger.sqlite-shm",
        "demo-summary.json",
    ):
        path = root / name
        if path.exists():
            path.unlink()
    for name in ("snapshots", "backups", "logs", "samples", "anomaly"):
        path = root / name
        if path.exists():
            shutil.rmtree(path, onerror=_remove_readonly)
    root.mkdir(parents=True, exist_ok=True)


def _remove_readonly(function: Any, path: str, exc_info: Any) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass
    function(path)


def _advance(ledger: Ledger, transaction_id: str, states: Iterable[str]) -> None:
    for state in states:
        ledger.advance_state(transaction_id, state)


def _forced_termination_worker(
    database_path: Path,
    transaction_id: str,
    state: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "txnmsg",
            "_crash-worker",
            "--database",
            str(database_path),
            "--transaction-id",
            transaction_id,
            "--state",
            state,
        ],
        cwd=str(REPOSITORY_ROOT),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _stale_cache(database_path: Path) -> dict[str, Any]:
    with Ledger(database_path, readonly=True, auto_recover=False) as ledger:
        return ledger.persisted_prefix_rows()


def _error_record(error: BusinessError) -> dict[str, Any]:
    payload = error.to_dict()
    payload["status"] = "ERROR"
    return payload


def run_demo(workdir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    root = Path(workdir or "demo_workspace").resolve()
    reset_workspace(root)
    sample_paths = write_samples(root)
    log_path = root / "logs" / "business.jsonl"
    logger = StructuredLogger(log_path)
    database_path = root / "ledger.sqlite"

    partitions = load_partitions(sample_paths["partitions"], logger)
    transactions = load_transactions(sample_paths["transactions"], partitions, logger)
    transaction_map = {transaction.transaction_id: transaction for transaction in transactions}
    result: dict[str, Any] = {
        "status": "PASS",
        "workdir": str(root),
        "samples": sample_paths,
        "input": {
            "partition_count": len(partitions),
            "transaction_count": len(transactions),
            "first_transaction_id": transactions[0].transaction_id,
            "last_transaction_id": transactions[-1].transaction_id,
        },
        "logs": str(log_path),
    }

    with Ledger(database_path, logger=logger) as ledger:
        ledger.initialise_partitions(partitions)
        ledger.ingest(transactions)
        _advance(ledger, "tx-order-001", ("prepared", "committing"))

    committed_crash = _forced_termination_worker(
        database_path, "tx-order-001", "committed"
    )
    if committed_crash.returncode != CRASH_EXIT_CODE:
        raise RuntimeError(
            "expected forced termination after committed event, got "
            f"{committed_crash.returncode}: {committed_crash.stderr}"
        )
    stale_after_commit = _stale_cache(database_path)
    with Ledger(database_path, logger=logger) as ledger:
        recovery_after_commit = ledger.last_recovery
        if recovery_after_commit is None:
            raise RuntimeError("recovery report missing after committed-event crash")
        result["committed_event_crash"] = {
            "exit_code": committed_crash.returncode,
            "persisted_prefix_cache_before_recovery": stale_after_commit,
            "recovery": recovery_after_commit.as_dict(),
            "prefix_after_recovery": [item.as_dict() for item in ledger.query_prefixes()],
        }
        _advance(ledger, "tx-order-002", ("prepared", "committing", "committed"))
        _advance(ledger, "tx-pay-001", ("prepared", "committing", "committed"))
        _advance(ledger, "tx-order-003", ("prepared",))

    unknown_crash = _forced_termination_worker(
        database_path, "tx-order-003", "committing"
    )
    if unknown_crash.returncode != CRASH_EXIT_CODE:
        raise RuntimeError(
            "expected forced termination during committing, got "
            f"{unknown_crash.returncode}: {unknown_crash.stderr}"
        )
    stale_before_unknown = _stale_cache(database_path)
    with Ledger(database_path, logger=logger) as ledger:
        recovery_unknown = ledger.last_recovery
        if recovery_unknown is None:
            raise RuntimeError("recovery report missing after committing crash")
        result["committing_crash"] = {
            "exit_code": unknown_crash.returncode,
            "persisted_prefix_cache_before_recovery": stale_before_unknown,
            "recovery": recovery_unknown.as_dict(),
            "prefix": [item.as_dict() for item in ledger.query_prefixes()],
            "exit_report_available": False,
        }
        unknown_snapshot = create_snapshot(
            ledger, next_snapshot_path(root)
        )
        unknown_replay = replay_verify(ledger, unknown_snapshot.path)
        result["unknown_replay"] = unknown_replay.as_dict()

        _advance(ledger, "tx-order-003", ("committed",))
        _advance(ledger, "tx-order-004", ("prepared", "aborted"))
        _advance(ledger, "tx-pay-002", ("prepared", "committing", "committed"))
        final_snapshot = create_snapshot(ledger, next_snapshot_path(root))
        final_conclusion = ledger.conclusion()
        result["final_snapshot"] = {
            "path": final_snapshot.path,
            "ledger_digest": final_snapshot.ledger_digest,
            "content_digest": final_snapshot.content_digest,
            "event_cursor": final_snapshot.event_cursor,
            "partition_set_version": final_snapshot.partition_set_version,
            "transaction_set_version": final_snapshot.transaction_set_version,
            "partitions": [item.as_dict() for item in final_conclusion.partitions],
        }

        increment = load_transactions(sample_paths["increment"], partitions, logger)
        ledger.ingest(increment)
        _advance(ledger, "tx-pay-003", ("prepared", "committing", "committed"))
        diverged_conclusion = ledger.conclusion()
        result["diverged_after_snapshot"] = {
            "ledger_digest": diverged_conclusion.ledger_digest,
            "event_cursor": diverged_conclusion.event_cursor,
            "transaction_set_version": diverged_conclusion.transaction_set_version,
            "partitions": [item.as_dict() for item in diverged_conclusion.partitions],
        }

        restore_result = restore_snapshot(ledger, final_snapshot.path, logger=logger)
        backup = Path(restore_result.backup_path) if restore_result.backup_path else None
        result["snapshot_restore"] = {
            **restore_result.as_dict(),
            "backup_readonly": bool(
                backup is not None and (backup.stat().st_mode & 0o222) == 0
            ),
        }
        replay_result = replay_verify(ledger, final_snapshot.path)
        result["replay_after_restore"] = replay_result.as_dict()
        view_before = build_diagnostics(ledger, root)
        digest_before = ledger.computed_ledger_digest()
        view_text_one = render_text(build_diagnostics(ledger, root))
        view_text_two = render_text(build_diagnostics(ledger, root))
        digest_after = ledger.computed_ledger_digest()
        if view_text_one != view_text_two or digest_before != digest_after:
            raise RuntimeError("diagnostics rendering changed ledger content or ordering")
        result["diagnostics"] = {
            "stable_render": True,
            "ledger_unchanged": True,
            "view": view_before,
            "text": view_text_one,
        }
        result["snapshots"] = [item.as_dict() for item in list_snapshots(root)]

    anomalies: dict[str, Any] = {}
    corrupt_path = inject_corrupt_batch(root)
    try:
        load_transactions(corrupt_path, partitions, logger)
        raise RuntimeError("corrupt batch was accepted")
    except BusinessError as error:
        anomalies["corrupt_batch"] = {
            "input": corrupt_path,
            **_error_record(error),
        }

    duplicate_path = inject_duplicate_position(root)
    try:
        load_transactions(duplicate_path, partitions, logger)
        raise RuntimeError("duplicate positions were accepted")
    except BusinessError as error:
        anomalies["duplicate_position"] = {
            "input": duplicate_path,
            **_error_record(error),
        }

    tampered_ledger_path = inject_ledger_digest(database_path, root)
    try:
        Ledger(tampered_ledger_path, auto_recover=False, logger=logger)
        raise RuntimeError("tampered ledger digest was accepted")
    except BusinessError as error:
        anomalies["ledger_digest"] = {
            "input": tampered_ledger_path,
            **_error_record(error),
        }
    result["anomalies"] = anomalies

    expected_codes = {
        "corrupt_batch": "E_SPEC",
        "duplicate_position": "E_ORDER",
        "ledger_digest": "E_SNAPSHOT",
    }
    for key, code in expected_codes.items():
        if anomalies[key]["code"] != code:
            raise RuntimeError(f"unexpected anomaly code for {key}: {anomalies[key]}")

    log_records: list[dict[str, Any]] = []
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    log_records.append(json.loads(line))
    result["structured_log"] = {
        "path": str(log_path),
        "codes": sorted({record["code"] for record in log_records}),
        "records": log_records,
    }
    result["ui"] = {
        "tkinter_available": tkinter_available(),
        "fallback_text_rendered": True,
        "note": "Tk runtime is optional for headless verification; the UI module is import-safe.",
    }
    summary_path = root / "demo-summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(result))
        handle.write("\n")
    result["summary_path"] = str(summary_path)
    return result
