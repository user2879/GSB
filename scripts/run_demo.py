"""Run the complete deterministic acceptance demonstration.

Outputs are written under artifacts/.  The script uses only local files and
SQLite; it never accesses the network.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leasebench.engine import LeaseEngine
from leasebench.logging_utils import StructuredLogger
from leasebench.models import canonical_json, content_digest
from leasebench.service import (
    Workbench,
    group_events,
    load_view_from_ledger,
    replay_ledger,
)
from leasebench.ui import render_view_text, write_text
from leasebench.validation import (
    load_declaration,
    load_events,
    load_session,
    parse_declaration,
)

sys.path.insert(0, str(ROOT / "scripts"))
import generate_demo


ARTIFACTS = ROOT / "artifacts"
SAMPLES = ROOT / "samples"


def clean(pattern: str) -> None:
    for path in ARTIFACTS.glob(pattern):
        if path.is_file():
            path.unlink()


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "leasebench", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def cli_result(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "exit_code": process.returncode,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }


def collect_operations(db: Path) -> list[dict[str, Any]]:
    from leasebench.ledger import Ledger

    with Ledger(db) as ledger:
        batches = ledger.load_batches()
    return [
        operation
        for batch in batches
        for operation in batch["payload"]["operations"]
    ]


def function_order_determinism(events, session) -> dict[str, Any]:
    raw = json.loads((SAMPLES / "declarations_v1.json").read_text(encoding="utf-8"))
    declaration = load_declaration(SAMPLES / "declarations_v1.json")
    reversed_raw = json.loads(json.dumps(raw))
    reversed_raw["functions"] = list(reversed(reversed_raw["functions"]))
    reversed_declaration = parse_declaration(reversed_raw)

    def payloads(item_declaration):
        engine = LeaseEngine(item_declaration, session)
        values = []
        for group in group_events(events):
            engine.begin_batch(engine.make_batch_id(group))
            values.append(engine.run_batch(group))
        return values

    first = payloads(declaration)
    second = payloads(reversed_declaration)
    first_json = [canonical_json(item) for item in first]
    second_json = [canonical_json(item) for item in second]
    return {
        "identical": first_json == second_json,
        "payload_sha256": hashlib.sha256("".join(first_json).encode("utf-8")).hexdigest(),
        "declaration_digest_equal": declaration.digest == reversed_declaration.digest,
    }


def rewrite_first_batch_grant(db: Path) -> None:
    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    try:
        rows = list(connection.execute("SELECT * FROM batches ORDER BY sequence_no"))
        previous = ""
        mutated = False
        for row in rows:
            payload = json.loads(row["result_json"])
            if not mutated:
                for operation in payload["operations"]:
                    if operation["type"] == "GRANT":
                        operation["lease"]["concurrency"] += 1
                        mutated = True
                        break
            content = content_digest(payload)
            chain = hashlib.sha256((previous + content).encode("utf-8")).hexdigest()
            connection.execute(
                "UPDATE batches SET result_json=?, content_digest=?, prev_chain_digest=?, chain_digest=? WHERE batch_id=?",
                (
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    content,
                    previous,
                    chain,
                    row["batch_id"],
                ),
            )
            previous = chain
        if not mutated:
            raise RuntimeError("semantic replay injection found no grant")
        connection.execute("UPDATE meta SET value=? WHERE key='chain_digest'", (previous,))
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    generate_demo.write()
    for pattern in (
        "demo.sqlite*",
        "crash_demo.sqlite*",
        "cycle_demo.sqlite*",
        "quota_demo.sqlite*",
        "snapshot_demo.sqlite*",
        "replay_demo.sqlite*",
        "*.log",
        "demo_*.json",
        "demo_*.txt",
        "view_*.txt",
    ):
        clean(pattern)

    start_mono = time.monotonic_ns()
    end_mono = time.monotonic_ns()
    declaration_v1 = load_declaration(SAMPLES / "declarations_v1.json")
    declaration_v2 = load_declaration(SAMPLES / "declarations_v2.json")
    events = load_events(SAMPLES / "events.ndjson", declaration_v1)
    session = load_session(1000, 50)
    logger = StructuredLogger(ARTIFACTS / "demo.log")
    db = ARTIFACTS / "demo.sqlite"

    determinism = function_order_determinism(events, session)

    with Workbench(
        db_path=db,
        declaration=declaration_v1,
        events=events,
        session=session,
        logger=logger,
    ) as workbench:
        first = workbench.run(limit_event_seq=6)
        write_text(ARTIFACTS / "view_v1_prefix.txt", render_view_text(first.view))
    first_ops = collect_operations(db)
    first_loans = [op["loan"] for op in first_ops if op["type"] == "BORROW"]
    first_preemptions = [
        op["preemption"] for op in first_ops if op["type"] == "PREEMPT_SCHEDULE"
    ]

    with Workbench(
        db_path=db,
        declaration=declaration_v2,
        events=events,
        session=session,
        logger=logger,
    ) as workbench:
        second = workbench.run()
        final_view = workbench.current_view()
    final_text = render_view_text(final_view)
    write_text(ARTIFACTS / "view_v2_final.txt", final_text)
    (ARTIFACTS / "demo_final_view.json").write_text(
        json.dumps(final_view, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    replay = replay_ledger(db_path=db, logger=logger)
    headless_one = render_view_text(load_view_from_ledger(db))
    headless_two = render_view_text(load_view_from_ledger(db))
    render_digest = hashlib.sha256(headless_one.encode("utf-8")).hexdigest()

    all_ops = collect_operations(db)
    release_dispositions = Counter(
        op["release"]["disposition"] for op in all_ops if op["type"] == "RELEASE"
    )
    loan_dispositions = Counter(
        op["loan"]["disposition"] for op in all_ops if op["type"] == "LOAN_RETURN"
    )
    preemption_statuses = Counter(
        item["status"] for item in final_view["preemptions"]
    )

    try:
        import tkinter

        tk_available = True
        tk_error = None
    except Exception as exc:
        tk_available = False
        tk_error = f"{type(exc).__name__}: {exc}"

    crash_db = ARTIFACTS / "crash_demo.sqlite"
    crash_log = ARTIFACTS / "crash_demo.log"
    crash = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(crash_db),
        "--inject-crash-after-pending",
        "--log",
        str(crash_log),
    )
    recovered_path = ARTIFACTS / "crash_demo_recovered.json"
    recover = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(crash_db),
        "--json-out",
        str(recovered_path),
        "--log",
        str(crash_log),
    )
    recovered = json.loads(recovered_path.read_text(encoding="utf-8"))

    cycle_log = ARTIFACTS / "cycle_demo.log"
    cycle = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(ARTIFACTS / "cycle_demo.sqlite"),
        "--inject-cycle",
        "--log",
        str(cycle_log),
    )
    quota_log = ARTIFACTS / "quota_demo.log"
    quota = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(ARTIFACTS / "quota_demo.sqlite"),
        "--inject-quota",
        "--log",
        str(quota_log),
    )
    snapshot_db = ARTIFACTS / "snapshot_demo.sqlite"
    snapshot_log = ARTIFACTS / "snapshot_demo.log"
    snapshot_seed = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(snapshot_db),
        "--limit-events",
        "6",
        "--log",
        str(snapshot_log),
    )
    snapshot = cli(
        "replay",
        "--db",
        str(snapshot_db),
        "--inject-snapshot",
        "--log",
        str(snapshot_log),
    )

    replay_db = ARTIFACTS / "replay_demo.sqlite"
    replay_log = ARTIFACTS / "replay_demo.log"
    replay_seed = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(replay_db),
        "--limit-events",
        "6",
        "--log",
        str(replay_log),
    )
    rewrite_first_batch_grant(replay_db)
    replay_error = cli("replay", "--db", str(replay_db), "--log", str(replay_log))

    report = {
        "status": "PASS",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": __import__("numpy").__version__,
            "sqlite": sqlite3.sqlite_version,
            "tkinter_available": tk_available,
            "tkinter_error": tk_error,
            "monotonic_ns_increases": end_mono > start_mono,
        },
        "samples": {
            "declaration_v1_digest": declaration_v1.digest,
            "declaration_v2_digest": declaration_v2.digest,
            "event_count": len(events),
            "watermark_scale": session.watermark_scale,
            "rounding": "FLOOR_TOWARD_ZERO",
        },
        "determinism": determinism,
        "v1_prefix": {
            "cursor": first.final_cursor,
            "batches": first.batches_written,
            "declaration_version": first.declaration_version,
            "loans": first_loans,
            "preemptions": first_preemptions,
            "view_preemptions": first.view["preemptions"],
            "view_loans": first.view["loans"],
            "view_queue": first.view["queue"],
        },
        "v2_resume": {
            "cursor": second.final_cursor,
            "batches": second.batches_written,
            "declaration_version": second.declaration_version,
            "recovery_status": second.recovery_status,
            "invalidations": final_view["ledger_invalidations"],
            "active_lease_ids": [item["lease_id"] for item in final_view["leases"]],
            "queue_ids": [item["queue_id"] for item in final_view["queue"]],
            "watermarks": final_view["watermarks"],
        },
        "attribution": {
            "release_dispositions": dict(sorted(release_dispositions.items())),
            "loan_return_dispositions": dict(sorted(loan_dispositions.items())),
            "preemption_statuses": dict(sorted(preemption_statuses.items())),
            "v1_prefix_preemption_statuses": dict(
                sorted(
                    Counter(item["status"] for item in first.view["preemptions"]).items()
                )
            ),
            "release_ids_unique": len(
                [op for op in all_ops if op["type"] == "RELEASE"]
            )
            == len(
                {
                    op["release"]["lease_id"]
                    for op in all_ops
                    if op["type"] == "RELEASE"
                }
            ),
            "loan_return_ids_unique": len(
                [op for op in all_ops if op["type"] == "LOAN_RETURN"]
            )
            == len(
                {
                    op["loan"]["loan_id"]
                    for op in all_ops
                    if op["type"] == "LOAN_RETURN"
                }
            ),
        },
        "replay": replay.as_dict(),
        "ui": {
            "mode": "headless deterministic renderer",
            "stable": headless_one == headless_two,
            "sha256": render_digest,
            "tkinter_available": tk_available,
            "note": (
                "Tk UI is available for interactive use."
                if tk_available
                else "This Linux validation container lacks Tk shared libraries; the Windows Tk UI code is present, and its Tk-independent rendering is validated headlessly."
            ),
        },
        "forced_termination": {
            "crash_exit_code": crash.returncode,
            "crash_stdout": crash.stdout.strip(),
            "recovery_exit_code": recover.returncode,
            "recovery_status": recovered["recovery_status"],
            "recomputed_unknown_lease_ids": recovered["recomputed_unknown_lease_ids"],
            "final_cursor": recovered["final_cursor"],
            "note": "A forced termination intentionally has no exit report; only committed batches are authoritative.",
        },
        "injections": {
            "cycle": cli_result(cycle),
            "quota": cli_result(quota),
            "snapshot": cli_result(snapshot),
            "replay_mismatch": cli_result(replay_error),
            "seed_commands_ok": all(
                item.returncode == 0 for item in (snapshot_seed, replay_seed)
            ),
        },
    }

    expected_codes = {
        "cycle": '"code": "E_DECL"' in cycle.stdout and cycle.returncode == 2,
        "quota": '"code": "E_QUOTA"' in quota.stdout and quota.returncode == 3,
        "snapshot": '"code": "E_SNAPSHOT"' in snapshot.stdout
        and snapshot.returncode == 4,
        "replay": '"code": "E_REPLAY"' in replay_error.stdout
        and replay_error.returncode == 5,
        "replay_ok": replay.ok,
        "determinism": determinism["identical"],
        "ui_stable": headless_one == headless_two,
        "crash": crash.returncode == 17 and recovered["recovery_status"] == "CLEAN",
        "version_evolution": bool(final_view["ledger_invalidations"]),
        "preemption_demo": bool(first_preemptions),
        "borrowing_demo": bool(first_loans),
    }
    report["checks"] = expected_codes
    if not all(expected_codes.values()):
        report["status"] = "FAIL"

    (ARTIFACTS / "demo_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        f"status: {report['status']}",
        f"checks: {json.dumps(expected_codes, ensure_ascii=False, sort_keys=True)}",
        f"events: {len(events)}; replay batches: {replay.batches_checked}",
        f"v1 prefix cursor: {first.final_cursor}; v2 final cursor: {second.final_cursor}",
        f"invalidated lease/queue/preemption objects: {len(final_view['ledger_invalidations'])}",
        f"current preemption statuses: {dict(sorted(preemption_statuses.items()))}",
        f"v1 prefix preemption statuses: {report['attribution']['v1_prefix_preemption_statuses']}",
        f"release dispositions: {dict(sorted(release_dispositions.items()))}",
        f"loan returns: {dict(sorted(loan_dispositions.items()))}",
        f"headless render sha256: {render_digest}",
        f"Tk available in this validation environment: {tk_available}",
        f"forced termination exit: {crash.returncode}; recovery: {recovered['recovery_status']}",
    ]
    write_text(ARTIFACTS / "demo_summary.txt", "\n".join(summary))
    print("\n".join(summary))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
