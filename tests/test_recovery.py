from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from ossdiag.models import load_spec_data
from ossdiag.engine import Workbench, RunOptions

from conftest import KiB, object_doc, spec_doc, workbench, write_spec


def _clean_digests(tmp_path: Path, doc: dict) -> tuple[dict[int, str], str, dict[int, str]]:
    wb = workbench(tmp_path / "clean", doc=doc)
    summary = wb.run()
    assert summary.ok
    rows = wb.ledger.shard_rows("object/a.bin")
    shard_digests = {int(row["shard_index"]): row["content_digest"] for row in rows}
    object_digest = wb.ledger.object_row("object/a.bin")["object_digest"]
    file_hashes = {
        int(row["shard_index"]): hashlib.sha256(
            wb.ledger.shard_path("object/a.bin", int(row["shard_index"])).read_bytes()
        ).hexdigest()
        for row in rows
    }
    wb.close()
    return shard_digests, object_digest, file_hashes


def _crash_child(spec_path: Path, root: Path, crash_after: int) -> int:
    code = f"""
from pathlib import Path
from ossdiag.models import load_spec
from ossdiag.engine import Workbench, RunOptions
session = load_spec({str(spec_path)!r})
wb = Workbench(session, ledger_path={str(root/'ledger.sqlite')!r}, data_dir={str(root/'data')!r}, log_path={str(root/'events.jsonl')!r}, backup_dir={str(root/'ledger_backups')!r})
wb.run(RunOptions(crash_after_batches={crash_after}))
"""
    completed = subprocess.run([sys.executable, "-c", code])
    return completed.returncode


def test_force_kill_recovery_unknown_and_digest_equivalence(tmp_path) -> None:
    doc = spec_doc(
        objects=[object_doc("object/a.bin", total_bytes=256 * KiB, shard_size=128 * KiB, batch_size=64 * KiB, concurrency=1)],
        global_lease_quota=1,
        worker_threads=1,
        throughput_budget_bytes=512 * KiB,
    )
    clean_shards, clean_object, clean_files = _clean_digests(tmp_path, doc)
    spec_path = write_spec(tmp_path / "spec.json", doc)
    crash_root = tmp_path / "crash"
    crash_root.mkdir()
    assert _crash_child(spec_path, crash_root, 2) == 137

    session = load_spec_data(doc)
    wb = Workbench(
        session,
        ledger_path=crash_root / "ledger.sqlite",
        data_dir=crash_root / "data",
        log_path=crash_root / "resume-events.jsonl",
        backup_dir=crash_root / "ledger_backups",
    )
    reconcile = wb.initialize()
    assert reconcile.replaced is False
    assert reconcile.recovered_unknown
    assert reconcile.reclaimed_leases
    assert reconcile.recovered_unknown[0]["shard_index"] == 0
    assert reconcile.recovered_unknown[0]["requires_rerun"] is True
    rows = wb.ledger.shard_rows("object/a.bin")
    assert rows[0]["status"] == "unknown"
    assert rows[0]["unknown"] == 1
    assert rows[0]["progress_cursor"] == 64 * KiB
    assert wb.ledger.shard_path("object/a.bin", 0).stat().st_size == 64 * KiB
    assert len(wb.ledger.batch_rows("object/a.bin", 0)) == 1
    assert wb.ledger.object_row("object/a.bin")["selected_path"]
    assert wb.scale_exponent == 6

    resumed = wb.run()
    assert resumed.ok is True
    resumed_rows = wb.ledger.shard_rows("object/a.bin")
    resumed_shards = {int(row["shard_index"]): row["content_digest"] for row in resumed_rows}
    resumed_object = wb.ledger.object_row("object/a.bin")["object_digest"]
    resumed_files = {
        int(row["shard_index"]): hashlib.sha256(
            wb.ledger.shard_path("object/a.bin", int(row["shard_index"])).read_bytes()
        ).hexdigest()
        for row in resumed_rows
    }
    assert resumed_shards == clean_shards
    assert resumed_object == clean_object
    assert resumed_files == clean_files
    assert all(row["unknown"] == 0 for row in resumed_rows)
    assert "强制终止时的退出报告不可用" in resumed.note
    wb.close()


def test_fully_committed_shards_survive_restart_without_rerun(tmp_path) -> None:
    doc = spec_doc(objects=[object_doc(total_bytes=128 * KiB)])
    wb = workbench(tmp_path, doc=doc)
    first = wb.run()
    digest = wb.ledger.meta()["ledger_digest"]
    wb.close()

    reopened = workbench(tmp_path, session=load_spec_data(doc))
    reconcile = reopened.initialize()
    assert reconcile.replaced is False
    assert reconcile.recovered_unknown == ()
    second = reopened.run()
    assert second.ok is True
    assert second.throughput["used_bytes"] <= second.throughput["limit_bytes"]
    assert reopened.ledger.meta()["ledger_digest"] == digest
    assert first.objects[0]["object_digest"] == second.objects[0]["object_digest"]
    reopened.close()
