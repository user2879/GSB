from __future__ import annotations

import os
import sqlite3

from ossdiag.engine import RunOptions
from ossdiag.models import load_spec_data

from conftest import KiB, object_doc, spec_doc, workbench


def test_injected_snapshot_conflict_creates_read_only_backup_and_replaces_all(tmp_path) -> None:
    doc = spec_doc(objects=[object_doc(total_bytes=128 * KiB)])
    wb = workbench(tmp_path, doc=doc)
    first = wb.run()
    assert first.ok
    replaced_count = len(wb.ledger.all_shard_rows())
    wb.close()

    reopened = workbench(tmp_path, session=load_spec_data(doc))
    summary = reopened.run(RunOptions(inject_snapshot_conflict=True))
    assert summary.ok is False
    assert any(item["code"] == "E_SNAPSHOT" for item in summary.errors)
    assert summary.reconcile["replaced"] is True
    assert len(summary.reconcile["replaced_shards"]) == replaced_count
    replaced = [(item["object_key"], item["shard_index"]) for item in summary.reconcile["replaced_shards"]]
    assert replaced == sorted(replaced)
    snapshots = reopened.ledger.list_snapshots()
    assert len(snapshots) == 1
    backup = snapshots[0]["backup_path"]
    assert os.path.exists(backup)
    assert os.stat(backup).st_mode & 0o222 == 0
    assert reopened.ledger.replacement_shards(1)
    assert summary.objects[0]["status"] == "complete"
    reopened.close()


def test_snapshot_injection_works_after_workbench_was_already_initialized(tmp_path) -> None:
    doc = spec_doc(objects=[object_doc(total_bytes=128 * KiB)])
    wb = workbench(tmp_path, doc=doc)
    wb.initialize()
    summary = wb.run(RunOptions(inject_snapshot_conflict=True))
    assert summary.reconcile["replaced"] is True
    assert any(item["code"] == "E_SNAPSHOT" for item in summary.errors)
    assert wb.ledger.list_snapshots()
    wb.close()


def test_format_version_corruption_triggers_whole_replacement(tmp_path) -> None:
    doc = spec_doc(objects=[object_doc(total_bytes=128 * KiB)])
    wb = workbench(tmp_path, doc=doc)
    wb.run()
    wb.close()
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute("UPDATE meta SET value='999' WHERE key='format_version'")
        conn.commit()
    reopened = workbench(tmp_path, session=load_spec_data(doc))
    result = reopened.initialize()
    assert result.replaced is True
    assert result.code == "E_SNAPSHOT"
    assert reopened.ledger.list_snapshots()
    reopened.close()


def test_blank_ledger_digest_triggers_whole_replacement(tmp_path) -> None:
    doc = spec_doc(objects=[object_doc(total_bytes=128 * KiB)])
    wb = workbench(tmp_path, doc=doc)
    wb.run()
    wb.close()
    with sqlite3.connect(tmp_path / "ledger.sqlite") as conn:
        conn.execute("UPDATE meta SET value='' WHERE key='ledger_digest'")
        conn.commit()
    reopened = workbench(tmp_path, session=load_spec_data(doc))
    result = reopened.initialize()
    assert result.replaced is True
    assert result.code == "E_SNAPSHOT"
    reopened.close()


def test_disk_corruption_triggers_whole_replacement(tmp_path) -> None:
    doc = spec_doc(objects=[object_doc(total_bytes=128 * KiB)])
    wb = workbench(tmp_path, doc=doc)
    wb.run()
    path = wb.ledger.shard_path("object/a.bin", 0)
    wb.close()
    with path.open("r+b") as handle:
        handle.seek(17)
        original = handle.read(1)
        handle.seek(17)
        handle.write(bytes([original[0] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())
    reopened = workbench(tmp_path, session=load_spec_data(doc))
    result = reopened.initialize()
    assert result.replaced is True
    assert result.code == "E_SNAPSHOT"
    assert any(issue["kind"].startswith("disk_") for issue in result.issues)
    reopened.close()
