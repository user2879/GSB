from __future__ import annotations

import json
import time

from ossdiag.engine import RunOptions, Workbench
from ossdiag.gui import dashboard_tables
from ossdiag.scheduler import ShardScheduler

from conftest import KiB, MiB, object_doc, session_from, spec_doc, workbench


def test_scheduler_completes_objects_and_respects_constraints(tmp_path) -> None:
    doc = spec_doc(
        objects=[
            object_doc("a", total_bytes=256 * KiB, concurrency=2, seed_base=1),
            object_doc("b", total_bytes=128 * KiB, concurrency=1, seed_base=50),
        ],
        global_lease_quota=2,
        throughput_budget_bytes=384 * KiB,
        worker_threads=4,
    )
    wb = workbench(tmp_path, doc=doc)
    summary = wb.run()
    assert summary.ok is True
    assert summary.throughput["used_bytes"] <= summary.throughput["limit_bytes"]
    assert summary.leases["peak_global"] <= 2
    assert summary.leases["peak_per_object"]["a"] <= 2
    assert summary.leases["peak_per_object"]["b"] <= 1
    assert all(item["digest_status"] == "PASS" for item in summary.objects)
    assert all(item["consistent"] for item in summary.path_verification)
    for item in summary.objects:
        for shard in item["shards"]:
            assert shard["status"] == "complete"
            assert shard["content_digest"]
            assert shard["lease_version"] > 0
            assert shard["completed_mono_ms"] is not None
    for task in summary.tasks:
        assert task["wait_start_monotonic_ms"] <= task["wait_end_monotonic_ms"]
        assert task["wait_start_monotonic_ms"] <= task["budget_wait_start_monotonic_ms"]
        assert task["budget_wait_start_monotonic_ms"] <= task["budget_wait_end_monotonic_ms"]
    wb.close()


def test_ledger_persists_every_committed_batch_field(tmp_path) -> None:
    wb = workbench(
        tmp_path,
        doc=spec_doc(objects=[object_doc(total_bytes=128 * KiB, shard_size=128 * KiB, batch_size=64 * KiB)]),
    )
    wb.run()
    rows = wb.ledger.batch_rows("object/a.bin", 0)
    assert len(rows) == 2
    for row in rows:
        assert row["byte_end"] - row["byte_start"] == 64 * KiB
        assert len(row["content_digest"]) == 64
        assert row["lease_version"] > 0
        assert row["renew_count"] >= 0
        assert row["completed_mono_ms"] > 0
        assert row["progress_cursor"] == row["byte_end"]
    wb.close()


def test_budget_shortage_fails_without_overflow(tmp_path) -> None:
    wb = workbench(
        tmp_path,
        doc=spec_doc(objects=[object_doc(total_bytes=128 * KiB)], throughput_budget_bytes=64 * KiB),
    )
    summary = wb.run()
    assert summary.ok is False
    assert any(item["code"] == "E_BUDGET" for item in summary.errors)
    assert summary.throughput["used_bytes"] <= summary.throughput["limit_bytes"]
    wb.close()


def test_dashboard_refresh_is_read_only_and_stably_sorted(tmp_path) -> None:
    wb = workbench(
        tmp_path,
        doc=spec_doc(
            objects=[object_doc("z", total_bytes=64 * KiB), object_doc("a", total_bytes=64 * KiB)],
            global_lease_quota=2,
        ),
    )
    wb.run()
    digest_before = wb.ledger.meta()["ledger_digest"]
    shards_before = wb.ledger.all_shard_rows()
    first = wb.build_dashboard()
    second = wb.build_dashboard()
    assert wb.ledger.meta()["ledger_digest"] == digest_before
    assert wb.ledger.all_shard_rows() == shards_before
    assert dashboard_tables(first) == dashboard_tables(second)
    object_rows = dashboard_tables(first)["objects"]
    assert [row[0] for row in object_rows] == ["a", "z"]
    shard_rows = dashboard_tables(first)["shards"]
    assert [(row[0], row[1]) for row in shard_rows] == sorted((row[0], row[1]) for row in shard_rows)
    wb.close()


def test_digest_injection_reports_e_digest_and_leaves_unknown(tmp_path) -> None:
    wb = workbench(tmp_path, doc=spec_doc(objects=[object_doc(total_bytes=128 * KiB)]))
    summary = wb.run(RunOptions(inject_digest_mismatch=True))
    assert summary.ok is False
    assert any(item["code"] == "E_DIGEST" for item in summary.errors)
    shards = wb.ledger.shard_rows("object/a.bin")
    assert any(row["status"] == "unknown" and row["unknown"] == 1 for row in shards)
    assert wb.ledger.meta()["ledger_digest"] == wb.ledger.ledger_digest()
    wb.close()


def test_digest_injection_still_reports_on_completed_session(tmp_path) -> None:
    wb = workbench(tmp_path, doc=spec_doc(objects=[object_doc(total_bytes=128 * KiB)]))
    clean = wb.run()
    assert clean.ok
    digest_before = wb.ledger.meta()["ledger_digest"]
    summary = wb.run(RunOptions(inject_digest_mismatch=True))
    assert summary.ok is False
    assert any(item["code"] == "E_DIGEST" for item in summary.errors)
    assert wb.ledger.meta()["ledger_digest"] == digest_before
    wb.close()


def test_lease_expiring_during_final_verification_is_reacquired_and_shard_rerun(tmp_path, monkeypatch) -> None:
    doc = spec_doc(
        objects=[object_doc(total_bytes=64 * KiB, lease_ms=100, renew_ms=10)],
        throughput_budget_bytes=MiB,
    )
    original = ShardScheduler._verify_full_shard
    calls = {"count": 0}

    def delayed_verify(self, **kwargs):
        if calls["count"] == 0:
            calls["count"] += 1
            time.sleep(0.25)
        return original(self, **kwargs)

    monkeypatch.setattr(ShardScheduler, "_verify_full_shard", delayed_verify)
    wb = workbench(tmp_path, doc=doc)
    summary = wb.run()
    assert summary.ok is True
    assert calls["count"] == 1
    row = wb.ledger.shard_rows("object/a.bin")[0]
    assert row["status"] == "complete"
    assert row["unknown"] == 0
    assert wb.ledger.meta()["ledger_digest"] == wb.ledger.ledger_digest()
    wb.close()
