import pytest

from gsb.ledger import LEDGER_FORMAT_VERSION, SNAPSHOT_VERSION, Ledger
from gsb.merge import GrayConclusion
from gsb.receipts import NormalizedReceipt
from gsb.routing import HitRecord
from gsb.scheduler import WaitRecord


def _hit(request_id="q1", batch_id="batch-1", param_version=1):
    return HitRecord(
        request_id=request_id,
        batch_id=batch_id,
        rule_id="r1",
        rule_version=1,
        client_key="c1",
        bucket=1234,
        weight=5000,
        route="gray",
        backend_id="b1",
        param_version=param_version,
    )


def _receipt(batch_id="batch-1"):
    return NormalizedReceipt(
        batch_id=batch_id,
        route="gray",
        output_digest="a" * 64,
        param_digest="p" * 64,
        quant_profile="int8",
        elapsed_ms=1,
        sample_ids=("q1",),
        param_version=1,
        source_format="v3",
    )


def _conclusion(batch_id="batch-1"):
    return GrayConclusion(
        batch_id=batch_id,
        route="gray",
        sample_ids=("q1",),
        param_digest="p" * 64,
        param_version=1,
        source_formats=("v3",),
        adopted_seq=0,
        adopted_format="v3",
    )


def _commit(ledger, batch_id="batch-1", cursor=1):
    ledger.commit_batch(
        hits=[_hit(batch_id=batch_id)],
        receipts=[("b1", _receipt(batch_id))],
        conclusion=_conclusion(batch_id),
        waits=[WaitRecord(batch_id, "b1", 0, 100, 105)],
        active_versions={"b1": 1},
        version_digests={("b1", 1): "p" * 64},
        cursor_value=cursor,
    )


def test_commit_and_readback(tmp_path):
    ledger = Ledger.open(tmp_path / "l.db", content_digest="d" * 64)
    assert ledger.diffs == []
    _commit(ledger)
    assert ledger.committed_batches() == ["batch-1"]
    hits = ledger.hits_for("batch-1")
    assert hits[0]["bucket"] == 1234
    assert ledger.conclusion_for("batch-1")["route"] == "gray"
    assert ledger.cursor_value() == 1
    assert ledger.route_distribution() == {"gray": 1}
    meta = ledger.meta()
    assert meta["snapshot_version"] == str(SNAPSHOT_VERSION)
    assert meta["ledger_format_version"] == str(LEDGER_FORMAT_VERSION)
    ledger.close()


def test_snapshot_digest_mismatch_full_replace(tmp_path):
    path = tmp_path / "l.db"
    ledger = Ledger.open(path, content_digest="a" * 64)
    _commit(ledger)
    ledger.close()
    # 相同摘要重开：数据保留
    ledger = Ledger.open(path, content_digest="a" * 64)
    assert ledger.diffs == []
    assert ledger.committed_batches() == ["batch-1"]
    ledger.close()
    # 摘要变化：整体替换并稳定列出差异项
    ledger = Ledger.open(path, content_digest="b" * 64)
    assert ledger.diffs == [f"content_digest: {'a'*12}... -> {'b'*12}..."]
    assert ledger.committed_batches() == []
    assert ledger.cursor_value() == 0
    ledger.close()


def test_snapshot_version_mismatch_full_replace(tmp_path):
    import sqlite3

    path = tmp_path / "l.db"
    ledger = Ledger.open(path, content_digest="a" * 64)
    _commit(ledger)
    ledger.close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value='0' WHERE key='snapshot_version'")
    conn.commit()
    conn.close()
    ledger = Ledger.open(path, content_digest="a" * 64)
    assert ledger.diffs == [f"snapshot_version: 0 -> {SNAPSHOT_VERSION}"]
    assert ledger.committed_batches() == []
    ledger.close()
