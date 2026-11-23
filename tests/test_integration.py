"""端到端：真实子进程后端 + 热更新隔离 + 多格式归并 + 快照恢复重放。"""

import sqlite3

import pytest

from gsb.errors import GSBError
from gsb.ledger import Ledger
from gsb.pipeline import (
    SimulatedCrash,
    recover_session,
    resume_session,
    run_session,
)
from gsb.spec import input_content_digest

SAMPLE_DIR = "samples"
HOT_UPDATES = {2: [("b-gray-a", 777)]}


def test_full_session_with_hot_update(tmp_path):
    db = tmp_path / "ledger.db"
    report = run_session(SAMPLE_DIR, db, hot_updates=HOT_UPDATES)
    assert len(report.completed) == 4

    ledger = Ledger.open(db, input_content_digest(SAMPLE_DIR))
    try:
        # 多格式归并：batch-1 覆盖 v2/v3 来源；batch-3 覆盖全部三种格式
        conclusion = ledger.conclusion_for("batch-1")
        assert set(conclusion["source_formats"]) == {"v2", "v3"}
        conclusion3 = ledger.conclusion_for("batch-3")
        assert set(conclusion3["source_formats"]) == {"v1", "v2", "v3"}
        # 热更新隔离：batch-1/2 全部用 v1 参数；batch-3/4 灰度后端升至 v2，
        # 且同一批次同一后端不得混装两个参数版本
        iso: dict[tuple[str, str], set] = {}
        for hit_batch in ledger.committed_batches():
            for hit in ledger.hits_for(hit_batch):
                iso.setdefault((hit["batch_id"], hit["backend_id"]), set()).add(
                    hit["param_version"]
                )
        for versions in iso.values():
            assert len(versions) == 1, "同一批次同一后端不得混装两个参数版本"
        assert iso[("batch-1", "b-gray-a")] == {1}
        assert iso[("batch-3", "b-gray-a")] == {2}
        assert iso[("batch-4", "b-gray-a")] == {2}
        # 账本逐请求记录规则版本、分桶、权重与去向
        hits = ledger.hits_for("batch-1")
        assert {h["rule_version"] for h in hits} == {1, 2}
        assert all(0 <= h["bucket"] < 10000 for h in hits)
        # 等待观测存在且非负
        assert all(
            w["wait_end_ms"] >= w["wait_start_ms"] for w in ledger.all_waits()
        )
    finally:
        ledger.close()

    # 恢复重放：全部批次已提交，无未知批次
    recovery = recover_session(SAMPLE_DIR, db)
    assert recovery.replay_verified == 4
    assert recovery.unknown_batches == []
    assert recovery.exit_report_available


def test_routing_stable_across_reruns(tmp_path):
    db1, db2 = tmp_path / "a.db", tmp_path / "b.db"
    run_session(SAMPLE_DIR, db1)
    run_session(SAMPLE_DIR, db2)
    ledger1 = Ledger.open(db1, input_content_digest(SAMPLE_DIR))
    ledger2 = Ledger.open(db2, input_content_digest(SAMPLE_DIR))
    try:
        for batch_id in ledger1.committed_batches():
            hits1 = [
                (h["request_id"], h["bucket"], h["route"])
                for h in ledger1.hits_for(batch_id)
            ]
            hits2 = [
                (h["request_id"], h["bucket"], h["route"])
                for h in ledger2.hits_for(batch_id)
            ]
            assert hits1 == hits2
    finally:
        ledger1.close()
        ledger2.close()


def test_crash_recovery_and_resume(tmp_path):
    db = tmp_path / "ledger.db"
    with pytest.raises(SimulatedCrash):
        run_session(
            SAMPLE_DIR, db, hot_updates=HOT_UPDATES, crash_before_commit={2}
        )
    # 恢复：只有 batch-1、batch-2 完整提交；batch-3/4 未知需重跑
    recovery = recover_session(SAMPLE_DIR, db)
    assert recovery.replay_verified == 2
    assert recovery.unknown_batches == ["batch-3", "batch-4"]
    assert not recovery.exit_report_available
    # 重跑未知批次后重放全部通过
    resumed = resume_session(SAMPLE_DIR, db, hot_updates=HOT_UPDATES)
    assert [r.batch_id for r in resumed.completed] == ["batch-3", "batch-4"]
    assert resumed.replay_verified == 4
    recovery2 = recover_session(SAMPLE_DIR, db)
    assert recovery2.unknown_batches == []
    assert recovery2.exit_report_available


def test_replay_detects_tampered_bucket(tmp_path):
    db = tmp_path / "ledger.db"
    run_session(SAMPLE_DIR, db)
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT request_id, batch_id, bucket FROM hits ORDER BY request_id LIMIT 1"
    ).fetchone()
    conn.execute(
        "UPDATE hits SET bucket=? WHERE request_id=?",
        ((row[2] + 1) % 10000, row[0]),
    )
    conn.commit()
    conn.close()
    with pytest.raises(GSBError) as exc:
        recover_session(SAMPLE_DIR, db)
    assert exc.value.code == "E_REPLAY"
    assert exc.value.context["batch_id"] == row[1]
    assert exc.value.context["request_id"] == row[0]
    assert exc.value.context["field"] == "bucket"


def test_replay_detects_tampered_conclusion(tmp_path):
    db = tmp_path / "ledger.db"
    run_session(SAMPLE_DIR, db)
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT batch_id, conclusion_json FROM batches ORDER BY rowid LIMIT 1"
    ).fetchone()
    import json

    conclusion = json.loads(row[1])
    conclusion["route"] = "stable" if conclusion["route"] == "gray" else "gray"
    conn.execute(
        "UPDATE batches SET conclusion_json=? WHERE batch_id=?",
        (json.dumps(conclusion, sort_keys=True), row[0]),
    )
    conn.commit()
    conn.close()
    with pytest.raises(GSBError) as exc:
        recover_session(SAMPLE_DIR, db)
    assert exc.value.code == "E_REPLAY"
    assert exc.value.context["batch_id"] == row[0]
    assert exc.value.context["field"] == "conclusion.route"


def test_snapshot_replace_on_spec_change(tmp_path):
    import shutil

    changed = tmp_path / "samples_changed"
    shutil.copytree(SAMPLE_DIR, changed)
    db = tmp_path / "ledger.db"
    run_session(SAMPLE_DIR, db)
    # 修改权重后重开：整体替换并列出差异项
    rules_path = changed / "rules.json"
    rules_path.write_text(rules_path.read_text().replace("5000", "6000"))
    report = run_session(changed, db)
    assert report.snapshot_diffs, "应报告快照差异"
    assert "content_digest" in report.snapshot_diffs[0]
