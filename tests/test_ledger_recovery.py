"""要求7/8：参数演进失效传播、账本事务、崩溃恢复与独立重放。"""

import pytest

from slo_burn.engine import run_pipeline
from slo_burn.errors import ReplayError
from slo_burn.ledger import Ledger
from slo_burn.lease_client import EvaluatorProcess
from slo_burn.loader import load_param_edits, load_service_specs
from slo_burn.params import build_version_chain
from slo_burn.recovery import recover
from slo_burn.replay import verify_ledger
from slo_burn.scheduler import WindowScheduler
from slo_burn.lease_client import LeaseClient


def _run(workspace, ledger_path, global_limit=3, per_service_limit=2):
    socket_path = str(ledger_path) + ".sock"
    with EvaluatorProcess(socket_path, global_limit, per_service_limit):
        with LeaseClient(socket_path) as client:
            scheduler = WindowScheduler(client, global_limit, per_service_limit)
            return run_pipeline(workspace, ledger_path, scheduler=scheduler)


def test_ledger_commit_carries_format_version_and_digest(demo_workspace, tmp_path):
    ledger_path = tmp_path / "ledger.db"
    report = _run(demo_workspace, ledger_path)
    assert report["latest"]["format_version"] == 1
    assert len(report["latest"]["content_digest"]) == 64
    with Ledger(ledger_path) as ledger:
        assert ledger.connection.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == 1


def test_stale_conclusions_excluded_from_active_view(demo_workspace, tmp_path):
    ledger_path = tmp_path / "ledger.db"
    _run(demo_workspace, ledger_path)
    with Ledger(ledger_path) as ledger:
        rows = ledger.connection.execute(
            "SELECT service_id, param_version, stale, MIN(minute), MAX(minute), COUNT(*)"
            " FROM conclusions GROUP BY 1, 2, 3 ORDER BY 1, 2"
        ).fetchall()
    auth_versions = [row for row in rows if row[0] == "auth-service"]
    # 参数演进后每分钟只归属当前版本：三个版本区间互不重叠且连续
    assert [row[1] for row in auth_versions] == [1, 2, 3]
    assert auth_versions[0][3:5] == (0, 60)
    assert auth_versions[1][3:5] == (61, 120)
    assert auth_versions[2][3:5] == (121, 149)
    # 每个 (service, minute) 恰有一条活跃结论
    with Ledger(ledger_path) as ledger:
        duplicate_active = ledger.connection.execute(
            "SELECT service_id, minute FROM conclusions WHERE stale=0"
            " GROUP BY service_id, minute HAVING COUNT(*) > 1"
        ).fetchall()
        assert duplicate_active == []


def test_mark_stale_replaces_old_conclusions_on_re_evaluation(demo_workspace, tmp_path):
    ledger_path = tmp_path / "ledger.db"
    _run(demo_workspace, ledger_path)
    with Ledger(ledger_path) as ledger:
        seq = ledger.mark_stale("auth-service", range(0, 61))
        assert seq >= 2
        stale_count = ledger.connection.execute(
            "SELECT COUNT(*) FROM conclusions WHERE service_id='auth-service' AND stale=1"
        ).fetchone()[0]
        assert stale_count == 61
        # 陈旧结论不参与活跃展示
        active = ledger.active_conclusions()
        assert all(
            not (item["service_id"] == "auth-service" and item["minute"] <= 60)
            for item in active
        )


def test_param_chain_versions_are_monotonic(demo_workspace):
    specs = load_service_specs(demo_workspace / "services.json")
    edits = load_param_edits(demo_workspace / "param_edits.json", specs)
    chains = build_version_chain(specs, edits)
    auth_versions = [item.spec.param_version for item in chains["auth-service"]]
    assert auth_versions == [1, 2, 3]
    v2 = next(item for item in chains["auth-service"] if item.spec.param_version == 2)
    assert v2.effective_from == 60
    assert v2.effective_to == 120


def test_replay_matches_ledger(demo_workspace, tmp_path):
    ledger_path = tmp_path / "ledger.db"
    _run(demo_workspace, ledger_path)
    result = verify_ledger(
        ledger_path,
        demo_workspace / "services.json",
        demo_workspace / "batches",
        demo_workspace / "param_edits.json",
    )
    assert result["ok"] is True
    assert result["rows"] == 450


def test_replay_detects_tampered_ledger(demo_workspace, tmp_path):
    ledger_path = tmp_path / "ledger.db"
    _run(demo_workspace, ledger_path)
    with Ledger(ledger_path) as ledger:
        ledger.connection.execute(
            "UPDATE conclusions SET remaining_budget_scaled = remaining_budget_scaled + 1"
            " WHERE stale=0 AND service_id='checkout-api'"
        )
    with pytest.raises(ReplayError) as exc:
        verify_ledger(
            ledger_path,
            demo_workspace / "services.json",
            demo_workspace / "batches",
            demo_workspace / "param_edits.json",
        )
    assert exc.value.code == "E_REPLAY"
    assert exc.value.context["service_id"] == "checkout-api"


def test_recovery_only_uses_committed_records(demo_workspace, tmp_path):
    ledger_path = tmp_path / "ledger.db"
    _run(demo_workspace, ledger_path)
    with Ledger(ledger_path) as ledger:
        ledger.register_pending(
            "checkout-api",
            999,
            {"minute": 999, "successes": 1, "failures": 1},
        )
    report = recover(ledger_path)
    assert report.last_commit_seq == 1
    assert report.unknown_batches[0]["status"] == "unknown"
    assert report.unknown_batches[0]["minute"] == 999
    assert report.reclaimed_lease_quota[0]["action"] == "release-lease-quota-before-rerun"
    assert "退出报告" in report.note


def test_format_version_mismatch_triggers_full_replacement(demo_workspace, tmp_path):
    ledger_path = tmp_path / "ledger.db"
    _run(demo_workspace, ledger_path)
    with Ledger(ledger_path) as ledger:
        ledger.connection.execute(
            "UPDATE meta SET value='999' WHERE key='format_version'"
        )
    with pytest.raises(Exception):
        Ledger(ledger_path)


def test_snapshot_mismatch_requires_full_replacement_not_merge(tmp_path):
    ledger_path = tmp_path / "replace.db"
    with Ledger(ledger_path) as ledger:
        ledger.write_commit(kind="seed", conclusions=[])
        assert ledger.latest_commit() is not None
        ledger.replace_all()
        assert ledger.latest_commit() is None
        assert ledger.connection.execute("SELECT COUNT(*) FROM conclusions").fetchone()[0] == 0
        # 格式版本 meta 保留，业务表整体清空而非增量合并
        assert ledger.connection.execute(
            "SELECT value FROM meta WHERE key='format_version'"
        ).fetchone()[0] == "1"
