import json
from pathlib import Path

import pytest

from dispatch_tokens.errors import DiffError
from dispatch_tokens.ledger import UNKNOWN, Ledger
from dispatch_tokens.runner import (Runner,
                                    verify_incremental_vs_rebuild)
from dispatch_tokens.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[1] / "samples" / "demo_spec.json"


def test_offline_runner_passes_rebuild_check(tmp_path):
    spec = load_spec(SPEC_PATH)
    report = Runner(spec, tmp_path / "offline.db",
                    with_workers=False).run()
    verify_incremental_vs_rebuild(spec, report)
    assert report["error_bounds"]["comparison_error_bound"] == 2


def test_live_runner_with_real_workers(tmp_path):
    spec = load_spec(SPEC_PATH)
    report = Runner(spec, tmp_path / "live.db",
                    with_workers=True).run()
    # 真实子进程为 DISPATCHED 分片产出 NumPy 聚合摘要。
    dispatched = [sid for sid, item in report["outcomes"].items()
                  if item["status"] == "DISPATCHED"]
    assert dispatched
    for shard_id in dispatched:
        summary = report["outcomes"][shard_id]["summary"]
        assert summary is not None
        assert summary["checksum"] >= 0
        assert isinstance(summary["elapsed_ms"], int)
    # 续约 RPC 同时覆盖成功与 E_LEASE 拒绝。
    outcomes = [item["node_accepted"] for item in report["renew_results"]]
    assert True in outcomes
    assert False in outcomes


def test_diff_check_detects_tampered_incremental(tmp_path):
    spec = load_spec(SPEC_PATH)
    report = Runner(spec, tmp_path / "d.db", with_workers=False).run()
    first_shard = next(iter(report["outcomes"]))
    report["outcomes"][first_shard]["param_version"] = 4242
    with pytest.raises(DiffError) as exc:
        verify_incremental_vs_rebuild(spec, report)
    assert exc.value.code == "E_DIFF"
    assert exc.value.location == f"shards[{first_shard}]"
    assert exc.value.field == "param_version"


def test_ledger_records_unknown_after_forced_kill(tmp_path):
    spec = load_spec(SPEC_PATH)
    db_path = tmp_path / "kill.db"
    Runner(spec, db_path, with_workers=False).run()
    ledger = Ledger(db_path, spec)
    ledger.mark_unknown("sh-3", "node-a", 99, 9999,
                        {"balance": 0, "param_version": 2})
    stored = ledger.load_outcomes()
    assert stored["sh-3"]["status"] == UNKNOWN
    assert stored["sh-3"]["ownership"]["decision"] == \
        "UNKNOWN_RERUN_RECLAIM_FIRST"
    ledger.close()
