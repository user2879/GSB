import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from mvcc_lab.digest import digest
from mvcc_lab.errors import ReplayError
from mvcc_lab.loader import load_spec
from mvcc_lab.ledger import Ledger
from mvcc_lab.runner import Runner, compare_decisions
from mvcc_lab.visibility import decide_all

ROOT = Path(__file__).parents[1]
SPEC = ROOT / "samples" / "sample_spec.json"
CHANGED = ROOT / "samples" / "sample_spec_changed.json"


def test_round_atomic_commit_and_cursor(tmp_path):
    spec = load_spec(SPEC)
    db = tmp_path / "led.sqlite"
    runner = Runner(spec, str(SPEC), str(db), batch_size=2)
    report = runner.run_full_round()
    runner.close()
    conn = sqlite3.connect(db)
    state = conn.execute(
        "SELECT state, content_digest FROM rounds WHERE round_id=?",
        (report["round_id"],),
    ).fetchone()
    assert state[0] == "committed"
    assert state[1] == report["content_digest"]
    count = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE round_id=?",
        (report["round_id"],),
    ).fetchone()[0]
    assert count == len(spec.queries)
    traces = conn.execute(
        "SELECT COUNT(*) FROM schedule_traces WHERE round_id=?",
        (report["round_id"],),
    ).fetchone()[0]
    assert traces == len(spec.queries)


def test_identity_mismatch_triggers_full_replacement(tmp_path):
    db = tmp_path / "led.sqlite"
    first = Runner(load_spec(SPEC), str(SPEC), str(db))
    first.run_full_round()
    first.close()
    second = Runner(load_spec(CHANGED), str(CHANGED), str(db))
    identity = second.ledger.identify_or_replace()
    second.close()
    assert identity["action"] == "replaced"
    assert identity["backup"] and Path(identity["backup"]).exists()
    backup = Path(identity["backup"])
    assert oct(backup.stat().st_mode & 0o777) == "0o444"
    assert any(d["field"] == "content_digest" for d in identity["diff"])
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM replacement_log").fetchone()[0] == 1


def test_force_kill_leaves_open_round_and_recovery_reruns(tmp_path):
    db = tmp_path / "kill.sqlite"
    env = dict(os.environ)
    kill_cmd = [
        sys.executable, "-m", "mvcc_lab",
        "--config", str(SPEC), "--ledger", str(db),
        "--kill-after", "3",
    ]
    killed = subprocess.run(kill_cmd, cwd=ROOT, env=env,
                            capture_output=True, text=True)
    # Python 子进程被 SIGKILL 时返回负信号码 -9（shell 视角为 137）。
    assert killed.returncode == -9
    # 强制终止时不生成退出报告。
    assert killed.stdout.strip() == ""

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT state FROM rounds").fetchall() == [("open",)]
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    unknown = conn.execute(
        "SELECT unknown_query_ids FROM progress"
    ).fetchone()[0]
    assert "Q1" in unknown
    conn.close()

    # 重启恢复：仅依据已完整提交记录，open 轮次全部标记未知并重跑。
    runner = Runner(load_spec(SPEC), str(SPEC), str(db), replay=True)
    report = runner.run_full_round()
    runner.close()
    assert report["recovery"]["open_rounds"] == [1]
    assert len(report["recovery"]["unknown_query_ids"]) == 8
    assert report["replay_check"]["status"] == "consistent"

    conn = sqlite3.connect(db)
    states = conn.execute(
        "SELECT round_id, state FROM rounds ORDER BY round_id"
    ).fetchall()
    assert states == [(1, "open"), (2, "committed")]


def test_replay_mismatch_raises_e_replay_with_location():
    base = {
        "query_id": "Q1", "read_point_id": "RP-1", "row_key": "R-A",
        "visible_version": 1, "visible_txn_id": "T1", "excluded": [],
        "reason_chain": [], "tie_applied": False,
        "error_influenced": False, "causes": [], "rule_version": "v",
    }
    bad = dict(base)
    bad["reason_chain"] = ["superseded"]
    with pytest.raises(ReplayError) as info:
        compare_decisions([base], [bad])
    assert info.value.code == "E_REPLAY"
    assert info.value.field == "reason_chain"
    assert info.value.object_id == "RP-1/R-A/Q1"


def test_replay_consistent_with_one_shot_decisions(tmp_path):
    spec = load_spec(SPEC)
    db = tmp_path / "rep.sqlite"
    runner = Runner(spec, str(SPEC), str(db), replay=True)
    report = runner.run_full_round()
    runner.close()
    expected = [d.summary() for d in decide_all(spec)]
    compare_decisions(expected, report["summaries"])  # 不抛即一致
