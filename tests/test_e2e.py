import json
import os
import subprocess
import sys

import pytest

from chaos import recover, replay, timeline
from chaos.errors import E_REPLAY, ChaosError
from chaos.ledger import Ledger
from chaos.orchestrator import Orchestrator, check_observation_isolation
from chaos.structlog import StructLog

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def paths(tmp_path):
    return {
        "ledger": str(tmp_path / "ledger.sqlite"),
        "snap": str(tmp_path / "snapshots"),
        "log": StructLog(str(tmp_path / "run.jsonl")),
    }


def _prepare(resolved_spec, paths):
    entries = timeline.generate(resolved_spec)
    tdigest = timeline.timeline_digest(entries)
    ledger, report = recover.reconcile(resolved_spec, tdigest,
                                       paths["ledger"], paths["snap"],
                                       paths["log"])
    return entries, tdigest, ledger, report


def test_full_run_and_replay(resolved_spec, paths):
    entries, tdigest, ledger, _ = _prepare(resolved_spec, paths)
    orch = Orchestrator(resolved_spec, ledger, entries, paths["log"])
    result = orch.run()
    assert result["errors"] == []
    assert result["committed"] == 4
    # every injection reached reverted with 3 observation phases each
    rows = ledger.conn.execute(
        "SELECT state, COUNT(*) FROM injections GROUP BY state").fetchall()
    assert rows == [("reverted", 4)]
    obs = ledger.conn.execute(
        "SELECT COUNT(*), SUM(decidable) FROM observations").fetchone()
    assert obs[0] == 12
    # the suspended service cannot answer: its active window falls below
    # min_samples and must be marked statistically undecidable
    sus_seq = next(e.seq for e in entries if e.type == "suspend")
    row = ledger.conn.execute(
        "SELECT decidable FROM observations WHERE inj_seq=? AND phase='active'",
        (sus_seq,)).fetchone()
    assert row[0] == 0
    assert obs[1] == 11
    assert check_observation_isolation(ledger) == []
    count = replay.replay(resolved_spec, entries, ledger)
    assert count == 4
    # latency injection actually inflated latency vs pre window
    lat_seq = next(e.seq for e in entries if e.type == "latency")
    pre = ledger.conn.execute(
        "SELECT p50_fixed FROM observations WHERE inj_seq=? AND phase='pre'",
        (lat_seq,)).fetchone()[0]
    act = ledger.conn.execute(
        "SELECT p50_fixed FROM observations WHERE inj_seq=? AND phase='active'",
        (lat_seq,)).fetchone()[0]
    assert act > pre
    ledger.close()


def test_replay_detects_tamper(resolved_spec, paths):
    entries, tdigest, ledger, _ = _prepare(resolved_spec, paths)
    Orchestrator(resolved_spec, ledger, entries, paths["log"]).run()
    seq = ledger.conn.execute("SELECT MIN(seq) FROM injections").fetchone()[0]
    with ledger.transaction() as cur:
        cur.execute("UPDATE observations SET rate_fixed = rate_fixed + 1"
                    " WHERE inj_seq=? AND phase='active'", (seq,))
    with pytest.raises(ChaosError) as exc:
        replay.replay(resolved_spec, entries, ledger)
    assert exc.value.code == E_REPLAY
    assert exc.value.evidence["seq"] == seq
    assert "rate_fixed" in exc.value.evidence["field"]
    ledger.close()


def _write_spec(tmp_path, resolved_spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(resolved_spec.to_dict()))
    return str(path)


def test_forced_termination_and_resume(tmp_path, resolved_spec):
    spec_path = _write_spec(tmp_path, resolved_spec)
    ledger = str(tmp_path / "ledger.sqlite")
    snap = str(tmp_path / "snap")
    logs = str(tmp_path / "logs")
    env = dict(os.environ, PYTHONPATH=REPO)
    # 1. run that crashes hard after the first commit
    proc = subprocess.run(
        [sys.executable, "-m", "chaos.cli", "run", "--spec", spec_path,
         "--ledger", ledger, "--snapshot-dir", snap, "--log-dir", logs,
         "--kill-after", "1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 137, proc.stderr
    # 2. resume: unknown injection marked, redo plan executed, replay passes
    proc2 = subprocess.run(
        [sys.executable, "-m", "chaos.cli", "resume", "--spec", spec_path,
         "--ledger", ledger, "--snapshot-dir", snap, "--log-dir", logs],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert proc2.returncode == 0, proc2.stderr
    summary = json.loads(proc2.stdout)
    recovery = summary["recovery"]
    assert recovery["unknown_count"] >= 1
    assert recovery["recovered_count"] >= 1
    assert recovery["redo_plan"] == recovery["unknown"]
    assert summary["replayed_injections"] == 4
    assert summary["isolation_violations"] == []
    # 3. ledger shows unknown transitions and final reverted states
    conn = Ledger(ledger)
    states = [r[0] for r in conn.conn.execute(
        "SELECT DISTINCT to_state FROM transitions")]
    assert "unknown" in states
    final = conn.conn.execute(
        "SELECT state, COUNT(*) FROM injections GROUP BY state"
        " ORDER BY state").fetchall()
    assert ("reverted", 4) in final
    conn.close()
    # 4. verify subcommand passes
    proc3 = subprocess.run(
        [sys.executable, "-m", "chaos.cli", "verify", "--spec", spec_path,
         "--ledger", ledger, "--snapshot-dir", snap, "--log-dir", logs],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert proc3.returncode == 0, proc3.stderr


@pytest.fixture(autouse=True, scope="module")
def _cleanup_services():
    yield
    if os.name != "nt":
        subprocess.run(["pkill", "-f", "[c]haos.service_sim"],
                       capture_output=True)
