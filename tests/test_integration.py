import json
import os
import signal
import sqlite3
import subprocess
import sys

import pytest

from receipt_ledger.errors import PreviewError, ReplayError
from receipt_ledger.ledger import Ledger
from receipt_ledger.merge import build_batch_plan
from receipt_ledger.preview import preview, recompute, rederive_conclusions
from receipt_ledger.scheduler import Scheduler, WorkerPool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "samples", "v1")


def _kill_sample_workers():
    """回收硬崩溃测试遗留的工作器子进程（按端口匹配命令行）。"""
    import subprocess as sp
    listing = sp.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True
    )
    for line in listing.stdout.splitlines():
        if "receipt_ledger.worker" not in line or "--workers" not in line:
            continue
        pid = int(line.strip().split(maxsplit=1)[0])
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_engine(spec, db_path, crash_after=None):
    with Ledger(db_path) as ledger:
        ledger.register_spec(spec)
        plan = build_batch_plan(spec)
        with WorkerPool(spec) as pool:
            scheduler = Scheduler(spec, ledger, pool,
                                  crash_after_commits=crash_after)
            scheduler.run(plan)
        return ledger, plan


def test_end_to_end_three_formats_and_byte_identical_preview(isolated_spec, db_path):
    v1_spec = isolated_spec
    ledger, plan = _run_engine(v1_spec, db_path)
    with Ledger(db_path) as reopened:
        data = preview(reopened, v1_spec)
        stats = recompute(reopened, v1_spec)
        assert stats["matched"] == len(plan)
        assert len(data["batches"]) == len(plan)
        receipts = reopened.receipts_of_batch(
            v1_spec.spec_version, plan[0].batch_id
        )
        formats = sorted(item["source_format"] for item in receipts)
        assert formats == ["r1", "r2", "r3"]
        adopted = [item for item in receipts if item["adopted"]]
        assert len(adopted) == 1 and adopted[0]["source_format"] == "r3"


def test_concurrency_limits_respected(isolated_spec, db_path):
    v1_spec = isolated_spec
    observed = {"active": 0, "max_active": 0, "per_worker": {}}
    original = Scheduler._per_request_outputs

    def wrapped(self, worker, task, winner_view):
        # 此方法在双重信号量均已获取后执行，观测值即真实并发占用。
        observed["active"] += 1
        observed["max_active"] = max(
            observed["max_active"], observed["active"]
        )
        counters = observed["per_worker"].setdefault(task.worker_id, [0, 0])
        counters[0] += 1
        counters[1] = max(counters[1], counters[0])
        try:
            return original(self, worker, task, winner_view)
        finally:
            counters[0] -= 1
            observed["active"] -= 1

    Scheduler._per_request_outputs = wrapped
    try:
        _run_engine(v1_spec, db_path)
    finally:
        Scheduler._per_request_outputs = original
    assert observed["max_active"] <= v1_spec.global_concurrency
    assert max(peak for _, peak in observed["per_worker"].values()) <= \
        v1_spec.per_worker_concurrency


def test_preview_drift_raises_e_preview(isolated_spec, db_path):
    v1_spec = isolated_spec
    _run_engine(v1_spec, db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT conclusion_json FROM batches ORDER BY seq LIMIT 1"
    ).fetchone()
    conclusion = json.loads(row[0])
    conclusion["output_digest"] = "f" * 64
    conn.execute(
        "UPDATE batches SET conclusion_json=? WHERE batch_id=?",
        (json.dumps(conclusion, sort_keys=True), conclusion["batch_id"]),
    )
    conn.commit()
    conn.close()
    with Ledger(db_path) as ledger:
        with pytest.raises(PreviewError) as exc:
            recompute(ledger, v1_spec)
        assert exc.value.code == "E_PREVIEW"
        assert exc.value.location["batch_id"] == conclusion["batch_id"]
        assert exc.value.location["field"] == "output_digest"


def test_forced_kill_recovery_and_unknown_marking(isolated_spec, db_path):
    v1_spec = isolated_spec
    plan = build_batch_plan(v1_spec)
    env = dict(os.environ)
    workers_path = v1_spec.source_paths["workers"]
    requests_path = v1_spec.source_paths["requests"]
    rules_path = v1_spec.source_paths["rules"]
    crash_code = (
        "from receipt_ledger.spec import load_spec;"
        "from receipt_ledger.ledger import Ledger;"
        "from receipt_ledger.merge import build_batch_plan;"
        "from receipt_ledger.scheduler import Scheduler, WorkerPool;"
        f"spec=load_spec(r'{workers_path}',r'{requests_path}',"
        f"r'{rules_path}');"
        f"ledger=Ledger(r'{db_path}');ledger.register_spec(spec);"
        "plan=build_batch_plan(spec);"
        "pool=WorkerPool(spec);pool.start();"
        "s=Scheduler(spec,ledger,pool,crash_after_commits=3);"
        "s.run(plan)"
    )
    proc = subprocess.run([sys.executable, "-c", crash_code],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 137
    _kill_sample_workers()

    with Ledger(db_path) as reopened:
        reopened.register_spec(v1_spec)
        state = reopened.recovery_state(v1_spec, plan)
        assert len(state["committed"]) == 3
        assert state["unknown"] == [t.batch_id for t in plan[3:]]
        assert state["exit_report"] is None
        with WorkerPool(v1_spec) as pool:
            scheduler = Scheduler(v1_spec, reopened, pool)
            scheduler.run(plan, start_seq=state["resume_from_seq"])
        reopened.verify_chain(v1_spec.spec_version)
        rederived = rederive_conclusions(v1_spec, plan)
        reopened.verify_replay(v1_spec.spec_version, rederived)


def test_replay_mismatch_locates_batch_seq_field(isolated_spec, db_path):
    v1_spec = isolated_spec
    _run_engine(v1_spec, db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT per_request_outputs,batch_id FROM batches ORDER BY seq LIMIT 1"
    ).fetchone()
    outputs = json.loads(row[0])
    outputs[0][1] = "9" * 64
    conn.execute(
        "UPDATE batches SET per_request_outputs=? WHERE batch_id=?",
        (json.dumps(outputs, sort_keys=True), row[1]),
    )
    conn.commit()
    conn.close()
    plan = build_batch_plan(v1_spec)
    with Ledger(db_path) as ledger:
        with pytest.raises(ReplayError) as exc:
            ledger.verify_replay(
                v1_spec.spec_version, rederive_conclusions(v1_spec, plan)
            )
        assert exc.value.code == "E_REPLAY"
        assert exc.value.location["batch_id"] == row[1]
        assert exc.value.location["seq"] == outputs[0][0]
        assert exc.value.location["field"] == "output_digest"
