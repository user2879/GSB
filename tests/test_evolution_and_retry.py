import json
import os
import sqlite3
import subprocess
import sys

from receipt_ledger.ledger import Ledger
from receipt_ledger.merge import build_batch_plan
from receipt_ledger.spec import load_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "samples", "v1")
V2 = os.path.join(ROOT, "samples", "v2")


def test_stale_isolation_and_full_replacement(isolated_spec, db_path, tmp_path):
    v1_spec = isolated_spec
    with Ledger(db_path) as ledger:
        ledger.register_spec(v1_spec)
        plan = build_batch_plan(v1_spec)
        from receipt_ledger.scheduler import Scheduler, WorkerPool
        with WorkerPool(v1_spec) as pool:
            Scheduler(v1_spec, ledger, pool).run(plan)

    v2_workers = tmp_path / "workers.v2.json"
    with open(os.path.join(V2, "workers.json"), encoding="utf-8") as handle:
        workers_v2 = json.load(handle)
    port_map = {
        old.worker_id: old.port for old in v1_spec.workers
    }
    for item in workers_v2:
        if item["worker_id"] in port_map:
            item["port"] = port_map[item["worker_id"]]
    free_handle = __import__("socket").socket()
    free_handle.bind(("127.0.0.1", 0))
    delta_port = free_handle.getsockname()[1]
    free_handle.close()
    for item in workers_v2:
        if item["worker_id"] == "worker-delta":
            item["port"] = delta_port
    v2_workers.write_text(json.dumps(workers_v2), encoding="utf-8")
    spec_v2 = load_spec(
        str(v2_workers),
        os.path.join(V2, "requests.json"),
        os.path.join(V2, "rules.json"),
        spec_version=2,
    )
    with Ledger(db_path) as ledger:
        diff = ledger.register_spec(spec_v2)
        assert diff["replaced"] is True
        assert "worker-delta" in diff["added"]
        assert "worker-beta" in diff["removed"]
        assert any(item[0] == "worker-alpha" for item in diff["changed"])
        assert os.path.exists(diff["backup"])
        assert not os.access(diff["backup"], os.W_OK)
        rows = ledger.conn.execute(
            "SELECT spec_version,status,COUNT(*) FROM batches "
            "GROUP BY spec_version,status ORDER BY spec_version"
        ).fetchall()
        assert [tuple(r) for r in rows] == [(1, "stale", 7)]
        assert ledger.preview_rows(2) == []

        plan2 = build_batch_plan(spec_v2)
        from receipt_ledger.scheduler import Scheduler, WorkerPool
        with WorkerPool(spec_v2) as pool:
            Scheduler(spec_v2, ledger, pool).run(plan2)
        assert len(ledger.preview_rows(2)) == 7


def test_retry_preserves_attribution_and_digest(isolated_spec, db_path):
    v1_spec = isolated_spec
    env = dict(os.environ, RL_FAULT_BATCHES="rule-gray-b0000:2")
    workers_path = v1_spec.source_paths["workers"]
    requests_path = v1_spec.source_paths["requests"]
    rules_path = v1_spec.source_paths["rules"]
    code = (
        "from receipt_ledger.spec import load_spec;"
        "from receipt_ledger.ledger import Ledger;"
        "from receipt_ledger.merge import build_batch_plan;"
        "from receipt_ledger.scheduler import Scheduler, WorkerPool;"
        f"spec=load_spec(r'{workers_path}',r'{requests_path}',"
        f"r'{rules_path}');"
        f"ledger=Ledger(r'{db_path}');ledger.register_spec(spec);"
        "plan=build_batch_plan(spec);"
        "pool=WorkerPool(spec);pool.start();"
        "Scheduler(spec,ledger,pool).run(plan)"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT retry_count,output_digest FROM batches "
        "WHERE batch_id='rule-gray-b0000'"
    ).fetchone()
    assert row[0] == 2
    assert len(row[1]) == 64


def test_latency_fixed_point_stats(isolated_spec, db_path):
    v1_spec = isolated_spec
    from tests.test_integration import _run_engine
    from receipt_ledger.preview import preview, recompute

    _run_engine(v1_spec, db_path)
    with Ledger(db_path) as ledger:
        preview(ledger, v1_spec)
        preview(ledger, v1_spec)
        recompute(ledger, v1_spec)
        stats = ledger.latency_stats("preview")
        assert stats["count"] == 2
        assert stats["avg_q1000"] >= 0 and stats["max_ms"] >= 0
