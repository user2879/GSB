from __future__ import annotations

import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .cluster import PROJECT_ROOT, Cluster
from .engine import Engine
from .ledger import Ledger
from .replay import verify_ledger
from .report import conclusion_from_engine
from .runner import Runner
from .scheduler import monotonic_ms
from .spec import load_bundle


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def prepare_acceptance_files(work_dir: Path) -> Any:
    work_dir.mkdir(parents=True, exist_ok=True)
    examples = PROJECT_ROOT / "examples"
    nodes_doc = json.loads((examples / "nodes.json").read_text(encoding="utf-8"))
    for node in nodes_doc["nodes"]:
        node["port"] = find_free_port()
    artifacts_text = (examples / "artifacts.json").read_text(encoding="utf-8")
    events_text = (examples / "events.json").read_text(encoding="utf-8")
    (work_dir / "nodes.json").write_text(json.dumps(nodes_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (work_dir / "artifacts.json").write_text(artifacts_text, encoding="utf-8")
    (work_dir / "events.json").write_text(events_text, encoding="utf-8")
    return load_bundle(work_dir / "nodes.json", work_dir / "artifacts.json", work_dir / "events.json")


def run_acceptance(work_dir: str | Path, *, base_port: int | None = None) -> dict[str, Any]:
    del base_port
    work = Path(work_dir)
    bundle = prepare_acceptance_files(work)
    ledger_path = work / "cacheflow.sqlite3"
    steps: list[dict[str, Any]] = []

    with Runner(bundle, work, rpc_timeout_seconds=10.0) as runner:
        run_result = runner.run()
    pre_kill_verify = verify_ledger(bundle, ledger_path)
    (work / "pre-kill-replay.canonical.json").write_bytes(pre_kill_verify["canonical_bytes"])
    steps.append(
        {
            "step": "normal_pipeline",
            "snapshot_digest": run_result["snapshot_digest"],
            "peaks": run_result["peaks"],
            "wait_record_count": len(run_result["wait_records"]),
            "unique_holding_ok": pre_kill_verify["unique_holding_ok"],
        }
    )

    cluster = Cluster(list(bundle.nodes), ledger_path=ledger_path).start_all()
    ledger = Ledger(ledger_path)
    ledger.initialize_schema()
    crash_event = {
        "seq": 100,
        "type": "grant",
        "lease_id": "lease-crash-rerun",
        "artifact_key": "art-delta",
        "node_id": "node-a",
        "quota_version": 2,
        "ttl_ms": 600_000,
    }
    planner = Engine(bundle)
    crash_plan: dict[str, Any] | None = None
    for event in bundle.events:
        planner.apply_spec_event(event)
    crash_plan = planner.apply_spec_event(crash_event)
    request_id = "req:100:lease-crash-rerun"
    ledger.begin_inflight(
        request_id=request_id,
        event_seq=100,
        node_id="node-a",
        lease_id=crash_event["lease_id"],
        op="grant",
        payload={**crash_event, "block_forever": True},
        wait_started_ms=monotonic_ms(),
        wait_ended_ms=monotonic_ms(),
        provisional_holds=[dict(hold, status="provisional") for hold in crash_plan["holds"]],
    )

    def blocked_grant() -> dict[str, Any]:
        client = cluster.client("node-a", timeout_seconds=5.0)
        try:
            return client.call(
                "grant",
                lease_id=crash_event["lease_id"],
                artifact_key=crash_event["artifact_key"],
                quota_version=2,
                ttl_ms=600_000,
                block_forever=True,
                block_timeout_seconds=30.0,
            )
        finally:
            client.close()

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crash-probe")
    future = pool.submit(blocked_grant)
    time.sleep(0.5)
    forced_returncode = cluster.kill_node("node-a")
    blocked_error: dict[str, Any] | None = None
    try:
        future.result(timeout=5.0)
    except Exception as exc:
        blocked_error = {
            "type": type(exc).__name__,
            "code": getattr(exc, "code", "E_SOCKET"),
            "message": str(exc),
        }
    pool.shutdown(wait=False, cancel_futures=True)
    steps.append(
        {
            "step": "force_kill_inflight",
            "node_id": "node-a",
            "forced_returncode": forced_returncode,
            "blocked_request_error": blocked_error,
            "node_exit_report_promised": False,
        }
    )

    unknown_reclaimed = ledger.recover_unknown_inflight()
    steps.append(
        {
            "step": "unknown_marked_and_quota_reclaimed",
            "request_ids": [item["reclaim_id"] for item in unknown_reclaimed],
            "hold_ids": [hold_id for item in unknown_reclaimed for hold_id in item["hold_ids"]],
            "status": "unknown_rerun_required",
        }
    )

    cluster.restart_node("node-a")
    status_client = cluster.client("node-a", timeout_seconds=5.0)
    try:
        node_status = status_client.call("status")
        rerun_response = status_client.call(
            "grant",
            lease_id=crash_event["lease_id"],
            artifact_key=crash_event["artifact_key"],
            quota_version=2,
            ttl_ms=600_000,
        )
    finally:
        status_client.close()

    ledger.commit_bundle(
        event=crash_event,
        status="committed",
        result=rerun_response,
        transitions=crash_plan.get("transitions"),
        leases=crash_plan.get("leases"),
        holds=crash_plan.get("holds"),
        decisions=crash_plan.get("decisions"),
        reclaims=crash_plan.get("reclaims"),
        inflight_id=request_id,
    )
    final_conclusion = conclusion_from_engine(bundle, planner, cutoff_seq=100)
    final_digest, _ = ledger.save_snapshot(cutoff_seq=100, report=final_conclusion)
    ledger.close()

    post_kill_verify = verify_ledger(bundle, ledger_path)
    replay_path = work / "post-kill-replay.canonical.json"
    replay_path.write_bytes(post_kill_verify["canonical_bytes"])
    repeat_verify = verify_ledger(bundle, ledger_path)
    bytewise_repeat = repeat_verify["canonical_bytes"] == replay_path.read_bytes()

    recovered_lease_status = [
        lease
        for lease in node_status.get("leases", [])
        if lease.get("lease_id") in {"lease-beta", "lease-delta"}
    ]
    capacity = json.loads(post_kill_verify["canonical_bytes"])["capacity"]
    result = {
        "ok": True,
        "code": "E_OK",
        "work_dir": str(work),
        "ledger_path": str(ledger_path),
        "replay_path": str(replay_path),
        "pre_kill_digest": pre_kill_verify["digest"],
        "post_kill_digest": post_kill_verify["digest"],
        "final_snapshot_digest": final_digest,
        "bytewise_repeat": bytewise_repeat,
        "unique_holding_ok": post_kill_verify["unique_holding_ok"],
        "capacity": capacity,
        "recovered_node_leases": recovered_lease_status,
        "steps": steps,
    }
    (work / "acceptance-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cluster.close()
    return result
