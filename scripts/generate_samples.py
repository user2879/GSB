"""Generate the deterministic sample scene JSON files under samples/.

Seed/rate pairs were chosen against ``mock_service.deterministic_bucket`` so
the outcome sequence of every node is provable, e.g. seed 0 with failure_rate
4000 yields bucket [0, 4729, 9458] -> [failure, success, ...], while seed 2
with slow_rate 9000 and a slow delay above the timeout yields three real
socket timeouts and an E_RETRY terminal marker.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def node(node_id: str, port: int, process_ms: int, failure_rate: int,
         slow_rate: int = 0, slow_extra_ms: int = 0, seed: int = 0) -> dict:
    return {
        "node_id": node_id,
        "host": "127.0.0.1",
        "port": port,
        "process_ms": process_ms,
        "failure_rate": failure_rate,
        "slow_rate": slow_rate,
        "slow_extra_ms": slow_extra_ms,
        "seed": seed,
    }


def write(name: str, payload: dict) -> None:
    target = SAMPLES / name
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {target}")


def base(scene_version: str, nodes: list[dict], task: dict, scheduling: dict,
         slo: dict) -> dict:
    return {
        "scene_version": scene_version,
        "nodes": nodes,
        "task": task,
        "scheduling": scheduling,
        "slo": slo,
    }


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)

    # 1) cold start: every delivery succeeds, but two nodes need a retry first
    #    (one deterministic HTTP failure, one genuine socket timeout).
    task_cold = {
        "task_id": "payload-broadcast-A",
        "payload_bytes": 4096,
        "max_attempts": 3,
        "attempt_timeout_ms": 250,
        "backoff_ms": [20, 40, 80],
        "deliveries_per_node": 1,
    }
    scene1 = base(
        "cold-start-v1",
        [
            node("n1", 48801, process_ms=20, failure_rate=1000, seed=1),
            node("n2", 48802, process_ms=25, failure_rate=4000, seed=0),
            node("n3", 48803, process_ms=25, failure_rate=0, slow_rate=4000,
                 slow_extra_ms=400, seed=0),
            node("n4", 48804, process_ms=30, failure_rate=1500, seed=1),
            node("n5", 48805, process_ms=30, failure_rate=2000, seed=1),
        ],
        task_cold,
        {"global_concurrency": 2, "node_concurrency": 1, "session_budget_ms": 60000},
        {"success_rate_percent": 90, "p95_latency_ms": 450},
    )
    write("scene_cold_start.json", scene1)

    # 2) tightened SLO + higher global concurrency; the SLO thresholds are now
    #    unreachable and one node exhausts all attempts (E_RETRY).
    task_tune = dict(task_cold)
    task_tune["attempt_timeout_ms"] = 120
    task_tune["backoff_ms"] = [10, 20, 40]
    scene2 = base(
        "retune-v2",
        [
            node("n1", 48901, process_ms=20, failure_rate=1000, seed=1),
            node("n2", 48902, process_ms=20, failure_rate=7000, seed=2),
            node("n3", 48903, process_ms=20, failure_rate=0, slow_rate=9000,
                 slow_extra_ms=300, seed=2),
            node("n4", 48904, process_ms=25, failure_rate=1500, seed=1),
            node("n5", 48905, process_ms=25, failure_rate=2000, seed=1),
        ],
        task_tune,
        {"global_concurrency": 5, "node_concurrency": 1, "session_budget_ms": 60000},
        {"success_rate_percent": 99, "p95_latency_ms": 90},
    )
    write("scene_retune.json", scene2)

    # 3) crash/recovery: two deliveries per node, low global concurrency so the
    #    queueing/wait ledger is exercised; timeout pressure is deterministic.
    task_crash = {
        "task_id": "payload-broadcast-A",
        "payload_bytes": 4096,
        "max_attempts": 3,
        "attempt_timeout_ms": 120,
        "backoff_ms": [10, 20, 40],
        "deliveries_per_node": 2,
    }
    scene3 = base(
        "crash-recovery-v1",
        [
            node("n1", 49001, process_ms=15, failure_rate=1000, seed=1),
            node("n2", 49002, process_ms=18, failure_rate=4000, seed=0),
            node("n3", 49003, process_ms=18, failure_rate=0, slow_rate=9000,
                 slow_extra_ms=250, seed=2),
            node("n4", 49004, process_ms=20, failure_rate=1500, seed=1),
            node("n5", 49005, process_ms=20, failure_rate=2000, seed=1),
        ],
        task_crash,
        {"global_concurrency": 2, "node_concurrency": 1, "session_budget_ms": 60000},
        {"success_rate_percent": 80, "p95_latency_ms": 450},
    )
    write("scene_crash_recovery.json", scene3)


if __name__ == "__main__":
    main()
