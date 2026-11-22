"""Generate deterministic local JSON samples using only the stdlib json module."""
from __future__ import annotations

import json
from pathlib import Path


def write_samples(out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pools = {
        "pools": [
            {"pool_id": "pool-a", "gpu_count": 4, "memory_per_gpu_mib": 16384,
             "compute_units": 96, "quota_share": 40},
            {"pool_id": "pool-b", "gpu_count": 4, "memory_per_gpu_mib": 8192,
             "compute_units": 64, "quota_share": 35},
            {"pool_id": "pool-c", "gpu_count": 2, "memory_per_gpu_mib": 4096,
             "compute_units": 32, "quota_share": 25},
        ]
    }

    tasks = {
        "tasks": [
            {"task_id": "T1", "gpu_need": 2, "memory_per_gpu_mib": 4096,
             "priority": 5, "duration_ms": 100, "shard_count": 3,
             "submit_time_ms": 0, "memory_budget_mib": 24576, "deps": []},
            {"task_id": "T2", "gpu_need": 2, "memory_per_gpu_mib": 2048,
             "priority": 2, "duration_ms": 80, "shard_count": 4,
             "submit_time_ms": 0, "memory_budget_mib": 16384, "deps": []},
            {"task_id": "T3", "gpu_need": 4, "memory_per_gpu_mib": 4096,
             "priority": 9, "duration_ms": 120, "shard_count": 2,
             "submit_time_ms": 50, "memory_budget_mib": 32768, "deps": []},
            {"task_id": "T4", "gpu_need": 1, "memory_per_gpu_mib": 1024,
             "priority": 3, "duration_ms": 60, "shard_count": 2,
             "submit_time_ms": 70, "memory_budget_mib": 4096, "deps": ["T1"]},
            {"task_id": "T5", "gpu_need": 1, "memory_per_gpu_mib": 2048,
             "priority": 1, "duration_ms": 50, "shard_count": 2,
             "submit_time_ms": 90, "memory_budget_mib": 4096, "deps": []},
        ],
        "shards": [
            {"task_id": "T4", "index": 0, "deps": ["T1#0"]},
            {"task_id": "T4", "index": 1, "deps": ["T4#0", "T1#1"]},
            {"task_id": "T1", "index": 1, "deps": ["T1#0"]},
            {"task_id": "T1", "index": 2, "deps": ["T1#1"]},
        ],
    }

    params = {
        "global_gpu_quota": 8,
        "pool_concurrency_limit": 2,
        "scale_factor": 1000,
        "quota_share_cap": 100,
        "ledger_format_version": 1,
        "pool_version": "pool-v1",
        "task_version": "task-v1",
        "scale_version": "scale-v1",
    }

    for name, payload in (("pools.json", pools), ("tasks.json", tasks),
                          ("params.json", params)):
        (out / name).write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False),
            encoding="utf-8")


def inject_fault(samples_dir: str | Path, fault: str) -> None:
    """Fault injection only mutates input files / parameters, never engine code."""
    samples = Path(samples_dir)
    if fault == "memory":
        tasks = json.loads((samples / "tasks.json").read_text(encoding="utf-8"))
        tasks["tasks"][0]["memory_per_gpu_mib"] = 99999
        (samples / "tasks.json").write_text(json.dumps(tasks, sort_keys=True, indent=2),
                                            encoding="utf-8")
    elif fault == "quota":
        pools = json.loads((samples / "pools.json").read_text(encoding="utf-8"))
        pools["pools"][0]["quota_share"] = 90
        (samples / "pools.json").write_text(json.dumps(pools, sort_keys=True, indent=2),
                                            encoding="utf-8")
    elif fault == "cycle":
        tasks = json.loads((samples / "tasks.json").read_text(encoding="utf-8"))
        for task in tasks["tasks"]:
            if task["task_id"] == "T1":
                task["deps"] = ["T4"]
        (samples / "tasks.json").write_text(json.dumps(tasks, sort_keys=True, indent=2),
                                            encoding="utf-8")
    else:
        raise ValueError(f"unknown input fault: {fault}")


def write_preemption_samples(out_dir: str | Path) -> None:
    """Scenario engineered so a high-priority task blocks on in-flight low ones."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pools = {"pools": [
        {"pool_id": "pool-a", "gpu_count": 8, "memory_per_gpu_mib": 16384,
         "compute_units": 96, "quota_share": 100},
    ]}
    tasks = {"tasks": [
        {"task_id": "LOW", "gpu_need": 4, "memory_per_gpu_mib": 2048,
         "priority": 1, "duration_ms": 100, "shard_count": 3,
         "submit_time_ms": 0, "memory_budget_mib": 32768, "deps": []},
        {"task_id": "MID", "gpu_need": 4, "memory_per_gpu_mib": 2048,
         "priority": 2, "duration_ms": 100, "shard_count": 2,
         "submit_time_ms": 0, "memory_budget_mib": 16384, "deps": []},
        {"task_id": "HIGH", "gpu_need": 8, "memory_per_gpu_mib": 2048,
         "priority": 9, "duration_ms": 100, "shard_count": 2,
         "submit_time_ms": 60, "memory_budget_mib": 32768, "deps": []},
    ], "shards": []}
    params = {
        "global_gpu_quota": 8, "pool_concurrency_limit": 3,
        "scale_factor": 1000, "quota_share_cap": 100,
        "ledger_format_version": 1, "pool_version": "pool-v1",
        "task_version": "task-v2", "scale_version": "scale-v1",
    }
    for name, payload in (("pools.json", pools), ("tasks.json", tasks),
                          ("params.json", params)):
        (out / name).write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False),
            encoding="utf-8")
