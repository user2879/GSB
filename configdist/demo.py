from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evolution import evolve_spec_data
from .planner import compile_plan
from .service import (
    execute_run,
    preview_ledger,
    replay_restored_snapshot,
    verify_run,
)
from .validation import validate_spec


def _reverse_input_order(raw: dict[str, Any]) -> dict[str, Any]:
    reversed_raw = json.loads(json.dumps(raw))
    reversed_raw["clusters"].reverse()
    reversed_raw["revisions"].reverse()
    reversed_raw["tasks"].reverse()
    for task in reversed_raw["tasks"]:
        task["config_keys"].reverse()
        task["target_clusters"].reverse()
    return reversed_raw


def run_demo(out_path: Path) -> dict[str, Any]:
    output: list[str] = []

    def emit(text: str = "") -> None:
        output.append(text)
        print(text)

    root = Path(__file__).resolve().parents[1]
    spec_path = root / "sample" / "spec.json"
    demo_dir = root / "artifacts" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = demo_dir / "ledger.sqlite3"
    snapshot_path = demo_dir / "session.snapshot.json"
    for path in [
        *demo_dir.glob("ledger.sqlite3*"),
        *demo_dir.glob("session.snapshot.json*"),
    ]:
        path.unlink(missing_ok=True)

    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = validate_spec(raw)
    plan = compile_plan(spec)
    permuted_spec = validate_spec(_reverse_input_order(raw))
    permuted_plan = compile_plan(permuted_spec)
    if permuted_plan["plan_digest"] != plan["plan_digest"]:
        raise AssertionError("permuted input produced a different plan")
    if [item["shard_id"] for item in permuted_plan["shards"]] != [item["shard_id"] for item in plan["shards"]]:
        raise AssertionError("permuted input produced a different shard sequence")
    if [item["shard_id"] for item in plan["shards"]] != sorted(item["shard_id"] for item in plan["shards"]):
        raise AssertionError("shard sequence is not stable lexical order")

    emit("=== 1. 清单校验与确定性分发计划 ===")
    emit(f"plan_digest={plan['plan_digest']}")
    emit(f"shards={len(plan['shards'])} permuted_digest={permuted_plan['plan_digest']}")
    emit("shard_ids=" + ",".join(item["shard_id"] for item in plan["shards"]))

    baseline = execute_run(spec, plan, ledger_path, snapshot_path=snapshot_path)
    emit("")
    emit("=== 2. 三重预算并行调度 ===")
    emit(f"run_id={baseline['run_id']} counts={baseline['counts']}")
    emit(f"peak={baseline['peak']}")
    emit(f"limits={baseline['limits']}")
    peak = baseline["peak"]
    limits = baseline["limits"]
    if peak["resource_units"] > limits["global_resource_units"]:
        raise AssertionError("global budget exceeded")
    for cluster_id, value in peak["cluster_concurrency"].items():
        if value > limits["cluster_quotas"][cluster_id]:
            raise AssertionError(f"cluster quota exceeded: {cluster_id}")
    for task_id, value in peak["task_concurrency"].items():
        if value > limits["task_parallel_limits"][task_id]:
            raise AssertionError(f"task limit exceeded: {task_id}")
    wait_samples = [
        trace for trace in baseline["trace"] if trace["outcome"] == "completed"
    ]
    emit(
        "sample_trace="
        + json.dumps(wait_samples[0], ensure_ascii=False, sort_keys=True)
    )

    emit("")
    emit("=== 3. 账本缓存、权威重算与快照重放 ===")
    preview = preview_ledger(ledger_path)
    emit(f"preview_run={preview['run_id']} cursor={preview['completed_cursor']} stale={preview['stale_shards']}")
    verify = verify_run(ledger_path, use_nodes=True)
    emit(f"authoritative_verify={json.dumps(verify, ensure_ascii=False, sort_keys=True)}")
    replay = replay_restored_snapshot(snapshot_path)
    emit(f"snapshot_replay={json.dumps(replay, ensure_ascii=False, sort_keys=True)}")

    evolved_raw = evolve_spec_data(
        raw,
        config_updates={
            "edge-a": {"mode": "canary", "region": "east", "policy": "strict"},
        },
        target_updates={"deploy-core": ["edge-a", "edge-c"]},
    )
    evolved_spec = validate_spec(evolved_raw)
    evolved_plan = compile_plan(evolved_spec)
    evolved = execute_run(evolved_spec, evolved_plan, ledger_path)
    emit("")
    emit("=== 4. 修订演进与整体替换 ===")
    emit(f"new_plan_digest={evolved['plan_digest']} counts={evolved['counts']}")
    emit(f"stale_old_shards={evolved['stale_shards_after']}")
    emit(f"replacement_reasons={evolved['replacement']['reasons']}")
    emit(f"changed_shards={len(evolved['replacement']['changed_shards'])}")
    emit(f"affected_shards={len(evolved['replacement']['affected_shards'])}")
    emit("affected_ids=" + ",".join(evolved["replacement"]["affected_shards"]))
    current_preview = preview_ledger(ledger_path)
    if current_preview["stale_shards"] != len(baseline["trace"]):
        raise AssertionError("old generation was not entirely stale")
    for group in current_preview["groups"]:
        progress = group["progress"]
        if progress["completed"] != progress["total"]:
            raise AssertionError(f"current generation is incomplete: {group}")
    emit(
        f"current_preview_run={current_preview['run_id']} "
        f"current_cursor={current_preview['completed_cursor']} "
        f"stale={current_preview['stale_shards']} groups={len(current_preview['groups'])}"
    )
    evolved_verify = verify_run(ledger_path, use_nodes=True)
    emit(f"evolved_authoritative_verify={json.dumps(evolved_verify, ensure_ascii=False, sort_keys=True)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    emit("")
    emit(f"acceptance_output={out_path}")
    return {
        "ok": True,
        "plan_digest": plan["plan_digest"],
        "permuted_plan_digest": permuted_plan["plan_digest"],
        "baseline": baseline["counts"],
        "stale_shards": evolved["stale_shards_after"],
        "output": str(out_path),
    }
