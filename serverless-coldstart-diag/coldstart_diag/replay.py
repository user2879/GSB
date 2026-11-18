"""恢复后的一致性重放与不可复现报告（E_REPLAY）。"""

from .engine import Engine
from .errors import ReplayError
from .spec import load_manifest

# 需要逐项比对的确定性字段（有序，稳定定位首个不一致项）
COMPARE_FIELDS = (
    "calls",
    "burst_calls",
    "eff_concurrency",
    "required_instances_raw",
    "required_instances",
    "global_cap_instances",
    "memory_cap_instances",
    "available_instances",
    "warm_target",
    "warm_instances",
    "prewarm_starts",
    "cold_starts",
    "init_starts",
    "hot_calls",
    "cold_calls",
    "queued_calls",
    "total_wait_ms",
    "avg_wait_ms",
    "reserved_used",
    "reserved_target",
    "cold_ratio_q",
    "wait_ratio_q",
    "occupancy_q",
    "reserved_occupancy_q",
    "instance_set",
    "warm_set",
    "cold_set",
    "allocation",
    "arbitration_reasons",
    "error_block",
)


def replay_run(manifest_path, ledger, run_id, call_sequence, function_id, function_version):
    manifest = load_manifest(manifest_path)
    run = ledger.get_run(run_id)
    if run is None:
        raise ReplayError("账本中不存在该运行", path="run_id", value=run_id)

    # 先单独区分“配置/缩放因子版本不一致”导致的不可复现
    config_problems = []
    if run["scale"] != manifest["scale"]:
        config_problems.append(
            {"field": "scale", "ledger": run["scale"], "manifest": manifest["scale"]}
        )
    if run["declaration_version"] != manifest["declaration_version"]:
        config_problems.append(
            {
                "field": "declaration_version",
                "ledger": run["declaration_version"],
                "manifest": manifest["declaration_version"],
            }
        )
    if config_problems:
        raise ReplayError(
            "缩放因子或声明版本不一致，无法按原参数复现",
            path="config",
            details=config_problems,
        )

    recorded = {p["minute"]: p for p in ledger.load_minutes(run_id)}
    engine = Engine(manifest, function_id, function_version)
    for item in sorted(call_sequence, key=lambda x: x["minute"]):
        replayed = engine.run_minute(item).to_dict()
        minute = item["minute"]
        if minute not in recorded:
            raise ReplayError(
                "账本缺少该分钟记录，不可复现",
                path=f"minute[{minute}]",
                value="MISSING",
            )
        expected = recorded[minute]
        for field in COMPARE_FIELDS:
            if replayed.get(field) != expected.get(field):
                raise ReplayError(
                    "重放结果与账本不一致，定位到首个不可复现字段",
                    path=f"minute[{minute}].{field}",
                    value={"expected_from_ledger": expected.get(field),
                           "replayed": replayed.get(field)},
                )
    return {
        "run_id": run_id,
        "minutes_verified": len(recorded),
        "status": "REPRODUCIBLE",
        "fields_checked": list(COMPARE_FIELDS),
    }
