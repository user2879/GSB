"""构建报告的确定性输出与预算审计。

报告分两层，兼顾「含等待/耗时」与「逐字节可复现」：
- ``report.full.json``：完整报告，含单调时钟等待时长、执行耗时等测量值；
- ``report.canonical.json``：确定性子集，剔除全部墙钟/单调时钟测量字段，
  同输入重复构建逐字节一致；并对其计算 ``canonical_digest``。

序列化顺序固定为：阶段按拓扑序、阶段内按指令 seq；所有 dict 键排序输出。
审计不通过时以 E_AUDIT 报告首个偏差指令（阶段 + seq + 字段 + 依据）。
"""
from __future__ import annotations

import json
import os

from . import digest
from .errors import (
    E_AUDIT, E_COMMAND, E_TIMEOUT, LayerBenchError, OK,
    STATUS_COMMITTED, STATUS_SKIPPED,
)

CANONICAL_FILENAME = "report.canonical.json"
FULL_FILENAME = "report.full.json"

# 确定性子集允许出现的字段（显式白名单，测量字段一律排除）
# cache_hit 属于“本次构建运行态”（冷/热构建不同），不进入确定性子集；
# 层摘要、结果摘要、退出码、stdout 摘要等才是跨构建必须逐字节一致的事实。
_NODE_FIELDS = (
    "node_uid", "stage", "seq", "id", "type", "params", "status",
    "result_code", "precursor", "layer_digest",
    "result_digest", "exit_code", "stdout_digest", "skipped_trigger",
)


def build_report(manifest, records: dict, scheduler, ledger, build_id: str,
                 cache_fingerprint: str) -> dict:
    nodes_out = []
    for stage_name in manifest.topo_order:
        for ins in manifest.stage_by_name[stage_name].instructions:
            uid = f"{stage_name}#{ins.seq}:{ins.id}"
            nodes_out.append(records[uid].as_dict())

    stage_usage = {}
    for stage_name in manifest.topo_order:
        stage_usage[stage_name] = {
            "quota": manifest.stage_by_name[stage_name].quota,
            "peak_running": scheduler.peak_stage[stage_name],
        }

    codes = sorted({r["result_code"] for r in nodes_out})
    build_code = OK
    for bad in (E_AUDIT, E_COMMAND, E_TIMEOUT):
        if bad in codes:
            build_code = bad
            break

    report = {
        "report_version": 1,
        "build_id": build_id,
        "manifest_version": manifest.version,
        "global_slots": manifest.global_slots,
        "cache_capacity": manifest.cache_capacity,
        "cache_fingerprint": cache_fingerprint,
        "build_code": build_code,
        "budget": {
            "peak_global_running": scheduler.peak_global,
            "stage_usage": stage_usage,
        },
        "evictions": [dict(e) for e in scheduler.cache.evictions],
        "nodes": nodes_out,
    }
    audit = audit_report(manifest, report, ledger, build_id)
    report["audit"] = audit
    if audit["ok"] is False:
        report["build_code"] = E_AUDIT
    return report


def canonical_payload(report: dict) -> dict:
    nodes = []
    for node in report["nodes"]:
        nodes.append({key: node[key] for key in _NODE_FIELDS})
    return {
        "report_version": report["report_version"],
        "manifest_version": report["manifest_version"],
        "global_slots": report["global_slots"],
        "cache_capacity": report["cache_capacity"],
        "build_code": report["build_code"],
        "budget": {
            "global_slots": report["global_slots"],
            "stage_quota": {
                name: usage["quota"]
                for name, usage in report["budget"]["stage_usage"].items()
            },
        },
        "nodes": nodes,
        "audit_ok": report["audit"]["ok"],
    }


def write_reports(directory: str, report: dict) -> tuple[str, str, str]:
    os.makedirs(directory, exist_ok=True)
    canonical = canonical_payload(report)
    canonical_text = digest.canonical_json(canonical) + "\n"
    canonical_path = os.path.join(directory, CANONICAL_FILENAME)
    full_path = os.path.join(directory, FULL_FILENAME)
    with open(canonical_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_text)
    full_text = json.dumps(report, sort_keys=True, ensure_ascii=False,
                           indent=2) + "\n"
    with open(full_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(full_text)
    return canonical_path, full_path, digest.sha256_hex(canonical_text)


def audit_report(manifest, report: dict, ledger, build_id: str) -> dict:
    """返回审计结果；首个偏差以 E_AUDIT 定位到阶段/seq/字段/依据。"""
    checks: list[dict] = []

    def fail(node, field: str, basis: str) -> dict:
        return {
            "ok": False,
            "code": E_AUDIT,
            "first_deviation": {
                "stage": node["stage"], "seq": node["seq"],
                "node_uid": node["node_uid"], "field": field, "basis": basis,
            },
            "checks": checks,
        }

    node_by_uid = {n["node_uid"]: n for n in report["nodes"]}

    all_events = ledger.get_events(build_id)
    # 按事件 detail.phase 精确划分执行段：续建只审计 resume 段，
    # 普通构建全部 slot 事件均为 initial 段。崩溃前孤儿事件属于 initial，
    # 不会污染续建段审计。
    def _phase(event):
        try:
            return json.loads(event["detail_json"] or "{}").get("phase", "initial")
        except (ValueError, TypeError):
            return "initial"

    is_resume = any(e["kind"] == "resume_start" for e in all_events)
    wanted_phase = "resume" if is_resume else "initial"
    slot_events = [e for e in all_events
                   if e["kind"] in ("slot_acquired", "slot_released")
                   and _phase(e) == wanted_phase]

    # 1) 命中层不得真实执行：cache_hit=True 的 run 在本执行段内不得有
    #    slot_acquired（崩溃前历史已被 resume_threshold 排除）
    acquired = {e["node_uid"] for e in slot_events
                if e["kind"] == "slot_acquired"}
    for node in report["nodes"]:
        if node["cache_hit"]:
            if node["type"] == "run" and node["node_uid"] in acquired:
                checks.append({"name": "hit_not_executed", "ok": False})
                return fail(node, "cache_hit",
                            "缓存命中的 run 指令却出现 slot_acquired（被真实执行）")
            checks.append({"name": "hit_not_executed", "ok": True})

    # 2) 失败传播与跳过集合自洽
    for node in report["nodes"]:
        if node["status"] == STATUS_SKIPPED:
            trigger_uid = node.get("skipped_trigger")
            if not trigger_uid or trigger_uid not in node_by_uid:
                return fail(node, "skipped_trigger", "跳过指令缺少有效上游触发者")
            trigger = node_by_uid[trigger_uid]
            if trigger["result_code"] not in (E_COMMAND, E_TIMEOUT):
                return fail(node, "status",
                            f"跳过触发者 {trigger_uid} 并非失败状态")
            if node["layer_digest"] is not None or node["cache_hit"]:
                return fail(node, "layer_digest", "跳过指令不应产生层摘要或缓存命中")
    # 3) 预算：从事件时间线独立重建占用，复核任一时刻不超限
    timeline_peak_global, timeline_peak_stage, violation = _rebuild_occupancy(
        ledger, build_id, manifest, slot_events
    )
    if violation is not None:
        node = node_by_uid.get(violation["node_uid"], report["nodes"][0])
        return fail(node, "budget", violation["basis"])
    if timeline_peak_global != report["budget"]["peak_global_running"]:
        return fail(report["nodes"][0], "peak_global_running",
                    f"事件时间线峰值 {timeline_peak_global} 与调度器记录 "
                    f"{report['budget']['peak_global_running']} 不一致")
    for stage_name, usage in report["budget"]["stage_usage"].items():
        if timeline_peak_stage.get(stage_name, 0) != usage["peak_running"]:
            return fail(report["nodes"][0], "stage_peak_running",
                        f"阶段 {stage_name} 时间线峰值与记录不一致")
        if usage["peak_running"] > usage["quota"]:
            return fail(report["nodes"][0], "quota",
                        f"阶段 {stage_name} 峰值 {usage['peak_running']} 超过配额")
    if report["budget"]["peak_global_running"] > manifest.global_slots:
        return fail(report["nodes"][0], "global_slots", "全局峰值超过槽位上限")

    # 4) 成功/失败提交指令都必须带层摘要
    for node in report["nodes"]:
        if node["status"] == STATUS_COMMITTED and not node["layer_digest"]:
            return fail(node, "layer_digest", "已提交指令缺少层摘要")

    return {"ok": True, "code": OK, "first_deviation": None, "checks": checks}


def _rebuild_occupancy(ledger, build_id: str, manifest, events=None):
    """独立从 slot_acquired/released 事件重建并发占用时间线。

    events 由调用方按执行段过滤（续建时仅 resume_start 之后），
    确保崩溃前的孤儿事件不参与本次审计。
    """
    if events is None:
        events = [e for e in ledger.get_events(build_id)
                  if e["kind"] in ("slot_acquired", "slot_released")]
    current_global = 0
    peak_global = 0
    current_stage = {s.name: 0 for s in manifest.stages}
    peak_stage = {s.name: 0 for s in manifest.stages}
    violation = None
    for event in events:
        node_uid = event["node_uid"]
        stage = node_uid.split("#", 1)[0]
        if event["kind"] == "slot_acquired":
            current_global += 1
            current_stage[stage] += 1
            peak_global = max(peak_global, current_global)
            peak_stage[stage] = max(peak_stage[stage], current_stage[stage])
            if current_global > manifest.global_slots:
                violation = {"node_uid": node_uid,
                             "basis": "全局运行数超过 global_slots"}
            if current_stage[stage] > manifest.stage_by_name[stage].quota:
                violation = {"node_uid": node_uid,
                             "basis": f"阶段 {stage} 运行数超过 quota"}
        else:
            current_global = max(0, current_global - 1)
            current_stage[stage] = max(0, current_stage[stage] - 1)
    return peak_global, peak_stage, violation
