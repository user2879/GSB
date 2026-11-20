"""租户 / 资源池 JSON 与请求流 NDJSON 的加载及强校验。

任何不合法都抛出 SpecError(E_SPEC) 整体拒绝，携带首个出错对象标识与字段；
注入场景下全局到达时刻倒退抛 OrderError(E_ORDER)，
优先级序缺失租户抛 PriorityError(E_PRIORITY)。
"""
from __future__ import annotations

import json
import os

from .errors import OrderError, PriorityError, SpecError
from .models import Pool, Request, Tenant


def _is_nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_pos_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_identifier(value) -> bool:
    return isinstance(value, str) and len(value.strip()) > 0


def load_spec(spec_path: str):
    """加载并强校验租户与资源池声明，返回 (tenants, pools)。"""
    if not os.path.exists(spec_path):
        raise SpecError(f"声明文件不存在: {spec_path}", ref_id=None, field="file")
    try:
        with open(spec_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SpecError(f"声明 JSON 无法解析: {exc}", ref_id=None, field="json")

    if not isinstance(data, dict) or "tenants" not in data or "pools" not in data:
        raise SpecError("声明顶层必须是含 tenants 与 pools 的对象", field="root")

    raw_tenants = data["tenants"]
    raw_pools = data["pools"]
    if not isinstance(raw_tenants, list) or not isinstance(raw_pools, list):
        raise SpecError("tenants 与 pools 必须是数组", field="root")

    tenants: list[Tenant] = []
    seen_tids: set[str] = set()
    for index, item in enumerate(raw_tenants):
        ref = item.get("tid") if isinstance(item, dict) else None
        if not isinstance(item, dict):
            raise SpecError("租户声明必须是对象", ref_id=ref, field="tenant")
        for field_name in ("tid", "quota", "concurrency_cap", "burst"):
            if field_name not in item:
                raise SpecError(f"租户缺少字段 {field_name}", ref_id=ref, field=field_name)
        if not _is_identifier(item["tid"]):
            raise SpecError("租户标识必须是非空字符串", ref_id=ref, field="tid")
        if item["tid"] in seen_tids:
            raise SpecError("租户标识重复", ref_id=item["tid"], field="tid")
        if not _is_pos_int(item["quota"]):
            raise SpecError("配额份额必须是正整数", ref_id=item["tid"], field="quota")
        if not _is_pos_int(item["concurrency_cap"]):
            raise SpecError("并发事务上限必须是正整数", ref_id=item["tid"], field="concurrency_cap")
        if not _is_nonneg_int(item["burst"]):
            raise SpecError("突发额度必须是非负整数", ref_id=item["tid"], field="burst")
        seen_tids.add(item["tid"])
        tenants.append(
            Tenant(item["tid"], item["quota"], item["concurrency_cap"], item["burst"])
        )

    if not tenants:
        raise SpecError("至少需要一个租户", field="tenants")

    pools: list[Pool] = []
    seen_pids: set[str] = set()
    for item in raw_pools:
        ref = item.get("pid") if isinstance(item, dict) else None
        if not isinstance(item, dict):
            raise SpecError("资源池声明必须是对象", ref_id=ref, field="pool")
        for field_name in ("pid", "capacity", "priority_order"):
            if field_name not in item:
                raise SpecError(f"资源池缺少字段 {field_name}", ref_id=ref, field=field_name)
        if not _is_identifier(item["pid"]):
            raise SpecError("资源池标识必须是非空字符串", ref_id=ref, field="pid")
        if item["pid"] in seen_pids:
            raise SpecError("资源池标识重复", ref_id=item["pid"], field="pid")
        if not _is_pos_int(item["capacity"]):
            raise SpecError("总容量单位必须是正整数", ref_id=item["pid"], field="capacity")
        order = item["priority_order"]
        if not isinstance(order, list) or not order:
            raise SpecError("租户优先级序必须是非空数组", ref_id=item["pid"], field="priority_order")
        local_seen: set[str] = set()
        for tid in order:
            if not _is_identifier(tid):
                raise SpecError("优先级序条目必须是非空字符串", ref_id=item["pid"], field="priority_order")
            if tid in local_seen:
                raise SpecError("优先级序内租户重复", ref_id=item["pid"], field="priority_order", detail={"tid": tid})
            if tid not in seen_tids:
                # 注入路径以 E_PRIORITY 报告缺失租户，常规输入按 E_SPEC 强校验拒绝。
                raise PriorityError(
                    "资源池优先级序引用了不存在的租户",
                    ref_id=item["pid"],
                    field="priority_order",
                    detail={"missing_tid": tid},
                )
            local_seen.add(tid)
        seen_pids.add(item["pid"])
        pools.append(Pool(item["pid"], item["capacity"], tuple(order)))

    if not pools:
        raise SpecError("至少需要一个资源池", field="pools")
    return tenants, pools


def load_requests(requests_path: str, tenants, pools, *, check_global_order: bool = False):
    """加载 NDJSON 请求流并强校验，返回按文件顺序的 Request 列表。"""
    if not os.path.exists(requests_path):
        raise SpecError(f"请求流文件不存在: {requests_path}", field="file")

    tenant_ids = {t.tid for t in tenants}
    pool_ids = {p.pid for p in pools}
    requests: list[Request] = []
    seen_rids: set[str] = set()
    last_arrival_by_tenant: dict[str, int] = {}
    global_last_arrival = -1

    with open(requests_path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SpecError(f"第 {line_no} 行 NDJSON 无法解析: {exc}", field="ndjson")
            if not isinstance(item, dict):
                raise SpecError("请求行必须是对象", field="request", detail={"line_no": line_no})
            ref = item.get("rid")
            for field_name in ("rid", "tid", "pid", "demand", "arrival", "priority"):
                if field_name not in item:
                    raise SpecError(f"请求缺少字段 {field_name}", ref_id=ref, field=field_name)
            if not _is_identifier(item["rid"]):
                raise SpecError("请求标识必须是非空字符串", ref_id=ref, field="rid")
            if item["rid"] in seen_rids:
                raise SpecError("请求标识重复", ref_id=item["rid"], field="rid")
            if item["tid"] not in tenant_ids:
                raise SpecError("请求引用了不存在的租户", ref_id=item["rid"], field="tid", detail={"tid": item["tid"]})
            if item["pid"] not in pool_ids:
                raise SpecError("请求引用了不存在的资源池", ref_id=item["rid"], field="pid", detail={"pid": item["pid"]})
            if not _is_pos_int(item["demand"]):
                raise SpecError("资源需求单位必须是正整数", ref_id=item["rid"], field="demand")
            if not _is_nonneg_int(item["arrival"]):
                raise SpecError("到达时刻必须是非负整数", ref_id=item["rid"], field="arrival")
            if not _is_nonneg_int(item["priority"]):
                raise SpecError("请求优先级必须是非负整数", ref_id=item["rid"], field="priority")

            arrival = item["arrival"]
            tid = item["tid"]
            if tid in last_arrival_by_tenant and arrival == last_arrival_by_tenant[tid]:
                raise SpecError(
                    "同一租户内到达时刻重复",
                    ref_id=item["rid"],
                    field="arrival",
                    detail={"tid": tid, "arrival": arrival},
                )
            if tid in last_arrival_by_tenant and arrival < last_arrival_by_tenant[tid]:
                raise SpecError(
                    "同一租户内到达时刻倒退",
                    ref_id=item["rid"],
                    field="arrival",
                    detail={"tid": tid, "arrival": arrival},
                )
            if check_global_order and arrival < global_last_arrival:
                raise OrderError(
                    "请求流到达时刻相对全局顺序倒退",
                    ref_id=item["rid"],
                    field="arrival",
                    detail={"arrival": arrival, "previous": global_last_arrival},
                )
            last_arrival_by_tenant[tid] = arrival
            global_last_arrival = max(global_last_arrival, arrival)
            seen_rids.add(item["rid"])
            requests.append(
                Request(item["rid"], tid, item["pid"], item["demand"], arrival, item["priority"])
            )

    if not requests:
        raise SpecError("请求流为空", field="requests")
    return requests
