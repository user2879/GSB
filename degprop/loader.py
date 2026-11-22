"""拓扑 JSON 与事件 NDJSON 的加载及强校验。

校验失败一律不返回部分结果；错误按文件中的稳定扫描顺序定位首个违规，
携带字段名与物理行号。依赖图成环单独以 E_CYCLE 报告，容量/配额乘积
越界以 E_CAPACITY 报告，其余结构违规以 E_TOPO 报告。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .errors import CapacityOverflowError, CycleError, TopologyError
from .fixedpoint import FP_MAX
from .models import (
    CallEvent,
    DegradeStrategy,
    DomainQuota,
    EdgeDef,
    LOOPBACK,
    PropagationMode,
    ServiceDef,
    SessionParams,
    Topology,
)


def _find_line(raw: str, needle: str) -> int:
    for idx, line in enumerate(raw.splitlines(), start=1):
        if needle in line:
            return idx
    return 1


def _require(cond: bool, message: str, field: str, line: int) -> None:
    if not cond:
        raise TopologyError(message, field=field, line=line)


def _is_pos_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def load_topology(path: str | Path) -> Topology:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TopologyError(
            f"拓扑 JSON 解析失败: {exc.msg}",
            field="<document>",
            line=exc.lineno,
        ) from exc

    _require(isinstance(doc, dict), "拓扑根节点必须是对象", "<document>", 1)
    version = str(doc.get("topology_version", "")).strip()
    _require(bool(version), "topology_version 缺失或为空", "topology_version",
             _find_line(raw, "topology_version"))

    params = _load_params(doc.get("session_params", {}), raw)
    factor = params.factor

    domains_raw = doc.get("domains", [])
    _require(isinstance(domains_raw, list) and domains_raw, "domains 必须是非空数组",
             "domains", _find_line(raw, "domains"))
    domains: dict[str, DomainQuota] = {}
    for item in domains_raw:
        line = _find_line(raw, f'"{item.get("domain_id", "")}"')
        did = item.get("domain_id")
        _require(isinstance(did, str) and did, "故障域标识必须是非空字符串",
                 "domain_id", line)
        _require(did not in domains, f"故障域标识重复: {did}", "domain_id", line)
        quota = item.get("in_flight_quota")
        _require(_is_pos_int(quota), "故障域并发配额必须是正整数",
                 "in_flight_quota", line)
        domains[did] = DomainQuota(did, quota)

    services_raw = doc.get("services", [])
    _require(isinstance(services_raw, list) and services_raw,
             "services 必须是非空数组", "services", _find_line(raw, "services"))
    services: dict[str, ServiceDef] = {}
    used_ports: dict[int, str] = {}
    for item in services_raw:
        sid = item.get("service_id")
        line = _find_line(raw, f'"service_id": "{sid}"')
        _require(isinstance(sid, str) and sid, "服务标识必须是非空字符串",
                 "service_id", line)
        _require(sid not in services, f"服务标识重复: {sid}", "service_id", line)

        host = item.get("host")
        _require(host == LOOPBACK, f"host 必须是字面量 {LOOPBACK}", "host", line)
        port = item.get("port")
        _require(_is_pos_int(port) and 1024 <= port <= 65535,
                 "port 必须是 1024..65535 的正整数", "port", line)
        _require(port not in used_ports, f"端口全局重复: {port}", "port", line)
        used_ports[port] = sid

        did = item.get("domain_id")
        _require(isinstance(did, str) and did in domains,
                 f"服务 {sid} 引用了未声明的故障域 {did}", "domain_id", line)

        strategy = item.get("degrade_strategy")
        _require(strategy in DegradeStrategy.values(),
                 f"未知降级策略: {strategy}", "degrade_strategy", line)

        cap = item.get("capacity_limit")
        _require(_is_pos_int(cap), "降级容量上限必须是正整数",
                 "capacity_limit", line)

        seed = item.get("seed", 0)
        _require(_nonneg_int(seed), "seed 必须是非负整数", "seed", line)

        # E_CAPACITY：容量上限与故障域配额的原始乘积越过 64 位定点声明域。
        quota = domains[did].in_flight_quota
        guarded_product = cap * quota * factor
        if guarded_product > FP_MAX:
            raise CapacityOverflowError(
                f"服务 {sid} 容量上限 {cap} × 域配额 {quota} × 缩放 {factor} "
                f"= {guarded_product} 越过定点声明域 {FP_MAX}",
                field="capacity_limit",
                line=line,
                detail={"product": guarded_product, "fp_max": FP_MAX},
            )

        services[sid] = ServiceDef(sid, did, host, port, strategy, cap, seed, line)

    edges_raw = doc.get("edges", [])
    _require(isinstance(edges_raw, list), "edges 必须是数组",
             "edges", _find_line(raw, "edges"))
    edges: list[EdgeDef] = []
    edge_keys: set[tuple[str, str]] = set()
    for item in edges_raw:
        up, down = item.get("upstream"), item.get("downstream")
        line = _find_line(raw, f'"upstream": "{up}"')
        _require(isinstance(up, str) and up in services,
                 f"边上游服务不存在: {up}", "upstream", line)
        _require(isinstance(down, str) and down in services,
                 f"边下游服务不存在: {down}", "downstream", line)
        _require(up != down, "依赖边不能自指", "downstream", line)
        _require((up, down) not in edge_keys, f"依赖边重复: {up}->{down}",
                 "upstream", line)
        edge_keys.add((up, down))

        mode = item.get("propagation_mode")
        _require(mode in {m.value for m in PropagationMode},
                 f"未知传播模式: {mode}", "propagation_mode", line)
        threshold = item.get("threshold")
        if mode == PropagationMode.THRESHOLD.value:
            _require(_nonneg_int(threshold) and threshold <= factor,
                     f"阈值传播边必须给出 0..{factor} 的定点阈值",
                     "threshold", line)
        else:
            _require(threshold is None, "强制传播边不得声明阈值",
                     "threshold", line)
        edges.append(EdgeDef(up, down, mode, threshold, line))

    order_index, adjacency, reverse = _build_graph(services, edges, raw)
    return Topology(version, services, domains, edges, params,
                    order_index, adjacency, reverse)


def _load_params(raw_params: dict, raw_text: str) -> SessionParams:
    if not isinstance(raw_params, dict):
        raise TopologyError("session_params 必须是对象",
                            field="session_params", line=_find_line(raw_text, "session_params"))
    allowed = set(SessionParams.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(raw_params) - allowed
    if unknown:
        key = sorted(unknown)[0]
        raise TopologyError(f"未知会话参数: {key}", field=key,
                            line=_find_line(raw_text, key))
    kwargs = dict(raw_params)
    exponent = kwargs.get("scale_exponent", 2)
    if not (isinstance(exponent, int) and 0 <= exponent <= 3):
        raise TopologyError("scale_exponent 必须是 0..3 的整数",
                            field="scale_exponent",
                            line=_find_line(raw_text, "scale_exponent"))
    positive_keys = ("global_concurrency", "window_size", "socket_timeout_ms",
                     "limited_threshold", "unavailable_threshold")
    for key in positive_keys:
        if key in kwargs and not _is_pos_int(kwargs[key]):
            raise TopologyError(f"{key} 必须是正整数", field=key,
                                line=_find_line(raw_text, key))
    if "max_retries" in kwargs and not _nonneg_int(kwargs["max_retries"]):
        raise TopologyError("max_retries 必须是非负整数", field="max_retries",
                            line=_find_line(raw_text, "max_retries"))
    return SessionParams(**kwargs)


def _build_graph(services: dict[str, ServiceDef], edges: list[EdgeDef],
                 raw: str) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    adjacency: dict[str, list[str]] = {sid: [] for sid in services}
    reverse: dict[str, list[str]] = {sid: [] for sid in services}
    indegree = {sid: 0 for sid in services}
    for edge in edges:
        adjacency[edge.upstream].append(edge.downstream)
        reverse[edge.downstream].append(edge.upstream)
        indegree[edge.downstream] += 1
    for sid in adjacency:
        adjacency[sid].sort()
        reverse[sid].sort()

    # Kahn 拓扑排序；候选集合按标识稳定排序，杜绝哈希/字典遍历顺序依赖。
    ready = sorted(sid for sid, deg in indegree.items() if deg == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    if len(order) != len(services):
        remaining = sorted(sid for sid in services if sid not in order)
        raise CycleError(
            f"服务依赖图存在环，无法完成拓扑排序，涉及: {remaining}",
            field="edges",
            line=_find_line(raw, "edges"),
            detail={"services_in_cycle": remaining},
        )
    return order, adjacency, reverse


def load_events(path: str | Path, topology: Topology) -> list[CallEvent]:
    path = Path(path)
    events: list[CallEvent] = []
    expected_seq = 1
    last_arrival = -1
    edge_pairs = {(e.upstream, e.downstream) for e in topology.edges}
    with path.open("r", encoding="utf-8") as handle:
        for physical_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TopologyError(
                    f"NDJSON 第 {physical_line} 行解析失败: {exc.msg}",
                    field="<line>", line=physical_line,
                ) from exc
            seq = item.get("seq")
            if not _is_pos_int(seq):
                raise TopologyError("事件序号必须是正整数", field="seq",
                                    line=physical_line)
            if seq != expected_seq:
                raise TopologyError(
                    f"事件序号必须从 1 连续唯一，期望 {expected_seq} 实际 {seq}",
                    field="seq", line=physical_line,
                )
            arrival = item.get("arrival_ms")
            if not (_nonneg_int(arrival) and arrival >= last_arrival):
                raise TopologyError(
                    "到达时刻必须是非负整数且单调不减",
                    field="arrival_ms", line=physical_line,
                )
            last_arrival = arrival
            up, down = item.get("upstream"), item.get("downstream")
            if up not in topology.services:
                raise TopologyError(f"事件上游服务不存在: {up}",
                                    field="upstream", line=physical_line)
            if down not in topology.services:
                raise TopologyError(f"事件下游服务不存在: {down}",
                                    field="downstream", line=physical_line)
            if (up, down) not in edge_pairs:
                raise TopologyError(
                    f"事件调用边 {up}->{down} 未在拓扑中声明",
                    field="upstream", line=physical_line,
                )
            client_key = item.get("client_key")
            if not isinstance(client_key, str) or not client_key:
                raise TopologyError("client_key 必须是非空字符串",
                                    field="client_key", line=physical_line)
            events.append(CallEvent(seq, arrival, up, down, client_key,
                                    physical_line))
            expected_seq += 1
    if not events:
        raise TopologyError("事件流为空", field="<document>", line=1)
    return events


def topology_digest(topology: Topology) -> str:
    """拓扑内容摘要：规范化 JSON 后取 SHA-256，作为快照版本识别依据。"""
    payload = {
        "topology_version": topology.topology_version,
        "services": [
            {
                "service_id": s.service_id,
                "domain_id": s.domain_id,
                "host": s.host,
                "port": s.port,
                "degrade_strategy": s.strategy,
                "capacity_limit": s.capacity_limit,
                "seed": s.seed,
            }
            for s in sorted(topology.services.values(), key=lambda s: s.service_id)
        ],
        "edges": [
            {
                "upstream": e.upstream,
                "downstream": e.downstream,
                "propagation_mode": e.mode,
                "threshold": e.threshold,
            }
            for e in sorted(topology.edges, key=lambda e: (e.upstream, e.downstream))
        ],
        "domains": [
            {"domain_id": d.domain_id, "in_flight_quota": d.in_flight_quota}
            for d in sorted(topology.domains.values(), key=lambda d: d.domain_id)
        ],
        "session_params": topology.params.__dict__,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
