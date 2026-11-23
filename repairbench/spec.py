"""副本组规格加载与强校验（需求 1）。

JSON 声明格式：
{
  "group_id": "alpha",
  "session": {
    "concurrency_limit": 3,          # 全局修复并发上限，非负整数
    "global_budget_bytes": 200000,   # 全局修复字节预算，非负整数
    "segment_bytes": 4096,           # 批次字节段长度，须为受支持整数
    "scale_exponent": 3,             # 定点缩放指数 0..6
    "transfer_window_ms": 40,        # 批次目标传输窗口（正整数，可选，默认 40）
    "time_scale": 0.05               # 模拟传输时间缩放（正数，可选，默认 1.0）
  },
  "nodes": [
    {"node_id": "n1", "host": "127.0.0.1", "port": 24101,
     "bandwidth_kbps": 64, "priority_weight": 5}
  ],
  "blocks": [
    {"block_id": "b1", "bytes": 10000, "seed": "alpha-b1",
     "node_id": "n1", "mismatch_permille": 800}
  ]
}

校验规则（任一违反即以 E_SPEC 拒绝并指明首个非法条目与字段）：
  - 节点标识、块标识全局唯一；
  - host 必须为回环字面量 "127.0.0.1"，端口为 1024..65535 整数且全局不重复；
  - 带宽配额、优先级权重为非负整数；
  - 块字节数为正整数；seed 为非空字符串；node_id 须引用已声明节点；
  - mismatch_permille 为 0..1000 整数或缺省（缺省 = 证据不足）；
  - segment_bytes 属于受支持集合且不大于最大块字节数（须能覆盖块长度）；
  - concurrency_limit、global_budget_bytes 为非负整数；
  - scale_exponent 为 0..6 整数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .errors import E_SPEC, WorkbenchError
from .fixedpoint import SCALE_EXP_MAX, SCALE_EXP_MIN

SUPPORTED_SEGMENT_BYTES = tuple(1024 << k for k in range(0, 11))  # 1KB..1MB 的 2 幂
LOOPBACK_LITERAL = "127.0.0.1"
PORT_MIN = 1024
PORT_MAX = 65535
DEFAULT_TRANSFER_WINDOW_MS = 40


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fail(entry: str, fld: str, message: str) -> None:
    raise WorkbenchError(E_SPEC, message, entry=entry, field=fld)


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    bandwidth_kbps: int
    priority_weight: int


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    byte_count: int
    seed: str
    node_id: str
    mismatch_permille: int | None  # None = 证据不足


@dataclass(frozen=True)
class SessionParams:
    concurrency_limit: int
    global_budget_bytes: int
    segment_bytes: int
    scale_exponent: int
    transfer_window_ms: int = DEFAULT_TRANSFER_WINDOW_MS
    time_scale: float = 1.0


@dataclass
class GroupSpec:
    group_id: str
    params: SessionParams
    nodes: list = field(default_factory=list)   # list[NodeSpec]
    blocks: list = field(default_factory=list)  # list[BlockSpec]

    def node(self, node_id: str) -> NodeSpec:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(node_id)

    def content_fingerprint(self) -> str:
        """规格规范化指纹（用于账本内容摘要与快照比对）。"""
        import hashlib

        canonical = {
            "group_id": self.group_id,
            "params": {
                "concurrency_limit": self.params.concurrency_limit,
                "global_budget_bytes": self.params.global_budget_bytes,
                "segment_bytes": self.params.segment_bytes,
                "scale_exponent": self.params.scale_exponent,
                "transfer_window_ms": self.params.transfer_window_ms,
            },
            "nodes": [
                [n.node_id, n.host, n.port, n.bandwidth_kbps, n.priority_weight]
                for n in sorted(self.nodes, key=lambda n: n.node_id)
            ],
            "blocks": [
                [b.block_id, b.byte_count, b.seed, b.node_id, b.mismatch_permille]
                for b in sorted(self.blocks, key=lambda b: b.block_id)
            ],
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _validate_session(raw: dict, entry: str) -> SessionParams:
    if not isinstance(raw, dict):
        _fail(entry, "session", "session 必须为对象")
    cl = raw.get("concurrency_limit")
    if not _is_int(cl) or cl < 0:
        _fail(entry, "session.concurrency_limit", "全局修复并发上限须为非负整数")
    gb = raw.get("global_budget_bytes")
    if not _is_int(gb) or gb < 0:
        _fail(entry, "session.global_budget_bytes", "全局修复字节预算须为非负整数")
    seg = raw.get("segment_bytes")
    if not _is_int(seg) or seg not in SUPPORTED_SEGMENT_BYTES:
        _fail(
            entry,
            "session.segment_bytes",
            f"批次字节段长度须为受支持整数 {SUPPORTED_SEGMENT_BYTES}",
        )
    se = raw.get("scale_exponent")
    if not _is_int(se) or not (SCALE_EXP_MIN <= se <= SCALE_EXP_MAX):
        _fail(entry, "session.scale_exponent", "缩放指数须为 0..6 整数")
    tw = raw.get("transfer_window_ms", DEFAULT_TRANSFER_WINDOW_MS)
    if not _is_int(tw) or tw <= 0:
        _fail(entry, "session.transfer_window_ms", "传输窗口须为正整数毫秒")
    ts = raw.get("time_scale", 1.0)
    if not isinstance(ts, (int, float)) or isinstance(ts, bool) or ts <= 0:
        _fail(entry, "session.time_scale", "时间缩放须为正数")
    return SessionParams(cl, gb, seg, se, tw, float(ts))


def _validate_nodes(raw_nodes, entry: str, used_ports: set) -> list:
    if not isinstance(raw_nodes, list) or not raw_nodes:
        _fail(entry, "nodes", "节点集合须为非空数组")
    seen_ids = set()
    nodes = []
    for idx, raw in enumerate(raw_nodes):
        eid = f"{entry}.nodes[{idx}]"
        if not isinstance(raw, dict):
            _fail(eid, "node", "节点条目须为对象")
        nid = raw.get("node_id")
        if not isinstance(nid, str) or not nid:
            _fail(eid, "node_id", "节点标识须为非空字符串")
        if nid in seen_ids:
            _fail(eid, "node_id", f"节点标识重复: {nid}")
        seen_ids.add(nid)
        host = raw.get("host")
        if host != LOOPBACK_LITERAL:
            _fail(eid, "host", f"端口主机须为回环字面量 {LOOPBACK_LITERAL}")
        port = raw.get("port")
        if not _is_int(port) or not (PORT_MIN <= port <= PORT_MAX):
            _fail(eid, "port", f"端口须为 {PORT_MIN}..{PORT_MAX} 整数")
        if port in used_ports:
            _fail(eid, "port", f"端口全局重复: {port}")
        used_ports.add(port)
        bw = raw.get("bandwidth_kbps")
        if not _is_int(bw) or bw < 0:
            _fail(eid, "bandwidth_kbps", "带宽配额须为非负整数 KB/s")
        pw = raw.get("priority_weight")
        if not _is_int(pw) or pw < 0:
            _fail(eid, "priority_weight", "优先级权重须为非负整数")
        nodes.append(NodeSpec(nid, host, port, bw, pw))
    return nodes


def _validate_blocks(raw_blocks, entry: str, node_ids: set) -> list:
    if not isinstance(raw_blocks, list) or not raw_blocks:
        _fail(entry, "blocks", "块集合须为非空数组")
    seen_ids = set()
    blocks = []
    for idx, raw in enumerate(raw_blocks):
        eid = f"{entry}.blocks[{idx}]"
        if not isinstance(raw, dict):
            _fail(eid, "block", "块条目须为对象")
        bid = raw.get("block_id")
        if not isinstance(bid, str) or not bid:
            _fail(eid, "block_id", "块标识须为非空字符串")
        if bid in seen_ids:
            _fail(eid, "block_id", f"块标识重复: {bid}")
        seen_ids.add(bid)
        bc = raw.get("bytes")
        if not _is_int(bc) or bc <= 0:
            _fail(eid, "bytes", "块字节数须为正整数")
        seed = raw.get("seed")
        if not isinstance(seed, str) or not seed:
            _fail(eid, "seed", "分片内容种子须为非空字符串")
        nid = raw.get("node_id")
        if nid not in node_ids:
            _fail(eid, "node_id", f"块引用了未声明节点: {nid}")
        mp = raw.get("mismatch_permille")
        if mp is not None and (not _is_int(mp) or not (0 <= mp <= 1000)):
            _fail(eid, "mismatch_permille", "失配程度须为 0..1000 整数或缺省")
        blocks.append(BlockSpec(bid, bc, seed, nid, mp))
    return blocks


def validate_group(raw: dict, entry: str = "group", used_ports: set | None = None) -> GroupSpec:
    """校验单个副本组声明；used_ports 跨组共享以保证端口全局不重复。"""
    if not isinstance(raw, dict):
        _fail(entry, "group", "副本组声明须为对象")
    gid = raw.get("group_id")
    if not isinstance(gid, str) or not gid:
        _fail(entry, "group_id", "副本组标识须为非空字符串")
    params = _validate_session(raw.get("session"), entry)
    ports = used_ports if used_ports is not None else set()
    nodes = _validate_nodes(raw.get("nodes"), entry, ports)
    blocks = _validate_blocks(raw.get("blocks"), entry, {n.node_id for n in nodes})
    # 批次段长度须能覆盖块长度：段长不超过最大块字节数（保证至少一个块可被单段覆盖）。
    max_block = max(b.byte_count for b in blocks)
    if params.segment_bytes > max_block and params.segment_bytes > SUPPORTED_SEGMENT_BYTES[0]:
        _fail(
            entry,
            "session.segment_bytes",
            f"批次段长度 {params.segment_bytes} 无法有效覆盖最大块长度 {max_block}",
        )
    return GroupSpec(gid, params, nodes, blocks)


def load_spec(path: str) -> list:
    """加载规格文件：单个组对象或 {"groups": [...]}，端口全局唯一。"""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "groups" in raw:
        raw_groups = raw["groups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            _fail("spec", "groups", "groups 须为非空数组")
    elif isinstance(raw, dict):
        raw_groups = [raw]
    else:
        _fail("spec", "root", "规格根须为对象")
    used_ports: set = set()
    groups = []
    seen_gid = set()
    for idx, raw_group in enumerate(raw_groups):
        entry = f"groups[{idx}]" if len(raw_groups) > 1 else "group"
        group = validate_group(raw_group, entry=entry, used_ports=used_ports)
        if group.group_id in seen_gid:
            _fail(entry, "group_id", f"副本组标识重复: {group.group_id}")
        seen_gid.add(group.group_id)
        groups.append(group)
    return groups
