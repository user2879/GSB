"""内容声明、缓存策略与请求序列的加载及强校验。

任一违规整体拒绝（E_SPEC），报首个违规路径与字段，不做部分加载。
"""

import base64
import hashlib
import json
from dataclasses import dataclass, field

from .derive import COMBINE_RULES, recompute_derived, topo_order
from .errors import GraphError, SpecError

EVICTION_RULES = ("lru", "fifo")


@dataclass
class Content:
    path: str
    etag: str
    body: bytes
    ttl_ms: int
    derived_from: tuple = ()
    combine: str = None


@dataclass
class ContentSpec:
    spec_version: int
    contents: dict  # path -> Content（插入序即拓扑序）
    digest: str

    def to_jsonable(self):
        return {
            "spec_version": self.spec_version,
            "contents": [
                {
                    "path": c.path,
                    "etag": c.etag,
                    "body_base64": base64.b64encode(c.body).decode("ascii"),
                    "ttl_ms": c.ttl_ms,
                    "derived_from": list(c.derived_from),
                    "combine": c.combine,
                }
                for c in self.contents.values()
            ],
        }


@dataclass
class NodePolicy:
    node_id: str
    capacity: int
    eviction: str


@dataclass
class Policy:
    policy_version: int
    cache_key_fields: tuple
    nodes: tuple  # tuple[NodePolicy]，按 node_id 排序
    global_inflight: int
    per_node_inflight: int
    pull_timeout_ms: int
    max_retries: int
    lease_ttl_ms: int
    handshake_timeout_ms: int
    port_range: tuple


@dataclass
class RequestItem:
    node: str
    path: str
    tick_ms: int


@dataclass
class Batch:
    batch: int
    requests: list = field(default_factory=list)


def content_digest(contents):
    """内容集合摘要：路径、ETag、TTL、派生关系与字节内容的确定性散列。"""
    parts = []
    for path in sorted(contents):
        c = contents[path]
        body_sha = hashlib.sha256(c.body).hexdigest()
        parts.append(
            f"{path}|{c.etag}|{c.ttl_ms}|{','.join(c.derived_from)}|{c.combine or ''}|{body_sha}"
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _fail(path, field_name, message):
    raise SpecError(message, first_path=path, field=field_name)


def load_content_spec(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    entries = raw.get("contents")
    if not isinstance(entries, list) or not entries:
        _fail("<root>", "contents", "内容声明缺少非空 contents 数组")
    contents = {}
    for item in entries:
        p = item.get("path")
        if not isinstance(p, str) or not p:
            _fail(str(p), "path", "路径缺失或非字符串")
        if p in contents:
            _fail(p, "path", "路径重复")
        ttl = item.get("ttl_ms")
        if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl <= 0:
            _fail(p, "ttl_ms", "TTL 必须为正整数毫秒")
        derived = item.get("derived_from") or []
        if not isinstance(derived, list) or any(not isinstance(d, str) for d in derived):
            _fail(p, "derived_from", "派生引用必须为字符串数组")
        if len(set(derived)) != len(derived):
            _fail(p, "derived_from", "派生引用重复")
        combine = item.get("combine")
        if derived:
            if combine not in COMBINE_RULES:
                _fail(p, "combine", f"未知组合规则 {combine!r}")
        body_b64 = item.get("body_base64")
        etag = item.get("etag")
        if not derived:
            if not isinstance(etag, str) or not etag:
                _fail(p, "etag", "非派生路径必须声明非空 ETag")
            if not isinstance(body_b64, str):
                _fail(p, "body_base64", "非派生路径必须声明 body_base64")
            try:
                body = base64.b64decode(body_b64, validate=True)
            except Exception:
                _fail(p, "body_base64", "body_base64 无法解码")
        else:
            body = b""
            etag = ""
        contents[p] = Content(
            path=p,
            etag=etag,
            body=body,
            ttl_ms=ttl,
            derived_from=tuple(derived),
            combine=combine,
        )
    for p, c in contents.items():
        for parent in c.derived_from:
            if parent not in contents:
                _fail(p, "derived_from", f"派生引用不存在: {parent}")
    try:
        order = topo_order(contents)
    except GraphError as exc:
        _fail(exc.details.get("first_path", "?"), "derived_from", "派生图存在环")
    contents = {p: contents[p] for p in order}
    recompute_derived(contents)
    return ContentSpec(
        spec_version=int(raw.get("spec_version", 1)),
        contents=contents,
        digest=content_digest(contents),
    )


def load_policy(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    key_fields = raw.get("cache_key_fields")
    if (
        not isinstance(key_fields, list)
        or not key_fields
        or any(not isinstance(f, str) for f in key_fields)
    ):
        _fail("<policy>", "cache_key_fields", "缓存键组成字段必须为非空字符串数组")
    if len(set(key_fields)) != len(key_fields):
        _fail("<policy>", "cache_key_fields", "缓存键组成字段存在重复")
    nodes = []
    seen = set()
    for item in raw.get("nodes") or []:
        nid = item.get("node_id")
        if not isinstance(nid, str) or not nid:
            _fail(str(nid), "node_id", "节点 ID 缺失")
        if nid in seen:
            _fail(nid, "node_id", "节点 ID 重复")
        seen.add(nid)
        cap = item.get("capacity")
        if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
            _fail(nid, "capacity", "节点容量必须为正整数")
        rule = item.get("eviction")
        if rule not in EVICTION_RULES:
            _fail(nid, "eviction", f"未知逐出规则 {rule!r}")
        nodes.append(NodePolicy(node_id=nid, capacity=cap, eviction=rule))
    if len(nodes) < 3:
        _fail("<policy>", "nodes", "至少需要三个边缘节点")
    nodes.sort(key=lambda n: n.node_id)
    sched = raw.get("scheduler") or {}
    lease = raw.get("lease") or {}
    port_range = raw.get("port_range") or [49152, 49252]
    return Policy(
        policy_version=int(raw.get("policy_version", 1)),
        cache_key_fields=tuple(key_fields),
        nodes=tuple(nodes),
        global_inflight=int(sched.get("global_inflight", 2)),
        per_node_inflight=int(sched.get("per_node_inflight", 1)),
        pull_timeout_ms=int(sched.get("pull_timeout_ms", 2000)),
        max_retries=int(sched.get("max_retries", 2)),
        lease_ttl_ms=int(lease.get("ttl_ms", 1500)),
        handshake_timeout_ms=int(raw.get("handshake_timeout_ms", 3000)),
        port_range=(int(port_range[0]), int(port_range[1])),
    )


def load_requests(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    batches = []
    for b in raw.get("batches") or []:
        items = []
        for r in b.get("requests") or []:
            tick = r.get("tick_ms")
            if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
                _fail(str(r.get("path")), "tick_ms", "请求 tick_ms 必须为非负整数")
            items.append(
                RequestItem(node=r["node"], path=r["path"], tick_ms=tick)
            )
        batches.append(Batch(batch=int(b.get("batch", 0)), requests=items))
    if not batches:
        _fail("<requests>", "batches", "请求序列缺少 batches")
    return batches
