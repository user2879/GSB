"""节点声明、内容清单与事件流的本地 JSON 加载及强校验。

任何不合法输入抛 SpecError(E_SPEC)，location 指明首个非法字段路径或事件序号。
校验通过后返回不可变风格的普通 dict/列表结构，供后续模块只读使用。
"""

import json
from pathlib import Path

from .errors import SpecError

EVENT_TYPES = {"write", "read", "renew", "release", "cancel", "quota_change"}
WRITE_TTL_FIELD = "ttl_ms"


def _is_pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonneg_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _err(message, location):
    raise SpecError(message, location)


def load_spec(path) -> dict:
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        _err(f"无法读取声明文件: {exc}", str(p))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _err(f"JSON 解析失败: {exc.msg}", f"{p}:line {exc.lineno}")
    if not isinstance(data, dict):
        _err("声明根必须是 JSON 对象", "$")
    return validate_spec(data)


def validate_spec(data: dict) -> dict:
    if not isinstance(data, dict):
        _err("声明根必须是 JSON 对象", "$")

    for field in ("session", "nodes", "contents", "events"):
        if field not in data:
            _err(f"缺少必填字段 {field}", f"$.{field}")

    session = _validate_session(data["session"])
    nodes = _validate_nodes(data["nodes"])
    contents = _validate_contents(data["contents"])
    events = _validate_events(data["events"], nodes, contents)

    return {
        "format_version": 1,
        "session": session,
        "nodes": nodes,
        "contents": contents,
        "events": events,
    }


def _validate_session(session):
    if not isinstance(session, dict):
        _err("session 必须是对象", "$.session")
    out = {}
    required = {
        "session_id": str,
        "global_lease_limit": _is_pos_int,
        "global_byte_budget": _is_pos_int,
        "session_scale": _is_pos_int,
    }
    for key, check in required.items():
        if key not in session:
            _err(f"session 缺少必填字段 {key}", f"$.session.{key}")
        value = session[key]
        if check is str:
            if not isinstance(value, str) or not value:
                _err(f"{key} 必须是非空字符串", f"$.session.{key}")
        elif not check(value):
            _err(f"{key} 必须是正整数", f"$.session.{key}")
        out[key] = value
    return out


def _validate_nodes(nodes):
    if not isinstance(nodes, list) or not nodes:
        _err("nodes 必须是非空数组", "$.nodes")
    out = []
    seen_ids = set()
    seen_ports = set()
    for i, node in enumerate(nodes):
        base = f"$.nodes[{i}]"
        if not isinstance(node, dict):
            _err("节点声明必须是对象", base)
        for field in ("node_id", "host", "port", "capacity_slots",
                      "per_node_lease_limit", "node_seed", "quota_version"):
            if field not in node:
                _err(f"节点缺少必填字段 {field}", f"{base}.{field}")
        node_id = node["node_id"]
        if not isinstance(node_id, str) or not node_id:
            _err("node_id 必须是非空字符串", f"{base}.node_id")
        if node_id in seen_ids:
            _err(f"节点标识重复: {node_id}", f"{base}.node_id")
        seen_ids.add(node_id)

        host = node["host"]
        if host != "127.0.0.1":
            _err("host 必须是回环字面量 127.0.0.1", f"{base}.host")
        port = node["port"]
        if not _is_pos_int(port) or not (1 <= port <= 65535):
            _err("port 必须是 1..65535 的正整数", f"{base}.port")
        if port in seen_ports:
            _err(f"回环端口全局重复: {port}", f"{base}.port")
        seen_ports.add(port)

        for field in ("capacity_slots", "per_node_lease_limit"):
            if not _is_pos_int(node[field]):
                _err(f"{field} 必须是正整数", f"{base}.{field}")
        if node["per_node_lease_limit"] > node["capacity_slots"]:
            _err("per_node_lease_limit 不得超过 capacity_slots",
                 f"{base}.per_node_lease_limit")
        if not _is_pos_int(node["node_seed"]):
            _err("node_seed 必须是正整数", f"{base}.node_seed")
        if not _is_nonneg_int(node["quota_version"]):
            _err("quota_version 必须是非负整数", f"{base}.quota_version")

        out.append({
            "node_id": node_id,
            "host": host,
            "port": port,
            "capacity_slots": node["capacity_slots"],
            "per_node_lease_limit": node["per_node_lease_limit"],
            "node_seed": node["node_seed"],
            "quota_version": node["quota_version"],
        })
    return out


def _validate_contents(contents):
    if not isinstance(contents, list) or not contents:
        _err("contents 必须是非空数组", "$.contents")
    out = []
    seen_keys = set()
    for i, item in enumerate(contents):
        base = f"$.contents[{i}]"
        if not isinstance(item, dict):
            _err("内容条目必须是对象", base)
        for field in ("content_key", "total_bytes", "shard_count",
                      "content_seed"):
            if field not in item:
                _err(f"内容缺少必填字段 {field}", f"{base}.{field}")
        key = item["content_key"]
        if not isinstance(key, str) or not key:
            _err("content_key 必须是非空字符串", f"{base}.content_key")
        if key in seen_keys:
            _err(f"内容键重复: {key}", f"{base}.content_key")
        seen_keys.add(key)
        total_bytes = item["total_bytes"]
        shard_count = item["shard_count"]
        if not _is_pos_int(total_bytes):
            _err("total_bytes 必须是正整数", f"{base}.total_bytes")
        if not _is_pos_int(shard_count):
            _err("shard_count 必须是正整数", f"{base}.shard_count")
        if total_bytes % shard_count != 0:
            _err("shard_count 必须整除 total_bytes",
                 f"{base}.shard_count")
        if not _is_pos_int(item["content_seed"]):
            _err("content_seed 必须是正整数", f"{base}.content_seed")
        out.append({
            "content_key": key,
            "total_bytes": total_bytes,
            "shard_count": shard_count,
            "content_seed": item["content_seed"],
            "shard_byte_size": total_bytes // shard_count,
        })
    return out


def _validate_events(events, nodes, contents):
    if not isinstance(events, list) or not events:
        _err("events 必须是非空数组", "$.events")
    node_ids = {n["node_id"] for n in nodes}
    content_map = {c["content_key"]: c for c in contents}
    out = []
    prev_seq = None
    for i, event in enumerate(events):
        base = f"$.events[{i}]"
        if not isinstance(event, dict):
            _err("事件必须是对象", base)
        if "seq" not in event:
            _err("事件缺少必填字段 seq", f"{base}.seq")
        seq = event["seq"]
        if not _is_pos_int(seq):
            _err("事件 seq 必须是正整数", f"event#{seq if isinstance(seq, int) else i}.seq")
        if prev_seq is not None and seq <= prev_seq:
            _err(f"事件 seq 必须严格单调递增，重复或回退: {seq}",
                 f"event#{seq}.seq")
        prev_seq = seq

        etype = event.get("type")
        if etype not in EVENT_TYPES:
            _err(f"未知事件类型: {etype!r}", f"event#{seq}.type")

        ev = {"seq": seq, "type": etype}
        if etype in ("write", "read", "renew", "release", "cancel"):
            node_id = event.get("node_id")
            if node_id not in node_ids:
                _err(f"事件引用了不存在的节点: {node_id!r}",
                     f"event#{seq}.node_id")
            ev["node_id"] = node_id
            key = event.get("content_key")
            if key not in content_map:
                _err(f"事件引用了不存在的内容键: {key!r}",
                     f"event#{seq}.content_key")
            ev["content_key"] = key
            shard_index = event.get("shard_index")
            content = content_map[key]
            if not _is_nonneg_int(shard_index):
                _err("shard_index 必须是非负整数",
                     f"event#{seq}.shard_index")
            if shard_index >= content["shard_count"]:
                _err(
                    f"分片序号越界: {shard_index} >= {content['shard_count']}",
                    f"event#{seq}.shard_index")
            ev["shard_index"] = shard_index

        if etype in ("write", "renew"):
            ttl = event.get("ttl_ms")
            if not _is_pos_int(ttl):
                _err(f"{etype} 事件 ttl_ms 必须是正整数",
                     f"event#{seq}.ttl_ms")
            ev["ttl_ms"] = ttl
        elif etype == "quota_change":
            node_id = event.get("node_id")
            if node_id not in node_ids:
                _err(f"事件引用了不存在的节点: {node_id!r}",
                     f"event#{seq}.node_id")
            ev["node_id"] = node_id
            new_quota = event.get("new_quota_version")
            if not _is_nonneg_int(new_quota):
                _err("new_quota_version 必须是非负整数",
                     f"event#{seq}.new_quota_version")
            ev["new_quota_version"] = new_quota
        out.append(ev)
    return out


def spec_digest(spec: dict) -> str:
    """声明内容摘要：规范化 JSON 后取 sha256，用于快照版本识别。"""
    import hashlib
    canonical = json.dumps(
        {
            "session": spec["session"],
            "nodes": spec["nodes"],
            "contents": spec["contents"],
            "events": spec["events"],
        },
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
