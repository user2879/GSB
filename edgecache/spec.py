"""Strict local JSON / JSONL declaration loading with source locations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SpecError
from .model import Config, ContentSpec, Event, Manifest, NodeSpec

_EVENT_TYPES = {"apply", "renew", "release", "quota"}
_TOP_LEVEL_KEYS = {"format_version", "config", "nodes", "contents", "events"}
_CONFIG_KEYS = {
    "sample_scale",
    "lease_ttl_ms",
    "global_lease_limit",
    "global_byte_budget",
    "socket_timeout_ms",
    "handshake_timeout_ms",
    "max_workers",
    "drain_timeout_ms",
}
_NODE_KEYS = {"node_id", "host", "port", "slots", "lease_limit", "seed", "quota_version"}
_CONTENT_KEYS = {"content_key", "total_bytes", "shard_count", "content_seed"}
_BASE_EVENT_KEYS = {"seq", "type", "delay_ms"}
_EVENT_KEYS = {
    "apply": _BASE_EVENT_KEYS | {"node_id", "content_key", "shards"},
    "renew": _BASE_EVENT_KEYS | {"lease_id", "expected_version", "extend_ms"},
    "release": _BASE_EVENT_KEYS | {"lease_id", "expected_version"},
    "quota": _BASE_EVENT_KEYS | {"quota_version", "node_ids"},
}


@dataclass
class _Issue:
    line: int
    order: int
    field: str
    message: str
    path: str


class _Source:
    def __init__(self, text: str, *, jsonl: bool = False) -> None:
        self.text = text
        self.lines = text.splitlines()
        self.jsonl = jsonl
        self.entity_lines: dict[str, list[int]] = {"node": [], "content": [], "event": []}
        self.config_line = 1

    def first_line(self, token: str) -> int:
        pattern = re.compile(token)
        for index, line in enumerate(self.lines, start=1):
            if pattern.search(line):
                return index
        return 1

    def token_lines(self, token: str) -> list[int]:
        pattern = re.compile(token)
        return [index for index, line in enumerate(self.lines, start=1) if pattern.search(line)]

    def array_line(self, name: str) -> int:
        return self.first_line(r'"' + re.escape(name) + r'"\s*:')

    def identity_line(self, key: str, value: Any, fallback: int, occurrence: int = 0) -> int:
        token = r'"' + re.escape(key) + r'"\s*:\s*' + re.escape(json.dumps(value, ensure_ascii=False))
        found = self.token_lines(token)
        if found:
            return found[min(occurrence, len(found) - 1)]
        return fallback

    def field_line(self, field: str, fallback: int = 1) -> int:
        return self.first_line(r'"' + re.escape(field) + r'"\s*:')

    def line_after(self, field: str, anchor: int, limit: int | None = None) -> int:
        pattern = re.compile(r'"' + re.escape(field) + r'"\s*:')
        stop = len(self.lines) if limit is None else min(limit, len(self.lines))
        for index in range(max(0, anchor - 1), stop):
            if pattern.search(self.lines[index]):
                return index + 1
        return anchor


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _jsonl_payload(text: str, source: _Source) -> dict[str, Any]:
    result: dict[str, Any] = {"nodes": [], "contents": [], "events": []}
    saw_config = False
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"invalid JSON: {exc.msg}",
                line=line_no,
                field="<json>",
                path=f"line[{line_no}]",
            ) from exc
        if not isinstance(item, dict):
            raise SpecError("JSONL record must be an object", line=line_no, field="<record>", path=f"line[{line_no}]")
        kind = item.get("kind")
        if kind == "meta":
            result["format_version"] = item.get("format_version")
            continue
        if kind == "config":
            if saw_config:
                raise SpecError("duplicate config record", line=line_no, field="kind", path=f"line[{line_no}].kind")
            saw_config = True
            result["format_version"] = item.get("format_version")
            source.config_line = line_no
            config = item.get("config")
            if config is None:
                config = {key: value for key, value in item.items() if key not in {"kind", "format_version"}}
            result["config"] = config
            continue
        if kind in {"node", "content", "event"}:
            body = item.get(kind)
            if body is None:
                body = {key: value for key, value in item.items() if key != "kind"}
            result[f"{kind}s"].append(body)
            source.entity_lines[kind].append(line_no)
            continue
        raise SpecError("unknown JSONL kind", line=line_no, field="kind", path=f"line[{line_no}].kind")
    if not saw_config:
        result.setdefault("format_version", None)
    source.jsonl = True
    return result


def _load_object(path: Path) -> tuple[dict[str, Any], _Source]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        source = _Source(text, jsonl=True)
        return _jsonl_payload(text, source), source
    source = _Source(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            line=exc.lineno,
            field="<json>",
            path=f"line[{exc.lineno}]",
        ) from exc
    if not isinstance(data, dict):
        raise SpecError("top-level value must be an object", line=1, field="<root>", path="$")
    return data, source


def _resolve_path(sample: str | Path) -> Path:
    path = Path(sample)
    if path.is_dir():
        for name in ("manifest.json", "spec.json", "sample.json"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise SpecError("sample directory contains no manifest.json", line=1, field="<path>", path=str(path))
    if not path.exists():
        raise SpecError("sample file does not exist", line=1, field="<path>", path=str(path))
    return path


def load_manifest(sample: str | Path) -> Manifest:
    path = _resolve_path(sample)
    data, source = _load_object(path)
    issues: list[_Issue] = []
    issue_order = 0

    def add(line: int, field: str, message: str, path_text: str) -> None:
        nonlocal issue_order
        issues.append(_Issue(line=max(1, int(line)), order=issue_order, field=field, message=message, path=path_text))
        issue_order += 1

    def finish() -> None:
        if not issues:
            return
        issue = min(issues, key=lambda item: (item.line, item.order))
        raise SpecError(issue.message, line=issue.line, field=issue.field, path=issue.path)

    for key in data:
        if key not in _TOP_LEVEL_KEYS:
            add(source.field_line(key), key, "unknown top-level field", key)
    finish()

    format_version = data.get("format_version")
    if not _is_int(format_version) or format_version != 1:
        add(1, "format_version", "format_version must be integer 1", "format_version")
    config_obj = data.get("config")
    if not isinstance(config_obj, dict):
        add(source.field_line("config"), "config", "config must be an object", "config")
        config_obj = {}
    else:
        for key in config_obj:
            if key not in _CONFIG_KEYS:
                add(source.field_line(key), key, "unknown config field", f"config.{key}")

    config_defaults = {
        "socket_timeout_ms": 3000,
        "handshake_timeout_ms": 4000,
        "max_workers": 8,
        "drain_timeout_ms": 5000,
    }
    config_values: dict[str, Any] = dict(config_defaults)
    for key in _CONFIG_KEYS:
        if key in config_obj:
            config_values[key] = config_obj[key]
    config_anchor = source.config_line if source.jsonl else source.first_line(r'"config"\s*:')
    for key in _CONFIG_KEYS:
        value = config_values.get(key)
        line = source.line_after(key, config_anchor, source.array_line("nodes"))
        if key == "sample_scale":
            if not _is_int(value) or not 1 <= value <= 1_000_000:
                add(line, key, "sample_scale must be an integer in [1, 1000000]", f"config.{key}")
        elif key in {"socket_timeout_ms", "handshake_timeout_ms"}:
            if not _is_int(value) or value <= 0:
                add(line, key, f"{key} must be a positive integer", f"config.{key}")
        elif not _is_int(value) or value <= 0:
            add(line, key, f"{key} must be a positive integer", f"config.{key}")

    nodes_obj = data.get("nodes")
    if not isinstance(nodes_obj, list) or not nodes_obj:
        add(source.array_line("nodes"), "nodes", "nodes must be a non-empty array", "nodes")
        nodes_obj = []
    contents_obj = data.get("contents")
    if not isinstance(contents_obj, list) or not contents_obj:
        add(source.array_line("contents"), "contents", "contents must be a non-empty array", "contents")
        contents_obj = []
    events_obj = data.get("events")
    if not isinstance(events_obj, list):
        add(source.array_line("events"), "events", "events must be an array", "events")
        events_obj = []

    nodes: list[NodeSpec] = []
    seen_nodes: dict[str, int] = {}
    seen_ports: dict[int, str] = {}
    node_line_fallback = source.array_line("nodes")
    for index, obj in enumerate(nodes_obj):
        fallback = (
            source.entity_lines["node"][index]
            if source.jsonl and index < len(source.entity_lines["node"])
            else node_line_fallback + index if source.jsonl else node_line_fallback
        )
        if not isinstance(obj, dict):
            add(fallback, "<node>", "node entry must be an object", f"nodes[{index}]")
            continue
        node_id = obj.get("node_id")
        line = source.identity_line("node_id", node_id, fallback, index) if isinstance(node_id, str) else fallback
        def node_field_line(field: str) -> int:
            return source.line_after(field, line)
        for key in obj:
            if key not in _NODE_KEYS:
                add(node_field_line(key), key, "unknown node field", f"nodes[{index}].{key}")
        if not isinstance(node_id, str) or not node_id:
            add(line, "node_id", "node_id must be a non-empty string", f"nodes[{index}].node_id")
            node_id = f"<node-{index}>"
        elif node_id in seen_nodes:
            add(line, "node_id", "node_id must be unique", f"nodes[{index}].node_id")
        else:
            seen_nodes[node_id] = index
        host = obj.get("host")
        if host != "127.0.0.1":
            add(node_field_line("host"), "host", "host must be the literal string 127.0.0.1", f"nodes[{index}].host")
        port = obj.get("port")
        if not _is_int(port) or not 1 <= port <= 65535:
            add(node_field_line("port"), "port", "port must be an integer in [1, 65535]", f"nodes[{index}].port")
        elif port in seen_ports:
            add(node_field_line("port"), "port", f"port duplicates node {seen_ports[port]}", f"nodes[{index}].port")
        else:
            seen_ports[port] = node_id
        slots = obj.get("slots")
        if not _is_int(slots) or slots <= 0:
            add(node_field_line("slots"), "slots", "slots must be a positive integer", f"nodes[{index}].slots")
        lease_limit = obj.get("lease_limit")
        if not _is_int(lease_limit) or lease_limit <= 0:
            add(node_field_line("lease_limit"), "lease_limit", "lease_limit must be a positive integer", f"nodes[{index}].lease_limit")
        seed = obj.get("seed")
        if not _is_int(seed):
            add(node_field_line("seed"), "seed", "seed must be an integer", f"nodes[{index}].seed")
        quota_version = obj.get("quota_version")
        if not _is_int(quota_version) or quota_version <= 0:
            add(node_field_line("quota_version"), "quota_version", "quota_version must be a positive integer", f"nodes[{index}].quota_version")
        if all(
            value is not None
            for value in (node_id, port, slots, lease_limit, seed, quota_version)
        ) and host == "127.0.0.1":
            try:
                nodes.append(
                    NodeSpec(
                        node_id=node_id,
                        host=host,
                        port=port,
                        slots=slots,
                        lease_limit=lease_limit,
                        seed=seed,
                        quota_version=quota_version,
                    )
                )
            except TypeError:
                pass

    contents: list[ContentSpec] = []
    seen_contents: dict[str, int] = {}
    content_line_fallback = source.array_line("contents")
    for index, obj in enumerate(contents_obj):
        fallback = (
            source.entity_lines["content"][index]
            if source.jsonl and index < len(source.entity_lines["content"])
            else content_line_fallback + index if source.jsonl else content_line_fallback
        )
        if not isinstance(obj, dict):
            add(fallback, "<content>", "content entry must be an object", f"contents[{index}]")
            continue
        content_key = obj.get("content_key")
        line = (
            source.identity_line("content_key", content_key, fallback, index)
            if isinstance(content_key, str)
            else fallback
        )
        def content_field_line(field: str) -> int:
            return source.line_after(field, line)
        for key in obj:
            if key not in _CONTENT_KEYS:
                add(content_field_line(key), key, "unknown content field", f"contents[{index}].{key}")
        if not isinstance(content_key, str) or not content_key:
            add(line, "content_key", "content_key must be a non-empty string", f"contents[{index}].content_key")
            content_key = f"<content-{index}>"
        elif content_key in seen_contents:
            add(line, "content_key", "content_key must be unique", f"contents[{index}].content_key")
        else:
            seen_contents[content_key] = index
        total_bytes = obj.get("total_bytes")
        if not _is_int(total_bytes) or total_bytes <= 0:
            add(content_field_line("total_bytes"), "total_bytes", "total_bytes must be a positive integer", f"contents[{index}].total_bytes")
        shard_count = obj.get("shard_count")
        if not _is_int(shard_count) or shard_count <= 0:
            add(content_field_line("shard_count"), "shard_count", "shard_count must be a positive integer", f"contents[{index}].shard_count")
        elif _is_int(total_bytes) and total_bytes > 0 and total_bytes % shard_count != 0:
            add(content_field_line("shard_count"), "shard_count", "shard_count must divide total_bytes", f"contents[{index}].shard_count")
        content_seed = obj.get("content_seed")
        if not _is_int(content_seed):
            add(content_field_line("content_seed"), "content_seed", "content_seed must be an integer", f"contents[{index}].content_seed")
        if all(value is not None for value in (content_key, total_bytes, shard_count, content_seed)):
            try:
                contents.append(
                    ContentSpec(
                        content_key=content_key,
                        total_bytes=total_bytes,
                        shard_count=shard_count,
                        content_seed=content_seed,
                    )
                )
            except TypeError:
                pass

    events: list[Event] = []
    active_nodes = set(seen_nodes)
    expected_seq = 1
    current_quota = max((node.quota_version for node in nodes), default=None)
    event_line_fallback = source.array_line("events")
    for index, obj in enumerate(events_obj):
        fallback = (
            source.entity_lines["event"][index]
            if source.jsonl and index < len(source.entity_lines["event"])
            else event_line_fallback + index if source.jsonl else event_line_fallback
        )
        if not isinstance(obj, dict):
            add(fallback, "<event>", "event entry must be an object", f"events[{index}]")
            continue
        seq = obj.get("seq")
        line = source.identity_line("seq", seq, fallback, index) if _is_int(seq) else fallback
        def event_field_line(field: str) -> int:
            return source.line_after(field, line)
        if not _is_int(seq):
            add(line, "seq", "seq must be an integer", f"events[{index}].seq")
        elif seq != expected_seq:
            add(line, "seq", f"seq must be contiguous; expected {expected_seq}", f"events[{index}].seq")
            expected_seq = seq + 1
        else:
            expected_seq += 1
        event_type = obj.get("type")
        if event_type not in _EVENT_TYPES:
            add(line, "type", "type must be apply, renew, release or quota", f"events[{index}].type")
            continue
        for key in obj:
            if key not in _EVENT_KEYS[event_type]:
                add(event_field_line(key), key, f"unknown field for {event_type} event", f"events[{index}].{key}")
        delay_ms = obj.get("delay_ms", 0)
        if not _is_int(delay_ms) or delay_ms < 0:
            add(event_field_line("delay_ms"), "delay_ms", "delay_ms must be a non-negative integer", f"events[{index}].delay_ms")
            delay_ms = 0
        if event_type == "apply":
            node_id = obj.get("node_id")
            if not isinstance(node_id, str) or node_id not in seen_nodes:
                add(line, "node_id", "node_id must reference a declared node", f"events[{index}].node_id")
                continue
            if node_id not in active_nodes:
                add(line, "node_id", "node_id is not active at this event sequence", f"events[{index}].node_id")
            content_key = obj.get("content_key")
            if not isinstance(content_key, str) or content_key not in seen_contents:
                add(line, "content_key", "content_key must reference a declared content", f"events[{index}].content_key")
                continue
            content = next((item for item in contents if item.content_key == content_key), None)
            shards = obj.get("shards")
            if not isinstance(shards, list) or not shards:
                add(line, "shards", "shards must be a non-empty array", f"events[{index}].shards")
                continue
            valid_shards: list[int] = []
            seen_shards: set[int] = set()
            for shard_position, shard in enumerate(shards):
                if not _is_int(shard):
                    add(line, "shards", "every shard index must be an integer", f"events[{index}].shards[{shard_position}]")
                elif content is not None and not 0 <= shard < content.shard_count:
                    add(line, "shards", "shard index out of range", f"events[{index}].shards[{shard_position}]")
                elif shard in seen_shards:
                    add(line, "shards", "shard indices must be unique within an event", f"events[{index}].shards[{shard_position}]")
                else:
                    seen_shards.add(shard)
                    valid_shards.append(shard)
            events.append(
                Event(
                    seq=seq if _is_int(seq) else expected_seq - 1,
                    type=event_type,
                    delay_ms=delay_ms,
                    node_id=node_id,
                    content_key=content_key,
                    shards=tuple(valid_shards),
                )
            )
        elif event_type == "renew":
            lease_id = obj.get("lease_id")
            if not isinstance(lease_id, str) or not lease_id:
                add(line, "lease_id", "lease_id must be a non-empty string", f"events[{index}].lease_id")
            expected_version = obj.get("expected_version")
            if not _is_int(expected_version) or expected_version <= 0:
                add(line, "expected_version", "expected_version must be a positive integer", f"events[{index}].expected_version")
            extend_ms = obj.get("extend_ms")
            if extend_ms is not None and (not _is_int(extend_ms) or extend_ms <= 0):
                add(line, "extend_ms", "extend_ms must be a positive integer", f"events[{index}].extend_ms")
            events.append(
                Event(
                    seq=seq if _is_int(seq) else expected_seq - 1,
                    type=event_type,
                    delay_ms=delay_ms,
                    lease_id=lease_id if isinstance(lease_id, str) else None,
                    expected_version=expected_version if _is_int(expected_version) else None,
                    extend_ms=extend_ms if _is_int(extend_ms) else None,
                )
            )
        elif event_type == "release":
            lease_id = obj.get("lease_id")
            if not isinstance(lease_id, str) or not lease_id:
                add(line, "lease_id", "lease_id must be a non-empty string", f"events[{index}].lease_id")
            expected_version = obj.get("expected_version")
            if expected_version is not None and (not _is_int(expected_version) or expected_version <= 0):
                add(line, "expected_version", "expected_version must be a positive integer", f"events[{index}].expected_version")
            events.append(
                Event(
                    seq=seq if _is_int(seq) else expected_seq - 1,
                    type=event_type,
                    delay_ms=delay_ms,
                    lease_id=lease_id if isinstance(lease_id, str) else None,
                    expected_version=expected_version if _is_int(expected_version) else None,
                )
            )
        else:
            quota_version = obj.get("quota_version")
            if not _is_int(quota_version) or quota_version <= 0:
                add(line, "quota_version", "quota_version must be a positive integer", f"events[{index}].quota_version")
            elif current_quota is not None and quota_version <= current_quota:
                add(line, "quota_version", "quota_version must strictly increase", f"events[{index}].quota_version")
            else:
                current_quota = quota_version
            node_ids = obj.get("node_ids")
            if not isinstance(node_ids, list) or not node_ids:
                add(line, "node_ids", "node_ids must be a non-empty array", f"events[{index}].node_ids")
                node_ids = []
            seen: set[str] = set()
            valid_nodes: list[str] = []
            for node_position, node_id in enumerate(node_ids):
                if not isinstance(node_id, str) or node_id not in seen_nodes:
                    add(line, "node_ids", "node_ids must reference declared nodes", f"events[{index}].node_ids[{node_position}]")
                elif node_id in seen:
                    add(line, "node_ids", "node_ids must be unique", f"events[{index}].node_ids[{node_position}]")
                else:
                    seen.add(node_id)
                    valid_nodes.append(node_id)
            if node_ids and not valid_nodes:
                add(line, "node_ids", "a quota event must retain at least one active node", f"events[{index}].node_ids")
            active_nodes = set(valid_nodes) if valid_nodes else active_nodes
            events.append(
                Event(
                    seq=seq if _is_int(seq) else expected_seq - 1,
                    type=event_type,
                    delay_ms=delay_ms,
                    quota_version=quota_version if _is_int(quota_version) else None,
                    node_ids=tuple(valid_nodes),
                )
            )

    finish()
    if not nodes or not contents:
        finish()
    if not all(isinstance(value, int) for value in config_values.values()):
        finish()
    config = Config(**config_values)
    return Manifest(
        format_version=1,
        config=config,
        nodes=tuple(nodes),
        contents=tuple(contents),
        events=tuple(events),
        source_path=str(path.resolve()),
    )


def describe_manifest(manifest: Manifest) -> dict[str, Any]:
    return {
        "ok": True,
        "format_version": manifest.format_version,
        "manifest_digest": __import__("edgecache.content", fromlist=["manifest_digest"]).manifest_digest(manifest),
        "nodes": [node.node_id for node in manifest.nodes],
        "contents": [content.content_key for content in manifest.contents],
        "event_count": len(manifest.events),
    }
