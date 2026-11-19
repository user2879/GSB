"""v1/v2/v3 快照的版本化编解码与归一。

格式定义（顶层始终带 format_version/state_version）：
- v1: entries[].key, expire_ms；v1 没有命名空间概念，统一归一到 default。
- v2: 增加 entries[].annotations（条目级注释依赖集合）。
- v3: 再增加 namespaces/parents 继承链与 digest_algorithm 值摘要算法标识。

轨迹 rounds 在三个版本中都持久化；v1 条目没有命名空间信息，因此
升级恢复后的重放会在首个轮次以 E_REPLAY 如实报告口径不可复现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import (
    STATE_VERSION,
    SUPPORTED_DIGEST_ALGORITHMS,
    SUPPORTED_FORMAT_VERSIONS,
)
from .errors import EFormat, ESnapshot
from .jsonpos import IncompleteJsonError, JsonPosError, LocDict, parse_text
from .loader import build_graph
from .model import EntryID, Graph, RoundRecord

DEFAULT_NAMESPACE = "default"


@dataclass
class SnapshotData:
    format_version: int
    state_version: int
    graph: Graph
    rounds: list[RoundRecord]


def _line(doc: object, tokens: list[str | int]) -> int | None:
    from .jsonpos import locate

    return locate(doc, tokens)


def _field_path(*tokens: str | int) -> str:
    parts: list[str] = []
    for token in tokens:
        if isinstance(token, int):
            parts[-1] = f"{parts[-1]}[{token}]"
        else:
            parts.append(token)
    return ".".join(parts)


def _require(payload: Any, field: str, tokens: list[str | int],
             doc: Any) -> Any:
    if not isinstance(payload, dict) or field not in payload:
        line = _line(doc, tokens + [field])
        if line is None and isinstance(doc, LocDict):
            line = doc.line
        if line is None:
            line = 1
        raise EFormat(
            f"快照缺少必需字段 {field}",
            field_path=_field_path(*tokens, field),
            line=line,
        )
    return payload[field]


def decode_snapshot_text(text: str) -> SnapshotData:
    try:
        doc = parse_text(text)
    except IncompleteJsonError as exc:
        raise ESnapshot(
            f"快照文件尾部截断，JSON 结构不完整: {exc}",
            field_path="<document>", line=exc.line,
            details={"incomplete": True},
        ) from exc
    except JsonPosError as exc:
        raise ESnapshot(
            f"快照 JSON 无法解析: {exc}",
            field_path="<document>", line=exc.line,
        ) from exc

    if not isinstance(doc, dict):
        raise EFormat("快照顶层必须是对象", field_path="$",
                      line=getattr(doc, "line", None))
    format_version = _require(doc, "format_version", [], doc)
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise EFormat("format_version 必须是整数",
                      field_path="format_version",
                      line=_line(doc, ["format_version"]))
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise EFormat(
            f"未知快照格式版本: {format_version}",
            field_path="format_version",
            line=_line(doc, ["format_version"]),
            details={"supported": list(SUPPORTED_FORMAT_VERSIONS)},
        )
    state_version = _require(doc, "state_version", [], doc)
    if not isinstance(state_version, int) or isinstance(state_version, bool):
        raise EFormat("state_version 必须是整数",
                      field_path="state_version",
                      line=_line(doc, ["state_version"]))

    raw_entries = _require(doc, "entries", [], doc)
    if not isinstance(raw_entries, list):
        raise EFormat("entries 必须是数组", field_path="entries",
                      line=_line(doc, ["entries"]))

    if format_version == 1:
        namespace_doc, entry_doc = _normalize_v1(raw_entries, doc)
    elif format_version == 2:
        namespace_doc, entry_doc = _normalize_v2(raw_entries, doc)
    else:
        namespace_doc, entry_doc = _normalize_v3(doc, raw_entries)

    graph = build_graph(namespace_doc, entry_doc,
                        located_namespace_doc=namespace_doc,
                        located_entry_doc=entry_doc)
    rounds = _decode_rounds(doc, format_version)
    return SnapshotData(
        format_version=format_version,
        state_version=state_version,
        graph=graph,
        rounds=rounds,
    )


def _decode_rounds(doc: Any, format_version: int) -> list[RoundRecord]:
    raw_rounds = doc.get("rounds", [])
    if not isinstance(raw_rounds, list):
        raise EFormat("rounds 必须是数组", field_path="rounds",
                      line=_line(doc, ["rounds"]))
    rounds: list[RoundRecord] = []
    for index, raw in enumerate(raw_rounds):
        tokens = ["rounds", index]
        if not isinstance(raw, dict):
            raise EFormat("轮次记录必须是对象",
                          field_path=_field_path(*tokens),
                          line=_line(doc, tokens))
        for field in ("round_no", "clock_ms", "affected",
                      "annotation_versions", "trigger_reason"):
            _require(raw, field, tokens, doc)
        payload = dict(raw)
        payload.setdefault("explicit_triggers", [])
        payload.setdefault("expired_triggers", [])
        payload.setdefault("refresh_triggers", [])
        payload.setdefault("format_version", format_version)
        rounds.append(RoundRecord.from_dict(payload))
    rounds.sort(key=lambda item: item.round_no)
    return rounds


def _normalize_v1(raw_entries: Any, doc: Any) -> tuple[dict, dict]:
    normalized_entries: list[dict] = []
    for index, raw in enumerate(raw_entries):
        tokens = ["entries", index]
        if not isinstance(raw, dict):
            raise EFormat("v1 条目必须是对象",
                          field_path=_field_path(*tokens),
                          line=_line(doc, tokens))
        key = _require(raw, "key", tokens, doc)
        expire_ms = _require(raw, "expire_ms", tokens, doc)
        if not isinstance(key, int) or isinstance(key, bool):
            raise EFormat("v1 条目 key 必须是整数",
                          field_path=_field_path(*tokens, "key"),
                          line=_line(doc, tokens + ["key"]))
        if not isinstance(expire_ms, int) or isinstance(expire_ms, bool):
            raise EFormat("v1 expire_ms 必须是整数",
                          field_path=_field_path(*tokens, "expire_ms"),
                          line=_line(doc, tokens + ["expire_ms"]))
        normalized_entries.append({
            "key": key,
            "namespace": DEFAULT_NAMESPACE,
            "expire_ms": expire_ms,
            "digest": "",
            "digest_algorithm": "none",
            "annotations": [],
        })
    return (
        {"namespaces": [{"name": DEFAULT_NAMESPACE, "parents": []}]},
        {"entries": normalized_entries},
    )


def _normalize_v2(raw_entries: Any, doc: Any) -> tuple[dict, dict]:
    normalized_entries: list[dict] = []
    seen_keys: set[int] = set()
    for index, raw in enumerate(raw_entries):
        tokens = ["entries", index]
        if not isinstance(raw, dict):
            raise EFormat("v2 条目必须是对象",
                          field_path=_field_path(*tokens),
                          line=_line(doc, tokens))
        key = _require(raw, "key", tokens, doc)
        expire_ms = _require(raw, "expire_ms", tokens, doc)
        annotations = raw.get("annotations", [])
        if not isinstance(key, int) or isinstance(key, bool):
            raise EFormat("v2 条目 key 必须是整数",
                          field_path=_field_path(*tokens, "key"),
                          line=_line(doc, tokens + ["key"]))
        if not isinstance(expire_ms, int) or isinstance(expire_ms, bool):
            raise EFormat("v2 expire_ms 必须是整数",
                          field_path=_field_path(*tokens, "expire_ms"),
                          line=_line(doc, tokens + ["expire_ms"]))
        if not isinstance(annotations, list):
            raise EFormat("v2 annotations 必须是数组",
                          field_path=_field_path(*tokens, "annotations"),
                          line=_line(doc, tokens + ["annotations"]))
        if key in seen_keys:
            raise EFormat(f"v2 全局条目键重复: {key}",
                          field_path=_field_path(*tokens, "key"),
                          line=_line(doc, tokens + ["key"]))
        seen_keys.add(key)
        normalized_entries.append({
            "key": key,
            "namespace": DEFAULT_NAMESPACE,
            "expire_ms": expire_ms,
            "digest": raw.get("digest", ""),
            "digest_algorithm": "none",
            "annotations": [
                ref if isinstance(ref, dict) else {"key": ref}
                for ref in annotations
            ],
        })
    # v2 引用同样落在 default 命名空间，补全 namespace 以便统一解析。
    for entry in normalized_entries:
        for ref in entry["annotations"]:
            ref.setdefault("namespace", DEFAULT_NAMESPACE)
    return (
        {"namespaces": [{"name": DEFAULT_NAMESPACE, "parents": []}]},
        {"entries": normalized_entries},
    )


def _normalize_v3(doc: Any, raw_entries: Any) -> tuple[dict, dict]:
    raw_namespaces = _require(doc, "namespaces", [], doc)
    if not isinstance(raw_namespaces, list):
        raise EFormat("v3 namespaces 必须是数组",
                      field_path="namespaces",
                      line=_line(doc, ["namespaces"]))
    normalized_namespaces: list[dict] = []
    for index, raw in enumerate(raw_namespaces):
        tokens = ["namespaces", index]
        if not isinstance(raw, dict):
            raise EFormat("v3 命名空间必须是对象",
                          field_path=_field_path(*tokens),
                          line=_line(doc, tokens))
        name = _require(raw, "name", tokens, doc)
        parents = raw.get("parents", [])
        if not isinstance(parents, list):
            raise EFormat("v3 parents 必须是数组",
                          field_path=_field_path(*tokens, "parents"),
                          line=_line(doc, tokens + ["parents"]))
        normalized_namespaces.append({"name": name, "parents": list(parents)})

    normalized_entries: list[dict] = []
    for index, raw in enumerate(raw_entries):
        tokens = ["entries", index]
        if not isinstance(raw, dict):
            raise EFormat("v3 条目必须是对象",
                          field_path=_field_path(*tokens),
                          line=_line(doc, tokens))
        for field in ("key", "namespace", "expire_ms", "digest_algorithm"):
            _require(raw, field, tokens, doc)
        algorithm = raw["digest_algorithm"]
        if algorithm not in SUPPORTED_DIGEST_ALGORITHMS:
            raise EFormat(
                f"不支持的值摘要算法标识: {algorithm!r}",
                key=str(raw.get("key")),
                field_path=_field_path(*tokens, "digest_algorithm"),
                line=_line(doc, tokens + ["digest_algorithm"]),
                details={"supported": list(SUPPORTED_DIGEST_ALGORITHMS)},
            )
        normalized_entries.append({
            "key": raw["key"],
            "namespace": raw["namespace"],
            "expire_ms": raw["expire_ms"],
            "digest": raw.get("digest", ""),
            "digest_algorithm": algorithm,
            "annotation_version": int(raw.get("annotation_version", 0)),
            "annotations": list(raw.get("annotations", [])),
        })
    return (
        {"namespaces": normalized_namespaces},
        {"entries": normalized_entries},
    )


def decode_snapshot_file(path: str | Path) -> SnapshotData:
    return decode_snapshot_text(Path(path).read_text(encoding="utf-8"))


def encode_snapshot(graph: Graph, rounds: list[RoundRecord],
                    format_version: int, *,
                    state_version: int = STATE_VERSION) -> str:
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise EFormat(
            f"未知快照格式版本: {format_version}",
            field_path="format_version",
            details={"supported": list(SUPPORTED_FORMAT_VERSIONS)},
        )
    payload: dict[str, Any] = {
        "format_version": format_version,
        "state_version": state_version,
    }
    if format_version == 3:
        payload["namespaces"] = [
            {"name": name, "parents": list(graph.namespace_parents[name])}
            for name in sorted(graph.namespaces)
        ]
    payload["entries"] = _encode_entries(graph, format_version)
    if format_version == 1:
        payload["rounds"] = [
            _downgrade_round_identifiers(round_record)
            for round_record in rounds
        ]
    else:
        payload["rounds"] = [
            round_record.to_dict() for round_record in rounds
        ]
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)


def _downgrade_round_identifiers(record: RoundRecord) -> dict:
    payload = record.to_dict()

    def normalize(identifier: str) -> str:
        return f"{DEFAULT_NAMESPACE}/{EntryID.parse(identifier).key}"

    for field in ("explicit_triggers", "expired_triggers",
                  "refresh_triggers", "affected"):
        payload[field] = [normalize(item) for item in payload[field]]
    payload["annotation_versions"] = {
        normalize(key): value
        for key, value in payload["annotation_versions"].items()
    }
    payload["format_version"] = 1
    return payload


def _encode_entries(graph: Graph, format_version: int) -> list[dict]:
    result: list[dict] = []
    for entry_id in graph.entry_ids_sorted():
        entry = graph.entries[entry_id]
        if format_version == 1:
            result.append({
                "key": entry_id.key,
                "expire_ms": entry.expire_ms,
            })
            continue
        targets = sorted(graph.annotation_edges[entry_id])
        if format_version == 2:
            result.append({
                "key": entry_id.key,
                "expire_ms": entry.expire_ms,
                "digest": entry.digest,
                "annotations": [
                    {"key": target.key} for target in targets
                ],
            })
        else:
            result.append({
                "key": entry_id.key,
                "namespace": entry_id.namespace,
                "expire_ms": entry.expire_ms,
                "digest": entry.digest,
                "digest_algorithm": entry.digest_algorithm,
                "annotation_version": entry.annotation_version,
                "annotations": [
                    {"namespace": target.namespace, "key": target.key}
                    for target in targets
                ],
            })
    return result


def write_snapshot(path: str | Path, graph: Graph,
                   rounds: list[RoundRecord], format_version: int, *,
                   state_version: int = STATE_VERSION) -> None:
    Path(path).write_text(
        encode_snapshot(graph, rounds, format_version,
                        state_version=state_version),
        encoding="utf-8",
    )
