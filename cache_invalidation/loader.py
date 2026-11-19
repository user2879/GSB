"""命名空间与条目声明文件的加载和强校验。

输入为两个本地 JSON 文件：
- 命名空间声明: {"namespaces": [{"name": str, "parents": [str, ...]}]}
- 条目声明: {"entries": [{
      "key": int, "namespace": str, "expire_ms": int,
      "digest": str, "digest_algorithm": str,
      "annotations": [{"namespace": str, "key": int} | int, ...]
  }]}

整数注释引用仅在全局键唯一时允许；歧义或缺失均按 E_GRAPH 拒绝，
保证同一套解析规则在快照归一与声明加载中一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import SUPPORTED_DIGEST_ALGORITHMS
from .errors import EGraph, EInherit, EFormat
from .jsonpos import JsonPosError, parse_text
from .model import Entry, EntryID, Graph, Namespace


def _path(*tokens: str | int) -> str:
    parts: list[str] = []
    for token in tokens:
        if isinstance(token, int):
            parts[-1] = f"{parts[-1]}[{token}]"
        else:
            parts.append(token)
    return ".".join(parts)


def _line(document: object, tokens: list[str | int]) -> int | None:
    from .jsonpos import locate

    return locate(document, tokens)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_declarations(namespace_path: str | Path,
                      entry_path: str | Path) -> Graph:
    namespace_text = Path(namespace_path).read_text(encoding="utf-8")
    entry_text = Path(entry_path).read_text(encoding="utf-8")
    try:
        namespace_doc = parse_text(namespace_text)
    except JsonPosError as exc:
        raise EGraph(
            f"命名空间声明 JSON 无法解析: {exc}",
            field_path="<document>", line=exc.line,
        ) from exc
    try:
        entry_doc = parse_text(entry_text)
    except JsonPosError as exc:
        raise EGraph(
            f"条目声明 JSON 无法解析: {exc}",
            field_path="<document>", line=exc.line,
        ) from exc
    return build_graph(namespace_doc, entry_doc,
                       located_namespace_doc=namespace_doc,
                       located_entry_doc=entry_doc)


def build_graph(namespace_doc: Any, entry_doc: Any, *,
                located_namespace_doc: Any | None = None,
                located_entry_doc: Any | None = None,
                default_digest_algorithm: str = "none") -> Graph:
    ns_source = (located_namespace_doc if located_namespace_doc is not None
                 else namespace_doc)
    en_source = (located_entry_doc if located_entry_doc is not None
                 else entry_doc)
    namespaces, ns_parents = _load_namespaces(namespace_doc, ns_source)
    entries, edges = _load_entries(entry_doc, en_source, namespaces,
                                   default_digest_algorithm)
    _validate_annotation_graph(entries, edges, en_source)
    return Graph(namespaces=namespaces, entries=entries,
                 annotation_edges=edges, namespace_parents=ns_parents)


def _load_namespaces(doc: Any, located: Any) -> tuple[
        dict[str, Namespace], dict[str, tuple[str, ...]]]:
    if not isinstance(doc, dict) or "namespaces" not in doc:
        raise EGraph("命名空间声明缺少 namespaces 数组",
                     field_path="namespaces",
                     line=_line(located, ["namespaces"]))
    raw_namespaces = doc["namespaces"]
    if not isinstance(raw_namespaces, list):
        raise EGraph("namespaces 必须是数组", field_path="namespaces",
                     line=_line(located, ["namespaces"]))

    namespaces: dict[str, Namespace] = {}
    parents_map: dict[str, tuple[str, ...]] = {}
    for index, raw in enumerate(raw_namespaces):
        tokens = ["namespaces", index]
        if not isinstance(raw, dict) or "name" not in raw:
            raise EGraph("命名空间声明缺少 name",
                         field_path=_path(*tokens, "name"),
                         line=_line(located, tokens + ["name"]))
        name = raw["name"]
        if not isinstance(name, str) or not name:
            raise EGraph("命名空间 name 必须是非空字符串",
                         key=str(name), field_path=_path(*tokens, "name"),
                         line=_line(located, tokens + ["name"]))
        if name in namespaces:
            raise EGraph(f"命名空间标识重复: {name}", key=name,
                         field_path=_path(*tokens, "name"),
                         line=_line(located, tokens + ["name"]))
        raw_parents = raw.get("parents", [])
        if not isinstance(raw_parents, list) or not all(
                isinstance(item, str) for item in raw_parents):
            raise EGraph("parents 必须是字符串数组", key=name,
                         field_path=_path(*tokens, "parents"),
                         line=_line(located, tokens + ["parents"]))
        if len(set(raw_parents)) != len(raw_parents):
            raise EGraph(
                f"命名空间 {name} 的继承链存在跨层重复父级",
                key=name, field_path=_path(*tokens, "parents"),
                line=_line(located, tokens + ["parents"]),
                details={"parents": list(raw_parents)},
            )
        namespaces[name] = Namespace(name, tuple(raw_parents))
        parents_map[name] = tuple(raw_parents)

    for name, parents in parents_map.items():
        for parent in parents:
            if parent not in namespaces:
                index = next(
                    i for i, item in enumerate(raw_namespaces)
                    if isinstance(item, dict) and item.get("name") == name
                )
                raise EInherit(
                    f"命名空间 {name} 引用了不存在的上游命名空间 {parent}",
                    key=name,
                    field_path=_path("namespaces", index, "parents"),
                    line=_line(located, ["namespaces", index, "parents"]),
                    details={"missing_parent": parent},
                )

    _validate_inheritance_acyclic(parents_map, located)
    return namespaces, parents_map


def _validate_inheritance_acyclic(
        parents_map: dict[str, tuple[str, ...]], located: Any) -> None:
    color: dict[str, int] = {name: 0 for name in parents_map}
    stack: list[str] = []

    def visit(name: str) -> None:
        color[name] = 1
        stack.append(name)
        for parent in parents_map[name]:
            if color[parent] == 1:
                cycle_start = stack.index(parent)
                cycle = stack[cycle_start:] + [parent]
                raise EGraph(
                    "命名空间继承链存在环: " + " -> ".join(cycle),
                    key=parent,
                    field_path="namespaces",
                    line=_line(located, ["namespaces"]),
                    cycle=cycle,
                )
            if color[parent] == 0:
                visit(parent)
        stack.pop()
        color[name] = 2

    for name in sorted(parents_map):
        if color[name] == 0:
            visit(name)


def _load_entries(doc: Any, located: Any,
                  namespaces: dict[str, Namespace],
                  default_algorithm: str) -> tuple[
        dict[EntryID, Entry], dict[EntryID, set[EntryID]]]:
    if not isinstance(doc, dict) or "entries" not in doc:
        raise EGraph("条目声明缺少 entries 数组", field_path="entries",
                     line=_line(located, ["entries"]))
    raw_entries = doc["entries"]
    if not isinstance(raw_entries, list):
        raise EGraph("entries 必须是数组", field_path="entries",
                     line=_line(located, ["entries"]))

    entries: dict[EntryID, Entry] = {}
    key_to_ns: dict[int, list[str]] = {}
    for index, raw in enumerate(raw_entries):
        tokens = ["entries", index]
        if not isinstance(raw, dict):
            raise EGraph("条目声明必须是对象",
                         field_path=_path(*tokens),
                         line=_line(located, tokens))
        key = raw.get("key")
        namespace = raw.get("namespace")
        if not _is_int(key):
            raise EGraph("条目 key 必须是整数",
                         key=str(key), field_path=_path(*tokens, "key"),
                         line=_line(located, tokens + ["key"]))
        if not isinstance(namespace, str) or namespace not in namespaces:
            raise EGraph(
                f"条目 {key} 所属命名空间不存在: {namespace!r}",
                key=str(key), field_path=_path(*tokens, "namespace"),
                line=_line(located, tokens + ["namespace"]),
            )
        entry_id = EntryID(namespace, key)
        if entry_id in entries:
            raise EGraph(
                f"命名空间 {namespace} 内条目键重复: {key}",
                key=entry_id.to_str(),
                field_path=_path(*tokens, "key"),
                line=_line(located, tokens + ["key"]),
            )
        expire_ms = raw.get("expire_ms")
        if not _is_int(expire_ms) or expire_ms < 0:
            raise EGraph(
                f"条目 {entry_id} 的 expire_ms 必须是非负整数毫秒",
                key=entry_id.to_str(),
                field_path=_path(*tokens, "expire_ms"),
                line=_line(located, tokens + ["expire_ms"]),
            )
        digest = raw.get("digest", "")
        if not isinstance(digest, str):
            raise EGraph(f"条目 {entry_id} 的 digest 必须是字符串",
                         key=entry_id.to_str(),
                         field_path=_path(*tokens, "digest"),
                         line=_line(located, tokens + ["digest"]))
        algorithm = raw.get("digest_algorithm", default_algorithm)
        if algorithm not in SUPPORTED_DIGEST_ALGORITHMS:
            raise EFormat(
                f"不支持的值摘要算法标识: {algorithm!r}",
                key=entry_id.to_str(),
                field_path=_path(*tokens, "digest_algorithm"),
                line=_line(located, tokens + ["digest_algorithm"]),
                details={"supported": list(SUPPORTED_DIGEST_ALGORITHMS)},
            )
        entries[entry_id] = Entry(
            id=entry_id, expire_ms=expire_ms, digest=digest,
            digest_algorithm=algorithm,
            annotation_version=int(raw.get("annotation_version", 0)),
        )
        key_to_ns.setdefault(key, []).append(namespace)

    edges: dict[EntryID, set[EntryID]] = {eid: set() for eid in entries}
    for index, raw in enumerate(raw_entries):
        source = EntryID(raw["namespace"], raw["key"])
        tokens = ["entries", index, "annotations"]
        raw_refs = raw.get("annotations", [])
        if not isinstance(raw_refs, list):
            raise EGraph("annotations 必须是数组",
                         key=source.to_str(), field_path=_path(*tokens),
                         line=_line(located, tokens))
        for ref_index, ref in enumerate(raw_refs):
            target = _resolve_reference(ref, entries, key_to_ns, source,
                                        located, tokens, ref_index)
            if target == source:
                raise EGraph(
                    f"条目 {source} 不能注释自身",
                    key=source.to_str(),
                    field_path=_path(*tokens, ref_index),
                    line=_line(located, tokens + [ref_index]),
                )
            edges[source].add(target)
    return entries, edges


def _resolve_reference(ref: Any, entries: dict[EntryID, Entry],
                       key_to_ns: dict[int, list[str]], source: EntryID,
                       located: Any, tokens: list[str | int],
                       ref_index: int) -> EntryID:
    ref_tokens = tokens + [ref_index]
    if isinstance(ref, dict):
        if "namespace" not in ref or "key" not in ref:
            raise EGraph("注释引用必须包含 namespace 与 key",
                         key=source.to_str(),
                         field_path=_path(*ref_tokens),
                         line=_line(located, ref_tokens))
        namespace = ref["namespace"]
        key = ref["key"]
        if not isinstance(namespace, str) or not _is_int(key):
            raise EGraph("注释引用的 namespace 须为字符串、key 须为整数",
                         key=source.to_str(),
                         field_path=_path(*ref_tokens),
                         line=_line(located, ref_tokens))
        target = EntryID(namespace, key)
    elif _is_int(ref):
        namespaces = key_to_ns.get(ref, [])
        if len(namespaces) != 1:
            raise EGraph(
                f"条目 {source} 的整数注释引用 {ref} 无法唯一解析",
                key=source.to_str(),
                field_path=_path(*ref_tokens),
                line=_line(located, ref_tokens),
                details={"candidate_namespaces": sorted(namespaces)},
            )
        target = EntryID(namespaces[0], ref)
    else:
        raise EGraph("注释引用必须是整数或 {namespace,key} 对象",
                     key=source.to_str(), field_path=_path(*ref_tokens),
                     line=_line(located, ref_tokens))
    if target not in entries:
        raise EGraph(
            f"条目 {source} 注释引用的目标不存在: {target}",
            key=source.to_str(), field_path=_path(*ref_tokens),
            line=_line(located, ref_tokens),
            details={"missing_target": target.to_str()},
        )
    return target


def _validate_annotation_graph(
        entries: dict[EntryID, Entry], edges: dict[EntryID, set[EntryID]],
        located: Any) -> None:
    color: dict[EntryID, int] = {eid: 0 for eid in entries}
    stack: list[EntryID] = []

    def visit(node: EntryID) -> None:
        color[node] = 1
        stack.append(node)
        for target in sorted(edges[node]):
            if color[target] == 1:
                cycle_start = stack.index(target)
                cycle = [item.to_str() for item in stack[cycle_start:]]
                cycle.append(target.to_str())
                raise EGraph(
                    "注释依赖图存在环: " + " -> ".join(cycle),
                    key=target.to_str(), field_path="entries",
                    line=_line(located, ["entries"]), cycle=cycle,
                )
            if color[target] == 0:
                visit(target)
        stack.pop()
        color[node] = 2

    for entry_id in sorted(entries):
        if color[entry_id] == 0:
            visit(entry_id)
