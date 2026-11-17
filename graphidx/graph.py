"""属性图的加载、强校验与增量变更集应用。

校验失败一律以 E_GRAPH 整体拒绝，并输出出错元素与字段；
合法变更集应用后图版本号单调递增，内容摘要由显式排序规则确定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canon import digest
from .errors import GraphError


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _err(element: str, field: str, message: str) -> dict[str, str]:
    return {"element": element, "field": field, "message": message}


@dataclass(frozen=True)
class Edge:
    id: str
    src: int
    dst: int
    weight: int
    label: str

    def row(self) -> list[Any]:
        return [self.id, self.src, self.dst, self.weight, self.label]


@dataclass
class ChangeRecord:
    """一次成功变更集应用的完整记录，供受影响条目推导与视图失效使用。"""

    changeset_id: str
    version_before: int
    version_after: int
    digest_before: str
    digest_after: str
    ops_detail: list[dict[str, Any]] = field(default_factory=list)
    structural_nodes: list[int] = field(default_factory=list)
    structural: bool = False


class PropertyGraph:
    def __init__(
        self,
        nodes: dict[int, str] | None = None,
        edges: dict[str, Edge] | None = None,
        version: int = 0,
    ):
        self.nodes: dict[int, str] = nodes or {}
        self.edges: dict[str, Edge] = edges or {}
        self.version = version

    # ------------------------------------------------------------------
    # 加载与强校验
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Any, version: int = 0) -> "PropertyGraph":
        errors: list[dict[str, str]] = []
        if not isinstance(data, dict):
            raise GraphError(
                "graph document must be a JSON object",
                [_err("$", "type", "expected object")],
            )
        raw_nodes = data.get("nodes")
        raw_edges = data.get("edges")
        nodes: dict[int, str] = {}
        if not isinstance(raw_nodes, list):
            errors.append(_err("nodes", "type", "expected list of nodes"))
            raw_nodes = []
        for pos, item in enumerate(raw_nodes):
            elem = f"nodes[{pos}]"
            if not isinstance(item, dict):
                errors.append(_err(elem, "type", "expected object"))
                continue
            nid = item.get("id")
            label = item.get("label")
            if not _is_int(nid):
                errors.append(_err(elem, "id", "node id must be an integer"))
                continue
            if nid in nodes:
                errors.append(_err(elem, "id", f"duplicate node id {nid}"))
                continue
            if not isinstance(label, str) or not label:
                errors.append(_err(elem, "label", "node label must be a non-empty string"))
                continue
            nodes[nid] = label
        edges: dict[str, Edge] = {}
        if raw_edges is not None and not isinstance(raw_edges, list):
            errors.append(_err("edges", "type", "expected list of edges"))
            raw_edges = []
        for pos, item in enumerate(raw_edges or []):
            elem = f"edges[{pos}]"
            if not isinstance(item, dict):
                errors.append(_err(elem, "type", "expected object"))
                continue
            eid = item.get("id")
            src = item.get("src")
            dst = item.get("dst")
            weight = item.get("weight")
            label = item.get("label")
            ok = True
            if not isinstance(eid, str) or not eid:
                errors.append(_err(elem, "id", "edge id must be a non-empty string"))
                ok = False
            elif eid in edges:
                errors.append(_err(elem, "id", f"duplicate edge id {eid!r}"))
                ok = False
            for name, value in (("src", src), ("dst", dst)):
                if not _is_int(value):
                    errors.append(_err(elem, name, "endpoint must be an integer node id"))
                    ok = False
                elif value not in nodes:
                    errors.append(_err(elem, name, f"endpoint node {value} does not exist"))
                    ok = False
            if not _is_int(weight):
                errors.append(_err(elem, "weight", "edge weight must be an integer"))
                ok = False
            if not isinstance(label, str) or not label:
                errors.append(_err(elem, "label", "edge label must be a non-empty string"))
                ok = False
            if ok:
                edges[eid] = Edge(eid, src, dst, weight, label)
        if errors:
            raise GraphError("invalid graph document", errors)
        return cls(nodes, edges, version)

    @classmethod
    def from_content(cls, content: dict[str, Any], version: int = 0) -> "PropertyGraph":
        """从规范内容行（to_content 的输出）重建图。"""
        data = {
            "nodes": [{"id": nid, "label": label} for nid, label in content["nodes"]],
            "edges": [
                {"id": eid, "src": src, "dst": dst, "weight": weight, "label": label}
                for eid, src, dst, weight, label in content["edges"]
            ],
        }
        return cls.from_dict(data, version=version)

    # ------------------------------------------------------------------
    # 规范内容与摘要
    # ------------------------------------------------------------------
    def to_content(self) -> dict[str, Any]:
        return {
            "nodes": [[nid, self.nodes[nid]] for nid in sorted(self.nodes)],
            "edges": [self.edges[eid].row() for eid in sorted(self.edges)],
        }

    def content_digest(self) -> str:
        return digest(self.to_content())

    # ------------------------------------------------------------------
    # 变更集应用（强校验、整体拒绝、版本单调递增）
    # ------------------------------------------------------------------
    def apply_changeset(self, changeset: dict[str, Any]) -> ChangeRecord:
        if not isinstance(changeset, dict):
            raise GraphError(
                "changeset must be a JSON object",
                [_err("changeset", "type", "expected object")],
            )
        cs_id = changeset.get("id")
        ops = changeset.get("ops")
        errors: list[dict[str, str]] = []
        if not isinstance(cs_id, str) or not cs_id:
            errors.append(_err("changeset", "id", "changeset id must be a non-empty string"))
        if not isinstance(ops, list):
            errors.append(_err("changeset", "ops", "ops must be a list"))
            ops = []

        nodes = dict(self.nodes)
        edges = dict(self.edges)
        ops_detail: list[dict[str, Any]] = []
        structural_nodes: set[int] = set()
        structural = False

        for pos, op in enumerate(ops):
            elem = f"ops[{pos}]"
            if not isinstance(op, dict):
                errors.append(_err(elem, "type", "expected object"))
                continue
            kind = op.get("op")
            if kind == "add_node":
                nid, label = op.get("node"), op.get("label")
                if not _is_int(nid):
                    errors.append(_err(elem, "node", "node id must be an integer"))
                    continue
                if nid in nodes:
                    errors.append(_err(elem, "node", f"node {nid} already exists"))
                    continue
                if not isinstance(label, str) or not label:
                    errors.append(_err(elem, "label", "node label must be a non-empty string"))
                    continue
                nodes[nid] = label
                structural = True
                structural_nodes.add(nid)
                ops_detail.append({"op": kind, "node": nid, "label": label})
            elif kind == "remove_node":
                nid = op.get("node")
                if not _is_int(nid):
                    errors.append(_err(elem, "node", "node id must be an integer"))
                    continue
                if nid not in nodes:
                    errors.append(_err(elem, "node", f"node {nid} does not exist"))
                    continue
                incident = sorted(eid for eid, e in edges.items() if e.src == nid or e.dst == nid)
                if incident:
                    errors.append(
                        _err(elem, "edges", f"node {nid} still has incident edges {incident}; remove them first")
                    )
                    continue
                old_label = nodes.pop(nid)
                structural = True
                structural_nodes.add(nid)
                ops_detail.append({"op": kind, "node": nid, "old_label": old_label})
            elif kind == "add_edge":
                eid = op.get("edge")
                src, dst = op.get("src"), op.get("dst")
                weight, label = op.get("weight"), op.get("label")
                ok = True
                if not isinstance(eid, str) or not eid:
                    errors.append(_err(elem, "edge", "edge id must be a non-empty string"))
                    ok = False
                elif eid in edges:
                    errors.append(_err(elem, "edge", f"edge {eid!r} already exists"))
                    ok = False
                for name, value in (("src", src), ("dst", dst)):
                    if not _is_int(value):
                        errors.append(_err(elem, name, "endpoint must be an integer node id"))
                        ok = False
                    elif value not in nodes:
                        errors.append(_err(elem, name, f"endpoint node {value} does not exist"))
                        ok = False
                if not _is_int(weight):
                    errors.append(_err(elem, "weight", "edge weight must be an integer"))
                    ok = False
                if not isinstance(label, str) or not label:
                    errors.append(_err(elem, "label", "edge label must be a non-empty string"))
                    ok = False
                if not ok:
                    continue
                edges[eid] = Edge(eid, src, dst, weight, label)
                structural = True
                structural_nodes.update((src, dst))
                ops_detail.append(
                    {"op": kind, "edge": eid, "src": src, "dst": dst, "weight": weight, "label": label}
                )
            elif kind == "remove_edge":
                eid = op.get("edge")
                if not isinstance(eid, str) or not eid:
                    errors.append(_err(elem, "edge", "edge id must be a non-empty string"))
                    continue
                edge = edges.get(eid)
                if edge is None:
                    errors.append(_err(elem, "edge", f"edge {eid!r} does not exist"))
                    continue
                del edges[eid]
                structural = True
                structural_nodes.update((edge.src, edge.dst))
                ops_detail.append(
                    {
                        "op": kind,
                        "edge": eid,
                        "src": edge.src,
                        "dst": edge.dst,
                        "old_weight": edge.weight,
                        "label": edge.label,
                    }
                )
            elif kind == "set_edge_weight":
                eid, weight = op.get("edge"), op.get("weight")
                if not isinstance(eid, str) or not eid:
                    errors.append(_err(elem, "edge", "edge id must be a non-empty string"))
                    continue
                edge = edges.get(eid)
                if edge is None:
                    errors.append(_err(elem, "edge", f"edge {eid!r} does not exist"))
                    continue
                if not _is_int(weight):
                    errors.append(_err(elem, "weight", "edge weight must be an integer"))
                    continue
                edges[eid] = Edge(edge.id, edge.src, edge.dst, weight, edge.label)
                structural = True
                structural_nodes.update((edge.src, edge.dst))
                ops_detail.append(
                    {
                        "op": kind,
                        "edge": eid,
                        "src": edge.src,
                        "dst": edge.dst,
                        "old_weight": edge.weight,
                        "new_weight": weight,
                    }
                )
            elif kind == "set_node_label":
                nid, label = op.get("node"), op.get("label")
                if not _is_int(nid):
                    errors.append(_err(elem, "node", "node id must be an integer"))
                    continue
                if nid not in nodes:
                    errors.append(_err(elem, "node", f"node {nid} does not exist"))
                    continue
                if not isinstance(label, str) or not label:
                    errors.append(_err(elem, "label", "node label must be a non-empty string"))
                    continue
                old_label = nodes[nid]
                nodes[nid] = label
                ops_detail.append({"op": kind, "node": nid, "old_label": old_label, "new_label": label})
            else:
                errors.append(_err(elem, "op", f"unknown op {kind!r}"))

        if errors:
            raise GraphError(f"changeset {cs_id!r} rejected as a whole", errors)

        record = ChangeRecord(
            changeset_id=cs_id,
            version_before=self.version,
            version_after=self.version + 1,
            digest_before=self.content_digest(),
            digest_after="",
            ops_detail=ops_detail,
            structural_nodes=sorted(structural_nodes),
            structural=structural,
        )
        self.nodes = nodes
        self.edges = edges
        self.version += 1
        record.digest_after = self.content_digest()
        return record
