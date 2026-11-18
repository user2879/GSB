"""链路拓扑：节点、链路与流到链路的映射。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Link:
    link_id: str
    src_node: str
    dst_node: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "src_node": self.src_node,
            "dst_node": self.dst_node,
            "metadata": self.metadata,
        }


@dataclass
class Topology:
    name: str = "local-loopback"
    links: Dict[str, Link] = field(default_factory=dict)
    flow_links: Dict[str, str] = field(default_factory=dict)

    def add_link(self, link: Link) -> None:
        self.links[link.link_id] = link

    def link_for_flow(self, flow: str) -> Optional[str]:
        return self.flow_links.get(flow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "links": {k: v.to_dict() for k, v in self.links.items()},
            "flow_links": dict(self.flow_links),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Topology":
        topo = cls(name=data.get("name", "local-loopback"))
        for item in data.get("links", {}).values():
            topo.add_link(Link(**item))
        topo.flow_links = dict(data.get("flow_links", {}))
        return topo

    @classmethod
    def default_for_flows(cls, flows: List[str]) -> "Topology":
        """为演示生成确定性的单链路拓扑：所有流走 L1。"""
        topo = cls(name="local-loopback")
        topo.add_link(Link("L1", "host-a", "host-b",
                           {"kind": "loopback", "deterministic": True}))
        for flow in flows:
            topo.flow_links[flow] = "L1"
        return topo
