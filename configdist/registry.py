"""节点注册表加载与校验。"""

import json

from .canon import PROFILES
from .errors import FormatError

VIEWS = ("s1", "s2")


class NodeSpec:
    def __init__(self, node_id, port, view, profile, fudge=0):
        self.id = node_id
        self.port = port
        self.view = view
        self.profile = profile
        self.fudge = fudge  # 演示/测试用：对聚合结果加固定偏移以制造偏差

    def __repr__(self):
        return f"NodeSpec({self.id!r}, port={self.port}, view={self.view!r})"


def load_registry(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    nodes = raw.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise FormatError("注册表 nodes 缺失或为空", path="nodes",
                          source_version="registry")
    specs = {}
    used_ports = set()
    for i, rec in enumerate(nodes):
        p = f"nodes[{i}]"
        node_id = rec.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise FormatError("节点 id 缺失", path=f"{p}.id",
                              source_version="registry")
        if node_id in specs:
            raise FormatError(f"节点 id 重复: {node_id!r}", path=f"{p}.id",
                              source_version="registry")
        port = rec.get("port")
        if not isinstance(port, int) or not (1024 <= port <= 65535):
            raise FormatError("端口必须为 1024-65535 的整数", path=f"{p}.port",
                              source_version="registry")
        if port in used_ports:
            raise FormatError(f"端口重复: {port}", path=f"{p}.port",
                              source_version="registry")
        used_ports.add(port)
        view = rec.get("view")
        if view not in VIEWS:
            raise FormatError(f"未知订阅视图: {view!r}", path=f"{p}.view",
                              source_version="registry")
        profile = rec.get("profile")
        if profile not in PROFILES:
            raise FormatError(f"未知聚合口径: {profile!r}", path=f"{p}.profile",
                              source_version="registry")
        fudge = rec.get("fudge", 0)
        if not isinstance(fudge, int):
            raise FormatError("fudge 必须为整数", path=f"{p}.fudge",
                              source_version="registry")
        specs[node_id] = NodeSpec(node_id, port, view, profile, fudge)
    return specs
