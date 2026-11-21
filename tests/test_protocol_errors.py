import json
import os
import tempfile
import unittest

from configbench.errors import FieldError, HandshakeError, PortError
from configbench.manifest import load_all
from configbench.platform import Platform
from configbench.transport import NodeProcess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_registry(tmp, nodes):
    path = os.path.join(tmp, "registry.json")
    with open(path, "w") as f:
        json.dump({"nodes": nodes}, f)
    return path


import itertools

# 跨测试单调递增的高位端口，避免动态 bind(0) 复用刚释放端口造成竞态。
_PORT = itertools.count(48100)


def _free_port():
    return next(_PORT)


def _node(node_id, port, fail=None):
    return {"node_id": node_id, "host": "127.0.0.1", "port": port,
            "group_key": "region", "scale": 3, "fail_inject": fail}


class ProtocolErrorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.manifest, cls.samples = load_all(
            os.path.join(ROOT, "samples/registry.json"),
            os.path.join(ROOT, "samples/manifest.json"),
            os.path.join(ROOT, "samples/samples.json"))
        cls.tmp = tempfile.mkdtemp()

    def _load(self, nodes):
        reg_path = _write_registry(self.tmp, nodes)
        from configbench.manifest import load_registry
        return load_registry(reg_path)

    def test_port_conflict_raises_e_port(self):
        registry = self._load([_node("node-alpha", _free_port(), "port")])
        proc = NodeProcess(registry.nodes["node-alpha"])
        with self.assertRaises(PortError) as ctx:
            proc.start()
        self.assertEqual(ctx.exception.code, "E_PORT")
        self.assertIn("port", ctx.exception.field)

    def test_handshake_timeout_raises_e_handshake(self):
        registry = self._load([_node("node-alpha", _free_port(), "handshake")])
        proc = NodeProcess(registry.nodes["node-alpha"]).start()
        self.addCleanup(proc.stop)
        from configbench.transport import dispatch_once
        node = registry.nodes["node-alpha"]
        req = {"type": "slice", "request_id": "r", "config_version": 1,
               "items": [{"key": "a", "value_type": "int", "value": 3,
                          "scale": 0, "targets": ["node-alpha"],
                          "derives_from": []}],
               "samples": {"seed": 1, "n_samples": 10,
                                        "n_groups": 2, "value_range": 5,
                                        "declared_scale": 0}}
        with self.assertRaises(HandshakeError) as ctx:
            dispatch_once(node, req, connect_timeout_ms=800,
                          handshake_timeout_ms=800, ack_timeout_ms=800)
        self.assertEqual(ctx.exception.code, "E_HANDSHAKE")

    def test_missing_field_raises_e_field(self):
        registry = self._load([_node("node-alpha", _free_port(), "field")])
        proc = NodeProcess(registry.nodes["node-alpha"]).start()
        self.addCleanup(proc.stop)
        from configbench.transport import dispatch_once
        node = registry.nodes["node-alpha"]
        req = {"type": "slice", "request_id": "r", "config_version": 1,
               "items": [], "samples": {"seed": 1, "n_samples": 10,
                                        "n_groups": 2, "value_range": 5,
                                        "declared_scale": 0}}
        with self.assertRaises(FieldError) as ctx:
            dispatch_once(node, req, handshake_timeout_ms=1000,
                          ack_timeout_ms=1000)
        self.assertEqual(ctx.exception.code, "E_FIELD")

    def test_failed_node_excluded_from_success_stats(self):
        # 一个健康节点 + 一个 field 故障节点；故障节点不得计入成功回执。
        registry = self._load([
            _node("node-alpha", _free_port(), None),
            _node("node-beta", _free_port(), "field"),
        ])
        # 用一个只含 node-alpha/node-beta 目标的小型清单。
        manifest = self._two_node_manifest(registry)
        p = Platform(registry, manifest, self.samples,
                     os.path.join(self.tmp, "fault.db"))
        p.start_nodes()
        self.addCleanup(p.close)
        pub = p.publish_version(1)
        self.assertEqual(pub["receipts"]["total"], 2)
        self.assertEqual(pub["receipts"]["success"], 1)
        self.assertEqual(pub["receipts"]["success_task_ids"], ["v1:node-alpha"])
        self.assertEqual(p.cache["node-beta"].progress, "FAILED")

    def _two_node_manifest(self, registry):
        from configbench.manifest import load_manifest
        path = os.path.join(self.tmp, "mini-manifest.json")
        with open(path, "w") as f:
            json.dump({
                "latency_budget_ms": 500,
                "versions": [{"version": 1, "items": {
                    "base-rate": {"value_type": "int", "value": 5,
                                  "targets": ["node-alpha", "node-beta"]}}}]}, f)
        return load_manifest(path, registry)


if __name__ == "__main__":
    unittest.main()
