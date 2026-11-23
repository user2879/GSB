import json
import socket
import tempfile
import unittest
from pathlib import Path

from diagconsole import aggregate as agg
from diagconsole.engine import Engine, NodeKey
from diagconsole.errors import DiagError
from diagconsole.ledger import Ledger
from diagconsole.model import Cluster, Spec, Task, Revision, load_spec
from diagconsole.orchestrator import Orchestrator

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def single_cluster_spec(port: int) -> Spec:
    cluster = Cluster("cluster-test", "127.0.0.1", port,
                      ("node-1", "node-2"), 20240777, 48)
    task = Task("task-test", ("rate_limit", "burst"),
                ("cluster-test",), 1)
    revisions = (
        Revision(1, None, {"rate_limit": 10, "burst": 2}, "canary", ()),
        Revision(2, 1, {"rate_limit": 20}, "rolling", (1,)),
    )
    return Spec(clusters=(cluster,), tasks=(task,), revisions=revisions)


class OrchestratorTests(unittest.TestCase):
    def test_dispatch_roundtrip_matches_local(self):
        port = free_port()
        spec = single_cluster_spec(port)
        orch = Orchestrator(spec)
        orch.start_all()
        try:
            tmpdir = tempfile.mkdtemp()
            ledger = Ledger(Path(tmpdir) / "l.db")
            engine = Engine(spec, ledger, orch)
            key = NodeKey("task-test", "cluster-test", 2)
            remote = engine.dispatch_node(key)
            local = engine.compute_node_locally(key)
            self.assertEqual(remote["aggregate"], local["aggregate"])
            self.assertEqual(remote["content_digest"], local["content_digest"])
            ledger.close()
        finally:
            orch.stop_all()

    def test_port_conflict_is_e_port(self):
        port = free_port()
        holder = socket.socket()
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        spec = single_cluster_spec(port)
        orch = Orchestrator(spec)
        try:
            with self.assertRaises(DiagError) as ctx:
                orch.start_all()
            self.assertEqual(ctx.exception.code, "E_PORT")
            self.assertEqual(ctx.exception.location["cluster"], "cluster-test")
        finally:
            holder.close()

    def test_missing_reply_field_is_e_reply(self):
        port = free_port()
        spec = single_cluster_spec(port)
        orch = Orchestrator(spec)
        orch.start_all()
        try:
            bad_request = {"type": "dispatch", "cluster_id": "cluster-test",
                           "task_id": "task-test", "version": 1}
            with self.assertRaises(DiagError) as ctx:
                orch.dispatch(bad_request)
            self.assertEqual(ctx.exception.code, "E_REPLY")
        finally:
            orch.stop_all()

    def test_handshake_timeout_is_e_handshake(self):
        class StubProcess:
            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        port = free_port()
        spec = single_cluster_spec(port)
        orch = Orchestrator(spec)
        original_popen = __import__("diagconsole.orchestrator",
                                    fromlist=["subprocess"]).subprocess.Popen

        class SilentPopen:
            def __init__(self, *args, **kwargs):
                empty = type("S", (), {"readline": lambda self: "",
                                       "close": lambda self: None})()
                self.stdout = empty
                self.stderr = empty
                self.pid = 0

            def poll(self):
                return None

            def terminate(self):
                pass

            def kill(self):
                pass

            def wait(self, timeout=None):
                return 0

        __import__("diagconsole.orchestrator",
                   fromlist=["subprocess"]).subprocess.Popen = SilentPopen
        original_timeout = __import__("diagconsole.orchestrator",
                                      fromlist=["READY_TIMEOUT_SECONDS"])
        mod = original_timeout
        old = mod.READY_TIMEOUT_SECONDS
        mod.READY_TIMEOUT_SECONDS = 0.2
        try:
            with self.assertRaises(DiagError) as ctx:
                orch.start_all()
            self.assertEqual(ctx.exception.code, "E_HANDSHAKE")
        finally:
            mod.READY_TIMEOUT_SECONDS = old
            __import__("diagconsole.orchestrator",
                       fromlist=["subprocess"]).subprocess.Popen = original_popen


if __name__ == "__main__":
    unittest.main()
