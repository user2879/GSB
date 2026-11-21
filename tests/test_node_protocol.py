from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from replica_repair.errors import BlockError, IntegrityError
from replica_repair.node_service import NodeClient, NodeServer
from replica_repair.sample import init_sample, sample_manifest_path
from replica_repair.util import atomic_write_bytes
class NodeProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        init_sample(self.root)
        self.manifest = sample_manifest_path(self.root, "interop-v3")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_digest_data_pull_restart_and_integrity_checks(self) -> None:
        server = NodeServer(self.root, self.manifest, "v3-a", port=0).start()
        try:
            self.assertGreater(server.port, 1024)
            client = NodeClient(server.host, server.port)
            ping = client.ping()
            self.assertEqual(ping["node_id"], "v3-a")
            digest = client.call("get_digest", block_id="blk-0001")
            block = client.call("get_block", block_id="blk-0001")
            self.assertEqual(digest["actual_digest"], block["actual_digest"])
            self.assertEqual(digest["size_bytes"], 4096)
        finally:
            server.stop()

        restarted = NodeServer(self.root, self.manifest, "v3-a", port=0).start()
        try:
            client = NodeClient(restarted.host, restarted.port)
            self.assertEqual(
                client.call("get_digest", block_id="blk-0001")["actual_digest"],
                digest["actual_digest"],
            )
            path = self.root / "nodes" / "v3-a" / "blocks" / "blk-0001.bin"
            original = path.read_bytes()
            atomic_write_bytes(path, original[:-1])
            with self.assertRaises(BlockError) as caught:
                client.call("get_digest", block_id="blk-0001")
            self.assertEqual(caught.exception.code, "E_BLOCK")

            corrupted = bytes([original[0] ^ 0xFF]) + original[1:]
            atomic_write_bytes(path, corrupted)
            with self.assertRaises(IntegrityError) as caught:
                client.fetch_observation(
                    "blk-0001",
                    4096,
                    digest["manifest_digest"],
                    strict_manifest=True,
                )
            self.assertEqual(caught.exception.code, "E_INTEGRITY")
            observation = client.fetch_observation(
                "blk-0001",
                4096,
                digest["manifest_digest"],
                strict_manifest=False,
            )
            self.assertTrue(observation.manifest_mismatch)
            atomic_write_bytes(path, original)
        finally:
            restarted.stop()

    def test_put_block_is_atomic_and_leaves_no_temporary_file(self) -> None:
        server = NodeServer(self.root, self.manifest, "v3-a", port=0).start()
        try:
            client = NodeClient(server.host, server.port)
            source_path = self.root / "nodes" / "v3-b" / "blocks" / "blk-0001.bin"
            data = source_path.read_bytes()
            digest = client.call("get_digest", block_id="blk-0001")["manifest_digest"]
            client.put_block("blk-0001", data, digest)
            directory = self.root / "nodes" / "v3-a" / "blocks"
            self.assertEqual([path.name for path in directory.glob("*.tmp")], [])
            self.assertEqual(client.call("get_digest", block_id="blk-0001")["actual_digest"], digest)
        finally:
            server.stop()

    def test_serve_node_process_restarts_from_persisted_files(self) -> None:
        port_file = self.root / "node.port.json"
        arguments = [
            "serve-node",
            "--root",
            str(self.root),
            "--manifest",
            str(self.manifest),
            "--node",
            "v3-a",
            "--port",
            "0",
            "--port-file",
            str(port_file),
        ]
        import os
        import subprocess
        import sys

        from tests.helpers import REPO_ROOT
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        first = subprocess.Popen(
            [sys.executable, "-m", "replica_repair", *arguments],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            port = self._wait_for_port(port_file)
            client = NodeClient("127.0.0.1", port)
            before = client.call("get_digest", block_id="blk-0001")["actual_digest"]
            self.assertEqual(client.ping()["node_id"], "v3-a")
        finally:
            first.kill()
            first.wait(timeout=10)
        port_file.unlink(missing_ok=True)
        second = subprocess.Popen(
            [sys.executable, "-m", "replica_repair", *arguments],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            port2 = self._wait_for_port(port_file)
            client = NodeClient("127.0.0.1", port2)
            after = client.call("get_digest", block_id="blk-0001")["actual_digest"]
            self.assertEqual(before, after)
        finally:
            second.kill()
            second.wait(timeout=10)

    @staticmethod
    def _wait_for_port(path: Path) -> int:
        deadline = time.time() + 10
        while time.time() < deadline:
            if path.exists():
                try:
                    return int(json.loads(path.read_text())["port"])
                except (json.JSONDecodeError, KeyError):
                    pass
            time.sleep(0.05)
        raise AssertionError("node did not publish its port")


if __name__ == "__main__":
    unittest.main()
