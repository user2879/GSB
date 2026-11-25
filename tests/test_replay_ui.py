from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from edgecache.dashboard import dashboard_snapshot
from edgecache.scheduler import Coordinator
from edgecache.spec import load_manifest

from tests.helpers import free_ports, manifest_data, write_manifest


class DashboardTests(unittest.TestCase):
    def test_refresh_is_stable_and_read_only(self) -> None:
        ports = free_ports(1)
        events = [
            {"seq": 1, "type": "apply", "node_id": "n1", "content_key": "clip", "shards": [0]},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_manifest(
                root / "ui.json",
                manifest_data(ports, ttl_ms=5000, events=events, node_count=1, slots=1, lease_limit=1),
            )
            db = root / "ui.sqlite"
            Coordinator(load_manifest(path), db, sample_path=path).run()
            before = hashlib.sha256(db.read_bytes()).hexdigest()
            first = dashboard_snapshot(db)
            second = dashboard_snapshot(db)
            after = hashlib.sha256(db.read_bytes()).hexdigest()
            self.assertEqual(first, second)
            self.assertEqual(before, after)
            self.assertEqual(len(first["leases"]), 1)
            self.assertIn("stale_views", first)
            self.assertIn("unknown_leases", first)
            self.assertIn("batches", first)


if __name__ == "__main__":
    unittest.main()
