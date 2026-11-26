from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from gsb.errors import DivergeError
from gsb.reconcile import compare_outputs, run_evolution_check
from gsb.spec import load_spec

from tests.factories import free_tcp_port


def evolution_spec(port: int, *, changed: bool = False) -> dict:
    shards = []
    for index in range(3):
        shards.append(
            {
                "shard_id": f"s{index}",
                "node_id": "n1",
                "dependencies": [] if index == 0 else [f"s{index - 1}"],
                "rows": 16 if changed and index == 0 else 12,
                "resource_units": 1,
                "priority": 10 - index,
            }
        )
    return {
        "session_id": "evolution-test",
        "scale_factor": 100,
        "budget": {"resource_units": 10},
        "scheduling": {
            "lease_ttl_ms": 10_000,
            "request_timeout_ms": 10_000,
            "handshake_timeout_ms": 10_000,
            "poll_interval_ms": 5,
        },
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": port,
                "concurrency": 2,
                "lease_limit": 3,
                "seed": 99,
                "capacity_version": 1,
                "fixed_scale": 100,
                "processing_delay_ms": 2,
            }
        ],
        "shards": shards,
    }


class ReconcileTests(unittest.TestCase):
    def test_incremental_matches_blank_full_and_preserves_unaffected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port = free_tcp_port()
            old_path = root / "old.json"
            new_path = root / "new.json"
            old_path.write_text(json.dumps(evolution_spec(port)), encoding="utf-8")
            new_path.write_text(json.dumps(evolution_spec(port, changed=True)), encoding="utf-8")

            report = run_evolution_check(old_spec_path=old_path, new_spec_path=new_path, work_dir=root / "evo")
            self.assertEqual(report["invalid_set"], ["s0", "s1", "s2"])
            self.assertTrue(all(item["byte_identical"] for item in report["unaffected_byte_checks"]))
            self.assertEqual(len(report["comparisons"]), 3)

            # There are no unaffected shards in a chain, so add a direct
            # E_DIVERGE probe by corrupting one compared water mark.
            new_spec = load_spec(new_path)
            inc_db = Path(report["incremental_db"])
            full_db = Path(report["full_db"])
            with sqlite3.connect(inc_db) as conn:
                conn.execute("UPDATE outcomes SET water_q=water_q+1 WHERE shard_id='s1'")
                conn.commit()
            with self.assertRaises(DivergeError) as ctx:
                compare_outputs(new_spec, inc_db, full_db)
            self.assertEqual(ctx.exception.code, "E_DIVERGE")
            self.assertIn("s1.water_q", ctx.exception.field or "")


if __name__ == "__main__":
    unittest.main()
