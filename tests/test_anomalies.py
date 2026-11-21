from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from replica_repair.engine import run_pipeline
from replica_repair.errors import BlockError, BudgetError, IntegrityError
from replica_repair.planner import RepairBudget
from replica_repair.sample import init_sample, sample_ledger_path, sample_manifest_path
from replica_repair.util import atomic_write_bytes
from tests.helpers import group_servers


class AnomalyTests(unittest.TestCase):
    def _root(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        init_sample(root)
        return temp, root

    def test_truncated_block_reports_e_block(self) -> None:
        temp, root = self._root()
        try:
            with group_servers(root) as (manifest, clients, _servers):
                path = root / "nodes" / "v3-a" / "blocks" / "blk-0001.bin"
                atomic_write_bytes(path, path.read_bytes()[:-1])
                with self.assertRaises(BlockError) as caught:
                    run_pipeline(
                        root=root,
                        manifest_path=sample_manifest_path(root, "interop-v3"),
                        ledger_path=sample_ledger_path(root, "interop-v3"),
                        clients=clients,
                        budget=RepairBudget(3, 3),
                    )
            self.assertEqual(caught.exception.code, "E_BLOCK")
            self.assertEqual(caught.exception.details["node_id"], "v3-a")
            self.assertEqual(caught.exception.details["block_id"], "blk-0001")
        finally:
            temp.cleanup()

    def test_digest_corruption_reports_e_integrity(self) -> None:
        temp, root = self._root()
        try:
            with group_servers(root) as (manifest, clients, _servers):
                path = root / "nodes" / "v3-b" / "blocks" / "blk-0001.bin"
                data = path.read_bytes()
                atomic_write_bytes(path, bytes([data[0] ^ 0xFF]) + data[1:])
                with self.assertRaises(IntegrityError) as caught:
                    run_pipeline(
                        root=root,
                        manifest_path=sample_manifest_path(root, "interop-v3"),
                        ledger_path=sample_ledger_path(root, "interop-v3"),
                        clients=clients,
                        budget=RepairBudget(3, 3),
                    )
            self.assertEqual(caught.exception.code, "E_INTEGRITY")
            self.assertEqual(caught.exception.details["node_id"], "v3-b")
        finally:
            temp.cleanup()

    def test_invalid_budget_reports_e_budget(self) -> None:
        temp, root = self._root()
        try:
            with group_servers(root) as (manifest, clients, _servers):
                with self.assertRaises(BudgetError) as caught:
                    run_pipeline(
                        root=root,
                        manifest_path=sample_manifest_path(root, "interop-v3"),
                        ledger_path=sample_ledger_path(root, "interop-v3"),
                        clients=clients,
                        budget=RepairBudget(0, 3),
                    )
            self.assertEqual(caught.exception.code, "E_BUDGET")
            self.assertEqual(caught.exception.path, "$.budget.max_parallel")
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
