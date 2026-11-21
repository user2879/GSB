from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from replica_repair.engine import run_pipeline
from replica_repair.ledger import Ledger
from replica_repair.manifest_repo import ManifestRepository
from replica_repair.model import ChecksumDerivation, Manifest, RELATION_ALGORITHM, derive_relation_digest
from replica_repair.planner import RepairBudget
from replica_repair.sample import init_sample, sample_ledger_path, sample_manifest_path
from tests.helpers import group_servers


class EvolutionTests(unittest.TestCase):
    def test_derivation_and_version_changes_stale_previous_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_sample(root)
            manifest_path = sample_manifest_path(root, "interop-v3")
            ledger_path = sample_ledger_path(root, "interop-v3")
            with group_servers(root) as (manifest, clients, _servers):
                run_pipeline(
                    root=root,
                    manifest_path=manifest_path,
                    ledger_path=ledger_path,
                    clients=clients,
                    budget=RepairBudget(3, 3),
                )
            current = Manifest.from_file(manifest_path)
            previous_version = current.manifest_version
            nodes = []
            for node in current.nodes:
                blocks = node.block_map()
                source_digest = blocks["blk-0001"].digest
                relation = ChecksumDerivation(
                    algorithm=RELATION_ALGORITHM,
                    source_block_ids=("blk-0001",),
                    digest=derive_relation_digest(
                        RELATION_ALGORITHM,
                        ("blk-0001",),
                        {"blk-0001": source_digest},
                    ),
                )
                blocks["blk-0003"] = replace(blocks["blk-0003"], checksum_derivation=relation)
                nodes.append(
                    replace(
                        node,
                        blocks=tuple(blocks[block_id] for block_id in sorted(blocks)),
                    )
                )
            result = ManifestRepository(manifest_path).save_current(
                replace(current, nodes=tuple(nodes)),
                recompute_relations=True,
            )
            self.assertTrue(result.version_incremented)
            self.assertEqual(result.manifest.manifest_version, previous_version + 1)
            self.assertEqual(
                set(result.changes.affected_blocks),
                {"blk-0001", "blk-0002", "blk-0003"},
            )
            self.assertEqual(
                set(result.changes.affected_nodes),
                {"v3-a", "v3-b", "v3-c", "v3-d"},
            )

            with Ledger(ledger_path) as ledger:
                recovery = ledger.reopen_from_disk()
                self.assertIsNotNone(recovery)
                self.assertTrue(recovery.replaced)
                self.assertIsNone(ledger.current_report()["current"])
                self.assertIn("blk-0003", recovery.affected_blocks)
                self.assertEqual(
                    set(recovery.affected_nodes),
                    {"v3-a", "v3-b", "v3-c", "v3-d"},
                )

            # A pure manifest-version advance also invalidates all old conclusions.
            latest = Manifest.from_file(manifest_path)
            version_only = ManifestRepository(manifest_path).save_current(
                replace(latest, manifest_version=latest.manifest_version + 1)
            )
            self.assertEqual(
                {item["reason"] for item in version_only.changes.details},
                {"manifest_version"},
            )
            with Ledger(ledger_path) as ledger:
                recovery = ledger.reopen_from_disk()
                self.assertIsNotNone(recovery)
                self.assertTrue(recovery.replaced)
                self.assertEqual(
                    set(recovery.affected_blocks),
                    {"blk-0001", "blk-0002", "blk-0003"},
                )


if __name__ == "__main__":
    unittest.main()
