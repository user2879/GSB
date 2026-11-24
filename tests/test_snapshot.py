from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from matgen.cache import CacheStore
from matgen.engine import MaterialEngine
from matgen.errors import E_SNAPSHOT_VERSION
from matgen.injection import InjectionRegistry
from matgen.loader import input_fingerprint, load_input
from matgen.snapshot import SnapshotManager, phase_hashes_from_result

from tests.helpers import BASIC


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_input(BASIC)
        cls.input_hash = input_fingerprint(cls.loaded)

    def _commit(self, manager: SnapshotManager, cache: CacheStore, result):
        return manager.commit(
            cache=cache,
            config=self.loaded.config,
            input_config_hash=self.input_hash,
            commands_hash="commands-hash",
            command_cursor=0,
            completed_batches_cursor=result.batches_consumed,
            phase_hashes=phase_hashes_from_result(result),
            ledger=result.ledger_entries,
        )

    def test_restore_cache_avoids_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = SnapshotManager(Path(temporary) / "snapshots")
            cache = CacheStore()
            first = MaterialEngine(cache, InjectionRegistry()).build(
                self.loaded.config, self.loaded.materials, self.loaded.textures
            )
            commit = self._commit(manager, cache, first)
            self.assertEqual(commit["status"], "committed")

            new_cache = CacheStore()
            restore_issues = manager.restore_cache(new_cache)
            self.assertEqual(restore_issues, [])
            second = MaterialEngine(new_cache, InjectionRegistry()).build(
                self.loaded.config, self.loaded.materials, self.loaded.textures
            )
            self.assertEqual(second.final_digest, first.final_digest)
            self.assertEqual(second.affected_layers, [])
            self.assertEqual(second.affected_channels, [])
            self.assertTrue(second.final_reused)

    def test_version_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = SnapshotManager(Path(temporary) / "snapshots")
            cache = CacheStore()
            result = MaterialEngine(cache, InjectionRegistry()).build(
                self.loaded.config, self.loaded.materials, self.loaded.textures
            )
            self._commit(manager, cache, result)
            usable, _, issues = manager.check_resume(
                input_config_hash=self.input_hash,
                commands_hash="commands-hash",
                expected_version=999,
            )
            self.assertFalse(usable)
            self.assertIn(E_SNAPSHOT_VERSION, {issue.code for issue in issues})
            usable, _, issues = manager.check_resume(
                input_config_hash="wrong-hash",
                commands_hash="commands-hash",
            )
            self.assertFalse(usable)
            self.assertIn(E_SNAPSHOT_VERSION, {issue.code for issue in issues})

    def test_interrupted_directory_swap_recovers_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "snapshots"
            manager = SnapshotManager(root)
            cache = CacheStore()
            result = MaterialEngine(cache, InjectionRegistry()).build(
                self.loaded.config, self.loaded.materials, self.loaded.textures
            )
            self._commit(manager, cache, result)
            self.assertIsNotNone(manager.load_manifest())
            backup = root / ".previous-simulated"
            manager.current.rename(backup)
            recovered = SnapshotManager(root)
            self.assertIsNotNone(recovered.load_manifest())
            self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()
