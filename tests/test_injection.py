from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from matgen.cache import CacheStore
from matgen.engine import MaterialEngine
from matgen.injection import InjectionRegistry, load_injection_config
from matgen.loader import input_fingerprint, load_input
from matgen.snapshot import SnapshotManager, phase_hashes_from_result

from tests.helpers import BASIC, INJECTED


class InjectionTests(unittest.TestCase):
    def test_expected_code_mapping_for_load_anomalies(self) -> None:
        loaded = load_input(INJECTED)
        injections, issues = load_injection_config(INJECTED / "injection_config.json")
        self.assertEqual(issues, [])
        registry = InjectionRegistry(injections)
        for issue in loaded.issues:
            registry.record_observed(issue.code, target=issue.item, stage="load")
        events = {event["id"]: event for event in registry.events()}
        for injection_id in ("INJ-LOAD-001", "INJ-LOAD-002", "INJ-LOAD-003"):
            self.assertTrue(events[injection_id]["matched"])

    def test_runtime_material_failure_is_isolated(self) -> None:
        loaded = load_input(INJECTED)
        injections, _ = load_injection_config(INJECTED / "injection_config.json")
        registry = InjectionRegistry(injections)
        result = MaterialEngine(CacheStore(), registry).build(
            loaded.config,
            loaded.materials,
            loaded.textures,
        )
        codes = {issue.code for issue in result.issues}
        self.assertIn("E_MATERIAL_RUNTIME", codes)
        self.assertIn("brushed_steel", result.material_artifacts)
        self.assertNotIn("painted_plastic", result.material_artifacts)
        events = {event["id"]: event for event in registry.events()}
        self.assertTrue(events["INJ-MAT-001"]["matched"])

    def test_snapshot_version_mismatch_injection_mapping(self) -> None:
        loaded = load_input(BASIC)
        injections, _ = load_injection_config(INJECTED / "injection_config.json")
        registry = InjectionRegistry(injections)
        with tempfile.TemporaryDirectory() as temporary:
            manager = SnapshotManager(Path(temporary) / "snapshots")
            cache = CacheStore()
            result = MaterialEngine(cache, InjectionRegistry()).build(
                loaded.config,
                loaded.materials,
                loaded.textures,
            )
            manager.commit(
                cache=cache,
                config=loaded.config,
                input_config_hash=input_fingerprint(loaded),
                commands_hash="commands-hash",
                command_cursor=0,
                completed_batches_cursor=result.batches_consumed,
                phase_hashes=phase_hashes_from_result(result),
                ledger=result.ledger_entries,
            )
            usable, _, issues = manager.check_resume(
                input_config_hash=input_fingerprint(loaded),
                commands_hash="commands-hash",
                expected_version=999,
            )
            self.assertFalse(usable)
            self.assertIn("E_SNAPSHOT_VERSION", {issue.code for issue in issues})
            registry.record_code_for_action(
                "E_SNAPSHOT_VERSION",
                action="snapshot_hash_mismatch",
                stage="snapshot",
            )
        events = {event["id"]: event for event in registry.events()}
        self.assertTrue(events["INJ-SNAP-001"]["matched"])


if __name__ == "__main__":
    unittest.main()
