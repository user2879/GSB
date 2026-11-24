from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from matgen.cache import CacheStore
from matgen.engine import MaterialEngine
from matgen.injection import InjectionRegistry, load_injection_config
from matgen.interactive import InteractiveRunner
from matgen.loader import input_fingerprint, load_input
from matgen.snapshot import SnapshotManager

from tests.helpers import BASIC, INJECTED


class InteractiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.basic = load_input(BASIC)
        cls.injected = load_input(INJECTED)

    def test_basic_responses_are_fresh_within_budget(self) -> None:
        cache = CacheStore()
        engine = MaterialEngine(cache, InjectionRegistry())
        runner = InteractiveRunner(
            engine=engine,
            textures=self.basic.textures,
            input_config_hash=input_fingerprint(self.basic),
            commands=self.basic.commands,
        )
        output = runner.run(self.basic.config, self.basic.materials)
        self.assertEqual(output["stale_responses"], 0)
        self.assertEqual(output["within_budget_responses"], 6)
        for response in output["responses"]:
            self.assertFalse(response["stale"])
            self.assertEqual(response["completed_layers"], 4)
            self.assertEqual(response["progress"], 1.0)
            self.assertIn("final_channel_summary", response)
            self.assertLessEqual(response["elapsed_ms"], response["latency_budget_ms"])

    def test_slow_injection_returns_stale_and_keeps_queue(self) -> None:
        injections, issues = load_injection_config(INJECTED / "injection_config.json")
        self.assertEqual(issues, [])
        registry = InjectionRegistry(injections)
        for issue in self.injected.issues:
            registry.record_observed(issue.code, target=issue.item, stage="load")
        engine = MaterialEngine(CacheStore(), registry)
        runner = InteractiveRunner(
            engine=engine,
            textures=self.injected.textures,
            input_config_hash=input_fingerprint(self.injected),
            commands=self.injected.commands,
        )
        output = runner.run(self.injected.config, self.injected.materials)
        self.assertEqual(output["stale_responses"], 6)
        self.assertEqual(output["within_budget_responses"], 0)
        self.assertTrue(all(response["stale"] for response in output["responses"]))
        self.assertTrue(all("E_RESPONSE_STALE" in response["codes"] for response in output["responses"]))
        self.assertEqual(output["final_revision"], 6)
        self.assertIsNotNone(output["final_result"])
        events = {event["id"]: event for event in registry.events()}
        self.assertTrue(events["INJ-LAYER-001"]["matched"])
        self.assertTrue(events["INJ-MAT-001"]["matched"])

    def test_snapshot_resume_processes_only_remaining_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root = Path(temporary) / "snapshots"
            first_manager = SnapshotManager(snapshot_root)
            first_engine = MaterialEngine(CacheStore(), InjectionRegistry())
            first_runner = InteractiveRunner(
                engine=first_engine,
                textures=self.basic.textures,
                input_config_hash=input_fingerprint(self.basic),
                commands=self.basic.commands,
                snapshot_manager=first_manager,
            )
            first_output = first_runner.run(self.basic.config, self.basic.materials, stop_after=3)
            self.assertEqual(first_manager.load_manifest()["command_cursor"], 3)

            second_manager = SnapshotManager(snapshot_root)
            second_engine = MaterialEngine(CacheStore(), InjectionRegistry())
            second_runner = InteractiveRunner(
                engine=second_engine,
                textures=self.basic.textures,
                input_config_hash=input_fingerprint(self.basic),
                commands=self.basic.commands,
                snapshot_manager=second_manager,
                resume=True,
            )
            second_output = second_runner.run(self.basic.config, self.basic.materials)
            self.assertTrue(second_output["resume"]["usable"])
            self.assertEqual(second_output["commands_started_at"], 3)
            self.assertEqual(second_output["commands_processed"], 3)

            full_engine = MaterialEngine(CacheStore(), InjectionRegistry())
            full_runner = InteractiveRunner(
                engine=full_engine,
                textures=self.basic.textures,
                input_config_hash=input_fingerprint(self.basic),
                commands=self.basic.commands,
            )
            full_output = full_runner.run(self.basic.config, self.basic.materials)
            self.assertEqual(second_output["final_result"].final_digest, full_output["final_result"].final_digest)


if __name__ == "__main__":
    unittest.main()
