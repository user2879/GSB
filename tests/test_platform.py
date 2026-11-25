import dataclasses
import tempfile
import unittest
from pathlib import Path

from arbplatform.platform import ArbitrationPlatform
from arbplatform.samplegen import generate_observations
from arbplatform.spec import load_manifest

MANIFEST = Path(__file__).resolve().parent.parent / "config" / "manifest.json"


class PlatformIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "ledger.sqlite"
        self.manifest = load_manifest(MANIFEST)
        self.platform = ArbitrationPlatform(self.manifest, self.db)
        self.platform.start_services()
        events = generate_observations(self.manifest)
        self.platform.ingest([o.to_event() for o in events], batch_size=5)

    def tearDown(self):
        self.platform.shutdown()
        self.tmp.cleanup()

    def test_full_run_is_deterministic_across_batch_sizes(self):
        first = self.platform.run_view("view-all", use_scheduler=False)
        digest_a = first.result.content_digest

        shifted = dataclasses.replace(
            self.manifest,
            environments=tuple(
                dataclasses.replace(env, port=env.port + 1000)
                for env in self.manifest.environments))
        platform_b = ArbitrationPlatform(
            shifted, Path(self.tmp.name) / "b.sqlite")
        platform_b.start_services()
        events = generate_observations(self.manifest)
        platform_b.ingest([o.to_event() for o in events], batch_size=13)
        second = platform_b.run_view("view-all", use_scheduler=False)
        platform_b.shutdown()
        self.assertEqual(digest_a, second.result.content_digest)

    def test_incremental_preserves_unaffected_keys_byte_identical(self):
        full = self.platform.run_view("view-all", use_scheduler=False)
        before = {k: v.to_record() for k, v in full.result.verdicts.items()}

        # Replace one observation affecting only cache.ttl in env-alpha.
        target = next(
            oid for oid, o in self.platform.store.items()
            if o["env_id"] == "env-alpha" and o["key"] == "cache.ttl"
            and o["seq"] == 6
        )
        change = self.platform.replace_observation(target, 999)
        report = self.platform.run_view(
            "view-all", changes=[change], use_scheduler=True)

        self.assertEqual(report.path_decision["parameters"]["affected_key_count"], 1)
        after = {k: v.to_record() for k, v in
                 report.result.verdicts.items()}
        # Unaffected keys remain byte-identical.
        for key in ("pool.size", "rate.window"):
            self.assertEqual(before[key], after[key])
        self.assertNotEqual(before["cache.ttl"], after["cache.ttl"])
        self.assertEqual(report.result.view_version, 2)

    def test_force_kill_recovery_and_replay(self):
        report = self.platform.run_view("view-all", use_scheduler=False)
        committed_digest = report.result.content_digest

        # Open a pending (uncommitted) view, then force-kill its service.
        self.platform.ledger.begin_view("view-alpha-bravo", 1, 0)
        exit_code = self.platform.force_kill_service("env-bravo")
        self.assertNotEqual(exit_code, 0)
        recovery = self.platform.recover_after_kill()
        self.assertIn("view-alpha-bravo", recovery["unknown_views"])

        restart = self.platform.restart_service("env-bravo")
        self.assertGreater(restart["repushed"], 0)

        rerun = self.platform.run_view("view-alpha-bravo", use_scheduler=False)
        self.assertIsNotNone(rerun.result.content_digest)

        outcome = self.platform.replay_ledger()
        total_keys = sum(len(v.policies) for v in self.manifest.views)
        self.assertEqual(outcome.matched, total_keys)
        # The previously committed view is still exactly reproducible.
        self.assertEqual(
            self.platform.last_result("view-all").content_digest,
            committed_digest)

    def test_scheduler_resources_reported(self):
        target = next(
            oid for oid, o in self.platform.store.items()
            if o["key"] == "pool.size"
        )
        change = self.platform.replace_observation(target, 42)
        report = self.platform.run_view(
            "view-all", changes=[change], use_scheduler=True,
            force_path="incremental")
        scheduling = report.scheduling
        self.assertIsNotNone(scheduling)
        self.assertLessEqual(
            scheduling["max_global_inflight"],
            self.manifest.global_concurrency)
        for env in self.manifest.environments:
            self.assertLessEqual(
                scheduling["max_per_env_inflight"][env.env_id],
                env.lease_capacity)


if __name__ == "__main__":
    unittest.main()
