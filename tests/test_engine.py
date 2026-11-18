import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tier_mover.engine import Engine
from tier_mover.errors import CapacityError, IntegrityError, ReplayError, VersionConflictError
from tier_mover.faults import Injection
from tier_mover.ledger import Ledger
from tier_mover.logging_util import JsonlLogger
from tier_mover.manifest import load_manifest
from tier_mover.replay import replay
from tier_mover.samplegen import build_sample
from tier_mover.storage import staging_dir


class EngineTestBase(unittest.TestCase):
    def make_workdir(self, name: str, **kwargs) -> Path:
        workdir = Path(self.tmp.name) / name
        build_sample(workdir, **kwargs)
        return workdir

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def engine(self, workdir: Path, injection: Injection | None = None) -> Engine:
        manifest = load_manifest(workdir / "manifest.json")
        logger = JsonlLogger(workdir / "logs" / "test.jsonl")
        self.addCleanup(logger.close)
        return Engine(workdir, manifest, logger, injection)

    def assert_within_capacity(self, workdir: Path) -> None:
        ledger = Ledger.open(workdir)
        try:
            for row in ledger.tier_rows():
                self.assertLessEqual(row["used_bytes"], row["capacity_bytes"])
                self.assertEqual(row["free_bytes"],
                                 row["capacity_bytes"] - row["used_bytes"])
        finally:
            ledger.close()


class HappyPathTest(EngineTestBase):
    def test_initial_then_cycle_then_replay(self) -> None:
        workdir = self.make_workdir(
            "happy", hot_capacity=16384, cold_capacity=65536,
            hot_watermark=12288, cold_reclaim=32768, batch_size=4,
        )
        engine = self.engine(workdir)
        engine.run_initial()
        engine.run_cycle()
        self.assert_within_capacity(workdir)
        result = replay(str(workdir), load_manifest(workdir / "manifest.json"))
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["migrations_replayed"], 0)

    def test_deterministic_conclusions(self) -> None:
        def build(name: str) -> bytes:
            workdir = self.make_workdir(name)
            self.engine(workdir).run_initial()
            self.engine(workdir).run_cycle()
            return (workdir / "cycle_conclusions.json").read_bytes()
        self.assertEqual(build("da"), build("db"))


class FaultInjectionTest(EngineTestBase):
    def test_capacity_injection(self) -> None:
        workdir = self.make_workdir("cap")
        engine = self.engine(workdir, Injection(kind="capacity", capacity_override=1))
        with self.assertRaises(CapacityError) as ctx:
            engine.run_initial()
        self.assertEqual(ctx.exception.code, "E_CAPACITY")

    def test_truncate_injection(self) -> None:
        workdir = self.make_workdir("trunc")
        target = load_manifest(workdir / "manifest.json").objects[0]
        path = staging_dir(workdir) / target.digest
        path.write_bytes(path.read_bytes()[: max(1, target.size_bytes // 2)])
        engine = self.engine(workdir)
        with self.assertRaises(IntegrityError) as ctx:
            engine.run_initial()
        self.assertEqual(ctx.exception.code, "E_INTEGRITY")
        self.assertEqual(ctx.exception.context["key"], target.key)

    def test_version_injection(self) -> None:
        from dataclasses import replace
        workdir = self.make_workdir("ver")
        self.engine(workdir).run_initial()
        manifest = load_manifest(workdir / "manifest.json")
        changed = replace(manifest, policy_version="policy-X")
        logger = JsonlLogger(workdir / "logs" / "ver.jsonl")
        self.addCleanup(logger.close)
        engine = Engine(workdir, changed, logger)
        with self.assertRaises(VersionConflictError) as ctx:
            engine.run_cycle()
        self.assertEqual(ctx.exception.code, "E_VERSION")


class CrashRecoveryTest(EngineTestBase):
    def _run_cli(self, workdir: Path, *args: str) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-m", "tier_mover.cli", *args]
        return subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True)

    def test_kill_recover_replay(self) -> None:
        workdir = self.make_workdir(
            "kill", hot_capacity=12288, hot_watermark=8192,
            cold_reclaim=40960, batch_size=4,
        )
        manifest_arg = str(workdir / "manifest.json")
        self.assertEqual(self._run_cli(workdir, "initial", str(workdir),
                                       "--manifest", manifest_arg).returncode, 0)
        killed = self._run_cli(
            workdir, "cycle", str(workdir), "--manifest", manifest_arg,
            "--inject", "kill", "--kill-after-commit", "1",
        )
        self.assertEqual(killed.returncode, 9)

        ledger = Ledger.open(workdir)
        try:
            self.assertTrue(ledger.in_flight_objects())
        finally:
            ledger.close()

        recovered = self._run_cli(workdir, "recover", str(workdir),
                                  "--manifest", manifest_arg)
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        report = json.loads((workdir / "recovery_report.json").read_text())
        self.assertEqual(len(report["resolved_unknown_objects"]), 1)
        self.assertTrue(report["reran_migrations"])

        result = replay(str(workdir), load_manifest(workdir / "manifest.json"))
        self.assertEqual(result["status"], "ok")


class ReplayMismatchTest(EngineTestBase):
    def test_changed_digest_raises_replay(self) -> None:
        workdir = self.make_workdir("replay")
        self.engine(workdir).run_initial()
        self.engine(workdir).run_cycle()
        manifest = load_manifest(workdir / "manifest.json")
        from dataclasses import replace
        first = manifest.objects[0]
        tampered = replace(first, digest="0" * 64)
        objects = [tampered if o.key == first.key else o for o in manifest.objects]
        changed = replace(manifest, objects=objects)
        with self.assertRaises(ReplayError) as ctx:
            replay(str(workdir), changed)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.context["first_mismatch_key"], first.key)


if __name__ == "__main__":
    unittest.main()
