import json
import signal
import tempfile
import unittest
from pathlib import Path

from _helpers import base_manifest, mutate

from haptic_engine.engine import HapticEngine, SignalController
from haptic_engine.errors import E_DIGEST, E_SYNC, HapticError
from haptic_engine.parser import parse_manifest
from haptic_engine.scheduler import InterruptRequested


def make_engine(root: Path, payload: dict | None = None, serial: bool = False,
                signals: SignalController | None = None, delay: float = 0.0):
    data = payload or base_manifest(str(root))
    manifest = parse_manifest(data)
    return HapticEngine(manifest, serial=serial, signals=signals or SignalController(),
                        block_delay=delay)


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_serial_and_parallel_streams_identical(self):
        serial_engine = make_engine(self.root / "serial", serial=True)
        serial_engine.run()
        parallel_engine = make_engine(self.root / "parallel", serial=False)
        parallel_engine.run()
        serial_streams = json.loads((self.root / "serial" / "commands.json").read_text())["streams"]
        parallel_streams = json.loads((self.root / "parallel" / "commands.json").read_text())["streams"]
        self.assertEqual(serial_streams, parallel_streams)
        # Sorted by device id then sample order.
        for stream in parallel_streams.values():
            starts = [item["start_sample"] for item in stream]
            self.assertEqual(starts, sorted(starts))

    def test_inflight_budget_never_exceeded(self):
        engine = make_engine(self.root / "budget")
        report = engine.run()
        self.assertLessEqual(
            report["peak_inflight_commands"], engine.manifest.global_config.inflight_limit
        )

    def test_interrupt_at_block_boundary_then_resume_matches_clean_run(self):
        root = self.root / "interrupt"
        clean = make_engine(self.root / "clean")
        clean.run()

        signals = SignalController()
        engine = make_engine(root, signals=signals, delay=0.01)
        signals._event.set()
        signals.pending_signal = signal.Signals.SIGINT
        with self.assertRaises(InterruptRequested):
            engine.run()
        exit_code = engine.handle_interrupt_and_exit()
        self.assertEqual(exit_code, 130)
        marker = json.loads((root / "state" / "interrupt.json").read_text())
        self.assertTrue(marker["pending_resume"])
        # No temp block files are mistaken for completed snapshots.
        self.assertFalse(list(Path(root / "state").rglob(".*.tmp-*")))

        resumed = make_engine(root)
        report = resumed.run()
        self.assertEqual(report["interrupt_count"], 1)
        interrupted_streams = json.loads((root / "commands.json").read_text())["streams"]
        clean_streams = json.loads((self.root / "clean" / "commands.json").read_text())["streams"]
        self.assertEqual(interrupted_streams, clean_streams)

    def test_corrupt_block_is_detected_and_recomputed(self):
        root = self.root / "corrupt"
        engine = make_engine(root)
        engine.run()
        device_id = engine.manifest.devices[0].device_id
        target = root / "state" / "blocks" / device_id / "block_000000.json"
        payload = json.loads(target.read_text())
        payload["block"]["commands"][0]["level"] += 1
        target.write_text(json.dumps(payload))

        resumed = make_engine(root)
        report = resumed.run()
        self.assertGreaterEqual(report["devices"][device_id]["digest_failures"], 1)
        events = report["devices"][device_id]["digest_events"]
        self.assertTrue(any(event["code"] == E_DIGEST for event in events))
        clean = make_engine(self.root / "clean2")
        clean.run()
        repaired = json.loads((root / "commands.json").read_text())["streams"]
        reference = json.loads((self.root / "clean2" / "commands.json").read_text())["streams"]
        self.assertEqual(repaired, reference)

    def test_declared_command_count_mismatch_invalidates_block(self):
        root = self.root / "count"
        engine = make_engine(root)
        engine.run()
        device_id = engine.manifest.devices[0].device_id
        target = root / "state" / "blocks" / device_id / "block_000000.json"
        payload = json.loads(target.read_text())
        payload["block"]["command_count"] += 5
        target.write_text(json.dumps(payload))
        resumed = make_engine(root)
        report = resumed.run()
        self.assertGreaterEqual(report["devices"][device_id]["digest_failures"], 1)

    def test_version_change_archives_old_snapshot_and_reruns(self):
        root = self.root / "version"
        engine = make_engine(root)
        report_one = engine.run()
        changed = mutate(base_manifest(str(root)))
        changed["devices"][0]["levels"] = 32
        changed_engine = make_engine(root, payload=changed)
        report_two = changed_engine.run()
        self.assertNotEqual(report_one["version_id"], report_two["version_id"])
        archive = root / "state" / "archive"
        self.assertTrue(archive.exists())
        self.assertTrue(any(path.is_dir() for path in archive.iterdir()))
        current_version = json.loads((root / "state" / "version.json").read_text())
        self.assertEqual(current_version["version_id"], report_two["version_id"])

    def test_sync_violation_exports_streams_and_raises_sync_code(self):
        payload = mutate(base_manifest(str(self.root / "sync")))
        # 37 ms is not an integer sample on the 800 Hz device (29.6 -> 30),
        # producing a command-start offset relative to the 1000 Hz reference.
        payload["sync_anchors"][0]["time_ms"] = 37
        payload["sync_anchors"][0]["allowed_samples"] = 0
        engine = make_engine(self.root / "sync", payload=payload)
        with self.assertRaises(HapticError) as ctx:
            engine.store.initialize()
            engine._prepare_plans()
            engine._run_scheduler()
            engine._final_verify_and_repair()
            from haptic_engine.sync import evaluate_anchors
            evaluate_anchors(
                engine.manifest,
                engine.device_map,
                {k: v.blocks for k, v in engine._results.items()},
            )
        self.assertEqual(ctx.exception.code, E_SYNC)
        # Full engine path still exports valid streams and an exit-3 report.
        report = engine.run()
        self.assertFalse(report["ok"])
        self.assertTrue((self.root / "sync" / "commands.json").exists())

    def test_second_run_reuses_completed_blocks(self):
        root = self.root / "reuse"
        first = make_engine(root)
        first.run()
        second = make_engine(root)
        report = second.run()
        for device_report in report["devices"].values():
            self.assertEqual(device_report["recomputed_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
