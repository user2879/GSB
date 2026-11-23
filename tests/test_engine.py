import tempfile
import unittest
from pathlib import Path

from arcore.clock import MonotonicClock
from arcore.engine import run_round
from arcore.ledger import (
    commit_round,
    open_ledger,
    replay_rounds,
)
from arcore.spec import load_declaration
from arcore.stream import generate_stream

SAMPLE = Path(__file__).resolve().parents[1] / "sample" / "declaration.json"


class EngineTest(unittest.TestCase):
    def test_round_payload_is_deterministic_and_replays_clean(self):
        declaration = load_declaration(SAMPLE)
        stream = generate_stream(declaration)
        result = run_round(declaration, 1, stream=stream)
        payload = result.payload
        self.assertEqual(payload["progress_cursor"]["processed_alert_count"],
                         len(stream.alerts))
        self.assertIn("path_decision", payload)
        self.assertIn("error_bounds", payload)
        self.assertIn("root_causes", payload)
        self.assertIn("batch_specs", payload)
        self.assertIn("lease_occupancy", payload)
        # A second independent run reproduces every deterministic field.
        replayed = run_round(
            declaration, 1, stream=generate_stream(declaration)).payload
        with tempfile.TemporaryDirectory() as tmp:
            connection, _ = open_ledger(Path(tmp) / "led.sqlite3")
            commit_round(connection, 1, payload, result.stats,
                         MonotonicClock().now_ms())
            diffs = replay_rounds(connection, [replayed])
            self.assertEqual(diffs, [])
            connection.close()

    def test_renewal_hook_records_renew_events(self):
        declaration = load_declaration(SAMPLE)

        def hook(batch, lease):
            from arcore.errors import LeaseError
            try:
                lease_manager.renew(lease.lease_id, lease.holder,
                                    lease.version, extension_ms=1000)
            except LeaseError:
                pass

        lease_manager = __import__(
            "arcore.lease", fromlist=["LeaseManager"]).LeaseManager(
            declaration.parameters.global_concurrency,
            declaration.parameters.service_concurrency,
            ttl_ms=10_000, clock=MonotonicClock())
        result = run_round(
            declaration, 1, lease_manager=lease_manager, work_hook=hook)
        self.assertGreaterEqual(
            result.payload["lease_occupancy"]["renewed"], 1)


if __name__ == "__main__":
    unittest.main()
