from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_session_fixture, read_json, run_cli


class ValidationTests(unittest.TestCase):
    def test_invalid_media_returns_business_error_without_state_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = make_session_fixture(
                root,
                fragments=[
                    {
                        "duration_ms": 0,
                        "representations": [{"bitrate_bps": 200_000, "size_bytes": 50_000}],
                    }
                ],
            )
            code, payload = run_cli(session)
            self.assertNotEqual(code, 0)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["code"], "E_INPUT")
            self.assertFalse((root / "state").exists())
            self.assertFalse((root / "snapshots").exists())

    def test_startup_target_above_max_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = make_session_fixture(
                root,
                client={"startup_buffer_ms": 5000, "max_buffer_ms": 4000},
            )
            code, payload = run_cli(session)
            self.assertNotEqual(code, 0)
            self.assertEqual(payload["error"]["code"], "E_INPUT")
            self.assertIn("startup_buffer_ms", payload["error"]["path"])
            self.assertFalse((root / "state").exists())

    def test_jump_past_media_duration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = make_session_fixture(
                root,
                events=[{"at_ms": 1000, "action": "jump", "target_ms": 9000}],
            )
            code, payload = run_cli(session)
            self.assertNotEqual(code, 0)
            self.assertEqual(payload["error"]["code"], "E_EVENT")
            self.assertFalse((root / "state").exists())

    def test_negative_rtt_and_bandwidth_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = make_session_fixture(
                root,
                trace=[{"bandwidth_bps": -1, "rtt_ms": -2}],
            )
            code, payload = run_cli(session)
            self.assertNotEqual(code, 0)
            self.assertIn(payload["error"]["code"], {"E_INPUT", "E_NETWORK"})
            self.assertFalse((root / "state").exists())

    def test_deterministic_integer_millisecond_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = make_session_fixture(
                root,
                fragments=[
                    {
                        "duration_ms": 1000,
                        "representations": [
                            {"bitrate_bps": 333_333, "size_bytes": 41_667},
                        ],
                    }
                ],
                trace=[{"bandwidth_bps": 1_000_000, "rtt_ms": 0}],
                client={"startup_buffer_ms": 0, "max_buffer_ms": 2000},
            )
            code, payload = run_cli(session)
            self.assertEqual(code, 0)
            start = next(e for e in payload["events"] if e["kind"] == "download_start")
            self.assertEqual(start["duration_ms"], 334)
            self.assertEqual(start["download_complete_ms"], 334)
            self.assertEqual(payload["events"][-1]["kind"], "finish")


if __name__ == "__main__":
    unittest.main()
