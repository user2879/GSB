from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gsb.errors import DiagnosticError
from gsb.spec import load_session


def _base() -> dict:
    return {
        "format_version": 1,
        "global_budget": 3,
        "shards": [
            {
                "shard_id": "a",
                "host": "127.0.0.1",
                "port": 39001,
                "lease_limit": 1,
                "concurrency_limit": 1,
                "seed": 7,
                "parameter_version": 1,
                "sample_dim": 2,
                "lease_ttl_ms": 1000,
                "layers": [
                    {
                        "layer_id": "l1",
                        "input_dim": 2,
                        "output_dim": 2,
                        "scale": 1000,
                        "depends_on": [],
                    }
                ],
            }
        ],
        "checkpoints": [
            {
                "checkpoint_id": "cp",
                "shard_id": "a",
                "parameter_version": 1,
                "loss_baseline_digest": "0" * 64,
            }
        ],
        "batches": [
            {
                "batch_id": "b",
                "shard_id": "a",
                "sample_ids": [1],
                "resource_units": 1,
                "order": 1,
            }
        ],
        "events": [
            {
                "seq": 2,
                "type": "commit",
                "shard_id": "a",
                "checkpoint_id": "cp",
            }
        ],
    }


class SpecValidationTests(unittest.TestCase):
    def _write(self, payload: dict, directory: str, name: str = "session.json") -> Path:
        path = Path(directory) / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(_base(), directory)
            session = load_session(config_path=path)
            self.assertEqual(session.shards[0].shard_id, "a")
            self.assertEqual(len(session.batches), 1)

    def test_duplicate_port_reports_field_and_source_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = _base()
            second = json.loads(json.dumps(payload["shards"][0]))
            second["shard_id"] = "b"
            second["port"] = payload["shards"][0]["port"]
            second["layers"] = [dict(second["layers"][0], layer_id="other")]
            payload["shards"].append(second)
            path = self._write(payload, directory)
            lines = path.read_text(encoding="utf-8").splitlines()
            expected_line = [
                index
                for index, line in enumerate(lines, 1)
                if line.strip().startswith('"port": 39001')
            ][1]
            with self.assertRaises(DiagnosticError) as caught:
                load_session(config_path=path)
            error = caught.exception
            self.assertEqual(error.code, "E_SPEC")
            self.assertEqual(error.field, "shards[1].port")
            self.assertEqual(error.source, str(path))
            self.assertEqual(error.line, expected_line)

    def test_cycle_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = _base()
            payload["shards"][0]["layers"].append(
                {
                    "layer_id": "l2",
                    "input_dim": 2,
                    "output_dim": 2,
                    "scale": 1000,
                    "depends_on": ["l1"],
                }
            )
            payload["shards"][0]["layers"][0]["depends_on"] = ["l2"]
            path = self._write(payload, directory)
            with self.assertRaises(DiagnosticError) as caught:
                load_session(config_path=path)
            self.assertEqual(caught.exception.code, "E_SPEC")
            self.assertIn("cycle", caught.exception.message)

    def test_unknown_field_fails_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = _base()
            payload["shards"][0]["unexpected"] = 1
            path = self._write(payload, directory)
            with self.assertRaises(DiagnosticError) as caught:
                load_session(config_path=path)
            self.assertEqual(caught.exception.field, "shards[0].unexpected")
            self.assertIsInstance(caught.exception.line, int)

    def test_separate_documents_keep_original_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = _base()
            shards_path = Path(directory) / "shards.json"
            checkpoints_path = Path(directory) / "checkpoints.json"
            batches_path = Path(directory) / "batches.json"
            events_path = Path(directory) / "events.json"
            shards_path.write_text(json.dumps(payload["shards"], indent=2) + "\n")
            checkpoints_path.write_text(json.dumps(payload["checkpoints"], indent=2) + "\n")
            batches_path.write_text(json.dumps(payload["batches"], indent=2) + "\n")
            events_path.write_text(json.dumps(payload["events"], indent=2) + "\n")
            session = load_session(
                shards_path=shards_path,
                checkpoints_path=checkpoints_path,
                batches_path=batches_path,
                events_path=events_path,
                global_budget=3,
            )
            self.assertEqual(session.shards[0].port, 39001)
            self.assertEqual(session.events[0].type, "commit")

    def test_separate_shards_report_original_source_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = _base()
            second = json.loads(json.dumps(payload["shards"][0]))
            second["shard_id"] = "b"
            second["layers"] = [dict(second["layers"][0], layer_id="other")]
            shards = [payload["shards"][0], second]
            shards_path = Path(directory) / "shards.json"
            checkpoints_path = Path(directory) / "checkpoints.json"
            shards_path.write_text(json.dumps(shards, indent=2) + "\n", encoding="utf-8")
            checkpoints_path.write_text(json.dumps(payload["checkpoints"], indent=2) + "\n")
            lines = shards_path.read_text(encoding="utf-8").splitlines()
            expected_line = [
                index
                for index, line in enumerate(lines, 1)
                if line.strip().startswith('"port": 39001')
            ][1]
            with self.assertRaises(DiagnosticError) as caught:
                load_session(
                    shards_path=shards_path,
                    checkpoints_path=checkpoints_path,
                    global_budget=3,
                )
            self.assertEqual(caught.exception.field, "shards[1].port")
            self.assertEqual(caught.exception.source, str(shards_path))
            self.assertEqual(caught.exception.line, expected_line)


if __name__ == "__main__":
    unittest.main()
