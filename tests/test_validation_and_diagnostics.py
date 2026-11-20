from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from streamparse.engine import full_decode
from streamparse.errors import AppError, E_DECODE_PARAM_INVALID, E_INPUT_INVALID
from streamparse.loader import load_config, load_dataset

from tests.helpers import make_sample


class ValidationAndDiagnosticsTests(unittest.TestCase):
    def test_injections_are_hit_isolated_and_mapped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample_dir = make_sample(root)
            dataset, config, injections, load_failures = load_dataset(sample_dir)
            report = full_decode(
                dataset,
                config,
                injections=injections,
                initial_failures=load_failures,
            )

        items = {item["id"]: item for item in report["injection_report"]["items"]}
        expected_codes = {
            "INJ-001-NONFINITE": E_INPUT_INVALID,
            "INJ-002-FRAME-COUNT": E_INPUT_INVALID,
            "INJ-003-OVERLAP": E_INPUT_INVALID,
            "INJ-004-ZERO-LOOKAHEAD": E_DECODE_PARAM_INVALID,
        }
        for injection_id, code in expected_codes.items():
            self.assertEqual(items[injection_id]["hit_count"], 1)
            self.assertEqual(items[injection_id]["observed_business_code"], code)
        self.assertEqual(report["injection_report"]["unhit_ids"], [])

        failure_order = [
            (item["session_id"], item["shard_seq"], item["injection_id"])
            for item in report["failures"]
        ]
        self.assertEqual(
            failure_order,
            [
                ("S-INJ-FRAME", 0, "INJ-002-FRAME-COUNT"),
                ("S-INJ-LOOKAHEAD", None, "INJ-004-ZERO-LOOKAHEAD"),
                ("S-INJ-OVERLAP", None, "INJ-003-OVERLAP"),
                ("S-INJ-SHARD", 1, "INJ-001-NONFINITE"),
            ],
        )
        sessions = {item["session_id"]: item for item in report["sessions"]}
        # A failed shard does not abort the rest of its session.
        self.assertEqual(sessions["S-INJ-SHARD"]["status"], "partial")
        self.assertGreater(sessions["S-INJ-SHARD"]["frame_count"], 0)
        self.assertEqual(sessions["S01"]["status"], "ok")
        self.assertEqual(sessions["S02"]["status"], "ok")

    def test_overlap_ownership_is_auditable_and_later_shard_wins(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sample_dir = make_sample(Path(raw))
            dataset, config, injections, load_failures = load_dataset(sample_dir)
            report = full_decode(
                dataset,
                config,
                injections=injections,
                initial_failures=load_failures,
            )
        s01 = next(item for item in report["sessions"] if item["session_id"] == "S01")
        conflicts = s01["ownership_conflicts"]
        self.assertEqual([item["global_frame"] for item in conflicts], [3, 4, 5, 6])
        self.assertEqual([item["winning_shard_seq"] for item in conflicts], [1, 1, 2, 2])
        self.assertEqual([item["overwritten_shard_seq"] for item in conflicts], [0, 0, 1, 1])
        self.assertEqual([item["overwritten_frame_in_shard"] for item in conflicts], [3, 4, 2, 3])
        for frame in s01["ownership_frames"]:
            if frame["global_frame"] in (3, 4, 5, 6):
                self.assertTrue(frame["overwritten"])

    def test_smoothing_is_marked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            sample_dir = make_sample(Path(raw))
            dataset, config, injections, load_failures = load_dataset(sample_dir)
            report = full_decode(
                dataset,
                config,
                injections=injections,
                initial_failures=load_failures,
            )
        s02 = next(item for item in report["sessions"] if item["session_id"] == "S02")
        self.assertEqual(s02["smoothed_frames"], [4])
        smoothing_decision = next(
            item for item in s02["frame_decisions"] if item["global_frame"] == 4
        )
        self.assertTrue(smoothing_decision["smoothed"])
        self.assertEqual(smoothing_decision["token"], "<sil>")

    def test_invalid_score_dimension_invalidates_only_that_shard(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample_dir = make_sample(root)
            scores_path = sample_dir / "scores.jsonl"
            rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
            target = next(
                row
                for row in rows
                if row["session_id"] == "S01" and row["shard_seq"] == 1 and row["frame_in_shard"] == 0
            )
            target["scores"] = target["scores"][:-1]
            scores_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            dataset, config, injections, load_failures = load_dataset(sample_dir)
            report = full_decode(
                dataset,
                config,
                injections=injections,
                initial_failures=load_failures,
            )
        s01 = next(item for item in report["sessions"] if item["session_id"] == "S01")
        self.assertEqual(s01["status"], "partial")
        self.assertTrue(any(item["shard_seq"] == 1 for item in s01["failures"]))
        self.assertTrue(any(item["business_code"] == E_INPUT_INVALID for item in s01["failures"]))
        s02 = next(item for item in report["sessions"] if item["session_id"] == "S02")
        self.assertEqual(s02["status"], "ok")

    def test_session_sequence_gap_invalidates_only_session(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample_dir = make_sample(root)
            sessions_path = sample_dir / "sessions.jsonl"
            rows = [json.loads(line) for line in sessions_path.read_text(encoding="utf-8").splitlines()]
            rows = [
                row
                for row in rows
                if not (row["session_id"] == "S01" and row["shard_seq"] == 1)
            ]
            sessions_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            dataset, config, injections, load_failures = load_dataset(sample_dir)
            report = full_decode(
                dataset,
                config,
                injections=injections,
                initial_failures=load_failures,
            )
        s01 = next(item for item in report["sessions"] if item["session_id"] == "S01")
        self.assertEqual(s01["status"], "failed")
        self.assertTrue(
            any(
                item["business_code"] == E_INPUT_INVALID
                and "gap" in item["message"]
                for item in s01["failures"]
            )
        )
        s02 = next(item for item in report["sessions"] if item["session_id"] == "S02")
        self.assertEqual(s02["status"], "ok")

    def test_non_positive_slot_bounds_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample_dir = make_sample(root)
            config_path = sample_dir / "decode_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["slot_min_frames"] = 0
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(AppError) as raised:
                load_config(config_path)
            self.assertEqual(raised.exception.code, E_INPUT_INVALID)


if __name__ == "__main__":
    unittest.main()
