from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from streamparse.engine import full_decode, semantic_payload
from streamparse.loader import load_extra_scores

from tests.helpers import assert_incremental_matches_full, make_app, make_sample


class EquivalenceAndCacheTests(unittest.TestCase):
    def test_strategy_switch_shares_grid_and_preserves_stable_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app = make_app(Path(raw))
            try:
                before = app.latest_report
                s01_before = next(item for item in before["sessions"] if item["session_id"] == "S01")
                response = app.execute(
                    {"cmd": "set_strategy", "session_id": "S01", "strategy": "greedy"}
                )
                self.assertEqual(response["status"], "ok")
                s01_after = next(
                    item
                    for item in app.latest_report["sessions"]
                    if item["session_id"] == "S01"
                )
                self.assertEqual(
                    s01_before["fingerprints"]["ownership_fp"],
                    s01_after["fingerprints"]["ownership_fp"],
                )
                self.assertEqual(
                    s01_before["fingerprints"]["grid_fp"],
                    s01_after["fingerprints"]["grid_fp"],
                )
                self.assertEqual(
                    s01_before["stable_prefix"],
                    s01_after["stable_prefix"],
                )
                self.assertNotEqual(
                    s01_before["fingerprints"]["tail_fp"],
                    s01_after["fingerprints"]["tail_fp"],
                )
                self.assertEqual(s01_after["cache_actions"]["grid"], "reuse")
                self.assertEqual(s01_after["cache_actions"]["stable_prefix"], "reuse")
                self.assertEqual(s01_after["cache_actions"]["tail_decode"], "invalidate")
                self.assertEqual(s01_after["decode_mode"], "tail_reuse")
                self.assertEqual(
                    s01_after["decode_reused_prefix_frames"],
                    s01_after["stable_prefix"]["length"],
                )
                assert_incremental_matches_full(self, app)
            finally:
                app.close()

    def test_incremental_edit_sequence_matches_full_decode_after_every_step(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            app = make_app(root)
            try:
                append_rows = load_extra_scores(
                    app.sample_dir / "extras" / "append_scores.jsonl", app.dataset.vocab
                )
                replace_rows = load_extra_scores(
                    app.sample_dir / "extras" / "replace_scores.jsonl", app.dataset.vocab
                )
                new_rows = load_extra_scores(
                    app.sample_dir / "extras" / "new_session_scores.jsonl", app.dataset.vocab
                )
                commands = [
                    {"cmd": "set_strategy", "session_id": "S01", "strategy": "greedy"},
                    {"cmd": "set_strategy", "session_id": "S01", "strategy": "lookahead"},
                    {"cmd": "set_param", "session_id": "S01", "key": "lookahead_frames", "value": 2},
                    {"cmd": "set_param", "session_id": "S01", "key": "overlap_frames", "value": 1},
                    {
                        "cmd": "append_shard",
                        "session_id": "S01",
                        "shard_seq": 3,
                        "start_frame": 8,
                        "end_frame": 11,
                        "scores": append_rows,
                    },
                    {
                        "cmd": "replace_shard",
                        "session_id": "S01",
                        "shard_seq": 1,
                        "start_frame": 3,
                        "end_frame": 6,
                        "scores": replace_rows,
                    },
                    {
                        "cmd": "add_session",
                        "session_id": "S05",
                        "shards": [
                            {
                                "shard_seq": 0,
                                "start_frame": 0,
                                "end_frame": 4,
                                "scores": new_rows[:5],
                            }
                        ],
                    },
                    {"cmd": "delete_session", "session_id": "S03"},
                    {"cmd": "set_param", "key": "lm_weight", "value": 0.7},
                    {"cmd": "set_strategy", "strategy": "greedy"},
                ]
                for command in commands:
                    response = app.execute(command)
                    self.assertEqual(
                        response["status"],
                        "ok",
                        msg=f"command failed/pending: {command} -> {response}",
                    )
                    assert_incremental_matches_full(self, app)
                self.assertNotIn("S03", app.dataset.sessions)
                self.assertIn("S05", app.dataset.sessions)
            finally:
                app.close()

    def test_append_replace_and_parameter_invalidation_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            app = make_app(root)
            try:
                append_rows = load_extra_scores(
                    app.sample_dir / "extras" / "append_scores.jsonl", app.dataset.vocab
                )
                replace_rows = load_extra_scores(
                    app.sample_dir / "extras" / "replace_scores.jsonl", app.dataset.vocab
                )

                app.execute({"cmd": "set_param", "session_id": "S01", "key": "overlap_frames", "value": 1})
                s01 = next(item for item in app.latest_report["sessions"] if item["session_id"] == "S01")
                self.assertEqual(s01["cache_actions"]["grid"], "reuse")
                self.assertEqual(s01["cache_actions"]["stable_prefix"], "reuse")
                self.assertEqual(s01["cache_actions"]["tail_decode"], "invalidate")
                self.assertEqual(s01["affected_intervals"][0]["reason"], "strategy_or_overlap_tail")

                app.execute(
                    {
                        "cmd": "append_shard",
                        "session_id": "S01",
                        "shard_seq": 3,
                        "start_frame": 8,
                        "end_frame": 11,
                        "scores": append_rows,
                    }
                )
                s01 = next(item for item in app.latest_report["sessions"] if item["session_id"] == "S01")
                self.assertEqual(s01["cache_actions"]["grid"], "partial_invalidate")
                self.assertIn(
                    s01["cache_actions"]["stable_prefix"],
                    ("reuse", "reuse_with_extension"),
                )
                self.assertEqual(s01["cache_actions"]["tail_decode"], "invalidate")
                self.assertEqual(
                    s01["affected_intervals"],
                    [{"start_frame": 9, "end_frame": 11, "reason": "append_new_and_overlap"}],
                )

                app.execute(
                    {
                        "cmd": "replace_shard",
                        "session_id": "S01",
                        "shard_seq": 1,
                        "start_frame": 3,
                        "end_frame": 6,
                        "scores": replace_rows,
                    }
                )
                s01 = next(item for item in app.latest_report["sessions"] if item["session_id"] == "S01")
                self.assertEqual(s01["affected_intervals"][0]["reason"], "replace_shard_and_downstream")
                self.assertEqual(s01["affected_intervals"][0]["start_frame"], 3)
                self.assertGreaterEqual(s01["affected_intervals"][0]["end_frame"], 11)
                self.assertEqual(s01["cache_actions"]["tail_decode"], "invalidate")
                assert_incremental_matches_full(self, app)
            finally:
                app.close()

    def test_same_input_and_config_are_frame_stable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample_dir = make_sample(root)
            from streamparse.loader import load_dataset

            dataset, config, injections, load_failures = load_dataset(sample_dir)
            first = full_decode(dataset, config, injections=injections, initial_failures=load_failures)
            second = full_decode(dataset, config, injections=injections, initial_failures=load_failures)
            self.assertEqual(semantic_payload(first), semantic_payload(second))
            greedy_config = replace(config, strategy="greedy", lookahead_frames=0)
            greedy = full_decode(
                dataset,
                greedy_config,
                injections=injections,
                initial_failures=load_failures,
            )
            lookahead = full_decode(dataset, config, injections=injections, initial_failures=load_failures)
            by_greedy = {item["session_id"]: item for item in greedy["sessions"]}
            by_lookahead = {item["session_id"]: item for item in lookahead["sessions"]}
            self.assertEqual(
                by_greedy["S01"]["fingerprints"]["grid_fp"],
                by_lookahead["S01"]["fingerprints"]["grid_fp"],
            )
            self.assertEqual(greedy["overlap_ownership"], lookahead["overlap_ownership"])
            self.assertEqual(
                by_greedy["S01"]["stable_prefix"],
                by_lookahead["S01"]["stable_prefix"],
            )


if __name__ == "__main__":
    unittest.main()
