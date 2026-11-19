from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rts_pipeline.canonical import atomic_write_json
from rts_pipeline.chunker import load_manifest, prepare_chunks
from rts_pipeline.errors import PipelineError
from rts_pipeline.runner import load_workspace, run_workspace

from .helpers import make_workspace


class ChunkTests(unittest.TestCase):
    def test_chunk_manifest_is_repeatable_and_tail_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generated = make_workspace(
                Path(first), seed=211, ticks=10, chunk_ticks=4
            )
            game, rules, events = load_workspace(Path(first))
            first_manifest = prepare_chunks(Path(first), events, rules, game, 4)
            second_manifest = prepare_chunks(Path(second), events, rules, game, 4)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                [(chunk["start_tick"], chunk["end_tick"]) for chunk in first_manifest["chunks"]],
                [(0, 3), (4, 7), (8, 9)],
            )
            self.assertEqual(first_manifest["chunks"][-1]["tick_count"], 2)
            actual_events = sum(chunk["event_count"] for chunk in first_manifest["chunks"])
            self.assertEqual(actual_events, len(generated["events"]))
            for chunk in first_manifest["chunks"]:
                chunk_path = Path(first) / chunk["file"]
                other_path = Path(second) / chunk["file"]
                self.assertEqual(chunk_path.read_bytes(), other_path.read_bytes())
            self.assertEqual(
                (Path(first) / "chunks" / "manifest.json").read_bytes(),
                (Path(second) / "chunks" / "manifest.json").read_bytes(),
            )

    def test_corrupt_chunk_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            make_workspace(Path(root), seed=221, ticks=8, chunk_ticks=4)
            manifest_path = Path(root) / "chunks" / "manifest.json"
            game, rules, events = load_workspace(Path(root))
            manifest = prepare_chunks(Path(root), events, rules, game, 4)
            manifest["chunks"][0]["event_count"] += 1
            atomic_write_json(manifest_path, manifest)
            with self.assertRaises(PipelineError) as caught:
                load_manifest(Path(root))
            self.assertEqual(caught.exception.code, "E_CHUNK_BOUNDARY")


class SchedulerTests(unittest.TestCase):
    def test_thread_and_budget_changes_preserve_business_hash(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            make_workspace(Path(root), seed=301, ticks=16, chunk_ticks=3)
            game, rules, events = load_workspace(Path(root))
            manifest = prepare_chunks(Path(root), events, rules, game, 3)
            max_block = max(item["estimated_memory_bytes"] for item in manifest["chunks"])
            configs = (
                (1, max_block),
                (2, max_block * 2),
                (4, max_block * 4),
                (8, max_block),
            )
            runs = []
            for threads, budget in configs:
                result = run_workspace(
                    Path(root),
                    threads=threads,
                    memory_budget_bytes=budget,
                    chunk_ticks=3,
                )
                self.assertEqual(result["status"], "ok")
                self.assertTrue(result["verification"]["full_consistency"])
                self.assertTrue(result["truth_comparison"]["full_consistency"])
                runs.append(result)
            self.assertEqual(
                len({run["parallel_ledger"]["business_hash"] for run in runs}), 1
            )
            degraded = runs[-1]["parallel_ledger"]["scheduler"]
            self.assertLess(degraded["effective_parallelism"], 8)
            self.assertTrue(degraded["degradations"])
            self.assertEqual(degraded["merge_order_basis"], "start_tick_then_chunk_id")

    def test_chunk_length_changes_partition_not_business_hash(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            make_workspace(Path(root), seed=351, ticks=15, chunk_ticks=4)
            hashes = []
            for chunk_ticks in (2, 7):
                result = run_workspace(
                    Path(root),
                    threads=3,
                    memory_budget_bytes=8 * 1024 * 1024,
                    chunk_ticks=chunk_ticks,
                )
                self.assertEqual(result["status"], "ok")
                hashes.append(result["parallel_ledger"]["business_hash"])
            self.assertEqual(len(set(hashes)), 1)

    def test_memory_below_single_block_denies_every_chunk_and_writes_no_exit(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            make_workspace(Path(root), seed=401, ticks=9, chunk_ticks=3)
            result = run_workspace(
                Path(root), threads=4, memory_budget_bytes=1, chunk_ticks=3
            )
            self.assertEqual(result["status"], "denied")
            self.assertEqual(result["result_code"], "E_BUDGET_DENIED")
            self.assertFalse(result["exit_report_written"])
            self.assertFalse((Path(root) / "output" / "exit_report.json").exists())
            self.assertTrue((Path(root) / "output" / "error.json").exists())
            report = result["scheduler_report"]
            self.assertGreater(len(report["denied_blocks"]), 0)
            self.assertTrue(
                all(item["code"] == "E_BUDGET_DENIED" for item in report["denied_blocks"])
            )


if __name__ == "__main__":
    unittest.main()
