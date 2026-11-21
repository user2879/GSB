from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workbench.engine import RunConfig, Workbench
from workbench.errors import ConfigError, SnapshotStaleError
from workbench.loader import load_plan
from workbench.util import read_json, stable_hash


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def run_workbench(
    output: Path,
    *,
    step: int,
    mode: str = "incremental",
    serial: bool = False,
    budget: int | None = None,
    config_override: Path | None = None,
) -> dict:
    return Workbench(
        RunConfig(
            samples_dir=SAMPLES,
            output_dir=output,
            step=step,
            mode=mode,
            serial=serial,
            budget_override=budget,
            config_override=config_override,
        )
    ).run()


def plan_steps() -> list[int]:
    return [step["step"] for step in load_plan(SAMPLES)["steps"]]


class WorkbenchTests(unittest.TestCase):
    def test_incremental_rolling_equals_blank_full_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incremental = root / "incremental"
            full = root / "full"
            run_workbench(incremental, step=0)
            for step in plan_steps():
                report = run_workbench(incremental, step=step)
            run_workbench(full, step=max(plan_steps()), mode="full")
            inc_canonical = read_json(incremental / "canonical.json")
            full_canonical = read_json(full / "canonical.json")
            self.assertEqual(inc_canonical, full_canonical)
            self.assertEqual(stable_hash(inc_canonical), stable_hash(full_canonical))
            self.assertGreater(report["cache"]["totals"]["reused"], 0)
            self.assertGreater(report["cache"]["totals"]["invalidated"], 0)
            self.assertEqual(
                report["failure_list"],
                read_json(full / "report.json")["failure_list"],
            )

    def test_serial_and_thread_pool_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parallel = root / "parallel"
            serial = root / "serial"
            run_workbench(parallel, step=4, mode="full")
            run_workbench(serial, step=4, mode="full", serial=True)
            self.assertEqual(
                read_json(parallel / "canonical.json"),
                read_json(serial / "canonical.json"),
            )

    def test_merge_authority_cross_channel_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            report = run_workbench(output, step=1, mode="full")
            lexical = [
                record
                for record in report["merge_records"]
                if record["request_id"] == "R001"
                and record["item_id"] == "I001"
                and record["channel"] == "lexical"
            ][0]
            semantic = [
                record
                for record in report["merge_records"]
                if record["request_id"] == "R001"
                and record["item_id"] == "I001"
                and record["channel"] == "semantic"
            ][0]
            self.assertEqual(lexical["authority"]["score"], 0.95)
            self.assertEqual(lexical["authority"]["shard_seq"], 1)
            self.assertEqual(lexical["occurrences_count"], 2)
            self.assertFalse(lexical["overridden"][0]["overlaps_authority"])
            self.assertEqual(semantic["authority"]["score"], 0.8)
            self.assertEqual(semantic["occurrences_count"], 2)
            self.assertTrue(semantic["overridden"][0]["overlaps_authority"])
            self.assertNotEqual(lexical["cache_key"], semantic["cache_key"])
            request_records = [
                record for record in report["merge_records"] if record["request_id"] == "R001"
            ]
            self.assertEqual(
                request_records,
                sorted(
                    request_records,
                    key=lambda record: (
                        record["order"]["channel"],
                        record["order"]["coverage_start_ms"],
                        record["order"]["coverage_end_ms"],
                        record["order"]["shard_seq"],
                        record["order"]["line_no"],
                        record["item_id"],
                        record["channel"],
                    ),
                ),
            )

    def test_fixed_point_fusion_boundaries_and_rerank_discounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            report = run_workbench(output, step=4, mode="full")
            i017 = [
                record
                for record in report["fusion_records"]
                if record["request_id"] == "R004" and record["item_id"] == "I017"
            ][0]
            self.assertEqual(i017["fusion_fixed"], 10000)
            self.assertEqual(i017["fusion_score"], 1.0)
            self.assertIn("fusion_score_exact_clip_boundary", i017["edge_flags"])
            tied = [
                record
                for record in report["fusion_records"]
                if record["request_id"] == "R003"
                and record["item_id"] in {"I010", "I011"}
            ]
            self.assertEqual(len(tied), 2)
            self.assertEqual(tied[0]["fusion_fixed"], tied[1]["fusion_fixed"])
            criteria = {sample["criterion"] for sample in report["boundary_samples"]}
            self.assertIn("fusion_score_tie_item_id_order", criteria)
            self.assertIn("similarity_equal_threshold_no_discount", criteria)
            self.assertTrue(report["discount_records"])
            for record in report["discount_records"]:
                self.assertEqual(
                    record["before_fixed"] - record["after_fixed"],
                    record["discount_fixed"],
                )
                self.assertGreater(record["similarity_fixed"], 6000)
            hit_ids = {hit["id"] for hit in report["injection_hits"] if hit["hit"]}
            self.assertTrue({"INJ-001", "INJ-002", "INJ-003", "INJ-005", "INJ-006"} <= hit_ids)
            failures = report["failure_list"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["code"], "E_SHARD_INVALID")
            self.assertEqual(failures[0]["injection_id"], "INJ-001")

    def test_budget_pending_priority_and_eventual_full_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            low = root / "low"
            high = root / "high"
            high_report = run_workbench(high, step=4, mode="full")
            high_fusion = {
                (record["request_id"], record["item_id"]): record
                for record in high_report["fusion_records"]
            }
            rounds = []
            for _ in range(20):
                report = run_workbench(low, step=4, budget=3)
                rounds.append(report["budget"]["current_round"])
                first_completed = {
                    (record["request_id"], record["item_id"]): record
                    for record in report["fusion_records"]
                }
                for key, low_record in first_completed.items():
                    self.assertEqual(low_record["fusion_fixed"], high_fusion[key]["fusion_fixed"])
                    self.assertEqual(low_record["contributions"], high_fusion[key]["contributions"])
                if not report["pending"]:
                    break
            else:
                self.fail("budget scheduler did not finish within 20 rounds")
            first_round = report["budget"]["rounds"][0]
            self.assertEqual(
                [(item["request_id"], item["item_id"]) for item in first_round["deductions"]],
                [("R001", "I001"), ("R001", "I002"), ("R001", "I003")],
            )
            self.assertEqual(report["budget"]["rounds"][1]["input_pending_count"], 18)
            self.assertEqual(report["pending"], [])
            self.assertEqual(report["canonical_hash"], high_report["canonical_hash"])
            self.assertLessEqual(report["budget"]["rounds"][-1]["consumed"], 3)

    def test_stale_snapshot_is_rejected_with_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            run_workbench(output, step=0)
            manifest_path = output / "live" / "manifest.json"
            manifest = read_json(manifest_path)
            manifest["config_hash"] = "tampered"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            with self.assertRaises(SnapshotStaleError) as context:
                run_workbench(output, step=1)
            self.assertEqual(context.exception.code, "E_SNAPSHOT_STALE")
            error = read_json(output / "error_report.json")
            self.assertEqual(error["code"], "E_SNAPSHOT_STALE")

    def test_snapshot_stage_versions_and_atomic_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            run_workbench(output, step=0)
            first = read_json(output / "live" / "manifest.json")
            second_report = run_workbench(output, step=1)
            second = read_json(output / "live" / "manifest.json")
            self.assertEqual(second["global_version"], first["global_version"] + 1)
            self.assertNotEqual(
                second["stage_versions"]["alignment"]["version"],
                first["stage_versions"]["alignment"]["version"],
            )
            for stage in ("alignment", "merge", "normalize", "fusion", "rerank"):
                self.assertIn(stage, second["stage_versions"])
                self.assertIn("content_hash", second["stage_versions"][stage])
            for filename in (
                "manifest.json",
                "state.json",
                "cache.json",
                "alignment.json",
                "merge.json",
                "fusion.json",
                "rerank.json",
                "canonical.json",
                "report.json",
            ):
                self.assertTrue((output / "live" / filename).exists(), filename)
            self.assertFalse(list((output / ".tmp").iterdir()))
            self.assertEqual(second_report["snapshots"]["global_version"], 2)

    def test_invalid_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = read_json(SAMPLES / "rerank_config.json")
            config["dedup_similarity_threshold"] = 1.5
            config_path = root / "bad_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigError):
                run_workbench(root / "out", step=0, config_override=config_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
