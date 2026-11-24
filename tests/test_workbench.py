from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from arbitration_workbench.canonical import fixed_str
from arbitration_workbench.editing import apply_edit
from arbitration_workbench.engine import run_workbench
from arbitration_workbench.loading import load_evidence_file, load_inputs
from arbitration_workbench.sample import generate_sample


def business_projection(report: dict) -> dict:
    summary = report["summary"]
    return {
        "workorders": report["workorders"],
        "failures": report["failures"],
        "summary": {
            "workorder_count": summary["workorder_count"],
            "successful_count": summary["successful_count"],
            "failure_count": summary["failure_count"],
            "pending_count": summary["pending_count"],
            "boundary_workorders": summary["boundary_workorders"],
            "boundary_by_priority": summary["boundary_by_priority"],
            "priority_distribution": summary["priority_distribution"],
            "failure_code_distribution": summary["failure_code_distribution"],
        },
    }


class WorkbenchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="awb-test-")
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        generate_sample(self.data)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_report(self, state: str, *, mode: str = "parallel", injections: Path | None = None) -> dict:
        return run_workbench(
            self.data,
            self.root / state,
            mode=mode,
            injections_path=injections,
        )

    def test_serial_parallel_and_repeat_are_identical(self) -> None:
        serial = self.run_report("state-serial", mode="serial")
        parallel = self.run_report("state-parallel", mode="parallel")
        self.assertFalse(serial["fatal"])
        self.assertFalse(parallel["fatal"])
        self.assertEqual(business_projection(serial), business_projection(parallel))
        self.assertEqual(serial["ledger"]["budget"], parallel["ledger"]["budget"])

        repeat = run_workbench(self.data, self.root / "state-serial", mode="parallel")
        self.assertEqual(repeat["ledger"]["recomputed_workorders"], [])
        self.assertEqual(repeat["ledger"]["stage_reuse_counts"], {"decay": 8, "relations": 8, "aggregation": 8})
        self.assertEqual(business_projection(serial), business_projection(repeat))
        self.assertEqual(serial["summary"]["boundary_workorders"], ["WO-0008"])

        by_id = {row["workorder_id"]: row for row in serial["workorders"]}
        boundary = by_id["WO-0008"]
        self.assertEqual(boundary["decision_score"], "0.2500")
        self.assertTrue(boundary["boundary"])
        self.assertEqual(boundary["matched_threshold_priority"], "P2")
        self.assertEqual(boundary["decay"]["metadata"]["exact_half_life_boundaries"], [1])

        wo1 = by_id["WO-0001"]
        kinds = [row["kind"] for row in wo1["relations"]["adjustments"]]
        self.assertTrue(kinds)
        self.assertEqual(kinds, sorted(kinds, key=lambda kind: 0 if kind == "mutual_exclusion" else 1))

    def test_budget_pending_quota_and_next_batch_priority(self) -> None:
        config_path = self.data / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["batch_evidence_budget"] = 7
        config_path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")

        serial = self.run_report("state-budget-serial", mode="serial")
        parallel = self.run_report("state-budget-parallel", mode="parallel")
        self.assertEqual(business_projection(serial), business_projection(parallel))
        self.assertEqual(serial["ledger"]["budget"]["consumed_this_run"], 6)
        self.assertEqual(serial["ledger"]["budget"]["remaining_this_run"], 1)
        self.assertEqual(
            serial["ledger"]["pending_workorders"],
            ["WO-0003", "WO-0004", "WO-0005", "WO-0006", "WO-0007"],
        )
        self.assertEqual([row["workorder_id"] for row in serial["workorders"]], ["WO-0001", "WO-0002", "WO-0008"])
        failure_rows = [(row["workorder_id"], row["code"]) for row in serial["failures"]]
        self.assertEqual(
            failure_rows,
            [
                ("WO-0003", "E_EVIDENCE_BUDGET_PENDING"),
                ("WO-0004", "E_EVIDENCE_BUDGET_PENDING"),
                ("WO-0005", "E_EVIDENCE_BUDGET_PENDING"),
                ("WO-0006", "E_EVIDENCE_BUDGET_PENDING"),
                ("WO-0007", "E_EVIDENCE_BUDGET_PENDING"),
            ],
        )

        second = run_workbench(self.data, self.root / "state-budget-serial", mode="serial")
        self.assertTrue(
            second["ledger"]["scheduled_decay_order"][:5]
            == ["WO-0003", "WO-0004", "WO-0005", "WO-0006", "WO-0007"]
        )
        self.assertEqual(second["ledger"]["pending_workorders"], ["WO-0005", "WO-0006", "WO-0007"])
        self.assertEqual(second["ledger"]["reused_workorders"], ["WO-0001", "WO-0002", "WO-0008"])

        quota_data = self.root / "quota-data"
        shutil.copytree(self.data, quota_data)
        quota_path = quota_data / "config.json"
        quota_config = json.loads(quota_path.read_text(encoding="utf-8"))
        quota_config["workorder_evidence_quota"] = 2
        quota_config["batch_evidence_budget"] = 100
        quota_path.write_text(json.dumps(quota_config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        quota = run_workbench(quota_data, self.root / "state-quota", mode="parallel")
        quota_failures = [(row["workorder_id"], row["code"]) for row in quota["failures"]]
        self.assertEqual(len(quota_failures), 6)
        self.assertTrue(all(code == "E_WORKORDER_BUDGET_EXHAUSTED" for _, code in quota_failures))
        self.assertEqual(quota_failures, sorted(quota_failures))

    def test_edit_sequence_reuses_unaffected_and_matches_full_rebuild(self) -> None:
        self.run_report("state-incr", mode="parallel")
        commands = json.loads((self.data / "edits.json").read_text(encoding="utf-8"))["commands"]
        reports = []
        for index, command in enumerate(commands, start=1):
            apply_edit(self.data, command)
            report = run_workbench(self.data, self.root / "state-incr", mode="parallel")
            reports.append(report)
            self.assertEqual(
                report["ledger"]["reused_workorders"],
                sorted(report["ledger"]["reused_workorders"]),
            )
            self.assertEqual(
                report["ledger"]["recomputed_workorders"],
                sorted(report["ledger"]["recomputed_workorders"]),
            )
            current = sorted(row["workorder_id"] for row in report["workorders"]) + [
                row["workorder_id"] for row in report["failures"]
            ]
            self.assertEqual(
                sorted(current),
                sorted(report["ledger"]["reused_workorders"] + report["ledger"]["recomputed_workorders"]),
            )
            if index == 1:
                self.assertEqual(report["ledger"]["recomputed_workorders"], ["WO-0002"])
                self.assertEqual(len(report["ledger"]["reused_workorders"]), 7)
            if index == 4:
                self.assertEqual(report["ledger"]["recomputed_workorders"], ["WO-0001"])

        full_data = self.root / "full-data"
        full_data.mkdir()
        for name in ("evidence.jsonl", "rules.json", "config.json", "input_meta.json", "edits.json", "injections.json"):
            shutil.copy2(self.data / name, full_data / name)
        full = run_workbench(full_data, self.root / "state-full", mode="serial")
        self.assertEqual(business_projection(reports[-1]), business_projection(full))

    def test_injections_are_isolated_and_mapped(self) -> None:
        report = run_workbench(
            self.data,
            self.root / "state-injected",
            mode="parallel",
            injections_path=self.data / "injections.json",
        )
        self.assertFalse(report["fatal"])
        self.assertEqual([row["workorder_id"] for row in report["workorders"]], ["WO-0006", "WO-0007", "WO-0008"])
        failures = [(row["workorder_id"], row["code"], row.get("injection_id")) for row in report["failures"]]
        self.assertEqual(
            failures,
            [
                ("WO-0001", "E_UNKNOWN_EVIDENCE_TYPE", "inj-001"),
                ("WO-0002", "E_CONFIDENCE_RANGE", "inj-002"),
                ("WO-0003", "E_EVENT_TIME_INVALID", "inj-003"),
                ("WO-0004", "E_HALF_LIFE_ZERO", "inj-004"),
                ("WO-0005", "E_RELATION_UNKNOWN_TYPE", "inj-005"),
            ],
        )
        self.assertEqual([row["injection_id"] for row in report["injections"]["hits"]], [f"inj-{i:03d}" for i in range(1, 6)])
        self.assertEqual(report["injections"]["misses"], [])

    def test_snapshot_restart_rejects_changed_config(self) -> None:
        first = self.run_report("state-snapshot", mode="serial")
        self.assertFalse(first["fatal"])
        state = self.root / "state-snapshot"
        pointer = json.loads((state / "current.json").read_text(encoding="utf-8"))
        snapshot_dir = state / "snapshots" / pointer["snapshot"]
        for name in ("manifest.json", "load.jsonl", "decay.jsonl", "relations.jsonl", "aggregation.jsonl"):
            self.assertTrue((snapshot_dir / name).is_file())
        self.assertEqual(len(first["snapshot"]["commits"]), 4)

        config_path = self.data / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["concurrency"] = 5
        config_path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        rejected = self.run_report("state-snapshot", mode="serial")
        self.assertTrue(rejected["fatal"])
        self.assertEqual(rejected["failures"][0]["code"], "E_SNAPSHOT_CONFIG_MISMATCH")

        rebuilt = run_workbench(
            self.data,
            state,
            mode="serial",
            rebuild_state=True,
        )
        self.assertFalse(rebuilt["fatal"])

    def test_validation_failures_are_not_silently_skipped(self) -> None:
        evidence_path = self.data / "evidence.jsonl"
        with evidence_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "workorder_id": "WO-0001",
                        "evidence_seq": 99,
                        "evidence_type": "not_in_weights",
                        "source_role": "missing_role",
                        "event_ms": -5,
                        "terms": [],
                        "confidence": 1.5,
                    }
                )
                + "\n"
            )
        records, structural = load_evidence_file(evidence_path)
        self.assertTrue(records)
        report = self.run_report("state-invalid")
        codes = {row["code"] for row in report["failures"] if row["workorder_id"] == "WO-0001"}
        self.assertTrue(
            {
                "E_UNKNOWN_EVIDENCE_TYPE",
                "E_UNKNOWN_SOURCE_ROLE",
                "E_EVENT_TIME_INVALID",
                "E_TERMS_EMPTY",
                "E_CONFIDENCE_RANGE",
            }.issubset(codes)
        )
        self.assertEqual(report["summary"]["successful_count"], 7)

        config_path = self.data / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["fixed_decimal_width"] = 5
        config_path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        fatal = self.run_report("state-bad-config")
        self.assertTrue(fatal["fatal"])
        self.assertEqual(fatal["failures"][0]["code"], "E_CONFIG_FIXED_WIDTH")
        self.assertFalse(structural)


if __name__ == "__main__":
    unittest.main()
