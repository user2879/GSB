from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rope_diag.config import load_config
from rope_diag.pruning import load_pruning_policies
from rope_diag.pipeline import DiagnosticRunner
from rope_diag.strategy import select_strategy

from tests.support import config_dict, rewrite_json, sample_rows, write_case


def run_case(root: Path, *, force_full: bool = False):
    paths = root / "sample_data"
    return DiagnosticRunner(
        config_path=paths / "config.json",
        samples_path=paths / "samples.jsonl",
        pruning_path=paths / "pruning_strategies.json",
        injections_path=paths / "injections.json",
        workspace=root,
        output_path=root / "artifacts" / "report.json",
        force_full=force_full,
    ).run()


def stage_map(report: dict) -> dict[tuple[str, str], dict]:
    return {
        (event["scheme"], event["stage"]): event
        for event in report["cache"]["stage_events"]
    }


class PipelineTests(unittest.TestCase):
    def test_incremental_reuse_scale_invalidation_and_full_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_case(root)
            first = run_case(root)
            self.assertEqual(first["validation"]["sample_invalid_count"], 0)
            self.assertTrue(all(payload["status"] == "ok" for payload in first["schemes"].values()))
            self.assertTrue(all(
                payload["statistics"]["extrapolation"]["score_count"] > 0
                for payload in first["schemes"].values()
            ))

            second = run_case(root)
            second_stages = stage_map(second)
            self.assertEqual(second["semantic_digest"], first["semantic_digest"])
            for key in [
                ("rope", "position"),
                ("rope", "scoring"),
                ("rope", "trace"),
                ("rope", "sensitivity"),
                ("linear_interpolation", "position"),
                ("linear_interpolation", "scoring"),
                ("linear_interpolation", "trace"),
                ("linear_interpolation", "sensitivity"),
                ("ntk_aware", "position"),
                ("ntk_aware", "scoring"),
                ("ntk_aware", "trace"),
                ("ntk_aware", "sensitivity"),
            ]:
                self.assertEqual(second_stages[key]["status"], "reused", key)
                self.assertEqual(second_stages[key]["computed_count"], 0, key)

            config = json.loads(paths["config"].read_text(encoding="utf-8"))
            config["scaling_factor"] = 6.0
            rewrite_json(paths["config"], config)
            changed = run_case(root)
            changed_stages = stage_map(changed)
            self.assertEqual(changed_stages[("rope", "position")]["status"], "reused")
            self.assertEqual(changed_stages[("rope", "scoring")]["status"], "reused")
            self.assertEqual(changed_stages[("rope", "trace")]["status"], "reused")
            self.assertEqual(changed_stages[("linear_interpolation", "position")]["status"], "recomputed")
            self.assertEqual(changed_stages[("ntk_aware", "position")]["status"], "recomputed")
            self.assertNotEqual(changed["semantic_digest"], first["semantic_digest"])

            full = run_case(root, force_full=True)
            self.assertEqual(full["semantic_digest"], changed["semantic_digest"])
            self.assertTrue(all(
                event["status"] in {"recomputed", "injected_fault", "not_run"}
                for event in full["cache"]["stage_events"]
            ))

    def test_single_sample_replacement_only_recomputes_that_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_case(root)
            run_case(root)
            rows = [json.loads(line) for line in paths["samples"].read_text(encoding="utf-8").splitlines()]
            rows[1]["query_features"][0] += 0.25
            with paths["samples"].open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            changed = run_case(root)
            stages = stage_map(changed)
            for scheme in ("rope", "linear_interpolation", "ntk_aware"):
                position = stages[(scheme, "position")]
                self.assertEqual(position["cache_hit_count"], 3)
                self.assertEqual(position["computed_count"], 1)
                self.assertEqual(position["status"], "partial")
                self.assertEqual(stages[(scheme, "trace")]["status"], "recomputed")
                self.assertEqual(stages[(scheme, "sensitivity")]["status"], "recomputed")

    def test_budget_pending_resume_and_serial_concurrent_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = config_dict(point_budget=20, concurrency=1)
            write_case(root, config=config)
            first = run_case(root)
            self.assertGreater(first["scheduler"]["pending_count"], 0)
            self.assertTrue(any(
                failure["code"] == "E_POINT_BUDGET_PENDING"
                for failure in first["failures"]
            ))
            completed_first = len(first["snapshot"]["manifest"]["completed_tasks"])
            self.assertGreater(completed_first, 0)

            second = run_case(root)
            self.assertTrue(second["snapshot"]["resume_compatible"])
            self.assertGreater(len(second["snapshot"]["manifest"]["completed_tasks"]), completed_first)

        with tempfile.TemporaryDirectory() as temporary_one, tempfile.TemporaryDirectory() as temporary_four:
            root_one = Path(temporary_one)
            root_four = Path(temporary_four)
            write_case(root_one, config=config_dict(concurrency=1))
            write_case(root_four, config=config_dict(concurrency=4))
            serial = run_case(root_one)
            concurrent = run_case(root_four)
            self.assertEqual(serial["semantic_digest"], concurrent["semantic_digest"])
            self.assertEqual(
                serial["scheduler"]["budget_deductions"],
                concurrent["scheduler"]["budget_deductions"],
            )
            self.assertEqual(serial["failures"], concurrent["failures"])
            self.assertEqual(serial["comparison"], concurrent["comparison"])

    def test_injection_isolation_and_default_sample_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            injections = [
                {
                    "injection_id": "INJ-SAMPLE",
                    "scheme": "linear_interpolation",
                    "sample_id": "s0002",
                    "stage": "scoring",
                    "action": "raise",
                    "message": "sample fault",
                },
                {
                    "injection_id": "INJ-AGG",
                    "scheme": "ntk_aware",
                    "stage": "sensitivity",
                    "action": "nonfinite",
                    "message": "aggregate fault",
                },
            ]
            write_case(root, injections=injections)
            report = run_case(root)
            self.assertEqual(report["schemes"]["linear_interpolation"]["sample_summary"]["failed"], 1)
            self.assertEqual(report["schemes"]["linear_interpolation"]["sample_summary"]["completed"], 3)
            self.assertEqual(report["schemes"]["rope"]["status"], "ok")
            self.assertEqual(report["schemes"]["ntk_aware"]["sensitivity"]["status"], "E_INJECTED_FAULT")
            self.assertEqual(
                {item["injection_id"] for item in report["injections"]["triggered"]},
                {"INJ-SAMPLE", "INJ-AGG"},
            )
            self.assertTrue(all(
                item["code"] == "E_INJECTED_FAULT"
                for item in report["failures"]
            ))

    def test_no_extrapolation_continues_other_schemes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_case(root, samples=sample_rows(count=2, no_extrapolation=True))
            report = run_case(root)
            self.assertEqual(len(report["schemes"]), 3)
            for payload in report["schemes"].values():
                self.assertEqual(payload["status"], "statistics_error")
                self.assertEqual(
                    payload["statistics"]["status"], "E_NO_EXTRAPOLATION_SAMPLE"
                )
            no_ext_failures = [
                failure for failure in report["failures"]
                if failure["code"] == "E_NO_EXTRAPOLATION_SAMPLE"
            ]
            self.assertEqual(len(no_ext_failures), 3)

    def test_strategy_selector_changes_dimensions_chunks_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_case(root, config=config_dict(dimension=64, training_context_length=16, extrapolation_target_length=128))
            config, failures, _ = load_config(paths["config"])
            self.assertEqual(failures, [])
            policies, policy_failures, _ = load_pruning_policies(paths["pruning"])
            self.assertEqual(policy_failures, [])
            wide = select_strategy(
                config, sample_count=20, total_points=400, policies=policies
            )
            lean = select_strategy(
                config, sample_count=3, total_points=30, policies=policies
            )
            self.assertEqual(wide["strategy_id"], "S_WIDE")
            self.assertEqual(lean["strategy_id"], "S_LEAN")
            self.assertNotEqual(wide["dimension_indices"], lean["dimension_indices"])
            self.assertNotEqual(wide["chunk_boundaries"], lean["chunk_boundaries"])
            self.assertNotEqual(wide["strategy_fingerprint"], lean["strategy_fingerprint"])


if __name__ == "__main__":
    unittest.main()
