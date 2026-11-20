from __future__ import annotations

import copy
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from deploy_workbench.canonical import canonical_json
from deploy_workbench.engine import resolve_all, resolve_environment
from deploy_workbench.errors import E_BIND, E_ENV, E_GRID, E_REPLAY, E_SPEC, WorkbenchError
from deploy_workbench.fixed import FixedQuantizer
from deploy_workbench.ledger import Ledger
from deploy_workbench.service import WorkbenchService
from deploy_workbench.snapshot import read_snapshot, write_snapshot
from deploy_workbench.spec import load_spec, write_injected_spec


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_spec.json"


def read_sample() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


class TemporarySpec:
    def __init__(self, directory: Path, raw: dict | None = None) -> None:
        self.path = directory / "spec.json"
        self.path.write_text(json.dumps(read_sample() if raw is None else raw, ensure_ascii=False, indent=2), encoding="utf-8")


class SpecValidationTests(unittest.TestCase):
    def assert_error(self, raw: dict, code: str, *, environment: str | None = None, path_contains: str | None = None):
        with tempfile.TemporaryDirectory() as tmp:
            spec = TemporarySpec(Path(tmp), raw)
            with self.assertRaises(WorkbenchError) as caught:
                load_spec(spec.path)
        self.assertEqual(caught.exception.code, code)
        if environment is not None:
            self.assertEqual(caught.exception.environment, environment)
        if path_contains is not None:
            self.assertIn(path_contains, caught.exception.path)
        return caught.exception

    def test_duplicate_environment_id_is_spec_rejection(self):
        raw = read_sample()
        raw["environments"][1]["id"] = raw["environments"][0]["id"]
        error = self.assert_error(raw, E_SPEC, environment="dev", path_contains=".id")
        self.assertEqual(error.parent_code, E_SPEC)

    def test_negative_depth_is_spec_rejection(self):
        raw = read_sample()
        raw["environments"][0]["depth"] = -1
        self.assert_error(raw, E_SPEC, environment="dev", path_contains=".depth")

    def test_unsupported_numeric_type_is_spec_rejection(self):
        raw = read_sample()
        raw["environments"][0]["entries"]["base"]["cpu_cores"] = "4"
        self.assert_error(raw, E_SPEC, environment="dev", path_contains="cpu_cores")

    def test_formula_missing_key_is_spec_rejection(self):
        raw = read_sample()
        raw["tasks"][0]["formula"] = "1.0*not_declared"
        self.assert_error(raw, E_SPEC, path_contains=".formula")

    def test_formula_without_keys_is_spec_rejection(self):
        raw = read_sample()
        raw["tasks"][0]["formula"] = "1 + 2"
        self.assert_error(raw, E_SPEC, path_contains=".formula")

    def test_duplicate_task_id_is_spec_rejection(self):
        raw = read_sample()
        raw["tasks"][1]["id"] = raw["tasks"][0]["id"]
        self.assert_error(raw, E_SPEC, path_contains=".id")

    def test_binding_missing_source_is_bind_code(self):
        raw = read_sample()
        raw["environments"][0]["bindings"]["headroom"]["allowed_source_ids"].append("ghost")
        error = self.assert_error(raw, E_BIND, environment="dev", path_contains="allowed_source_ids")
        self.assertEqual(error.parent_code, E_SPEC)

    def test_invalid_scale_power_is_grid_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(WorkbenchError) as caught:
                load_spec(SAMPLE, scale_power_override=99)
        self.assertEqual(caught.exception.code, E_GRID)
        self.assertEqual(caught.exception.path, "$.scale_power")

    def test_malformed_environment_json_is_env_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "broken.json"
            write_injected_spec(SAMPLE, target, "env")
            with self.assertRaises(WorkbenchError) as caught:
                load_spec(target)
        self.assertEqual(caught.exception.code, E_ENV)

    def test_bind_injection_changes_input_file_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bind.json"
            write_injected_spec(SAMPLE, target, "bind")
            with self.assertRaises(WorkbenchError) as caught:
                load_spec(target)
        self.assertEqual(caught.exception.code, E_BIND)
        self.assertEqual(caught.exception.details["missing_source_id"], "__missing_injected_source__")


class DeterministicResolutionTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec(SAMPLE)
        self.quantizer = FixedQuantizer(self.spec.scale_power, self.spec.overflow_policy)

    def test_chain_order_binding_and_byte_stability(self):
        first = resolve_all(self.spec, self.quantizer)
        second = resolve_all(self.spec, self.quantizer)
        self.assertEqual(canonical_json(first), canonical_json(second))
        dev_headroom = first["dev"]["keys"]["headroom"]
        self.assertEqual(
            [step["source_id"] for step in dev_headroom["chain"]],
            ["rt_dev", "file_dev", "base"],
        )
        self.assertEqual(dev_headroom["effective_source_id"], "file_dev")
        self.assertFalse(dev_headroom["chain"][0]["selected"])
        self.assertFalse(dev_headroom["chain"][0]["binding_satisfied"])
        self.assertTrue(dev_headroom["chain"][1]["selected"])

    def test_declaration_list_order_does_not_change_chains(self):
        raw = read_sample()
        raw["sources"] = list(reversed(raw["sources"]))
        raw["environments"] = list(reversed(raw["environments"]))
        raw["tasks"] = list(reversed(raw["tasks"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reordered.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            reordered = load_spec(path)
        first = resolve_all(self.spec, self.quantizer)
        second = resolve_all(reordered, FixedQuantizer(reordered.scale_power, reordered.overflow_policy))
        self.assertEqual(canonical_json(first), canonical_json(second))

    def test_binding_can_skip_higher_priority_file_source(self):
        prod_mem_floor = resolve_all(self.spec, self.quantizer)["prod"]["keys"]["mem_floor"]
        self.assertEqual(
            [step["source_id"] for step in prod_mem_floor["chain"]],
            ["file_prod_primary", "file_prod_fallback", "base"],
        )
        self.assertEqual(prod_mem_floor["effective_source_id"], "file_prod_fallback")


class FixedPointAndAttributionTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec(SAMPLE)
        self.quantizer = FixedQuantizer(self.spec.scale_power, self.spec.overflow_policy)
        self.result = resolve_all(self.spec, self.quantizer)

    def test_actual_error_is_inside_declared_bound(self):
        for environment in self.result.values():
            for task in environment["tasks"].values():
                if task["status"] != "ok":
                    continue
                self.assertLessEqual(Decimal(task["actual_error"]), Decimal(task["error_bound"]))
                self.assertLessEqual(Decimal(task["actual_error"]), Decimal(task["theoretical_error_bound"]))
                self.assertTrue(task["bounds_valid"])

    def test_segments_sum_to_error_bound(self):
        task = self.result["dev"]["tasks"]["capacity"]
        segment_sum = sum(Decimal(segment["contribution"]) for segment in task["attribution"]["segments"])
        self.assertEqual(segment_sum, Decimal(task["error_bound"]))
        self.assertEqual(
            [segment["segment"] for segment in task["attribution"]["segments"]],
            ["source_quantization", "formula_truncation", "aggregation_synthesis"],
        )
        self.assertIn(task["attribution"]["dominant_segment"], {"source_quantization", "formula_truncation", "aggregation_synthesis"})

    def test_intermediates_are_explicit(self):
        task = self.result["prod"]["tasks"]["capacity"]
        keys = {item["key"] for item in task["intermediates"]}
        self.assertEqual(keys, set(task["referenced_keys"]))
        for item in task["intermediates"]:
            self.assertIn("value_residual", item)
            self.assertIn("coefficient_residual", item)
            self.assertIn("source_quantization_error", item)

    def test_comparison_inside_combined_bound_uses_stable_key_id(self):
        comparisons = self.result["staging"]["comparisons"]
        uncertain = next(
            item for item in comparisons if {item["key_a"], item["key_b"]} == {"mem_floor", "memory_gb"}
        )
        self.assertTrue(uncertain["affected_by_error_bound"])
        self.assertEqual(uncertain["ordered_keys"], ["mem_floor", "memory_gb"])
        self.assertEqual(uncertain["decision_basis"], "error_bound_then_key_id")

    def test_per_key_quantization_bound(self):
        for environment in self.result.values():
            for key in environment["keys"].values():
                if key["resolved"]:
                    self.assertEqual(key["quantization_error_bound"], "0.001")
                    self.assertLessEqual(Decimal(key["quantization_error_abs"]), Decimal("0.001"))

    def test_task_execution_order_does_not_change_payload(self):
        raw = read_sample()
        raw["tasks"] = list(reversed(raw["tasks"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            changed = load_spec(path)
        other = resolve_all(changed, FixedQuantizer(changed.scale_power, changed.overflow_policy))
        for env_id in self.result:
            self.assertEqual(
                canonical_json(self.result[env_id]["tasks"]),
                canonical_json(other[env_id]["tasks"]),
            )


class VersionAndLedgerTests(unittest.TestCase):
    def test_version_bump_replace_stale_view_and_readonly_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            spec_path = root / "spec.json"
            raw = read_sample()
            spec_path.write_text(json.dumps(raw), encoding="utf-8")
            service = WorkbenchService(spec_path, state)
            first = service.run_all()
            self.assertEqual(first["status"], "complete")
            self.assertEqual(service.versions.version_tuple(), (1, 1, 1))
            old_payload = service.get_current_view()["environments"][0]["payload"]
            old_record_digest = first["ledger_content_digest"]
            service.close()

            raw["environments"][0]["entries"]["file_dev"]["headroom"] = 0.5
            spec_path.write_text(json.dumps(raw), encoding="utf-8")
            service = WorkbenchService(spec_path, state)
            self.assertEqual(service.versions.environment_version, 2)
            self.assertEqual(service.versions.priority_version, 1)
            stale_view = service.get_current_view()
            self.assertTrue(stale_view["stale"])
            self.assertEqual(stale_view["environments"], [])
            second = service.run_all()
            self.assertEqual(second["status"], "complete")
            self.assertEqual(second["stats"]["affected_environment_count"], 1)
            self.assertGreaterEqual(second["stats"]["affected_key_count"], 1)
            self.assertIn("dev", second["stats"]["affected_environments"])
            current = service.get_current_view()
            self.assertFalse(current["stale"])
            self.assertEqual(current["run"]["run_id"], second["run_id"])
            self.assertTrue(all(record["payload"] != old_payload for record in current["environments"]))
            versions = service.list_versions()
            old_version = next(item for item in versions if item["run_id"] == first["run_id"])
            self.assertTrue(old_version["stale"])
            self.assertTrue(old_version["read_only"])
            historical = service.get_historical_view(first["run_id"])
            self.assertTrue(historical["read_only"])
            with self.assertRaises(sqlite3.DatabaseError):
                with service.ledger.connection:
                    service.ledger.connection.execute(
                        "UPDATE environment_records SET content_digest = 'tampered' WHERE run_id = ?",
                        (first["run_id"],),
                    )
            self.assertNotEqual(old_record_digest, second["ledger_content_digest"])
            service.close()

    def test_priority_and_scale_versions_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            spec_path = root / "spec.json"
            raw = read_sample()
            spec_path.write_text(json.dumps(raw), encoding="utf-8")
            service = WorkbenchService(spec_path, state)
            service.run_all()
            service.close()

            raw["environments"][2]["source_priority"] = ["rt_prod", "file_prod_fallback", "file_prod_primary", "base"]
            spec_path.write_text(json.dumps(raw), encoding="utf-8")
            service = WorkbenchService(spec_path, state)
            self.assertEqual(service.versions.version_tuple(), (1, 2, 1))
            result = service.run_all()
            self.assertEqual(result["status"], "complete")
            service.close()

            service = WorkbenchService(spec_path, state, scale_power_override=4)
            self.assertEqual(service.versions.version_tuple(), (1, 2, 2))
            result = service.run_all()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["versions"]["scale_version"], 2)
            service.close()


class RecoveryTests(unittest.TestCase):
    def test_forced_interruption_recovers_only_committed_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            spec = load_spec(SAMPLE)
            ledger = Ledger(state / "ledger.sqlite3")
            fingerprints = spec.fingerprints()
            versions = ledger.plan_versions(fingerprints)
            ledger.apply_pending_versions(versions)
            expected = [env.id for env in spec.environment_order()]
            run_id = ledger.begin_run(versions, expected, {"state": "computing"})
            ledger.mark_computing(run_id, "dev")
            quantizer = FixedQuantizer(spec.scale_power, spec.overflow_policy)
            dev_result = resolve_environment(spec, spec.environments["dev"], quantizer)
            ledger.commit_environment(run_id, "dev", dev_result, versions, sequence=1)
            ledger.mark_computing(run_id, "staging")
            # Simulate SIGKILL here: no finalize, no clean exit report.
            ledger.close()

            ledger = Ledger(state / "ledger.sqlite3")
            report = ledger.recover_incomplete()
            self.assertEqual(report["recovered_runs"][0]["completed_environments"], ["dev"])
            self.assertEqual(report["recovered_runs"][0]["unknown_environments"], ["prod", "staging"])
            run = ledger.get_run(run_id)
            self.assertEqual(run["status"], "interrupted")
            statuses = {
                row["env_id"]: row["status"]
                for row in ledger.connection.execute(
                    "SELECT env_id, status FROM environment_progress WHERE run_id = ?", (run_id,)
                )
            }
            self.assertEqual(statuses["dev"], "committed")
            self.assertEqual(statuses["staging"], "unknown")
            self.assertEqual(statuses["prod"], "unknown")
            view = ledger.current_view()
            self.assertEqual(view["completed_environments"], ["dev"])
            self.assertEqual(view["unknown_environments"], ["prod", "staging"])
            self.assertIn("no forced-exit report", report["note"])
            ledger.close()

            service = WorkbenchService(SAMPLE, state)
            rerun = service.run_all()
            self.assertEqual(rerun["status"], "complete")
            self.assertNotEqual(rerun["run_id"], run_id)
            self.assertEqual(service.get_current_view()["completed_environments"], ["dev", "prod", "staging"])
            service.close()


class SnapshotReplayTests(unittest.TestCase):
    def _run_service(self, root: Path) -> WorkbenchService:
        state = root / "state"
        service = WorkbenchService(SAMPLE, state)
        service.run_all()
        return service

    def test_replay_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._run_service(Path(tmp))
            result = service.replay_snapshot()
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["environment_count"], 3)
            self.assertEqual(result["snapshot_conclusion_digest"], result["replay_conclusion_digest"])
            service.close()

    def test_first_key_field_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._run_service(Path(tmp))
            snapshot = read_snapshot(service.latest_snapshot_path)
            snapshot["environments"]["dev"]["keys"]["cpu_cores"]["effective_fixed"] += 1
            write_snapshot(service.latest_snapshot_path, snapshot)
            with self.assertRaises(WorkbenchError) as caught:
                service.replay_snapshot()
            self.assertEqual(caught.exception.code, E_REPLAY)
            self.assertEqual(caught.exception.environment, "dev")
            self.assertEqual(caught.exception.details["key"], "cpu_cores")
            self.assertEqual(caught.exception.details["field"], "effective_fixed")
            self.assertEqual(caught.exception.details["expected"], snapshot["environments"]["dev"]["keys"]["cpu_cores"]["effective_fixed"])
            service.close()

    def test_snapshot_version_mismatch_replaces_whole_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            spec_path = root / "spec.json"
            raw = read_sample()
            spec_path.write_text(json.dumps(raw), encoding="utf-8")
            service = WorkbenchService(spec_path, state)
            first = service.run_all()
            old_snapshot = state / "snapshots" / "old.json"
            old_snapshot.write_bytes(service.latest_snapshot_path.read_bytes())
            service.close()

            raw["environments"][0]["entries"]["file_dev"]["headroom"] = 0.25
            spec_path.write_text(json.dumps(raw), encoding="utf-8")
            service = WorkbenchService(spec_path, state)
            result = service.replay_snapshot(old_snapshot)
            self.assertEqual(result["status"], "replaced")
            self.assertIn("version_tuple", result["mismatch_fields"])
            self.assertEqual(result["replacement"]["status"], "complete")
            self.assertGreater(result["replacement"]["run_id"], first["run_id"])
            current = service.get_current_view()
            self.assertFalse(current["stale"])
            self.assertEqual(current["run"]["run_id"], result["replacement"]["run_id"])
            service.close()

    def test_missing_completed_environment_is_replay_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._run_service(Path(tmp))
            snapshot = read_snapshot(service.latest_snapshot_path)
            snapshot["completed_environments"].remove("prod")
            snapshot["environments"].pop("prod")
            write_snapshot(service.latest_snapshot_path, snapshot)
            with self.assertRaises(WorkbenchError) as caught:
                service.replay_snapshot()
            self.assertEqual(caught.exception.code, E_REPLAY)
            self.assertEqual(caught.exception.environment, "prod")
            service.close()


class CliInjectionTests(unittest.TestCase):
    def _run(self, state: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "deploy_workbench", *args, "--spec", str(SAMPLE), "--state-root", str(state), "--max-json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_three_injections_have_business_codes_and_structured_logs(self):
        cases = [
            ("--inject-env", E_ENV, 2, "$"),
            ("--inject-bind", E_BIND, 3, "$.environments"),
            ("--inject-grid", E_GRID, 4, "$.scale_power"),
        ]
        for flag, code, exit_code, path_prefix in cases:
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp)
                completed = self._run(state, "run", flag)
                self.assertEqual(completed.returncode, exit_code, completed.stderr)
                report = json.loads(completed.stdout)
                self.assertEqual(report["code"], code)
                self.assertTrue(report["path"].startswith(path_prefix))
                log_path = state / "logs" / "workbench.jsonl"
                self.assertTrue(log_path.exists())
                records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(records[-1]["code"], code)
                self.assertEqual(records[-1]["event"], "domain_rejection")

    def test_valid_cli_scale_override_reaches_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            completed = self._run(state, "run", "--scale-power", "4")
            self.assertEqual(completed.returncode, 0, completed.stdout)
            report = json.loads(completed.stdout)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["versions"]["scale_version"], 1)


class PackagingTests(unittest.TestCase):
    def test_all_python_files_compile(self):
        files = sorted((ROOT / "deploy_workbench").glob("*.py"))
        self.assertTrue(files)
        for path in files:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
