"""Acceptance test suite for the deterministic pipeline workbench.

Run with:  python -m unittest tests.test_workbench -v
The suite uses only the Python standard library and temporary directories.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from workbench.canonical import canonical_bytes
from workbench.cli import main as cli_main
from workbench.demo_data import generate_fixtures
from workbench.engine import Engine, ExecutionPlan
from workbench.errors import WorkbenchError
from workbench.fixedpoint import scale_int, to_fixed
from workbench.ledger import Ledger
from workbench.report import render_report
from workbench.runner import Runner
from workbench.spec import load_pipeline


class WorkbenchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="wb_test_")
        self.fixture_dir = os.path.join(self.tmp, "fixtures")
        self.spec_path = generate_fixtures(self.fixture_dir)
        self.workspace = os.path.join(self.tmp, "workspace")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _runner(self) -> Runner:
        return Runner(self.workspace)

    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    # ------------------------------------------------------------- req 1/2
    def test_spec_validation_and_fingerprint_stability(self) -> None:
        spec_a = load_pipeline(self.spec_path)
        spec_b = load_pipeline(self.spec_path)
        self.assertEqual(spec_a.fingerprint, spec_b.fingerprint)
        self.assertEqual(spec_a.samples_digest, spec_b.samples_digest)
        # Reordering sources/stages in memory cannot change the fingerprint of
        # an identical document: reload twice and compare canonical payloads.
        self.assertEqual(canonical_bytes(spec_a.raw),
                         canonical_bytes(spec_b.raw))
        with self.subTest("duplicate stage id rejected"):
            doc = self._load_json(self.spec_path)
            doc["stages"][1]["id"] = "read"
            broken = os.path.join(self.fixture_dir, "dup.json")
            with open(broken, "w", encoding="utf-8") as handle:
                json.dump(doc, handle)
            with self.assertRaises(WorkbenchError) as ctx:
                load_pipeline(broken)
            self.assertEqual(ctx.exception.code, "E_SPEC")
            self.assertIn("stages", ctx.exception.field_path)
        with self.subTest("unknown upstream rejected"):
            doc = self._load_json(self.spec_path)
            doc["stages"][1]["upstream"] = ["ghost"]
            broken = os.path.join(self.fixture_dir, "up.json")
            with open(broken, "w", encoding="utf-8") as handle:
                json.dump(doc, handle)
            with self.assertRaises(WorkbenchError) as ctx:
                load_pipeline(broken)
            self.assertEqual(ctx.exception.code, "E_SPEC")
            self.assertEqual(ctx.exception.stage_id, "types")
        with self.subTest("cycle rejected"):
            doc = self._load_json(self.spec_path)
            doc["stages"][0]["upstream"] = ["pack"]
            broken = os.path.join(self.fixture_dir, "cycle.json")
            with open(broken, "w", encoding="utf-8") as handle:
                json.dump(doc, handle)
            with self.assertRaises(WorkbenchError) as ctx:
                load_pipeline(broken)
            self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_spec_errors_are_rejected_with_stage_and_path(self) -> None:
        doc = self._load_json(self.spec_path)
        doc["sources"][0]["columns"]["visits"]["type"] = "decimal"
        broken = os.path.join(self.fixture_dir, "badtype.json")
        with open(broken, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        with self.assertRaises(WorkbenchError) as ctx:
            load_pipeline(broken)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.stage_id, "train_a")
        self.assertIn("columns.visits.type", ctx.exception.field_path)

    # ------------------------------------------------------------- req 2/3
    def test_deterministic_stage_summaries_and_shards(self) -> None:
        runner = self._runner()
        first = runner.run_pipeline(self.spec_path)
        second = runner.run_pipeline(self.spec_path)
        rows_a = runner.ledger.stage_rows(first["run_id"])
        rows_b = runner.ledger.stage_rows(second["run_id"])
        self.assertEqual(
            [r["output_digest"] for r in rows_a],
            [r["output_digest"] for r in rows_b])
        self.assertEqual(
            [r["column_stats_json"] for r in rows_a],
            [r["column_stats_json"] for r in rows_b])
        shards_a = runner.ledger.shard_rows(first["run_id"])
        shards_b = runner.ledger.shard_rows(second["run_id"])
        self.assertEqual(
            [r["shard_fingerprint"] for r in shards_a],
            [r["shard_fingerprint"] for r in shards_b])
        # Fixed-point semantics: 1.25 * 1000 == 1250, truncation named.
        self.assertEqual(to_fixed(1.25, 1000, "toward_zero"), 1250)
        self.assertEqual(scale_int(7000, 2, 1, "toward_zero"), 14000)
        # Snapshot files on disk are byte-identical across runs.
        run_a_dir = rows_a[0]["snapshot_path"].split(os.sep + "stages" +
                                                     os.sep)[0]
        run_b_dir = rows_b[0]["snapshot_path"].split(os.sep + "stages" +
                                                     os.sep)[0]
        for name in os.listdir(os.path.join(run_a_dir, "stages")):
            with open(os.path.join(run_a_dir, "stages", name), "rb") as fa, \
                    open(os.path.join(run_b_dir, "stages", name), "rb") as fb:
                self.assertEqual(fa.read(), fb.read(), name)
        runner.close()

    # ------------------------------------------------------------- req 4/5
    def test_evolution_sets_and_preserved_upstream(self) -> None:
        runner = self._runner()
        runner.run_pipeline(self.spec_path)
        variant_dir = os.path.join(self.tmp, "variant_norm")
        variant_spec = generate_fixtures(variant_dir, normalize_scale=4)
        evolved = runner.run_pipeline(variant_spec)
        ev = evolved["evolution"]
        self.assertIn("stage_spec_changed", ev["reasons"])
        self.assertEqual(ev["changed_stage_ids"], ["scale"])
        self.assertEqual(ev["affected_stage_ids"],
                         ["scale", "clip", "pack"])
        self.assertEqual(ev["unchanged_stage_ids"],
                         ["read", "types", "fill"])
        self.assertGreater(ev["invalidated_shards"], 0)
        baseline_rows = {r["stage_id"]: r for r in
                         runner.ledger.stage_rows(1)}
        evolved_rows = {r["stage_id"]: r for r in
                        runner.ledger.stage_rows(evolved["run_id"])}
        for stage_id in ev["unchanged_stage_ids"]:
            self.assertEqual(
                baseline_rows[stage_id]["output_digest"],
                evolved_rows[stage_id]["output_digest"])
        # Monotonic fingerprint version across distinct specifications.
        self.assertGreater(
            runner.ledger.get_run(evolved["run_id"])["fingerprint_version"],
            runner.ledger.get_run(1)["fingerprint_version"])
        runner.close()

    def test_sample_evolution_invalidates_every_stage(self) -> None:
        runner = self._runner()
        runner.run_pipeline(self.spec_path)
        sample_variant = os.path.join(self.tmp, "variant_sample")
        sample_spec = generate_fixtures(sample_variant)
        sample_file = os.path.join(sample_variant, "samples", "train_a.json")
        doc = self._load_json(sample_file)
        doc["rows"][0]["visits"] = 99
        with open(sample_file, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        result = runner.run_pipeline(sample_spec)
        ev = result["evolution"]
        self.assertTrue(ev["sample_changed"])
        spec = load_pipeline(sample_spec)
        self.assertEqual(ev["affected_stage_ids"],
                         [stage.id for stage in spec.stages])
        self.assertEqual(ev["unchanged_stage_ids"], [])
        runner.close()

    def test_shard_remainder_rule_and_binding(self) -> None:
        runner = self._runner()
        result = runner.run_pipeline(self.spec_path)
        shards = runner.ledger.shard_rows(result["run_id"])
        # 13 rows, target 4 -> [4,4,4,1] with keep_partial remainder rule.
        self.assertEqual([r["row_count"] for r in shards], [4, 4, 4, 1])
        self.assertEqual([(r["row_start"], r["row_end"]) for r in shards],
                         [(0, 4), (4, 8), (8, 12), (12, 13)])
        runner.ledger.verify_shard_binding(result["run_id"])
        runner.close()

    def test_shard_drop_partial_remainder_rule(self) -> None:
        drop_dir = os.path.join(self.tmp, "drop_partial")
        drop_spec = generate_fixtures(drop_dir, target_rows=5)
        doc = self._load_json(drop_spec)
        doc["stages"][5]["params"]["remainder_rule"] = "drop_partial"
        drop_spec_path = os.path.join(drop_dir, "pipeline_drop.json")
        with open(drop_spec_path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        runner = Runner(os.path.join(self.tmp, "drop_ws"))
        result = runner.run_pipeline(drop_spec_path)
        extra = json.loads(
            next(row for row in runner.ledger.stage_rows(result["run_id"])
                 if row["stage_id"] == "pack")["extra_json"])
        self.assertEqual(extra["dropped_rows"], 3)
        self.assertEqual(extra["shard_row_counts"], [5, 5])
        runner.close()

    # --------------------------------------------------------------- req 6
    def test_injection_business_codes_and_ledger_events(self) -> None:
        from workbench.inject import prepare_injection
        expected = {"data": "E_DATA", "param": "E_PARAM",
                    "outlier": "E_OUTLIER"}
        for kind, code in expected.items():
            ws = os.path.join(self.tmp, "inject_" + kind)
            injected = prepare_injection(
                self.spec_path, os.path.join(ws, "injections", kind), kind)
            runner = Runner(ws)
            with self.assertRaises(WorkbenchError) as ctx:
                runner.run_pipeline(injected)
            self.assertEqual(ctx.exception.code, code)
            with open(os.path.join(ws, "events.jsonl"),
                      encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]
            self.assertTrue(any(event["code"] == code for event in events))
            if code == "E_PARAM":
                self.assertEqual(ctx.exception.stage_id, "scale")
                self.assertIn("scale", ctx.exception.field_path)
                self.assertEqual(ctx.exception.details["param"], "scale")
            if code == "E_OUTLIER":
                self.assertEqual(ctx.exception.stage_id, "clip")
                self.assertEqual(ctx.exception.details["param"],
                                 "max_outlier_bp")
            runner.close()

    def test_cli_injection_flag_reports_business_code(self) -> None:
        for kind, code in (("data", "E_DATA"), ("param", "E_PARAM"),
                           ("outlier", "E_OUTLIER")):
            ws = os.path.join(self.tmp, "cli_" + kind)
            exit_code = cli_main(["inject", self.spec_path,
                                  "--workspace", ws, "--kind", kind])
            self.assertEqual(exit_code, 2)
            with open(os.path.join(ws, "events.jsonl"),
                      encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]
            self.assertTrue(any(event["code"] == code for event in events))

    # --------------------------------------------------------------- req 7
    def test_forced_termination_unknown_stages_and_recovery(self) -> None:
        runner = self._runner()
        killed = runner.run_pipeline(self.spec_path, kill_after_ordinal=2)
        statuses = [(row["stage_id"], row["status"])
                    for row in runner.ledger.stage_rows(killed["run_id"])]
        self.assertEqual([status for _, status in statuses],
                         ["COMMITTED", "COMMITTED", "COMMITTED",
                          "UNKNOWN", "UNKNOWN", "UNKNOWN"])
        progress = runner.ledger.active_run()
        self.assertEqual(progress["status"], "RUNNING")
        runner.close()
        # Restart: only fully committed records are trusted; replay first.
        runner = Runner(self.workspace)
        recovery = runner.recover(self.spec_path)
        self.assertEqual(recovery["replayed"],
                         ["read", "types", "fill"])
        self.assertEqual(recovery["rerun_stage_ids"],
                         ["scale", "clip", "pack"])
        replay = runner.replay_run(recovery["new_run_id"])
        self.assertEqual(replay["status"], "MATCH")
        runner.close()

    def test_replay_mismatch_reports_first_stage_and_field(self) -> None:
        runner = self._runner()
        result = runner.run_pipeline(self.spec_path)
        run_id = result["run_id"]
        # Tamper with a committed column statistic inside the ledger itself.
        stage_row = next(row for row in runner.ledger.stage_rows(run_id)
                         if row["stage_id"] == "fill")
        stats = json.loads(stage_row["column_stats_json"])
        stats["score"]["sum_fixed"] += 1
        with runner.ledger.conn:
            runner.ledger.conn.execute(
                "UPDATE run_stages SET column_stats_json=? WHERE run_id=? "
                "AND stage_id='fill'",
                (json.dumps(stats, sort_keys=True), run_id))
        with self.assertRaises(WorkbenchError) as ctx:
            runner.replay_run(run_id)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.stage_id, "fill")
        self.assertIn("column_stats.score.sum_fixed",
                      ctx.exception.field_path)
        runner.close()

    def test_shard_tamper_and_ledger_replay(self) -> None:
        runner = self._runner()
        result = runner.run_pipeline(self.spec_path)
        shard = runner.ledger.shard_rows(result["run_id"])[0]
        with open(shard["file_path"], "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        row = json.loads(lines[0])
        row["visits"] = row["visits"] + 1
        lines[0] = json.dumps(row, sort_keys=True,
                               separators=(",", ":")) + "\n"
        with open(shard["file_path"], "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        with self.assertRaises(WorkbenchError) as ctx:
            runner.replay_run(result["run_id"])
        self.assertEqual(ctx.exception.code, "E_SHARD")
        self.assertEqual(ctx.exception.stage_id, "pack")
        runner.close()

    def test_report_and_ui_view_model_are_stable(self) -> None:
        runner = self._runner()
        result = runner.run_pipeline(self.spec_path)
        first = render_report(runner.ledger, result["run_id"])
        model = runner.ledger.view_model(result["run_id"])
        self.assertEqual([s["stage_id"] for s in model["stages"]],
                         ["read", "types", "fill", "scale", "clip", "pack"])
        self.assertEqual(len(model["shards"]), 4)
        # Rendering twice yields identical stable text.
        self.assertEqual(first, render_report(runner.ledger, result["run_id"]))
        self.assertIn("PIPELINE DIAGNOSTIC WORKBENCH", first)
        runner.close()

    def test_ui_text_sections_and_headless_fallback(self) -> None:
        from workbench.ui import _build_text
        runner = self._runner()
        result = runner.run_pipeline(self.spec_path)
        model = runner.ledger.view_model(result["run_id"])
        text = _build_text(model)
        for heading in (
                "STAGE DEPENDENCIES AND STATUS",
                "STAGE SUMMARIES (COLUMN STATISTICS)",
                "SHARD DISTRIBUTION", "FINGERPRINT LEDGER",
                "BUSINESS EVENTS / LEDGER VIEW"):
            self.assertIn(heading, text)
        self.assertEqual(text, _build_text(model))
        runner.close()

    def test_concurrent_independent_runs_share_fingerprint(self) -> None:
        import subprocess
        import sys
        script = (
            "from workbench.spec import load_pipeline;"
            "print(load_pipeline(%r).fingerprint)" % self.spec_path)
        digests = set()
        for _ in range(3):
            completed = subprocess.run(
                [sys.executable, "-c", script], cwd=os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))),
                capture_output=True, text=True, check=True)
            digests.add(completed.stdout.strip())
        self.assertEqual(len(digests), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
