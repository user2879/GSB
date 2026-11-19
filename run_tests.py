#!/usr/bin/env python3
"""Acceptance & unit tests for the aggwin pipeline."""

import json
import os
import shutil
import sys
import tempfile
import unittest

from aggwin import faults
from aggwin.catalog import Catalog, canonical_query_id, param_set_fingerprint
from aggwin.cli import main as cli_main
from aggwin.engine import Engine
from aggwin.errors import PipelineError
from aggwin.persistence import (StateLogWriter, read_log, read_snapshot,
                                write_snapshot)
from aggwin.recovery import (compare_states, rebuild_from_snapshot,
                             replay_log, state_digest)
from aggwin.schema import load_queries_file, load_tables_file
from aggwin.state import EngineState
from aggwin.verify import compare_results, compare_runs

ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(ROOT, "sample", "tables.json")
QUERIES = os.path.join(ROOT, "sample", "queries.json")


def load():
    tables = load_tables_file(TABLES)
    query_set = load_queries_file(QUERIES, tables)
    return tables, query_set


class WorkdirMixin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aggwin-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_cli(self, *argv):
        code = cli_main(list(argv))
        return code

    def run_ok(self, workdir, *extra):
        argv = ["run", "--tables", TABLES, "--queries", QUERIES,
                "--workdir", workdir, *extra]
        self.assertEqual(self.run_cli(*argv), 0, msg=str(argv))

    def read_results(self, workdir):
        with open(os.path.join(workdir, "results.json"),
                  encoding="utf-8") as fh:
            return json.load(fh)


class TestFormatValidation(unittest.TestCase):
    def _expect_e_format(self, mutate, needle):
        tables = load_tables_file(TABLES)
        with open(TABLES, encoding="utf-8") as fh:
            doc = json.load(fh)
        mutate(doc)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(doc, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        with self.assertRaises(PipelineError) as ctx:
            load_tables_file(tmp.name)
        self.assertEqual(ctx.exception.code, "E_FORMAT")
        self.assertIn(needle, ctx.exception.details["field_path"])

    def test_unknown_field_rejected(self):
        def mutate(doc):
            doc["tables"][0]["bogus"] = 1
        self._expect_e_format(mutate, "$.tables[0].bogus")

    def test_duplicate_column_rejected(self):
        def mutate(doc):
            doc["tables"][0]["columns"].append(
                dict(doc["tables"][0]["columns"][0]))
        self._expect_e_format(mutate, "name")

    def test_null_bitmap_length_checked(self):
        def mutate(doc):
            for col in doc["tables"][0]["columns"]:
                if "null_bitmap" in col:
                    col["null_bitmap"] = col["null_bitmap"][:-1]
        self._expect_e_format(mutate, "null_bitmap")

    def test_timestamp_column_must_exist(self):
        def mutate(doc):
            doc["tables"][0]["timestamp_column"] = "nope"
        self._expect_e_format(mutate, "timestamp_column")

    def test_query_validation(self):
        tables = load_tables_file(TABLES)
        with open(QUERIES, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["queries"][1]["interval"]["preceding"] = -3
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(doc, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        with self.assertRaises(PipelineError) as ctx:
            load_queries_file(tmp.name, tables)
        self.assertEqual(ctx.exception.code, "E_FORMAT")
        self.assertIn("interval.preceding",
                      ctx.exception.details["field_path"])


class TestPersistence(WorkdirMixin):
    def test_log_versions_roundtrip(self):
        for fmt in (1, 2):
            log_dir = os.path.join(self.tmp, f"log-v{fmt}")
            writer = StateLogWriter(log_dir, format_version=fmt)
            writer.write_batch("events", [0, 1, 2], 3)
            writer.write_checkpoint({"state": {"cursors": {"events": 3}}})
            writer.write_batch("events", [3, 4], 5)
            records = read_log(log_dir)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0].payload["rows"], [0, 1, 2])
            self.assertEqual(records[2].payload["cursor_after"], 5)

    def test_log_truncation_reports_segment_and_offset(self):
        log_dir = os.path.join(self.tmp, "log-trunc")
        writer = StateLogWriter(log_dir)
        writer.write_batch("events", [0, 1], 2)
        faults.truncate_log_tail(log_dir, 5)
        with self.assertRaises(PipelineError) as ctx:
            read_log(log_dir)
        self.assertEqual(ctx.exception.code, "E_LOG")
        self.assertIn("segment", ctx.exception.details)
        self.assertIn("offset", ctx.exception.details)

    def test_snapshot_versions_normalise_identically(self):
        state = {"cursors": {"events": 4}, "applied": {"events": [[0, 4]]},
                 "aggs": {}, "windows": {}}
        records = []
        for fmt in (1, 2):
            path = os.path.join(self.tmp, f"snap-v{fmt}.json")
            write_snapshot(path, state, 7, f"snap-v{fmt}",
                           format_version=fmt)
            records.append(read_snapshot(path))
        self.assertEqual(records[0].state, records[1].state)
        self.assertEqual(records[0].query_param_version, 7)

    def test_snapshot_digest_corruption(self):
        path = os.path.join(self.tmp, "snap.json")
        write_snapshot(path, {"cursors": {}, "applied": {}, "aggs": {},
                              "windows": {}}, 1, "s1")
        faults.corrupt_snapshot_digest(path)
        with self.assertRaises(PipelineError) as ctx:
            read_snapshot(path)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")


class TestEngineSemantics(WorkdirMixin):
    def test_idempotent_reapply(self):
        tables, query_set = load()
        engine = Engine(tables, query_set, 1)
        engine.apply_rows("events", [0, 1, 2])
        digest_once = state_digest(engine.state)
        engine.apply_rows("events", [0, 1, 2])  # duplicate: no-op
        self.assertEqual(digest_once, state_digest(engine.state))

    def test_batch_size_invariance(self):
        tables, query_set = load()
        digests = []
        for batch in (1, 3, 7, 64):
            qs = load_queries_file(QUERIES, tables)
            qs.session["batch_size"] = batch
            engine = Engine(tables, qs, 1)
            engine.run_all()
            digests.append(state_digest(engine.state))
        self.assertEqual(len(set(digests)), 1)

    def test_arrival_order_invariance_with_recompute(self):
        tables, query_set = load()
        outcomes = []
        for seed, window in ((None, 0), (5, 4), (9, 6)):
            qs = load_queries_file(QUERIES, tables)
            qs.session["arrival"] = {"shuffle_seed": seed,
                                     "shuffle_window": window}
            engine = Engine(tables, qs, 1)
            engine.run_all()
            outcomes.append(engine.results())
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], outcomes[2])

    def test_mean_error_bound_documented(self):
        tables, query_set = load()
        engine = Engine(tables, query_set, 1)
        engine.run_all()
        results = engine.results()
        self.assertEqual(results["mean_error_bound"], "1/1000")


class TestRecoveryAndDivergence(WorkdirMixin):
    def test_replay_matches_snapshot(self):
        workdir = os.path.join(self.tmp, "run")
        self.run_ok(workdir)
        code = self.run_cli(
            "recover-check", "--tables", TABLES, "--queries", QUERIES,
            "--workdir", workdir, "--snapshot",
            os.path.join(workdir, "snapshots", "manifest-v2-final.json"))
        self.assertEqual(code, 0)

    def test_resume_bitwise_identical(self):
        full = os.path.join(self.tmp, "full")
        resumed = os.path.join(self.tmp, "resumed")
        self.run_ok(full)
        self.run_ok(resumed, "--resume-from",
                    os.path.join(full, "snapshots", "manifest-v2-mid.json"),
                    "--batch-size", "5")
        code = self.run_cli("diverge-check", "--tables", TABLES,
                            "--queries", QUERIES,
                            "--workdir-a", full, "--workdir-b", resumed)
        self.assertEqual(code, 0)
        self.assertEqual(self.read_results(full)["state_digest"],
                         self.read_results(resumed)["state_digest"])

    def test_batch_and_shuffle_combinations(self):
        tables, query_set = load()
        ref_engine = None
        for batch, seed, window in ((1, None, 0), (4, 3, 5), (11, 8, 6),
                                    (60, None, 0)):
            qs = load_queries_file(QUERIES, tables)
            qs.session["batch_size"] = batch
            qs.session["arrival"] = {"shuffle_seed": seed,
                                     "shuffle_window": window}
            engine = Engine(tables, qs, 1)
            engine.run_all()
            if ref_engine is None:
                ref_engine = engine
            else:
                compare_results(ref_engine, engine)

    def test_e_recover_on_tampered_snapshot(self):
        workdir = os.path.join(self.tmp, "run")
        self.run_ok(workdir)
        tables, query_set = load()
        log_engine = replay_log(os.path.join(workdir, "statelog"), tables,
                                query_set, 1)
        snap_path = os.path.join(workdir, "snapshots",
                                 "manifest-v2-final.json")
        _record, snap_state = rebuild_from_snapshot(snap_path)
        group = next(iter(snap_state.aggs["agg_by_grp"].values()))
        group.count += 1
        with self.assertRaises(PipelineError) as ctx:
            compare_states(log_engine.state, snap_state)
        self.assertEqual(ctx.exception.code, "E_RECOVER")
        self.assertIn("group_key", ctx.exception.details)
        self.assertEqual(ctx.exception.details["field"], "count")

    def test_e_diverge_on_tampered_run(self):
        tables, query_set = load()
        engine_a = Engine(tables, query_set, 1)
        engine_a.run_all()
        engine_b = Engine(tables, query_set, 1)
        engine_b.run_all()
        group = next(iter(engine_b.state.aggs["agg_by_grp"].values()))
        group.sum += 1
        with self.assertRaises(PipelineError) as ctx:
            compare_runs(engine_a, engine_b)
        self.assertEqual(ctx.exception.code, "E_DIVERGE")


class TestVersionEvolution(WorkdirMixin):
    def test_param_version_monotonic_and_stale_snapshot(self):
        tables, query_set = load()
        catalog = Catalog(os.path.join(self.tmp, "catalog.db"))
        fp1 = param_set_fingerprint(query_set)
        v1 = catalog.param_version(fp1)
        path = os.path.join(self.tmp, "snap.json")
        write_snapshot(path, EngineState().to_dict(), v1, "snap-old")
        catalog.register_snapshot("snap-old", path, v1, 2, "digest-x")

        evolved = load_queries_file(QUERIES, tables)
        evolved.session["scale"] = 100  # parameter evolution
        fp2 = param_set_fingerprint(evolved)
        v2 = catalog.param_version(fp2)
        self.assertGreater(v2, v1)
        catalog.mark_stale_before(v2)
        rows = catalog.list_snapshots()
        self.assertEqual(rows[0][5], 1)  # stale flag set

        # Reopening the stale snapshot replaces state wholesale.
        record = read_snapshot(path)
        self.assertNotEqual(record.query_param_version, v2)

    def test_canonical_id_stable(self):
        _tables, query_set = load()
        ids1 = [canonical_query_id(q) for q in query_set.queries]
        _tables2, query_set2 = load()
        ids2 = [canonical_query_id(q) for q in query_set2.queries]
        self.assertEqual(ids1, ids2)
        self.assertTrue(all(i.startswith("q-") for i in ids1))


class TestFaultInjection(WorkdirMixin):
    def _inject(self, name):
        workdir = os.path.join(self.tmp, f"inj-{name}")
        code = self.run_cli("run", "--tables", TABLES, "--queries", QUERIES,
                            "--workdir", workdir, "--inject", name)
        self.assertEqual(code, 2)
        with open(os.path.join(workdir, "logs", "structured.jsonl"),
                  encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    def test_inject_snapshot_digest(self):
        events = self._inject("snapshot-digest")
        self.assertTrue(any(e.get("context", {}).get("code") == "E_SNAPSHOT"
                            for e in events))

    def test_inject_log_truncate(self):
        events = self._inject("log-truncate")
        self.assertTrue(any(e.get("context", {}).get("code") == "E_LOG"
                            for e in events))

    def test_inject_batch_mid_window(self):
        events = self._inject("batch-mid-window")
        self.assertTrue(any(e.get("context", {}).get("code") == "E_BATCH"
                            for e in events))

    def test_inject_out_of_order(self):
        events = self._inject("out-of-order")
        self.assertTrue(any(e.get("context", {}).get("code") == "E_ORDER"
                            for e in events))


class TestCrossVersionInterop(WorkdirMixin):
    def test_v1_log_and_v1_snapshot(self):
        workdir = os.path.join(self.tmp, "run-v1")
        self.run_ok(workdir, "--log-format", "1", "--snapshot-format", "1")
        code = self.run_cli(
            "recover-check", "--tables", TABLES, "--queries", QUERIES,
            "--workdir", workdir, "--snapshot",
            os.path.join(workdir, "snapshots", "manifest-v1-final.json"))
        self.assertEqual(code, 0)

    def test_cross_version_same_output(self):
        workdir_v1 = os.path.join(self.tmp, "run-v1x")
        workdir_v2 = os.path.join(self.tmp, "run-v2x")
        self.run_ok(workdir_v1, "--log-format", "1", "--snapshot-format", "1")
        self.run_ok(workdir_v2, "--log-format", "2", "--snapshot-format", "2")
        self.assertEqual(self.read_results(workdir_v1)["state_digest"],
                         self.read_results(workdir_v2)["state_digest"])


class TestGuiModel(WorkdirMixin):
    def test_build_model(self):
        sys.path.insert(0, ROOT)
        import gui
        workdir = os.path.join(self.tmp, "run")
        self.run_ok(workdir)
        model = gui.build_model(workdir)
        self.assertEqual(len(model["queries"]), 2)
        self.assertTrue(model["aggs"])
        self.assertTrue(model["windows"])
        self.assertTrue(model["cursors"])
        self.assertEqual(len(model["snapshots"]), 2)
        self.assertTrue(all(q["canonical_id"].startswith("q-")
                            for q in model["queries"]))


class TestVerifyTool(WorkdirMixin):
    def test_verify_tool_ok(self):
        import verify_tool
        workdir = os.path.join(self.tmp, "run")
        self.run_ok(workdir)
        code = verify_tool.main([
            "--tables", TABLES, "--queries", QUERIES,
            "--snapshot", os.path.join(workdir, "snapshots",
                                       "manifest-v2-final.json"),
            "--statelog", os.path.join(workdir, "statelog"),
            "--catalog", os.path.join(workdir, "catalog.db"),
            "--results", os.path.join(workdir, "results.json")])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
