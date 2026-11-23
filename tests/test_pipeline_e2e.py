import json
import shutil
import tempfile
import unittest
from pathlib import Path

from alert_pipeline import samplegen
from alert_pipeline.errors import (FieldError, ReplayError, SnapshotError,
                                   SpecError)
from alert_pipeline.ledger import Ledger
from alert_pipeline.pipeline import Pipeline, PipelineConfig
from alert_pipeline.util import write_json


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="alertpipe-"))
        self.paths = samplegen.generate_all(self.tmp / "samples", seed=77)
        samplegen.generate_injected(self.tmp / "samples")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, **over):
        kwargs = dict(
            graph_path=self.paths["graph"], rules_path=self.paths["rules"],
            stream_path=self.paths["stream"],
            ledger_path=str(self.tmp / "ledger.sqlite"),
            log_dir=str(self.tmp / "logs"), batch_size=10, global_limit=3,
            per_service_limit=2, task_timeout_ms=1000)
        kwargs.update(over)
        return PipelineConfig(**kwargs)

    def test_full_run_statistics_and_repeatability(self):
        pipe = Pipeline(self.cfg())
        result = pipe.run()
        pipe.close()
        self.assertEqual(result["status"], "OK")
        self.assertGreater(result["total_aggregates"], 0)
        stats = result["statistics"]
        self.assertEqual(stats["scale_factor"], 10 ** 4)
        self.assertEqual(stats["truncation"], "floor")
        # Re-running into a fresh ledger reproduces identical aggregates.
        second_ledger = self.tmp / "ledger2.sqlite"
        pipe2 = Pipeline(self.cfg(ledger_path=str(second_ledger)))
        again = pipe2.run()
        pipe2.close()
        self.assertEqual(again["total_aggregates"],
                         result["total_aggregates"])

    def test_crash_unknown_batch_recovery_and_replay(self):
        cfg = self.cfg()
        pipe = Pipeline(cfg)
        crashed = pipe.run(crash_at_batch=2)
        pipe.close()
        self.assertEqual(crashed["status"], "CRASHED")
        ledger = Ledger(cfg.ledger_path)
        states = dict((r[0], r[1]) for r in ledger.conn.execute(
            "SELECT batch_id, state FROM batches"))
        self.assertEqual(states[2], "UNKNOWN")
        ledger.close()

        pipe = Pipeline(cfg)  # restart
        recovered = pipe.resume()
        pipe.close()
        self.assertEqual(recovered["status"], "OK")
        self.assertEqual(recovered["reran_unknown_batches"], [2])
        self.assertEqual(recovered["replayed_batches"], 1)
        ledger = Ledger(cfg.ledger_path)
        self.assertFalse(any(r[0] == "UNKNOWN" for r in ledger.conn.execute(
            "SELECT DISTINCT state FROM batches")))
        ledger.close()

    def test_replay_divergence_detected_as_e_replay(self):
        cfg = self.cfg()
        pipe = Pipeline(cfg)
        pipe.run()
        pipe.close()
        # Tamper with a committed aggregate payload -> replay digest mismatch.
        ledger = Ledger(cfg.ledger_path)
        bid = ledger.conn.execute(
            "SELECT MIN(batch_id) FROM aggregates").fetchone()[0]
        aid = ledger.conn.execute(
            "SELECT aggregate_id FROM aggregates WHERE batch_id=? LIMIT 1",
            (bid,)).fetchone()[0]
        record = json.loads(ledger.conn.execute(
            "SELECT record_json FROM aggregates WHERE batch_id=? AND "
            "aggregate_id=?", (bid, aid)).fetchone()[0])
        record["suppressed_count"] += 1
        # Tamper content but keep the old digest: snapshot verification still
        # passes (it recomputes content->digest only in verify_snapshots), while
        # deterministic replay comparison detects the first field divergence.
        ledger.conn.execute(
            "UPDATE aggregates SET record_json=? WHERE batch_id=? AND "
            "aggregate_id=?",
            (json.dumps(record, sort_keys=True), bid, aid))
        ledger.conn.commit()
        ledger.close()
        # Re-open and force replay against the altered committed conclusion.
        pipe = Pipeline(cfg)
        trusted = pipe.ledger.last_committed_cursor(pipe.version)
        with self.assertRaises(ReplayError) as ctx:
            aggs, corrs, _ = pipe._reprocess_through(trusted)
            from alert_pipeline.replay import verify_replay
            _, ledger_aggs, ledger_corrs = pipe.ledger.load_committed(
                pipe.version)
            verify_replay(
                sorted((a.to_record() for a in aggs),
                       key=lambda r: r["aggregate_id"]),
                sorted(ledger_aggs, key=lambda r: r["aggregate_id"]),
                sorted((c.to_record() for c in corrs),
                       key=lambda r: r["aggregate_id"]),
                sorted(ledger_corrs, key=lambda r: r["aggregate_id"]))
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertTrue(ctx.exception.detail.startswith("agg_"))
        self.assertIn(ctx.exception.field,
                      ("aggregate_id", "suppressed_count"))
        pipe.close()

    def test_evolution_marks_stale_and_replaces_view(self):
        cfg = self.cfg()
        pipe = Pipeline(cfg)
        pipe.run()
        rules = json.loads(Path(self.paths["rules"]).read_text(encoding="utf-8"))
        rules["declared_version"] = 2
        rules["window_version"] = 2
        rules["default_silence_window_ms"] = 4000
        new_rules = self.tmp / "rules_v2.json"
        write_json(new_rules, rules)
        result = pipe.evolve(self.paths["graph"], str(new_rules),
                             self.paths["stream"])
        pipe.close()
        self.assertEqual(result["view_diff"]["new_version"], 2)
        self.assertTrue(result["stale_batches_marked"] >= 1)
        ledger = Ledger(cfg.ledger_path)
        stale_aggs = ledger.conn.execute(
            "SELECT COUNT(*) FROM aggregates WHERE stale=1").fetchone()[0]
        live_aggs, _ = ledger.current_view(2)
        self.assertGreater(stale_aggs, 0)
        self.assertTrue(all(
            r[0] == 2 for r in ledger.conn.execute(
                "SELECT DISTINCT declared_version FROM aggregates "
                "WHERE stale=0")))
        self.assertGreater(len(live_aggs), 0)
        ledger.close()

    def test_injected_spec_field_snapshot_errors(self):
        with self.assertRaises(SpecError) as ctx:
            Pipeline(self.cfg(
                graph_path=str(self.tmp / "samples" / "injected" /
                               "graph_cycle.json"),
                ledger_path=str(self.tmp / "c.sqlite")))
        self.assertEqual(ctx.exception.code, "E_SPEC")

        with self.assertRaises(FieldError) as ctx:
            Pipeline(self.cfg(
                stream_path=str(self.tmp / "samples" / "injected" /
                                "alerts_negative_window.ndjson"),
                ledger_path=str(self.tmp / "f.sqlite")))
        self.assertEqual(ctx.exception.code, "E_FIELD")

        cfg = self.cfg()
        pipe = Pipeline(cfg)
        pipe.run()
        pipe.close()
        ledger = Ledger(cfg.ledger_path)
        bid = ledger.conn.execute(
            "SELECT MIN(batch_id) FROM aggregates").fetchone()[0]
        ledger.corrupt_for_injection("aggregates", bid)
        ledger.close()
        with self.assertRaises(SnapshotError) as ctx:
            pipe = Pipeline(cfg)
            pipe.run()
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()
