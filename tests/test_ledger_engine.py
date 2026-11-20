import tempfile
import unittest
from pathlib import Path

from elastic_scaler.common import canonical_dumps
from elastic_scaler.engine import ForcedKill, ScalingEngine
from elastic_scaler.errors import (
    DivergenceError,
    ReplayError,
    VersionConflictError,
)
from elastic_scaler.gui import ScalerWorkbench, build_view_model
from elastic_scaler.spec import load_spec

V1 = "samples/cluster_v1.json"
V2 = "samples/cluster_v2.json"


class EngineHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.db = str(root / "ledger.sqlite3")
        self.log = str(root / "events.jsonl")

    def engine(self, run_id: str = "run-current") -> ScalingEngine:
        return ScalingEngine(self.db, self.log, run_id=run_id)


class LedgerAndReplayTests(EngineHarness):
    def test_minute_commit_is_idempotent_and_byte_identical(self):
        with self.engine() as engine:
            spec = load_spec(V1)
            engine.initialize(spec)
            first = engine.advance_minute(0)
            first_again = engine.advance_minute(0)
        with self.engine() as reopened:
            reopened.recover()
            self.assertEqual(
                canonical_dumps(first.to_dict()),
                canonical_dumps(first_again.to_dict()),
            )
            self.assertEqual(
                canonical_dumps(first.to_dict()),
                canonical_dumps(reopened.authoritative_query(0).to_dict()),
            )
            # No duplicate service rows after the idempotent rewrite.
            count = reopened.ledger.conn.execute(
                "SELECT COUNT(*) AS c FROM minute_services"
            ).fetchone()["c"]
            self.assertEqual(count, len(spec.services))

    def test_replay_report_covers_every_committed_row(self):
        with self.engine() as engine:
            engine.initialize(load_spec(V1))
            engine.advance_all()
            report = engine.verify_replay()
            self.assertTrue(report.ok)
            self.assertEqual(report.checked_minutes, 8)
            self.assertEqual(report.checked_service_rows, 24)

    def test_forced_kill_leaves_unknown_minute_and_resumes(self):
        with self.engine("run-crash") as engine:
            engine.initialize(load_spec(V1))
            with self.assertRaises(ForcedKill):
                engine.advance_all(simulate_kill_at=3)
        with self.engine("run-crash") as engine:
            report = engine.recover()
            self.assertEqual(list(report.committed_minutes), [0, 1, 2])
            self.assertEqual(list(report.unknown_minutes), [3])
            self.assertIn("re-run", report.note)
            resumed = engine.resume_after_recovery()
            self.assertEqual([d.minute for d in resumed], [3, 4, 5, 6, 7])
            self.assertTrue(engine.verify_replay().ok)


class FastPathTests(EngineHarness):
    def test_fast_query_matches_authoritative_replay(self):
        with self.engine() as engine:
            engine.initialize(load_spec(V1))
            engine.advance_all()
            engine.verify_fast_vs_authoritative()
            fast = engine.fast_query(4)
            authoritative = engine.authoritative_query(4)
            self.assertEqual(fast.to_dict(), authoritative.to_dict())

    def test_cache_corruption_is_reported_as_e_diverge(self):
        with self.engine() as engine:
            engine.initialize(load_spec(V1))
            engine.advance_all()
            engine.debug_corrupt_cache(4, "llm")
            with self.assertRaises(DivergenceError) as caught:
                engine.verify_fast_vs_authoritative()
            context = caught.exception.context
            self.assertEqual(context["minute"], 4)
            self.assertEqual(context["service"], "llm")
            self.assertEqual(context["field"], "allocated_replicas")
            self.assertEqual(context["cache_basis_spec_version"], 1)


class VersionTests(EngineHarness):
    def test_spec_replacement_replaces_whole_view_and_keeps_old_readonly(self):
        with self.engine("run-v1") as engine:
            engine.initialize(load_spec(V1))
            engine.advance_all()
            replacement = engine.replace_spec(load_spec(V2))
            self.assertEqual((replacement.old_version, replacement.new_version), (1, 2))
            self.assertEqual(replacement.affected_services, ("llm",))
            self.assertTrue(any(
                diff["path"] == "/global_replica_budget"
                for diff in replacement.parameter_diffs
            ))
            engine.advance_all()
            self.assertEqual(engine.cache_basis_version, 2)
            old_view = engine.get_version_view(1)
            self.assertTrue(old_view["read_only"])
            self.assertEqual(old_view["status"], "READ_ONLY")

    def test_snapshot_version_conflict_raises_e_version(self):
        with self.engine() as engine:
            engine.initialize(load_spec(V1))
            engine.advance_minute(0)
            engine.save_snapshot("good", 0)
            engine.replace_spec(load_spec(V2))
            with self.assertRaises(VersionConflictError) as caught:
                engine.restore_snapshot("good")
            self.assertEqual(caught.exception.code, "E_VERSION")


class GuiAndStatsTests(EngineHarness):
    def test_headless_view_is_read_only_and_latency_is_integer_fixed_point(self):
        with self.engine() as engine:
            engine.initialize(load_spec(V1))
            engine.advance_all()
            workbench = ScalerWorkbench(engine)
            self.assertTrue(workbench.headless)
            view = workbench.refresh()
            self.assertEqual(view["service_order"], ["asr", "embedding", "llm"])
            self.assertEqual(len(view["budget_curve"]), 8)
            report = workbench.latency_report()
            overall = report["overall"]
            self.assertGreater(overall["samples"], 0)
            self.assertEqual(overall["quantization_error_bound_us"], 1)
            for value in (overall["mean_ms_fixed3"], overall["p95_ms_fixed3"],
                          overall["max_ms_fixed3"]):
                self.assertRegex(value, r"^\d+\.\d{3}$")
            # Refresh must not mutate the ledger.
            before = engine.ledger.committed_minutes(engine.run_id)
            build_view_model(engine)
            self.assertEqual(before, engine.ledger.committed_minutes(engine.run_id))


if __name__ == "__main__":
    unittest.main()


class LedgerTamperTests(EngineHarness):
    def test_tampered_ledger_row_is_reported_as_e_replay(self):
        with self.engine() as engine:
            engine.initialize(load_spec(V1))
            engine.advance_all()
            import json
            row = engine.ledger.get_minute(engine.run_id, 2)
            decision = row.decision
            for allocation in decision["allocations"]:
                if allocation["service"] == "embedding":
                    allocation["allocated_replicas"] += 1
            engine.ledger.conn.execute(
                "UPDATE minute_progress SET decision_json = ? WHERE run_id = ? AND minute = ?",
                (json.dumps(decision, sort_keys=True, separators=(",", ":")),
                 engine.run_id, 2),
            )
            engine.ledger.conn.commit()
        with self.engine() as engine:
            engine.recover()
            with self.assertRaises(ReplayError) as caught:
                engine.verify_replay()
            context = caught.exception.context
            self.assertEqual(context["minute"], 2)
            self.assertEqual(context["service"], "embedding")
            self.assertIn("allocated_replicas", context["field"])
            self.assertIn("spec_version=", context["basis"])
