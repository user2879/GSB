import os
import tempfile
import unittest

from bench.mw.config import load_events, load_spec
from bench.mw.errors import (OrderError, ReplayError, SnapshotError,
                             StructuredLogger, ViewMismatchError,
                             WatermarkError)
from bench.mw.fault_inject import (inject_corrupt_event_file,
                                   inject_snapshot_digest,
                                   inject_timestamp_regression)
from bench.mw.latency import compute_stats
from bench.mw.recovery import (continue_from_snapshot, unknown_after_crash,
                               verify_against_snapshot)
from bench.mw.session import Session, restore_session

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASIC_SPEC = os.path.join(ROOT, "samples", "basic", "spec.json")
BASIC_EVENTS = os.path.join(ROOT, "samples", "basic", "events.ndjson")
RESUME_SPEC = os.path.join(ROOT, "samples", "resume", "spec.json")
RESUME_P1 = os.path.join(ROOT, "samples", "resume", "events_part1.ndjson")
RESUME_P2 = os.path.join(ROOT, "samples", "resume", "events_part2.ndjson")


class SessionHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger_path = os.path.join(self.tmp.name, "l.db")
        self.snap_path = os.path.join(self.tmp.name, "s.json")
        self.log_path = os.path.join(self.tmp.name, "e.ndjson")
        self.logger = StructuredLogger(self.log_path)

    def tearDown(self):
        self.logger.close()
        self.tmp.cleanup()

    def session(self, spec_path=BASIC_SPEC, events=None):
        spec = load_spec(spec_path)
        session = Session(spec, self.ledger_path, self.logger,
                          event_file=events)
        return session


class CacheVsReplayTests(SessionHarness):
    def test_views_byte_identical(self):
        session = self.session(events=BASIC_EVENTS)
        session.ingest_file(BASIC_EVENTS)
        session.drain()
        result = session.verify_views()
        self.assertEqual(result["status"], "identical")
        self.assertEqual(session.cache_view_bytes(),
                         session.authoritative_view_bytes())
        session.close()

    def test_e_view_on_tampered_cache(self):
        session = self.session(events=BASIC_EVENTS)
        session.ingest_file(BASIC_EVENTS)
        session.drain()
        ledger = session.ledger
        ledger.conn.execute(
            "UPDATE partition_state SET finalized_count=finalized_count+1 "
            "WHERE partition='p0'")
        ledger.conn.commit()
        with self.assertRaises(ViewMismatchError) as ctx:
            session.verify_views()
        self.assertEqual(ctx.exception.code, "E_VIEW")
        self.assertEqual(ctx.exception.details["first_partition"], "p0")
        self.assertEqual(ctx.exception.details["field"], "finalized_count")
        session.close()

    def test_late_records_in_log(self):
        session = self.session(events=BASIC_EVENTS)
        session.ingest_file(BASIC_EVENTS)
        session.drain()
        late_rows = session.ledger.late_log_rows(session.param_version)
        self.assertEqual({r["partition"] for r in late_rows}, {"p0", "p1"})
        with open(self.log_path) as handle:
            codes = [line for line in handle if '"code":"E_LATE"' in line]
        self.assertTrue(codes)
        session.close()


class LatencyTests(SessionHarness):
    def test_query_latency_budget_and_stats(self):
        session = self.session()
        session.ingest_file(BASIC_EVENTS)
        session.drain()
        normal = session.query()
        self.assertFalse(normal.over_budget)
        over = session.query(inject_delay_ms=80)
        self.assertTrue(over.over_budget)
        self.assertEqual(over.code, "E_LAT")
        report = session.latency_report()
        self.assertEqual(report["truth_source"], "ledger.query_log")
        self.assertEqual(report["overall"]["samples"], 2)
        self.assertGreaterEqual(report["overall"]["max_fixed"], 8000)
        self.assertEqual(report["over_budget_count"], 1)
        # latency must not change aggregates
        verify = session.verify_views()
        self.assertEqual(verify["status"], "identical")
        session.close()

    def test_stats_deterministic(self):
        samples = [5, 8, 3, 12, 7]
        first = compute_stats(samples, 2).as_dict()
        second = compute_stats(list(reversed(samples)), 2).as_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["p95_fixed"], 1200)
        self.assertEqual(first["max_fixed"], 1200)


class ParameterVersionTests(SessionHarness):
    def test_replacement_hides_stale_versions(self):
        session = self.session()
        session.ingest_file(BASIC_EVENTS)
        session.drain()
        session.query()
        new_spec_path = os.path.join(self.tmp.name, "new_spec.json")
        import json
        new_spec = {
            "out_of_order_ms": 250, "latency_budget_ms": 30, "scale": 4,
            "partitions": [
                {"key": "p0", "aggregations": ["sum", "count"]},
                {"key": "p1", "aggregations": ["sum", "count"]},
                {"key": "p2", "aggregations": ["sum", "count"]}]}
        with open(new_spec_path, "w") as fh:
            json.dump(new_spec, fh)
        replacement = session.replace_parameters(load_spec(new_spec_path))
        self.assertEqual(replacement["new_param_version"], 2)
        self.assertEqual(replacement["stale_versions"], [1])
        # current views are empty brand-new state, not merged
        views = session.ledger.cache_views(2, session.spec)
        self.assertTrue(all(v["finalized_count"] == 0 for v in views))
        # stale version remains in ledger but excluded from current report
        report = session.latency_report()
        self.assertEqual(report["param_version"], 2)
        self.assertEqual(report["overall"]["samples"], 0)
        affected = replacement["affected_partitions"]
        self.assertEqual({a["partition"] for a in affected},
                         {"p0", "p1", "p2"})
        self.assertTrue(all("old_watermark_ms" in a and
                            "new_watermark_ms" in a for a in affected))
        session.close()


class SnapshotRecoveryTests(SessionHarness):
    def test_snapshot_restore_and_resume_replay(self):
        session = self.session(RESUME_SPEC, events=RESUME_P1)
        session.ingest_file(RESUME_P1)
        session.save_snapshot(self.snap_path)
        session.query()
        session.close()

        restored = restore_session(self.snap_path, self.ledger_path,
                                   self.logger)
        self.assertFalse(restored.replacement_required)
        verify = verify_against_snapshot(restored, restored.snapshot_info)
        self.assertEqual(verify["status"], "identical")
        continue_from_snapshot(restored, RESUME_P2, self.logger)
        restored.drain()
        views = {v["partition"]: v for v in
                 restored.ledger.cache_views(restored.param_version,
                                             restored.spec)}
        self.assertEqual(views["p0"]["finalized_count"], 4)
        self.assertEqual(views["p0"]["sum_fixed"], 1000000)
        self.assertEqual(views["p1"]["finalized_count"], 4)
        self.assertEqual(views["p1"]["sum_fixed"], 34000)
        restored.close()

    def test_replay_divergence_detected(self):
        session = self.session(RESUME_SPEC, events=RESUME_P1)
        session.ingest_file(RESUME_P1)
        session.save_snapshot(self.snap_path)
        session.close()
        restored = restore_session(self.snap_path, self.ledger_path,
                                   self.logger)
        # corrupt committed state then replay must flag E_REPLAY
        restored.ledger.conn.execute(
            "UPDATE partition_state SET raw_sum=raw_sum+999 WHERE partition='p0'")
        restored.ledger.conn.commit()
        with self.assertRaises(ReplayError) as ctx:
            verify_against_snapshot(restored, restored.snapshot_info)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.details["partition"], "p0")
        self.assertEqual(ctx.exception.details["field"], "sum_raw")
        restored.close()

    def test_snapshot_digest_tampering_rejected(self):
        session = self.session(RESUME_SPEC, events=RESUME_P1)
        session.ingest_file(RESUME_P1)
        session.save_snapshot(self.snap_path)
        session.close()
        bad = os.path.join(self.tmp.name, "bad_snap.json")
        inject_snapshot_digest(self.snap_path, bad)
        with self.assertRaises(SnapshotError) as ctx:
            restore_session(bad, self.ledger_path, self.logger, tampered=True)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")

    def test_crash_unknown_reload_no_exit_report(self):
        session = self.session(RESUME_SPEC, events=RESUME_P1)
        session.ingest_file(RESUME_P1)
        unknown = unknown_after_crash(session)
        self.assertTrue(all(u["status"] == "unknown_reload_required"
                            for u in unknown))
        cursors = {u["partition"]: u["committed_cursor_seq"]
                   for u in unknown}
        self.assertEqual(cursors, {"p0": 1, "p1": 1})
        session.close()


class FaultInjectionTests(SessionHarness):
    def test_corrupt_event_file_is_e_spec(self):
        target = os.path.join(self.tmp.name, "corrupt.ndjson")
        inject_corrupt_event_file(BASIC_EVENTS, target)
        spec = load_spec(BASIC_SPEC)
        from bench.mw.errors import SpecError
        with self.assertRaises(SpecError) as ctx:
            load_events(target, spec)
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_timestamp_regression_is_e_order(self):
        target = os.path.join(self.tmp.name, "reg.ndjson")
        # part2 contains seq beyond snapshot cursor; push seq2 far backwards.
        inject_timestamp_regression(
            RESUME_P2, target, partition="p0", seq=2, new_ts_ms=10)
        session = self.session(RESUME_SPEC, events=RESUME_P1)
        session.ingest_file(RESUME_P1)
        session.save_snapshot(self.snap_path)
        session.close()
        restored = restore_session(self.snap_path, self.ledger_path,
                                   self.logger)
        with self.assertRaises(OrderError) as ctx:
            continue_from_snapshot(restored, target, self.logger)
        self.assertEqual(ctx.exception.code, "E_ORDER")
        self.assertEqual(ctx.exception.details["partition"], "p0")
        self.assertEqual(ctx.exception.details["seq"], 2)
        restored.close()

    def test_watermark_rollback_code(self):
        session = self.session()
        session.ingest_file(BASIC_EVENTS)
        with self.assertRaises(WatermarkError) as ctx:
            session.advance_watermark(-1)
        self.assertEqual(ctx.exception.code, "E_WATER")
        session.close()


class NumpyCrossCheckTests(SessionHarness):
    def test_numpy_agrees_with_ledger(self):
        from bench.mw.numpy_check import cross_check_aggregates, cross_check_percentile
        session = self.session()
        session.ingest_file(BASIC_EVENTS)
        session.drain()
        replay = session.ledger.authoritative_replay(
            session.param_version, session.spec)
        values: dict[str, list[int]] = {}
        for event in session.ledger.raw_events_for_version(session.param_version):
            part = session.engine.parts[event.partition]
            if event.ts_ms >= part.watermark_ms or True:
                pass
        # derive finalized values directly from engine replay
        from bench.mw.engine import SessionEngine
        engine = SessionEngine(session.spec)
        engine.apply_events(
            session.ledger.raw_events_for_version(session.param_version))
        engine.drain()
        for key, part in engine.parts.items():
            values[key] = [f.value for f in part.finalized]
        cross = cross_check_aggregates(replay, values, session.spec.scale)
        self.assertTrue(all(all(match.values()) for match in cross.values()))
        self.assertEqual(cross_check_percentile([1, 2, 3, 4, 5]), 5)
        session.close()


if __name__ == "__main__":
    unittest.main()
