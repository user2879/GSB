"""End-to-end and unit tests for the slow-request profiler.

Run with: python -m unittest discover -s tests -v
NumPy is used only for independent numeric cross-validation and is optional.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from slowprof.errors import (
    RangeError,
    ReplayError,
    SampleError,
)
from slowprof.fixedpoint import QuantParams, build_share_report, quantize
from slowprof.folding import (
    cross_check,
    fold_adaptive,
    fold_full,
    fold_streaming,
    select_path,
)
from slowprof.generator import generate
from slowprof.ranking import build_contexts, rank_frames
from slowprof.samples import (
    iter_samples,
    load_samples,
    make_interval,
    sample_stream_digest,
)
from slowprof.session import ForcedKill, ProfileSession, SessionConfig

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:  # pragma: no cover
    HAS_NUMPY = False


def write_ndjson(path: Path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def base_row(index, service="billing", ts=None, weight=1,
             frames=("billing.entry", "billing.charge", "db.query")):
    if ts is None:
        ts = 1000 + index
    return {
        "request_id": f"req-{index}",
        "timestamp_ms": ts,
        "duration_ms": 10 + index,
        "service_id": service,
        "weight": weight,
        "frames": list(frames),
    }


class SampleValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "s.ndjson"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_load_and_interval(self):
        write_ndjson(self.path, [
            base_row(0, ts=100),
            base_row(1, ts=200),
            base_row(2, service="billing", ts=300),
        ])
        samples = load_samples(self.path)
        self.assertEqual(len(samples), 3)
        window = load_samples(self.path, make_interval(150, 250))
        self.assertEqual([s.timestamp_ms for s in window], [200])

    def test_bad_weight_reports_line_and_field(self):
        rows = [base_row(0), base_row(1)]
        rows[1]["weight"] = 0
        write_ndjson(self.path, rows)
        with self.assertRaises(SampleError) as ctx:
            list(iter_samples(self.path))
        self.assertEqual(ctx.exception.code, "E_SAMPLE")
        self.assertEqual(ctx.exception.line_number, 2)
        self.assertEqual(ctx.exception.field, "weight")

    def test_bad_duration(self):
        rows = [base_row(0)]
        rows[0]["duration_ms"] = -5
        write_ndjson(self.path, rows)
        with self.assertRaises(SampleError) as ctx:
            list(iter_samples(self.path))
        self.assertEqual(ctx.exception.field, "duration_ms")

    def test_bad_frame_shape(self):
        rows = [base_row(0, frames=("billing.entry", "no_dot_segment"))]
        write_ndjson(self.path, rows)
        with self.assertRaises(SampleError) as ctx:
            list(iter_samples(self.path))
        self.assertEqual(ctx.exception.code, "E_SAMPLE")
        self.assertTrue(ctx.exception.field.startswith("frames[1]"))

    def test_empty_frame_segment(self):
        rows = [base_row(0, frames=("billing.", "db.query"))]
        write_ndjson(self.path, rows)
        with self.assertRaises(SampleError):
            list(iter_samples(self.path))

    def test_missing_field(self):
        row = base_row(0)
        del row["service_id"]
        write_ndjson(self.path, [row])
        with self.assertRaises(SampleError) as ctx:
            list(iter_samples(self.path))
        self.assertEqual(ctx.exception.field, "service_id")

    def test_invalid_json_line_number(self):
        self.path.write_text(
            json.dumps(base_row(0)) + "\n{not valid json}\n",
            encoding="utf-8")
        with self.assertRaises(SampleError) as ctx:
            list(iter_samples(self.path))
        self.assertEqual(ctx.exception.line_number, 2)
        self.assertEqual(ctx.exception.field, "<json>")

    def test_monotone_timestamp_per_service(self):
        write_ndjson(self.path, [
            base_row(0, service="a", ts=100),
            base_row(1, service="b", ts=50),  # different service is fine
            base_row(2, service="a", ts=99),  # regress inside service a
        ])
        with self.assertRaises(SampleError) as ctx:
            list(iter_samples(self.path))
        self.assertEqual(ctx.exception.field, "timestamp_ms")
        self.assertEqual(ctx.exception.line_number, 3)

    def test_boolean_rejected_as_integer(self):
        row = base_row(0)
        row["weight"] = True
        write_ndjson(self.path, [row])
        with self.assertRaises(SampleError):
            list(iter_samples(self.path))

    def test_invalid_interval_is_e_range(self):
        with self.assertRaises(RangeError) as ctx:
            make_interval(500, 100)
        self.assertEqual(ctx.exception.code, "E_RANGE")


class FoldingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "s.ndjson"
        self.stack_a = ("svc.entry", "svc.work", "db.query")
        self.stack_b = ("svc.entry", "svc.work", "cache.get")
        write_ndjson(self.path, [
            base_row(0, weight=2, frames=self.stack_a),
            base_row(1, weight=3, frames=self.stack_b),
            base_row(2, weight=4, frames=self.stack_a),
        ])
        self.samples = load_samples(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_self_and_cumulative_semantics(self):
        result = fold_full(self.samples)
        metrics = result.metric_map()
        # stack_a total weight 6, stack_b 3
        self.assertEqual(metrics["db.query"].self_weight, 6)
        self.assertEqual(metrics["cache.get"].self_weight, 3)
        # svc.entry and svc.work appear in every sample: cumulative 9
        self.assertEqual(metrics["svc.entry"].cumulative_weight, 9)
        self.assertEqual(metrics["svc.work"].cumulative_weight, 9)
        # non-leaf frames have zero self weight
        self.assertEqual(metrics["svc.entry"].self_weight, 0)
        self.assertEqual(result.total_weight, 9)

    def test_order_independence(self):
        reordered = [self.samples[2], self.samples[0], self.samples[1]]
        self.assertEqual(
            fold_full(self.samples).canonical_bytes(),
            fold_full(reordered).canonical_bytes())

    def test_dual_path_byte_identical(self):
        same, message = cross_check(self.samples)
        self.assertTrue(same, message)
        self.assertEqual(
            fold_full(self.samples).canonical_bytes(),
            fold_streaming(self.samples).canonical_bytes())

    def test_threshold_boundary_selects_streaming(self):
        self.assertEqual(select_path(4, 10, max_unique_stacks=5,
                                     max_total_weight=20), "full")
        self.assertEqual(select_path(5, 10, max_unique_stacks=5,
                                     max_total_weight=20), "streaming")
        self.assertEqual(select_path(4, 20, max_unique_stacks=5,
                                     max_total_weight=20), "streaming")

    def test_adaptive_force_paths_agree(self):
        full = fold_adaptive(self.samples, force_path="full")
        streaming = fold_adaptive(self.samples, force_path="streaming")
        self.assertEqual(full.canonical_bytes(),
                         streaming.canonical_bytes())

    def test_repeated_frame_counted_once_cumulative(self):
        rows = [base_row(0, weight=2,
                         frames=("a.m", "b.m", "a.m"))]
        write_ndjson(self.path, rows)
        result = fold_full(load_samples(self.path))
        self.assertEqual(result.metric_map()["a.m"].cumulative_weight, 2)


class RankingContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "s.ndjson"
        write_ndjson(self.path, [
            base_row(0, weight=5,
                     frames=("root.a", "mid.b", "leaf.c")),
            base_row(1, weight=7,
                     frames=("root.a", "mid.b", "leaf.d")),
            base_row(2, weight=3,
                     frames=("root.a", "mid.e", "leaf.c")),
        ])
        self.samples = load_samples(self.path)
        self.result = fold_full(self.samples)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stable_ranking_rule(self):
        ranks = rank_frames(self.result, top_k=3)
        # leaf.d self=7 first, leaf.c self=5+3=8 actually -> leaf.c first
        names = [r.frame for r in ranks]
        self.assertEqual(names[0], "leaf.c")
        self.assertEqual(names[1], "leaf.d")
        # ranks are contiguous and 1-based
        self.assertEqual([r.rank for r in ranks], [1, 2, 3])

    def test_ranking_order_independent(self):
        reordered = list(reversed(self.samples))
        ranks_a = [r.frame for r in rank_frames(self.result, 4)]
        ranks_b = [
            r.frame for r in rank_frames(fold_full(reordered), 4)]
        self.assertEqual(ranks_a, ranks_b)

    def test_caller_callee_context(self):
        ranks = rank_frames(self.result, 2)
        contexts = build_contexts(self.result, ranks)
        leaf_c = contexts["leaf.c"]
        callers = {e.counterpart: e.edge_weight for e in leaf_c.callers}
        # leaf.c called by mid.b (w5) and mid.e (w3)
        self.assertEqual(callers, {"mid.b": 5, "mid.e": 3})
        self.assertEqual(leaf_c.callees, ())
        mid_b = contexts.get("mid.b")
        # mid.b not in top-2 leaves, so absent by design
        self.assertIsNone(mid_b)

    def test_context_deterministic_sorting(self):
        ranks = rank_frames(self.result, 4)
        contexts = build_contexts(self.result, ranks)
        for ctx in contexts.values():
            weights = [e.edge_weight for e in ctx.callers]
            self.assertEqual(weights, sorted(weights, reverse=True))


class FixedPointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "s.ndjson"
        write_ndjson(self.path, [
            base_row(0, weight=1, frames=("a.x", "leaf.p")),
            base_row(1, weight=1, frames=("a.x", "leaf.q")),
            base_row(2, weight=1, frames=("a.x", "leaf.r")),
        ])
        self.result = fold_full(load_samples(self.path))
        self.ranks = rank_frames(self.result, 3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_quantize_bounds(self):
        self.assertEqual(quantize(1, 3, 100), 33)
        self.assertEqual(quantize(3, 3, 100), 100)
        self.assertEqual(quantize(0, 3, 100), 0)

    def test_scale_must_be_positive_integer(self):
        with self.assertRaises(RangeError):
            QuantParams(0)
        with self.assertRaises(RangeError):
            QuantParams(1.5)  # type: ignore[arg-type]

    def test_report_contains_true_top_k_and_bounds(self):
        report = build_share_report(
            self.result, self.ranks, QuantParams(100))
        self.assertTrue(report.top_k_contains_true_top_k)
        self.assertEqual(report.aggregate_error_bound_num,
                         len(self.ranks))
        self.assertEqual(report.aggregate_error_bound_den, 100)
        for share in report.self_shares:
            self.assertEqual(share.error_bound_num, 1)
            self.assertEqual(share.error_bound_den, 100)
            self.assertLess(
                abs(share.real_numerator / share.real_denominator
                    - share.quantized / 100),
                1 / 100,
            )

    def test_error_affected_tie_annotation(self):
        # Three equal leaves quantize to the same value -> affected flag.
        report = build_share_report(
            self.result, self.ranks, QuantParams(10))
        self.assertTrue(
            all(s.error_affected for s in report.self_shares))
        self.assertTrue(len(report.error_affected_pairs) >= 1)

    @unittest.skipUnless(HAS_NUMPY, "NumPy not available")
    def test_numpy_independent_numeric_crosscheck(self):
        scale = 10_000
        weights = np.array(
            [m.self_weight for m in self.result.frame_metrics],
            dtype=np.int64)
        total = self.result.total_weight
        numpy_q = np.floor(weights * scale // total).astype(np.int64)
        ours = np.array(
            [quantize(int(w), total, scale) for w in weights],
            dtype=np.int64)
        self.assertTrue(np.array_equal(numpy_q, ours))
        # Error envelope check vectorized.
        real = weights / total
        error = np.abs(real - ours / scale)
        self.assertTrue(np.all(error < 1 / scale))


class SessionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.samples_path = self.tmp / "s.ndjson"
        self.ledger_path = self.tmp / "ledger.sqlite"
        self.snapshot_dir = self.tmp / "snap"
        generate(rows=2_500, services=3, min_depth=3, max_depth=7,
                 weight_mode="skewed", seed=42,
                 output=str(self.samples_path))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def config(self, **overrides):
        cfg = SessionConfig(
            sample_path=str(self.samples_path),
            ledger_path=str(self.ledger_path),
            snapshot_dir=str(self.snapshot_dir),
            batch_size=500,
            top_k=8,
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_full_run_then_replay_matches(self):
        session = ProfileSession(self.config())
        summary = session.run()
        self.assertEqual(summary["batches"], 5)
        session.close()

        verifier = ProfileSession(self.config())
        outcome = verifier.replay()
        self.assertTrue(outcome["matched"])
        self.assertEqual(outcome["checked_frames"],
                         len(session.result.frame_metrics))
        verifier.close()

    def test_forced_kill_recover_resume(self):
        session = ProfileSession(self.config())
        with self.assertRaises(ForcedKill):
            session.run(crash_after_batches=2)
        info = session.recover()
        self.assertEqual(info.recoverable_batches, 2)
        self.assertEqual(info.committed_weight,
                         sum(b.weight_added for b
                             in session.ledger.list_batches()))
        self.assertGreater(info.needs_rerun_from_line,
                           info.last_committed_line)
        session.close()

        # A second process reopens the same ledger and only trusts commits.
        reopened = ProfileSession(self.config())
        reopen_info = reopened.recover()
        self.assertEqual(reopen_info.recoverable_batches, 2)
        summary = reopened.resume()
        self.assertEqual(summary["batches"], 5)
        reopened.close()

        # Final totals equal an independent single-shot run on a fresh ledger.
        fresh_cfg = self.config()
        fresh_cfg.ledger_path = str(self.tmp / "fresh.sqlite")
        fresh_cfg.snapshot_dir = str(self.tmp / "fresh_snap")
        fresh = ProfileSession(fresh_cfg)
        fresh.run()
        self.assertEqual(
            reopened.result.canonical_bytes()
            if reopened.result else None,
            fresh.result.canonical_bytes())
        fresh.close()

    def test_uncommitted_batch_not_visible(self):
        session = ProfileSession(self.config())
        with self.assertRaises(ForcedKill):
            session.run(crash_after_batches=1)
        session.close()
        conn = sqlite3.connect(str(self.ledger_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM batches").fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()

    def test_parameter_change_replaces_state(self):
        first = ProfileSession(self.config(scale=1_000))
        first.run()
        first.close()
        second = ProfileSession(self.config(scale=100))
        second.run()
        self.assertEqual(second.ledger.get_meta("scale"), 100)
        second.close()

    def test_interval_version_change_forces_snapshot_replace(self):
        first = ProfileSession(self.config())
        first.run()
        first.close()
        changed = ProfileSession(self.config(interval_version=2))
        summary = changed.run()
        self.assertTrue(summary["snapshot_replaced"])
        self.assertIn("version", summary["replacement_reason"])
        changed.close()


class ReplayMismatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.samples_path = self.tmp / "s.ndjson"
        self.ledger_path = self.tmp / "ledger.sqlite"
        self.snapshot_dir = self.tmp / "snap"
        write_ndjson(self.samples_path, [base_row(i) for i in range(12)])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _config(self):
        return SessionConfig(
            sample_path=str(self.samples_path),
            ledger_path=str(self.ledger_path),
            snapshot_dir=str(self.snapshot_dir),
            batch_size=4,
        )

    def test_tampered_snapshot_locates_first_mismatch(self):
        session = ProfileSession(self._config())
        session.run()
        session.close()

        snap_file = self.snapshot_dir / "snapshot.json"
        record = json.loads(snap_file.read_text())
        first_metric = record["result"]["frame_metrics"][0]
        target_frame = first_metric["frame"]
        original_value = first_metric["self_weight"]
        first_metric["self_weight"] = original_value + 999
        snap_file.write_text(json.dumps(record, sort_keys=True, indent=2))

        verifier = ProfileSession(self._config())
        with self.assertRaises(ReplayError) as ctx:
            verifier.replay()
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.mismatch.frame_name, target_frame)
        self.assertEqual(ctx.exception.mismatch.field, "self_weight")
        self.assertEqual(ctx.exception.mismatch.expected,
                         original_value + 999)
        verifier.close()


class GeneratorDeterminismTests(unittest.TestCase):
    def test_repeat_run_byte_identical_and_summary_stable(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            a = tmp / "a.ndjson"
            b = tmp / "b.ndjson"
            _, summary_a = generate(
                rows=1_000, services=2, min_depth=2, max_depth=6,
                weight_mode="uniform", seed=7, output=str(a))
            _, summary_b = generate(
                rows=1_000, services=2, min_depth=2, max_depth=6,
                weight_mode="uniform", seed=7, output=str(b))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            sa = {k: v for k, v in summary_a.items() if k != "output"}
            sb = {k: v for k, v in summary_b.items() if k != "output"}
            self.assertEqual(sa, sb)
            digest_a = sample_stream_digest(a)
            digest_b = sample_stream_digest(b)
            self.assertEqual(digest_a, digest_b)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ReadOnlyDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.samples_path = self.tmp / "s.ndjson"
        self.ledger_path = self.tmp / "ledger.sqlite"
        self.snapshot_dir = self.tmp / "snap"
        write_ndjson(self.samples_path, [base_row(i) for i in range(20)])
        self.cfg = SessionConfig(
            sample_path=str(self.samples_path),
            ledger_path=str(self.ledger_path),
            snapshot_dir=str(self.snapshot_dir),
            batch_size=5,
        )
        session = ProfileSession(self.cfg)
        session.run()
        session.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_render_is_read_only_and_stable(self):
        from slowprof.diagnostics import render_from_snapshot

        def fingerprint():
            conn = sqlite3.connect(str(self.ledger_path))
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "ORDER BY name").fetchall()
            counts = {}
            for (table,) in rows:
                counts[table] = conn.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            conn.close()
            snap = (self.snapshot_dir / "snapshot.json").read_bytes()
            return counts, snap

        before_counts, before_snap = fingerprint()
        first = render_from_snapshot(
            str(self.snapshot_dir), str(self.ledger_path),
            make_interval(None, None))
        second = render_from_snapshot(
            str(self.snapshot_dir), str(self.ledger_path),
            make_interval(None, None))
        after_counts, after_snap = fingerprint()
        self.assertEqual(first, second)          # stable rendering
        self.assertEqual(before_counts, after_counts)
        self.assertEqual(before_snap, after_snap)  # no state mutation
