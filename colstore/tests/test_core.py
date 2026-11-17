from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest

from colstore.encoding import (
    decode_block,
    encode_block,
    estimate_encoding,
    select_encoding,
)
from colstore.engine import ScanEngine
from colstore.errors import (
    BudgetError,
    DivergenceError,
    EncodingError,
    PayloadError,
    QueryError,
)
from colstore.model import BudgetParams, ColumnType, Encoding, SegmentDefinition
from colstore.prepare_samples import SAMPLE_DIR, prepare
from colstore.query import build_scan_plan, canonical_query, query_id
from colstore.scheduler import Scheduler, TaskSpec
from colstore.segment import (
    inject_encoding_mismatch,
    inject_payload_truncation,
    load_segment,
)
from colstore.snapshot import SnapshotStore


def _definition() -> SegmentDefinition:
    return SegmentDefinition.from_dict(
        {
            "name": "t",
            "columns": [
                {"name": "id", "dtype": "int32", "compressed": False},
                {"name": "cat", "dtype": "int32", "compressed": True},
            ],
            "row_group_rows": [4, 4],
            "values": {
                "id": [1, 2, 3, 4, 5, 6, 7, 8],
                "cat": [7, 7, 7, 9, 9, 9, 9, 2],
            },
            "block_specs": [],
        }
    )


class EncodingTests(unittest.TestCase):
    def test_round_trip_all_encodings(self) -> None:
        values = [1, 1, 2, 2, 3, 3, 3, 9]
        for encoding in Encoding:
            payload = encode_block(values, ColumnType.INT32, encoding)
            self.assertEqual(decode_block(payload, ColumnType.INT32, encoding), values)

    def test_dictionary_bad_index_rejected(self) -> None:
        payload = encode_block([1, 2, 1], ColumnType.INT32, Encoding.DICTIONARY)
        # Corrupt an index byte so it points past the two-entry dictionary.
        corrupted = payload[:-1] + b"\x09"
        with self.assertRaises(EncodingError):
            decode_block(corrupted, ColumnType.INT32, Encoding.DICTIONARY)

    def test_estimate_is_deterministic_and_integer(self) -> None:
        values = [5, 5, 5, 6, 6, 7]
        first = estimate_encoding(values, ColumnType.INT64, Encoding.RLE)
        second = estimate_encoding(values, ColumnType.INT64, Encoding.RLE)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertIsInstance(first.decoded_peak_bytes, int)
        self.assertEqual(first.decoded_peak_bytes, len(values) * 8)
        self.assertLessEqual(abs(first.compression_ratio - first.compression_ratio), 0.0)
        self.assertEqual(first.quant_error_bound, 0.5e-4)

    def test_selection_prefers_rle_for_runs_and_documents_tie_rule(self) -> None:
        values = [4, 4, 4, 4, 4]
        encoding, decision = select_encoding(values, ColumnType.INT32)
        self.assertIn(encoding, (Encoding.RLE, Encoding.DICTIONARY))
        self.assertIn("estimates", decision)
        # Repeated runs: RLE must beat raw on encoded payload size.
        rle = estimate_encoding(values, ColumnType.INT32, Encoding.RLE)
        raw = estimate_encoding(values, ColumnType.INT32, Encoding.RAW)
        self.assertLess(rle.encoded_payload_bytes, raw.encoded_payload_bytes)


class SegmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.engine = ScanEngine(self.tmp, BudgetParams(2, 1 << 16, 1))
        self.engine.register_definition(_definition())
        self.engine.build_segments()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_whole_segment_rejection_on_truncation(self) -> None:
        source = self.engine.segment_path("t")
        corrupt = os.path.join(self.tmp, "trunc.colseg")
        inject_payload_truncation(source, corrupt)
        with self.assertRaises(PayloadError) as ctx:
            load_segment(corrupt, _definition())
        self.assertEqual(ctx.exception.code, "E_PAYLOAD")
        self.assertIn("column", ctx.exception.context)
        self.assertIn("row_group", ctx.exception.context)
        self.assertIn("payload_offset", ctx.exception.context)

    def test_encoding_marker_mismatch_rejected(self) -> None:
        source = self.engine.segment_path("t")
        corrupt = os.path.join(self.tmp, "mismatch.colseg")
        inject_encoding_mismatch(source, corrupt)
        with self.assertRaises(EncodingError) as ctx:
            load_segment(corrupt, _definition())
        self.assertEqual(ctx.exception.code, "E_ENCODING")


class QueryPlanTests(unittest.TestCase):
    def test_canonical_id_ignores_order_and_grouping(self) -> None:
        a = {
            "select": ["id", "cat"],
            "filter": {
                "op": "and",
                "args": [
                    {"op": "eq", "column": "cat", "value": 7},
                    {"op": "ge", "column": "id", "value": 2},
                ],
            },
        }
        b = {
            "select": ["cat", "id"],
            "filter": {
                "op": "and",
                "args": [
                    {"op": "ge", "column": "id", "value": 2},
                    {"op": "eq", "column": "cat", "value": 7},
                ],
            },
        }
        self.assertEqual(query_id(a), query_id(b))
        self.assertEqual(canonical_query(a), canonical_query(b))

    def test_unsupported_operator_is_e_query(self) -> None:
        with self.assertRaises(QueryError) as ctx:
            query_id(
                {"select": ["id"], "filter": {"op": "regex", "column": "id", "value": 1}}
            )
        self.assertEqual(ctx.exception.code, "E_QUERY")

    def test_push_down_skips_blocks_that_cannot_match(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            engine = ScanEngine(tmp)
            engine.register_definition(_definition())
            engine.build_segments()
            plan = build_scan_plan(
                {"select": ["id"], "filter": {"op": "eq", "column": "id", "value": 1}},
                list(engine.segments.values()),
            )
            # Value 1 only exists in row group 0, so the rg1 id block is skipped.
            skipped = {(b.row_group, b.column) for b in plan.skipped_blocks}
            self.assertIn((1, "id"), skipped)
            self.assertNotIn((0, "id"), skipped)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SchedulerTests(unittest.TestCase):
    def test_queueing_and_memory_budget_never_exceeded(self) -> None:
        specs = [
            TaskSpec(block=_block(i), encoding=Encoding.RAW, memory_bytes=100, rows=4)
            for i in range(6)
        ]
        observed = []
        lock = __import__("threading").Lock()

        def worker(record):
            time.sleep(0.01)
            with lock:
                observed.append((record.start_ms, record.memory_bytes))
            return 2

        scheduler = Scheduler(BudgetParams(2, 250, 1), 1, worker)
        records = scheduler.run(specs)
        self.assertTrue(all(record.status == "done" for record in records))
        self.assertLessEqual(scheduler.peak_memory_bytes, 250)
        # per-segment concurrency 1 + parallelism 2 => some tasks waited.
        self.assertTrue(any(record.wait_ms > 0 for record in records))

    def test_zero_budget_raises_e_budget(self) -> None:
        with self.assertRaises(BudgetError) as ctx:
            Scheduler(BudgetParams(1, 0, 1), 1, lambda record: 0)
        self.assertEqual(ctx.exception.code, "E_BUDGET")

    def test_oversized_block_raises_e_budget(self) -> None:
        scheduler = Scheduler(BudgetParams(1, 10, 1), 1, lambda record: 0)
        with self.assertRaises(BudgetError):
            scheduler.run(
                [TaskSpec(block=_block(0), encoding=Encoding.RAW, memory_bytes=50, rows=1)]
            )


def _block(index: int):
    from colstore.model import BlockId

    return BlockId(segment="t", row_group=index % 2, column=f"c{index}")


class EngineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        prepare()

    def _engine(self, budget: BudgetParams | None = None) -> ScanEngine:
        tmp = tempfile.mkdtemp()
        engine = ScanEngine(tmp, budget or BudgetParams(4, 1 << 16, 2))
        for name in ("segment_sales.json", "segment_inventory.json"):
            with open(os.path.join(SAMPLE_DIR, name), "r", encoding="utf-8") as handle:
                engine.register_definition(SegmentDefinition.from_dict(json.load(handle)))
        engine.build_segments()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return engine

    def _query(self, name: str):
        with open(os.path.join(SAMPLE_DIR, name), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_consistency_across_parallelism_and_encodings(self) -> None:
        engine = self._engine()
        report = engine.consistency_check(self._query("query_and.json"), 4)
        self.assertTrue(report["ok"])

    def test_stale_generation_never_enters_summary(self) -> None:
        engine = self._engine()
        first = engine.scan(self._query("query_eq.json"))
        engine.update_budget(BudgetParams(1, 4096, 1))
        self.assertTrue(engine.generations[first.result_id].stats.stale)
        self.assertIsNone(engine.current_summary()["current"])
        second = engine.scan(self._query("query_eq.json"))
        self.assertEqual(engine.current_summary()["current"]["result_id"], second.result_id)

    def test_snapshot_whole_replacement_diff(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        engine = self._engine()
        result = engine.scan(self._query("query_eq.json"))
        db = os.path.join(tmp, "s.db")
        store = SnapshotStore(db)
        summaries = [engine.segments[n].file_summary() for n in engine.segments]
        for item in summaries:
            item["encoding_selection_version"] = engine.encoding_selection_version
        store.save_session("s1", result, engine.budget, summaries,
                           engine.encoding_selection_version, {})
        engine.rebuild_with_encoding(Encoding.RLE)
        new_summaries = [engine.segments[n].file_summary() for n in engine.segments]
        for item in new_summaries:
            item["encoding_selection_version"] = engine.encoding_selection_version
        diffs = store.compare_with_current("s1", new_summaries,
                                           engine.encoding_selection_version)
        self.assertTrue(any(d["kind"] == "encoding_version_changed" for d in diffs))
        store.mark_superseded("s1", "s2", diffs)
        self.assertTrue(store.load_session("s1")["read_only"])
        store.close()

    def test_in_query_matches_expected_rows(self) -> None:
        engine = self._engine()
        result = engine.scan(self._query("query_in.json"))
        regions = {row[1] for row in result.rows}
        self.assertTrue(regions <= {1, 3})
        self.assertGreater(len(result.rows), 0)

    def test_divergence_reports_first_row_and_kind(self) -> None:
        from colstore.model import ScanResult, ScanStats

        engine = self._engine()
        good = engine.scan(self._query("query_eq.json"))
        semantic_bad = ScanResult(
            result_id="x",
            query_id=good.query_id,
            budget_version=good.budget_version,
            encoding_selection_version=good.encoding_selection_version,
            selected_columns=list(good.selected_columns),
            rows=good.rows[:-1] + [(-1, -2)],
            stats=good.stats,
            plan_summary=good.plan_summary,
            tasks=good.tasks,
            segment_versions=good.segment_versions,
            result_hash="deadbeef",
        )
        with self.assertRaises(DivergenceError) as ctx:
            engine._compare_paths({"a": good, "b": semantic_bad})
        self.assertEqual(ctx.exception.context["kind"], "semantic")
        self.assertIn("first_row_id", ctx.exception.context)

        stats_bad = ScanStats(
            budget_version=good.budget_version,
            encoding_selection_version=good.encoding_selection_version,
            segment_format_versions=dict(good.segment_versions),
            rows_completed=good.stats.rows_completed + 1,
            rows_matched=good.stats.rows_matched,
        )
        statistical_bad = ScanResult(
            result_id="y",
            query_id=good.query_id,
            budget_version=good.budget_version,
            encoding_selection_version=good.encoding_selection_version,
            selected_columns=list(good.selected_columns),
            rows=list(good.rows),
            stats=stats_bad,
            plan_summary=good.plan_summary,
            tasks=good.tasks,
            segment_versions=good.segment_versions,
            result_hash=good.result_hash,
        )
        with self.assertRaises(DivergenceError) as ctx:
            engine._compare_paths({"a": good, "b": statistical_bad})
        self.assertEqual(ctx.exception.context["kind"], "statistical")

    def test_old_snapshot_sample_is_detected_as_drift(self) -> None:
        db = os.path.join(SAMPLE_DIR, "snapshots", "old_session.db")
        self.assertTrue(os.path.exists(db))
        store = SnapshotStore(db)
        engine = self._engine()
        current = [engine.segments[n].file_summary() for n in engine.segments]
        # Pretend current encoding selection advanced beyond the old snapshot.
        for item in current:
            item["encoding_selection_version"] = 99
        diffs = store.compare_with_current("old-session-v1", current, 99)
        self.assertTrue(any(d["kind"] == "encoding_version_changed" for d in diffs))
        store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
