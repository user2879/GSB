import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowquery_diag.cli import _serialize  # noqa: E402
from slowquery_diag.engine import DiagnosticEngine  # noqa: E402
from slowquery_diag.invalidation import affected_queries, diff_scheme  # noqa: E402
from slowquery_diag.models import canonical_json

EXAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)


class EngineReloadTests(unittest.TestCase):
    def _workspace(self, tmpdir):
        decl_src = os.path.join(EXAMPLE_DIR, "declaration.json")
        decl_dst = os.path.join(tmpdir, "declaration.json")
        with open(decl_src, encoding="utf-8") as handle:
            payload = json.load(handle)
        for source in payload["sources"]:
            source["frame_file"] = os.path.join(
                EXAMPLE_DIR, os.path.basename(source["frame_file"])
            )
        with open(decl_dst, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return decl_dst, payload

    def test_scheme_version_monotonic_and_view_replaced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            decl_path, payload = self._workspace(tmpdir)
            engine = DiagnosticEngine(
                os.path.join(tmpdir, "ledger.db"), scale_exponent=1
            )
            first = engine.run(decl_path)
            self.assertEqual(first.scheme_version, 1)
            self.assertEqual(len(first.view_replacement.added_entries), 6)
            self.assertEqual(first.view_replacement.removed_entries, ())

            # Edit one source parameter: only its queries should be affected.
            payload["sources"][0]["sample_period_ms"] = 7000
            with open(decl_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            second = engine.run(decl_path)
            self.assertEqual(second.scheme_version, 2)
            self.assertIn("src_db_primary", second.diff.replaced_sources)
            self.assertEqual(second.diff.removed_sources, ())
            stale_keys = {entry.entry_key for entry in second.stale_entries}
            self.assertTrue(all(
                key in stale_keys for key in (
                    (101, "latency_us"), (101, "rows_scanned"),
                    (102, "latency_us"), (102, "rows_scanned"),
                    (103, "latency_us"), (103, "rows_scanned"),
                )
            ))
            self.assertTrue(all(
                "src_db_primary" in entry.related_sources
                for entry in second.stale_entries
            ))
            # Whole-view replacement still lists every current entry.
            current_keys = {
                (qid, metric)
                for qid in second.verdicts
                for metric in ("latency_us", "rows_scanned")
            }
            self.assertEqual(
                set(second.view_replacement.replaced_entries), current_keys
            )
            engine.close()

    def test_removed_candidate_propagates_to_query_invalidation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            decl_path, payload = self._workspace(tmpdir)
            removed = payload["candidates"].pop()
            with open(decl_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            # Direct invalidation helper check.
            candidate_to_query = {removed["candidate_id"]: 103}
            fake_diff = diff_scheme(
                tuple(), tuple(), (type("C", (), {
                    "candidate_id": removed["candidate_id"],
                    "original_query": removed["original_query"],
                    "rewritten_sql": removed["rewritten_sql"],
                    "target_sources": tuple(removed["target_sources"]),
                    "expected_gain_bps": removed["expected_gain_bps"],
                })(),), tuple(),
            )
            affected = affected_queries(
                fake_diff, {}, candidate_to_query
            )
            self.assertEqual(affected, {103})

    def test_repeated_runs_are_deterministic_and_replayable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            decl_path, _ = self._workspace(tmpdir)
            catalog_path = os.path.join(tmpdir, "catalog.json")

            def run_once(ledger_name):
                engine = DiagnosticEngine(
                    os.path.join(tmpdir, ledger_name), scale_exponent=1
                )
                result = engine.run(decl_path)
                engine.build_replay_catalog(result.bundle, catalog_path)
                config = {
                    "catalog_path": catalog_path,
                    "global_concurrency": 2,
                    "per_source_quota": {
                        s.source_id: 1 for s in result.bundle.sources
                    },
                    "socket_timeout_ms": 2000,
                }
                result = engine.run(
                    decl_path, replay=True, replay_config=config
                )
                payload = _serialize(result)
                # Monotonic-clock queue durations legitimately vary across
                # runs; drop them before comparing the deterministic core.
                for item in payload["replay"]:
                    item.pop("wait_ms", None)
                rendered = canonical_json(payload)
                engine.close()
                return rendered

            first = run_once("a.db")
            second = run_once("b.db")
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
