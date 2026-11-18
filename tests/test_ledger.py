import os
import tempfile
import unittest

from swa.errors import BusinessError
from swa.ledger import Ledger
from swa.spec import parse_spec


def spec():
    return parse_spec(
        {
            "sources": [{"source_id": "a", "seed": 1, "event_count": 4}],
            "rules": [
                {
                    "rule_id": "r",
                    "window_seconds": 2,
                    "group_key": "group_key",
                    "aggregations": ["count", "sum", "max"],
                    "concurrency": 1,
                    "memory_budget": 4,
                    "version": 1,
                }
            ],
        }
    )


def result(start=0, version=1, digest="d1"):
    from swa.engine import WindowResult

    return WindowResult(
        rule_id="r",
        rule_version=version,
        window_start_ms=start,
        window_end_ms=start + 2000,
        groups=[{"group_key": "a", "count": 1, "sum": 2, "max": 2}],
        event_count=1,
        content_digest=digest,
    )


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "ledger.sqlite")

    def test_format_version_is_stored_as_integer(self):
        ledger = Ledger(self.path)
        value = ledger.conn.execute("SELECT value FROM ledger_meta WHERE key='format_version'").fetchone()[0]
        self.assertIsInstance(value, int)
        self.assertEqual(value, 1)
        ledger.close()

    def test_unsupported_ledger_version_is_e_ledger(self):
        with self.assertRaises(BusinessError) as ctx:
            Ledger(self.path, expected_format_version=999)
        self.assertEqual(ctx.exception.code, "E_LEDGER")

    def test_window_commits_are_idempotent_and_unique(self):
        ledger = Ledger(self.path)
        ledger.register_rule_versions(spec().rules)
        ledger.start_run("run-1", spec())
        self.assertEqual(ledger.commit_window("run-1", result()), "inserted")
        self.assertEqual(ledger.commit_window("run-1", result()), "reused")
        count = ledger.conn.execute("SELECT count(*) FROM window_results").fetchone()[0]
        self.assertEqual(count, 1)
        ledger.close()

    def test_rule_version_change_marks_old_windows_stale(self):
        ledger = Ledger(self.path)
        ledger.register_rule_versions(spec().rules)
        ledger.start_run("run-1", spec())
        ledger.commit_window("run-1", result(0))
        evolved = parse_spec(
            {
                "sources": [{"source_id": "a", "seed": 1, "event_count": 4}],
                "rules": [
                    {
                        "rule_id": "r",
                        "window_seconds": 3,
                        "group_key": "group_key",
                        "aggregations": ["count", "sum"],
                        "concurrency": 1,
                        "memory_budget": 4,
                        "version": 2,
                    }
                ],
            }
        )
        replaced = ledger.register_rule_versions(evolved.rules)
        self.assertIn("r", replaced)
        status = ledger.conn.execute("SELECT status FROM window_results").fetchone()[0]
        self.assertEqual(status, "stale")
        ledger.close()

    def test_same_version_reregistration_is_idempotent(self):
        ledger = Ledger(self.path)
        ledger.register_rule_versions(spec().rules)
        # Identical version and signature re-registers without error.
        replaced = ledger.register_rule_versions(spec().rules)
        self.assertEqual(replaced, {})
        ledger.close()

    def test_lower_rule_version_is_rejected(self):
        ledger = Ledger(self.path)
        ledger.register_rule_versions(spec().rules)
        import dataclasses

        lower = dataclasses.replace(spec().rules[0], version=0)
        with self.assertRaises(BusinessError) as ctx:
            ledger.register_rule_versions((lower,))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        ledger.close()

    def test_rolling_back_to_a_recorded_older_version_is_rejected(self):
        ledger = Ledger(self.path)
        ledger.register_rule_versions(spec().rules)
        evolved = parse_spec(
            {
                "sources": [{"source_id": "a", "seed": 1, "event_count": 4}],
                "rules": [
                    {
                        "rule_id": "r",
                        "window_seconds": 3,
                        "group_key": "group_key",
                        "aggregations": ["count"],
                        "concurrency": 1,
                        "memory_budget": 4,
                        "version": 2,
                    }
                ],
            }
        )
        ledger.register_rule_versions(evolved.rules)
        with self.assertRaises(BusinessError) as ctx:
            ledger.register_rule_versions(spec().rules)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        ledger.close()

    def test_same_version_parameter_drift_reports_every_window(self):
        ledger = Ledger(self.path)
        ledger.register_rule_versions(spec().rules)
        ledger.start_run("run-1", spec())
        ledger.commit_window("run-1", result(0))
        drift = parse_spec(
            {
                "sources": [{"source_id": "a", "seed": 1, "event_count": 4}],
                "rules": [
                    {
                        "rule_id": "r",
                        "window_seconds": 2,
                        "group_key": "group_key",
                        "aggregations": ["count", "sum", "max"],
                        "concurrency": 1,
                        "memory_budget": 2,
                        "version": 1,
                    }
                ],
            }
        )
        with self.assertRaises(BusinessError) as ctx:
            ledger.register_rule_versions(drift.rules)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        ledger.close()


if __name__ == "__main__":
    unittest.main()
