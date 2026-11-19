import json
import tempfile
import unittest
from pathlib import Path

from cache_invalidation.cli import main
from cache_invalidation.engine import Engine
from cache_invalidation.envcheck import check_environment
from cache_invalidation.gui import build_view_model
from cache_invalidation.model import EntryID
from cache_invalidation.store import Store
from tests.support import sample_graph


ROOT = Path(__file__).resolve().parent.parent


class GuiViewModelTest(unittest.TestCase):
    def test_view_model_rows_are_stable_and_complete(self):
        engine = Engine(sample_graph(), store=Store())
        engine.explicit_invalidate(EntryID("catalog", 100), 100)
        model_a = build_view_model(engine, "OK")
        model_b = build_view_model(engine, "OK")
        self.assertEqual(model_a, model_b)
        self.assertEqual(model_a.format_version, 3)
        self.assertEqual(model_a.status_code, "OK")
        self.assertEqual(
            model_a.inheritance_rows[0],
            ("catalog", "<root>"))
        sources = [row[0] for row in model_a.edge_rows]
        self.assertEqual(sources, sorted(sources))
        self.assertEqual(len(model_a.round_rows), 1)
        self.assertIn(
            ("catalog/100", 1, "invalidated"),
            model_a.version_rows)


class EnvCheckTest(unittest.TestCase):
    def test_environment_reports_stdlib_and_loopback(self):
        result = check_environment()
        self.assertTrue(result["python_minor_ok"])
        self.assertTrue(result["loopback"]["bindable"])
        for module_name in ("sqlite3", "json", "threading", "subprocess"):
            self.assertEqual(
                result["modules"][module_name], "available")
        self.assertIn("tkinter", result["modules"])


class CliInjectionTest(unittest.TestCase):
    def test_three_injection_commands_return_error_code(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        base = Path(work.name)
        ns = ROOT / "samples" / "namespaces.json"
        entries = ROOT / "samples" / "entries.json"

        cases = [
            ["inject", "--fault", "annotation_cycle",
             "--namespaces", str(ns), "--entries", str(entries)],
            ["inject", "--fault", "missing_inherit",
             "--namespaces", str(ns), "--entries", str(entries)],
        ]
        for argv in cases:
            self.assertEqual(main(argv), 1)

        engine = Engine(sample_graph(), store=Store())
        engine.explicit_invalidate(EntryID("catalog", 100), 100)
        snapshot = base / "snapshot.json"
        engine.save_snapshot(snapshot, 3)
        self.assertEqual(
            main(["inject", "--fault", "truncated_snapshot",
                  "--snapshot", str(snapshot)]),
            1)

    def test_log_file_contains_structured_business_code(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        log_path = Path(work.name) / "events.jsonl"
        ns = ROOT / "samples" / "namespaces.json"
        entries = ROOT / "samples" / "entries.json"
        code = main([
            "--log-file", str(log_path),
            "inject", "--fault", "annotation_cycle",
            "--namespaces", str(ns), "--entries", str(entries),
        ])
        self.assertEqual(code, 1)
        lines = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(lines[0]["code"], "E_GRAPH")
        self.assertIn("time", lines[0])
        self.assertIn("key", lines[0])


if __name__ == "__main__":
    unittest.main()
