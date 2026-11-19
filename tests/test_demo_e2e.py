import json
import tempfile
import unittest
from pathlib import Path

from dwb_diag.cli import main
from dwb_diag.gui_model import collect_model


class TestDemoEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "demo"

    def tearDown(self):
        self.tmp.cleanup()

    def test_demo_command_full_pipeline(self):
        rc = main(["demo", "--root", str(self.root)])
        self.assertEqual(rc, 0)
        # 关键产物全部存在
        for name in [
            "doublewrite.dwb",
            "main.pages",
            "page_versions.json",
            "views.json",
            "crash_markers.jsonl",
            "sessions.sqlite",
            "events.log.jsonl",
        ]:
            self.assertTrue((self.root / name).exists(), name)

    def test_inject_command_reports_business_codes(self):
        main(["demo", "--root", str(self.root)])
        for kind, code in [
            ("main_corrupt", "E_CORRUPT"),
            ("buffer_truncate", "E_PAGE"),
            ("depend_cycle", "E_DEPEND"),
            ("snapshot_stale", "E_SNAPSHOT"),
        ]:
            target = self.root.parent / f"inj-{kind}"
            import shutil

            shutil.copytree(self.root, target)
            rc = main(["inject", "--root", str(target), "--kind", kind, "--page", "0"])
            self.assertEqual(rc, 0)
            log = (target / "events.log.jsonl").read_text(encoding="utf-8")
            self.assertIn(code, log, f"{kind} should produce {code}")

    def test_replay_tamper_command_e_replay(self):
        main(["demo", "--root", str(self.root)])
        page = 0
        rc = main(["replay", "--root", str(self.root), "--tamper-page", str(page)])
        self.assertEqual(rc, 0)
        log = (self.root / "events.log.jsonl").read_text(encoding="utf-8")
        self.assertIn("E_REPLAY", log)

    def test_gui_model_stable_tabs_without_tk(self):
        main(["demo", "--root", str(self.root)])
        model = collect_model(self.root, 1024, 64)
        self.assertEqual(
            set(model["tabs"]), {"buffer", "main", "markers", "codes", "snapshots"}
        )
        offsets = [row["offset"] for row in model["tabs"]["buffer"]["mirrors"]]
        self.assertEqual(offsets, sorted(offsets))
        page_nos = [row["page_no"] for row in model["tabs"]["main"]["pages"]]
        self.assertEqual(page_nos, sorted(page_nos))

    def test_keyspace_blocked_until_recovery_then_answers(self):
        main(["demo", "--root", str(self.root)])
        rc = main(["keyspace", "--root", str(self.root)])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
