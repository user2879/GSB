from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from edge_ledger.engine import build_workspace
from edge_ledger.workspace import WorkspaceEditor, apply_command_file
from edge_ledger.loader import write_yaml

from .helpers import make_sample_workspace


class WorkspaceEditorTests(unittest.TestCase):
    def test_editor_saves_relative_files_under_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            editor = WorkspaceEditor.from_root(root)
            editor.apply_command(
                {
                    "op": "add_samples",
                    "batch_id": "batch-extra",
                    "samples": [
                        {
                            "sample_id": "S-EXTRA",
                            "template_id": "tpl_note",
                            "template_revision": 1,
                            "answer_template_id": "tpl_note",
                            "answer_template_revision": 1,
                            "source_shard": "shard-z",
                            "slot_fills": [
                                {"name": "topic", "value": "relative save"},
                                {"name": "body", "value": "must stay under root"},
                            ],
                        }
                    ],
                }
            )
            editor.save()
            self.assertTrue(any("S-EXTRA" in path.read_text(encoding="utf-8") for path in (root / "data" / "samples").glob("*.yaml")))
            # A path that does not exist under this temporary root would fail this build.
            result = build_workspace(root)
            self.assertIn("S-EXTRA", {row["sample_id"] for row in result.result["samples"]})

    def test_invalid_command_file_is_transactional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            commands = root / "commands" / "bad.yaml"
            write_yaml(
                commands,
                {
                    "commands": [
                        {"id": "BAD-1", "op": "not_a_real_operation"},
                        {"id": "BAD-2", "op": "delete_sample", "sample_id": "missing"},
                    ]
                },
            )
            before = (root / "data" / "templates.yaml").read_bytes()
            editor, failures = apply_command_file(root, commands)
            self.assertIsNone(editor)
            self.assertEqual(len(failures), 2)
            self.assertTrue(all(failure.code == "E_COMMAND_INVALID" for failure in failures))
            self.assertEqual(before, (root / "data" / "templates.yaml").read_bytes())


class GuiSmokeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("DISPLAY"), "DISPLAY is not configured")
    def test_gui_builds_and_displays_authoritative_quick_view(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:
            self.skipTest(f"tkinter is unavailable: {exc}")
        from edge_ledger.gui import DiagnosticApp

        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            master = tk.Tk()
            try:
                app = DiagnosticApp(master, root)
                master.update_idletasks()
                self.assertGreater(len(app.template_tree.get_children()), 0)
                self.assertGreater(len(app.sample_tree.get_children()), 0)
                self.assertEqual(app.quick_match_var.get(), "Quick view: MATCH")
                app.threshold_var.set("850")
                app.apply_threshold()
                master.update_idletasks()
                self.assertEqual(app._threshold(), 850)
                self.assertEqual(app.quick_match_var.get(), "Quick view: MATCH")
                template_id = "tpl_note@1"
                app.template_tree.selection_set(template_id)
                app._on_template_select()
                original_digest = next(
                    row["render_digest"]
                    for row in app.result.result["samples"]
                    if row["sample_id"] == "S0009"
                )
                app.instruction_edit.delete("1.0", "end")
                app.instruction_edit.insert("1.0", "Edited note about {topic}: {body}")
                app.save_template_edit()
                master.update_idletasks()
                changed_digest = next(
                    row["render_digest"]
                    for row in app.result.result["samples"]
                    if row["sample_id"] == "S0009"
                )
                self.assertNotEqual(original_digest, changed_digest)
                self.assertEqual(app.quick_match_var.get(), "Quick view: MATCH")
            finally:
                master.destroy()


if __name__ == "__main__":
    unittest.main()
