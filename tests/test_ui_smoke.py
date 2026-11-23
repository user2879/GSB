from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests.support import build_engine, copy_samples, read_json


class TkUiSmokeTests(unittest.TestCase):
    def _tk_ready(self) -> bool:
        try:
            import tkinter  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Tk runtime unavailable: {exc}")
        if not os.environ.get("DISPLAY"):
            self.skipTest("no DISPLAY service")
        return True

    def test_window_renders_and_edits_observation(self) -> None:
        self._tk_ready()
        from config_arbiter.ui import ArbiterApp

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, specs, environment_path, views_path = build_engine(root)
            try:
                engine.run_all()
            finally:
                ledger.close()
            try:
                app = ArbiterApp(
                    environment_path=str(environment_path),
                    views_path=str(views_path),
                    db_path=str(root / "ledger.sqlite"),
                    snapshot_path=str(root / "snapshot.json"),
                    log_path=str(root / "log.jsonl"),
                )
            except Exception as exc:
                self.skipTest(f"cannot open Tk display: {exc}")

            errors: list[Exception] = []

            def exercise() -> None:
                try:
                    self.assertIsNotNone(app.model)
                    self.assertEqual(len(app.model["views"]), 3)
                    self.assertGreater(len(app.environment_tree.get_children()), 0)
                    self.assertGreater(len(app.chain_tree.get_children()), 0)
                    self.assertGreater(len(app.authority_tree.get_children()), 0)
                    self.assertGreater(len(app.error_tree.get_children()), 0)
                    self.assertGreater(len(app.version_tree.get_children()), 0)
                    target = next(
                        row_id
                        for row_id in app.environment_tree.get_children()
                        if app._env_rows[row_id]["key"] == "http.timeout_ms"
                        and app._env_rows[row_id]["environment_id"] == "prod-cn"
                    )
                    app.environment_tree.selection_set(target)
                    app._on_environment_select(None)
                    app.edit_vars["value_text"].set("3009.125")
                    app.save_environment_edit()
                    document = read_json(environment_path)
                    values = {
                        source["source_id"]: observation["value"]
                        for environment in document["environments"]
                        if environment["environment_id"] == "prod-cn"
                        for source in environment["sources"]
                        for observation in source["observations"]
                        if observation["key"] == "http.timeout_ms"
                    }
                    self.assertIn(3009.125, values.values())
                except Exception as exc:  # pragma: no cover - assertion bridge
                    errors.append(exc)
                finally:
                    app.destroy()

            app.after(900, exercise)
            app.mainloop()
            if errors:
                raise errors[0]

    def test_empty_ledger_shows_unknown_views_and_rerun_rows(self) -> None:
        self._tk_ready()
        from config_arbiter.ui import ArbiterApp

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment_path, views_path = copy_samples(root)
            try:
                app = ArbiterApp(
                    environment_path=str(environment_path),
                    views_path=str(views_path),
                    db_path=str(root / "ledger.sqlite"),
                    snapshot_path=str(root / "snapshot.json"),
                    log_path=str(root / "log.jsonl"),
                )
            except Exception as exc:
                self.skipTest(f"cannot open Tk display: {exc}")

            errors: list[Exception] = []

            def exercise() -> None:
                try:
                    self.assertEqual(app.model["completed_views"], [])
                    self.assertEqual(len(app.model["unknown_views"]), 3)
                    self.assertEqual(len(app.version_tree.get_children()), 3)
                    self.assertIn("待重跑 3", app.status_var.get())
                except Exception as exc:  # pragma: no cover - assertion bridge
                    errors.append(exc)
                finally:
                    app.destroy()

            app.after(700, exercise)
            app.mainloop()
            if errors:
                raise errors[0]


if __name__ == "__main__":
    unittest.main()
