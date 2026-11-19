from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mesh_audit.engine import CancellationToken
from mesh_audit.errors import BusinessError, ErrorCode
from mesh_audit.gui import load_workspace_view
from mesh_audit.persistence import StateStore
from tests.helpers import make_mesh, make_script, run_audit


def _snapshot_payloads(state_dir: Path) -> dict[int, dict]:
    records = StateStore(state_dir).valid_snapshots_for_resume(
        StateStore(state_dir).load_manifest()["version_id"]
    )
    return {
        record.step: record.payload
        for record in records
        if record.payload is not None
    }


class PersistenceTests(unittest.TestCase):
    def test_cancel_at_boundary_and_resume_matches_uninterrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, _ = run_audit(
                root, state_name="state-baseline", output_name="output-baseline"
            )
            mesh = make_mesh()
            script = make_script()
            from mesh_audit.engine import RunEngine

            interrupted_engine = RunEngine(
                mesh, script, root / "state-resume", root / "output-resume"
            )
            token = CancellationToken()

            def progress(step: int, _total: int, _label: str) -> None:
                if step == 2:
                    token.request(130, "TEST_BOUNDARY")

            interrupted = interrupted_engine.run(cancellation=token, progress=progress)
            self.assertEqual(interrupted.exit_code, 130)
            self.assertEqual(interrupted.completed_steps, 2)
            self.assertFalse((root / "output-resume" / "report.json").exists())

            resumed = interrupted_engine.run()
            self.assertEqual(resumed.exit_code, 0)
            self.assertEqual(resumed.resumed_from_step, 2)
            baseline_payloads = _snapshot_payloads(root / "state-baseline")
            resumed_payloads = _snapshot_payloads(root / "state-resume")
            self.assertEqual(sorted(baseline_payloads), sorted(resumed_payloads))
            for step in baseline_payloads:
                self.assertEqual(
                    baseline_payloads[step]["mesh"], resumed_payloads[step]["mesh"]
                )
                self.assertEqual(
                    baseline_payloads[step]["execution_count"],
                    resumed_payloads[step]["execution_count"],
                )
                self.assertEqual(
                    baseline_payloads[step]["operator_effect"],
                    resumed_payloads[step]["operator_effect"],
                )
                self.assertEqual(
                    baseline_payloads[step]["error"]["vertex_errors"],
                    resumed_payloads[step]["error"]["vertex_errors"],
                )

    def test_tampered_snapshot_is_rejected_and_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, engine = run_audit(root)
            self.assertEqual(result.exit_code, 0)
            snapshot = root / "state" / "current" / "step-0002.json"
            original_text = snapshot.read_text(encoding="utf-8")
            wrapper = json.loads(original_text)
            wrapper["payload"]["mesh"]["fixed"][0][0] += 1
            snapshot.write_text(json.dumps(wrapper), encoding="utf-8")
            tampered_text = snapshot.read_text(encoding="utf-8")
            with self.assertRaises(BusinessError) as caught:
                engine.run()
            self.assertEqual(caught.exception.code, ErrorCode.E_SNAPSHOT_INVALID)
            self.assertEqual(snapshot.read_text(encoding="utf-8"), tampered_text)
            inspection = StateStore(root / "state").inspect()
            self.assertEqual(inspection.latest_contiguous_valid_step, 1)
            invalid = [record for record in inspection.snapshots if not record.valid]
            self.assertEqual(len(invalid), 1)
            self.assertEqual(invalid[0].step, 2)

            self.assertEqual(inspection.latest_contiguous_valid_step, 1)

    def test_gui_view_marks_tampered_snapshot_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh_path = root / "mesh.json"
            script_path = root / "script.json"
            from mesh_audit.sample import deterministic_mesh_payload, deterministic_repair_script

            mesh_path.write_text(json.dumps(deterministic_mesh_payload()), encoding="utf-8")
            script_path.write_text(json.dumps(deterministic_repair_script()), encoding="utf-8")
            run_audit(root)
            snapshot = root / "state" / "current" / "step-0002.json"
            wrapper = json.loads(snapshot.read_text(encoding="utf-8"))
            wrapper["payload"]["mesh"]["reference"][0][0] += 0.1
            snapshot.write_text(json.dumps(wrapper), encoding="utf-8")
            view = load_workspace_view(str(mesh_path), str(script_path), str(root / "state"))
            step_two = next(step for step in view.steps if step.step == 2)
            self.assertFalse(step_two.valid)
            self.assertIn("integrity", step_two.reason or "")
            self.assertIn(ErrorCode.E_SNAPSHOT_INVALID, view.load_codes)

    def test_version_change_archives_previous_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = run_audit(root, script=make_script({"concurrency": 2}))
            first_version = first.version_id
            second, engine = run_audit(
                root,
                script=make_script({"concurrency": 6}),
                restart=False,
            )
            self.assertNotEqual(first.version_id, second.version_id)
            archives = engine.store.archive_entries()
            self.assertEqual(len(archives), 1)
            archived_manifest = json.loads(
                (Path(archives[0]) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(archived_manifest["version_id"], first_version)
            current_manifest = engine.store.load_manifest()
            self.assertEqual(current_manifest["version_id"], second.version_id)


if __name__ == "__main__":
    unittest.main()
