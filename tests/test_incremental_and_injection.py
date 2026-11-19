from __future__ import annotations

import tempfile
import unittest

from spectral_workspace.errors import DomainError
from spectral_workspace.injection import validate_injection_config
from spectral_workspace.synth import demo_operations, generate_sample_scene
from spectral_workspace.workspace import open_workspace


class IncrementalAndInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = generate_sample_scene()
        self.operations = demo_operations(self.sample)

    def test_incremental_closures_and_reuse_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = open_workspace(directory)
            results = []
            for operation in self.operations[:6]:
                results.append(engine.apply(operation))
            self.assertEqual(engine.verify()["status"], "ok")

            revision2 = results[1]["delta"]
            self.assertEqual(revision2["recomputed_count"], 5)
            self.assertEqual(revision2["reused_count"], 10)
            self.assertEqual(
                {item["stage_id"] for item in revision2["recomputed"]},
                {
                    "spectral_xyz:mat_oxide:light_day",
                    "spectral_xyz:mat_oxide:light_warm",
                    "adapted_xyz:mat_oxide:light_day",
                    "adapted_xyz:mat_oxide:light_warm",
                    "material_color:mat_oxide",
                },
            )

            revision3 = results[2]["delta"]
            self.assertEqual(
                [item["stage_id"] for item in revision3["recomputed"]],
                ["material_color:mat_polymer"],
            )
            self.assertEqual(revision3["reused_count"], 14)

            revision4 = results[3]["delta"]
            self.assertEqual(revision4["recomputed_count"], 9)
            self.assertFalse(
                any(":light_day" in item["stage_id"] for item in revision4["recomputed"])
            )
            reused = {item["stage_id"]: item for item in revision4["reused"]}
            day_stage = reused["spectral_xyz:mat_oxide:light_day"]
            self.assertTrue(day_stage["reused"])
            self.assertGreaterEqual(day_stage["source_version"], 1)
            self.assertTrue(day_stage["content_hash"].startswith("sha256:"))

            self.assertTrue(all(operation["version"] for operation in engine.operations))
            self.assertEqual(len(engine.history), 6)
            engine.close()

    def test_switching_observer_does_not_reuse_old_cmf_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = open_workspace(directory)
            engine.apply(self.operations[0])
            before = {
                stage_id: stage["content_hash"]
                for stage_id, stage in engine.stages.items()
                if stage["kind"] == "spectral_xyz"
            }
            result = engine.apply(
                {
                    "version": 2,
                    "op": "set_observer",
                    "payload": {"observer": "CIE1964_10"},
                }
            )
            after = {
                stage_id: stage["content_hash"]
                for stage_id, stage in engine.stages.items()
                if stage["kind"] == "spectral_xyz"
            }
            self.assertEqual(set(before), set(after))
            self.assertTrue(all(before[key] != after[key] for key in before))
            self.assertEqual(result["delta"]["reused_count"], 0)
            engine.close()

    def test_stage_failure_injection_returns_stale_and_commits_nothing(self) -> None:
        injection = validate_injection_config(
            {
                "schema": "spectral-injection/v1",
                "stage_failure": {
                    "operation_version": 4,
                    "stage_kind": "adapted_xyz",
                    "material_id": "mat_oxide",
                    "light_id": "light_day",
                    "delta_xyz": [0.25, 0.0, 0.0],
                },
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            engine = open_workspace(directory, injection=injection)
            for operation in self.operations[:3]:
                engine.apply(operation)
            with self.assertRaises(DomainError) as context:
                engine.apply(self.operations[3])
            error = context.exception
            self.assertEqual(error.code, "E_STAGE_STALE")
            self.assertEqual(
                error.details["stage_id"], "adapted_xyz:mat_oxide:light_day"
            )
            self.assertEqual(error.details["first_divergence"]["field"], "/product/xyz/0")
            self.assertEqual(engine.revision, 3)
            self.assertEqual(engine.snapshot_version, 3)
            engine.close()


if __name__ == "__main__":
    unittest.main()
