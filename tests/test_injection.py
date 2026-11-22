import tempfile
import unittest
from pathlib import Path

from mm_pipeline.injection import load_injections
from mm_pipeline.pipeline import execute_pipeline

from .fixtures import build_corpus


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="mm-inj-"))
        # Need at least 4 samples for the targeted sample ids.
        self.samples, self.config_path = build_corpus(self.base, count=5)
        # sample ids produced by the fixture are sample_000..sample_004.
        self.injection_path = Path(__file__).parent.parent / "configs" / "injections.yaml"

    def test_injection_ids_hit_and_map_to_codes(self):
        injections = load_injections(self.injection_path)
        ids = {inj.injection_id for inj in injections}
        self.assertEqual(
            ids, {"INJ-001", "INJ-002", "INJ-003", "INJ-004", "INJ-005"}
        )

        # Parameter-level injections make the config invalid; run them in
        # isolation so the pipeline can still produce a full diagnostic report
        # for the file-level faults.
        file_only = self.base / "injections_file.yaml"
        file_only.write_text(
            "injections:\n"
            "  - injection_id: INJ-001\n    injection_type: truncate_image\n"
            "    sample_id: sample_001\n    params: {keep_bytes: 24}\n"
            "  - injection_id: INJ-002\n    injection_type: missing_assertion_field\n"
            "    sample_id: sample_002\n    params: {assertion_index: 0, field: value}\n"
            "  - injection_id: INJ-003\n    injection_type: dangling_relation_region\n"
            "    sample_id: sample_003\n    params: {relation_index: 0, bad_region: 9999}\n"
        )
        report = execute_pipeline(
            self.samples,
            self.config_path,
            self.base / "work_file",
            injection_path=file_only,
            force_reset=True,
        )
        hits = {item["injection_id"]: item for item in report["injections"]}
        for inj_id, expected_code in (
            ("INJ-001", "E_IMAGE_FORMAT"),
            ("INJ-002", "E_ASSERTION_SCHEMA"),
            ("INJ-003", "E_RELATION_REF"),
        ):
            self.assertTrue(hits[inj_id]["hit"], f"{inj_id} did not hit")
            self.assertEqual(hits[inj_id]["expected_code"], expected_code)

        # Failure records carry sample, stage and business code.
        codes = {(f["sample"], f["code"], f["stage"]) for f in report["failures"]}
        self.assertIn(("sample_001", "E_IMAGE_FORMAT", "load"), codes)
        self.assertIn(("sample_002", "E_ASSERTION_SCHEMA", "load"), codes)
        self.assertIn(("sample_003", "E_RELATION_REF", "load"), codes)

        # Healthy samples still produced conclusions.
        healthy = {row["sample_id"] for row in report["region_results"]}
        self.assertIn("sample_000", healthy)
        self.assertIn("sample_004", healthy)
        self.assertNotIn("sample_001", healthy)

    def test_parameter_injections_raise_config_invalid(self):
        for inj_type, payload in (
            ("nonfinite_fusion_param", "value: .nan"),
            ("threshold_out_of_range", "threshold: 1.7"),
        ):
            with self.subTest(inj_type=inj_type):
                path = self.base / f"inj_{inj_type}.yaml"
                path.write_text(
                    "injections:\n"
                    f"  - injection_id: X\n    injection_type: {inj_type}\n"
                    f"    params: {{{payload}}}\n"
                )
                from mm_pipeline.errors import PipelineError
                with self.assertRaises(PipelineError) as ctx:
                    execute_pipeline(
                        self.samples,
                        self.config_path,
                        self.base / f"work_{inj_type}",
                        injection_path=path,
                        force_reset=True,
                    )
                self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")


if __name__ == "__main__":
    unittest.main()
