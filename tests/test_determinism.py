import shutil
import tempfile
import unittest

from fppdiag.engine import PipelineEngine
from fppdiag.errorchain import build_error_chains
from fppdiag.samplegen import build_sample_doc
from fppdiag.spec import load_spec


class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fppdet_")
        self.doc = build_sample_doc(rows=120)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, block_size, schedule):
        doc = self._doc_with_block(block_size)
        spec = load_spec(doc)
        eng = PipelineEngine(spec, self.tmp)
        eng.execute(schedule)
        values = {sid: run.all_values() for sid, run in eng.runs.items()}
        digests = {sid: run.digest() for sid, run in eng.runs.items()}
        chains = build_error_chains(spec, eng)
        return values, digests, chains

    def _doc_with_block(self, block_size):
        doc = self.doc
        doc = {**doc, "block_size": block_size}
        return doc

    def test_block_partition_byte_identical(self):
        v1, d1, c1 = self._run(7, "stage_major")
        v2, d2, c2 = self._run(64, "stage_major")
        self.assertEqual(v1, v2)
        self.assertEqual(d1, d2)
        self.assertEqual({k: c["bound_ulp"] for k, c in c1.items()},
                         {k: c["bound_ulp"] for k, c in c2.items()})

    def test_schedule_byte_identical(self):
        v1, d1, _ = self._run(13, "stage_major")
        v2, d2, _ = self._run(13, "pipeline_major")
        self.assertEqual(v1, v2)
        self.assertEqual(d1, d2)

    def test_null_propagation_present(self):
        spec = load_spec(self._doc_with_block(16))
        eng = PipelineEngine(spec, self.tmp)
        eng.execute()
        for sid, run in eng.runs.items():
            self.assertIn(None, run.all_values(), f"{sid} should propagate nulls")

    def test_saturation_flags(self):
        from fppdiag import fixedpoint as fp
        from fppdiag.spec import StageSpec
        from fppdiag.stages import run_standardize
        stage = StageSpec("z", "standardize", 0, None, (),
                          {"mean": 0, "sd": 1, "sd_scale_exp": 0})
        res = run_standardize(stage, [fp.INT32_MAX, None, 1], 0)
        self.assertTrue(res.saturated[0])
        self.assertIsNone(res.values[1])
        self.assertFalse(res.saturated[1])


if __name__ == "__main__":
    unittest.main()
