import shutil
import tempfile
import unittest

from fppdiag.engine import PipelineEngine
from fppdiag.errorchain import DOMINANT_SHARE_BPS
from fppdiag.samplegen import build_sample_doc
from fppdiag.spec import load_spec


class TestErrorChain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fpperr_")
        self.spec = load_spec(build_sample_doc(rows=120))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _chains(self):
        eng = PipelineEngine(self.spec, self.tmp)
        eng.execute()
        return eng.error_bounds

    def test_bounds_are_nonnegative_finite_integers(self):
        for c in self._chains().values():
            self.assertIsInstance(c["bound_ulp"], int)
            self.assertGreaterEqual(c["bound_ulp"], 0)

    def test_product_amplifies_upstream_bounds(self):
        chains = self._chains()
        # product of two standardized features must bound both inputs' errors
        self.assertGreaterEqual(
            chains["interact_pv"]["bound_ulp"],
            max(chains["std_price"]["bound_ulp"],
                chains["std_vol"]["bound_ulp"]))
        entry = chains["interact_pv"]["entries"][0]
        self.assertEqual(entry["source"], "product")
        self.assertIn("amplification_ulp", entry)

    def test_truncate_preserves_or_grows_bound(self):
        chains = self._chains()
        self.assertGreaterEqual(chains["clip_interact"]["bound_ulp"],
                                chains["interact_pv"]["bound_ulp"])

    def test_dominant_threshold_declared_and_resolvable(self):
        chains = self._chains()
        for c in chains.values():
            self.assertEqual(c["dominant_threshold_bps"], DOMINANT_SHARE_BPS)
        # bucket stage's own contribution dominates its feature chain
        self.assertEqual(chains["bucket_price"]["dominant_stage_id"],
                         "bucket_price")

    def test_order_independent_of_traversal(self):
        eng1 = PipelineEngine(self.spec, self.tmp)
        eng1.execute("stage_major")
        eng2 = PipelineEngine(self.spec, self.tmp)
        eng2.execute("pipeline_major")
        b1 = {k: c["bound_ulp"] for k, c in eng1.error_bounds.items()}
        b2 = {k: c["bound_ulp"] for k, c in eng2.error_bounds.items()}
        self.assertEqual(b1, b2)

    def test_log_routine_bound_respected(self):
        from fppdiag.errorchain import LOG_ROUTINE_BOUND
        chains = self._chains()
        self.assertLessEqual(chains["log_size"]["bound_ulp"],
                             LOG_ROUTINE_BOUND + 1)


if __name__ == "__main__":
    unittest.main()
