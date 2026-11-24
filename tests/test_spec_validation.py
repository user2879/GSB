import unittest

from fppdiag.errors import SpecError
from fppdiag.samplegen import build_sample_doc
from fppdiag.spec import load_spec


def base_doc():
    return build_sample_doc(rows=64)


class TestSpecValidation(unittest.TestCase):
    def assert_spec_error(self, doc, location=None, field=None):
        with self.assertRaises(SpecError) as ctx:
            load_spec(doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        if location is not None:
            self.assertEqual(ctx.exception.location, location)
        if field is not None:
            self.assertEqual(ctx.exception.field, field)

    def test_valid_doc_loads(self):
        spec = load_spec(base_doc())
        self.assertEqual(spec.session_scale_exp, 4)
        self.assertEqual(len(spec.stages), 6)

    def test_duplicate_stage_id(self):
        doc = base_doc()
        doc["stages"][1]["id"] = doc["stages"][0]["id"]
        doc["thresholds_bps"] = {s["id"]: 1500 for s in doc["stages"]}
        self.assert_spec_error(doc, doc["stages"][1]["id"], "id")

    def test_unknown_upstream_ref(self):
        doc = base_doc()
        doc["stages"][2]["refs"] = ["ghost"]
        self.assert_spec_error(doc, doc["stages"][2]["id"], "refs")

    def test_cycle_detected(self):
        doc = base_doc()
        # make clip_interact feed back into bucket_price via a ref loop
        doc["stages"][2]["refs"] = ["clip_interact"]
        doc["stages"][5]["refs"] = ["bucket_price"]
        self.assert_spec_error(doc, field="refs")

    def test_bad_scale_exponent(self):
        doc = base_doc()
        doc["stages"][0]["scale_exp"] = 9
        self.assert_spec_error(doc, doc["stages"][0]["id"], "scale_exp")

    def test_window_gap_rejected(self):
        doc = base_doc()
        doc["window"]["obs_start"] = doc["window"]["ref_end"] + 1
        self.assert_spec_error(doc, "<window>", "obs_start")

    def test_window_out_of_bounds(self):
        doc = base_doc()
        doc["window"]["obs_end"] = doc["window"]["obs_end"] + 5
        self.assert_spec_error(doc, "<window>", "obs_end")

    def test_bitmap_length_mismatch(self):
        doc = base_doc()
        doc["sources"][0]["null_bitmap_len"] = doc["sources"][0]["rows"] - 1
        self.assert_spec_error(doc, doc["sources"][0]["id"], "null_bitmap_len")

    def test_threshold_bounds(self):
        doc = base_doc()
        doc["thresholds_bps"]["std_price"] = 0
        self.assert_spec_error(doc, "std_price", "thresholds_bps")
        doc2 = base_doc()
        doc2["thresholds_bps"]["std_price"] = 10001
        self.assert_spec_error(doc2, "std_price", "thresholds_bps")

    def test_product_requires_two_refs(self):
        doc = base_doc()
        doc["stages"][4]["refs"] = ["std_price"]
        self.assert_spec_error(doc, "interact_pv", "refs")

    def test_first_violation_reported(self):
        doc = base_doc()
        doc["session_scale_exp"] = 7       # root-level violation appears first
        with self.assertRaises(SpecError) as ctx:
            load_spec(doc)
        self.assertEqual(ctx.exception.field, "session_scale_exp")


if __name__ == "__main__":
    unittest.main()
