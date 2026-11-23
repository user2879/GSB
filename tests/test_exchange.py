import unittest

from tspworkbench.constants import V1_LAYOUT, v1_line_length
from tspworkbench.errors import E_FIELD_LAYOUT, E_DUPLICATE_KEY, E_LAG_NEGATIVE, E_MODEL_ENUM
from tspworkbench.exchange.ir import ExchangeIR, WindowWeight
from tspworkbench.exchange.v1_codec import parse_v1, write_v1, encode_line, fingerprint_v1
from tspworkbench.exchange.v2_codec import write_v2, parse_v2
from tspworkbench.exchange.converter import v2_to_v1, v1_to_v2, fingerprint_v2


def _ir():
    return ExchangeIR(
        exchange_version="v2", config_fingerprint="cfg123", version_fingerprint="",
        missing_strategy="mean_fill",
        weights=[
            WindowWeight("S000", 0, "naive", 0.2, 0, "exact"),
            WindowWeight("S000", 0, "moving_average", 0.30047, 5, "exact"),
        ],
        predictions=[], metrics=[],
    )


class ExchangeTests(unittest.TestCase):
    def test_v1_roundtrip_and_layout_error(self):
        ir = _ir()
        ir.version_fingerprint = fingerprint_v2(ir)
        v1_ir, sidecar = v2_to_v1(ir)
        text = ""
        # 手动渲染以读取文本
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.v1.txt"
            write_v1(v1_ir, p)
            text = p.read_text(encoding="utf-8")
        failures = []
        parsed = parse_v1(text, failures)
        self.assertEqual(failures, [])
        self.assertEqual(parsed.weights[0].weight, 0.2)
        # 截断残差
        ma = next(w for w in sidecar.entries if w.model == "moving_average")
        self.assertAlmostEqual(ma.residual, 0.00047, places=7)

        # 破坏一行长度 -> E_FIELD_LAYOUT
        broken = text.splitlines()
        broken[1] = broken[1][:-3]
        failures2 = []
        parse_v1("\n".join(broken), failures2)
        self.assertTrue(any(f["code"] == E_FIELD_LAYOUT for f in failures2))

    def test_sidecar_restores_named_info(self):
        ir = _ir()
        ir.version_fingerprint = fingerprint_v2(ir)
        v1_ir, sidecar = v2_to_v1(ir)
        # v1 默认缺失策略是 forward_fill，旁路里保留 mean_fill
        self.assertEqual(v1_ir.missing_strategy, "forward_fill")
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.v1.txt"
            write_v1(v1_ir, p)
            back = v1_to_v2(p.read_text(encoding="utf-8"), sidecar)
        self.assertEqual(back.missing_strategy, "mean_fill")
        ma = next(w for w in back.weights if w.model == "moving_average")
        self.assertAlmostEqual(ma.weight, 0.30047, places=7)
        self.assertEqual(ma.precision_label, "exact")

    def test_validation_duplicate_and_enum(self):
        ir = _ir()
        ir.weights.append(WindowWeight("S000", 0, "naive", 0.1, 0))
        with self.assertRaises(Exception) as ctx:
            ir.validate()
        self.assertEqual(ctx.exception.code, E_DUPLICATE_KEY)

        bad = _ir()
        bad.weights[0].lag_order = -1
        with self.assertRaises(Exception) as ctx:
            bad.validate()
        self.assertEqual(ctx.exception.code, E_LAG_NEGATIVE)

        bad2 = _ir()
        bad2.weights[0].model = "not-a-model"
        with self.assertRaises(Exception) as ctx:
            bad2.validate()
        self.assertEqual(ctx.exception.code, E_MODEL_ENUM)

    def test_v2_json_roundtrip(self):
        ir = _ir()
        ir.version_fingerprint = fingerprint_v2(ir)
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.v2.json"
            write_v2(ir, p)
            back = parse_v2(p.read_text(encoding="utf-8"))
        self.assertEqual(back.missing_strategy, "mean_fill")
        self.assertAlmostEqual(back.weights[1].weight, 0.30047, places=7)

    def test_sidecar_fingerprint_mismatch_rejected(self):
        from tspworkbench.errors import E_SIDECAR_MISMATCH
        ir = _ir()
        ir.version_fingerprint = fingerprint_v2(ir)
        v1_ir, sidecar = v2_to_v1(ir)
        sidecar.config_fingerprint = "tampered"
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.v1.txt"
            write_v1(v1_ir, p)
            with self.assertRaises(Exception) as ctx:
                v1_to_v2(p.read_text(encoding="utf-8"), sidecar)
        self.assertEqual(ctx.exception.code, E_SIDECAR_MISMATCH)


if __name__ == "__main__":
    unittest.main()
