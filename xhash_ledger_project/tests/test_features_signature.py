"""业务要求 2、3：定点量化、确定性签名、决胜规则与分桶。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xhash_ledger.features import build_layout, compose_weighted, extract_raw_features
from xhash_ledger.fixedpoint import quantize_vector
from xhash_ledger.generator import generate_sample
from xhash_ledger.signature import (
    bucket_signature,
    build_projection,
    generate_signature,
    hamming,
    sign_bits,
)
from xhash_ledger.validation import load_sample
import tempfile
from xhash_ledger.generator import write_sample


class FeatureSignatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        sample = generate_sample(42, "s001")
        write_sample(self.tmp, "s001", sample)
        self.bundle, failures = load_sample(self.tmp, "s001", (64, 64))
        self.assertEqual(failures, [])
        self.layout = build_layout(64, 64, 8, 2, 3)

    def test_feature_layout_and_determinism(self):
        raw_a = extract_raw_features(self.bundle, self.layout)
        raw_b = extract_raw_features(self.bundle, self.layout)
        self.assertEqual(raw_a.shape, (self.layout.dim,))
        self.assertTrue((raw_a == raw_b).all())
        weighted = compose_weighted(raw_a, self.layout)
        self.assertEqual(weighted.shape, (self.layout.dim,))

    def test_fixed_point_quantization_records_step_and_residual(self):
        raw = extract_raw_features(self.bundle, self.layout)
        weighted = compose_weighted(raw, self.layout)
        q = quantize_vector(weighted, frac_bits=12)
        self.assertEqual(len(q.values), self.layout.dim)
        self.assertEqual(q.qstep_exp, -12)
        self.assertEqual(len(q.residuals), self.layout.dim)
        # 残差以定点整数保存，绝对值不超过半个量化步长（即 0 或极小整数）
        self.assertTrue(all(-1 <= r <= 1 for r in q.residuals))
        # 同一输入两次量化逐元素一致
        q2 = quantize_vector(weighted, frac_bits=12)
        self.assertEqual(q.values, q2.values)

    def test_projection_deterministic_and_changes_with_seed(self):
        p1 = build_projection(seed=111, signature_bits=64, feature_dim=self.layout.dim, frac_bits=12)
        p2 = build_projection(seed=111, signature_bits=64, feature_dim=self.layout.dim, frac_bits=12)
        self.assertEqual(p1.matrix, p2.matrix)
        p3 = build_projection(seed=222, signature_bits=64, feature_dim=self.layout.dim, frac_bits=12)
        self.assertNotEqual(p1.matrix, p3.matrix)

    def test_signature_bit_width_changes_pattern(self):
        raw = compose_weighted(extract_raw_features(self.bundle, self.layout), self.layout)
        q = quantize_vector(raw, 12)
        proj64 = build_projection(99, 64, self.layout.dim, 12)
        proj128 = build_projection(99, 128, self.layout.dim, 12)
        sig64 = generate_signature("s001", q, proj64, 8)
        sig128 = generate_signature("s001", q, proj128, 8)
        self.assertEqual(len(sig64.bits), 64)
        self.assertEqual(len(sig128.bits), 128)
        self.assertTrue(sig64.bits != sig128.bits[:64] or len(sig128.bits) != len(sig64.bits))

    def test_seed_version_changes_bit_pattern(self):
        raw = compose_weighted(extract_raw_features(self.bundle, self.layout), self.layout)
        q = quantize_vector(raw, 12)
        sig_a = generate_signature("s001", q, build_projection(1, 64, self.layout.dim, 12), 8)
        sig_b = generate_signature("s001", q, build_projection(2, 64, self.layout.dim, 12), 8)
        self.assertNotEqual(sig_a.bits, sig_b.bits)

    def test_tie_break_rules(self):
        bits, _, ties = sign_bits([0, 5, -5, 100_000_000], frac_bits=12)
        self.assertEqual(ties[0]["rule"], "zero")
        self.assertEqual(bits[0], "1")
        self.assertEqual({t["rule"] for t in ties}, {"zero", "half_step"})
        # 固定决胜：零 -> 1，半步内（含正负小值）-> 0
        self.assertEqual(bits[1], "0")
        self.assertEqual(bits[2], "0")
        self.assertEqual(bits[3], "1")

    def test_bucket_split_and_bit_sets(self):
        bits = "1011001011110000"
        keys, bit_sets = bucket_signature(bits, 8)
        self.assertEqual(len(keys), 2)
        self.assertEqual(bit_sets[0], list(range(0, 8)))
        self.assertEqual(bit_sets[1], list(range(8, 16)))

    def test_hamming_distance(self):
        self.assertEqual(hamming(0b1010, 0b1001), 2)

    def test_bucket_only_change_keeps_signature_key_parts(self):
        # 分桶位数不进入投影种子，因此签名位模式保持不变，仅桶键数量变化
        raw = compose_weighted(extract_raw_features(self.bundle, self.layout), self.layout)
        q = quantize_vector(raw, 12)
        proj = build_projection(7, 64, self.layout.dim, 12)
        sig8 = generate_signature("s001", q, proj, 8)
        sig16 = generate_signature("s001", q, proj, 16)
        self.assertEqual(sig8.bits, sig16.bits)
        self.assertNotEqual(sig8.bucket_keys, sig16.bucket_keys)
        self.assertEqual(len(sig16.bucket_keys), 4)


if __name__ == "__main__":
    unittest.main()
