from __future__ import annotations

import unittest

import numpy as np

from dp_grad_sim.errors import SimError
from dp_grad_sim.quantization import (
    compress_tensor,
    compute_common_quant_params,
    decode_packet,
    packet_size,
    parse_packet,
    payload_size_for,
)


class QuantizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.array(
            [[1.25, -0.75, 0.0], [0.125, 2.5, -3.0]], dtype=np.float32
        )

    def test_roundtrip_for_all_schemes_and_widths(self) -> None:
        for scheme in ("symmetric", "asymmetric"):
            for bits in (2, 3, 4, 6, 8, 12, 16):
                params = compute_common_quant_params("tensor", [self.values], bits, scheme)
                packet = compress_tensor(self.values, "tensor", bits, scheme, params)
                decoded = decode_packet(
                    packet,
                    expected_name="tensor",
                    expected_shape=self.values.shape,
                    expected_bits=bits,
                    expected_scheme=scheme,
                )
                self.assertEqual(decoded.q.shape, self.values.shape)
                self.assertEqual(decoded.values.shape, self.values.shape)
                self.assertEqual(decoded.header.scale, params.scale)
                self.assertEqual(decoded.header.zero_point, params.zero_point)
                self.assertLessEqual(float(np.max(np.abs(decoded.values - self.values))), params.scale)

    def test_packet_length_is_value_independent(self) -> None:
        other = np.array(
            [[100.0, -0.1, 9.0], [0.001, -8.0, 2.0]], dtype=np.float32
        )
        for scheme in ("symmetric", "asymmetric"):
            for bits in (4, 6, 8):
                p1 = compress_tensor(self.values, "tensor", bits, scheme)
                p2 = compress_tensor(other, "tensor", bits, scheme)
                self.assertEqual(len(p1), len(p2))
                self.assertEqual(len(p1), packet_size("tensor", self.values.shape, bits))
                self.assertEqual(
                    len(parse_packet(p1).payload), payload_size_for(self.values.size, bits)
                )

    def test_repeated_decompression_is_elementwise_identical(self) -> None:
        packet = compress_tensor(self.values, "tensor", 6, "asymmetric")
        first = decode_packet(packet)
        second = decode_packet(packet)
        self.assertTrue(np.array_equal(first.q, second.q))
        self.assertTrue(np.array_equal(first.values, second.values))
        self.assertTrue(np.array_equal(first.values.tobytes(), second.values.tobytes()))

    def test_header_contains_tensor_name_bits_and_scale(self) -> None:
        params = compute_common_quant_params("tensor", [self.values], 6, "symmetric")
        packet = compress_tensor(self.values, "tensor", 6, "symmetric", params)
        view = parse_packet(packet)
        self.assertEqual(view.header.tensor_name, "tensor")
        self.assertEqual(view.header.bits, 6)
        self.assertEqual(view.header.scale, params.scale)
        self.assertEqual(view.header.scheme, "symmetric")

    def test_header_and_payload_validation(self) -> None:
        packet = compress_tensor(self.values, "tensor", 6, "symmetric")
        with self.assertRaises(SimError) as context:
            decode_packet(packet, expected_name="other")
        self.assertEqual(context.exception.code, "E_QUANT_HEADER_INVALID")

        corrupted = bytearray(packet)
        corrupted[-1] ^= 0xFF
        with self.assertRaises(SimError) as context:
            decode_packet(bytes(corrupted))
        self.assertEqual(context.exception.code, "E_QUANT_HEADER_INVALID")

        truncated = packet[:-3]
        with self.assertRaises(SimError) as context:
            decode_packet(truncated)
        self.assertEqual(context.exception.code, "E_QUANT_HEADER_INVALID")

    def test_common_scale_is_shared(self) -> None:
        second = self.values * 2.0
        params = compute_common_quant_params("tensor", [self.values, second], 8, "symmetric")
        p1 = compress_tensor(self.values, "tensor", 8, "symmetric", params)
        p2 = compress_tensor(second, "tensor", 8, "symmetric", params)
        self.assertEqual(parse_packet(p1).header.scale, parse_packet(p2).header.scale)
        self.assertEqual(len(p1), len(p2))


if __name__ == "__main__":
    unittest.main()
