import unittest

from hashlot.hashing import encode_group_columns, hash_group_key, map_partition
from hashlot.spec import Column
from hashlot.fixedpoint import quantize


class FloatGroupKeyTest(unittest.TestCase):
    def test_float_key_quantized_before_hash(self):
        bitmap = (1, 1, 1)
        col = Column(name="price", dtype="float64", bitmap=bitmap,
                     values=(1.2349, 1.2340, 1.2351))
        columns = {"price": col}
        hashes = [
            hash_group_key(encode_group_columns(columns, i, ("price",), 1000))
            for i in range(3)
        ]
        # 1.2349 与 1.2340 量化后都等于 1234，必须同哈希同分区
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[1], hashes[2])
        self.assertEqual(quantize(1.2349, 1000), quantize(1.2340, 1000))
        partitions = [map_partition(h, 8) for h in hashes]
        self.assertEqual(partitions[0], partitions[1])
        self.assertTrue(all(0 <= p < 8 for p in partitions))

    def test_null_group_key_uses_null_tag(self):
        col = Column(name="price", dtype="float64", bitmap=(0, 1),
                     values=(None, 1.0))
        columns = {"price": col}
        null_encoded = encode_group_columns(columns, 0, ("price",), 1000)
        self.assertEqual(null_encoded[0][0], 0x00)
        live_encoded = encode_group_columns(columns, 1, ("price",), 1000)
        self.assertEqual(live_encoded[0][0], 0x22)
        self.assertNotEqual(hash_group_key(null_encoded),
                            hash_group_key(live_encoded))


if __name__ == "__main__":
    unittest.main()
