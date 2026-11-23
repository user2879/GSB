import os
import tempfile
import unittest

from repairbench.content import (
    BlockRegistry,
    block_digest,
    generate_content,
)
from repairbench.errors import E_INTEGRITY, WorkbenchError
from repairbench.spec import BlockSpec


class TestContent(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.block = BlockSpec("b1", 10000, "seed-1", "n1", 500)

    def test_deterministic_bytes_and_digest(self):
        c1 = generate_content("seed-1", "b1", 10000)
        c2 = generate_content("seed-1", "b1", 10000)
        self.assertEqual(c1, c2)
        self.assertEqual(len(c1), 10000)
        self.assertEqual(
            block_digest("b1", 10000, c1), block_digest("b1", 10000, c2)
        )

    def test_registry_materialize_idempotent(self):
        reg = BlockRegistry(self.dir)
        d1 = reg.materialize(self.block)
        with open(reg.path_for("b1"), "rb") as fh:
            bytes1 = fh.read()
        d2 = reg.materialize(self.block)
        with open(reg.path_for("b1"), "rb") as fh:
            bytes2 = fh.read()
        self.assertEqual(d1, d2)
        self.assertEqual(bytes1, bytes2)
        self.assertIn("b1", reg.digests)  # 修复前已登记

    def test_tamper_detected(self):
        reg = BlockRegistry(self.dir)
        reg.materialize(self.block)
        path = reg.path_for("b1")
        with open(path, "r+b") as fh:
            first = fh.read(1)
            fh.seek(0)
            fh.write(bytes([first[0] ^ 0xFF]))
        with self.assertRaises(WorkbenchError) as cm:
            reg.verify("b1")
        self.assertEqual(cm.exception.code, E_INTEGRITY)


if __name__ == "__main__":
    unittest.main()
