"""E_DIVERGE byte-identity verification tests."""
import tempfile
import unittest

from colval.errors import DivergeError
from colval.formats import content_digest, make_bitmap
from colval.store import BlockStore, block_from_columns
from colval.verify import compare_blocks, verify_store_against_canonical


def _blk(block_id="blk-v", values=None, null_rows=None):
    values = [1, 2, 3, 4] if values is None else values
    null_rows = [] if null_rows is None else null_rows
    return block_from_columns(block_id, [
        ("a", "int64", make_bitmap(null_rows, len(values)), values),
    ], 3, 1024)


class TestVerify(unittest.TestCase):
    def test_identical_passes(self):
        compare_blocks(_blk(), _blk())
        self.assertEqual(content_digest(_blk()), content_digest(_blk()))

    def test_value_diverge_locates_row(self):
        with self.assertRaises(DivergeError) as cm:
            compare_blocks(_blk(), _blk(values=[1, 2, 99, 4]))
        ctx = cm.exception.context
        self.assertEqual(cm.exception.code, "E_DIVERGE")
        self.assertEqual((ctx["block_id"], ctx["column"], ctx["row"]),
                         ("blk-v", "a", 2))

    def test_null_bitmap_diverge(self):
        with self.assertRaises(DivergeError) as cm:
            compare_blocks(_blk(), _blk(null_rows=[1]))
        self.assertEqual(cm.exception.context["column"], "a")
        self.assertEqual(cm.exception.context["row"], 1)

    def test_store_level_verify(self):
        d = tempfile.mkdtemp(prefix="colval-vfy-")
        store = BlockStore(d)
        store.put_block(_blk("blk-v"))
        store.close()
        store = BlockStore(d)
        verified = verify_store_against_canonical(store, {"blk-v": _blk("blk-v")})
        self.assertEqual(verified, ["blk-v"])
        store.close()


if __name__ == "__main__":
    unittest.main()
