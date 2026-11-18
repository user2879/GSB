import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.generate_samples import make_texture
from texmat.loader import scan_textures


def good_png(path: Path, size: int, seed: int) -> None:
    Image.fromarray(make_texture(size, size, seed, [(4, 4, 2)]), mode="RGB").save(path)


class LoaderFaultTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_decodable_plus_corrupt_isolation(self):
        size = 16
        good_png(self.tmp / "good1.png", size, 1)
        good_png(self.tmp / "good2.png", size, 2)
        (self.tmp / "notpng.png").write_bytes(b"this is definitely not a png file")
        image = Image.fromarray(make_texture(size, size, 3, [(2, 2, 1)]), mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raw = buffer.getvalue()
        (self.tmp / "truncated.png").write_bytes(raw[: len(raw) // 2])

        textures, failures = scan_textures(self.tmp, size)
        self.assertEqual(set(textures), {"good1", "good2"})
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(f.code == "E_TEXTURE_DECODE" for f in failures))
        self.assertEqual({f.target for f in failures}, {"notpng.png", "truncated.png"})
        keys = [f.order_key for f in failures]
        self.assertEqual(keys, sorted(keys))

    def test_size_mismatch_flagged(self):
        good_png(self.tmp / "sixteen.png", 16, 1)
        textures, failures = scan_textures(self.tmp, expected_size=32)
        self.assertFalse(textures)
        self.assertEqual(failures[0].code, "E_TEXTURE_DECODE")
        self.assertIn("size", failures[0].message)

    def test_pixels_uint8(self):
        size = 12
        good_png(self.tmp / "only.png", size, 5)
        textures, failures = scan_textures(self.tmp, size)
        self.assertFalse(failures)
        pixels = textures["only"].pixels
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertEqual(pixels.shape, (size, size, 3))


if __name__ == "__main__":
    unittest.main()
