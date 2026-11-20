import json
import os
import tempfile
import unittest

import numpy as np

from animpipe.animation import (
    CURRENT_SCHEMA,
    build_legacy_document,
    dump_json_animation,
    load_json_animation,
    migrate_schema,
    migration_hash,
)
from animpipe.binary import BinarySpec, decode_binary, dump_binary, encode_binary
from animpipe.errors import E_INDEX_OVERFLOW, E_SCHEMA_VERSION
from animpipe.samples import generate_all


class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.sample = generate_all(cls.tmp)
        cls.anim = cls.sample.source_anim

    def test_json_roundtrip(self):
        path = os.path.join(self.tmp, "a.json")
        dump_json_animation(self.anim, path)
        back = load_json_animation(path)
        np.testing.assert_allclose(
            back.rotations, self.anim.rotations, atol=1e-12
        )
        np.testing.assert_allclose(
            back.root_translations, self.anim.root_translations, atol=1e-12
        )
        self.assertEqual(back.interp, self.anim.interp)
        self.assertEqual(back.quat_order, self.anim.quat_order)
        self.assertEqual(back.joint_index(), self.anim.joint_index())

    def test_binary_roundtrip_and_byte_stability(self):
        spec = BinarySpec(endianness="<", index_width=2, quat_order="xyzw")
        path = os.path.join(self.tmp, "a.btrj")
        dump_binary(self.anim, path, spec)
        back = decode_binary(
            open(path, "rb").read(),
            self.anim.joint_names,
            interp=self.anim.interp,
        )
        np.testing.assert_allclose(
            back.rotations, self.anim.rotations, atol=2e-6
        )
        np.testing.assert_allclose(
            back.root_translations, self.anim.root_translations, atol=2e-6
        )
        self.assertEqual(encode_binary(back, spec), open(path, "rb").read())

    def test_big_endian_roundtrip(self):
        spec = BinarySpec(endianness=">", index_width=4, quat_order="wxyz")
        blob = encode_binary(self.anim, spec)
        back = decode_binary(blob, self.anim.joint_names, interp="slerp")
        np.testing.assert_allclose(
            back.rotations, self.anim.rotations, atol=2e-6
        )

    def test_index_overflow(self):
        spec = BinarySpec(index_width=1)
        big = self.anim
        names = tuple(f"j{i}" for i in range(300))
        rotations = np.zeros((big.frame_count, 300, 4))
        rotations[..., 0] = 1.0
        from animpipe.animation import Animation

        big_anim = Animation(
            joint_names=names,
            times=big.times.copy(),
            rotations=rotations,
            root_translations=np.zeros((big.frame_count, 3)),
        )
        with self.assertRaises(E_INDEX_OVERFLOW) as caught:
            encode_binary(big_anim, spec)
        self.assertEqual(caught.exception.context["index_width"], 1)

    def test_migration_idempotent_and_unsupported(self):
        legacy = build_legacy_document(self.anim)
        once = migrate_schema(legacy)
        twice = migrate_schema(once)
        self.assertEqual(once["schema_version"], CURRENT_SCHEMA)
        self.assertEqual(migration_hash(once), migration_hash(twice))
        self.assertIn("joint_index", once)
        self.assertIn("quat_order", once)
        self.assertIn("interp", once)
        for frame in once["frames"]:
            self.assertIn("root_translation", frame)
            self.assertIn("rotations", frame)
        loaded = load_json_animation  # ensure symbol stays public
        self.assertTrue(callable(loaded))

        bad = dict(legacy)
        bad["schema_version"] = "0.1"
        with self.assertRaises(E_SCHEMA_VERSION) as caught:
            migrate_schema(bad)
        self.assertEqual(caught.exception.context["field"], "schema_version")


if __name__ == "__main__":
    unittest.main()
