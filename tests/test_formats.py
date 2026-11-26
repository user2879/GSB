import unittest

from fxrender.formats import (FrameRecord, NOISE_CODE_TO_SCHEME,
                              parse_v1_line, parse_v2_dict, read_v1, read_v2,
                              record_to_v1_line, record_to_v2_dict,
                              restore_v2_from_sidecar, v1_to_v2, v2_to_v1,
                              write_v1, write_v2, V1_LINE_WIDTH)


def make_record(**overrides):
    base = dict(
        frame_id=3, step=2, tile_row=1, guidance_float=0.7004,
        noise_scheme="gaussian",
        latent_fp=[12, -34, 56, 78], step_delta_fp=[1, -2, 3, -4],
        pixel_digest="a" * 32, step_cost=4, merged=False,
        cache_key="denoise:abc", noise_scale_fp=100,
        condition_scale_fp=200, source_version=2, missing=[])
    base.update(overrides)
    return FrameRecord(**base)


class FormatRoundTripTests(unittest.TestCase):
    def test_v1_fixed_width_contract(self):
        record = v1_to_v2(parse_v1_line(record_to_v1_line(
            FrameRecord.from_canonical(make_record().canonical()))))
        line = record_to_v1_line(record)
        self.assertEqual(len(line), V1_LINE_WIDTH)

    def test_v2_to_v1_to_v2_field_equality(self):
        original = make_record()
        _, line, sidecar = v2_to_v1(original)
        v1_record = parse_v1_line(line)
        restored = restore_v2_from_sidecar(v1_record, sidecar)
        direct = parse_v2_dict(record_to_v2_dict(original))
        self.assertEqual(restored.canonical(), direct.canonical())

    def test_guidance_exact_conversion_and_residual(self):
        record = make_record(guidance_float=0.7004)
        v1_record, _line, sidecar = v2_to_v1(record)
        self.assertEqual(v1_record.guidance_permille, 700)
        self.assertAlmostEqual(sidecar["residuals"]["guidance"], 0.0004)
        self.assertIn("guidance_float", sidecar["overwritten_fields"])
        restored = restore_v2_from_sidecar(v1_record, sidecar)
        self.assertAlmostEqual(restored.guidance_float, 0.7004)

    def test_noise_code_mapping_and_precision_label(self):
        record = make_record(noise_scheme="laplacian")
        v1_record, _line, sidecar = v2_to_v1(record)
        self.assertEqual(v1_record.noise_code,
                         next(k for k, v in NOISE_CODE_TO_SCHEME.items()
                              if v == "laplacian"))
        self.assertTrue(sidecar["precision_label"])
        upgraded = v1_to_v2(v1_record)
        self.assertEqual(upgraded.noise_scheme, "laplacian")

    def test_missing_fields_preserved(self):
        # A genuinely sparse v1 record: guidance and noise code are absent.
        sparse = FrameRecord(frame_id=5, step=1, tile_row=0,
                             pixel_digest=None, step_cost=None,
                             source_version=1,
                             missing=["guidance_permille", "noise_code",
                                      "pixel_digest", "step_cost"])
        line = record_to_v1_line(sparse)
        parsed = parse_v1_line(line)
        self.assertIsNone(parsed.guidance_float)
        self.assertIsNone(parsed.noise_scheme)
        upgraded = v1_to_v2(parsed)
        self.assertIn("guidance_permille", upgraded.missing)
        self.assertIn("noise_code", upgraded.missing)
        # Downgrade then restore keeps the sparse representation intact.
        v1_record, _line, sidecar = v2_to_v1(upgraded)
        restored = restore_v2_from_sidecar(v1_record, sidecar)
        self.assertEqual(restored.canonical(), upgraded.canonical())
        self.assertIsNone(restored.guidance_float)
        self.assertIsNone(restored.noise_scheme)

    def test_file_roundtrip(self, tmp_path=None):
        import tempfile
        from pathlib import Path
        work = Path(tempfile.mkdtemp())
        records = [make_record(frame_id=i) for i in range(3)]
        write_v1(work / "a.txt", records)
        write_v2(work / "a.jsonl", records)
        self.assertEqual([r.frame_id for r in read_v1(work / "a.txt")],
                         [0, 1, 2])
        self.assertEqual([r.frame_id for r in read_v2(work / "a.jsonl")],
                         [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
