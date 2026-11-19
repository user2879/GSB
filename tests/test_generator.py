from __future__ import annotations

import hashlib
import shutil
import unittest

from forensic_workbench.avi import AVIVideoReader
from forensic_workbench.config import AnalysisWindow, ScopedRect
from forensic_workbench.errors import ForensicError
from forensic_workbench.features import SelectionResolver, compute_frame_payload
from forensic_workbench.source import SourceSpec, generate_source
from forensic_workbench.util import iter_jsonl, sha256_file
from tests.support import TemporaryWorkspace


class GeneratorTests(unittest.TestCase):
    def test_two_generations_are_byte_identical_and_truth_is_complete(self):
        spec = SourceSpec(
            width=160, height=90, fps=30, duration_seconds=2, seed=7, gop_size=10
        )
        with TemporaryWorkspace() as workspace:
            source_a = generate_source(workspace / "a", spec)
            source_b = generate_source(workspace / "b", spec)
            for filename in ("source.avi", "ground_truth.jsonl", "source_manifest.json"):
                self.assertEqual(
                    sha256_file(workspace / "a" / filename),
                    sha256_file(workspace / "b" / filename),
                    filename,
                )
            self.assertEqual(source_a.frame_count, 60)
            self.assertEqual(source_a.gop_count, 6)
            self.assertEqual(source_a.manifest_digest, source_b.manifest_digest)
            self.assertEqual(source_a.video_sha256, source_b.video_sha256)
            self.assertEqual(source_a.truth_sha256, source_b.truth_sha256)
            records = list(iter_jsonl(source_a.truth_path))
            self.assertEqual(len(records), 60)
            self.assertTrue(records[0]["shot_boundary"])
            self.assertTrue(any(record["shot_boundary"] for record in records[1:]))
            self.assertTrue(any(record["gop_boundary"] for record in records[1:]))
            self.assertTrue(all("watermark" in record for record in records))
            self.assertTrue(all(len(record["blocks"]) == 3 for record in records))
            digest = hashlib.sha256()
            with AVIVideoReader(source_a.video_path) as reader:
                self.assertEqual(reader.frame_count, 60)
                for _, frame in reader.iter_frames():
                    digest.update(frame.tobytes(order="C"))
            self.assertEqual(digest.hexdigest(), source_a.content_sha256)

    def test_same_frame_bytes_are_stable(self):
        with TemporaryWorkspace() as workspace:
            source = generate_source(workspace / "source")
            with AVIVideoReader(source.video_path) as reader:
                first = reader.read_frame(17)
                second = reader.read_frame(17)
            self.assertEqual(first.tobytes(), second.tobytes())
            self.assertEqual(
                hashlib.sha256(first.tobytes()).hexdigest(),
                hashlib.sha256(second.tobytes()).hexdigest(),
            )

    def test_frame_feature_hash_is_recomputable(self):
        with TemporaryWorkspace() as workspace:
            source = generate_source(workspace / "source")
            config = source.default_pipeline_config()
            resolver = SelectionResolver(
                source.spec.width, source.spec.height, config
            )
            selection = resolver.for_frame(17)
            with AVIVideoReader(source.video_path) as reader:
                frame = reader.read_frame(17)
            first = compute_frame_payload(
                frame_index=17,
                frame=frame,
                selection=selection,
                shot_boundaries=source.shot_boundaries,
                gop_size=config.gop_size,
                feature_version=config.feature_version,
            )
            second = compute_frame_payload(
                frame_index=17,
                frame=frame,
                selection=selection,
                shot_boundaries=source.shot_boundaries,
                gop_size=config.gop_size,
                feature_version=config.feature_version,
            )
            self.assertEqual(first["content_hash"], second["content_hash"])
            self.assertEqual(first["keyframe_signature"], second["keyframe_signature"])
            self.assertEqual(first["tile_hashes"], second["tile_hashes"])
            self.assertEqual(first["excluded_pixels"], second["excluded_pixels"])

    def test_seed_changes_generated_content(self):
        with TemporaryWorkspace() as workspace:
            base = SourceSpec(
                width=160, height=90, fps=30, duration_seconds=1, seed=7, gop_size=10
            )
            changed = SourceSpec(
                width=160, height=90, fps=30, duration_seconds=1, seed=8, gop_size=10
            )
            first = generate_source(workspace / "first", base)
            second = generate_source(workspace / "second", changed)
            self.assertNotEqual(first.video_sha256, second.video_sha256)
            self.assertNotEqual(first.content_sha256, second.content_sha256)
            self.assertNotEqual(first.truth_sha256, second.truth_sha256)

    def test_corrupt_frame_format_returns_business_code(self):
        with TemporaryWorkspace() as workspace:
            source = generate_source(workspace / "source")
            corrupt = workspace / "corrupt.avi"
            shutil.copyfile(source.video_path, corrupt)
            with corrupt.open("r+b") as handle:
                handle.write(b"NOPE")
            with self.assertRaises(ForensicError) as raised:
                AVIVideoReader(corrupt)
            self.assertEqual(raised.exception.code, "E_FRAME_FORMAT")
            self.assertIn("path", raised.exception.context)

    def test_region_bounds_and_mask_errors_are_distinct(self):
        with TemporaryWorkspace() as workspace:
            source = generate_source(workspace / "source")
            base = source.default_pipeline_config()
            bad_sampling = base.with_regions(
                sampling=[ScopedRect(-1, 0, 10, 10, 0, 5)]
            )
            with self.assertRaises(ForensicError) as raised:
                bad_sampling.validate(
                    source.spec.width, source.spec.height, source.frame_count
                )
            self.assertEqual(raised.exception.code, "E_REGION_BOUNDS")
            bad_mask = base.with_regions(mask=[ScopedRect(0, 0, 0, 10, 0, 5)])
            with self.assertRaises(ForensicError) as raised:
                bad_mask.validate(
                    source.spec.width, source.spec.height, source.frame_count
                )
            self.assertEqual(raised.exception.code, "E_MASK_INVALID")
            out_of_bounds_mask = base.with_regions(
                mask=[ScopedRect(source.spec.width - 2, 0, 10, 10, 0, 5)]
            )
            with self.assertRaises(ForensicError) as raised:
                out_of_bounds_mask.validate(
                    source.spec.width, source.spec.height, source.frame_count
                )
            self.assertEqual(raised.exception.code, "E_MASK_INVALID")
            bad_window = base.with_regions(window=AnalysisWindow(50, 10))
            with self.assertRaises(ForensicError) as raised:
                bad_window.validate(
                    source.spec.width, source.spec.height, source.frame_count
                )
            self.assertEqual(raised.exception.code, "E_REGION_BOUNDS")


if __name__ == "__main__":
    unittest.main()
