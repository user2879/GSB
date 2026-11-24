import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowquery_diag.codec import (  # noqa: E402
    FrameBuilder,
    decode_and_normalize,
)
from slowquery_diag.config import load_declaration  # noqa: E402
from slowquery_diag.errors import FormatError, SourceDeclarationError  # noqa: E402
from slowquery_diag.models import SOURCE_APP, SOURCE_DB, SOURCE_SLOW  # noqa: E402

EXAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)


class DeclarationValidationTests(unittest.TestCase):
    def test_valid_bundle_loads(self):
        bundle = load_declaration(os.path.join(EXAMPLE_DIR, "declaration.json"))
        self.assertEqual(len(bundle.sources), 6)
        self.assertEqual(len(bundle.candidates), 4)
        self.assertTrue(all(s.format_version >= 1 for s in bundle.sources))

    def _mutated_declaration(self, tmpdir, mutate):
        payload = json.load(
            open(os.path.join(EXAMPLE_DIR, "declaration.json"), encoding="utf-8")
        )
        mutate(payload)
        path = os.path.join(tmpdir, "decl.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_duplicate_source_id_reports_e_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def mutate(payload):
                payload["sources"][1]["source_id"] = payload["sources"][0]["source_id"]
            with self.assertRaises(SourceDeclarationError) as ctx:
                load_declaration(self._mutated_declaration(tmpdir, mutate))
            self.assertEqual(ctx.exception.code, "E_SOURCE")
            self.assertEqual(ctx.exception.field, "source_id")

    def test_negative_integer_reports_e_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def mutate(payload):
                payload["sources"][0]["sample_period_ms"] = -1
            with self.assertRaises(SourceDeclarationError) as ctx:
                load_declaration(self._mutated_declaration(tmpdir, mutate))
            self.assertEqual(ctx.exception.code, "E_SOURCE")
            self.assertEqual(ctx.exception.field, "sample_period_ms")

    def test_unknown_source_type_and_missing_candidate_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def mutate(payload):
                payload["sources"][0]["source_type"] = "telemetry"
            with self.assertRaises(SourceDeclarationError) as ctx:
                load_declaration(self._mutated_declaration(tmpdir, mutate))
            self.assertEqual(ctx.exception.field, "source_type")

            def mutate2(payload):
                for source in payload["sources"]:
                    source["frame_file"] = os.path.join(
                        EXAMPLE_DIR, os.path.basename(source["frame_file"])
                    )
                payload["candidates"][0]["target_sources"] = ["ghost"]
            with self.assertRaises(SourceDeclarationError) as ctx:
                load_declaration(self._mutated_declaration(tmpdir, mutate2))
            self.assertEqual(ctx.exception.code, "E_SOURCE")
            self.assertIn("target_sources", ctx.exception.field)

    def test_bool_is_not_accepted_as_integer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def mutate(payload):
                payload["sources"][0]["sample_rows"] = True
            with self.assertRaises(SourceDeclarationError):
                load_declaration(self._mutated_declaration(tmpdir, mutate))


class CodecNormalizationTests(unittest.TestCase):
    def _source(self, tmpdir, source_type, version, frame):
        from slowquery_diag.models import SourceDeclaration
        path = os.path.join(tmpdir, "frame.bin")
        with open(path, "wb") as handle:
            handle.write(frame)
        return SourceDeclaration(
            source_id="s", source_type=source_type, format_version=version,
            sample_rows=1, latency_caliber="avg", sample_period_ms=1000,
            latency_algebra={}, frame_file=path,
        )

    def test_db_v1_backfills_call_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = FrameBuilder(SOURCE_DB, 1).add(1, (7, 1234, 55)).build()
            source = self._source(tmpdir, SOURCE_DB, 1, frame)
            records = decode_and_normalize(frame, source)
            self.assertEqual(records[0].query_id, 7)
            self.assertEqual(records[0].latency_us, 1234)
            self.assertIn("call_count", records[0].filled_fields)
            self.assertTrue(records[0].fill_reason)

    def test_app_ms_converted_to_us_and_v1_backfills_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = FrameBuilder(SOURCE_APP, 1).add(1, (3, 12, 4)).build()
            source = self._source(tmpdir, SOURCE_APP, 1, frame)
            records = decode_and_normalize(frame, source)
            self.assertEqual(records[0].latency_us, 12_000)
            self.assertEqual(records[0].call_count, 4)
            self.assertEqual(records[0].filled_fields, ("rows_scanned",))

    def test_slow_interval_midpoint_truncates_floor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = FrameBuilder(SOURCE_SLOW, 1).add(
                1, (9, 3, 4, 2)
            ).build()
            source = self._source(tmpdir, SOURCE_SLOW, 1, frame)
            records = decode_and_normalize(frame, source)
            # floor((3+4)/2)=3 ms -> 3000 us
            self.assertEqual(records[0].latency_us, 3000)

    def test_unknown_version_localizes_first_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = bytearray(
                FrameBuilder(SOURCE_DB, 2).add(2, (1, 10, 10, 1)).build()
            )
            frame[13] = 99
            source = self._source(tmpdir, SOURCE_DB, 2, bytes(frame))
            with self.assertRaises(FormatError) as ctx:
                decode_and_normalize(bytes(frame), source)
            self.assertEqual(ctx.exception.code, "E_FORMAT")
            self.assertEqual(ctx.exception.row, 1)

    def test_inconsistent_frame_len_localizes_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = bytearray(
                FrameBuilder(SOURCE_DB, 2).add(2, (1, 10, 10, 1)).build()
            )
            frame[11] = 0x01  # claim 258-byte payload instead of 34
            frame[12] = 0x02
            source = self._source(tmpdir, SOURCE_DB, 2, bytes(frame))
            with self.assertRaises(FormatError) as ctx:
                decode_and_normalize(bytes(frame), source)
            self.assertEqual(ctx.exception.row, 1)


if __name__ == "__main__":
    unittest.main()
