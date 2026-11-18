from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import shutil
import tempfile
import unittest

from pagecache.aggregate import FIXED_SCALE, FixedPointAggregator
from pagecache.cache import PageCache
from pagecache.errors import EngineError, ErrorCode
from pagecache.manifest import read_manifest, write_manifest
from pagecache.pagefile import read_page_file
from pagecache.segment import read_segment_file
from tests.helpers import copy_overflow, make_samples, run_pipeline, version_paths


class CacheAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.data, cls.meta = make_samples(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_deterministic_spill_sequence_and_swap_counts(self) -> None:
        first_dir = Path(self.temp.name) / "deterministic-a"
        second_dir = Path(self.temp.name) / "deterministic-b"
        budget = int(self.meta["small_budget_bytes"])
        first_agg, first_cache, first = run_pipeline(data=self.data, meta=self.meta, overflow_dir=first_dir, budget_bytes=budget)
        second_agg, second_cache, second = run_pipeline(data=self.data, meta=self.meta, overflow_dir=second_dir, budget_bytes=budget)
        first_state = first_cache.state()
        second_state = second_cache.state()
        self.assertEqual(first_state, second_state)
        self.assertEqual(first["trace"], second["trace"])
        self.assertEqual(first_agg.output_bytes(), second_agg.output_bytes())
        self.assertGreater(first_state["swap_in_count"], 0)
        self.assertEqual(first_state["overflow_sequence"][0], "p0005")
        self.assertEqual(first_state["capacity_frames"], 3)
        inventory_a = [(path.name, path.read_bytes()) for path in sorted(first_dir.glob("*.seg"))]
        inventory_b = [(path.name, path.read_bytes()) for path in sorted(second_dir.glob("*.seg"))]
        self.assertEqual(inventory_a, inventory_b)

    def test_full_resident_and_small_budget_have_identical_output_and_bounds(self) -> None:
        full_dir = Path(self.temp.name) / "full-resident"
        small_dir = Path(self.temp.name) / "small-budget"
        full_agg, full_cache, _ = run_pipeline(
            data=self.data,
            meta=self.meta,
            overflow_dir=full_dir,
            budget_bytes=int(self.meta["full_budget_bytes"]),
        )
        small_agg, small_cache, _ = run_pipeline(
            data=self.data,
            meta=self.meta,
            overflow_dir=small_dir,
            budget_bytes=int(self.meta["small_budget_bytes"]),
        )
        self.assertEqual(full_agg.output_bytes(), small_agg.output_bytes())
        self.assertEqual(full_cache.state()["overflow_sequence"], [])
        self.assertGreater(len(small_cache.state()["overflow_sequence"]), 0)

        true_groups = self._true_group_means()
        output = small_agg.output()
        for group in output["groups"]:
            key = int(group["key"])
            true_sum, true_count = true_groups[key]
            self.assertEqual(int(group["count"]), true_count)
            reported_sum = Fraction(int(group["sum_scaled"]), FIXED_SCALE)
            reported_mean = Fraction(int(group["mean_scaled"]), FIXED_SCALE)
            self.assertLessEqual(abs(reported_sum - true_sum), Fraction(true_count, 2 * FIXED_SCALE))
            self.assertLessEqual(abs(reported_mean - true_sum / true_count), Fraction(3, 2 * FIXED_SCALE))
        self.assertEqual(
            output["fixed_point"]["mean_abs_error_upper_bound"],
            {"numerator": 3, "denominator": 2 * FIXED_SCALE},
        )

    def test_compressed_and_uncompressed_swap_in_return_identical_bytes(self) -> None:
        manifest_path, pages_dir, sample_overflow = version_paths(self.data, 3)
        compressed_dir = Path(self.temp.name) / "compressed-cache"
        copy_overflow(sample_overflow, compressed_dir)
        compressed_cache = PageCache(
            manifest=read_manifest(manifest_path),
            pages_dir=pages_dir,
            overflow_dir=compressed_dir,
            budget_bytes=1536,
            page_size=int(self.meta["page_size"]),
        )
        compressed_page = compressed_cache.get_page("p0001")

        plain_dir = Path(self.temp.name) / "plain-cache"
        plain_dir.mkdir(parents=True)
        plain_segment_source = version_paths(self.data, 2)[2] / "p0001.seg"
        shutil.copy2(plain_segment_source, plain_dir / "p0001.seg")
        entries = []
        for entry in read_manifest(manifest_path).records:
            if entry.page_id == "p0001":
                entries.append(replace(entry, compressed=False, overflow_ref="p0001.seg"))
            else:
                entries.append(entry)
        plain_manifest = Path(self.temp.name) / "plain.manifest"
        write_manifest(plain_manifest, entries, format_version=3)
        plain_cache = PageCache(
            manifest=read_manifest(plain_manifest),
            pages_dir=pages_dir,
            overflow_dir=plain_dir,
            budget_bytes=1536,
            page_size=int(self.meta["page_size"]),
        )
        plain_page = plain_cache.get_page("p0001")
        expected = read_page_file(pages_dir / "p0001.bin")
        self.assertEqual(compressed_page.raw_bytes, plain_page.raw_bytes)
        self.assertEqual(compressed_page.raw_bytes, expected.raw_bytes)
        self.assertEqual(read_segment_file(plain_dir / "p0001.seg").compressed, False)

    def test_manifest_compression_mismatch_is_e_payload_with_first_page(self) -> None:
        manifest_path, pages_dir, sample_overflow = version_paths(self.data, 3)
        overflow_dir = Path(self.temp.name) / "mismatch-cache"
        copy_overflow(sample_overflow, overflow_dir)
        entries = [replace(entry, compressed=False) if entry.page_id == "p0001" else entry for entry in read_manifest(manifest_path).records]
        mismatch_manifest = Path(self.temp.name) / "mismatch.manifest"
        write_manifest(mismatch_manifest, entries, format_version=3)
        cache = PageCache(
            manifest=read_manifest(mismatch_manifest),
            pages_dir=pages_dir,
            overflow_dir=overflow_dir,
            budget_bytes=1536,
            page_size=int(self.meta["page_size"]),
        )
        with self.assertRaises(EngineError) as caught:
            cache.get_page("p0001")
        self.assertIs(caught.exception.code, ErrorCode.E_PAYLOAD)
        self.assertEqual(caught.exception.page_id, "p0001")

    def test_zero_budget_is_e_budget(self) -> None:
        manifest_path, pages_dir, overflow = version_paths(self.data, 3)
        with self.assertRaises(EngineError) as caught:
            PageCache(
                manifest=read_manifest(manifest_path),
                pages_dir=pages_dir,
                overflow_dir=overflow,
                budget_bytes=0,
                page_size=int(self.meta["page_size"]),
            )
        self.assertIs(caught.exception.code, ErrorCode.E_BUDGET)

    def _true_group_means(self) -> dict[int, tuple[Fraction, int]]:
        _manifest, pages_dir, _overflow = version_paths(self.data, 3)
        sums: dict[int, Fraction] = {}
        counts: dict[int, int] = {}
        for page_id in self.meta["scan_order"]:
            page = read_page_file(pages_dir / f"{page_id}.bin")
            group_column = page.column(self.meta["group_column"])
            value_column = page.column(self.meta["value_column"])
            for group, value in zip(group_column.values, value_column.values, strict=True):
                if group is None or value is None:
                    continue
                key = int(group)
                sums[key] = sums.get(key, Fraction(0)) + Fraction(Decimal(str(value)))
                counts[key] = counts.get(key, 0) + 1
        return {key: (sums[key], counts[key]) for key in sorted(sums)}


if __name__ == "__main__":
    unittest.main()
