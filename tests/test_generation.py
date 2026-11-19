from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tactical_path.mapgen import generate_bundle, save_bundle

from .helpers import bundle, config


class GenerationTests(unittest.TestCase):
    def test_same_seed_and_tier_are_byte_identical(self) -> None:
        cfg = config()
        spec = cfg.tiers["small"]
        first = generate_bundle(spec, seed=cfg.seed, mobility_rules=cfg.mobility)
        second = generate_bundle(spec, seed=cfg.seed, mobility_rules=cfg.mobility)
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left"
            right = Path(directory) / "right"
            save_bundle(first, left)
            save_bundle(second, right)
            left_files = {
                path.name: path.read_bytes() for path in sorted(left.iterdir())
            }
            right_files = {
                path.name: path.read_bytes() for path in sorted(right.iterdir())
            }
            self.assertEqual(sorted(left_files), sorted(right_files))
            for name in left_files:
                self.assertEqual(left_files[name], right_files[name], name)

    def test_metadata_counts_and_hash(self) -> None:
        loaded = bundle("small")
        meta = loaded.metadata
        self.assertEqual(meta["region_count"], 16)
        self.assertEqual(meta["unit_count"], len(loaded.units))
        self.assertEqual(meta["query_count"], len(loaded.queries))
        self.assertEqual(len(meta["map_content_hash"]), 64)
        self.assertEqual(loaded.map_data.content_hash, meta["map_content_hash"])

