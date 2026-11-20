"""业务要求 2：确定性分块特征、固定投影、定点量化与截断记录。"""

from __future__ import annotations

import numpy as np

from vrep.config import Config
from vrep.features import (
    aggregate_embedding,
    chunk_cache_key,
    enumerate_tiles,
    extract_chunk,
    integer_bounds,
    projection_matrix,
    quantize,
    quantize_matrix,
    round_half_up,
)
from vrep.hashing import content_fingerprint, feature_config_fingerprint
from vrep.loader import scan_library

from ._base import SampleTestBase


class FeatureDeterminismTests(SampleTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.records, _, _ = scan_library(self.samples)
        self.config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        self.fp = feature_config_fingerprint(self.config)

    def test_tile_enumeration_row_major_and_truncation(self) -> None:
        tiles, truncated = enumerate_tiles(
            self.IMAGE_SIZE, self.IMAGE_SIZE, self.TILE_SIZE, max_tiles=10
        )
        self.assertEqual(len(tiles), 10)
        self.assertEqual([tile.seq for tile in tiles], list(range(10)))
        self.assertEqual(truncated, list(range(10, 16)))
        self.assertTrue(all(tiles[i].seq < tiles[i + 1].seq for i in range(9)))

    def test_projection_matrix_independent_of_global_rng(self) -> None:
        np.random.seed(12345)
        first = projection_matrix(self.config.feature_dim, self.fp)
        np.random.seed(99999)
        second = projection_matrix(self.config.feature_dim, self.fp)
        np.testing.assert_array_equal(first, second)

    def test_repeated_extraction_elementwise_identical(self) -> None:
        record = self.records[0]
        project = projection_matrix(self.config.feature_dim, self.fp)
        tiles, _ = enumerate_tiles(
            self.IMAGE_SIZE, self.IMAGE_SIZE,
            self.TILE_SIZE, self.config.max_tiles_per_image,
        )
        results = []
        for _ in range(3):
            chunks = [
                extract_chunk(
                    record.rgb, tile, self.TILE_SIZE, project,
                    self.config.quant_level, self.config.fixed_bits,
                )
                for tile in tiles
            ]
            results.append(chunks)
        for chunk_a, chunk_b in zip(results[0], results[1]):
            self.assertEqual(chunk_a, chunk_b)
        for chunk_a, chunk_c in zip(results[0], results[2]):
            self.assertEqual(chunk_a["vector"], chunk_c["vector"])
            self.assertEqual(chunk_a["residual"], chunk_c["residual"])

    def test_cache_key_stable_for_same_content_and_config(self) -> None:
        record = self.records[0]
        content_fp = content_fingerprint(record.rgb)
        tiles, _ = enumerate_tiles(
            self.IMAGE_SIZE, self.IMAGE_SIZE,
            self.TILE_SIZE, self.config.max_tiles_per_image,
        )
        key_a = chunk_cache_key(
            content_fp, self.fp, self.config.quant_level, tiles[0]
        )
        key_b = chunk_cache_key(
            content_fp, self.fp, self.config.quant_level, tiles[0]
        )
        self.assertEqual(key_a, key_b)

    def test_quantization_residual_and_bounds(self) -> None:
        values = np.array([0.123456, -0.777, 1.999, -1.999], dtype=np.float64)
        vector, residual = quantize(
            values, self.config.quant_level, self.config.fixed_bits
        )
        lower, upper = integer_bounds(self.config.quant_level)
        self.assertTrue(np.all(vector >= lower))
        self.assertTrue(np.all(vector <= upper))
        self.assertEqual(len(vector), len(values))
        self.assertEqual(len(residual), len(values))
        # 残差落在 [-0.5, 0.5) 半向上取整区间
        self.assertTrue(np.all(residual >= -0.5))
        self.assertTrue(np.all(residual < 0.5))

    def test_round_half_up_is_deterministic_for_negatives(self) -> None:
        np.testing.assert_array_equal(
            round_half_up(np.array([0.5, -0.5, 1.5, -1.5, 2.5])),
            np.array([1, 0, 2, -1, 3]),
        )

    def test_quantized_matrix_is_integer_and_bounded(self) -> None:
        matrix = projection_matrix(self.config.feature_dim, self.fp)
        integer_matrix = quantize_matrix(
            matrix, self.config.quant_level, self.config.fixed_bits
        )
        self.assertTrue(np.issubdtype(integer_matrix.dtype, np.integer))
        lower, upper = integer_bounds(self.config.quant_level)
        self.assertTrue(np.all(integer_matrix >= lower))
        self.assertTrue(np.all(integer_matrix <= upper))


    def test_tile_truncation_recorded_in_bundle(self) -> None:
        from vrep.engine import Engine
        from vrep.injection import InjectionManager
        import os

        config = Config(
            **{
                **self.config.to_dict(),
                "max_tiles_per_image": 6,
                "point_budget_per_batch": 200,
            }
        )
        engine = Engine(
            os.path.join(self.temp_root, "trunc-engine"), InjectionManager(None)
        )
        engine.seed_library(self.samples)
        engine.run_batch(config, batch_no=1)
        entry = engine.ledger["images"][self.records[0].name]
        # 64px/16px 为 4x4 共 16 块，上限 6 -> 记录 10 个被截断序号
        self.assertEqual(entry["feature"]["truncated_seqs"], list(range(6, 16)))
        self.assertEqual(len(entry["feature"]["chunk_keys"]), 6)

    def test_embedding_aggregation_order_stable(self) -> None:
        rng = np.random.default_rng(7)
        pairs = [
            (seq, rng.integers(-100, 100, size=6).tolist())
            for seq in range(5)
        ]
        matrix = quantize_matrix(
            rng.uniform(-0.5, 0.5, size=(4, 6)),
            self.config.quant_level,
            self.config.fixed_bits,
        )
        first = aggregate_embedding(
            list(pairs), matrix, self.config.fixed_bits, self.config.quant_level
        )
        shuffled = list(reversed(pairs))
        second = aggregate_embedding(
            shuffled, matrix, self.config.fixed_bits, self.config.quant_level
        )
        self.assertEqual(first, second)
