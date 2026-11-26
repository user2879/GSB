from __future__ import annotations

import concurrent.futures
import unittest

from gsb.params import (
    compute_batch_summary,
    get_parameter_set,
    quantization_error_bound,
)
from tests.helpers import make_shard


class DeterministicParamsTests(unittest.TestCase):
    def test_parameter_digest_is_stable(self) -> None:
        shard = make_shard()
        first = get_parameter_set(shard, 2)
        second = get_parameter_set(shard, 2)
        changed_version = get_parameter_set(shard, 1)
        self.assertEqual(first.digest, second.digest)
        self.assertNotEqual(first.digest, changed_version.digest)

    def test_sample_and_batch_order_do_not_change_digest(self) -> None:
        shard = make_shard()
        forward = compute_batch_summary(shard, "b", [1, 2, 3], 2)
        reverse = compute_batch_summary(shard, "b", [3, 1, 2], 2)
        self.assertEqual(forward, reverse)

    def test_concurrent_batch_dispatch_is_bit_identical(self) -> None:
        shard = make_shard()
        expected = compute_batch_summary(shard, "b-9", [9, 4, 7], 2)

        def run(_: int):
            return compute_batch_summary(shard, "b-9", [7, 9, 4], 2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, range(32)))
        self.assertTrue(all(result == expected for result in results))

    def test_scale_change_changes_digest_and_error_bound(self) -> None:
        shard = make_shard(scale=1000)
        changed = make_shard(scale=4000)
        self.assertNotEqual(
            get_parameter_set(shard, 2).digest,
            get_parameter_set(changed, 2).digest,
        )
        self.assertNotEqual(
            compute_batch_summary(shard, "b", [1, 2], 2)["loss_digest"],
            compute_batch_summary(changed, "b", [1, 2], 2)["loss_digest"],
        )
        coarse = make_shard(scale=100)
        small_bound = quantization_error_bound(coarse, [1, 2], 2)
        large_bound = quantization_error_bound(changed, [1, 2], 2)
        self.assertGreater(small_bound.loss_bound, 0)
        self.assertGreater(small_bound.loss_bound, large_bound.loss_bound)
        self.assertLessEqual(small_bound.per_layer["head"], large_bound.per_layer["head"])


if __name__ == "__main__":
    unittest.main()
