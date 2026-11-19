from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dp_grad_sim.config import load_config
from dp_grad_sim.data import load_shards
from dp_grad_sim.model import (
    derive_worker_gradient,
    derive_window_gradients,
    initialize_parameters,
    sanitize_nonfinite,
)
from dp_grad_sim.samplegen import generate_sample


class ModelDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        generate_sample(self.root / "sample", workers=3, steps=2, layers=2, hidden_dim=8)
        self.config = load_config(self.root / "sample" / "config.yaml")
        self.shards = load_shards(self.root / "sample" / "shards.jsonl", self.config.workers)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_parameter_initialization_is_deterministic(self) -> None:
        first = initialize_parameters(self.config)
        second = initialize_parameters(self.config)
        self.assertEqual(sorted(first), sorted(second))
        for name in first:
            self.assertTrue(np.array_equal(first[name], second[name]))
            self.assertEqual(first[name].dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(first[name])))

    def test_pseudo_gradient_is_deterministic_and_step_dependent(self) -> None:
        first = derive_worker_gradient(self.config, self.shards.by_worker[1], 1, 0)
        second = derive_worker_gradient(self.config, self.shards.by_worker[1], 1, 0)
        next_step = derive_worker_gradient(self.config, self.shards.by_worker[1], 1, 1)
        for name in first:
            self.assertTrue(np.array_equal(first[name], second[name]))
        self.assertTrue(any(not np.array_equal(first[name], next_step[name]) for name in first))

    def test_window_gradient_is_deterministic(self) -> None:
        first = derive_window_gradients(self.config, self.shards, 0, 2)
        second = derive_window_gradients(self.config, self.shards, 0, 2)
        for worker in first:
            for name in first[worker]:
                self.assertTrue(np.array_equal(first[worker][name], second[worker][name]))

    def test_nonfinite_is_reported_with_worker_and_tensor(self) -> None:
        gradients = derive_window_gradients(self.config, self.shards, 0, 1)
        tensor_name = "layer_0.weight"
        gradients[1][tensor_name][0, 0] = np.nan
        cleaned, failures = sanitize_nonfinite(gradients)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["code"], "E_PARAM_NONFINITE")
        self.assertEqual(failures[0]["worker_id"], 1)
        self.assertEqual(failures[0]["tensor"], tensor_name)
        self.assertTrue(np.all(np.isfinite(cleaned[1][tensor_name])))

    def test_shard_validation_rejects_missing_worker(self) -> None:
        path = self.root / "partial.jsonl"
        path.write_text(
            '{"worker_id":0,"shard_id":"a","samples":1,"feature_digest":"f","label_digest":"l"}\n',
            encoding="utf-8",
        )
        with self.assertRaises(Exception) as context:
            load_shards(path, 3)
        self.assertEqual(getattr(context.exception, "code", None), "E_DATA_INVALID")


if __name__ == "__main__":
    unittest.main()

