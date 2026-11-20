from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from synthml.classifiers import TrainingInstance, decide_labels, train_model
from synthml.config import feature_dimension, load_config, load_plan, parse_config
from synthml.errors import ConfigValidationError, PlanValidationError
from synthml.generate import generate_dataset
from synthml.imaging import decode_png, extract_features
from synthml.util import stable_hash


class CoreUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="synthml-core-")
        cls.root = Path(cls.temp.name)
        cls.samples = cls.root / "samples"
        cls.config_path = cls.root / "experiment.yaml"
        cls.plan_path = cls.root / "plan.json"
        generate_dataset(
            cls.samples,
            cls.config_path,
            cls.plan_path,
            sample_count=12,
            seed=12345,
            concurrency=2,
            snapshot_interval=100,
        )
        cls.config = load_config(cls.config_path)
        cls.raw_config = yaml.safe_load(cls.config_path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_feature_determinism_and_true_dimension_changes(self) -> None:
        png = sorted(self.samples.glob("*.png"))[0]
        decoded = decode_png(png.read_bytes(), self.config.resolution)
        array = decoded["array"]
        parameter_sets = []
        for strategy in (
            {"feature_method": "color", "color_bins": 8, "hog_bins": 9},
            {"feature_method": "color", "color_bins": 16, "hog_bins": 9},
            {"feature_method": "hog", "color_bins": 8, "hog_bins": 9},
            {"feature_method": "concat", "color_bins": 8, "hog_bins": 9},
        ):
            parameter_sets.append(self.config.feature_params(strategy))
        outputs = []
        for params in parameter_sets:
            first, metadata = extract_features(array, params)
            second, _ = extract_features(array, params)
            self.assertTrue(np.array_equal(first, second))
            self.assertEqual(first.shape[0], params["feature_dimension"])
            self.assertEqual(metadata["feature_method"], params["feature_method"])
            self.assertEqual(metadata["quantization_levels"], 4096)
            self.assertEqual(metadata["rounding_decimals"], 8)
            outputs.append(first)
        self.assertEqual([len(vector) for vector in outputs], [24, 48, 36, 60])
        self.assertFalse(np.array_equal(outputs[0], outputs[1]))
        fast = self.config.feature_params(parameter_sets[0], precision="fast")
        fast_vector, fast_metadata = extract_features(array, fast)
        self.assertEqual(fast_metadata["quantization_levels"], 1024)
        self.assertEqual(fast_metadata["rounding_decimals"], 6)
        self.assertFalse(np.array_equal(outputs[0], fast_vector))

    def test_all_classifiers_are_deterministic_and_bounded(self) -> None:
        labels = ["circle", "square"]
        instances = [
            TrainingInstance("a", np.array([0.0, 0.2]), ("circle",)),
            TrainingInstance("b", np.array([0.1, 0.0]), ("circle",)),
            TrainingInstance("c", np.array([4.8, 5.1]), ("square",)),
            TrainingInstance("d", np.array([5.0, 4.9]), ("square",)),
            TrainingInstance("e", np.array([2.4, 2.6]), ("circle", "square")),
        ]
        for classifier in ("logistic_regression", "nearest_centroid", "knn"):
            params = self.config.classifier_params(classifier)
            first = train_model(classifier, params, labels, instances)
            second = train_model(classifier, params, labels, instances)
            self.assertEqual(stable_hash(first), stable_hash(second))
            from synthml.classifiers import ClassifierModel

            model = ClassifierModel(first, self.config.evaluation["probability_decimals"])
            scored = model.score("query", np.array([0.2, 0.1]))
            self.assertEqual(sorted(scored["probabilities"]), labels)
            for value in scored["probabilities"].values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            for values in scored["raw_scores"].values():
                self.assertTrue(np.isfinite(values))
            if classifier == "logistic_regression":
                self.assertEqual(
                    len(first["loss_history"]["circle"]),
                    self.config.classifier_params(classifier)["epochs"],
                )
            if classifier == "knn":
                self.assertEqual(first["effective_k"], 3)

    def test_boundary_equality_is_positive_and_lexicographic(self) -> None:
        result = decide_labels(
            {"square": 0.5, "circle": 0.5, "bar": 0.2},
            threshold=0.5,
            boundary_policy="equality_positive_labels_lexicographic",
        )
        self.assertEqual(result["predicted_labels"], ["circle", "square"])
        self.assertEqual(result["boundary_labels"], ["circle", "square"])
        self.assertTrue(result["boundary"])
        self.assertEqual(result["decisions"]["circle"]["lexicographic_rank"], 1)
        self.assertEqual(result["decisions"]["square"]["lexicographic_rank"], 2)
        self.assertEqual(result["decisions"]["bar"]["comparison"], "less_than_threshold")

    def test_configuration_error_codes(self) -> None:
        cases = []
        unknown = copy.deepcopy(self.raw_config)
        unknown["illegal_key"] = True
        cases.append(unknown)
        empty_labels = copy.deepcopy(self.raw_config)
        empty_labels["labels"] = []
        cases.append(empty_labels)
        wrong_dimension = copy.deepcopy(self.raw_config)
        wrong_dimension["features"]["expected_dimensions"][0]["dimension"] += 1
        cases.append(wrong_dimension)
        bad_concurrency = copy.deepcopy(self.raw_config)
        bad_concurrency["concurrency"] = 0
        cases.append(bad_concurrency)
        messages = []
        for raw in cases:
            with self.assertRaises(ConfigValidationError) as context:
                parse_config(raw, self.config_path)
            self.assertEqual(context.exception.code, "E_CONFIG_INVALID")
            messages.append(context.exception.message)
        self.assertTrue(any("illegal configuration key" in message for message in messages))
        self.assertTrue(any("labels must be a non-empty list" in message for message in messages))
        self.assertTrue(any("feature dimension mismatch" in message for message in messages))
        self.assertTrue(any("concurrency must be a positive integer" in message for message in messages))

    def test_non_monotonic_scale_plan_is_rejected(self) -> None:
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["tiers"][0]["max_samples"] = 30
        plan["tiers"][1]["max_samples"] = 20
        broken = self.root / "broken_plan.json"
        broken.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaises(PlanValidationError) as context:
            load_plan(broken, self.config.labels)
        self.assertEqual(context.exception.code, "E_CONFIG_INVALID")
        self.assertTrue(any("not monotonic" in error for error in context.exception.errors))

    def test_dimension_equation(self) -> None:
        self.assertEqual(feature_dimension("color", 8, 9, (2, 2)), 24)
        self.assertEqual(feature_dimension("hog", 8, 9, (2, 2)), 36)
        self.assertEqual(feature_dimension("concat", 8, 9, (2, 2)), 60)


if __name__ == "__main__":
    unittest.main()
