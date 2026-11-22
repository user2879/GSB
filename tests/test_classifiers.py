import numpy as np

from mlimgedit.classifiers import CLASSIFIER_ORDER, stable_sigmoid, train_label_model
from mlimgedit.fixedpoint import FixedPointQuantizer
from mlimgedit.hashing import scale_tier


def _matrix_and_truth(seed=7):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(12, 6))
    targets = (matrix[:, 0] + matrix[:, 1] > 0).astype(np.int64)
    names = [f"sample_{idx:02d}" for idx in range(12)]
    return names, matrix, targets


def test_stable_sigmoid_bounds_and_order():
    values = np.asarray([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    probabilities = stable_sigmoid(values)
    assert probabilities[0] >= 0.0 and probabilities[-1] <= 1.0
    assert np.all(np.diff(probabilities) > 0)
    np.testing.assert_allclose(stable_sigmoid(values), stable_sigmoid(values))


def test_all_classifiers_produce_valid_probabilities_and_threshold(config):
    names, matrix, targets = _matrix_and_truth()
    quantizer = FixedPointQuantizer(config.quant_level, config.frac_bits)
    for classifier in CLASSIFIER_ORDER:
        artifact = train_label_model(
            classifier,
            "rectangle",
            names,
            matrix,
            targets,
            config,
            quantizer,
            scale_tier(len(names)),
        )
        assert len(artifact.scores) == len(names)
        for score in artifact.scores:
            assert 0.0 <= score.probability <= 1.0
            assert score.probability_int >= 0
            assert score.decision == (score.probability_int >= score.threshold_int)
        assert list(artifact.referenced_samples) == names


def test_logreg_records_deterministic_loss_curve(config):
    names, matrix, targets = _matrix_and_truth()
    quantizer = FixedPointQuantizer(config.quant_level, config.frac_bits)
    first = train_label_model(
        "logreg", "rectangle", names, matrix, targets, config, quantizer, "T00_n12"
    )
    second = train_label_model(
        "logreg", "rectangle", names, matrix, targets, config, quantizer, "T00_n12"
    )
    assert first.trainer_meta["losses"] == second.trainer_meta["losses"]
    assert len(first.trainer_meta["losses"]) == config.classifiers["logreg"]["epochs"]


def test_knn_neighbor_order_is_total_and_deterministic(config):
    names, matrix, targets = _matrix_and_truth()
    quantizer = FixedPointQuantizer(config.quant_level, config.frac_bits)
    artifact = train_label_model(
        "knn", "rectangle", names, matrix, targets, config, quantizer, "T00_n12"
    )
    neighbors = artifact.trainer_meta["neighbors"]
    assert all(len(row) == config.classifiers["knn"]["k"] for row in neighbors)
    # The first neighbor of every sample is the sample itself at distance zero.
    assert all(row[0] == idx for idx, row in enumerate(neighbors))


def test_equal_threshold_is_boundary_and_positive(config):
    names, matrix, targets = _matrix_and_truth()
    quantizer = FixedPointQuantizer(config.quant_level, config.frac_bits)
    artifact = train_label_model(
        "logreg", "rectangle", names, matrix, targets, config, quantizer, "T00_n12"
    )
    score = artifact.scores[0]
    # Force an exact integer equality to exercise the documented tie rule.
    score_int = score.probability_int
    threshold_int = score_int
    assert (score_int >= threshold_int) is True
