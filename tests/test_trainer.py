from __future__ import annotations

import math
import threading
import unittest

import numpy as np

from tabular_lab.errors import E_MODEL_BUDGET_EXHAUSTED
from tabular_lab.trainer import (
    TrainingFailure,
    TrainingHyperparameters,
    TrainingInterrupted,
    train_model,
)


def synthetic_xy() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(90210)
    x = rng.standard_normal((32, 3))
    signal = 1.2 * x[:, 0] - 0.7 * x[:, 1] + 0.35 * x[:, 2]
    y = (signal + 0.4 * rng.standard_normal(32) > 0).astype(np.float64)
    return x[:24], y[:24], x[24:], y[24:]


def hyperparameters(*, epochs: int = 5, snapshot_every: int = 1) -> TrainingHyperparameters:
    return TrainingHyperparameters(
        epochs=epochs,
        learning_rate=0.05,
        batch_size=4,
        tolerance=0.0,
        l1_zero_subgradient=0.0,
        snapshot_every_batches=snapshot_every,
    )


class TrainerTests(unittest.TestCase):
    def test_all_regularizers_are_explicit_and_deterministic(self) -> None:
        x_train, y_train, x_validation, y_validation = synthetic_xy()
        cases = [("l2", 0.15, None), ("l1", 0.03, None), ("elastic_net", 0.08, 0.6)]
        for scheme, penalty, mix in cases:
            kwargs = dict(
                model_name=f"{scheme}_model",
                scheme=scheme,
                penalty=penalty,
                mix=mix,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                hyperparameters=hyperparameters(),
                quota=20,
            )
            first = train_model(**kwargs)
            second = train_model(**kwargs)
            self.assertEqual(first, second)
            self.assertEqual(first["epochs_consumed"], 5)
            self.assertEqual(len(first["loss_trajectory"]), 5)
            for point in first["loss_trajectory"]:
                self.assertAlmostEqual(
                    point["loss"], point["data_loss"] + point["regularization_loss"], places=14
                )
                if scheme == "l2":
                    expected = penalty * 0.5 * point["l2_squared"]
                elif scheme == "l1":
                    expected = penalty * point["l1_norm"]
                else:
                    expected = penalty * ((1.0 - mix) * 0.5 * point["l2_squared"] + mix * point["l1_norm"])
                self.assertAlmostEqual(point["regularization_loss"], expected, places=14)
            self.assertTrue(all(point["coefficient_digest"] for point in first["loss_trajectory"]))

    def test_quota_exhaustion_is_a_business_failure(self) -> None:
        x_train, y_train, x_validation, y_validation = synthetic_xy()
        with self.assertRaises(TrainingFailure) as context:
            train_model(
                model_name="quota_model",
                scheme="l2",
                penalty=0.1,
                mix=None,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                hyperparameters=hyperparameters(epochs=5),
                quota=2,
            )
        self.assertEqual(context.exception.code, E_MODEL_BUDGET_EXHAUSTED)
        self.assertEqual(context.exception.epoch, 2)

    def test_checkpoint_resume_is_bit_identical_to_uninterrupted_training(self) -> None:
        x_train, y_train, x_validation, y_validation = synthetic_xy()
        hyper = hyperparameters(epochs=6, snapshot_every=1)
        direct = train_model(
            model_name="resume_model",
            scheme="elastic_net",
            penalty=0.07,
            mix=0.4,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            hyperparameters=hyper,
            quota=20,
        )

        stop = threading.Event()
        saved: dict = {}
        stopped_after_first = {"value": False}

        def callback(state: dict, reason: str) -> None:
            saved.clear()
            saved.update(state)
            if reason == "batch" and not stopped_after_first["value"]:
                stopped_after_first["value"] = True
                stop.set()

        with self.assertRaises(TrainingInterrupted):
            train_model(
                model_name="resume_model",
                scheme="elastic_net",
                penalty=0.07,
                mix=0.4,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                hyperparameters=hyper,
                quota=20,
                checkpoint_callback=callback,
                stop_event=stop,
            )
        self.assertTrue(saved)
        self.assertGreaterEqual(saved["completed_epochs"], 0)
        resumed = train_model(
            model_name="resume_model",
            scheme="elastic_net",
            penalty=0.07,
            mix=0.4,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            hyperparameters=hyper,
            quota=20,
            resume_state=saved,
        )
        self.assertEqual(resumed, direct)

    def test_validation_metrics_are_finite(self) -> None:
        x_train, y_train, x_validation, y_validation = synthetic_xy()
        result = train_model(
            model_name="metrics_model",
            scheme="l1",
            penalty=0.01,
            mix=None,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            hyperparameters=hyperparameters(epochs=3),
            quota=10,
        )
        metrics = result["validation_metrics"]
        for key in ("mse", "rmse", "mae", "accuracy@0.5", "log_loss@clipped"):
            self.assertTrue(math.isfinite(metrics[key]))


if __name__ == "__main__":
    unittest.main()
