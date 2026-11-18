"""Per-task low-rank adapter training and validation (NumPy, CPU only).

Each target layer gets a bypass LoRA pair ``B @ A`` (rank ``r``) scaled by
``alpha / r``. Inputs are the frozen base layer inputs cached during the
forward stage, so the base weights are never updated. Training uses closed-form
batch gradients and a fixed sample order for bitwise reproducibility.
"""

from __future__ import annotations

import math

import numpy as np

from .config import AdapterConfig
from .forward import (
    forward_index_by_sample,
    load_forward_entry,
    sample_content_fingerprint,
)
from .loader import BasePack
from .utils import derive_seed, rng_from_seed


def _readout_for(task: str, hidden_dim: int) -> np.ndarray:
    """Deterministic fixed readout row used to map hidden state to a target."""
    rng = rng_from_seed(derive_seed("readout", task))
    return rng.normal(0.0, 0.4, hidden_dim).astype(np.float64)


def _target_value(sample: dict) -> float:
    # Classification targets are mapped into fixed signed positions; regression
    # uses the bounded label directly.
    if isinstance(sample["label"], int) and not isinstance(sample["label"], bool):
        return float(-1.0 + 0.5 * int(sample["label"]))
    return float(sample["label"])


class LoRAAdapter:
    """Bypass low-rank matrices for one task across the configured layers."""

    def __init__(self, task: str, rank: int, alpha: float, layers: tuple[int, ...], hidden_dim: int) -> None:
        self.task = task
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.layers = tuple(sorted(layers))
        self.hidden_dim = hidden_dim
        self.a: dict[int, np.ndarray] = {}
        self.b: dict[int, np.ndarray] = {}
        for layer in self.layers:
            rng = rng_from_seed(derive_seed("lora", task, layer))
            self.a[layer] = rng.normal(0.0, 0.1, (rank, hidden_dim)).astype(np.float64)
            self.b[layer] = (
                rng.normal(0.0, 0.1, (hidden_dim, rank)).astype(np.float64) * 0.1
            )
        self.readout = _readout_for(task, hidden_dim)

    def layer_delta(self, layer: int, x: np.ndarray) -> np.ndarray:
        return self.scaling * (self.b[layer] @ (self.a[layer] @ x))

    def predict_batch(self, forward_caches: list[dict]) -> np.ndarray:
        outputs = []
        for cache in forward_caches:
            hidden = cache["final"].astype(np.float64).copy()
            for layer in self.layers:
                hidden = hidden + self.layer_delta(
                    layer, cache[f"layer{layer}.input"].astype(np.float64)
                )
            outputs.append(float(self.readout @ hidden))
        return np.asarray(outputs, dtype=np.float64)

    def loss(self, forward_caches: list[dict], targets: np.ndarray) -> float:
        predictions = self.predict_batch(forward_caches)
        return float(np.mean((predictions - targets) ** 2))

    def train_epoch(
        self,
        forward_caches: list[dict],
        targets: np.ndarray,
        learning_rate: float,
        batch_size: int,
    ) -> None:
        """One fixed-order pass with analytic mini-batch gradient descent."""
        sample_count = len(forward_caches)
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            batch_caches = forward_caches[start:end]
            batch_targets = targets[start:end]
            size = end - start
            grad_a = {layer: np.zeros_like(matrix) for layer, matrix in self.a.items()}
            grad_b = {layer: np.zeros_like(matrix) for layer, matrix in self.b.items()}
            for cache, target in zip(batch_caches, batch_targets):
                hidden = cache["final"].astype(np.float64).copy()
                # Accumulate deltas layer by layer; cached base inputs stay fixed.
                deltas: dict[int, np.ndarray] = {}
                for layer in self.layers:
                    x = cache[f"layer{layer}.input"].astype(np.float64)
                    delta = self.layer_delta(layer, x)
                    deltas[layer] = delta
                    hidden = hidden + delta
                residual = (2.0 / size) * (float(self.readout @ hidden) - float(target))
                upstream = residual * self.readout
                for layer in self.layers:
                    x = cache[f"layer{layer}.input"].astype(np.float64)
                    ax = self.a[layer] @ x
                    grad_b[layer] += self.scaling * np.outer(upstream, ax)
                    grad_a[layer] += self.scaling * (
                        (self.b[layer].T @ upstream)[:, None] @ x[None, :]
                    )
            for layer in self.layers:
                self.b[layer] -= learning_rate * grad_b[layer]
                self.a[layer] -= learning_rate * grad_a[layer]

    def fingerprint(self) -> str:
        from .utils import fingerprint_arrays

        arrays: list[tuple[str, np.ndarray]] = []
        for layer in self.layers:
            arrays.append((f"layer{layer}.A", self.a[layer]))
            arrays.append((f"layer{layer}.B", self.b[layer]))
        arrays.append(("readout", self.readout))
        return fingerprint_arrays(arrays)

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays = {}
        for layer in self.layers:
            arrays[f"layer{layer}.A"] = self.a[layer]
            arrays[f"layer{layer}.B"] = self.b[layer]
        arrays["readout"] = self.readout
        return arrays

    @classmethod
    def from_arrays(
        cls,
        task: str,
        rank: int,
        alpha: float,
        layers: tuple[int, ...],
        hidden_dim: int,
        arrays: dict[str, np.ndarray],
    ) -> "LoRAAdapter":
        adapter = cls(task, rank, alpha, layers, hidden_dim)
        for layer in adapter.layers:
            adapter.a[layer] = np.asarray(arrays[f"layer{layer}.A"], dtype=np.float64)
            adapter.b[layer] = np.asarray(arrays[f"layer{layer}.B"], dtype=np.float64)
        adapter.readout = np.asarray(arrays["readout"], dtype=np.float64)
        return adapter


def _prepare_samples(
    sample_ids: list[str],
    sample_by_id: dict[str, dict],
    forward_dir: str,
    forward_manifest: dict,
) -> tuple[list[dict], np.ndarray]:
    index = forward_index_by_sample(forward_manifest)
    ordered_ids = sorted(sample_ids)
    caches: list[dict] = []
    targets: list[float] = []
    for sample_id in ordered_ids:
        sample = sample_by_id[sample_id]
        caches.append(load_forward_entry(forward_dir, index[sample_id]))
        targets.append(_target_value(sample))
    return caches, np.asarray(targets, dtype=np.float64)


def _evaluate(
    adapter: LoRAAdapter,
    val_caches: list[dict],
    val_targets: np.ndarray,
    samples: list[dict],
    metric: str,
) -> dict:
    predictions = adapter.predict_batch(val_caches)
    mse = float(np.mean((predictions - val_targets) ** 2))
    result = {
        "metric": metric,
        "mse": mse,
        "sample_count": len(val_caches),
    }
    if metric == "accuracy":
        # Class labels are recovered by rounding the inverse of the signed map.
        predicted_labels = np.rint((predictions + 1.0) / 0.5).astype(int)
        true_labels = np.asarray([int(sample["label"]) for sample in samples])
        accuracy = float(np.mean(predicted_labels == true_labels))
        result["accuracy"] = accuracy
        result["score"] = accuracy
    else:
        # Bounded R^2-style score, higher is better and clipped into [0, 1].
        variance = float(np.mean((val_targets - val_targets.mean()) ** 2))
        r2 = 1.0 - mse / variance if variance > 1e-12 else 0.0
        result["score"] = max(0.0, min(1.0, r2))
    return result


def train_task(
    task_entry: dict,
    samples: list[dict],
    base: BasePack,
    config: AdapterConfig,
    forward_dir: str,
    forward_manifest: dict,
) -> dict:
    """Full deterministic train + validate run for one task."""
    task = task_entry["task"]
    rank = config.rank_for(task)
    if rank <= 0:
        from .errors import E_RANK_INVALID, WorkbenchError

        raise WorkbenchError(
            E_RANK_INVALID, f"task {task}: rank must be positive", stage="train"
        )
    if not task_entry["val_sample_ids"]:
        from .errors import E_VALIDATION_EMPTY, WorkbenchError

        raise WorkbenchError(
            E_VALIDATION_EMPTY,
            f"task {task}: validation sample list is empty",
            stage="validate",
        )
    alpha = config.alpha_for(task)
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    train_caches, train_targets = _prepare_samples(
        task_entry["train_sample_ids"],
        sample_by_id,
        forward_dir,
        forward_manifest,
    )
    ordered_val_ids = sorted(task_entry["val_sample_ids"])
    val_caches, val_targets = _prepare_samples(
        ordered_val_ids, sample_by_id, forward_dir, forward_manifest
    )
    val_samples = [sample_by_id[sample_id] for sample_id in ordered_val_ids]

    adapter = LoRAAdapter(
        task, rank, alpha, config.target_layers, base.hidden_dim
    )
    metric = task_entry.get("metric", config.metric)
    threshold = float(
        task_entry.get("metric_threshold", config.metric_threshold)
    )
    epoch_losses: list[float] = []
    epoch_metrics: list[dict] = []
    epoch_fingerprints: list[str] = []
    for epoch in range(config.epochs):
        adapter.train_epoch(
            train_caches,
            train_targets,
            config.learning_rate,
            config.batch_size,
        )
        epoch_losses.append(adapter.loss(train_caches, train_targets))
        metrics = _evaluate(
            adapter, val_caches, val_targets, val_samples, metric
        )
        epoch_metrics.append(metrics)
        epoch_fingerprints.append(adapter.fingerprint())

    # Higher score wins; ties break by epoch ascending then fingerprint lexicographic.
    best_epoch = min(
        range(config.epochs),
        key=lambda idx: (
            -epoch_metrics[idx]["score"],
            idx,
            epoch_fingerprints[idx],
        ),
    )
    best_metrics = dict(epoch_metrics[best_epoch])
    best_metrics["best_epoch"] = best_epoch
    best_metrics["threshold"] = threshold
    best_metrics["passed"] = bool(best_metrics["score"] >= threshold)

    # Rebuild the adapter state at the chosen epoch for a self-contained artifact.
    final_adapter = LoRAAdapter(
        task, rank, alpha, config.target_layers, base.hidden_dim
    )
    for epoch in range(best_epoch + 1):
        final_adapter.train_epoch(
            train_caches,
            train_targets,
            config.learning_rate,
            config.batch_size,
        )
    if not math.isfinite(epoch_losses[-1]):
        from .errors import E_TRAIN_TASK_FAULT, WorkbenchError

        raise WorkbenchError(
            E_TRAIN_TASK_FAULT,
            f"task {task}: non-finite training loss",
            stage="train",
        )
    return {
        "task": task,
        "rank": rank,
        "alpha": alpha,
        "target_layers": list(config.target_layers),
        "scaling": alpha / rank,
        "metric": metric,
        "epoch_losses": epoch_losses,
        "epoch_metrics": epoch_metrics,
        "epoch_fingerprints": epoch_fingerprints,
        "best": best_metrics,
        "final_fingerprint": final_adapter.fingerprint(),
        "arrays": final_adapter.to_arrays(),
        "train_sample_fingerprints": [
            sample_content_fingerprint(sample_by_id[sample_id])
            for sample_id in sorted(task_entry["train_sample_ids"])
        ],
        "val_sample_fingerprints": [
            sample_content_fingerprint(sample_by_id[sample_id])
            for sample_id in ordered_val_ids
        ],
    }
