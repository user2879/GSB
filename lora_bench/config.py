"""Adapter configuration parsing, validation and content hashing."""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import (
    E_CONFIG_INVALID,
    E_RANK_INVALID,
    E_TARGET_LAYER_UNKNOWN,
    WorkbenchError,
)
from .utils import hash_json, read_json

RESUME_CONTROL_KEYS = ("concurrency", "commit_interval")


@dataclass(frozen=True)
class TaskOverrides:
    rank: int | None = None
    alpha: float | None = None


@dataclass(frozen=True)
class AdapterConfig:
    rank: int
    alpha: float
    target_layers: tuple[int, ...]
    learning_rate: float
    batch_size: int
    epochs: int
    metric: str
    metric_threshold: float
    concurrency: int
    commit_interval: int
    task_overrides: dict[str, TaskOverrides] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def rank_for(self, task: str) -> int:
        override = self.task_overrides.get(task)
        if override and override.rank is not None:
            return override.rank
        return self.rank

    def alpha_for(self, task: str) -> float:
        override = self.task_overrides.get(task)
        if override and override.alpha is not None:
            return override.alpha
        return self.alpha

    def training_payload(self) -> dict:
        """Subset of the config that changes numerical training results."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_layers": list(self.target_layers),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "metric": self.metric,
            "metric_threshold": self.metric_threshold,
            "task_overrides": {
                task: {"rank": ov.rank, "alpha": ov.alpha}
                for task, ov in sorted(self.task_overrides.items())
            },
        }

    def config_hash(self) -> str:
        # Concurrency and commit interval affect scheduling/snapshots only,
        # never adapter numerics; they are excluded from the reconciliation hash.
        return hash_json(self.training_payload())


def load_config(path: str, layer_count: int) -> AdapterConfig:
    raw = read_json(path)
    return parse_config(raw, layer_count)


def parse_config(raw: dict, layer_count: int) -> AdapterConfig:
    required = (
        "rank",
        "alpha",
        "target_layers",
        "learning_rate",
        "batch_size",
        "epochs",
        "metric",
        "metric_threshold",
        "concurrency",
        "commit_interval",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise WorkbenchError(
            E_CONFIG_INVALID, f"config missing fields {sorted(missing)}"
        )
    target_layers = raw["target_layers"]
    if not isinstance(target_layers, list) or not target_layers:
        raise WorkbenchError(
            E_TARGET_LAYER_UNKNOWN, "target_layers must be a non-empty list"
        )
    bad_layers = [
        layer
        for layer in target_layers
        if not isinstance(layer, int) or layer < 0 or layer >= layer_count
    ]
    if bad_layers:
        raise WorkbenchError(
            E_TARGET_LAYER_UNKNOWN,
            f"target layers outside [0,{layer_count}): {bad_layers}",
        )
    layers = tuple(sorted(set(target_layers)))
    rank = raw["rank"]
    if not isinstance(rank, int) or rank <= 0:
        raise WorkbenchError(E_RANK_INVALID, f"rank must be positive, got {rank}")
    alpha = raw["alpha"]
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise WorkbenchError(
            E_CONFIG_INVALID, f"alpha must be positive, got {alpha}"
        )
    for key in ("learning_rate", "metric_threshold"):
        if not isinstance(raw[key], (int, float)) or raw[key] <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, f"{key} must be positive")
    for key in ("batch_size", "epochs", "concurrency", "commit_interval"):
        if not isinstance(raw[key], int) or raw[key] <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, f"{key} must be a positive int")
    if raw["metric"] not in ("accuracy", "mse"):
        raise WorkbenchError(
            E_CONFIG_INVALID, f"unsupported metric {raw['metric']}"
        )
    overrides: dict[str, TaskOverrides] = {}
    for task, entry in (raw.get("task_overrides") or {}).items():
        override_rank = entry.get("rank", rank)
        if override_rank is not None and (
            not isinstance(override_rank, int) or override_rank <= 0
        ):
            raise WorkbenchError(
                E_RANK_INVALID,
                f"task {task}: override rank must be positive",
            )
        overrides[task] = TaskOverrides(
            rank=override_rank,
            alpha=entry.get("alpha", alpha),
        )
    return AdapterConfig(
        rank=int(rank),
        alpha=float(alpha),
        target_layers=layers,
        learning_rate=float(raw["learning_rate"]),
        batch_size=int(raw["batch_size"]),
        epochs=int(raw["epochs"]),
        metric=raw["metric"],
        metric_threshold=float(raw["metric_threshold"]),
        concurrency=int(raw["concurrency"]),
        commit_interval=int(raw["commit_interval"]),
        task_overrides=overrides,
        raw=raw,
    )
