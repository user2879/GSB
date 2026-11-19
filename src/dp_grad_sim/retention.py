from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FidelityEntry:
    window: int
    step: int
    gradient_id: str
    tensors: dict[str, np.ndarray]

    def metadata(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "step": self.step,
            "gradient_id": self.gradient_id,
            "tensors": {
                name: {"shape": list(array.shape), "dtype": "<f4"}
                for name, array in sorted(self.tensors.items())
            },
        }


class FidelityStore:
    def __init__(self, generations: int, entries: list[FidelityEntry] | None = None) -> None:
        self.generations = generations
        self.entries = sorted(entries or [], key=lambda entry: (entry.window, entry.gradient_id))
        self.rotation_events: list[dict[str, Any]] = []

    def add(
        self,
        window: int,
        step: int,
        gradients: dict[str, np.ndarray],
    ) -> FidelityEntry | None:
        if self.generations <= 0:
            return None
        gradient_id = f"fidelity_w{window:06d}_s{step:06d}"
        if any(entry.gradient_id == gradient_id for entry in self.entries):
            return next(entry for entry in self.entries if entry.gradient_id == gradient_id)
        entry = FidelityEntry(
            window=window,
            step=step,
            gradient_id=gradient_id,
            tensors={
                name: np.asarray(values, dtype=np.float32).copy()
                for name, values in sorted(gradients.items())
            },
        )
        self.entries.append(entry)
        self.entries.sort(key=lambda item: (item.window, item.gradient_id))
        while len(self.entries) > self.generations:
            removed = self.entries.pop(0)
            self.rotation_events.append(
                {
                    "window": window,
                    "replaced_gradient_id": removed.gradient_id,
                    "replaced_window": removed.window,
                    "inserted_gradient_id": entry.gradient_id,
                    "retention_generations": self.generations,
                    "reason": "retention capacity exceeded; oldest window rotated out",
                }
            )
        return entry

    def ids(self) -> list[str]:
        return [entry.gradient_id for entry in self.entries]

    def by_id(self, gradient_id: str) -> FidelityEntry | None:
        return next((entry for entry in self.entries if entry.gradient_id == gradient_id), None)

