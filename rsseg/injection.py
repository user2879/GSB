"""Independent fault-injection configuration.

Injections are declared in a separate JSON file so the production sample set
stays clean. Each injection carries a stable id that is copied verbatim into
failure records and the diagnostic report to prove the fault was hit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BusinessError, ErrorCode


@dataclass(frozen=True)
class Injection:
    injection_id: str
    kind: str
    target: tuple[int, int] | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InjectionPlan:
    injections: tuple[Injection, ...]

    def for_index(self) -> Injection | None:
        for injection in self.injections:
            if injection.kind == "asymmetric_adjacency":
                return injection
        return None

    def for_tile(self, key: tuple[int, int]) -> Injection | None:
        for injection in self.injections:
            if injection.target == key and injection.kind in {
                "truncated_png", "non_finite_band", "shape_mismatch"
            }:
                return injection
        return None

    def budget_override(self) -> int | None:
        for injection in self.injections:
            if injection.kind == "zero_budget":
                return 0
        return None

    def by_id(self, injection_id: str) -> Injection | None:
        for injection in self.injections:
            if injection.injection_id == injection_id:
                return injection
        return None


def load_injections(path: str | Path) -> InjectionPlan:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    injections: list[Injection] = []
    for entry in raw.get("injections", []):
        kind = entry["kind"]
        target_raw = entry.get("target")
        target = tuple(target_raw) if target_raw else None
        injections.append(Injection(
            injection_id=str(entry["id"]),
            kind=kind,
            target=target,  # type: ignore[arg-type]
            payload=dict(entry.get("payload", {})),
        ))
    return InjectionPlan(tuple(injections))


def apply_index_injection(index_raw: dict, injection: Injection) -> dict:
    """Return a deep-copied index with one directed (asymmetric) neighbor edge."""

    if injection.kind != "asymmetric_adjacency":
        return index_raw
    data = json.loads(json.dumps(index_raw))
    target = injection.target
    for entry in data["tiles"]:
        if (entry["row"], entry["col"]) == target:
            neighbors = entry["neighbors"]
            if neighbors:
                removed = neighbors[0]
                entry["neighbors"] = neighbors[1:]
                entry["_injection"] = {"id": injection.injection_id, "removed": removed}
                break
    else:  # pragma: no cover - generator guarantees target exists
        raise BusinessError(ErrorCode.E_INJECTION_FAULT,
                            f"injection target {target} missing from index")
    return data


def truncated_png_bytes(path: Path, injection: Injection) -> None:
    keep_ratio = float(injection.payload.get("keep_ratio", 0.5))
    raw = path.read_bytes()
    cut = max(8, int(len(raw) * keep_ratio))
    path.write_bytes(raw[:cut])


def non_finite_patch(path: Path, shape: tuple[int, int, int]) -> None:
    import numpy as np

    patch = np.zeros(shape, dtype=np.float32)
    patch[0, 0, 0] = np.nan
    np.save(path, patch)
