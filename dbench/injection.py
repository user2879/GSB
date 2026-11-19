"""Independent fault-injection manifest and real input fault application.

The injection file is separate from the render config. Each item has a unique
``id``; when the fault fires, the id is recorded in the diagnostic report next
to the mapped business code, proving exactly which injection was hit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .config import LATENT_H, LATENT_W
from .errors import (
    E_CONFIG_INVALID,
    E_INJECTED_PATCH_OUT_OF_RANGE,
    E_NON_FINITE_LATENT,
    E_NOISE_ENTRY_TRUNCATED,
    WorkbenchError,
)

SUPPORTED_TYPES = {
    "non_finite_latent",
    "image_size_mismatch",
    "noise_entry_truncated",
    "patch_out_of_range",
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    fault_type: str
    stage: str
    step: int | None = None
    enabled: bool = True
    params: dict | None = None


def load_injections(path: str | None) -> tuple[Injection, ...]:
    if not path:
        return tuple()
    if not os.path.isfile(path):
        raise WorkbenchError(E_CONFIG_INVALID, f"injection file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("injections"), list):
        raise WorkbenchError(E_CONFIG_INVALID,
                             "injection file must contain an 'injections' array")
    items: list[Injection] = []
    seen_ids: set[str] = set()
    for idx, entry in enumerate(raw["injections"]):
        if not isinstance(entry, dict):
            raise WorkbenchError(E_CONFIG_INVALID, f"injection[{idx}] must be an object")
        for key in ("id", "type", "stage"):
            if key not in entry:
                raise WorkbenchError(
                    E_CONFIG_INVALID,
                    f"injection[{idx}] missing field '{key}'",
                )
        if entry["id"] in seen_ids:
            raise WorkbenchError(E_CONFIG_INVALID,
                                 f"duplicate injection id: {entry['id']}")
        seen_ids.add(entry["id"])
        if entry["type"] not in SUPPORTED_TYPES:
            raise WorkbenchError(E_CONFIG_INVALID,
                                 f"unsupported injection type: {entry['type']}")
        step = entry.get("step")
        if step is not None and not (isinstance(step, int) and not isinstance(step, bool)):
            raise WorkbenchError(E_CONFIG_INVALID,
                                 f"injection {entry['id']} step must be int or null")
        items.append(
            Injection(
                injection_id=str(entry["id"]),
                fault_type=str(entry["type"]),
                stage=str(entry["stage"]),
                step=step,
                enabled=bool(entry.get("enabled", True)),
                params=entry.get("params", {}) or {},
            )
        )
    return tuple(items)


def matching_injections(items: tuple[Injection, ...], stage: str,
                        step: int | None) -> list[Injection]:
    out = []
    for item in items:
        if not item.enabled or item.stage != stage:
            continue
        if item.step is None or item.step == step:
            out.append(item)
    return out


# ---- real fault applicators ------------------------------------------------

def apply_non_finite_latent(latent, injection: Injection):
    """Corrupt a real latent element into +inf (denoise output fault)."""
    import numpy as np

    result = latent.copy()
    coords = (injection.params or {}).get("coords")
    if coords is None:
        result.reshape(-1)[0] = np.inf
    else:
        c, h, w = coords
        result[c, h, w] = np.nan
    return result


def truncate_noise_entry(entry, injection: Injection):
    """Cut a real noise-table entry short so its length stops matching."""
    keep = int((injection.params or {}).get("keep", 1))
    keep = max(0, min(keep, len(entry) - 1))
    return entry[:keep]


def build_out_of_range_patch(injection: Injection, channels: int):
    """Construct a real patch whose rectangle lies outside the latent grid."""
    from .config import Patch

    params = injection.params or {}
    top = int(params.get("top", LATENT_H - 1))
    left = int(params.get("left", LATENT_W - 1))
    height = int(params.get("height", 2))
    width = int(params.get("width", 2))
    values = tuple(
        tuple(tuple(0.0 for _ in range(channels)) for _ in range(width))
        for _ in range(height)
    )
    return Patch(step=int(injection.step or 1), top=top, left=left,
                 height=height, width=width, values=values)


# Mapped business code per fault type (requirement 6).
FAULT_CODE = {
    "non_finite_latent": E_NON_FINITE_LATENT,
    "image_size_mismatch": "E_IMAGE_SIZE_MISMATCH",
    "noise_entry_truncated": E_NOISE_ENTRY_TRUNCATED,
    "patch_out_of_range": E_INJECTED_PATCH_OUT_OF_RANGE,
}
