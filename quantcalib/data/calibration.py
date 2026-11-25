"""Calibration sample JSON loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantcalib.errors import E_CONFIG_INVALID, Issue
from quantcalib.models import CalibrationSet


def load_calibration(path: str | Path) -> tuple[CalibrationSet | None, list[Issue]]:
    issues: list[Issue] = []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    inputs = np.asarray(payload["inputs"], dtype=np.float64)
    refs = np.asarray(payload["reference_outputs"], dtype=np.float64)
    seed = int(payload.get("seed", 0))

    if inputs.ndim != 2 or refs.ndim != 2 or inputs.shape[0] != refs.shape[0]:
        issues.append(Issue(E_CONFIG_INVALID,
                            "inputs/reference_outputs must be aligned 2-D matrices",
                            detail={"inputs": list(inputs.shape),
                                    "reference_outputs": list(refs.shape)}))
        return None, issues
    if not (np.all(np.isfinite(inputs)) and np.all(np.isfinite(refs))):
        issues.append(Issue(E_CONFIG_INVALID, "calibration data must be finite"))
        return None, issues

    return CalibrationSet(inputs=inputs, reference_outputs=refs, seed=seed), issues


def write_calibration(path: str | Path, calib: CalibrationSet) -> None:
    payload = {
        "seed": calib.seed,
        "inputs": calib.inputs.tolist(),
        "reference_outputs": calib.reference_outputs.tolist(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
