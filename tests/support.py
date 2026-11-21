from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "data" / "sample"


def ensure_sample_data() -> Path:
    if not (SAMPLE_DATA / "candidates_v1.txt").exists():
        from candidate_exchange.generator import generate_dataset

        generate_dataset(SAMPLE_DATA, seed=20261122, force=True)
    return SAMPLE_DATA


def normalized_state(report: dict[str, Any]) -> dict[str, Any]:
    conversion = copy.deepcopy(report.get("conversion"))
    if isinstance(conversion, dict):
        conversion.pop("outputs", None)
    tracks = []
    for track in report.get("tracks", []):
        item = dict(track)
        item.pop("reused", None)
        tracks.append(item)
    return {
        "active_version": report.get("active_version"),
        "frames": report.get("frames"),
        "tracks": tracks,
        "failures": report.get("failures"),
        "conversion": conversion,
    }
