from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def sample_spec() -> dict[str, Any]:
    return json.loads((SAMPLES / "spec.json").read_text(encoding="utf-8"))


def changed_spec() -> dict[str, Any]:
    return json.loads((SAMPLES / "spec_context_changed.json").read_text(encoding="utf-8"))


def sample_events(name: str = "normal") -> list[str]:
    return (SAMPLES / f"{name}.ndjson").read_text(encoding="utf-8").splitlines()

