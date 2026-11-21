from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from mvcc_workbench.loader import load_dataset


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def sample(name: str) -> dict[str, Any]:
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def base_dataset():
    return load_dataset(SAMPLES / "base.json")


def evolved_dataset():
    return load_dataset(SAMPLES / "evolved.json")


def temp_db(prefix: str = "mvcc-test-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix)) / "ledger.sqlite"


def deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)
