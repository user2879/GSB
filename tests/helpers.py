from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SPEC_V1 = ROOT / "samples" / "demo_cluster.json"
SPEC_V2 = ROOT / "samples" / "demo_cluster_budget_v2.json"


def sample_payload(path: Path = SPEC_V1) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mutated_sample(mutator) -> dict[str, Any]:
    payload = copy.deepcopy(sample_payload())
    mutator(payload)
    return payload

