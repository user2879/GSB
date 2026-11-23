from __future__ import annotations

import copy
import json
from pathlib import Path

from mixedmem.schema import parse_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "sample_plan.json"


def raw_plan() -> dict:
    with open(PLAN_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plan_copy() -> dict:
    return copy.deepcopy(raw_plan())


def parsed_plan():
    return parse_plan(raw_plan())
