"""Deterministic anomaly injection.

Injections describe real input faults applied at a declared stage. The loader
faults (non-finite score, out-of-range group, binary flag corruption) mutate the
JSONL record before validation so that the normal validation pipeline produces
the matching business error code. Snapshot-staleness is handled by the runner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

FAULT_NON_FINITE_SCORE = "non_finite_score"
FAULT_GROUP_OUT_OF_RANGE = "group_out_of_range"
FAULT_BAD_CLICK_FLAG = "bad_click_flag"
FAULT_STALE_SNAPSHOT = "stale_snapshot_reuse"

LOADER_FAULTS = {FAULT_NON_FINITE_SCORE, FAULT_GROUP_OUT_OF_RANGE, FAULT_BAD_CLICK_FLAG}


@dataclass(frozen=True)
class InjectionHit:
    injection_id: str
    stage: str
    fault: str
    target: str
    code: str

    def to_dict(self):
        return {
            "injection_id": self.injection_id,
            "stage": self.stage,
            "fault": self.fault,
            "target": self.target,
            "code": self.code,
        }


def _matches(record: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    for key in ("request_id", "item_id", "group", "user_id"):
        if key in spec and str(record.get(key)) != str(spec[key]):
            return False
    return True


def apply_loader_injections(
    records: List[Dict[str, Any]], injections: List[Dict[str, Any]]
) -> List[InjectionHit]:
    hits: List[InjectionHit] = []
    active = [s for s in injections if s.get("fault") in LOADER_FAULTS]
    hit_specs: set = set()
    for record in records:
        for spec in active:
            if not _matches(record, spec):
                continue
            spec_id = str(spec["id"])
            if spec_id in hit_specs:
                continue
            fault = spec["fault"]
            injection_id = spec_id
            if fault == FAULT_NON_FINITE_SCORE:
                record["score"] = math.nan
                code = "E_LOG_INVALID"
            elif fault == FAULT_GROUP_OUT_OF_RANGE:
                record["group"] = f"__bogus_{record['group']}__"
                code = "E_LOG_INVALID"
            else:
                record["clicked"] = 7
                code = "E_LOG_INVALID"
            hits.append(
                InjectionHit(
                    injection_id=injection_id,
                    stage="load",
                    fault=fault,
                    target=str(record.get("request_id", "")),
                    code=code,
                )
            )
            hit_specs.add(spec_id)
    hits.sort(key=lambda h: (h.injection_id, h.target))
    return hits
