"""Point-budget scheduling and fault-injection isolation."""
from __future__ import annotations

from dataclasses import dataclass, field

E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_NONFINITE_FEATURE = "E_NONFINITE_FEATURE"
E_KEY_POS_OOB = "E_KEY_POS_OOB"
E_BASE_ZERO = "E_BASE_ZERO"
E_SCALE_NONPOSITIVE = "E_SCALE_NONPOSITIVE"

FAULT_TO_CODE = {
    "nonfinite_feature": E_NONFINITE_FEATURE,
    "key_pos_oob": E_KEY_POS_OOB,
    "base_zero": E_BASE_ZERO,
    "scale_nonpositive": E_SCALE_NONPOSITIVE,
}


@dataclass
class ScheduledUnit:
    scheme: str
    sample_id: str
    points: int
    sort_key: tuple = field(init=False)

    def __post_init__(self):
        self.sort_key = (self.scheme, self.sample_id)


def schedule(units: list[ScheduledUnit], point_budget: int) -> dict:
    """Consume the per-batch point budget in stable (scheme, sample) order.

    Units that no longer fit stay in the pending queue and are flagged with
    E_POINT_BUDGET_PENDING.
    """
    ordered = sorted(units, key=lambda u: u.sort_key)
    admitted: list[ScheduledUnit] = []
    pending: list[dict] = []
    spent = 0
    for unit in ordered:
        if spent + unit.points <= point_budget:
            admitted.append(unit)
            spent += unit.points
        else:
            pending.append(
                {
                    "scheme": unit.scheme,
                    "sample_id": unit.sample_id,
                    "points": unit.points,
                    "code": E_POINT_BUDGET_PENDING,
                }
            )
    return {
        "admitted": admitted,
        "pending": pending,
        "spent_points": spent,
        "point_budget": int(point_budget),
    }


class InjectedFault(Exception):
    """A real input fault raised after an injection mutated the input."""

    def __init__(self, code: str, injection_id: str, message: str):
        super().__init__(message)
        self.code = code
        self.injection_id = injection_id


def injections_for(injection_manifest: dict, sample_id: str, stage: str) -> list[dict]:
    return [
        inj
        for inj in injection_manifest.get("injections", [])
        if inj["sample_id"] == sample_id and inj["stage"] == stage
    ]


def apply_load_injections(sample: dict, injections: list[dict]) -> dict:
    """Mutate a sample with real load-stage faults so genuine validation
    downstream detects them (non-finite features, out-of-bounds keys)."""
    mutated = {
        **sample,
        "query": list(sample["query"]),
        "keys": [list(k) for k in sample["keys"]],
        "key_positions": list(sample["key_positions"]),
    }
    for inj in injections:
        if inj["fault"] == "nonfinite_feature":
            mutated["query"][0] = float("nan")
        elif inj["fault"] == "key_pos_oob":
            mutated["key_positions"].append(int(inj.get("oob_position", 10**9)))
            mutated["keys"].append(list(sample["keys"][0]))
    return mutated


def apply_param_injections(params: dict, injections: list[dict]) -> dict:
    """Mutate scheme params with real faults (base=0, scale<=0)."""
    mutated = dict(params)
    for inj in injections:
        if inj["fault"] == "base_zero":
            mutated["base"] = 0.0
        elif inj["fault"] == "scale_nonpositive":
            mutated["scale"] = -abs(float(mutated["scale"]))
    return mutated


def validate_sample(sample: dict, target_len: int, injections: list[dict]) -> None:
    """Genuine input validation; maps injected faults to business codes."""
    import math

    fault_by_type = {inj["fault"]: inj for inj in injections}
    values = list(sample["query"]) + [v for row in sample["keys"] for v in row]
    if any(not math.isfinite(float(v)) for v in values):
        inj = fault_by_type.get("nonfinite_feature")
        raise InjectedFault(
            E_NONFINITE_FEATURE,
            inj["injection_id"] if inj else "NONE",
            f"sample {sample['sample_id']} carries non-finite feature values",
        )
    for pos in sample["key_positions"]:
        if not (0 <= int(pos) <= target_len):
            inj = fault_by_type.get("key_pos_oob")
            raise InjectedFault(
                E_KEY_POS_OOB,
                inj["injection_id"] if inj else "NONE",
                f"sample {sample['sample_id']} key position {pos} out of bounds",
            )
