"""Apply the injection manifest as real input mutations.

Injections are matched by (sample, stage) and turned into concrete corrupted
inputs or illegal encoder parameters.  The compute components then raise their
normal deterministic business error codes, so fault handling reuses the exact
same validation path as naturally bad data rather than a special branch.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Injection
from .errors import (
    E_BASE_ZERO,
    E_KEY_POS_OOR,
    E_NONFINITE_INPUT,
    E_SCALE_INVALID,
    DiagError,
)

KIND_TO_CODE = {
    "nonfinite_feature": E_NONFINITE_INPUT,
    "key_pos_out_of_range": E_KEY_POS_OOR,
    "base_zero": E_BASE_ZERO,
    "scale_invalid": E_SCALE_INVALID,
}


@dataclass
class TaskInjection:
    injection: Injection
    feature_overrides: dict | None
    base_override: float | None
    scale_override: float | None
    key_overrides: dict | None


class InjectionBook:
    def __init__(self, injections: list[Injection]):
        self._table: dict[tuple[str, str], list[Injection]] = {}
        for inj in injections:
            self._table.setdefault((inj.sample_id, inj.stage), []).append(inj)

    def for_task(self, sample_id: str, stage: str) -> TaskInjection | None:
        hits = self._table.get((sample_id, stage))
        if not hits:
            return None
        # deterministic: at most one injection per task in the manifest
        inj = hits[0]
        params = inj.params
        feature_overrides = None
        base_override = None
        scale_override = None
        key_overrides = None

        if inj.kind == "nonfinite_feature":
            value = float(params.get("value", "nan"))
            target = params.get("target", "query")
            if target == "query":
                feature_overrides = {
                    "query": [{"component": int(params["component"]),
                               "value": value}],
                    "key": [],
                }
            else:
                feature_overrides = {
                    "query": [],
                    "key": [{"key_index": int(params["key_index"]),
                             "component": int(params["component"]),
                             "value": value}],
                }
        elif inj.kind == "base_zero":
            base_override = 0.0
        elif inj.kind == "scale_invalid":
            scale_override = float(params.get("value", 0.0))
        elif inj.kind == "key_pos_out_of_range":
            key_overrides = {
                "key_index": int(params["key_index"]),
                "position": int(params["position"]),
            }
        else:
            raise DiagError("E_CONFIG",
                            "unknown injection kind {}".format(inj.kind))
        return TaskInjection(inj, feature_overrides, base_override,
                             scale_override, key_overrides)

    @staticmethod
    def expected_code(kind: str) -> str:
        return KIND_TO_CODE[kind]
