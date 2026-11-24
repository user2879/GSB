"""Independent fault injection configuration and business-code mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .errors import E_INPUT_INVALID, E_STAGE_BUDGET_EXCEEDED


@dataclass(frozen=True)
class Injection:
    injection_id: str
    kind: str
    target: str
    payload: Any
    expected_code: str


INJECTION_CATALOG = {
    "INF-001": ("non_finite_weight", E_INPUT_INVALID),
    "INF-002": ("zero_bucket_count", E_INPUT_INVALID),
    "INF-003": ("unknown_upstream", E_INPUT_INVALID),
    "INF-004": ("receipt_width_short", E_INPUT_INVALID),
    "INF-005": ("duplicate_receipt_artifact", E_INPUT_INVALID),
}


def load_injections(doc: Dict[str, Any]) -> Tuple[Injection, ...]:
    injections: List[Injection] = []
    for item in doc.get("injections", []):
        injection_id = item["id"]
        if injection_id not in INJECTION_CATALOG:
            raise KeyError(f"unknown injection {injection_id}")
        kind, expected_code = INJECTION_CATALOG[injection_id]
        injections.append(
            Injection(injection_id, kind, item["target"], item.get("payload"), expected_code)
        )
    return tuple(injections)


@dataclass(frozen=True)
class InjectionReport:
    injection_id: str
    kind: str
    expected_code: str
    observed_code: str
    hit: bool
    locator: str
    message: str

    def to_dict(self) -> dict:
        return {
            "injection_id": self.injection_id,
            "kind": self.kind,
            "expected_code": self.expected_code,
            "observed_code": self.observed_code,
            "hit": self.hit,
            "locator": self.locator,
            "message": self.message,
        }
