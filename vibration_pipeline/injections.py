from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import _is_int
from .util import hash_object


INJECTION_STAGES = {
    "non_finite_sample": "feature_extraction",
    "band_edge_out_of_range": "feature_extraction",
    "peak_factor_window_zero": "scoring",
    "threshold_equal_boundary": "threshold",
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    kind: str
    stage: str
    device_id: str
    segment_index: int
    algorithm: str | None

    def matches(self, *, stage: str, device_id: str, segment_index: int, algorithm: str | None = None) -> bool:
        return (
            self.stage == stage
            and self.device_id == device_id
            and self.segment_index == segment_index
            and (self.algorithm is None or algorithm is None or self.algorithm == algorithm)
        )


def parse_injections(raw: dict[str, Any] | None) -> tuple[list[Injection], list[str]]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return [], ["injection configuration root must be a mapping"]
    entries = raw.get("injections", [])
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        return [], ["injections must be a list"]
    parsed: list[Injection] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            errors.append(f"injection {index} must be a mapping")
            continue
        injection_id = item.get("id")
        if not isinstance(injection_id, str) or not injection_id:
            errors.append(f"injection {index} has no non-empty id")
            continue
        if injection_id in seen:
            errors.append(f"duplicate injection id {injection_id}")
            continue
        kind = item.get("kind")
        if kind not in INJECTION_STAGES:
            errors.append(f"injection {injection_id} has unknown kind {kind!r}")
            continue
        stage = item.get("stage", INJECTION_STAGES[kind])
        if stage != INJECTION_STAGES[kind]:
            errors.append(f"injection {injection_id} stage must be {INJECTION_STAGES[kind]}")
            continue
        device_id = item.get("device_id")
        segment_index = item.get("segment_index")
        if not isinstance(device_id, str) or not device_id:
            errors.append(f"injection {injection_id} device_id must be a non-empty string")
            continue
        if not _is_int(segment_index) or segment_index < 0:
            errors.append(f"injection {injection_id} segment_index must be a non-negative integer")
            continue
        algorithm = item.get("algorithm")
        if algorithm is not None and not isinstance(algorithm, str):
            errors.append(f"injection {injection_id} algorithm must be a string or null")
            continue
        if kind == "peak_factor_window_zero" and algorithm != "crest_factor":
            errors.append(f"injection {injection_id} peak_factor_window_zero requires algorithm=crest_factor")
            continue
        seen.add(injection_id)
        parsed.append(Injection(injection_id, kind, stage, device_id, int(segment_index), algorithm))
    return sorted(parsed, key=lambda value: value.injection_id), errors


def injection_hash(raw: dict[str, Any] | None) -> str:
    return hash_object(raw or {}, namespace="injection-config-v1")


class InjectionTracker:
    def __init__(self, injections: list[Injection]) -> None:
        self.injections = {injection.injection_id: injection for injection in injections}
        self.hits: dict[str, dict[str, Any]] = {}

    def matching(
        self,
        *,
        stage: str,
        device_id: str,
        segment_index: int,
        algorithm: str | None = None,
    ) -> list[Injection]:
        return [
            self.injections[injection_id]
            for injection_id in sorted(self.injections)
            if self.injections[injection_id].matches(
                stage=stage,
                device_id=device_id,
                segment_index=segment_index,
                algorithm=algorithm,
            )
        ]

    def mark(self, injection: Injection, *, effect: str, code: str | None = None) -> None:
        if injection.injection_id in self.hits:
            return
        self.hits[injection.injection_id] = {
            "id": injection.injection_id,
            "kind": injection.kind,
            "stage": injection.stage,
            "device_id": injection.device_id,
            "segment_index": injection.segment_index,
            "algorithm": injection.algorithm,
            "hit": True,
            "effect": effect,
            "code": code,
        }

    def report(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for injection_id in sorted(self.injections):
            if injection_id in self.hits:
                output.append(self.hits[injection_id])
            else:
                injection = self.injections[injection_id]
                output.append(
                    {
                        "id": injection.injection_id,
                        "kind": injection.kind,
                        "stage": injection.stage,
                        "device_id": injection.device_id,
                        "segment_index": injection.segment_index,
                        "algorithm": injection.algorithm,
                        "hit": False,
                        "effect": "target_not_reached",
                        "code": None,
                    }
                )
        return output
