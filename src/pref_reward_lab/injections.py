from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import read_json
from .errors import ConfigError

INJECTION_CODES = {
    "non_finite_weight": "E_NONFINITE_WEIGHT",
    "zero_hash_buckets": "E_HASH_BUCKETS_ZERO",
    "boundary_equal_threshold": "W_BOUNDARY_EQUAL_THRESHOLD",
    "missing_sidecar": "E_SIDECAR_MISSING",
}
INJECTION_STAGES = {
    "non_finite_weight": "scoring",
    "zero_hash_buckets": "features",
    "boundary_equal_threshold": "scoring",
    "missing_sidecar": "conversion",
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    pair_id: str
    stage: str
    kind: str

    @property
    def code(self) -> str:
        return INJECTION_CODES[self.kind]

    def to_dict(self, *, hit: bool, message: str = "") -> dict[str, Any]:
        return {
            "id": self.injection_id,
            "pair_id": self.pair_id,
            "stage": self.stage,
            "kind": self.kind,
            "hit": hit,
            "business_code": self.code,
            "message": message,
        }


@dataclass
class InjectionSet:
    items: tuple[Injection, ...]

    @classmethod
    def empty(cls) -> "InjectionSet":
        return cls(())

    @classmethod
    def load(cls, path: str | Path | None) -> "InjectionSet":
        if path is None:
            return cls.empty()
        try:
            value = read_json(path)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read injections: {exc}", field="injections") from exc
        raw_items = value.get("injections", value) if isinstance(value, dict) else value
        if not isinstance(raw_items, list):
            raise ConfigError("injections must be an array", field="injections")
        result: list[Injection] = []
        seen_ids: set[str] = set()
        seen_targets: set[tuple[str, str]] = set()
        for index, raw in enumerate(raw_items):
            prefix = f"injections[{index}]"
            if not isinstance(raw, dict):
                raise ConfigError("injection must be an object", field=prefix)
            injection_id = raw.get("id")
            pair_id = raw.get("pair_id")
            kind = raw.get("kind", raw.get("type"))
            stage = raw.get("stage")
            if not isinstance(injection_id, str) or not injection_id.strip():
                raise ConfigError("injection id is required", field=f"{prefix}.id")
            if injection_id in seen_ids:
                raise ConfigError("injection ids must be unique", field=f"{prefix}.id")
            seen_ids.add(injection_id)
            if not isinstance(pair_id, str) or not pair_id:
                raise ConfigError("injection pair_id is required", field=f"{prefix}.pair_id")
            if kind not in INJECTION_CODES:
                raise ConfigError(f"unknown injection kind: {kind}", field=f"{prefix}.kind")
            expected_stage = INJECTION_STAGES[str(kind)]
            if stage is not None and stage != expected_stage:
                raise ConfigError(
                    f"kind {kind} must target stage {expected_stage}",
                    field=f"{prefix}.stage",
                )
            target = (pair_id, expected_stage)
            if target in seen_targets:
                raise ConfigError(
                    "only one injection may target a given pair and stage",
                    field=f"{prefix}.pair_id",
                )
            seen_targets.add(target)
            result.append(
                Injection(
                    injection_id=injection_id.strip(),
                    pair_id=pair_id,
                    stage=expected_stage,
                    kind=str(kind),
                )
            )
        result.sort(key=lambda item: (item.injection_id, item.pair_id, item.stage))
        return cls(tuple(result))

    def at(self, pair_id: str, stage: str) -> Injection | None:
        for item in self.items:
            if item.pair_id == pair_id and item.stage == stage:
                return item
        return None

    def fingerprint(self, pair_id: str, stage: str) -> str:
        item = self.at(pair_id, stage)
        return "" if item is None else f"{item.injection_id}:{item.kind}"

    def build_report(self, hits: dict[str, bool], messages: dict[str, str] | None = None) -> list[dict[str, Any]]:
        messages = messages or {}
        return [
            item.to_dict(
                hit=bool(hits.get(item.injection_id, False)),
                message=messages.get(item.injection_id, ""),
            )
            for item in self.items
        ]
