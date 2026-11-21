"""Independent fault-injection manifest handling."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import E_CONFIG_INVALID, WorkbenchError


TRUNCATE_PNG = "truncate_png"
MISSING_ASSERTION_FIELD = "missing_assertion_field"
NONFINITE_FEATURE = "nonfinite_feature"
THRESHOLD_OUT_OF_RANGE = "threshold_out_of_range"
CHUNK_NOT_DIVISIBLE = "chunk_not_divisible"

KNOWN_TYPES = {
    TRUNCATE_PNG,
    MISSING_ASSERTION_FIELD,
    NONFINITE_FEATURE,
    THRESHOLD_OUT_OF_RANGE,
    CHUNK_NOT_DIVISIBLE,
}


@dataclass(frozen=True)
class Injection:
    id: str
    sample: str
    stage: str
    type: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def matches(self, sample: str, stages: Iterable[str], types: Iterable[str]) -> bool:
        return (
            self.enabled
            and self.type in set(types)
            and self.stage in set(stages)
            and (self.sample == "*" or self.sample == sample)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sample": self.sample,
            "stage": self.stage,
            "type": self.type,
            "enabled": self.enabled,
            "params": self.params,
        }


@dataclass
class InjectionHit:
    injection_id: str
    sample: str
    stage: str
    type: str
    code: str
    proof: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "sample": self.sample,
            "stage": self.stage,
            "type": self.type,
            "status": "hit",
            "code": self.code,
            "proof": self.proof,
        }


class InjectionSet:
    def __init__(self, items: list[Injection], source: Path | None = None) -> None:
        self.items = sorted(items, key=lambda item: (item.id, item.sample, item.stage, item.type))
        self.source = source
        self._hits: dict[str, InjectionHit] = {}
        self._miss_reasons: dict[str, str] = {}

    @classmethod
    def empty(cls) -> "InjectionSet":
        return cls([])

    @classmethod
    def load(cls, path: Path | str | None) -> "InjectionSet":
        if path is None:
            return cls.empty()
        path = Path(path)
        if not path.exists():
            raise WorkbenchError(E_CONFIG_INVALID, f"injection manifest not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkbenchError(E_CONFIG_INVALID, f"invalid injection JSON: {exc}") from exc
        items_raw = raw.get("injections", []) if isinstance(raw, dict) else raw
        if not isinstance(items_raw, list):
            raise WorkbenchError(E_CONFIG_INVALID, "injection manifest must contain a list")
        items: list[Injection] = []
        seen: set[str] = set()
        for index, item in enumerate(items_raw):
            if not isinstance(item, dict):
                raise WorkbenchError(E_CONFIG_INVALID, f"injection[{index}] must be an object")
            injection_id = item.get("id", f"INJ-{index + 1:03d}")
            sample = item.get("sample", item.get("target", "*"))
            stage = item.get("stage", "")
            fault_type = item.get("type", item.get("fault", ""))
            enabled = item.get("enabled", True)
            if not isinstance(injection_id, str) or not injection_id:
                raise WorkbenchError(E_CONFIG_INVALID, "injection id must be a non-empty string")
            if injection_id in seen:
                raise WorkbenchError(E_CONFIG_INVALID, f"duplicate injection id: {injection_id}")
            seen.add(injection_id)
            if not isinstance(sample, str) or not sample:
                raise WorkbenchError(E_CONFIG_INVALID, f"injection {injection_id} has invalid sample")
            if not isinstance(stage, str) or not stage:
                raise WorkbenchError(E_CONFIG_INVALID, f"injection {injection_id} has invalid stage")
            if fault_type not in KNOWN_TYPES:
                raise WorkbenchError(
                    E_CONFIG_INVALID,
                    f"injection {injection_id} has unknown type {fault_type!r}",
                )
            if not isinstance(enabled, bool):
                raise WorkbenchError(E_CONFIG_INVALID, f"injection {injection_id} enabled must be boolean")
            params = {key: value for key, value in item.items() if key not in {
                "id", "sample", "target", "stage", "type", "fault", "enabled"
            }}
            items.append(
                Injection(
                    id=injection_id,
                    sample=sample,
                    stage=stage,
                    type=fault_type,
                    enabled=enabled,
                    params=params,
                )
            )
        return cls(items, source=path)

    def find(
        self,
        sample: str,
        stages: Iterable[str],
        types: Iterable[str],
    ) -> Injection | None:
        for item in self.items:
            if item.matches(sample, stages, types):
                return item
        return None

    def applied_image_bytes(
        self, sample: str, data: bytes
    ) -> tuple[bytes, Injection | None]:
        item = self.find(sample, {"load", "load_image", "image"}, {TRUNCATE_PNG})
        if item is None:
            return data, None
        count = int(item.params.get("bytes_removed", item.params.get("truncate_bytes", 32)))
        count = max(1, min(count, len(data)))
        return data[: len(data) - count], item

    def applied_text_document(
        self, sample: str, document: dict[str, Any]
    ) -> tuple[dict[str, Any], Injection | None]:
        item = self.find(sample, {"load", "load_text", "text"}, {MISSING_ASSERTION_FIELD})
        if item is None:
            return document, None
        transformed = copy.deepcopy(document)
        field = str(item.params.get("field", "value"))
        assertions = transformed.get("assertions")
        removed = False
        if isinstance(assertions, list) and assertions:
            index = int(item.params.get("assertion_index", 0))
            index = min(max(index, 0), len(assertions) - 1)
            if isinstance(assertions[index], dict) and field in assertions[index]:
                assertions[index].pop(field)
                removed = True
        if not removed:
            item.params.setdefault("_not_applied", True)
        return transformed, item

    def sample_chunk_override(self, sample: str, default_size: int) -> tuple[int, Injection | None]:
        item = self.find(
            sample,
            {"feature", "visual", "visual_feature", "load_image", "config"},
            {CHUNK_NOT_DIVISIBLE},
        )
        if item is None:
            return default_size, None
        chunk_size = int(item.params.get("chunk_size", default_size + 1))
        return chunk_size, item

    def sample_nonfinite(
        self, sample: str, stages: Iterable[str] | None = None
    ) -> Injection | None:
        stage_filter = set(
            stages or {"feature", "visual", "visual_feature", "align", "alignment"}
        )
        return self.find(
            sample,
            stage_filter,
            {NONFINITE_FEATURE},
        )

    def threshold_fault(self, sample: str = "*") -> Injection | None:
        return self.find(
            sample,
            {"retrieval", "config", "align", "alignment"},
            {THRESHOLD_OUT_OF_RANGE},
        )

    def mark_hit(
        self,
        injection: Injection | None,
        *,
        sample: str,
        stage: str,
        code: str,
        proof: dict[str, Any] | None = None,
    ) -> None:
        if injection is None:
            return
        self._hits[injection.id] = InjectionHit(
            injection_id=injection.id,
            sample=sample,
            stage=stage,
            type=injection.type,
            code=code,
            proof=proof or {},
        )

    def mark_missed(self, injection_id: str, reason: str) -> None:
        if injection_id in self._hits:
            return
        self._miss_reasons.setdefault(injection_id, reason)

    def reset_audit(self) -> None:
        self._hits.clear()
        self._miss_reasons.clear()

    def audit(self) -> dict[str, Any]:
        hits = [self._hits[key].to_dict() for key in sorted(self._hits)]
        missed = []
        for item in self.items:
            if not item.enabled:
                missed.append(
                    {
                        "injection_id": item.id,
                        "sample": item.sample,
                        "stage": item.stage,
                        "type": item.type,
                        "status": "disabled",
                        "code": None,
                        "reason": "disabled by manifest",
                    }
                )
            elif item.id not in self._hits:
                missed.append(
                    {
                        "injection_id": item.id,
                        "sample": item.sample,
                        "stage": item.stage,
                        "type": item.type,
                        "status": "missed",
                        "code": "E_INJECTION_MISSED",
                        "reason": self._miss_reasons.get(item.id, "target did not reach stage"),
                    }
                )
        return {
            "source": str(self.source) if self.source else None,
            "configured": len(self.items),
            "hit_count": len(hits),
            "missed_count": len(missed),
            "hits": hits,
            "missed": missed,
        }
