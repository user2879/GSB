"""Independent, per-target fault injection plan."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import E_CONFIG_INVALID, E_INJECTION_HIT
from .util import read_json


_SUPPORTED_KINDS = {
    "truncate_png",
    "truth_bbox_out_of_bounds",
    "missing_template",
    "nonfinite_fusion_weight",
}


@dataclass
class InjectionItem:
    id: str
    kind: str
    target: str
    expected_error: str
    spec: dict[str, Any]
    applied: bool = False
    observed_codes: list[str] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return self.expected_error in self.observed_codes

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "expected_error": self.expected_error,
            "applied": self.applied,
            "hit": self.hit,
            "observed_codes": sorted(set(self.observed_codes)),
        }


@dataclass
class InjectionPlan:
    enabled: bool
    manifest_path: Path | None
    manifest_hash: str | None
    items: list[InjectionItem]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def by_target(self, kind: str) -> dict[str, list[InjectionItem]]:
        result: dict[str, list[InjectionItem]] = {}
        for item in self.items:
            if item.kind == kind:
                result.setdefault(item.target, []).append(item)
        return result

    def raw_overrides(self, samples_dir: Path) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for item in self.items:
            if item.kind != "truncate_png":
                continue
            path = samples_dir / item.target
            data = path.read_bytes()
            ratio = float(item.spec.get("truncate_at_ratio", 0.5))
            keep = max(1, min(len(data), int(round(len(data) * ratio))))
            result[item.target] = data[:keep]
            item.applied = True
        return result

    def truth_mutators(self) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
        result: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        for item in self.items:
            if item.kind != "truth_bbox_out_of_bounds":
                continue
            target = item.target

            def mutate(truth: dict[str, Any], item: InjectionItem = item) -> dict[str, Any]:
                mutated = copy.deepcopy(truth)
                column_index = int(item.spec.get("column_index", 0))
                char_index = int(item.spec.get("char_index", 0))
                columns = mutated.get("columns", [])
                if column_index < 0 or column_index >= len(columns):
                    raise ValueError("injection column_index outside truth")
                chars = columns[column_index].get("chars", [])
                if char_index < 0 or char_index >= len(chars):
                    raise ValueError("injection char_index outside truth")
                chars[char_index]["bbox"] = list(item.spec["bbox"])
                item.applied = True
                return mutated

            result[target] = mutate
        return result

    def effective_config(self, base_config: dict[str, Any], target: str) -> tuple[dict[str, Any], list[InjectionItem]]:
        config = copy.deepcopy(base_config)
        touched: list[InjectionItem] = []
        for item in self.items:
            if item.target != target or item.kind != "nonfinite_fusion_weight":
                continue
            index = int(item.spec.get("weight_index", 0))
            literal = str(item.spec.get("literal", "nan")).lower()
            value = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}.get(literal)
            if value is None:
                value = float("nan")
            if 0 <= index < len(config["fusion_weights"]):
                config["fusion_weights"][index] = value
                item.applied = True
                touched.append(item)
        return config, touched

    def missing_templates(self, target: str) -> tuple[set[str], list[InjectionItem]]:
        missing: set[str] = set()
        touched: list[InjectionItem] = []
        for item in self.items:
            if item.target == target and item.kind == "missing_template":
                missing.add(str(item.spec["template_id"]))
                item.applied = True
                touched.append(item)
        return missing, touched


def load_injection_plan(path: Path | None, enabled: bool) -> InjectionPlan:
    if path is None or not enabled:
        return InjectionPlan(enabled=False, manifest_path=path, manifest_hash=None, items=[])
    try:
        raw = read_json(path)
    except (OSError, ValueError) as exc:
        return InjectionPlan(
            enabled=True,
            manifest_path=path,
            manifest_hash=None,
            items=[],
            diagnostics=[{"code": E_INJECTION_HIT, "message": f"cannot read injection manifest: {exc}"}],
        )
    from .util import sha256_json

    items: list[InjectionItem] = []
    diagnostics: list[dict[str, Any]] = []
    for position, value in enumerate(raw.get("items", [])):
        if not isinstance(value, dict):
            diagnostics.append({"code": E_INJECTION_HIT, "message": f"item {position} is not an object"})
            continue
        kind = str(value.get("kind", ""))
        item = InjectionItem(
            id=str(value.get("id", f"INJ-{position + 1:03d}")),
            kind=kind,
            target=str(value.get("target", "")),
            expected_error=str(value.get("expected_error", "")),
            spec=copy.deepcopy(value),
        )
        if kind not in _SUPPORTED_KINDS or not item.target or not item.expected_error:
            diagnostics.append(
                {
                    "code": E_INJECTION_HIT,
                    "message": f"injection {item.id} has unsupported or incomplete specification",
                    "id": item.id,
                }
            )
        items.append(item)
    return InjectionPlan(
        enabled=True,
        manifest_path=path,
        manifest_hash=sha256_json(raw),
        items=items,
        diagnostics=diagnostics,
    )


def record_injection_observations(
    items: list[InjectionItem], observed: list[dict[str, Any]]
) -> None:
    for item in items:
        codes = {
            str(entry["code"])
            for entry in observed
            if entry.get("file") == item.target or entry.get("target") == item.target
        }
        item.observed_codes = sorted(codes)
