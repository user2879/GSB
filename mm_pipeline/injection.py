"""Independent fault injection configuration and real input mutation.

Injections are declared in a separate YAML file so they never alter the
production flow config. File-level injections physically mutate a *copy* of the
sample directory (truncated bytes / removed fields / dangling relation refs),
while parameter-level injections override the in-memory config used for a run.
Every injection carries a stable id that is echoed into the diagnostic report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import STAGE_LOAD, STAGE_FUSION
from .errors import (
    E_IMAGE_FORMAT,
    E_ASSERTION_SCHEMA,
    E_RELATION_REF,
    E_CONFIG_INVALID,
)

# injection_type -> (business code, stage at which it must surface)
INJECTION_SPEC: dict[str, tuple[str, str]] = {
    "truncate_image": (E_IMAGE_FORMAT, STAGE_LOAD),
    "missing_assertion_field": (E_ASSERTION_SCHEMA, STAGE_LOAD),
    "dangling_relation_region": (E_RELATION_REF, STAGE_LOAD),
    "nonfinite_fusion_param": (E_CONFIG_INVALID, STAGE_LOAD),
    "threshold_out_of_range": (E_CONFIG_INVALID, STAGE_LOAD),
    "crash_after_stage": ("E_INJECTION_HIT", STAGE_FUSION),
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    injection_type: str
    sample_id: Optional[str]
    stage: str
    target_file: Optional[str]
    params: dict[str, Any]

    @property
    def expected_code(self) -> str:
        return INJECTION_SPEC[self.injection_type][0]


def load_injections(path: str | Path) -> list[Injection]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not raw or "injections" not in raw:
        return []
    injections: list[Injection] = []
    for item in raw["injections"]:
        itype = item["injection_type"]
        if itype not in INJECTION_SPEC:
            raise ValueError(f"unknown injection_type: {itype}")
        code, stage = INJECTION_SPEC[itype]
        injections.append(
            Injection(
                injection_id=item["injection_id"],
                injection_type=itype,
                sample_id=item.get("sample_id"),
                stage=stage,
                target_file=item.get("target_file"),
                params=item.get("params", {}) or {},
            )
        )
    return sorted(injections, key=lambda inj: inj.injection_id)


def apply_file_injections(
    injections: list[Injection], sample_root: Path
) -> list[Injection]:
    """Physically apply file-level injections inside the sample directory."""
    applied: list[Injection] = []
    for injection in injections:
        if injection.injection_type in (
            "nonfinite_fusion_param",
            "threshold_out_of_range",
            "crash_after_stage",
        ):
            continue
        sample_dir = sample_root / (injection.sample_id or "")
        if injection.injection_type == "truncate_image":
            image_path = sample_dir / "image.png"
            data = image_path.read_bytes()
            keep = int(injection.params.get("keep_bytes", 40))
            image_path.write_bytes(data[: max(8, keep)])
        elif injection.injection_type == "missing_assertion_field":
            path = sample_dir / "assertions.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            field_name = injection.params.get("field", "value")
            index = int(injection.params.get("assertion_index", 0))
            doc["assertions"][index].pop(field_name, None)
            path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        elif injection.injection_type == "dangling_relation_region":
            path = sample_dir / "relations.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            index = int(injection.params.get("relation_index", 0))
            bad_region = int(injection.params.get("bad_region", 9999))
            doc["relations"][index]["region"] = bad_region
            path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        applied.append(injection)
    return applied


def config_overrides(injections: list[Injection]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for injection in injections:
        if injection.injection_type == "nonfinite_fusion_param":
            overrides["fusion_temperature"] = float("nan")
        elif injection.injection_type == "threshold_out_of_range":
            overrides["confidence_threshold"] = float(
                injection.params.get("threshold", 1.7)
            )
    return overrides
