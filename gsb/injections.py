"""Numbered, isolated fault-injection suite for diagnostics."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gsb.config import load_config, validate_config_file
from gsb.constants import E_CONFIG_INVALID
from gsb.errors import ValidationIssue
from gsb.fonts import load_templates
from gsb.samples import load_dataset, validate_png_bytes


INJECTION_KINDS = {
    "truncated_png",
    "non_integer_keypoint",
    "template_variant_dimension",
    "quantization_out_of_range",
}


@dataclass(frozen=True)
class InjectionSpec:
    injection_id: str
    kind: str
    payload: dict[str, Any]


def load_injection_config(path: str | Path) -> tuple[list[InjectionSpec], list[ValidationIssue]]:
    config_path = Path(path)
    issues: list[ValidationIssue] = []
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], [
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="injection config not found",
                source=str(config_path),
            )
        ]
    except yaml.YAMLError as exc:
        return [], [
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message=f"injection config YAML is invalid: {exc}",
                source=str(config_path),
            )
        ]
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        return [], [
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="injection config schema_version must be 1",
                source=str(config_path),
            )
        ]
    entries = raw.get("injections")
    if not isinstance(entries, list):
        return [], [
            ValidationIssue(
                code=E_CONFIG_INVALID,
                message="injections must be a list",
                source=str(config_path),
            )
        ]
    specs: list[InjectionSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            issues.append(
                ValidationIssue(
                    code=E_CONFIG_INVALID,
                    message=f"injections[{index}] must be an object",
                    source=str(config_path),
                )
            )
            continue
        injection_id = entry.get("id")
        kind = entry.get("kind")
        if not isinstance(injection_id, str) or not injection_id:
            issues.append(
                ValidationIssue(
                    code=E_CONFIG_INVALID,
                    message=f"injections[{index}].id must be declared",
                    source=str(config_path),
                )
            )
            continue
        if injection_id in seen:
            issues.append(
                ValidationIssue(
                    code=E_CONFIG_INVALID,
                    message=f"duplicate injection id: {injection_id}",
                    source=str(config_path),
                )
            )
            continue
        seen.add(injection_id)
        if kind not in INJECTION_KINDS:
            issues.append(
                ValidationIssue(
                    code=E_CONFIG_INVALID,
                    message=f"unsupported injection kind: {kind}",
                    source=str(config_path),
                )
            )
            continue
        payload = {key: value for key, value in entry.items() if key not in {"id", "kind"}}
        specs.append(InjectionSpec(injection_id=injection_id, kind=kind, payload=payload))
    return specs, issues


def _copy_samples(root: Path, temp_root: Path) -> Path:
    destination = temp_root / "samples"
    shutil.copytree(root / "samples", destination)
    return destination


def _truncate_png(root: Path, spec: InjectionSpec) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gsb-inj-") as temp:
        temp_root = Path(temp)
        samples_dir = _copy_samples(root, temp_root)
        sample_id = str(spec.payload.get("sample_id", "page_a"))
        image_path = samples_dir / f"{sample_id}.png"
        payload = image_path.read_bytes()
        image_path.write_bytes(payload[: max(8, len(payload) // 2)])
        config = load_config(root / "configs" / "default.yaml")
        state, issues = load_dataset(samples_dir / "dataset.yaml", config=config)
        injected_issues = [
            issue.to_dict()
            for issue in issues
            if issue.sample_id == sample_id
        ]
        return {
            "hit": bool(injected_issues),
            "code": injected_issues[0]["code"] if injected_issues else None,
            "sample_id": sample_id,
            "surviving_samples": list(state.sample_order),
            "details": injected_issues,
        }


def _non_integer_keypoint(root: Path, spec: InjectionSpec) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gsb-inj-") as temp:
        temp_root = Path(temp)
        samples_dir = _copy_samples(root, temp_root)
        sample_id = str(spec.payload.get("sample_id", "page_a"))
        keypoint_id = str(spec.payload.get("keypoint_id", "page_a-c0-r0"))
        manifest_path = samples_dir / f"{sample_id}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for keypoint in manifest.get("keypoints", []):
            if keypoint.get("id") == keypoint_id:
                keypoint["x"] = float(keypoint["x"]) + 0.5
                break
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        config = load_config(root / "configs" / "default.yaml")
        state, issues = load_dataset(samples_dir / "dataset.yaml", config=config)
        injected_issues = [
            issue.to_dict()
            for issue in issues
            if issue.sample_id == sample_id
        ]
        return {
            "hit": bool(injected_issues),
            "code": injected_issues[0]["code"] if injected_issues else None,
            "sample_id": sample_id,
            "keypoint_id": keypoint_id,
            "surviving_samples": list(state.sample_order),
            "details": injected_issues,
        }


def _template_dimension(root: Path, spec: InjectionSpec) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gsb-inj-") as temp:
        temp_root = Path(temp)
        config = load_config(root / "configs" / "default.yaml")
        source = root / "fonts" / "bitmap-v1.json"
        target = temp_root / "bitmap-v1.json"
        shutil.copyfile(source, target)
        raw = json.loads(target.read_text(encoding="utf-8"))
        char = str(spec.payload.get("char", "古"))
        variant_index = int(spec.payload.get("variant_index", 0))
        for glyph in raw.get("glyphs", []):
            if glyph.get("char") == char:
                glyph["variants"][variant_index]["width"] -= 1
                break
        target.write_text(
            json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        templates, issues = load_templates(
            target,
            expected_version=config.recognition.template_version,
            expected_width=config.recognition.template_width,
            expected_height=config.recognition.template_height,
            expected_weights=config.recognition.variant_weights,
        )
        return {
            "hit": templates is None and bool(issues),
            "code": issues[0].code if issues else None,
            "sample_id": None,
            "char": char,
            "variant_index": variant_index,
            "details": [issue.to_dict() for issue in issues],
        }


def _quantization_out_of_range(root: Path, spec: InjectionSpec) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gsb-inj-") as temp:
        temp_root = Path(temp)
        source = root / "configs" / "default.yaml"
        target = temp_root / "invalid-config.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw.setdefault("quantization", {})["level"] = str(spec.payload.get("level", "Q99"))
        raw.setdefault("quantization", {})["fixed_bits"] = int(spec.payload.get("fixed_bits", 99))
        target.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
        issues = validate_config_file(target)
        return {
            "hit": bool(issues),
            "code": issues[0].code if issues else None,
            "sample_id": None,
            "details": [issue.to_dict() for issue in issues],
        }


def run_injection_suite(
    *,
    root: str | Path,
    config_path: str | Path,
    dataset_path: str | Path,
    templates_path: str | Path,
    injections_path: str | Path,
) -> dict[str, Any]:
    del config_path, dataset_path, templates_path
    root_path = Path(root)
    specs, load_issues = load_injection_config(injections_path)
    results: list[dict[str, Any]] = []
    for spec in specs:
        if spec.kind == "truncated_png":
            outcome = _truncate_png(root_path, spec)
        elif spec.kind == "non_integer_keypoint":
            outcome = _non_integer_keypoint(root_path, spec)
        elif spec.kind == "template_variant_dimension":
            outcome = _template_dimension(root_path, spec)
        elif spec.kind == "quantization_out_of_range":
            outcome = _quantization_out_of_range(root_path, spec)
        else:
            outcome = {"hit": False, "code": None, "details": []}
        results.append(
            {
                "id": spec.injection_id,
                "kind": spec.kind,
                **outcome,
            }
        )
    return {
        "schema_version": 1,
        "config_issues": [issue.to_dict() for issue in load_issues],
        "injections": results,
        "all_hit": bool(results) and all(item["hit"] for item in results) and not load_issues,
    }
