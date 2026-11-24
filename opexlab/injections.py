"""Independent fault-injection runner with business-code evidence."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .config import LabConfig, default_state
from .engine import ExecutionEngine
from .errors import E_CONFIG, LabError
from .loaders import read_source_bundle
from .models import SourceBundle
from .util import read_json, write_json


def load_injection_manifest(path: Path) -> list[Dict[str, Any]]:
    try:
        value = read_json(path)
    except Exception as exc:
        raise LabError(E_CONFIG, f"cannot read injection manifest: {exc}", stage="injection")
    if not isinstance(value, dict) or not isinstance(value.get("injections"), list):
        raise LabError(E_CONFIG, "injection manifest must contain an injections list", stage="injection")
    result = []
    for item in value["injections"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise LabError(E_CONFIG, "every injection must have an id", stage="injection")
        if not isinstance(item.get("kind"), str):
            raise LabError(E_CONFIG, "every injection must have a kind", stage="injection")
        if not isinstance(item.get("expected_code"), str):
            raise LabError(E_CONFIG, "every injection must have an expected_code", stage="injection")
        result.append(item)
    return result


def _parse_v1_identity(line: str, widths: Dict[str, int]) -> tuple[str, int]:
    layer = line[: widths["layer"]].strip()
    row_text = line[widths["layer"] : widths["layer"] + widths["row"]].strip()
    row = int(row_text)
    return layer, row


def _find_v1_line_index(
    lines: list[str], widths: Dict[str, int], layer: str, row: int
) -> Optional[int]:
    total = sum(widths.values())
    for index, line in enumerate(lines):
        if len(line.rstrip("\r\n")) < total:
            continue
        try:
            found_layer, found_row = _parse_v1_identity(line.rstrip("\r\n"), widths)
        except Exception:
            continue
        if found_layer == layer and found_row == row:
            return index
    return None


def _activation_object(
    lines: list[str], layer: str
) -> Optional[tuple[int, Dict[str, Any]]]:
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("layer") == layer:
            return index, obj
    return None


def apply_injection(
    config: LabConfig,
    bundle: SourceBundle,
    item: Dict[str, Any],
) -> tuple[SourceBundle, Dict[str, Any]]:
    mutated = copy.deepcopy(bundle)
    target = item.get("target", {})
    if not isinstance(target, dict):
        target = {}
    layer = target.get("layer")
    row = target.get("row")
    kind = item["kind"]
    widths = config.field_widths
    budget_override: Optional[int] = None
    note: Dict[str, Any] = {"kind": kind, "target": target}
    if kind == "truncated_weight_line":
        if not isinstance(layer, str) or not isinstance(row, int):
            raise LabError(E_CONFIG, "truncated_weight_line requires layer and row", stage="injection")
        index = _find_v1_line_index(mutated.v1_lines, widths, layer, row)
        if index is None:
            raise LabError(E_CONFIG, "injection target v1 row was not found", stage="injection")
        mutated.v1_lines[index] = mutated.v1_lines[index][:-2]
        note["mutation"] = "removed final two fixed-width characters"
    elif kind == "nonfinite_activation":
        if not isinstance(layer, str) or not isinstance(row, int):
            raise LabError(E_CONFIG, "nonfinite_activation requires layer and row", stage="injection")
        found = _activation_object(mutated.activation_lines, layer)
        if found is None:
            raise LabError(E_CONFIG, "injection target activation layer was not found", stage="injection")
        index, obj = found
        activation = obj.get("activation")
        if not isinstance(activation, list) or row < 0 or row >= len(activation):
            raise LabError(E_CONFIG, "injection target activation row was not found", stage="injection")
        activation[row][0] = float("nan")
        mutated.activation_lines[index] = json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True
        )
        note["mutation"] = "set activation[0] to NaN"
    elif kind == "unknown_scheme":
        if not isinstance(layer, str) or not isinstance(row, int):
            raise LabError(E_CONFIG, "unknown_scheme requires layer and row", stage="injection")
        found = False
        for obj in mutated.v2_values:
            if isinstance(obj, dict) and obj.get("layer") == layer and obj.get("row") == row:
                obj["scheme"] = "layernorm_unknown_scheme"
                found = True
                break
        if not found:
            raise LabError(E_CONFIG, "injection target v2 row was not found", stage="injection")
        note["mutation"] = "replaced scheme with layernorm_unknown_scheme"
    elif kind == "width_out_of_range":
        if not isinstance(layer, str) or not isinstance(row, int):
            raise LabError(E_CONFIG, "width_out_of_range requires layer and row", stage="injection")
        index = _find_v1_line_index(mutated.v1_lines, widths, layer, row)
        if index is None:
            raise LabError(E_CONFIG, "injection target v1 row was not found", stage="injection")
        line = mutated.v1_lines[index]
        start = sum(widths[name] for name in ("layer", "row", "scale_permille", "bias_code"))
        width = widths["quant_bits"]
        mutated.v1_lines[index] = line[:start] + f"{7:>{width}d}" + line[start + width :]
        note["mutation"] = "set quant_bits to 7"
    elif kind == "budget_smaller_than_row":
        if not isinstance(layer, str) or not isinstance(row, int):
            raise LabError(E_CONFIG, "budget_smaller_than_row requires layer and row", stage="injection")
        found = _activation_object(mutated.activation_lines, layer)
        if found is None:
            raise LabError(E_CONFIG, "injection target activation layer was not found", stage="injection")
        _index, obj = found
        dim = obj.get("dim")
        if not isinstance(dim, int) or dim <= 1:
            raise LabError(E_CONFIG, "injection target dim is invalid", stage="injection")
        budget_override = dim - 1
        note["mutation"] = f"set batch_element_budget to {budget_override}"
    else:
        raise LabError(E_CONFIG, f"unknown injection kind: {kind}", stage="injection")
    mutated.injection_id = str(item["id"])
    note["budget_override"] = budget_override
    return mutated, note


def run_injections(
    config: LabConfig,
    *,
    manifest_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest_path = manifest_path or config.paths.get("injections")
    if manifest_path is None or not Path(manifest_path).exists():
        raise LabError(E_CONFIG, "injection manifest path is not configured", stage="injection")
    manifest_path = Path(manifest_path)
    report_path = report_path or (config.runtime_dir / "reports" / "injection_report.json")
    items = load_injection_manifest(manifest_path)
    base_bundle = read_source_bundle(config)
    (config.runtime_dir / "tmp").mkdir(parents=True, exist_ok=True)
    report_items = []
    for item in items:
        mutated, note = apply_injection(config, base_bundle, item)
        state = default_state(config)
        if note.get("budget_override") is not None:
            state["batch_element_budget"] = int(note["budget_override"])
        injection_runtime = Path(
            tempfile.mkdtemp(prefix=f"inject-{item['id']}-", dir=str(config.runtime_dir / "tmp"))
        )
        try:
            engine = ExecutionEngine(config, runtime_dir=injection_runtime)
            execution = engine.run(
                state,
                mode="full",
                bundle=mutated,
                commit_snapshot=False,
                injection_id=str(item["id"]),
            )
        finally:
            shutil.rmtree(injection_runtime, ignore_errors=True)
        matching = [
            failure
            for failure in execution["failures"]
            if failure.get("code") == item["expected_code"]
            and failure.get("injection_id") == item["id"]
        ]
        target = item.get("target", {})
        if isinstance(target, dict):
            targeted = [
                failure
                for failure in matching
                if failure.get("layer") == target.get("layer")
                and (
                    target.get("row") is None
                    or failure.get("row") == target.get("row")
                )
            ]
        else:
            targeted = []
        hit_count = len(matching)
        matched = bool(targeted)
        report_items.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "target": target,
                "expected_code": item["expected_code"],
                "matched": matched,
                "hit_count": hit_count,
                "mutation": note.get("mutation"),
                "budget_override": note.get("budget_override"),
                "observed_codes": sorted({failure["code"] for failure in execution["failures"]}),
                "sample_failures": matching[:3],
            }
        )
    report = {
        "schema": "opexlab.injection_report.v1",
        "manifest": str(manifest_path),
        "all_matched": all(item["matched"] for item in report_items),
        "items": report_items,
    }
    write_json(report_path, report)
    return report
