"""Real-file injection preparation used by the integration demo and tests."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .errors import QuantCalError
from .util import atomic_write_json, load_json, load_jsonl


def load_injection_manifest(path: str | Path) -> dict[str, Any]:
    document = load_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("injections"), list):
        raise QuantCalError("E_FORMAT_INVALID", "injection manifest must contain injections[]")
    return document


def copy_input_tree(source: str | Path, target: str | Path) -> Path:
    source_path = Path(source)
    target_path = Path(target)
    if target_path.exists():
        shutil.rmtree(target_path)
    shutil.copytree(source_path, target_path)
    return target_path


def _rewrite_json(path: Path, document: Any) -> None:
    atomic_write_json(path, document)


def _rewrite_jsonl(path: Path, rows: list[Any]) -> None:
    lines = []
    for row in rows:
        import json

        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_input_injection(
    base_input: str | Path,
    scenario_input: str | Path,
    injection: dict[str, Any],
) -> dict[str, Any]:
    """Copy the real input package and mutate one documented field in place."""

    scenario = copy_input_tree(base_input, scenario_input)
    kind = injection.get("kind")
    injection_id = injection.get("id")
    if not isinstance(injection_id, str) or not injection_id:
        raise QuantCalError("E_FORMAT_INVALID", "injection id must be non-empty")
    if kind == "nonfinite_sample":
        samples_path = scenario / str(injection.get("input_file", "samples.jsonl"))
        rows = load_jsonl(samples_path)
        sample_id = injection.get("jsonl_sample_id")
        tensor = injection.get("tensor")
        element_index = int(injection.get("element_index", 0))
        changed = False
        for row in rows:
            if row.get("sample_id") != sample_id:
                continue
            values = row.get("features", {}).get(tensor)
            if not isinstance(values, list) or not (0 <= element_index < len(values)):
                break
            values[element_index] = str(injection.get("value", "NaN"))
            changed = True
            break
        if not changed:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"non-finite injection target not found: {sample_id}/{tensor}[{element_index}]",
            )
        _rewrite_jsonl(samples_path, rows)
        hit = {
            "injection_id": injection_id,
            "kind": kind,
            "path": str(samples_path.resolve()),
            "json_pointer": f"/samples/{sample_id}/features/{tensor}/{element_index}",
            "expected_code": injection.get("expected_code", "E_NON_FINITE_SAMPLE"),
            "applied": True,
        }
    elif kind in {"quantile_bounds_inverted", "quantization_bits_out_of_range"}:
        config_path = scenario / str(injection.get("input_file", "config.json"))
        config = load_json(config_path)
        tensor = injection.get("tensor")
        if not isinstance(tensor, str):
            raise QuantCalError("E_FORMAT_INVALID", "config injection requires tensor")
        overrides = config.setdefault("tensor_overrides", {})
        target = overrides.setdefault(tensor, {})
        if kind == "quantile_bounds_inverted":
            target["quantile_lower_permille"] = int(injection.get("lower_permille", 900))
            target["quantile_upper_permille"] = int(injection.get("upper_permille", 100))
            pointer = f"/tensor_overrides/{tensor}/quantile_bounds"
            expected = injection.get("expected_code", "E_QUANT_RANGE_INVALID")
        else:
            target["quantization_bits"] = int(injection.get("quantization_bits", 1))
            pointer = f"/tensor_overrides/{tensor}/quantization_bits"
            expected = injection.get("expected_code", "E_QUANT_RANGE_INVALID")
        _rewrite_json(config_path, config)
        hit = {
            "injection_id": injection_id,
            "kind": kind,
            "path": str(config_path.resolve()),
            "json_pointer": pointer,
            "expected_code": expected,
            "applied": True,
        }
    elif kind == "snapshot_version_tamper":
        raise QuantCalError(
            "E_FORMAT_INVALID",
            "snapshot_version_tamper is applied after a valid run, not to an input copy",
        )
    else:
        raise QuantCalError("E_FORMAT_INVALID", f"unsupported injection kind {kind!r}")
    return hit


def tamper_snapshot_version(work_dir: str | Path) -> dict[str, Any]:
    snapshot_path = Path(work_dir) / "state" / "snapshot.json"
    snapshot = load_json(snapshot_path)
    original = snapshot.get("global_version")
    snapshot["global_version"] = int(original) + 1000
    # Deliberately do not refresh manifest_hash: startup must detect this as stale.
    _rewrite_json(snapshot_path, snapshot)
    return {
        "path": str(snapshot_path.resolve()),
        "json_pointer": "/global_version",
        "original": original,
        "tampered": snapshot["global_version"],
        "manifest_hash_preserved": True,
    }
