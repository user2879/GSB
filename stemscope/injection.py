from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any

from .package import parse_package
from .v3codec import HEADER_SIZE, RECORD_SIZE


def load_injection_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _patch_bytes(path: Path, offset: int, data_hex: str) -> None:
    data = bytearray(path.read_bytes())
    patch = bytes.fromhex(data_hex)
    if offset < 0 or offset + len(patch) > len(data):
        raise ValueError(f"byte patch at {offset}+{len(patch)} exceeds {path}")
    data[offset : offset + len(patch)] = patch
    path.write_bytes(bytes(data))


def _patch_v3_offset(path: Path, record_index: int, offset_value: int) -> None:
    data = bytearray(path.read_bytes())
    record_start = HEADER_SIZE + RECORD_SIZE * record_index
    if record_start + RECORD_SIZE > len(data):
        raise ValueError("v3 record index is out of range")
    offset_field = record_start + 20
    struct.pack_into("<I", data, offset_field, offset_value)
    path.write_bytes(bytes(data))


def _patch_json(path: Path, json_path: list[Any], *, value: Any = None, value_delta: Any = None) -> None:
    value_obj = json.loads(path.read_text(encoding="utf-8"))
    target = value_obj
    for key in json_path[:-1]:
        target = target[key]
    final_key = json_path[-1]
    if value_delta is not None:
        target[final_key] = target[final_key] + value_delta
    else:
        target[final_key] = value
    path.write_text(json.dumps(value_obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def apply_injection_case(
    sample_root: str | Path,
    destination_root: str | Path,
    case: dict[str, Any],
) -> Path:
    source = Path(sample_root).resolve()
    destination = Path(destination_root).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    target = destination / str(case["path"])
    operation = case["operation"]
    if operation == "patch_bytes":
        _patch_bytes(target, int(case["offset"]), str(case["data_hex"]))
    elif operation == "patch_v3_record_offset":
        _patch_v3_offset(target, int(case["record_index"]), int(case["offset"]))
    elif operation == "patch_json":
        _patch_json(
            target,
            list(case["json_path"]),
            value=case.get("value"),
            value_delta=case.get("value_delta"),
        )
    else:
        raise ValueError(f"unknown injection operation {operation!r}")
    return destination


def run_injection_case(
    sample_root: str | Path,
    working_root: str | Path,
    case: dict[str, Any],
) -> dict[str, Any]:
    destination = apply_injection_case(sample_root, working_root, case)
    clip_id = str(case.get("clip_id", "seg_000"))
    format_version = str(case["format"])
    result = parse_package(destination / "segments" / clip_id, format_version, clip_id=clip_id)
    failures = [error.to_dict() for error in result.failures]
    expected = str(case.get("expected_code", "E_TRACK_PARSE_FAILED"))
    passed = bool(failures) and all(item["code"] == expected for item in failures)
    return {
        "case_id": case.get("id", ""),
        "format_version": format_version,
        "clip_id": clip_id,
        "operation": case["operation"],
        "expected_code": expected,
        "observed_codes": [item["code"] for item in failures],
        "passed": passed,
        "failures": failures,
        "working_directory": str(destination),
    }
