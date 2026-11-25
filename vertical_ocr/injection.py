from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .codes import INPUT_INVALID, TEMPLATE_VERSION_UNKNOWN
from .config import load_config
from .ledger import atomic_write_json
from .pipeline import Pipeline


def default_injection_config() -> dict[str, Any]:
    return {
        "schema": "vertical-ocr-injection/v1",
        "injections": [
            {
                "id": "INJ-PNG-001",
                "kind": "truncated_png",
                "image": "page_002.png",
                "stage": "load_validation",
                "expected_code": INPUT_INVALID,
            },
            {
                "id": "INJ-KP-002",
                "kind": "non_integer_keypoint",
                "image": "page_001.png",
                "stage": "load_validation",
                "expected_code": INPUT_INVALID,
            },
            {
                "id": "INJ-SEQ-003",
                "kind": "reading_number_gap",
                "image": "page_002.png",
                "stage": "load_validation",
                "expected_code": INPUT_INVALID,
            },
            {
                "id": "INJ-WIN-004",
                "kind": "window_not_divisible",
                "stage": "config_validation",
                "expected_code": INPUT_INVALID,
            },
            {
                "id": "INJ-FONT-005",
                "kind": "unknown_template_version",
                "stage": "template_matching",
                "expected_code": TEMPLATE_VERSION_UNKNOWN,
            },
        ],
    }


def write_injection_config(root: str | Path) -> Path:
    path = Path(root) / "config" / "injection.json"
    atomic_write_json(path, default_injection_config())
    return path


def _copy_baseline(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(source / "input", target / "input")
    shutil.copytree(source / "config", target / "config")
    for directory in ("ledger", "reports", "exchange", "snapshots"):
        (target / directory).mkdir()


def apply_injection(case_root: Path, injection: dict[str, Any]) -> dict[str, Any]:
    kind = injection["kind"]
    applied = {"kind": kind}
    if kind == "truncated_png":
        image_path = case_root / "input" / injection["image"]
        data = image_path.read_bytes()
        image_path.write_bytes(data[: max(64, len(data) // 2)])
        applied["bytes_written"] = image_path.stat().st_size
    elif kind == "non_integer_keypoint":
        path = case_root / "input" / "keypoint_versions.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for sub in data["submissions"]:
            if sub.get("image") == injection.get("image"):
                sub["x"] = 12.5
                break
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    elif kind == "reading_number_gap":
        path = case_root / "input" / "layout_truth.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for image in data["images"]:
            if image["image"] == injection.get("image"):
                block = image["columns"][0]["blocks"][1]
                block["reading_no"] = int(block["reading_no"]) + 5
                applied["block_id"] = block["block_id"]
                break
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    elif kind == "window_not_divisible":
        path = case_root / "config" / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["sliding_window"]["width"] = 13
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    elif kind == "unknown_template_version":
        path = case_root / "config" / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["template_version"] = "dot-matrix-v999-unknown"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"unknown injection kind {kind}")
    return applied


def run_injection_suite(output_root: str | Path, baseline_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root)
    baseline_root = Path(baseline_root)
    output_root.mkdir(parents=True, exist_ok=True)
    injection_config_path = baseline_root / "config" / "injection.json"
    if injection_config_path.exists():
        config = json.loads(injection_config_path.read_text(encoding="utf-8"))
    else:
        config = default_injection_config()
    results = []
    for injection in config["injections"]:
        case_root = output_root / injection["id"]
        _copy_baseline(baseline_root, case_root)
        applied = apply_injection(case_root, injection)
        started = time.monotonic_ns()
        observed_code = None
        report = None
        config_error = None
        try:
            cfg = load_config(case_root / "config" / "config.json")
            report = Pipeline(case_root, cfg).run()
            codes = report.get("business_code_counts", {})
            expected = injection["expected_code"]
            observed_code = expected if codes.get(expected) else next(iter(codes), None)
        except Exception as exc:
            observed_code = getattr(exc, "code", "E_STAGE_EXCEPTION")
            config_error = str(exc)
        elapsed_ms = round((time.monotonic_ns() - started) / 1_000_000, 2)
        expected = injection["expected_code"]
        hit = observed_code == expected
        result = {
            "injection_id": injection["id"],
            "kind": injection["kind"],
            "stage": injection["stage"],
            "expected_code": expected,
            "observed_code": observed_code,
            "hit": hit,
            "elapsed_ms": elapsed_ms,
            "applied": applied,
            "config_error": config_error,
            "isolated_image_reports": (report or {}).get("images", []),
            "business_code_counts": (report or {}).get("business_code_counts", {}),
            "failure_count": len((report or {}).get("failures", [])),
        }
        results.append(result)
    summary = {
        "schema": "vertical-ocr-injection-report/v1",
        "all_hits": all(r["hit"] for r in results),
        "hit_count": sum(1 for r in results if r["hit"]),
        "total": len(results),
        "results": results,
    }
    atomic_write_json(output_root / "injection_report.json", summary)
    return summary
