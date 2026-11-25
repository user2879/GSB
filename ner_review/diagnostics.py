"""Isolated fault injection and locatable diagnostic reports."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .codes import Code, MESSAGES
from .io_utils import read_json, write_json
from .validation import load_package, parse_corpus


EXPECTED_CODES = {
    "truncate_package": Code.PACKAGE_TRUNCATED,
    "bad_tag_length": Code.TAG_LENGTH_MISMATCH,
    "unknown_tag": Code.UNKNOWN_TAG,
    "duplicate_id": Code.DUPLICATE_SAMPLE_ID,
    "invalid_bieso": Code.INVALID_BIESO_SPAN,
    "feature_dim_invalid": Code.INVALID_FEATURE_DIM,
    "feature_dim_out_of_range": Code.FEATURE_DIM_OUT_OF_RANGE,
    "non_finite_weight": Code.NON_FINITE_WEIGHT,
}


def _write_case_file(case_dir: Path, name: str, text: str) -> Path:
    path = case_dir / name
    path.write_text(text, encoding="utf-8")
    return path


def run_injection_suite(
    corpus_path: str | Path,
    package_path: str | Path,
    manifest_path: str | Path | dict[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    corpus_rows = [json.loads(line) for line in Path(corpus_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    package = read_json(package_path)
    manifest = read_json(manifest_path) if isinstance(manifest_path, (str, Path)) else manifest_path
    injections = manifest["injections"] if isinstance(manifest, dict) else manifest
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="ner-inject-", dir=output_root))

    results: list[dict[str, Any]] = []
    code_counts: dict[str, int] = {}
    for injection in injections:
        injection_id = injection["id"]
        kind = injection["kind"]
        case_dir = temporary_root / injection_id
        case_dir.mkdir(parents=True, exist_ok=True)
        expected_code = EXPECTED_CODES.get(kind, Code.INVALID_PACKAGE)
        record: dict[str, Any] = {
            "injection_id": injection_id,
            "kind": kind,
            "expected_code": expected_code,
            "isolated_directory": str(case_dir),
        }
        try:
            if kind == "truncate_package":
                raw = Path(package_path).read_text(encoding="utf-8")
                retained = int(injection.get("retain_bytes", injection.get("truncate_bytes", 96)))
                target = _write_case_file(case_dir, "truncated_package.json", raw[:retained])
                loaded = load_package(target)
                failures = loaded["failures"]
                record["failure"] = failures[0] if failures else None
                record["ok"] = bool(failures and failures[0]["code"] == expected_code)

            elif kind in {"bad_tag_length", "unknown_tag", "duplicate_id", "invalid_bieso"}:
                rows = [dict(row) for row in corpus_rows]
                index = int(injection.get("sample_index", 0))
                if kind == "bad_tag_length":
                    labels = rows[index]["label_sequence"].split()
                    rows[index]["label_sequence"] = " ".join(labels[:-1])
                elif kind == "unknown_tag":
                    labels = rows[index]["label_sequence"].split()
                    labels[min(2, len(labels) - 1)] = "X-UNKNOWN"
                    rows[index]["label_sequence"] = " ".join(labels)
                elif kind == "duplicate_id":
                    rows[index]["id"] = rows[0]["id"]
                elif kind == "invalid_bieso":
                    labels = rows[index]["label_sequence"].split()
                    if labels:
                        labels[0] = "I-PER"
                    rows[index]["label_sequence"] = " ".join(labels)
                target = case_dir / "injected_corpus.jsonl"
                target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
                parsed = parse_corpus(target)
                matching = [failure for failure in parsed["failures"] if failure["code"] == expected_code]
                record["failure"] = matching[0] if matching else (parsed["failures"][0] if parsed["failures"] else None)
                record["valid_sample_count"] = len(parsed["samples"])
                record["total_sample_count"] = len(rows)
                record["isolation_continued_other_samples"] = len(parsed["samples"]) >= len(rows) - 1
                record["ok"] = bool(matching)

            elif kind == "feature_dim_invalid":
                mutated = dict(package)
                mutated["feature_dim"] = injection.get("feature_dim", 0)
                target = case_dir / "feature_dim_zero.json"
                write_json(target, mutated)
                loaded = load_package(target)
                record["failure"] = loaded["failures"][0] if loaded["failures"] else None
                record["ok"] = bool(loaded["failures"] and loaded["failures"][0]["code"] == expected_code)

            elif kind == "feature_dim_out_of_range":
                mutated = dict(package)
                mutated["feature_dim"] = int(injection.get("feature_dim", 999))
                target = case_dir / "feature_dim_oob.json"
                write_json(target, mutated)
                loaded = load_package(target)
                record["failure"] = loaded["failures"][0] if loaded["failures"] else None
                record["ok"] = bool(loaded["failures"] and loaded["failures"][0]["code"] == expected_code)

            elif kind == "non_finite_weight":
                mutated = json.loads(json.dumps(package))
                mutated["emission_weights"][0][0][0] = float("nan")
                target = case_dir / "nonfinite_package.json"
                with target.open("w", encoding="utf-8") as handle:
                    json.dump(mutated, handle, ensure_ascii=False, allow_nan=True)
                loaded = load_package(target)
                record["failure"] = loaded["failures"][0] if loaded["failures"] else None
                record["ok"] = bool(loaded["failures"] and loaded["failures"][0]["code"] == expected_code)
            else:
                record["ok"] = False
                record["failure"] = {"code": Code.UNKNOWN_COMMAND, "message": f"未登记注入类型：{kind}"}

            actual_code = (record.get("failure") or {}).get("code", Code.OK)
            record["actual_code"] = actual_code
            record["diagnosis"] = MESSAGES.get(actual_code, "未得到预期故障")
            code_counts[actual_code] = code_counts.get(actual_code, 0) + 1
        except Exception as exc:
            record["ok"] = False
            record["actual_code"] = Code.INVALID_PACKAGE
            record["failure"] = {"code": Code.INVALID_PACKAGE, "message": f"{type(exc).__name__}: {exc}"}
            code_counts[Code.INVALID_PACKAGE] = code_counts.get(Code.INVALID_PACKAGE, 0) + 1
        results.append(record)

    report = {
        "ok": all(item.get("ok") for item in results),
        "injection_count": len(results),
        "passed_count": sum(1 for item in results if item.get("ok")),
        "business_code_counts": dict(sorted(code_counts.items())),
        "temporary_root": str(temporary_root),
        "cases": results,
    }
    write_json(output_root / "injection_report.json", report)
    return report


def remove_temporary_root(report: dict[str, Any]) -> None:
    path = report.get("temporary_root")
    if path:
        shutil.rmtree(path, ignore_errors=True)
