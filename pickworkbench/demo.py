"""Repeatable demo bundle: fixed samples, operations, injections, and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, content_hash, pretty_json
from .demo_cases import build_demo_cases
from .engine import execute_case
from .generator import GeneratorConfig, generate_sample_with_config, summarize_scene


def _case_index_report(report: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    return {
        "name": report["case"],
        "status": report["status"],
        "expected_error_codes": report["expected_error_codes"],
        "detected_error_codes": report["detected_error_codes"],
        "scene_hash": report["scene_hash"],
        "queries_hash": report["queries_hash"],
        "injection_hash": report["injection_hash"],
        "records_hash": report.get("records_hash"),
        "case_report_hash": report["content_hash"],
        "summary": report["summary"],
        "first_affected": None
        if report["first_affected"] is None
        else {
            "query_id": report["first_affected"]["query_id"],
            "query_index": report["first_affected"]["query_index"],
            "error_code": report["first_affected"]["error_code"],
            "message": report["first_affected"]["message"],
        },
        "witness_minimal": None if report.get("witness") is None else report["witness"]["minimal"],
        "query_hashes": report["query_hashes"],
        "files": files,
    }


def build_demo_files(config: GeneratorConfig | None = None) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Return every artifact as bytes plus the top-level report object."""

    config = config or GeneratorConfig()
    scene_a, queries_a = generate_sample_with_config(config)
    scene_b, queries_b = generate_sample_with_config(config)
    generation_identical = canonical_bytes(scene_a) == canonical_bytes(scene_b) and canonical_bytes(
        queries_a
    ) == canonical_bytes(queries_b)
    if not generation_identical:
        raise RuntimeError("Two independent generations were not byte-identical")

    cases = build_demo_cases(scene_a, queries_a)
    files: dict[str, bytes] = {}
    case_indexes: list[dict[str, Any]] = []

    for case in cases:
        name = str(case["name"])
        base = f"cases/{name}"
        report = execute_case(
            name,
            scene_a,
            queries_a,
            case["injection"],
            case["expected_error_codes"],
        )
        records = report.get("records", [])
        witness = report.get("witness")
        file_roles = {
            "scene": f"{base}/scene.json",
            "queries": f"{base}/queries.json",
            "injection": f"{base}/injection.json",
            "records": f"{base}/records.json",
            "report": f"{base}/report.json",
        }
        if witness is not None:
            file_roles["witness"] = f"{base}/witness.json"
        files[file_roles["scene"]] = pretty_json(scene_a).encode("utf-8")
        files[file_roles["queries"]] = pretty_json(queries_a).encode("utf-8")
        files[file_roles["injection"]] = pretty_json(case["injection"]).encode("utf-8")
        files[file_roles["records"]] = pretty_json(records).encode("utf-8")
        files[file_roles["report"]] = pretty_json(report).encode("utf-8")
        if witness is not None:
            files[file_roles["witness"]] = pretty_json(witness).encode("utf-8")
        case_indexes.append(_case_index_report(report, file_roles))

    top_payload = {
        "schema_version": 1,
        "demo": "pickworkbench-deterministic-demo",
        "generator_config": config.as_dict(),
        "source_summary": summarize_scene(scene_a, queries_a),
        "generation_repeat_check": {
            "independent_generations_identical": generation_identical,
            "scene_hash": content_hash(scene_a),
            "queries_hash": content_hash(queries_a),
        },
        "case_order": [case["name"] for case in cases],
        "cases": case_indexes,
        "overall_hash": content_hash(
            {
                "source_summary": summarize_scene(scene_a, queries_a),
                "cases": case_indexes,
            }
        ),
    }
    files["report.json"] = pretty_json(top_payload).encode("utf-8")
    files["manifest.json"] = pretty_json(
        {
            "schema_version": 1,
            "files": sorted(files, key=lambda item: (item != "report.json", item)),
            "overall_hash": top_payload["overall_hash"],
        }
    ).encode("utf-8")
    return files, top_payload


def write_demo_files(target: str | Path, files: dict[str, bytes]) -> None:
    root = Path(target)
    for relative, payload in sorted(files.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def verify_bundle_repeat(config: GeneratorConfig | None = None) -> tuple[dict[str, bytes], dict[str, Any]]:
    first_files, first_report = build_demo_files(config)
    second_files, second_report = build_demo_files(config)
    if first_files != second_files:
        different = sorted(
            key for key in set(first_files) | set(second_files) if first_files.get(key) != second_files.get(key)
        )
        raise RuntimeError(f"Demo bundle is not byte-repeatable: {different}")
    return first_files, first_report
