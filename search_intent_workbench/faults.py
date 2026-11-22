"""Run isolated fault-injection artifacts without modifying business source."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import read_json
from .loader import (
    load_config,
    load_rules,
    validate_query_records,
)
from .canonical import read_jsonl_strict
from .pipeline import Workbench


def _resolve(base: Path, relative: str) -> Path:
    return (base / relative).resolve()


def run_fault_suite(injection_file: str | Path) -> dict[str, Any]:
    injection_path = Path(injection_file).resolve()
    base = injection_path.parent
    suite = read_json(injection_path)
    injections = suite.get("items", []) if isinstance(suite, dict) else []
    results: list[dict[str, Any]] = []
    for item in injections:
        injection_id = item.get("injection_id")
        kind = item.get("kind")
        rules, rule_issues = load_rules(_resolve(base, item["rules_path"]))
        config, config_issues = load_config(_resolve(base, item["config_path"]), rules)
        rows, line_errors = read_jsonl_strict(_resolve(base, item["queries_path"]))
        vocabulary = set(config.get("vocabulary", []))
        intent_names = {
            intent["name"]
            for intent in config.get("intent_categories", [])
            if isinstance(intent, dict) and isinstance(intent.get("name"), str)
        }
        valid_queries, query_failures = validate_query_records(
            rows,
            vocabulary=vocabulary,
            intent_names=intent_names,
            line_errors=line_errors,
        )
        all_errors = rule_issues + config_issues + query_failures
        observed_codes = sorted({error.get("code", "") for error in all_errors})
        expected_codes = list(item.get("expected_codes", []))
        detected = bool(all_errors) and all(
            code in observed_codes for code in expected_codes
        )
        execution: dict[str, Any] = {}
        if not rule_issues and not config_issues:
            workbench = Workbench(config, rules, max_workers=config.get("max_workers", 1))
            run = workbench.run(valid_queries, input_failures=query_failures)
            execution = {
                "valid_query_count": len(valid_queries),
                "processed_query_count": len(run["queries"]),
                "run_failure_count": len(run["failures"]),
                "failed_query_ids": sorted(
                    {
                        failure["query_id"]
                        for failure in run["failures"]
                        if "query_id" in failure
                    }
                ),
            }
        else:
            execution = {
                "valid_query_count": len(valid_queries),
                "processed_query_count": 0,
                "run_failure_count": len(all_errors),
                "failed_query_ids": [],
                "skipped_reason": "configuration rejected before business execution",
            }
        results.append(
            {
                "injection_id": injection_id,
                "kind": kind,
                "detected": detected,
                "expected_codes": expected_codes,
                "observed_codes": observed_codes,
                "error_count": len(all_errors),
                "errors": all_errors,
                "execution": execution,
            }
        )
    hit_count = sum(1 for item in results if item["detected"])
    return {
        "suite_version": suite.get("version") if isinstance(suite, dict) else None,
        "summary": {
            "injection_count": len(results),
            "hit_count": hit_count,
            "miss_count": len(results) - hit_count,
            "all_detected": hit_count == len(results) and bool(results),
        },
        "injections": results,
    }
