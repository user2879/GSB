from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import validate_command, validate_request_stream, validate_rules, validate_workers
from .injections import apply_receipt_injection, apply_request_injection, selected_injections
from .utils import sha256_json


def _load_raw_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_workspace(
    data_dir: str | Path,
    *,
    injection_config_path: str | Path | None = None,
    injection_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str], str, list[dict[str, Any]]]:
    root = Path(data_dir)
    failures: list[dict[str, Any]] = []
    raw_workers = _load_raw_json(root / "workers.json")
    raw_stream = _load_raw_json(root / "request_stream.json")
    raw_rules = _load_raw_json(root / "rules.json")
    raw_commands = _load_raw_json(root / "commands.json")

    injection_config: dict[str, Any] = {"injections": []}
    injections: list[dict[str, Any]] = []
    if injection_config_path:
        injection_config = _load_raw_json(Path(injection_config_path))
        injections = selected_injections(injection_config, injection_ids)

    injection_results: list[dict[str, Any]] = []
    for injection in injections:
        result = {
            "id": injection["id"],
            "kind": injection["kind"],
            "expected_code": injection["expected_code"],
            "applied": False,
            "observed": False,
        }
        try:
            if injection["kind"] == "duplicate_request_id":
                raw_stream = apply_request_injection(raw_stream, injection)
            result["applied"] = True
        except Exception as exc:  # injection configuration is local and deterministic
            result["error"] = str(exc)
        injection_results.append(result)

    workers, worker_failures = validate_workers(raw_workers)
    failures.extend(worker_failures)
    worker_ids = {worker["worker_id"] for worker in workers}
    requests, batches, stream_failures = validate_request_stream(raw_stream, worker_ids)
    for injection in injections:
        if injection["kind"] == "duplicate_request_id":
            for item in stream_failures:
                if item.get("item_id") == injection.get("request_id") and item.get("code") == injection["expected_code"]:
                    item["injection_id"] = injection["id"]
    failures.extend(stream_failures)

    rules, rule_failures = validate_rules(raw_rules, worker_ids)
    failures.extend(rule_failures)

    commands: list[dict[str, Any]] = []
    if isinstance(raw_commands, dict) and isinstance(raw_commands.get("commands"), list):
        seen: set[str] = set()
        for index, raw_command in enumerate(raw_commands["commands"]):
            command, command_failure = validate_command(raw_command, index)
            if command_failure:
                failures.append(command_failure)
                continue
            if command["command_id"] in seen:
                from .config import failure
                failures.append(failure("commands.json", "duplicate command identifier", index=index,
                                        item_id=command["command_id"]))
                continue
            seen.add(command["command_id"])
            commands.append(command)

    receipt_dir = root / "receipts"
    receipt_texts: dict[str, str] = {}
    for path in sorted(receipt_dir.glob("*.log")):
        text = path.read_text(encoding="utf-8")
        for injection in injections:
            if injection.get("file") == path.name and injection["kind"] in {"truncate_line", "set_field"}:
                try:
                    text = apply_receipt_injection(text, injection)
                    for result in injection_results:
                        if result["id"] == injection["id"]:
                            result["applied"] = True
                except Exception as exc:
                    for result in injection_results:
                        if result["id"] == injection["id"]:
                            result["error"] = str(exc)
        receipt_texts[path.name] = text

    inputs = {
        "workers": workers,
        "requests": requests,
        "batches": batches,
        "rules": rules or {},
        "commands": commands,
        "failures": failures,
    }
    config_hash = sha256_json({
        "workers": raw_workers,
        "request_stream": raw_stream,
        "rules": raw_rules,
        "commands": raw_commands,
        "receipt_texts": receipt_texts,
    })
    return inputs, receipt_texts, config_hash, injection_results


def mark_injection_observations(injection_results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    codes_by_injection = {
        "INJ-001-truncated-receipt": ("E_RECEIPT_TRUNCATED", None),
        "INJ-002-unknown-param-version": ("E_VERSION_UNKNOWN", None),
        "INJ-003-nonfinite-latency": ("E_RECEIPT_NONFINITE_LATENCY", None),
        "INJ-004-duplicate-request-id": ("E_SCHEMA_INVALID", "REQ-0001"),
    }
    for result in injection_results:
        expected_code, item_id = codes_by_injection.get(result["id"], (result["expected_code"], None))
        matches = [
            failure for failure in failures
            if failure.get("code") == expected_code and (item_id is None or failure.get("item_id") == item_id)
        ]
        result["observed"] = bool(matches)
        result["business_code"] = expected_code
        if matches:
            result["matching_failure_keys"] = sorted({
                f"{failure.get('source')}:{failure.get('index', '')}:{failure.get('item_id', '')}"
                for failure in matches
            })
