from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import Code
from .utils import is_real_int, read_json


DEFAULT_PORT_RANGE = [43000, 43999]

WORKER_FIELDS = {
    "worker_id",
    "host",
    "port",
    "model_dim",
    "hidden_dim",
    "quota",
    "param_versions",
    "current_param_version",
}
BATCH_FIELDS = {"batch_id", "worker_id", "batch_size", "param_version", "latency_seed"}
REQUEST_FIELDS = {"seq", "request_id", "arrived_ms", "batch_id", "worker_id", "output_digest"}
RULE_FIELDS = {"rules_id", "rule_version", "allowed_workers", "worker_sets", "policy", "budgets", "resource"}

COMMAND_FIELDS: dict[str, set[str]] = {
    "set_param_version": {"command_id", "op", "worker_id", "param_version", "param_digest"},
    "replace_request": {"command_id", "op", "request_id", "request"},
    "switch_rules": {"command_id", "op", "rules"},
    "grant_budget": {"command_id", "op", "batch_budget", "worker_quota"},
    "normalize": {"command_id", "op", "reason"},
    "checkpoint": {"command_id", "op"},
}


def failure(
    source: str,
    message: str,
    *,
    index: int | None = None,
    item_id: str | None = None,
    field: str | None = None,
    code: str = Code.SCHEMA_INVALID,
    injection_id: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "source": source, "message": message}
    if index is not None:
        result["index"] = index
    if item_id is not None:
        result["item_id"] = item_id
    if field is not None:
        result["field"] = field
    if injection_id is not None:
        result["injection_id"] = injection_id
    return result


def load_json_lenient(path: Path, failures: list[dict[str, Any]]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(failure(str(path), "input file not found"))
    except json.JSONDecodeError as exc:
        failures.append(failure(str(path), f"JSON parse error: {exc.msg}", field=f"line {exc.lineno}"))
    return None


def _unknown_fields(record: dict[str, Any], allowed: set[str], source: str, index: int,
                    failures: list[dict[str, Any]]) -> bool:
    unknown = sorted(set(record) - allowed)
    if unknown:
        failures.append(failure(
            source,
            f"unknown field(s): {', '.join(unknown)}",
            index=index,
            item_id=str(record.get("id") or record.get("worker_id") or record.get("request_id") or ""),
            field=",".join(unknown),
        ))
        return True
    return False


def _positive_dim(value: Any) -> bool:
    return is_real_int(value) and value > 0


def validate_workers(payload: Any, port_range: list[int] | None = None,
                      source: str = "workers.json") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    if not isinstance(payload, dict) or not isinstance(payload.get("workers"), list):
        failures.append(failure(source, "must be an object with a workers list"))
        return workers, failures
    if set(payload) - {"schema_version", "workers", "allowed_port_range", "seed"}:
        failures.append(failure(source, f"unknown top-level field(s): {sorted(set(payload) - {'schema_version','workers','allowed_port_range','seed'})}"))
    allowed_range = port_range or payload.get("allowed_port_range") or DEFAULT_PORT_RANGE
    seen: set[str] = set()
    for index, raw in enumerate(payload["workers"]):
        if not isinstance(raw, dict):
            failures.append(failure(source, "worker entry must be an object", index=index))
            continue
        failure_count_before = len(failures)
        _unknown_fields(raw, WORKER_FIELDS, source, index, failures)
        worker_id = raw.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            failures.append(failure(source, "worker_id must be a non-empty string", index=index, field="worker_id"))
            continue
        if worker_id in seen:
            failures.append(failure(source, "duplicate worker identifier", index=index, item_id=worker_id, field="worker_id"))
            continue
        host = raw.get("host", "127.0.0.1")
        port = raw.get("port")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            failures.append(failure(source, "host is not a loopback address", index=index, item_id=worker_id, field="host"))
        if not is_real_int(port) or not (allowed_range[0] <= port <= allowed_range[1]):
            failures.append(failure(
                source,
                f"port must be an integer in [{allowed_range[0]}, {allowed_range[1]}]",
                index=index, item_id=worker_id, field="port",
            ))
        model_dim = raw.get("model_dim")
        hidden_dim = raw.get("hidden_dim")
        for field_name, value in (("model_dim", model_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, float) and not _finite(value):
                failures.append(failure(source, "dimension must be finite", index=index, item_id=worker_id, field=field_name))
            elif not _positive_dim(value):
                failures.append(failure(source, "dimension must be a positive integer", index=index, item_id=worker_id, field=field_name))
        quota = raw.get("quota", 4)
        if not _positive_dim(quota):
            failures.append(failure(source, "quota must be a positive integer", index=index, item_id=worker_id, field="quota"))
        versions = raw.get("param_versions", [1])
        if not isinstance(versions, list) or not versions or not all(_positive_dim(v) for v in versions):
            failures.append(failure(source, "param_versions must be a non-empty list of positive integers", index=index, item_id=worker_id, field="param_versions"))
        current = raw.get("current_param_version", versions[0] if isinstance(versions, list) and versions else 1)
        if not _positive_dim(current) or (isinstance(versions, list) and current not in versions):
            failures.append(failure(source, "current_param_version must be one of param_versions", index=index, item_id=worker_id, field="current_param_version"))
        if len(failures) > failure_count_before:
            continue
        seen.add(worker_id)
        workers.append({
            "worker_id": worker_id,
            "host": host,
            "port": port,
            "model_dim": model_dim,
            "hidden_dim": hidden_dim,
            "quota": quota if _positive_dim(quota) else 4,
            "param_versions": list(versions) if isinstance(versions, list) and all(_positive_dim(v) for v in versions) else [1],
            "current_param_version": current if _positive_dim(current) else 1,
        })
    return workers, failures


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def validate_request_stream(payload: Any, worker_ids: set[str],
                             source: str = "request_stream.json"
                             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        failures.append(failure(source, "must be an object"))
        return requests, batches, failures
    unknown_top = set(payload) - {"schema_version", "requests", "batches", "seed"}
    if unknown_top:
        failures.append(failure(source, f"unknown top-level field(s): {sorted(unknown_top)}"))

    seen_batch: set[tuple[str, str]] = set()
    for index, raw in enumerate(payload.get("batches", [])):
        if not isinstance(raw, dict):
            failures.append(failure(source, "batch entry must be an object", index=index))
            continue
        failure_count_before = len(failures)
        _unknown_fields(raw, BATCH_FIELDS, f"{source}#batches", index, failures)
        batch_id = raw.get("batch_id")
        worker_id = raw.get("worker_id")
        batch_size = raw.get("batch_size")
        key = (str(batch_id), str(worker_id))
        if not isinstance(batch_id, str) or not batch_id:
            failures.append(failure(source, "batch_id must be non-empty string", index=index, field="batch_id"))
        if worker_id not in worker_ids:
            failures.append(failure(source, "batch references unknown worker", index=index, item_id=str(batch_id), field="worker_id"))
        if key in seen_batch:
            failures.append(failure(source, "duplicate batch identifier for worker", index=index, item_id=str(batch_id), field="batch_id"))
        if not is_real_int(batch_size) or batch_size <= 0:
            failures.append(failure(source, "batch_size must be a positive integer; negative sizes are invalid", index=index, item_id=str(batch_id), field="batch_size"))
        param_version = raw.get("param_version", 1)
        if not _positive_dim(param_version):
            failures.append(failure(source, "param_version must be a positive integer", index=index, item_id=str(batch_id), field="param_version"))
        if len(failures) == failure_count_before:
            seen_batch.add(key)
            batches.append({
                "batch_id": batch_id,
                "worker_id": worker_id,
                "batch_size": batch_size,
                "param_version": param_version if _positive_dim(param_version) else 1,
                "latency_seed": raw.get("latency_seed", 0),
            })

    previous_seq: int | None = None
    seen_request_ids: set[str] = set()
    batch_keys = {(b["batch_id"], b["worker_id"]) for b in batches}
    for index, raw in enumerate(payload.get("requests", [])):
        if not isinstance(raw, dict):
            failures.append(failure(source, "request entry must be an object", index=index))
            continue
        failure_count_before = len(failures)
        _unknown_fields(raw, REQUEST_FIELDS, f"{source}#requests", index, failures)
        seq = raw.get("seq")
        request_id = raw.get("request_id")
        arrived_ms = raw.get("arrived_ms")
        batch_id = raw.get("batch_id")
        worker_id = raw.get("worker_id")
        if not is_real_int(seq) or seq <= 0:
            failures.append(failure(source, "seq must be a positive integer", index=index, item_id=str(request_id), field="seq"))
        elif previous_seq is not None and seq <= previous_seq:
            failures.append(failure(source, "request seq must be strictly increasing", index=index, item_id=str(request_id), field="seq"))
        else:
            previous_seq = seq
        if not isinstance(request_id, str) or not request_id:
            failures.append(failure(source, "request_id must be a non-empty string", index=index, field="request_id"))
        elif request_id in seen_request_ids:
            failures.append(failure(source, "duplicate request identifier", index=index, item_id=request_id, field="request_id"))
        else:
            seen_request_ids.add(request_id)
        if not is_real_int(arrived_ms) or arrived_ms < 0:
            failures.append(failure(source, "arrived_ms must be a non-negative integer", index=index, item_id=str(request_id), field="arrived_ms"))
        if worker_id not in worker_ids:
            failures.append(failure(source, "request references unknown worker", index=index, item_id=str(request_id), field="worker_id"))
        if not isinstance(batch_id, str) or not batch_id:
            failures.append(failure(source, "batch_id must be a non-empty string", index=index, item_id=str(request_id), field="batch_id"))
        elif (batch_id, worker_id) not in batch_keys:
            failures.append(failure(source, "request references unknown batch/worker pair", index=index, item_id=request_id, field="batch_id"))
        if len(failures) == failure_count_before:
            requests.append({
                "seq": seq,
                "request_id": request_id,
                "arrived_ms": arrived_ms,
                "batch_id": batch_id,
                "worker_id": worker_id,
                "output_digest": raw.get("output_digest", ""),
            })
    return requests, batches, failures


def validate_rules(payload: Any, worker_ids: set[str], source: str = "rules.json"
                   ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return None, [failure(source, "rules must be an object")]
    if _unknown_fields(payload, RULE_FIELDS, source, 0, failures):
        return None, failures
    rules_id = payload.get("rules_id")
    if not isinstance(rules_id, str) or not rules_id:
        failures.append(failure(source, "rules_id must be a non-empty string", field="rules_id"))
    allowed = payload.get("allowed_workers", sorted(worker_ids))
    if not isinstance(allowed, list) or not all(isinstance(x, str) for x in allowed):
        failures.append(failure(source, "allowed_workers must be a list of worker ids", field="allowed_workers"))
        allowed = []
    missing = sorted(set(allowed) - worker_ids)
    if missing:
        failures.append(failure(source, f"rule references unknown worker(s): {', '.join(missing)}", field="allowed_workers"))
    worker_sets = payload.get("worker_sets", {})
    if not isinstance(worker_sets, dict):
        failures.append(failure(source, "worker_sets must be an object of worker id lists", field="worker_sets"))
        worker_sets = {}
    for set_name, ids in worker_sets.items():
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            failures.append(failure(source, f"worker set {set_name} must be a list", field=f"worker_sets.{set_name}"))
            continue
        missing_set = sorted(set(ids) - worker_ids)
        if missing_set:
            failures.append(failure(source, f"worker set {set_name} references unknown worker(s): {', '.join(missing_set)}", item_id=set_name, field=f"worker_sets.{set_name}"))
    if failures:
        return None, failures
    return payload, []


def validate_command(raw: Any, index: int, source: str = "commands.json"
                     ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(raw, dict):
        return None, failure(source, "command must be an object", index=index)
    op = raw.get("op")
    if op not in COMMAND_FIELDS:
        return None, failure(source, f"unknown command op: {op!r}", index=index, item_id=str(raw.get("command_id", "")), field="op")
    unknown = sorted(set(raw) - COMMAND_FIELDS[op])
    if unknown:
        return None, failure(source, f"unknown command field(s): {', '.join(unknown)}", index=index,
                             item_id=str(raw.get("command_id", "")), field=",".join(unknown))
    command_id = raw.get("command_id")
    if not isinstance(command_id, str) or not command_id:
        return None, failure(source, "command_id must be a non-empty string", index=index, field="command_id")
    return dict(raw), None


def load_inputs(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    failures: list[dict[str, Any]] = []
    workers_payload = load_json_lenient(root / "workers.json", failures)
    workers, worker_failures = validate_workers(workers_payload)
    failures.extend(worker_failures)
    worker_ids = {w["worker_id"] for w in workers}

    stream_payload = load_json_lenient(root / "request_stream.json", failures)
    requests, batches, stream_failures = validate_request_stream(stream_payload, worker_ids)
    failures.extend(stream_failures)

    rules_payload = load_json_lenient(root / "rules.json", failures)
    rules, rule_failures = validate_rules(rules_payload, worker_ids)
    failures.extend(rule_failures)

    commands_payload = load_json_lenient(root / "commands.json", failures)
    commands: list[dict[str, Any]] = []
    if isinstance(commands_payload, dict) and isinstance(commands_payload.get("commands"), list):
        seen_command_ids: set[str] = set()
        for index, raw in enumerate(commands_payload["commands"]):
            command, command_failure = validate_command(raw, index)
            if command_failure:
                failures.append(command_failure)
                continue
            if command["command_id"] in seen_command_ids:
                failures.append(failure("commands.json", "duplicate command identifier", index=index,
                                        item_id=command["command_id"], field="command_id"))
                continue
            seen_command_ids.add(command["command_id"])
            commands.append(command)
    elif commands_payload is not None:
        failures.append(failure("commands.json", "must be an object with commands list"))

    return {
        "data_dir": str(root),
        "workers": workers,
        "requests": requests,
        "batches": batches,
        "rules": rules or {},
        "commands": commands,
        "failures": failures,
    }


def config_fingerprint(inputs: dict[str, Any]) -> dict[str, Any]:
    """Return the stable facts which require a format/config rebuild."""
    return {
        "workers": sorted(inputs.get("workers", []), key=lambda w: w["worker_id"]),
        "batches": sorted(inputs.get("batches", []), key=lambda b: (b["batch_id"], b["worker_id"])),
        "requests": sorted(inputs.get("requests", []), key=lambda r: r["seq"]),
        "rules": inputs.get("rules", {}),
    }
