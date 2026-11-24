from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    E_EVENT_INVALID,
    E_MANIFEST_INVALID,
    E_NODE_INVALID,
    EVENT_TYPES,
)
from .hashing import fingerprint


class InputValidationError(ValueError):
    pass


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_positive_int(value: Any) -> bool:
    return is_int(value) and value > 0


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_events_file(path: str | Path) -> list[Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        events: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                events.append({
                    "__parse_error__": f"line {line_number}: {exc.msg}",
                    "__line_number__": line_number,
                })
        return events
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("events"), list):
        return data["events"]
    if isinstance(data, list):
        return data
    raise InputValidationError("events file must be JSONL, an array, or an object with events[]")


def _failure(code: str, subject: str, message: str, **extra: Any) -> dict[str, Any]:
    item = {"code": code, "subject": subject, "message": message}
    item.update(extra)
    return item


def normalize_capacity_policies(raw: Any) -> tuple[dict[int, dict[str, int]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    policies: dict[int, dict[str, int]] = {}
    if raw is None:
        raw = [
            {"capacity_version": 1, "soft_permille": 600, "hard_permille": 850},
        ]
    if not isinstance(raw, list) or not raw:
        return {}, [_failure(E_NODE_INVALID, "capacity_policies", "must be a non-empty list")]

    previous_version = 0
    for index, item in enumerate(raw):
        subject = f"capacity_policies[{index}]"
        if not isinstance(item, dict):
            failures.append(_failure(E_NODE_INVALID, subject, "policy must be an object"))
            continue
        version = item.get("capacity_version")
        soft = item.get("soft_permille")
        hard = item.get("hard_permille")
        errors = []
        if not is_positive_int(version):
            errors.append("capacity_version must be a positive integer")
        elif version <= previous_version:
            errors.append("capacity_version values must be strictly increasing")
        if not is_int(soft) or not 0 <= soft <= 1000:
            errors.append("soft_permille must be an integer in [0, 1000]")
        if not is_int(hard) or not 0 <= hard <= 1000:
            errors.append("hard_permille must be an integer in [0, 1000]")
        if not errors and soft > hard:
            errors.append("soft_permille must not exceed hard_permille")
        if errors:
            failures.append(_failure(E_NODE_INVALID, subject, "; ".join(errors)))
            continue
        previous_version = version
        policies[version] = {
            "capacity_version": version,
            "soft_permille": soft,
            "hard_permille": hard,
        }
    return policies, failures


def validate_nodes(
    raw_nodes: Any,
    policies: dict[int, dict[str, int]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    nodes: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_nodes, list):
        return {}, [_failure(E_NODE_INVALID, "nodes", "must be a list")]

    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, raw in enumerate(raw_nodes):
        subject = f"nodes[{index}]"
        if not isinstance(raw, dict):
            failures.append(_failure(E_NODE_INVALID, subject, "node declaration must be an object"))
            continue

        node_id = raw.get("node_id") or raw.get("id")
        port = raw.get("port")
        slots = raw.get("capacity_slots")
        lease_limit = raw.get("lease_limit") or raw.get("max_leases_per_node")
        seed = raw.get("seed")
        capacity_version = raw.get("capacity_version")
        errors: list[str] = []

        if not isinstance(node_id, str) or not node_id:
            errors.append("node_id must be a non-empty string")
        elif node_id in seen_ids:
            errors.append(f"duplicate node_id {node_id!r}")

        if not is_int(port) or not 1 <= port <= 65535:
            errors.append("port must be an integer in [1, 65535]")
        elif port in seen_ports:
            errors.append(f"duplicate loopback port {port}")

        if not is_positive_int(slots):
            errors.append("capacity_slots must be a positive integer")
        if not is_positive_int(lease_limit):
            errors.append("lease_limit must be a positive integer")
        if not is_int(seed):
            errors.append("seed must be an integer")
        if not is_positive_int(capacity_version):
            errors.append("capacity_version must be a positive integer")
        elif capacity_version not in policies:
            errors.append(f"capacity_version {capacity_version} has no declared watermark policy")

        if errors:
            failures.append(_failure(
                E_NODE_INVALID,
                subject,
                "; ".join(errors),
                node_id=node_id if isinstance(node_id, str) else None,
                port=port if is_int(port) else None,
            ))
            continue

        seen_ids.add(node_id)
        seen_ports.add(port)
        nodes[node_id] = {
            "node_id": node_id,
            "host": "127.0.0.1",
            "port": port,
            "capacity_slots": slots,
            "lease_limit": lease_limit,
            "seed": int(seed),
            "capacity_version": capacity_version,
            "quota_version": int(raw.get("quota_version", 1)),
        }
        if not is_positive_int(nodes[node_id]["quota_version"]):
            failures.append(_failure(E_NODE_INVALID, f"{subject}.quota_version", "quota_version must be positive"))
            nodes.pop(node_id, None)
    return nodes, failures


def validate_manifest(raw_manifest: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if raw_manifest is None:
        return {"stages": [], "artifact_index": {}, "stage_index": {}}, failures
    if not isinstance(raw_manifest, dict):
        return None, [_failure(E_MANIFEST_INVALID, "manifest", "must be an object")]
    raw_stages = raw_manifest.get("stages")
    if not isinstance(raw_stages, list):
        return None, [_failure(E_MANIFEST_INVALID, "manifest.stages", "must be a list")]

    stages: list[dict[str, Any]] = []
    artifact_index: dict[str, str] = {}
    stage_index: dict[str, dict[str, Any]] = {}
    for index, raw_stage in enumerate(raw_stages):
        subject = f"manifest.stages[{index}]"
        if not isinstance(raw_stage, dict):
            failures.append(_failure(E_MANIFEST_INVALID, subject, "stage must be an object"))
            continue
        stage_id = raw_stage.get("stage_id") or raw_stage.get("stage")
        artifact_key = raw_stage.get("artifact_key") or raw_stage.get("key")
        needs = raw_stage.get("needs", [])
        upstream_stage_ids = raw_stage.get("upstream_stage_ids", [])
        quota = raw_stage.get("concurrency_quota")
        errors: list[str] = []

        if not isinstance(stage_id, str) or not stage_id:
            errors.append("stage_id must be a non-empty string")
        elif stage_id in stage_index:
            errors.append(f"duplicate stage_id {stage_id!r}")
        if not isinstance(artifact_key, str) or not artifact_key:
            errors.append("artifact_key must be a non-empty string")
        elif artifact_key in artifact_index:
            errors.append(f"duplicate artifact_key {artifact_key!r}")
        if not is_positive_int(quota):
            errors.append("concurrency_quota must be a positive integer")
        if not isinstance(needs, list) or not all(isinstance(item, str) and item for item in needs):
            errors.append("needs must be a list of non-empty artifact key strings")
            needs = []
        if not isinstance(upstream_stage_ids, list) or not all(
            isinstance(item, str) and item for item in upstream_stage_ids
        ):
            errors.append("upstream_stage_ids must be a list of stage id strings")
            upstream_stage_ids = []

        if errors:
            failures.append(_failure(E_MANIFEST_INVALID, subject, "; ".join(errors)))
            continue

        stage = {
            "stage_id": stage_id,
            "artifact_key": artifact_key,
            "needs": list(needs),
            "upstream_stage_ids": list(upstream_stage_ids),
            "concurrency_quota": quota,
        }
        stages.append(stage)
        stage_index[stage_id] = stage
        artifact_index[artifact_key] = stage_id

    # Existence checks are performed after collection so ordering does not hide errors.
    graph: dict[str, set[str]] = {stage["stage_id"]: set() for stage in stages}
    for stage in stages:
        subject = f"manifest.stages.{stage['stage_id']}"
        for artifact_key in stage["needs"]:
            producer = artifact_index.get(artifact_key)
            if producer is None:
                failures.append(_failure(
                    E_MANIFEST_INVALID,
                    subject,
                    f"referenced upstream artifact {artifact_key!r} does not exist",
                ))
            elif producer != stage["stage_id"]:
                graph[stage["stage_id"]].add(producer)
        for upstream_stage_id in stage["upstream_stage_ids"]:
            if upstream_stage_id not in stage_index:
                failures.append(_failure(
                    E_MANIFEST_INVALID,
                    subject,
                    f"referenced upstream stage {upstream_stage_id!r} does not exist",
                ))
            else:
                graph[stage["stage_id"]].add(upstream_stage_id)

    cycle = _find_cycle(graph)
    if cycle:
        failures.append(_failure(
            E_MANIFEST_INVALID,
            "manifest.stages",
            "dependency cycle detected: " + " -> ".join(cycle),
            cycle=cycle,
        ))

    manifest = {
        "stages": stages,
        "artifact_index": artifact_index,
        "stage_index": stage_index,
        "dependency_graph": {key: sorted(value) for key, value in graph.items()},
    }
    return manifest, failures


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt, 0) == 1:
                start = stack.index(nxt)
                return stack[start:] + [nxt]
            if state.get(nxt, 0) == 0:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if state.get(node, 0) == 0:
            found = visit(node)
            if found:
                return found
    return None


def _validate_node_ref(raw: dict[str, Any], nodes: dict[str, Any], subject: str, failures: list) -> list[str]:
    node_id = raw.get("node_id")
    node_ids = raw.get("node_ids")
    errors: list[str] = []
    if node_id is not None:
        if not isinstance(node_id, str) or node_id not in nodes:
            errors.append("node_id must reference a valid loaded node")
    elif node_ids is not None:
        if not isinstance(node_ids, list) or not node_ids:
            errors.append("node_ids must be a non-empty list")
        else:
            if len(node_ids) != len(set(node_ids)):
                errors.append("node_ids must be unique")
            for item in node_ids:
                if not isinstance(item, str) or item not in nodes:
                    errors.append(f"node_ids entry {item!r} must reference a valid loaded node")
    else:
        errors.append("node_id or node_ids is required")
    return errors


def validate_events(
    raw_events: Any,
    nodes: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    if not isinstance(raw_events, list):
        raw_events = []
        failures.append(_failure(E_EVENT_INVALID, "events", "must be a list or JSONL stream"))

    previous_seq = 0
    for index, raw in enumerate(raw_events):
        subject = f"events[{index}]"
        if not isinstance(raw, dict):
            parse_error = raw.get("__parse_error__") if isinstance(raw, dict) else None
            failures.append(_failure(
                E_EVENT_INVALID,
                subject,
                parse_error or "event must be an object",
            ))
            continue

        if "__parse_error__" in raw:
            failures.append(_failure(E_EVENT_INVALID, subject, raw["__parse_error__"]))
            continue

        seq = raw.get("seq")
        event_type = raw.get("type")
        errors: list[str] = []
        if not is_int(seq) or seq <= 0:
            errors.append("seq must be a positive integer")
        elif seq <= previous_seq:
            errors.append(f"seq {seq} is not strictly greater than {previous_seq}")
        if event_type not in EVENT_TYPES:
            errors.append(f"type must be one of {sorted(EVENT_TYPES)}")

        if not errors:
            errors.extend(_validate_event_fields(raw, nodes, manifest, subject))

        if errors:
            failures.append(_failure(
                E_EVENT_INVALID,
                subject,
                "; ".join(errors),
                seq=seq if is_int(seq) else None,
                event_type=event_type if isinstance(event_type, str) else None,
            ))
            if is_int(seq) and seq > previous_seq:
                previous_seq = seq
            continue

        previous_seq = seq
        event = dict(raw)
        event["seq"] = int(seq)
        event["type"] = event_type
        events.append(event)

    return events, failures


def _validate_event_fields(
    event: dict[str, Any],
    nodes: dict[str, Any],
    manifest: dict[str, Any] | None,
    subject: str,
) -> list[str]:
    etype = event["type"]
    errors: list[str] = []

    if etype in {"WRITE", "QUOTA_CHANGED", "POLICY_CHANGED"}:
        errors.extend(_validate_node_ref(event, nodes, subject, []))
    elif etype in {"APPLY", "READ", "RELEASE", "LEASE_RENEW", "RUN_EVICTION"}:
        node_id = event.get("node_id")
        if not isinstance(node_id, str) or node_id not in nodes:
            errors.append("node_id must reference a valid loaded node")

    if etype in {"WRITE", "APPLY"}:
        if not isinstance(event.get("key"), str) or not event["key"]:
            errors.append("key must be a non-empty artifact key")
        if "slots" in event and not is_positive_int(event["slots"]):
            errors.append("slots must be a positive integer")
        if "priority" in event and not is_int(event["priority"]):
            errors.append("priority must be an integer")
        if not is_positive_int(event.get("ttl_ms")):
            errors.append("ttl_ms must be a positive integer")
        stage_id = event.get("stage_id")
        if stage_id is not None:
            if not isinstance(stage_id, str):
                errors.append("stage_id must be a string")
            elif manifest is not None and stage_id not in manifest.get("stage_index", {}):
                errors.append(f"stage_id {stage_id!r} is not declared in manifest")

    if etype == "WRITE" and "content" not in event and "digest" not in event:
        errors.append("WRITE requires content or digest")

    if etype in {"LEASE_RENEW", "TICK"}:
        if not is_int(event.get("now_ms")) or event["now_ms"] < 0:
            errors.append("now_ms must be a non-negative integer")

    if etype in {"WRITE", "APPLY", "QUOTA_CHANGED", "POLICY_CHANGED", "RUN_EVICTION"} and "now_ms" in event:
        if not is_int(event["now_ms"]) or event["now_ms"] < 0:
            errors.append("now_ms must be a non-negative integer")
    if etype in {"WRITE", "APPLY"} and "now_ms" not in event:
        errors.append("now_ms is required for lease issuance")

    if etype == "LEASE_RENEW":
        if not is_positive_int(event.get("ttl_ms")):
            errors.append("ttl_ms must be a positive integer")
        if not isinstance(event.get("key"), str) or not event["key"]:
            errors.append("key is required")

    if etype == "READ":
        if not isinstance(event.get("key"), str) or not event["key"]:
            errors.append("key is required")
        if "renew_ttl_ms" in event and not is_positive_int(event["renew_ttl_ms"]):
            errors.append("renew_ttl_ms must be positive")
        if "renew_ttl_ms" in event:
            if not is_int(event.get("now_ms")) or event["now_ms"] < 0:
                errors.append("now_ms is required for renewal")

    if etype == "RELEASE":
        if not event.get("key") and not event.get("lease_id"):
            errors.append("RELEASE requires key or lease_id")

    if etype == "QUOTA_CHANGED":
        if not is_positive_int(event.get("quota_version")):
            errors.append("quota_version must be a positive integer")
        if "concurrency_quotas" in event:
            quotas = event["concurrency_quotas"]
            if not isinstance(quotas, dict):
                errors.append("concurrency_quotas must be an object")
            else:
                stage_index = manifest.get("stage_index", {}) if manifest else {}
                for stage_id, quota in quotas.items():
                    if stage_id not in stage_index:
                        errors.append(f"concurrency quota stage {stage_id!r} is not declared")
                    if not is_positive_int(quota):
                        errors.append(f"concurrency quota for {stage_id!r} must be positive")

    if etype == "POLICY_CHANGED":
        if event.get("capacity_version") is None and event.get("lease_limit") is None:
            errors.append("POLICY_CHANGED requires capacity_version and/or lease_limit")
        if "capacity_version" in event and not is_positive_int(event["capacity_version"]):
            errors.append("capacity_version must be positive")
        if "lease_limit" in event and not is_positive_int(event["lease_limit"]):
            errors.append("lease_limit must be positive")

    return errors


def load_and_validate(
    config_path: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
    events_path: str | Path | None = None,
    events: list[Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        if config_path is None:
            raise InputValidationError("config or config_path is required")
        config = load_json(config_path)
    if not isinstance(config, dict):
        raise InputValidationError("config root must be an object")

    failures: list[dict[str, Any]] = []
    policies, policy_failures = normalize_capacity_policies(config.get("capacity_policies"))
    failures.extend(policy_failures)

    nodes, node_failures = validate_nodes(config.get("nodes", []), policies)
    failures.extend(node_failures)

    manifest_raw = config.get("manifest")
    if manifest_raw is None and config_path is not None:
        candidate = Path(config_path).parent / "manifest.json"
        if candidate.exists():
            manifest_raw = load_json(candidate)
    manifest, manifest_failures = validate_manifest(manifest_raw)
    failures.extend(manifest_failures)

    raw_events: Any = config.get("events", [])
    if events is not None:
        raw_events = events
    elif events_path is not None:
        raw_events = load_events_file(events_path)
    elif config.get("events_path"):
        raw_events = load_events_file(Path(config["events_path"]))
    events, event_failures = validate_events(raw_events, nodes, manifest)
    failures.extend(event_failures)

    config_hash = fingerprint({
        "schema_version": config.get("schema_version"),
        "capacity_policies": config.get("capacity_policies"),
        "nodes": config.get("nodes"),
        "manifest": config.get("manifest"),
    })

    return {
        "schema_version": config.get("schema_version"),
        "clock_mode": config.get("clock_mode", "seeded"),
        "config": config,
        "config_hash": config_hash,
        "capacity_policies": policies,
        "nodes": nodes,
        "manifest": manifest,
        "events": events,
        "failures": failures,
        "valid": not failures,
    }
