"""Loading and strict validation of node registry and versioned manifests.

Validation is all-or-nothing: every violation is collected, sorted into a
stable listing and raised together as E_MANIFEST (or E_CYCLE for derived-from
cycles, which have their own business code).  No partial state is returned.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from .errors import BusinessError, ErrorCode
from .models import ConfigItem, ManifestVersion, NodeSpec, Strategy
from .util import stable_topological

KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
VALID_STRATEGIES = {"full", "shard"}
VALID_RETRY = {"none", "fixed"}


def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise BusinessError(
            ErrorCode.E_MANIFEST,
            f"input file not found: {path}",
            [{"path": path, "reason": "file_not_found"}],
        )
    except json.JSONDecodeError as exc:
        raise BusinessError(
            ErrorCode.E_MANIFEST,
            f"corrupt JSON file: {path}",
            [{"path": path, "reason": "invalid_json",
              "line": exc.lineno, "column": exc.colno, "detail": exc.msg}],
        )


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def load_registry(path: str) -> dict[str, NodeSpec]:
    data = _load_json(path)
    violations: list[dict[str, Any]] = []
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise BusinessError(
            ErrorCode.E_MANIFEST,
            "registry must be an object containing a 'nodes' array",
            [{"path": path, "reason": "schema_shape"}],
        )
    specs: list[NodeSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, raw in enumerate(data["nodes"]):
        prefix = f"nodes[{index}]"
        if not isinstance(raw, dict):
            violations.append({"entry": prefix, "reason": "not_an_object"})
            continue
        node_id = raw.get("node_id")
        if not isinstance(node_id, str) or not KEY_PATTERN.match(node_id or ""):
            violations.append({"entry": f"{prefix}.node_id", "value": node_id,
                               "reason": "illegal_or_missing_node_id"})
            continue
        if node_id in seen_ids:
            violations.append({"entry": f"{prefix}.node_id", "value": node_id,
                               "reason": "duplicate_node_id"})
        seen_ids.add(node_id)
        port = raw.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not (1024 <= port <= 65535):
            violations.append({"entry": f"{prefix}.port", "value": port,
                               "reason": "port_must_be_int_1024_65535"})
        elif port in seen_ports:
            violations.append({"entry": f"{prefix}.port", "value": port,
                               "reason": "duplicate_port"})
        else:
            seen_ports.add(port)
        host = raw.get("host", "127.0.0.1")
        if host != "127.0.0.1":
            violations.append({"entry": f"{prefix}.host", "value": host,
                               "reason": "loopback_only"})
        timeout = raw.get("timeout_ms", 2000)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            violations.append({"entry": f"{prefix}.timeout_ms", "value": timeout,
                               "reason": "positive_integer_required"})
        retry = raw.get("retry_policy", "none")
        if retry not in VALID_RETRY:
            violations.append({"entry": f"{prefix}.retry_policy", "value": retry,
                               "reason": f"one_of_{sorted(VALID_RETRY)}"})
        max_retries = raw.get("max_retries", 0)
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            violations.append({"entry": f"{prefix}.max_retries", "value": max_retries,
                               "reason": "non_negative_integer_required"})
        if retry == "fixed" and max_retries == 0:
            violations.append({"entry": f"{prefix}.max_retries", "value": max_retries,
                               "reason": "fixed_retry_requires_max_retries_gt_0"})
        if violations and any(v["entry"].startswith(prefix) for v in violations):
            continue
        specs.append(NodeSpec.from_dict(raw))
    if violations:
        raise BusinessError(ErrorCode.E_MANIFEST,
                            "node registry validation failed",
                            sorted(violations, key=lambda v: v["entry"]))
    return {spec.node_id: spec for spec in specs}


def _validate_items(raw_items: Any, violations: list[dict[str, Any]],
                    version: int) -> dict[str, ConfigItem]:
    entry_prefix = f"versions[{version}].items"
    if not isinstance(raw_items, list) or not raw_items:
        violations.append({"entry": entry_prefix, "reason": "items_must_be_nonempty_array"})
        return {}
    items: dict[str, ConfigItem] = {}
    for index, raw in enumerate(raw_items):
        prefix = f"{entry_prefix}[{index}]"
        if not isinstance(raw, dict):
            violations.append({"entry": prefix, "reason": "not_an_object"})
            continue
        key = raw.get("key")
        if not isinstance(key, str) or not KEY_PATTERN.match(key or ""):
            violations.append({"entry": f"{prefix}.key", "value": key,
                               "reason": "illegal_key_pattern"})
            continue
        if key in items:
            violations.append({"entry": f"{prefix}.key", "value": key,
                               "reason": "duplicate_item_key"})
            continue
        derived = raw.get("derived_from", {})
        value = raw.get("value")
        if derived is None:
            derived = {}
        if not isinstance(derived, dict) or not all(
            isinstance(k, str) for k in derived):
            violations.append({"entry": f"{prefix}.derived_from",
                               "reason": "mapping_of_dep_key_to_numeric_coeff"})
            continue
        bad_coeff = [dep for dep, coeff in derived.items() if not _is_number(coeff) or coeff == 0]
        for dep in bad_coeff:
            violations.append({"entry": f"{prefix}.derived_from.{dep}",
                               "value": derived[dep], "reason": "nonzero_numeric_coeff_required"})
        if derived:
            if any(isinstance(k, str) and k == key for k in derived):
                violations.append({"entry": f"{prefix}.derived_from", "value": key,
                                   "reason": "self_dependency"})
        elif not _is_number(value):
            violations.append({"entry": f"{prefix}.value", "value": value,
                               "reason": "base_item_requires_finite_numeric_value"})
        if value is not None and not _is_number(value):
            violations.append({"entry": f"{prefix}.value", "value": value,
                               "reason": "finite_numeric_value_required"})
        items[key] = ConfigItem(
            key=key,
            value=float(value) if value is not None else None,
            derived_from={dep: float(coeff) for dep, coeff in derived.items()
                          if _is_number(coeff) and coeff != 0},
        )
    # reference existence across the same version
    for item in items.values():
        for dep in item.derived_from:
            if dep not in items:
                violations.append({
                    "entry": f"{entry_prefix}.{item.key}.derived_from.{dep}",
                    "reason": "dependency_key_not_declared_in_version",
                })
    return items


def load_manifest(path: str, registry: dict[str, NodeSpec]) -> list[ManifestVersion]:
    data = _load_json(path)
    violations: list[dict[str, Any]] = []
    if not isinstance(data, dict) or not isinstance(data.get("versions"), list):
        raise BusinessError(
            ErrorCode.E_MANIFEST,
            "manifest must be an object containing a 'versions' array",
            [{"path": path, "reason": "schema_shape"}],
        )
    raw_versions = data["versions"]
    if not raw_versions:
        raise BusinessError(ErrorCode.E_MANIFEST, "manifest contains no versions",
                            [{"path": path, "reason": "empty_versions"}])
    parsed: list[tuple[int, ManifestVersion, dict[str, ConfigItem]]] = []
    previous_version: int | None = None
    seen_versions: set[int] = set()
    for index, raw in enumerate(raw_versions):
        prefix = f"versions[{index}]"
        if not isinstance(raw, dict):
            violations.append({"entry": prefix, "reason": "not_an_object"})
            continue
        version = raw.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            violations.append({"entry": f"{prefix}.version", "value": version,
                               "reason": "integer_version_required"})
        else:
            if version in seen_versions:
                violations.append({"entry": f"{prefix}.version", "value": version,
                                   "reason": "duplicate_version"})
            if previous_version is not None and version <= previous_version:
                violations.append({"entry": f"{prefix}.version", "value": version,
                                   "reason": "versions_must_strictly_increase",
                                   "previous": previous_version})
            seen_versions.add(version)
            previous_version = version
        targets = raw.get("targets")
        if not isinstance(targets, list) or not targets or not all(
            isinstance(t, str) for t in targets):
            violations.append({"entry": f"{prefix}.targets", "value": targets,
                               "reason": "nonempty_string_array_required"})
        else:
            for target in targets:
                if target not in registry:
                    violations.append({"entry": f"{prefix}.targets.{target}",
                                       "reason": "node_not_in_registry"})
        strategy_raw = raw.get("strategy")
        strategy: Strategy | None = None
        if not isinstance(strategy_raw, dict):
            violations.append({"entry": f"{prefix}.strategy",
                               "reason": "strategy_object_required"})
        else:
            mode = strategy_raw.get("mode")
            if mode not in VALID_STRATEGIES:
                violations.append({"entry": f"{prefix}.strategy.mode", "value": mode,
                                   "reason": f"one_of_{sorted(VALID_STRATEGIES)}"})
            shard_count = strategy_raw.get("shard_count", 1)
            if not isinstance(shard_count, int) or isinstance(shard_count, bool) or shard_count < 1:
                violations.append({"entry": f"{prefix}.strategy.shard_count",
                                   "value": shard_count,
                                   "reason": "positive_integer_required"})
            if mode == "full" and shard_count not in (1,):
                violations.append({"entry": f"{prefix}.strategy.shard_count",
                                   "value": shard_count,
                                   "reason": "full_mode_requires_shard_count_1"})
            else:
                strategy = Strategy.from_dict(strategy_raw)
        items = _validate_items(raw.get("items"), violations, index)
        if isinstance(version, int) and strategy is not None and items and isinstance(targets, list):
            parsed.append((index,
                           ManifestVersion(version=version, items=items,
                                           targets=list(targets), strategy=strategy),
                           items))
    # cycle detection per version (dedicated E_CYCLE code)
    for index, manifest_version, items in parsed:
        edges = {key: set(item.derived_from) for key, item in items.items()}
        try:
            stable_topological(set(items), edges)
        except ValueError as exc:
            raise BusinessError(
                ErrorCode.E_CYCLE,
                f"derived-from cycle in version {manifest_version.version}",
                [{"entry": f"versions[{index}]", "reason": str(exc)}],
            )
    if violations:
        raise BusinessError(
            ErrorCode.E_MANIFEST,
            "manifest validation failed; load rejected entirely",
            sorted(violations, key=lambda v: (v["entry"], str(v.get("reason")))),
        )
    return [manifest_version for _, manifest_version, _ in sorted(parsed, key=lambda t: t[0])]


def load_bundle(manifest_path: str, registry_path: str):
    registry = load_registry(registry_path)
    versions = load_manifest(manifest_path, registry)
    return registry, versions
