"""Compile a v3 manifest into a deterministic per-node distribution plan."""

from __future__ import annotations
import hashlib

from .derivation import node_digest
from .errors import BizError, ErrorCode
from .manifest import canonical_json


def build_plan(manifest: dict, node_ids: list[str]) -> dict:
    if manifest["format_version"] != 3:
        raise BizError(
            ErrorCode.E_FORMAT,
            "only current (v3) manifests can be compiled into plans",
            {"format_version": manifest["format_version"]},
        )
    node_set = set(node_ids)
    groups: dict[str, list[dict]] = {node_id: [] for node_id in node_ids}
    default_strategy = {
        "mode": "targeted",
        "retries": 2,
        "timeout_ms": 2000,
    }

    for index, entry in enumerate(manifest["entries"]):
        target = entry["node"]
        if target not in node_set:
            raise BizError(
                ErrorCode.E_FORMAT,
                f"entry '{entry['key']}' targets undeclared node '{target}'",
                {"field_path": f"$.entries[{index}].node", "node": target},
            )
        strategy = entry.get("strategy", dict(default_strategy))
        item = {
            "key": entry["key"],
            "value": entry["value"],
            "config_version": entry["config_version"],
            "strategy": strategy,
        }
        targets = node_ids if strategy.get("mode") == "broadcast" else [target]
        for node_id in targets:
            groups[node_id].append(item)

    derivations = manifest.get("derivations", [])
    derivation_nodes = {item["target_node"] for item in derivations}
    unknown = derivation_nodes - node_set
    if unknown:
        raise BizError(
            ErrorCode.E_FORMAT,
            "derivation targets undeclared node",
            {"nodes": sorted(unknown)},
        )

    slices = []
    for node_id in sorted(groups):
        node_entries = sorted(groups[node_id], key=lambda item: item["key"])
        slice_payload = {
            "node": node_id,
            "entries": node_entries,
            "expected_digest": node_digest(node_id, node_entries, derivations),
        }
        slices.append(slice_payload)

    plan = {
        "config_version": manifest["entries"][0]["config_version"],
        "slices": slices,
        "derivations": sorted(
            (
                {
                    "name": item["name"],
                    "inputs": sorted(item["inputs"]),
                    "op": item["op"],
                    "target_node": item["target_node"],
                }
                for item in derivations
            ),
            key=lambda item: item["name"],
        ),
    }
    plan["plan_id"] = "plan-" + hashlib.sha256(
        canonical_json(_plan_body(plan)).encode("utf-8")
    ).hexdigest()[:16]
    return plan


def _plan_body(plan: dict) -> dict:
    return {key: value for key, value in plan.items() if key not in ("plan_id", "created_at")}


def plan_fingerprint(plan: dict) -> str:
    return hashlib.sha256(canonical_json(_plan_body(plan)).encode("utf-8")).hexdigest()


def expected_digest_map(plan: dict) -> dict[str, str]:
    return {item["node"]: item["expected_digest"] for item in plan["slices"]}
