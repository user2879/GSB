from __future__ import annotations

from typing import Any

from .errors import DivergenceError
from .ledger import Ledger
from .spec import Config


def build_projection(ledger: Ledger, config: Config) -> dict[str, Any]:
    snapshot = ledger.latest_snapshot() or {"nodes": {}}
    latest_watermarks = {row["node_id"]: row for row in ledger.latest_watermarks()}
    projection: dict[str, Any] = {
        "config_digest": config.config_digest,
        "progress": {
            "last_ordinal": int(ledger.progress()["last_ordinal"]),
            "last_seq": int(ledger.progress()["last_seq"]),
        },
        "nodes": {},
    }
    for spec in config.nodes:
        node_snapshot = snapshot.get("nodes", {}).get(spec.node_id, {})
        watermark_row = latest_watermarks.get(spec.node_id, {})
        entries = {
            str(entry["artifact_key"]): entry for entry in node_snapshot.get("entries", [])
        }
        leases_by_artifact: dict[str, list[dict[str, Any]]] = {
            artifact.key: [] for artifact in config.artifacts
        }
        for lease in node_snapshot.get("leases", []):
            leases_by_artifact.setdefault(str(lease["artifact_key"]), []).append(
                {
                    "lease_id": str(lease["lease_id"]),
                    "digest": str(lease["digest"]),
                    "quota_version": int(lease["quota_version"]),
                    "apply_seq": int(lease["apply_seq"]),
                }
            )
        artifacts: dict[str, Any] = {}
        for artifact in config.artifacts:
            artifact_leases = sorted(
                leases_by_artifact.get(artifact.key, []), key=lambda item: item["lease_id"]
            )
            entry = entries.get(artifact.key)
            artifacts[artifact.key] = {
                "lease_owners": [spec.node_id] if artifact_leases else [],
                "active_leases": artifact_leases,
                "active_lease_count": len(artifact_leases),
                "digest": None if entry is None else str(entry["digest"]),
                "entry_quota_version": None if entry is None else int(entry["quota_version"]),
            }
        projection["nodes"][spec.node_id] = {
            "quota_version": int(watermark_row.get("quota_version", node_snapshot.get("quota_version", -1))),
            "capacity_slots": int(node_snapshot.get("capacity_slots", -1)),
            "lease_limit": int(node_snapshot.get("lease_limit", -1)),
            "watermark": {
                "used_slots": int(watermark_row.get("used_slots", -1)),
                "free_slots": int(watermark_row.get("free_slots", -1)),
                "active_leases": int(watermark_row.get("active_leases", -1)),
                "lease_remaining": int(watermark_row.get("lease_remaining", -1)),
                "watermark_units": int(watermark_row.get("watermark_units", -1)),
                "quantization_error_upper": int(
                    watermark_row.get("quantization_error_upper", -1)
                ),
            },
            "artifacts": artifacts,
        }
    return projection


def compare_projections(
    incremental: dict[str, Any], full: dict[str, Any], config: Config
) -> None:
    node_ids = sorted(set(incremental["nodes"]) | set(full["nodes"]))
    for node_id in node_ids:
        left_node = incremental["nodes"].get(node_id)
        right_node = full["nodes"].get(node_id)
        if left_node is None or right_node is None:
            raise DivergenceError(node_id, "*", "node_presence", "node exists on only one path")
        for field_name in ("quota_version", "capacity_slots", "lease_limit"):
            if left_node[field_name] != right_node[field_name]:
                raise DivergenceError(
                    node_id,
                    "*",
                    field_name,
                    f"incremental={left_node[field_name]!r} full={right_node[field_name]!r}",
                )
        for field_name in sorted(left_node["watermark"]):
            if left_node["watermark"][field_name] != right_node["watermark"][field_name]:
                raise DivergenceError(
                    node_id,
                    "*",
                    f"watermark.{field_name}",
                    f"incremental={left_node['watermark'][field_name]!r} full={right_node['watermark'][field_name]!r}",
                )
        artifact_keys = sorted(set(left_node["artifacts"]) | set(right_node["artifacts"]))
        for artifact_key in artifact_keys:
            left_artifact = left_node["artifacts"].get(artifact_key)
            right_artifact = right_node["artifacts"].get(artifact_key)
            if left_artifact is None or right_artifact is None:
                raise DivergenceError(node_id, artifact_key, "artifact_presence", "artifact exists on only one path")
            for field_name in (
                "lease_owners",
                "active_leases",
                "active_lease_count",
                "digest",
                "entry_quota_version",
            ):
                if left_artifact[field_name] != right_artifact[field_name]:
                    raise DivergenceError(
                        node_id,
                        artifact_key,
                        field_name,
                        f"incremental={left_artifact[field_name]!r} full={right_artifact[field_name]!r}",
                    )

