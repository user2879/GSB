"""Lineage, invalidation, and cache-reuse ledger."""

from __future__ import annotations

from typing import Any


class ReuseLedger:
    def __init__(self, previous_lineage: dict[str, dict[str, Any]] | None = None):
        self.previous = previous_lineage or {}
        self.entries: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._artifacts: set[str] = set()

    def record(
        self,
        *,
        artifact_id: str,
        stage: str,
        cache_key: str,
        cache_hit: bool,
        basis: str,
        sample_ids: list[str] | None = None,
        columns: list[str] | None = None,
        depends_on: list[str] | None = None,
        material: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if artifact_id in self._artifacts:
            raise RuntimeError(f"duplicate lineage artifact: {artifact_id}")
        self._artifacts.add(artifact_id)
        previous = self.previous.get(artifact_id)
        invalidation = None
        if previous is not None and previous.get("cache_key") != cache_key:
            invalidation = {
                "previous_key": previous.get("cache_key"),
                "reason": _invalidation_reason(previous.get("material") or {}, material or {}),
            }
        status = "reused" if cache_hit else "rebuilt"
        entry = {
            "artifact_id": artifact_id,
            "stage": stage,
            "cache_key": cache_key,
            "status": status,
            "reuse_basis": basis if cache_hit else f"cache_miss:{basis}",
            "sample_refs": sorted(sample_ids or []),
            "column_refs": list(columns or []),
            "depends_on": sorted(depends_on or []),
            "invalidation": invalidation,
            "material": material or {},
        }
        self.entries.append(entry)
        self.events.append({
            "artifact_id": artifact_id,
            "stage": stage,
            "cache_key": cache_key,
            "status": status,
            "reason": entry["reuse_basis"],
        })
        return entry

    def invalidate_removed(self, current_artifacts: set[str]) -> list[dict[str, Any]]:
        removed = []
        for artifact_id, previous in sorted(self.previous.items()):
            if artifact_id not in current_artifacts:
                record = {
                    "artifact_id": artifact_id,
                    "stage": previous.get("stage"),
                    "previous_key": previous.get("cache_key"),
                    "status": "invalidated_removed",
                    "reason": "artifact_or_reference_set_removed",
                    "sample_refs": previous.get("sample_refs", []),
                    "column_refs": previous.get("column_refs", []),
                }
                removed.append(record)
                self.events.append(record)
        return removed

    def lineage_manifest(self) -> list[dict[str, Any]]:
        return sorted(self.entries, key=lambda entry: (entry["stage"], entry["artifact_id"], entry["cache_key"]))

    def ordered_events(self) -> list[dict[str, Any]]:
        return sorted(self.events, key=lambda event: (event.get("stage") or "", event.get("artifact_id") or "", event.get("cache_key") or ""))

    def summary(self) -> dict[str, Any]:
        reused = sum(1 for entry in self.entries if entry["status"] == "reused")
        rebuilt = sum(1 for entry in self.entries if entry["status"] == "rebuilt")
        invalidated = sum(1 for event in self.events if event.get("status") == "invalidated_removed")
        changed = sum(1 for entry in self.entries if entry.get("invalidation"))
        return {
            "artifact_count": len(self.entries),
            "reused": reused,
            "rebuilt": rebuilt,
            "changed_keys": changed,
            "removed_artifacts": invalidated,
        }


def _invalidation_reason(old_material: dict[str, Any], new_material: dict[str, Any]) -> str:
    if old_material.get("fixed") != new_material.get("fixed"):
        return "fixed_point_or_quant_gear_changed"
    if old_material.get("alpha") != new_material.get("alpha"):
        return "alpha_level_changed"
    old_samples = old_material.get("sample_fingerprints")
    new_samples = new_material.get("sample_fingerprints")
    if old_samples != new_samples:
        old_map = dict(old_samples or [])
        new_map = dict(new_samples or [])
        changed = sorted(k for k in set(old_map) | set(new_map) if old_map.get(k) != new_map.get(k))
        if any(k not in old_map for k in changed):
            return "sample_appended_or_replaced"
        if any(k not in new_map for k in changed):
            return "sample_deleted"
        return f"sample_content_changed:{','.join(changed[:4])}"
    old_cols = old_material.get("column_fingerprints")
    new_cols = new_material.get("column_fingerprints")
    if old_cols != new_cols:
        return "column_content_or_schema_changed"
    if old_material.get("solver") != new_material.get("solver"):
        return "solver_parameters_changed"
    return "content_fingerprint_changed"
