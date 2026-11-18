from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .config import InferenceConfig


TRACKED_FIELDS = (
    "batch_size_limit",
    "precision",
    "quant_scale_power",
    "memory_budget_mib",
    "block_size_kb",
    "preallocation",
)

STALE_SECTIONS = ["batch_outputs", "memory_statistics", "error_conclusions"]


@dataclass(frozen=True)
class PendingTransition:
    previous_version: int | None
    version: int
    changed_fields: tuple[str, ...]
    previous_config: dict[str, Any] | None


class ConfigVersionManager:
    def __init__(self) -> None:
        self.current_version: int | None = None
        self.current_config: InferenceConfig | None = None
        self.current_batch_ids: tuple[str, ...] = ()

    def begin(self, new_config: InferenceConfig) -> PendingTransition:
        if self.current_version is None or self.current_config is None:
            return PendingTransition(None, 1, (), None)
        old_fields = self.current_config.version_fields()
        new_fields = new_config.version_fields()
        changed = tuple(sorted(field for field in TRACKED_FIELDS if old_fields[field] != new_fields[field]))
        version = self.current_version + (1 if changed else 0)
        return PendingTransition(
            previous_version=self.current_version,
            version=version,
            changed_fields=changed,
            previous_config=self.current_config.to_dict(),
        )

    def commit(
        self,
        pending: PendingTransition,
        new_config: InferenceConfig,
        new_batch_ids: Iterable[str],
    ) -> dict[str, Any]:
        old_batch_ids = set(self.current_batch_ids)
        current_batch_ids = tuple(sorted(str(batch_id) for batch_id in new_batch_ids))
        changed = list(pending.changed_fields)
        if changed:
            affected = sorted(old_batch_ids | set(current_batch_ids))
            stale_batch_ids = sorted(old_batch_ids)
        else:
            affected = []
            stale_batch_ids = []
        replacement = {
            "replaced_whole": True,
            "merge_mode": "full_replacement",
            "previous_version": pending.previous_version,
            "current_version": pending.version,
            "changed_fields": changed,
            "affected_batches": affected,
            "stale_marker": "STALE" if changed else "NONE",
            "stale_batch_ids": stale_batch_ids,
            "stale_sections": list(STALE_SECTIONS) if changed else [],
            "stale_results_excluded_from_current_report": True,
            "previous_configs_excluded_from_aggregates": True,
        }
        self.current_version = pending.version
        self.current_config = new_config
        self.current_batch_ids = current_batch_ids
        return replacement
