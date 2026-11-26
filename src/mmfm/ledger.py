"""Content-key reuse ledger and invalidation explanation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .canonical import content_digest


class Ledger:
    def __init__(self, prior_descriptors: dict[str, Any] | None = None) -> None:
        self.prior_descriptors = prior_descriptors or {}
        self.descriptors: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def descriptor(
        self,
        *,
        stage: str,
        logical_id: str,
        cache_key: str,
        deps: list[dict[str, str]],
        params: dict[str, Any],
        digest: str | None,
        hit: bool,
        hit_origin: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalised_deps = sorted(
            [{"logical_id": dep["logical_id"], "cache_key": dep["cache_key"]} for dep in deps],
            key=lambda item: item["logical_id"],
        )
        descriptor = {
            "stage": stage,
            "logical_id": logical_id,
            "cache_key": cache_key,
            "deps": normalised_deps,
            "params_digest": content_digest(params),
            "digest": digest,
            "hit": bool(hit),
        }
        if not hit:
            basis = self._basis(descriptor)
        elif hit_origin == "current_run":
            prior = self.prior_descriptors.get(logical_id)
            basis = {
                "reason": "current_run_content_reuse",
                "previous_cache_key": prior.get("cache_key") if prior else None,
            }
        else:
            basis = {"reason": "content_key_match"}
        event = {
            **descriptor,
            "event": "reuse_hit" if hit else "recompute",
            "invalidation_basis": basis,
            "hit_origin": hit_origin if hit else None,
        }
        if extra:
            event.update(extra)
        self.descriptors[logical_id] = descriptor
        self.events.append(event)
        return descriptor

    def pending(
        self,
        *,
        stage: str,
        logical_id: str,
        frame: int,
        step: int | None = None,
        chunk: int | None = None,
    ) -> None:
        self.events.append(
            {
                "event": "budget_pending",
                "stage": stage,
                "logical_id": logical_id,
                "frame": frame,
                "step": step,
                "chunk": chunk,
                "invalidation_basis": {"reason": "denoise_budget_exhausted"},
            }
        )

    def _basis(self, current: dict[str, Any]) -> dict[str, Any]:
        prior = self.prior_descriptors.get(current["logical_id"])
        if not prior:
            return {
                "reason": "new_or_uncommitted_task",
                "previous_cache_key": None,
            }
        if prior.get("cache_key") == current["cache_key"]:
            return {
                "reason": "matching_key_has_no_committed_artifact",
                "previous_cache_key": prior.get("cache_key"),
            }
        prior_deps = {
            item.get("logical_id"): item.get("cache_key") for item in prior.get("deps", [])
        }
        changed: list[dict[str, Any]] = []
        for dep in current["deps"]:
            previous_key = prior_deps.get(dep["logical_id"])
            if previous_key != dep["cache_key"]:
                changed.append(
                    {
                        "logical_id": dep["logical_id"],
                        "previous_cache_key": previous_key,
                        "current_cache_key": dep["cache_key"],
                    }
                )
        if changed:
            return {
                "reason": "upstream_content_changed",
                "changed_dependencies": changed,
                "previous_cache_key": prior.get("cache_key"),
            }
        if current["stage"] == "condition":
            return {
                "reason": "condition_source_changed",
                "previous_cache_key": prior.get("cache_key"),
            }
        if prior.get("params_digest") != current["params_digest"]:
            return {
                "reason": "parameter_changed",
                "previous_cache_key": prior.get("cache_key"),
            }
        return {
            "reason": "content_key_changed",
            "previous_cache_key": prior.get("cache_key"),
        }

    def merged_descriptors(self) -> dict[str, dict[str, Any]]:
        merged = dict(self.prior_descriptors)
        merged.update(self.descriptors)
        return merged

    def summary(self) -> dict[str, Any]:
        by_stage: dict[str, Counter[str]] = {}
        for event in self.events:
            stage = str(event.get("stage", "unknown"))
            by_stage.setdefault(stage, Counter())[str(event["event"])] += 1
        return {
            "events": len(self.events),
            "reuse_hits": sum(1 for event in self.events if event["event"] == "reuse_hit"),
            "recomputes": sum(1 for event in self.events if event["event"] == "recompute"),
            "pending": sum(1 for event in self.events if event["event"] == "budget_pending"),
            "by_stage": {stage: dict(counts) for stage, counts in sorted(by_stage.items())},
        }
