"""High-level workbench runtime tying inference, graph, leases and ledger."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from .errors import DiagnosticError, budget_error, replay_error
from .graph import DependencyGraph, build_graph
from .inference import (
    LayerBatchResult,
    aggregate_layer_digest,
    aggregate_layer_errors,
    attribute_errors,
    compute_batch,
    compute_layer_result,
)
from .lease import Lease, LeasePool, LeaseRequest
from .ledger import Ledger
from .samples import BatchSpec, SampleFactory, make_batches
from .schema import (
    Plan,
    change_layer,
    change_layers,
    layer_memory_bytes,
    remove_layer,
)
from .scheduler import schedule_batches
from .util import monotonic_ms


class InferenceRuntime:
    def __init__(
        self,
        plan: Plan,
        sample_count: int,
        *,
        batch_size: int | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if batch_size is None:
            batch_size = plan.batch_size_max
        if batch_size <= 0 or batch_size > plan.batch_size_max:
            raise ValueError("batch_size must be within plan.batch_size_max")
        self.plan = plan
        self.sample_count = sample_count
        self.batch_size = batch_size
        self.factory = SampleFactory(plan, sample_count)
        self.batches = make_batches(self.factory.ids, batch_size)
        self.graph = build_graph(plan)
        self.layer_versions: dict[str, int] = {layer.id: 1 for layer in plan.layers}
        self.results: dict[tuple[str, str], LayerBatchResult] = {}
        self.ledger = ledger
        self.lease_history: list[dict[str, Any]] = []
        self.current_lease_ledger: list[dict[str, Any]] = []
        self.invalidation_history: list[dict[str, Any]] = []
        self.last_schedule: dict[str, Any] | None = None
        self.last_attribution: dict[str, Any] | None = None
        self.restored_layer_summaries: dict[str, Any] | None = None
        self.version_pool = LeasePool(
            block_size_bytes=plan.block_size_bytes,
            memory_budget_bytes=plan.memory_budget_bytes,
            id_prefix="H",
        )
        self.held_leases: dict[str, Lease] = {}

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.factory.ids

    def _new_pool(self) -> LeasePool:
        return LeasePool(
            block_size_bytes=self.plan.block_size_bytes,
            memory_budget_bytes=self.plan.memory_budget_bytes,
        )

    def _requests_for_batch(self, batch: BatchSpec) -> list[LeaseRequest]:
        return [
            LeaseRequest(
                layer_id=layer_id,
                blocks=layer_memory_bytes(self.plan, layer_id, batch.size)
                // self.plan.block_size_bytes,
            )
            for layer_id in self.graph.order
        ]

    def run(
        self,
        *,
        worker_count: int = 4,
        crash_after_committed_layers: int | None = None,
        batches: Sequence[BatchSpec] | None = None,
        ledger: Ledger | None = None,
    ) -> dict[str, Any]:
        if ledger is not None:
            self.ledger = ledger
        if self.ledger is None:
            raise ValueError("a ledger is required for scheduled inference")
        pool = self._new_pool()
        selected = tuple(batches if batches is not None else self.batches)
        schedule = schedule_batches(
            plan=self.plan,
            graph=self.graph,
            factory=self.factory,
            batches=selected,
            layer_versions=self.layer_versions,
            ledger=self.ledger,
            pool=pool,
            worker_count=worker_count,
            crash_after_committed_layers=crash_after_committed_layers,
        )
        outcome_objects = schedule.pop("_outcome_objects")
        self.last_schedule = schedule
        self.restored_layer_summaries = None
        for outcome in outcome_objects:
            if outcome.error:
                continue
            for layer_id, result in outcome.results.items():
                self.results[(outcome.batch_id, layer_id)] = result
        self.lease_history.extend(
            lease for batch in schedule["batches"] for lease in batch.get("leases", [])
        )
        if not schedule["failed_batches"]:
            attribution = self.attribution()
            self.last_attribution = attribution
            self.ledger.record_attribution(self.plan.version, attribution)
        report = self.report()
        report["ok"] = not schedule["failed_batches"]
        report["schedule"] = schedule
        return report

    def _results_for_layer(self, layer_id: str) -> list[LayerBatchResult]:
        return [
            result
            for (batch_id, result_layer), result in sorted(self.results.items())
            if result_layer == layer_id
        ]

    def layer_summaries(self) -> dict[str, Any]:
        if self.restored_layer_summaries is not None and not self.results:
            return self.restored_layer_summaries
        summaries: dict[str, Any] = {}
        for layer_id in self.graph.order:
            items = self._results_for_layer(layer_id)
            if not items:
                continue
            digest = aggregate_layer_digest(items)
            errors = aggregate_layer_errors(items)
            bounds = [item.error_bound for item in items]
            components = {
                key: max(float(bound[key]) for bound in bounds)
                for key in ("quantization_truncation", "accumulation_order", "lease_reordering")
            }
            components["combined"] = sum(components.values())
            leases = sorted(
                (
                    {
                        "batch_id": item.batch_id,
                        "lease_id": item.lease_id,
                        "block_range": [item.block_start, (item.block_start or 0) + (item.block_count or 0)],
                        "block_count": item.block_count,
                    }
                    for item in items
                ),
                key=lambda item: (item["batch_id"], item["lease_id"] or ""),
            )
            versions = sorted({item.layer_version for item in items})
            summaries[layer_id] = {
                "layer_id": layer_id,
                "layer_version": versions[0] if len(versions) == 1 else versions,
                "plan_version": self.plan.version,
                "sample_count": digest["sample_count"],
                "float_digest": digest["float_digest"],
                "quant_digest": digest["quant_digest"],
                "combined_digest": digest["combined_digest"],
                "errors": errors,
                "error_bound": components,
                "saturation_count": errors["saturation_count"],
                "occupancy_bytes": max(item.occupancy_bytes for item in items),
                "occupancy_blocks": max(item.occupancy_blocks for item in items),
                "lease_ranges": leases,
                "lease_range": [item["block_range"] for item in leases],
            }
        return summaries

    def attribution(self) -> dict[str, Any]:
        results_by_layer: dict[str, list[LayerBatchResult]] = {}
        for layer_id in self.graph.order:
            items = self._results_for_layer(layer_id)
            if items:
                results_by_layer[layer_id] = items
        return attribute_errors(self.plan, self.graph, results_by_layer)

    def batch_timeline(self) -> list[dict[str, Any]]:
        if self.last_schedule is None:
            return []
        return [
            {
                "batch_id": batch["batch_id"],
                "sample_ids": batch["sample_ids"],
                "queue_start_ms": batch["queue_start_ms"],
                "queue_end_ms": batch["queue_end_ms"],
                "compute_start_ms": batch["compute_start_ms"],
                "end_ms": batch["end_ms"],
                "queue_wait_ms": batch["queue_wait_ms"],
                "compute_ms": batch["compute_ms"],
                "worker_name": batch["worker_name"],
                "lease_ids": batch["lease_ids"],
                "block_ranges": batch["block_ranges"],
                "status": "failed" if batch["error"] else "succeeded",
            }
            for batch in self.last_schedule["batches"]
        ]

    def lease_timeline(self) -> list[dict[str, Any]]:
        records = self.current_lease_ledger or self.lease_history
        return sorted(
            records,
            key=lambda item: (
                int(item.get("granted_ms") or 0),
                str(item.get("batch_id") or ""),
                str(item.get("layer_id") or ""),
                str(item.get("lease_id") or ""),
            ),
        )

    def report(self) -> dict[str, Any]:
        return {
            "ok": True,
            "plan_version": self.plan.version,
            "plan_content_digest": self.plan.content_digest,
            "quantization_scale_version": self.plan.quantization_scale_version,
            "sample_count": self.sample_count,
            "batch_size": self.batch_size,
            "batches": [batch.to_dict() for batch in self.batches],
            "dataset": self.factory.summary(),
            "graph": self.graph.to_dict(),
            "layer_versions": {
                layer_id: self.layer_versions[layer_id] for layer_id in self.graph.order
            },
            "layer_summaries": self.layer_summaries(),
            "attribution": self.attribution(),
            "batch_timeline": self.batch_timeline(),
            "lease_timeline": self.lease_timeline(),
            "invalidation_history": self.invalidation_history,
            "ledger_content_digest": (
                None if self.ledger is None else self.ledger.content_digest()
            ),
        }

    def _retag_result(
        self,
        result: LayerBatchResult,
        *,
        plan_version: int,
        layer_version: int,
        lease_id: str,
        block_start: int,
        block_count: int,
        granted_ms: int,
        expires_ms: int,
    ) -> LayerBatchResult:
        return LayerBatchResult(
            layer_id=result.layer_id,
            batch_id=result.batch_id,
            sample_ids=result.sample_ids,
            layer_version=layer_version,
            plan_version=plan_version,
            float_output=result.float_output,
            quant_output=result.quant_output,
            errors=result.errors,
            error_bound=result.error_bound,
            saturation_count=result.saturation_count,
            occupancy_bytes=result.occupancy_bytes,
            occupancy_blocks=result.occupancy_blocks,
            lease_id=lease_id,
            block_start=block_start,
            block_count=block_count,
            input_digest=result.input_digest,
            float_digest=result.float_digest,
            quant_digest=result.quant_digest,
            per_sample=result.per_sample,
        )

    def _build_range_ledger(
        self,
        *,
        old_results: Mapping[tuple[str, str], LayerBatchResult],
        invalidated: set[str],
        new_plan: Plan,
        new_graph: DependencyGraph,
        batches: Sequence[BatchSpec],
    ) -> tuple[dict[tuple[str, str], tuple[int, int]], list[dict[str, Any]]]:
        now = monotonic_ms()
        assignments: dict[tuple[str, str], tuple[int, int]] = {}
        records: list[dict[str, Any]] = []
        total_blocks = new_plan.memory_budget_bytes // new_plan.block_size_bytes
        for batch in batches:
            occupied: list[tuple[int, int]] = []
            block_counts: dict[str, int] = {
                layer_id: layer_memory_bytes(new_plan, layer_id, batch.size)
                // new_plan.block_size_bytes
                for layer_id in new_graph.order
            }
            if sum(block_counts.values()) > total_blocks:
                raise budget_error(
                    "replacement lease ledger exceeds budget for a batch",
                    batch_id=batch.batch_id,
                    required_blocks=sum(block_counts.values()),
                    budget_blocks=total_blocks,
                )
            for layer_id in new_graph.order:
                count = block_counts[layer_id]
                old = old_results.get((batch.batch_id, layer_id))
                start: int | None = None
                if layer_id not in invalidated and old and old.block_count == count:
                    candidate = int(old.block_start or 0)
                    if not any(candidate < end and start_existing < candidate + count for start_existing, end in occupied):
                        start = candidate
                if start is None:
                    for candidate in range(total_blocks - count + 1):
                        if not any(candidate < end and start_existing < candidate + count for start_existing, end in occupied):
                            start = candidate
                            break
                if start is None:
                    raise budget_error(
                        "cannot place replacement lease",
                        batch_id=batch.batch_id,
                        layer_id=layer_id,
                        blocks=count,
                    )
                occupied.append((start, start + count))
                assignments[(batch.batch_id, layer_id)] = (start, count)
                records.append(
                    {
                        "lease_id": f"V{new_plan.version}-{batch.batch_id}-{layer_id}",
                        "batch_id": batch.batch_id,
                        "layer_id": layer_id,
                        "plan_version": new_plan.version,
                        "block_range": [start, start + count],
                        "block_start": start,
                        "block_count": count,
                        "granted_ms": now,
                        "expires_ms": now + new_plan.lease_ttl_ms,
                        "status": "assigned",
                        "released_ms": None,
                        "reason": "version_ledger_replacement",
                    }
                )
        records.sort(key=lambda item: (item["batch_id"], item["layer_id"]))
        return assignments, records

    def apply_changes(
        self, edits: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Apply atomic multi-layer edits and recompute the invalidated cone."""

        if not self.results:
            raise ValueError("run inference before applying an incremental change")
        old_plan = self.plan
        old_graph = self.graph
        old_results = dict(self.results)
        old_versions = dict(self.layer_versions)
        old_summaries = self.layer_summaries()
        normalized = {layer_id: dict(updates) for layer_id, updates in edits.items()}
        new_plan = change_layers(old_plan, normalized)
        new_graph = build_graph(new_plan)
        report = new_graph.invalidation_report(
            sorted(normalized),
            reason="declared_field_change",
            changed_fields={layer_id: list(updates) for layer_id, updates in normalized.items()},
        )
        invalidated = set(report["invalidated"])
        new_versions = dict(old_versions)
        for affected in invalidated:
            new_versions[affected] = old_versions[affected] + 1
        new_factory = SampleFactory(new_plan, self.sample_count)
        new_results: dict[tuple[str, str], LayerBatchResult] = {}
        for batch in self.batches:
            available: dict[str, LayerBatchResult] = {}
            # Unaffected results are reused byte-for-byte for every batch.
            for candidate in new_graph.order:
                if candidate not in invalidated and (batch.batch_id, candidate) in old_results:
                    available[candidate] = old_results[(batch.batch_id, candidate)]
            for candidate in new_graph.order:
                if candidate not in invalidated:
                    continue
                layer = new_plan.layer_by_id[candidate]
                computed = compute_layer_result(
                    new_plan,
                    new_factory,
                    layer,
                    batch.sample_ids,
                    batch.batch_id,
                    new_versions[candidate],
                    available,
                )
                available[candidate] = computed
            for candidate in new_graph.order:
                if candidate in invalidated:
                    new_results[(batch.batch_id, candidate)] = available[candidate]
                else:
                    new_results[(batch.batch_id, candidate)] = old_results[
                        (batch.batch_id, candidate)
                    ]
        assignments, new_lease_records = self._build_range_ledger(
            old_results=old_results,
            invalidated=invalidated,
            new_plan=new_plan,
            new_graph=new_graph,
            batches=self.batches,
        )
        # A complete replacement happens only after the new result and ledger
        # structures have been built successfully.
        now = monotonic_ms()
        replaced_results: dict[tuple[str, str], LayerBatchResult] = {}
        for key, result in new_results.items():
            start, count = assignments[key]
            lease_record = next(
                record
                for record in new_lease_records
                if record["batch_id"] == key[0] and record["layer_id"] == key[1]
            )
            replaced_results[key] = self._retag_result(
                result,
                plan_version=new_plan.version,
                layer_version=new_versions[key[1]],
                lease_id=lease_record["lease_id"],
                block_start=start,
                block_count=count,
                granted_ms=now,
                expires_ms=lease_record["expires_ms"],
            )
        revoked = self.version_pool.revoke_plan(
            old_plan.version,
            reason=f"E_REVOKE:plan_version_{old_plan.version}_to_{new_plan.version}",
        )
        retained = [
            record
            for record in self.lease_history
            if record.get("status") not in ("active", "unknown")
        ]
        old_current = list(self.current_lease_ledger)
        self.plan = new_plan
        self.graph = new_graph
        self.factory = new_factory
        self.layer_versions = new_versions
        self.results = replaced_results
        self.restored_layer_summaries = None
        self.current_lease_ledger = new_lease_records
        self.lease_history = retained + new_lease_records
        changed_report = {
            **report,
            "old_plan_version": old_plan.version,
            "new_plan_version": new_plan.version,
            "layer_versions": {key: new_versions[key] for key in new_graph.order},
            "lease_replacement": {
                "atomic": True,
                "revocation_code": "E_REVOKE",
                "revocation_basis": f"plan version {old_plan.version} -> {new_plan.version}",
                "revoked": revoked,
                "released_block_ranges": sorted(
                    {tuple(item["released_block_range"]): item for item in revoked}.values(),
                    key=lambda item: (item["released_block_range"][0], item["lease_id"]),
                ),
                "read_only_retained": retained,
                "old_ledger": old_current,
                "new_leases": new_lease_records,
            },
            "unchanged_lease_ranges": {
                layer: old_summaries[layer]["lease_range"]
                for layer in new_graph.order
                if layer not in invalidated
            },
            "unchanged_layer_versions": {
                layer: new_versions[layer]
                for layer in new_graph.order
                if layer not in invalidated
            },
        }
        self.invalidation_history.append(changed_report)
        if self.ledger is not None:
            self.ledger.record_version_replacement(
                old_plan_version=old_plan.version,
                new_plan_version=new_plan.version,
                revoked=revoked,
                new_leases=new_lease_records,
                basis=f"plan_version_{old_plan.version}_to_{new_plan.version}",
            )
            self.ledger.record_invalidation(new_plan.version, changed_report)
            self.ledger.record_attribution(new_plan.version, self.attribution())
        return changed_report

    def apply_change(self, layer_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
        """Single-layer convenience wrapper around :meth:`apply_changes`."""

        return self.apply_changes({layer_id: updates})

    def verify_incremental(self) -> dict[str, Any]:
        """Compare the current cached subset result with a clean full rerun."""

        full_factory = SampleFactory(self.plan, self.sample_count)
        full_results: dict[tuple[str, str], LayerBatchResult] = {}
        for batch in self.batches:
            computed = compute_batch(
                self.plan,
                self.graph,
                full_factory,
                batch.sample_ids,
                batch.batch_id,
                self.layer_versions,
            )
            for layer_id, result in computed.items():
                full_results[(batch.batch_id, layer_id)] = result
        current = self._comparison_view(self.results)
        full = self._comparison_view(full_results)
        mismatch = self._first_comparison_mismatch(current, full)
        current_attr = self.attribution()
        saved_results = self.results
        try:
            self.results = full_results
            full_attr = self.attribution()
        finally:
            self.results = saved_results
        if mismatch is None and current_attr != full_attr:
            mismatch = {
                "layer_id": current_attr.get("max_deviation_layer"),
                "field": "attribution",
                "left": current_attr,
                "right": full_attr,
            }
        result = {
            "ok": mismatch is None,
            "comparison": "incremental_subset_vs_clean_full_recompute",
            "first_mismatch": mismatch,
            "layers_compared": list(self.graph.order),
            "fields_compared": [
                "float_digest",
                "quant_digest",
                "combined_digest",
                "max_abs_error",
                "mean_abs_error",
                "error_bound",
                "occupancy_bytes",
                "occupancy_blocks",
            ],
        }
        if mismatch is not None:
            raise replay_error(
                "incremental result differs from full recomputation",
                layer_id=mismatch.get("layer_id"),
                field=mismatch.get("field"),
                first_mismatch=mismatch,
            )
        return result

    def _comparison_view(
        self, results: Mapping[tuple[str, str], LayerBatchResult]
    ) -> dict[str, dict[str, Any]]:
        view: dict[str, dict[str, Any]] = {}
        for layer_id in self.graph.order:
            items = [
                result
                for (batch_id, candidate), result in sorted(results.items())
                if candidate == layer_id
            ]
            if not items:
                continue
            aggregate = aggregate_layer_digest(items)
            errors = aggregate_layer_errors(items)
            bounds = {
                key: max(float(item.error_bound[key]) for item in items)
                for key in ("quantization_truncation", "accumulation_order", "lease_reordering")
            }
            bounds["combined"] = sum(bounds.values())
            view[layer_id] = {
                "float_digest": aggregate["float_digest"],
                "quant_digest": aggregate["quant_digest"],
                "combined_digest": aggregate["combined_digest"],
                "max_abs_error": errors["max_abs_error"],
                "mean_abs_error": errors["mean_abs_error"],
                "error_bound": bounds,
                "occupancy_bytes": max(item.occupancy_bytes for item in items),
                "occupancy_blocks": max(item.occupancy_blocks for item in items),
            }
        return view

    @staticmethod
    def _first_comparison_mismatch(
        left: Mapping[str, Mapping[str, Any]],
        right: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        fields = (
            "float_digest",
            "quant_digest",
            "combined_digest",
            "max_abs_error",
            "mean_abs_error",
            "error_bound",
            "occupancy_bytes",
            "occupancy_blocks",
        )
        for layer_id in sorted(set(left) | set(right)):
            if layer_id not in left or layer_id not in right:
                return {"layer_id": layer_id, "field": "<missing>"}
            for field in fields:
                if left[layer_id].get(field) != right[layer_id].get(field):
                    return {
                        "layer_id": layer_id,
                        "field": field,
                        "left": left[layer_id].get(field),
                        "right": right[layer_id].get(field),
                    }
        return None

    def hold_version_leases(self, batch_id: str) -> list[Lease]:
        """Hold real session leases to exercise stale-version revocation."""

        batch = next((item for item in self.batches if item.batch_id == batch_id), None)
        if batch is None:
            raise ValueError(f"unknown batch: {batch_id}")
        leases = self.version_pool.acquire_bundle(
            batch.batch_id,
            self._requests_for_batch(batch),
            ttl_ms=self.plan.lease_ttl_ms,
            plan_version=self.plan.version,
        )
        for lease in leases:
            self.held_leases[lease.lease_id] = lease
        self.current_lease_ledger = [lease.to_dict() for lease in leases]
        if self.ledger is not None:
            self.ledger.record_held_leases(self.plan.version, leases)
        return leases

    def release_held_leases(self) -> list[dict[str, Any]]:
        released: list[dict[str, Any]] = []
        for lease_id in sorted(self.held_leases):
            lease = self.held_leases[lease_id]
            try:
                released.append(
                    self.version_pool.release(lease_id, batch_id=lease.batch_id, layer_id=lease.layer_id)
                )
            except DiagnosticError:
                released.append(self.version_pool.release(lease_id, batch_id=lease.batch_id))
        self.held_leases.clear()
        return released

    def snapshot_payload(
        self,
        *,
        ledger_content_digest: str | None = None,
        cursor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from .snapshot import build_snapshot

        return build_snapshot(
            self.plan,
            sample_count=self.sample_count,
            dataset_digest=self.factory.dataset_digest(),
            lease_records=self.current_lease_ledger or self.lease_history,
            layer_summaries=self.layer_summaries(),
            processed_batch_cursor=cursor,
            ledger_content_digest=(
                ledger_content_digest
                if ledger_content_digest is not None
                else (None if self.ledger is None else self.ledger.content_digest())
            ),
        )

    def restore_snapshot(
        self,
        snapshot_path: str,
        *,
        current_ledger_digest: str | None = None,
    ) -> dict[str, Any]:
        """Replace the complete in-memory plan, leases and summaries from a snapshot."""

        from .snapshot import load_snapshot

        loaded = load_snapshot(
            snapshot_path,
            current_plan=self.plan,
            current_dataset_digest=self.factory.dataset_digest(),
            current_ledger_digest=current_ledger_digest,
            current_lease_records=self.current_lease_ledger,
        )
        new_plan = loaded.plan
        new_factory = SampleFactory(new_plan, loaded.sample_count)
        new_batches = make_batches(new_factory.ids, min(self.batch_size, new_plan.batch_size_max))
        new_graph = build_graph(new_plan)
        restored_versions: dict[str, int] = {}
        for layer in new_plan.layers:
            summary = loaded.layer_summaries.get(layer.id, {})
            version = summary.get("layer_version", 1)
            restored_versions[layer.id] = int(version) if isinstance(version, int) else min(version)
        # Replacement is deliberately sparse: no arrays are merged from the
        # previous in-memory generation.
        self.plan = new_plan
        self.sample_count = loaded.sample_count
        self.batch_size = min(self.batch_size, new_plan.batch_size_max)
        self.factory = new_factory
        self.batches = new_batches
        self.graph = new_graph
        self.layer_versions = restored_versions
        self.results = {}
        self.restored_layer_summaries = loaded.layer_summaries
        self.current_lease_ledger = list(loaded.lease_records)
        self.lease_history = list(loaded.lease_records)
        self.invalidation_history = []
        self.last_schedule = None
        self.last_attribution = None
        self.held_leases = {}
        self.version_pool = LeasePool(
            block_size_bytes=new_plan.block_size_bytes,
            memory_budget_bytes=new_plan.memory_budget_bytes,
            id_prefix="H",
        )
        result = loaded.to_dict()
        result["memory_replaced"] = True
        return result


def compare_stored_result(
    computed: LayerBatchResult,
    stored: Mapping[str, Any],
    *,
    layer_order: Sequence[str],
) -> dict[str, Any] | None:
    fields = (
        "float_digest",
        "quant_digest",
        "per_sample",
        "errors",
        "error_bound",
        "occupancy_bytes",
        "occupancy_blocks",
        "input_digest",
    )
    # JSON normalises tuples/lists; compare the canonical stored forms.
    computed_payload = computed.ledger_payload()
    for field in fields:
        left = computed_payload.get(field)
        right = stored.get(field)
        if field == "per_sample" and isinstance(right, dict):
            right = {key: right[key] for key in sorted(right)}
            left = {key: left[key] for key in sorted(left)}
        if left != right:
            return {
                "layer_id": computed.layer_id,
                "field": field,
                "left": left,
                "right": right,
            }
    return None


def recover_and_resume(
    *,
    plan: Plan,
    sample_count: int,
    batch_size: int,
    ledger: Ledger,
    worker_count: int = 4,
) -> dict[str, Any]:
    """Verify committed rows, mark torn batches unknown, and resume missing work."""

    integrity = ledger.verify_integrity()
    pre_recovery = ledger.recovery_view()
    pre_attribution = ledger.load_attribution(plan.version)
    unknown = ledger.mark_unknown_uncommitted()
    committed = ledger.load_committed_results(plan.version)
    lease_rows = {row["lease_id"]: row for row in ledger.load_leases(plan.version)}
    factory = SampleFactory(plan, sample_count)
    graph = build_graph(plan)
    batches = make_batches(factory.ids, batch_size)
    first_mismatch: dict[str, Any] | None = None
    compared_layers = 0
    compared_leases = 0
    all_committed_before = True
    for batch in batches:
        # A batch is trustworthy only when every layer has a valid committed row.
        if not all((batch.batch_id, layer_id) in committed for layer_id in graph.order):
            all_committed_before = False
            continue
        versions = {
            layer_id: int(committed[(batch.batch_id, layer_id)]["layer_version"])
            for layer_id in graph.order
        }
        computed = compute_batch(
            plan,
            graph,
            factory,
            batch.sample_ids,
            batch.batch_id,
            versions,
        )
        for layer_id in graph.order:
            stored = committed[(batch.batch_id, layer_id)]
            lease_id = stored.get("lease_id")
            lease_row = lease_rows.get(str(lease_id)) if lease_id else None
            compared_leases += 1
            expected_lease = {
                "batch_id": batch.batch_id,
                "layer_id": layer_id,
                "block_start": stored.get("block_start"),
                "block_count": stored.get("block_count"),
            }
            actual_lease = None
            if lease_row is not None:
                actual_lease = {
                    "batch_id": lease_row["batch_id"],
                    "layer_id": lease_row["layer_id"],
                    "block_start": lease_row["block_start"],
                    "block_count": lease_row["block_count"],
                }
            if actual_lease != expected_lease:
                first_mismatch = {
                    "layer_id": layer_id,
                    "field": "lease_identity_or_block_range",
                    "left": expected_lease,
                    "right": actual_lease,
                }
                break
            if lease_row["status"] not in ("released", "expired", "revoked"):
                first_mismatch = {
                    "layer_id": layer_id,
                    "field": "lease_status",
                    "left": "read_only_released_or_expired",
                    "right": lease_row["status"],
                }
                break
            mismatch = compare_stored_result(
                computed[layer_id],
                stored,
                layer_order=graph.order,
            )
            compared_layers += 1
            if mismatch is not None and first_mismatch is None:
                first_mismatch = mismatch
                break
        if first_mismatch is not None:
            break
    if first_mismatch is not None:
        raise replay_error(
            "replay verification found a ledger mismatch",
            layer_id=first_mismatch.get("layer_id"),
            field=first_mismatch.get("field"),
            first_mismatch=first_mismatch,
        )
    complete_batch_ids = {
        batch.batch_id
        for batch in batches
        if all((batch.batch_id, layer_id) in committed for layer_id in graph.order)
    }
    missing = [batch for batch in batches if batch.batch_id not in complete_batch_ids]
    runtime = InferenceRuntime(
        plan,
        sample_count,
        batch_size=batch_size,
        ledger=ledger,
    )
    continuation: dict[str, Any] | None = None
    if missing:
        continuation = runtime.run(worker_count=worker_count, batches=missing)
    full_results: dict[str, list[LayerBatchResult]] = {layer_id: [] for layer_id in graph.order}
    versions_after = ledger.load_committed_results(plan.version)
    for batch in batches:
        batch_versions = {
            layer_id: int(versions_after[(batch.batch_id, layer_id)]["layer_version"])
            for layer_id in graph.order
            if (batch.batch_id, layer_id) in versions_after
        }
        if len(batch_versions) != len(graph.order):
            continue
        computed = compute_batch(
            plan,
            graph,
            factory,
            batch.sample_ids,
            batch.batch_id,
            batch_versions,
        )
        for layer_id in graph.order:
            full_results[layer_id].append(computed[layer_id])
    complete_full = all(full_results[layer_id] for layer_id in graph.order)
    attribution_verified = False
    attribution_recorded = False
    if complete_full:
        full_attribution = attribute_errors(plan, graph, full_results)
        if pre_attribution is not None and all_committed_before:
            if pre_attribution != full_attribution:
                raise replay_error(
                    "stored attribution differs from replay attribution",
                    layer_id=pre_attribution.get("max_deviation_layer"),
                    field="attribution",
                    expected=pre_attribution,
                    actual=full_attribution,
                )
            attribution_verified = True
        ledger.record_attribution(plan.version, full_attribution)
        attribution_recorded = True
    return {
        "ok": continuation is None or continuation.get("ok", False),
        "integrity": integrity,
        "pre_recovery": pre_recovery,
        "unknown_batches": unknown["unknown_batches"],
        "unknown_leases": unknown["unknown_leases"],
        "uncommitted_exit_report_available": False,
        "replay_verification": {
            "ok": True,
            "layers_compared": compared_layers,
            "leases_compared": compared_leases,
            "attribution_verified_against_stored": attribution_verified,
            "attribution_recorded_after_resume": attribution_recorded,
            "first_mismatch": None,
        },
        "resumed_batches": [batch.batch_id for batch in missing],
        "continuation": continuation,
        "complete_after_resume": continuation is None or continuation.get("ok", False),
    }
