"""Top-level arbitration orchestration.

The engine pushes deterministic observations into per-environment subprocess
services over real TCP sockets, collects environment authority receipts, runs
the three deterministic merge policies with fixed-point quantization, derives
explicit error bounds, and commits each view atomically to the SQLite ledger.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .fixedpoint import (
    FixedPoint,
    per_key_error_bound,
    view_aggregate_bound,
)
from .ledger import KeyRecord, Ledger
from .merge import merge_key
from .planning import PathDecision, decide
from .propagation import PropagationEngine, ViewInvalidation
from .sample import DEFAULT_KEYS, Observation, generate_all
from .service_manager import ServiceManager
from .spec import Declarations
from .errors import OrderError, ErrorContext


@dataclass
class ArbitrationContext:
    decls: Declarations
    fp: FixedPoint
    observations: dict[str, list[Observation]]
    receipts: dict[str, dict]
    digests: dict[str, dict[str, str]]


class ArbitrationEngine:
    def __init__(self, decls: Declarations, scale_exponent: int, ledger: Ledger,
                 batch_size: int = 7):
        self.decls = decls
        self.fp = FixedPoint(scale_exponent)
        self.ledger = ledger
        self.batch_size = batch_size
        self.propagation = PropagationEngine(decls)
        self.ctx: ArbitrationContext | None = None

    # ---- environment services over real sockets -------------------------
    def start_services_and_ingest(self, manager: ServiceManager) -> ArbitrationContext:
        observations = generate_all(self.decls)
        receipts: dict[str, dict] = {}
        digests: dict[str, dict[str, str]] = {}
        for env_id in self.decls.environments:
            env_obs = observations[env_id]
            batch_receipts = manager.push_in_batches(env_id, env_obs, self.batch_size)
            arb_receipt = manager.arbitrate(env_id)
            receipts[env_id] = {
                "batch_receipts": batch_receipts,
                "arbitration": arb_receipt,
            }
            digests[env_id] = arb_receipt.get("keys", {})
        self.ctx = ArbitrationContext(
            decls=self.decls, fp=self.fp, observations=observations,
            receipts=receipts, digests=digests,
        )
        return self.ctx

    # ---- per-view arbitration -------------------------------------------
    def arbitrate_view(
        self,
        view_id: str,
        version: int,
        only_keys: tuple[str, ...] | None = None,
    ) -> list[KeyRecord]:
        if self.ctx is None:
            raise RuntimeError("ingestion must run before arbitration")
        view = self.decls.views[view_id]
        weights = {env_id: self.decls.environments[env_id].weight
                   for env_id in view.envs}
        keys = tuple(only_keys) if only_keys is not None else self.decls.keys
        records: list[KeyRecord] = []
        per_key_bounds = []
        for key in keys:
            policy = self.decls.policy_for(view_id, key)
            result = merge_key(
                policy=policy,
                key=key,
                env_order=view.envs,
                observations_by_env=self.ctx.observations,
                weights=weights,
                digests=self.ctx.digests,
                fp=self.fp,
            )
            if result is None:
                continue
            bound = per_key_error_bound(self.fp, policy, len(view.envs))
            per_key_bounds.append(bound)
            records.append(
                KeyRecord(
                    view_id=view_id,
                    version=version,
                    key=key,
                    policy=policy,
                    authority_q=result.authority_q,
                    adopted_envs=result.adopted_envs,
                    rejected_envs=result.rejected_envs,
                    error_quant=bound.quantization_truncation,
                    error_boundary=bound.interval_boundary,
                    error_norm=bound.weight_normalization,
                    error_total=bound.total(),
                    attribution={
                        "quantization_truncation_units": bound.quantization_truncation,
                        "interval_boundary_units": bound.interval_boundary,
                        "weight_normalization_units": bound.weight_normalization,
                        "scale_exponent": self.fp.exponent,
                        "quantum": str(self.fp.quantum),
                        "tie_break": result.tie_break,
                    },
                    evidence=result.evidence,
                )
            )
        aggregate = view_aggregate_bound(per_key_bounds)
        for record in records:
            record.attribution["view_aggregate_upper_bound"] = {
                "quantization_truncation_units": aggregate.quantization_truncation,
                "interval_boundary_units": aggregate.interval_boundary,
                "weight_normalization_units": aggregate.weight_normalization,
            }
        return records

    def commit_view(self, view_id: str, version: int,
                    records: list[KeyRecord]) -> str:
        return self.ledger.commit_view(view_id, version, records)

    # ---- path selection --------------------------------------------------
    def choose_path(self, view_id: str, dirty_keys: tuple[str, ...] | None) -> PathDecision:
        view = self.decls.views[view_id]
        obs_total = sum(
            len(self.ctx.observations[env_id]) for env_id in view.envs
        ) if self.ctx else 0
        changed_env_count = len(view.envs) if dirty_keys is None else 1
        dirty_count = len(self.decls.keys) if dirty_keys is None else len(dirty_keys)
        return decide(
            env_count=len(view.envs),
            key_count=len(self.decls.keys),
            obs_total=obs_total,
            changed_env_count=changed_env_count,
            dirty_key_count=dirty_count,
        )

    # ---- full vs incremental runs ---------------------------------------
    def run_full(self) -> dict:
        outputs: dict[str, dict] = {}
        for view_id in sorted(self.decls.views):
            decision = self.choose_path(view_id, None)
            invalidation = self.propagation.mark_view_dirty_all(view_id)
            records = self.arbitrate_view(view_id, invalidation.new_version)
            digest = self.commit_view(view_id, invalidation.new_version, records)
            outputs[view_id] = {
                "path": "full",
                "decision": decision,
                "invalidation": invalidation,
                "digest": digest,
                "records": records,
            }
        return outputs

    def run_incremental(self, invalidations: list[ViewInvalidation]) -> dict:
        outputs: dict[str, dict] = {}
        for invalidation in invalidations:
            view_id = invalidation.view_id
            decision = self.choose_path(view_id, invalidation.dirty_keys)
            records = self.arbitrate_view(
                view_id, invalidation.new_version, invalidation.dirty_keys
            )
            digest = self.commit_view(view_id, invalidation.new_version, records)
            outputs[view_id] = {
                "path": "incremental",
                "decision": decision,
                "invalidation": invalidation,
                "digest": digest,
                "records": records,
            }
        return outputs

    # ---- determinism guard ----------------------------------------------
    @staticmethod
    def assert_byte_identical(runs: list[dict]) -> None:
        """Compare serialized outputs across orderings/batch splits."""
        serialized = [
            json.dumps(run, sort_keys=True, separators=(",", ":"), default=str)
            for run in runs
        ]
        reference = serialized[0]
        for index, blob in enumerate(serialized[1:], start=1):
            if blob != reference:
                # locate first unstable key deterministically
                ref = json.loads(reference)
                cur = json.loads(blob)
                for view_id in sorted(set(ref) | set(cur)):
                    a = ref.get(view_id, {})
                    b = cur.get(view_id, {})
                    for record_a, record_b in zip(
                        a.get("records", []), b.get("records", [])
                    ):
                        if record_a != record_b:
                            raise OrderError(
                                "arbitration output differs across runs",
                                ErrorContext(
                                    view_id, record_a.get("key", ""),
                                    f"run_0_vs_run_{index}",
                                ),
                            )
                raise OrderError(
                    "arbitration output differs across runs",
                    ErrorContext(detail=f"run_0_vs_run_{index}"),
                )
