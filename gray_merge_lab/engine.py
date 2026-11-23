from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .canonical import sha256_value
from .deterministic import (
    BackendParameters,
    InputTensor,
    build_backend_parameters,
    build_input_tensor,
    forward_fixed,
)
from .errors import LabError
from .fixedpoint import BatchFixedStats, validate_scale_exp
from .ledger import Ledger
from .merge import merge_receipts
from .receipts import Receipt, make_receipt
from .routing import BatchPlan, plan_batches
from .snapshot import SnapshotStore, build_manifest, build_payload
from .specs import (
    BackendSpec,
    LoadedSpec,
    RuleSpec,
    SampleSpec,
    validate_backend,
)


@dataclass(frozen=True)
class InferenceResult:
    output_digest: str
    sample_outputs: list[dict[str, Any]]
    duration_ms: int
    backend_param_digest: str
    backend_param_version: int
    input_digests: list[str]


class InferenceTransport(Protocol):
    def infer(self, backend_id: str, samples: list[SampleSpec], batch_id: str) -> InferenceResult:
        ...

    def update_backend(self, backend_id: str, spec: BackendSpec, version: int) -> None:
        ...


class ModelBank:
    def __init__(self, spec: LoadedSpec) -> None:
        self.spec = spec
        self.versions: dict[str, int] = {item.backend_id: 1 for item in spec.backends}
        self.params: dict[str, BackendParameters] = {}
        for item in spec.backends:
            self.params[item.backend_id] = build_backend_parameters(item, self.versions[item.backend_id])

    def get(self, backend_id: str) -> BackendParameters:
        try:
            return self.params[backend_id]
        except KeyError as exc:
            raise LabError("E_RUNTIME", f"unknown backend {backend_id!r}", field="backend_id") from exc

    def summaries(self) -> list[dict[str, Any]]:
        return [self.params[key].summary() for key in sorted(self.params)]

    def update(self, backend_id: str, changes: dict[str, Any]) -> BackendParameters:
        if backend_id not in self.params:
            raise LabError("E_SPEC", "unknown backend_id", field="backend_id")
        forbidden = set(changes) & {"backend_id", "host", "port"}
        if forbidden:
            field = sorted(forbidden)[0]
            raise LabError("E_SPEC", "hot update cannot change backend_id, host or port", field=field)
        current = next(item for item in self.spec.backends if item.backend_id == backend_id)
        candidate = current.evolve(**changes)
        other_ids = {item.backend_id for item in self.spec.backends if item.backend_id != backend_id}
        other_ports = {item.port for item in self.spec.backends if item.backend_id != backend_id}
        validate_backend(candidate.declaration(), 0, other_ids, other_ports)
        version = self.versions[backend_id] + 1
        params = build_backend_parameters(candidate, version)
        self.versions[backend_id] = version
        self.params[backend_id] = params
        self.spec = LoadedSpec(
            backends=tuple(
                candidate if item.backend_id == backend_id else item for item in self.spec.backends
            ),
            rules=self.spec.rules,
            samples=self.spec.samples,
        )
        return params


class InProcessTransport:
    def __init__(
        self,
        bank: ModelBank,
        duration_provider: Callable[[str, str, int], int] | None = None,
    ) -> None:
        self.bank = bank
        self.duration_provider = duration_provider

    def update_backend(self, backend_id: str, spec: BackendSpec, version: int) -> None:
        del backend_id, spec, version

    def infer(self, backend_id: str, samples: list[SampleSpec], batch_id: str) -> InferenceResult:
        params = self.bank.get(backend_id)
        inputs = [build_input_tensor(sample, params.model_dim) for sample in samples]
        if self.duration_provider is None:
            start = time.perf_counter_ns()
            output_digest, rows, _ = forward_fixed(params, inputs)
            elapsed = time.perf_counter_ns() - start
            duration_ms = max(1, (elapsed + 999_999) // 1_000_000)
        else:
            output_digest, rows, _ = forward_fixed(params, inputs)
            duration_ms = int(self.duration_provider(backend_id, batch_id, len(samples)))
        if duration_ms < 0:
            raise LabError("E_RUNTIME", "duration provider returned a negative duration")
        return InferenceResult(
            output_digest=output_digest,
            sample_outputs=rows,
            duration_ms=duration_ms,
            backend_param_digest=params.param_digest,
            backend_param_version=params.version,
            input_digests=[item.digest for item in inputs],
        )


@dataclass(frozen=True)
class BatchExecution:
    plan: BatchPlan
    record: dict[str, Any]
    conclusion: dict[str, Any]
    receipts: tuple[dict[str, Any], ...]


class GatewayEngine:
    def __init__(
        self,
        spec: LoadedSpec,
        ledger: Ledger,
        *,
        scale_exp: int = 3,
        snapshot_path: str | Path | None = None,
        transport: InferenceTransport | None = None,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.scale_exp = validate_scale_exp(scale_exp)
        if transport is not None and hasattr(transport, "bank"):
            self.bank = transport.bank  # type: ignore[attr-defined]
        else:
            self.bank = ModelBank(spec)
        self.transport = transport or InProcessTransport(self.bank)
        self.snapshot_store = SnapshotStore(snapshot_path) if snapshot_path is not None else None

    def manifest(self) -> dict[str, Any]:
        return build_manifest(self.spec, self.bank.summaries(), self.scale_exp)

    def payload(self) -> dict[str, Any]:
        return build_payload(self.ledger, self.scale_exp)

    def refresh_snapshot(self) -> dict[str, Any] | None:
        if self.snapshot_store is None:
            return None
        return self.snapshot_store.write(self.manifest(), self.payload())

    def initialize_snapshot(self) -> dict[str, Any] | None:
        return self.refresh_snapshot()

    def hot_update_backend(self, backend_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        params = self.bank.update(backend_id, changes)
        self.spec = self.bank.spec
        updated = next(item for item in self.spec.backends if item.backend_id == backend_id)
        self.transport.update_backend(backend_id, updated, params.version)
        self.refresh_snapshot()
        return params.summary()

    def _target_backend(self, plan: BatchPlan) -> tuple[str, RuleSpec]:
        rule = next(item for item in self.spec.rules if item.rule_id == plan.rule_id)
        target = rule.stable_backend if plan.destination == "stable" else rule.canary_backend
        return target, rule

    def _receipts(
        self,
        plan: BatchPlan,
        result: InferenceResult,
    ) -> tuple[list[Receipt], list[dict[str, Any]]]:
        shared = {
            "client_bucket": plan.buckets[0],
            "weight_bp": plan.canary_weight_bp,
            "duration_ms": result.duration_ms,
        }
        h1 = make_receipt("h1", plan.batch_id, plan.destination, result.output_digest)
        h2 = make_receipt("h2", plan.batch_id, plan.destination, result.output_digest, **shared)
        # The h3 merge sequence grows with parameter version. h1 and h2 have
        # implied sequences 0 and 1, so the first h3 is sequence 2.
        h3 = make_receipt(
            "h3",
            plan.batch_id,
            plan.destination,
            result.output_digest,
            sample_ids=plan.sample_ids(),
            backend_param_digest=result.backend_param_digest,
            backend_param_version=result.backend_param_version,
            merge_seq=result.backend_param_version + 1,
            **shared,
        )
        items = [h1, h2, h3]
        return items, [item.to_dict() for item in items]

    def _record_for_plan(
        self,
        plan: BatchPlan,
        result: InferenceResult,
        receipts: list[Receipt],
        raw_receipts: list[dict[str, Any]],
        conclusion: dict[str, Any],
        rule: RuleSpec,
    ) -> dict[str, Any]:
        fixed = BatchFixedStats(
            batch_id=plan.batch_id,
            raw_duration_ms=result.duration_ms,
            raw_samples=len(plan.samples),
            scale_exp=self.scale_exp,
        )
        preview = {
            "batch_id": plan.batch_id,
            "batch_index": 0,
            "rule_id": plan.rule_id,
            "destination": plan.destination,
            "client_key": plan.samples[0].client_key,
            "client_bucket": plan.buckets[0],
            "canary_weight_bp": plan.canary_weight_bp,
            "sample_ids": plan.sample_ids(),
            "raw_duration_ms": result.duration_ms,
            "raw_sample_count": len(plan.samples),
            "duration_q": fixed.duration_q,
            "samples_q": fixed.samples_q,
            "avg_duration_q": fixed.duration_q,
            "avg_samples_q": fixed.samples_q,
            "scale_exp": self.scale_exp,
            "truncation": "toward_zero",
            "single_value_error_bound_original": fixed.single_value_error_original,
            "output_digest": result.output_digest,
            "backend_param_digest": result.backend_param_digest,
            "backend_param_version": result.backend_param_version,
            "conclusion_digest": conclusion["conclusion_digest"],
            "sampling_error_affected": False,
        }
        param_versions = {result.backend_param_digest: result.backend_param_version}
        merge_input = {
            "receipts": raw_receipts,
            "expected_sample_ids": plan.sample_ids(),
            "param_versions": param_versions,
            "rule_id": plan.rule_id,
        }
        routing_input = {
            "rule": rule.declaration(),
            "samples": [sample.declaration() for sample in plan.samples],
            "buckets": list(plan.buckets),
            "destination": plan.destination,
            "batch_id": plan.batch_id,
            "batch_size": rule.batch_size,
        }
        return {
            "batch_id": plan.batch_id,
            "rule_id": plan.rule_id,
            "destination": plan.destination,
            "sample_ids": plan.sample_ids(),
            "raw_duration_ms": result.duration_ms,
            "raw_sample_count": len(plan.samples),
            "scale_exp": self.scale_exp,
            "duration_q": fixed.duration_q,
            "samples_q": fixed.samples_q,
            "output_digest": result.output_digest,
            "backend_param_digest": result.backend_param_digest,
            "backend_param_version": result.backend_param_version,
            "input_digests": result.input_digests,
            "conclusion": conclusion,
            "merge_input": merge_input,
            "routing_input": routing_input,
            "fixed_stats": fixed.to_dict(),
            "preview": preview,
            "record": routing_input,
        }

    def _execute_plan(self, plan: BatchPlan) -> BatchExecution:
        target_backend, rule = self._target_backend(plan)
        result = self.transport.infer(target_backend, list(plan.samples), plan.batch_id)
        receipts, raw_receipts = self._receipts(plan, result)
        conclusion = merge_receipts(
            receipts,
            expected_sample_ids=plan.sample_ids(),
            param_versions={result.backend_param_digest: result.backend_param_version},
            rule_id=plan.rule_id,
        )
        record = self._record_for_plan(plan, result, receipts, raw_receipts, conclusion, rule)
        stored = self.ledger.record_batch(record, raw_receipts)
        self.refresh_snapshot()
        return BatchExecution(plan=plan, record=stored, conclusion=conclusion, receipts=tuple(raw_receipts))

    def process_samples(self, samples: Iterable[SampleSpec]) -> list[BatchExecution]:
        plans = plan_batches(samples, self.spec.rules)
        if not plans:
            raise LabError("E_SPEC", "no samples matched a routing rule", field="samples")
        return [self._execute_plan(plan) for plan in plans]

    def plan(self, samples: Iterable[SampleSpec]) -> list[BatchPlan]:
        return plan_batches(samples, self.spec.rules)

    def reinfer_existing_batch(self, batch_id: str) -> dict[str, Any]:
        """Append a new h3 receipt after a hot update and re-merge the batch.

        h1/h2 remain untouched. The h3 merge sequence is tied to the current
        parameter version, so it supersedes the pre-update receipt deterministically.
        """
        existing = self.ledger.batch(batch_id)
        if existing is None:
            raise LabError("E_RUNTIME", "batch does not exist", batch_id=batch_id)
        routing = existing["routing_input"]
        samples = [SampleSpec(**item) for item in routing["samples"]]
        rule = RuleSpec(**routing["rule"])
        target_backend = rule.stable_backend if existing["destination"] == "stable" else rule.canary_backend
        result = self.transport.infer(target_backend, samples, batch_id)
        merge_seq = result.backend_param_version + 1
        old_receipts = self.ledger.receipt_rows(batch_id)
        if any(
            item.get("h") == "h3" and item.get("merge_seq") == merge_seq for item in old_receipts
        ):
            raise LabError(
                "E_FORMAT",
                "an h3 receipt already exists at the current merge sequence",
                batch_id=batch_id,
                details={"merge_seq": merge_seq},
            )
        new_h3 = make_receipt(
            "h3",
            batch_id,
            existing["destination"],
            result.output_digest,
            client_bucket=routing["buckets"][0],
            weight_bp=rule.canary_weight_bp,
            duration_ms=result.duration_ms,
            sample_ids=existing["sample_ids"],
            backend_param_digest=result.backend_param_digest,
            backend_param_version=result.backend_param_version,
            merge_seq=merge_seq,
        )
        raw_receipts = old_receipts + [new_h3.to_dict()]
        param_versions = {
            item["backend_param_digest"]: item["backend_param_version"]
            for item in raw_receipts
            if item.get("backend_param_digest") and item.get("backend_param_version")
        }
        conclusion = merge_receipts(
            raw_receipts,
            expected_sample_ids=existing["sample_ids"],
            param_versions=param_versions,
            rule_id=existing["rule_id"],
        )
        fixed = BatchFixedStats(
            batch_id=batch_id,
            raw_duration_ms=result.duration_ms,
            raw_samples=len(samples),
            scale_exp=self.scale_exp,
        )
        record = dict(existing)
        record.update(
            {
                "raw_duration_ms": result.duration_ms,
                "raw_sample_count": len(samples),
                "duration_q": fixed.duration_q,
                "samples_q": fixed.samples_q,
                "output_digest": conclusion["output_digest"],
                "backend_param_digest": conclusion["backend_param_digest"],
                "backend_param_version": conclusion["backend_param_version"],
                "input_digests": result.input_digests,
                "conclusion": conclusion,
                "merge_input": {
                    "receipts": raw_receipts,
                    "expected_sample_ids": existing["sample_ids"],
                    "param_versions": param_versions,
                    "rule_id": existing["rule_id"],
                },
                "fixed_stats": fixed.to_dict(),
            }
        )
        preview = dict(record["preview"])
        preview.update(
            {
                "raw_duration_ms": result.duration_ms,
                "duration_q": fixed.duration_q,
                "avg_duration_q": fixed.duration_q,
                "output_digest": conclusion["output_digest"],
                "backend_param_digest": conclusion["backend_param_digest"],
                "backend_param_version": conclusion["backend_param_version"],
                "conclusion_digest": conclusion["conclusion_digest"],
            }
        )
        record["preview"] = preview
        stored = self.ledger.record_batch(record, raw_receipts)
        self.refresh_snapshot()
        return {
            "batch_id": batch_id,
            "adopted_merge_seq": conclusion["merge_seq"],
            "backend_param_version": conclusion["backend_param_version"],
            "receipt_count": len(raw_receipts),
            "rejected_count": len(conclusion["rejected"]),
            "conclusion": conclusion,
            "record": stored,
        }
