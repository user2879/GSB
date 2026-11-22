"""End-to-end staged workbench pipeline and JSON audit report assembly."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .cache import GroupDecisionCache, SignatureCache
from .codes import (
    E_CONFIG_INVALID,
    E_GLOBAL_BUDGET_EXHAUSTED,
    E_SOURCE_QUOTA_EXCEEDED,
    E_SNAPSHOT_STALE,
)
from .config import (
    BalanceConfig,
    ConfigValidationError,
    DedupConfig,
    InjectionItem,
    SchedulerConfig,
    load_all_configs,
)
from .dedup import deduplicate
from .loader import LoadResult, load_documents
from .models import Document, Failure, SampleRecord, SignatureRecord, sorted_failures
from .sampling import allocate_source_counts, sample_documents
from .scheduler import BudgetScheduler
from .snapshots import SnapshotBundle, SnapshotStaleError, SnapshotStore
from .util import safe_relative, sha256_json, write_json_atomic


class Workbench:
    """Composable workbench façade.

    All paths may be relative. Generated snapshots, cache and reports stay below
    ``output_root``. A run is immutable unless ``reset=True``; use ``resume=True``
    to continue a snapshot after budget or checkpoint exhaustion.
    """

    def __init__(
        self,
        data_path: str | Path,
        balance_path: str | Path,
        dedup_path: str | Path,
        scheduler_path: str | Path,
        injection_path: str | Path | None = None,
        output_root: str | Path = "output",
        run_id: str = "demo",
        cache_root: str | Path | None = None,
    ):
        self.data_path = Path(data_path)
        self.balance_path = Path(balance_path)
        self.dedup_path = Path(dedup_path)
        self.scheduler_path = Path(scheduler_path)
        self.injection_path = Path(injection_path) if injection_path is not None else None
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.cache_root = Path(cache_root) if cache_root is not None else self.output_root / "cache"
        self.snapshot_store = SnapshotStore(self.output_root, run_id)
        self.report_dir = self.output_root / "reports"
        self.result_dir = self.output_root / "results"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        resume: bool = False,
        expected_version: int | None = None,
        stop_after_batches: int | None = None,
        reset: bool = False,
    ) -> dict[str, Any]:
        if reset:
            self.snapshot_store.reset()
            resume = False
        if resume and not self.snapshot_store.exists():
            failure = Failure(
                source="",
                doc_id="",
                stage="snapshot",
                code=E_SNAPSHOT_STALE,
                message="resume requested but no snapshot CURRENT pointer exists",
            )
            return self._write_plain_failure_report("stale", [failure], {})
        if not resume and self.snapshot_store.exists():
            failure = Failure(
                source="",
                doc_id="",
                stage="snapshot",
                code=E_SNAPSHOT_STALE,
                message="run snapshot already exists; use resume=True or reset=True",
            )
            return self._write_plain_failure_report("stale", [failure], {})

        start = time.perf_counter()
        stage_elapsed_ms: dict[str, float] = {}
        try:
            balance, dedup, scheduler, injections = load_all_configs(
                self.balance_path,
                self.dedup_path,
                self.scheduler_path,
                self.injection_path,
            )
        except ConfigValidationError as exc:
            failures = sorted_failures(exc.failures)
            return self._write_plain_failure_report(
                "config_invalid",
                failures,
                {"elapsed_total_ms": round((time.perf_counter() - start) * 1000, 3)},
            )

        config_hashes = {
            "balance": balance.hash(),
            "dedup": dedup.hash(),
            "scheduler": scheduler.hash(),
        }

        load_start = time.perf_counter()
        loaded = load_documents(self.data_path, balance.source_weights, injections)
        stage_elapsed_ms["load_validate"] = round((time.perf_counter() - load_start) * 1000, 3)

        if loaded.fatal:
            report = self._base_report(
                status="input_invalid",
                complete=False,
                failures=loaded.failures,
                loaded=loaded,
                balance=balance,
                dedup=dedup,
                scheduler=scheduler,
                injections=injections,
                config_hashes=config_hashes,
                stage_elapsed_ms=stage_elapsed_ms,
                start=start,
            )
            report["documents"] = {
                "loaded_valid": 0,
                "loaded_invalid": len(loaded.failures),
                "sampled_unique_before_dedup": 0,
                "after_dedup": 0,
                "retained": 0,
                "removed_as_duplicate": 0,
                "signature_failed": 0,
            }
            report["deterministic_result_digest"] = sha256_json(
                {"status": "input_invalid", "failures": report["failures"]}
            )
            self._write_report(report, None)
            return report

        bundle: SnapshotBundle | None = None
        if resume:
            try:
                bundle = self.snapshot_store.load_current()
                self.snapshot_store.verify_resume(
                    bundle,
                    expected_version=expected_version,
                    balance_hash=config_hashes["balance"],
                    dedup_hash=config_hashes["dedup"],
                    corpus_digest=loaded.corpus_digest,
                )
            except SnapshotStaleError as exc:
                report = self._base_report(
                    status="stale",
                    complete=False,
                    failures=[exc.failure()],
                    loaded=loaded,
                    balance=balance,
                    dedup=dedup,
                    scheduler=scheduler,
                    injections=injections,
                    config_hashes=config_hashes,
                    stage_elapsed_ms=stage_elapsed_ms,
                    start=start,
                )
                report["global_version"] = bundle.version
                report["snapshot"] = {
                    "reused": False,
                    "version": bundle.version,
                    "stage_versions": bundle.manifest.get("stage_versions", {}),
                    "stage_statuses": bundle.manifest.get("stage_statuses", {}),
                    "stale_details": exc.details,
                }
                report["deterministic_result_digest"] = sha256_json(
                    {
                        "status": "stale",
                        "version": bundle.version,
                        "failures": report["failures"],
                    }
                )
                self._write_report(report, None)
                return report

        state, stage_elapsed_ms = self._state_for_run(bundle, config_hashes, loaded.corpus_digest, stage_elapsed_ms)

        def commit(stage_name: str, stage_status: str) -> int:
            nonlocal state
            version, state = self.snapshot_store.commit(
                state,
                stage_name=stage_name,
                stage_status=stage_status,
                config_hashes=config_hashes,
                corpus_digest=loaded.corpus_digest,
                stage_elapsed_ms=stage_elapsed_ms,
            )
            return version

        corpus_changed = bool(
            bundle is not None
            and bundle.manifest.get("corpus_digest") != loaded.corpus_digest
        )
        validation_complete = bool(
            bundle is not None
            and state.get("validation") is not None
            and state.get("stage_statuses", {}).get("validation") == "complete"
            and not corpus_changed
        )
        if not validation_complete:
            state["validation"] = {
                "documents": [document.to_dict(include_text=True) for document in loaded.documents],
                "failures": [failure.to_dict() for failure in loaded.failures],
                "file_digests": loaded.file_digests,
            }
            commit("validation", "complete")

        sample_start = time.perf_counter()
        sampling_complete = bool(
            bundle is not None
            and state.get("samples") is not None
            and state.get("stage_statuses", {}).get("samples") == "complete"
        )
        fresh_samples: list[SampleRecord] | None = None
        fresh_sample_failures: list[Failure] | None = None
        sampling_reset = False
        if sampling_complete and corpus_changed:
            fresh_samples, fresh_sample_failures = sample_documents(loaded.documents, balance)
            stored_sample_values = state["samples"]["samples"]
            fresh_sample_values = [sample.to_dict() for sample in fresh_samples]
            if stored_sample_values != fresh_sample_values:
                sampling_complete = False
                sampling_reset = True
        if sampling_complete:
            samples = [SampleRecord(**self._sample_record_kwargs(item)) for item in state["samples"]["samples"]]
            sample_failures = [Failure.from_dict(item) for item in state["samples"].get("failures", [])]
            store_stats = state["samples"].get("source_counts", {})
        else:
            if fresh_samples is None or fresh_sample_failures is None:
                samples, sample_failures = sample_documents(loaded.documents, balance)
            else:
                samples, sample_failures = fresh_samples, fresh_sample_failures
            store_stats = {
                source: sum(1 for sample in samples if sample.source == source)
                for source in sorted({sample.source for sample in samples})
            }
            state["samples"] = {
                "samples": [sample.to_dict() for sample in samples],
                "failures": [failure.to_dict() for failure in sample_failures],
                "source_counts": store_stats,
                "allocations": allocate_source_counts(
                    balance.source_weights,
                    balance.target_sample_count,
                    {document.source for document in loaded.documents},
                ),
            }
            if bundle is not None:
                state["minhash"] = None
                state["dedup"] = None
                state["result"] = None
                state.setdefault("stage_statuses", {})["minhash"] = "pending"
                state.setdefault("stage_statuses", {})["dedup"] = "pending"
                state.setdefault("stage_statuses", {})["write"] = "pending"
            commit("samples", "complete")
        stage_elapsed_ms.setdefault("sample", round((time.perf_counter() - sample_start) * 1000, 3))

        signature_cache = SignatureCache(self.cache_root)
        decision_cache = GroupDecisionCache(self.cache_root)

        minhash_start = time.perf_counter()
        existing_minhash = state.get("minhash")
        minhash_exists = isinstance(existing_minhash, dict)
        minhash_status = existing_minhash.get("status") if minhash_exists else None
        existing_pending = (existing_minhash or {}).get("pending", [])

        doc_by_id = {document.doc_id: document for document in loaded.documents}
        sampled_docs: dict[str, Document] = {}
        for sample in samples:
            document = doc_by_id.get(sample.doc_id)
            if document is not None:
                sampled_docs[document.stable_key] = document
        sampled_documents = sorted(sampled_docs.values(), key=lambda document: (document.source, document.doc_id))
        input_sample_digest = sha256_json(
            [
                {
                    "source": document.source,
                    "doc_id": document.doc_id,
                    "fingerprint": document.fingerprint,
                }
                for document in sampled_documents
            ]
        )
        resume_minhash_stale = bool(
            corpus_changed
            or sampling_reset
            or (
                minhash_exists
                and existing_minhash.get("input_sample_digest") != input_sample_digest
            )
        )
        minhash_complete = bool(
            minhash_exists
            and minhash_status == "complete"
            and not existing_pending
            and not resume_minhash_stale
        )

        if minhash_complete:
            assert existing_minhash is not None
            signatures = self._records_from_state(existing_minhash)
            scheduler_failures = [Failure.from_dict(item) for item in existing_minhash.get("failures", [])]
            scheduler_result_state = existing_minhash
        else:
            def progress_callback(partial: dict[str, Any]) -> None:
                nonlocal state
                partial["input_sample_digest"] = input_sample_digest
                state["minhash"] = partial
                state["dedup"] = None
                state.setdefault("stage_statuses", {})["dedup"] = "pending"
                state["result"] = None
                state.setdefault("stage_statuses", {})["write"] = "pending"
                stage_elapsed_ms["minhash"] = round((time.perf_counter() - minhash_start) * 1000, 3)
                commit("minhash", str(partial.get("status", "running")))

            budget_scheduler = BudgetScheduler(
                scheduler,
                dedup,
                signature_cache,
                progress_callback=progress_callback,
            )
            result = budget_scheduler.execute(
                sampled_documents,
                resume_state=existing_minhash if isinstance(existing_minhash, dict) else None,
                stop_after_batches=stop_after_batches,
            )
            scheduler_result_state = result.state_dict()
            scheduler_result_state["input_sample_digest"] = input_sample_digest
            state["minhash"] = scheduler_result_state
            state["dedup"] = None
            state["result"] = None
            state.setdefault("stage_statuses", {})["dedup"] = "pending"
            state.setdefault("stage_statuses", {})["write"] = "pending"
            stage_elapsed_ms["minhash"] = round((time.perf_counter() - minhash_start) * 1000, 3)
            commit("minhash", "complete" if not result.pending else result.status)
            signatures = result.signatures
            scheduler_failures = result.failures

        stage_elapsed_ms.setdefault("minhash", round((time.perf_counter() - minhash_start) * 1000, 3))
        minhash_state = state["minhash"] or {}
        pending = minhash_state.get("pending", []) or []
        minhash_stopped_early = bool(minhash_state.get("stopped_early", False))

        dedup_start = time.perf_counter()
        dedup_result = deduplicate(
            signatures.values(),
            balance.source_weights,
            dedup,
            decision_cache,
        )
        dedup_status = "complete" if not pending and not minhash_stopped_early else "partial"
        state["dedup"] = {
            **dedup_result.to_dict(),
            "status": dedup_status,
            "group_decision_cache": decision_cache.stats(),
        }
        stage_elapsed_ms["dedup"] = round((time.perf_counter() - dedup_start) * 1000, 3)
        commit("dedup", dedup_status)

        all_failures = sorted_failures(
            loaded.failures + sample_failures + scheduler_failures
        )
        complete = bool(not pending and not minhash_stopped_early and not any(
            failure.code in {E_SOURCE_QUOTA_EXCEEDED, E_GLOBAL_BUDGET_EXHAUSTED}
            for failure in scheduler_failures
        ))
        if complete:
            run_status = "complete"
        elif minhash_stopped_early:
            run_status = "interrupted"
        else:
            run_status = "budget_exhausted"

        write_start = time.perf_counter()
        core = self._deterministic_core(
            balance=balance,
            dedup=dedup,
            scheduler=scheduler,
            loaded=loaded,
            samples=samples,
            dedup_result=dedup_result.to_dict(),
            minhash_state=minhash_state,
            failures=all_failures,
            run_status=run_status,
        )
        result_digest = sha256_json(core)
        state["result"] = {
            "complete": complete,
            "status": run_status,
            "result_digest": result_digest,
            "retained_documents": dedup_result.retained_documents,
            "removed_documents": dedup_result.removed_documents,
            "pending_queue": pending,
        }
        stage_elapsed_ms["write"] = round((time.perf_counter() - write_start) * 1000, 3)
        final_version = commit("write", "complete" if complete else "partial")

        report = self._base_report(
            status=run_status,
            complete=complete,
            failures=all_failures,
            loaded=loaded,
            balance=balance,
            dedup=dedup,
            scheduler=scheduler,
            injections=injections,
            config_hashes=config_hashes,
            stage_elapsed_ms=stage_elapsed_ms,
            start=start,
        )
        report.update(
            {
                "global_version": final_version,
                "sampling": {
                    "target_sample_count": balance.target_sample_count,
                    "sample_count": len(samples),
                    "unique_document_count": len({(sample.source, sample.doc_id) for sample in samples}),
                    "by_source": {
                        source: int(store_stats.get(source, 0))
                        for source in sorted(balance.source_weights)
                    },
                    "allocations": (state.get("samples") or {}).get("allocations", {}),
                    "samples": [sample.to_dict() for sample in samples],
                },
                "documents": {
                    "loaded_valid": len(loaded.documents),
                    "loaded_invalid": len(loaded.failures),
                    "sampled_unique_before_dedup": dedup_result.input_document_count,
                    "before_dedup": dedup_result.input_document_count,
                    "after_dedup": dedup_result.output_document_count,
                    "retained": dedup_result.output_document_count,
                    "removed_as_duplicate": len(dedup_result.removed_documents),
                    "signature_failed": len(
                        [failure for failure in scheduler_failures if failure.stage == "minhash" and failure.code not in {E_SOURCE_QUOTA_EXCEEDED, E_GLOBAL_BUDGET_EXHAUSTED}]
                    ),
                },
                "budget": self._budget_report(scheduler, minhash_state, pending),
                "dedup": dedup_result.to_dict(),
                "pending_queue": pending,
                "stage_timings_ms": stage_elapsed_ms,
                "cache_audit": {
                    "signatures": signature_cache.stats(),
                    "group_decisions": decision_cache.stats(),
                },
                "snapshot": {
                    "reused": bool(bundle is not None),
                    "directory": safe_relative(self.snapshot_store.root, Path.cwd()),
                    "version": final_version,
                    "stage_versions": state.get("stage_versions", {}),
                    "stage_statuses": state.get("stage_statuses", {}),
                },
                "deterministic_result_digest": result_digest,
            }
        )
        report["injection"] = self._injection_report(injections, loaded, all_failures)
        report["deterministic_core"] = core
        self._write_report(report, final_version)
        return report

    def _state_for_run(
        self,
        bundle: SnapshotBundle | None,
        config_hashes: dict[str, str],
        corpus_digest: str,
        stage_elapsed_ms: dict[str, float],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        if bundle is None:
            return (
                {
                    "global_version": 0,
                    "config_hashes": config_hashes,
                    "corpus_digest": corpus_digest,
                    "stage_versions": {},
                    "stage_statuses": {},
                    "validation": None,
                    "samples": None,
                    "minhash": None,
                    "dedup": None,
                    "result": None,
                },
                stage_elapsed_ms,
            )
        prior_elapsed = {
            str(stage): float(value)
            for stage, value in (bundle.manifest.get("stage_elapsed_ms", {}) or {}).items()
        }
        prior_elapsed.update(stage_elapsed_ms)
        return bundle.state(), prior_elapsed

    @staticmethod
    def _sample_record_kwargs(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": str(value["source"]),
            "doc_id": str(value["doc_id"]),
            "sample_index": int(value["sample_index"]),
            "order": int(value["order"]),
        }

    @staticmethod
    def _records_from_state(minhash_state: dict[str, Any]) -> dict[str, SignatureRecord]:
        records: dict[str, SignatureRecord] = {}
        for key, value in (minhash_state.get("signatures", {}) or {}).items():
            try:
                records[key] = SignatureRecord.from_dict(value)
            except (KeyError, TypeError, ValueError):
                continue
        return records

    def _budget_report(
        self,
        scheduler: SchedulerConfig,
        minhash_state: dict[str, Any],
        pending: list[dict[str, Any]],
    ) -> dict[str, Any]:
        consumed = int(minhash_state.get("consumed_points", 0) or 0)
        return {
            "global_shingle_budget": scheduler.global_shingle_budget,
            "consumed_points": consumed,
            "reclaimed_points": int(minhash_state.get("reclaimed_points", 0) or 0),
            "remaining_points": max(0, scheduler.global_shingle_budget - consumed),
            "source_quotas": scheduler.source_quotas,
            "source_points": dict(sorted((minhash_state.get("source_points", {}) or {}).items())),
            "deduction_sequence": minhash_state.get("deductions", []),
            "pending_count": len(pending),
            "cursor": minhash_state.get("cursor", {}),
            "batch_count": minhash_state.get("batch_count", 0),
            "max_workers": scheduler.max_workers,
            "batch_size": scheduler.batch_size,
            "snapshot_interval_batches": scheduler.snapshot_interval_batches,
        }

    def _base_report(
        self,
        *,
        status: str,
        complete: bool,
        failures: list[Failure],
        loaded: LoadResult | None,
        balance: BalanceConfig | None,
        dedup: DedupConfig | None,
        scheduler: SchedulerConfig | None,
        injections: list[InjectionItem],
        config_hashes: dict[str, str],
        stage_elapsed_ms: dict[str, float],
        start: float,
    ) -> dict[str, Any]:
        report = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": status,
            "complete": complete,
            "global_version": 0,
            "config_hashes": dict(config_hashes),
            "config": {
                "balance": None if balance is None else {
                    "source_weights": balance.source_weights,
                    "target_sample_count": balance.target_sample_count,
                    "sampling_salt": balance.sampling_salt,
                },
                "dedup": None if dedup is None else {
                    "shingle_length": dedup.shingle_length,
                    "signature_dim": dedup.signature_dim,
                    "jaccard_threshold": dedup.jaccard_threshold,
                    "representative_rule": dedup.representative_rule,
                },
                "scheduler": None if scheduler is None else {
                    "max_workers": scheduler.max_workers,
                    "global_shingle_budget": scheduler.global_shingle_budget,
                    "source_quotas": scheduler.source_quotas,
                    "batch_size": scheduler.batch_size,
                    "snapshot_interval_batches": scheduler.snapshot_interval_batches,
                },
            },
            "inputs": None if loaded is None else {
                "data_path": safe_relative(self.data_path, Path.cwd()),
                "corpus_digest": loaded.corpus_digest,
                "file_digests": loaded.file_digests,
            },
            "failures": [failure.to_dict() for failure in sorted_failures(failures)],
            "failure_count": len(failures),
            "stage_timings_ms": dict(stage_elapsed_ms),
            "elapsed_total_ms": round((time.perf_counter() - start) * 1000, 3),
            "injection": self._injection_report(injections, loaded, failures),
        }
        return report

    def _injection_report(
        self,
        injections: list[InjectionItem],
        loaded: LoadResult | None,
        failures: list[Failure],
    ) -> dict[str, Any]:
        failure_by_injection = {
            failure.injection_id: failure for failure in failures if failure.injection_id is not None
        }
        traces = loaded.injection_traces if loaded is not None else {}
        hits: list[dict[str, Any]] = []
        for item in injections:
            failure = failure_by_injection.get(item.injection_id)
            trace = traces.get(item.injection_id)
            matched = failure is not None or trace is not None
            code = failure.code if failure is not None else (trace or {}).get("code")
            hits.append(
                {
                    **item.to_dict(),
                    "matched": matched,
                    "observed_code": code,
                    "observed_phase": failure.stage if failure is not None else ("load_validate" if matched else None),
                    "location": None if trace is None else {
                        "file": trace.get("file"),
                        "line_no": trace.get("line_no"),
                    },
                }
            )
        return {
            "configured_count": len(injections),
            "matched_count": sum(1 for hit in hits if hit["matched"]),
            "items": hits,
        }

    def _deterministic_core(
        self,
        *,
        balance: BalanceConfig,
        dedup: DedupConfig,
        scheduler: SchedulerConfig,
        loaded: LoadResult,
        samples: list[SampleRecord],
        dedup_result: dict[str, Any],
        minhash_state: dict[str, Any],
        failures: list[Failure],
        run_status: str,
    ) -> dict[str, Any]:
        stable_dedup = {
            **dedup_result,
            "groups": [
                {key: value for key, value in group.items() if key != "decision_cache_hit"}
                for group in dedup_result.get("groups", [])
            ],
        }
        return {
            "status": run_status,
            "spread_config": {
                "source_weights": balance.source_weights,
                "target_sample_count": balance.target_sample_count,
                "sampling_salt": balance.sampling_salt,
            },
            "dedup_config_without_scheduler": {
                "shingle_length": dedup.shingle_length,
                "signature_dim": dedup.signature_dim,
                "jaccard_threshold": dedup.jaccard_threshold,
                "representative_rule": dedup.representative_rule,
            },
            "budget_config": {
                "global_shingle_budget": scheduler.global_shingle_budget,
                "source_quotas": scheduler.source_quotas,
                "batch_size": scheduler.batch_size,
            },
            "samples": [sample.to_dict() for sample in samples],
            "failures": [failure.to_dict() for failure in sorted_failures(failures)],
            "budget": {
                "consumed_points": int(minhash_state.get("consumed_points", 0) or 0),
                "source_points": dict(sorted((minhash_state.get("source_points", {}) or {}).items())),
                "deductions": minhash_state.get("deductions", []),
                "pending": minhash_state.get("pending", []),
            },
            "dedup": stable_dedup,
        }

    def _write_plain_failure_report(
        self,
        status: str,
        failures: list[Failure],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        report = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": status,
            "complete": False,
            "global_version": self.snapshot_store.current_version() or 0,
            "config_hashes": {},
            "config": {},
            "inputs": None,
            "failures": [failure.to_dict() for failure in sorted_failures(failures)],
            "failure_count": len(failures),
            "stage_timings_ms": {},
            "injection": {"configured_count": 0, "matched_count": 0, "items": []},
            "deterministic_result_digest": sha256_json(
                {"status": status, "failures": [failure.to_dict() for failure in sorted_failures(failures)]}
            ),
        }
        report.update(extra)
        self._write_report(report, None)
        return report

    def _write_report(self, report: dict[str, Any], version: int | None) -> None:
        report_path = self.report_dir / f"{self.run_id}.json"
        write_json_atomic(report_path, report)
        if version is not None:
            write_json_atomic(self.report_dir / f"{self.run_id}-v{version:06d}.json", report)
        write_json_atomic(self.result_dir / f"{self.run_id}.json", report)
