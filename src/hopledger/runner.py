from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import HopLedgerError
from .invalidation import plan_invalidation, verify_digests_unchanged
from .ledger import Ledger
from .metrics import attribute_errors, build_conclusion
from .proc import ProcessManager
from .scheduler import DispatchOutcome, DispatchScheduler
from .specs import SpecBundle, load_bundle


class Runner:
    def __init__(
        self,
        bundle: SpecBundle,
        ledger: Ledger,
        *,
        manager: ProcessManager | None = None,
    ) -> None:
        self.bundle = bundle
        self.ledger = ledger
        self.manager = manager
        self.receipt_node_ids = [hop.hop_id for hop in bundle.chain.hops] + [
            bundle.origin.origin_id
        ]
        self.max_observed_global = 0
        self.max_observed_hop = {node_id: 0 for node_id in self.receipt_node_ids}

    @classmethod
    def from_config_dir(
        cls,
        config_dir: str | Path,
        ledger_path: str | Path,
        *,
        log_dir: str | Path | None = None,
        start_processes: bool = True,
    ) -> "Runner":
        bundle = load_bundle(config_dir)
        ledger = Ledger(ledger_path)
        manager = None
        if start_processes:
            manager = ProcessManager(bundle, log_dir or Path(ledger_path).parent / "logs")
            manager.start_all()
        return cls(bundle, ledger, manager=manager)

    def close(self) -> None:
        if self.manager is not None:
            self.manager.stop_all()
        self.ledger.close()

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def reload_bundle(self) -> SpecBundle:
        self.bundle = load_bundle(self.bundle.config_dir)
        self.receipt_node_ids = [hop.hop_id for hop in self.bundle.chain.hops] + [
            self.bundle.origin.origin_id
        ]
        if self.manager is not None:
            self.manager.bundle = self.bundle
        return self.bundle

    def start_manager(self, log_dir: str | Path) -> ProcessManager:
        self.manager = ProcessManager(self.bundle, log_dir)
        self.manager.start_all()
        return self.manager

    def _message_for(self, request_id: str, node_id: str, seq: int) -> dict[str, Any]:
        sample = self.bundle.origin.sample_for(request_id)
        kind = "origin_call" if node_id == self.bundle.origin.origin_id else "hop_call"
        return {
            "protocol": "hopledger-jsonl/1",
            "type": kind,
            "seq": seq,
            "request_id": request_id,
            "node_id": node_id,
            "path": sample.path,
            "seed_offset": sample.seed_offset,
        }

    def invoke_node(
        self,
        request_id: str,
        node_id: str,
        prior_receipts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.manager is None:
            raise HopLedgerError("process manager is not running", code="E_NO_MANAGER")
        message = self._message_for(request_id, node_id, len(prior_receipts) + 1)
        return self.manager.call_node(node_id, message)

    def _failure_maps(self) -> dict[str, int]:
        return {hop.hop_id: hop.failure_ratio_bp for hop in self.bundle.chain.hops}

    def _build_success_records(
        self,
        outcome: DispatchOutcome,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        receipts = outcome.receipts
        expected_nodes = self.receipt_node_ids
        if len(receipts) != len(expected_nodes):
            raise HopLedgerError(
                "incomplete receipt chain",
                code="E_FIELD",
                request_id=outcome.request_id,
                field="receipts",
                expected_nodes=expected_nodes,
                actual_nodes=[receipt.get("node_id") for receipt in receipts],
            )
        conclusion = build_conclusion(
            request_id=outcome.request_id,
            node_ids=self.bundle.chain.node_ids,
            receipts=receipts,
            session_scale=self.bundle.diagnostic.session_scale,
            scale_caliber_version=self.bundle.diagnostic.scale_caliber_version,
            hop_set_version=self.bundle.chain.hop_set_version,
        )
        attribution = attribute_errors(
            conclusion,
            self.bundle.diagnostic.error_model,
            failure_ratio_bp_by_node=self._failure_maps(),
            expected_config_versions=self.bundle.chain.config_versions()
            | {self.bundle.origin.origin_id: self.bundle.origin.config_version},
            expected_scale_factors=self.bundle.chain.scale_factors()
            | {self.bundle.origin.origin_id: self.bundle.diagnostic.session_scale},
            expected_scale_caliber_version=self.bundle.diagnostic.scale_caliber_version,
        )
        return receipts, conclusion, attribution

    def _failure_records(
        self, outcome: DispatchOutcome
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        error = outcome.error or {"code": "E_FORWARD"}
        if error.get("code") != "E_FORWARD_INJECTED":
            raise HopLedgerError(
                error.get("message", "dispatch failed"),
                code=error.get("code", "E_DISPATCH"),
                request_id=outcome.request_id,
                **{k: v for k, v in error.items() if k not in {"code", "message"}},
            )
        conclusion = {
            "schema": "hopledger.request_failure/1",
            "request_id": outcome.request_id,
            "ok": False,
            "error": error,
            "chain_order": self.bundle.chain.node_ids,
            "receipts": outcome.receipts,
            "e2e_total_units": None,
        }
        attribution = {
            "schema": "hopledger.attribution/1",
            "request_id": outcome.request_id,
            "dominant_hop": None,
            "declared_dominant_hop": None,
            "is_dominant": False,
            "primary_cause": "forward_failure_injected",
            "cause_codes": ["forward_failure_injected"],
            "version_conflicts": [],
            "nodes": [],
            "ranking": [],
        }
        return outcome.receipts, conclusion, attribution

    def run_batch(
        self,
        batch_id: str,
        request_ids: list[str],
        *,
        seq: int,
    ) -> dict[str, Any]:
        self.ledger.plan_batch(batch_id, seq, request_ids)
        scheduler = DispatchScheduler(
            self.bundle.diagnostic.scheduler,
            self.receipt_node_ids,
            self.invoke_node,
        )
        try:
            outcomes = scheduler.run_requests(request_ids)
            stats = scheduler.stats()
        finally:
            scheduler.shutdown()

        self.max_observed_global = max(
            self.max_observed_global, stats["max_observed_global"]
        )
        for node_id, value in stats["max_observed_hop"].items():
            self.max_observed_hop[node_id] = max(
                self.max_observed_hop.get(node_id, 0), value
            )

        outcomes_by_id = {outcome.request_id: outcome for outcome in outcomes}
        ordered_outcomes = [outcomes_by_id[request_id] for request_id in sorted(request_ids)]

        receipts_by_request: dict[str, list[dict[str, Any]]] = {}
        conclusions: dict[str, dict[str, Any]] = {}
        attributions: dict[str, dict[str, Any]] = {}
        events_by_request: dict[str, list[dict[str, Any]]] = {}
        summaries: list[dict[str, Any]] = []

        # Build every record before the transaction. An infrastructure error
        # therefore leaves the batch planned/unknown and commits no partial
        # metrics, matching the forced-kill recovery contract.
        for outcome in ordered_outcomes:
            events_by_request[outcome.request_id] = outcome.events
            if outcome.ok:
                receipts, conclusion, attribution = self._build_success_records(outcome)
            else:
                receipts, conclusion, attribution = self._failure_records(outcome)
            receipts_by_request[outcome.request_id] = receipts
            conclusions[outcome.request_id] = conclusion
            attributions[outcome.request_id] = attribution
            summaries.append({
                "request_id": outcome.request_id,
                "ok": outcome.ok,
                "e2e_total_units": conclusion.get("e2e_total_units"),
                "dominant_hop": attribution.get("dominant_hop"),
                "primary_cause": attribution.get("primary_cause"),
            })

        digest = self.ledger.commit_batch(
            batch_id=batch_id,
            seq=seq,
            request_ids=sorted(request_ids),
            receipts_by_request=receipts_by_request,
            conclusions=conclusions,
            attributions=attributions,
            events_by_request=events_by_request,
            hop_set_version=self.bundle.chain.hop_set_version,
            scale_caliber_version=self.bundle.diagnostic.scale_caliber_version,
            session_scale=self.bundle.diagnostic.session_scale,
        )
        return {
            "batch_id": batch_id,
            "seq": seq,
            "request_ids": sorted(request_ids),
            "batch_digest": digest,
            "scheduler": stats,
            "summaries": summaries,
        }

    def run_all(self, request_ids: list[str] | None = None) -> dict[str, Any]:
        requested = list(request_ids or self.bundle.request_ids)
        committed = set(self.ledger.committed_request_ids())
        pending = [request_id for request_id in requested if request_id not in committed]
        batch_size = self.bundle.diagnostic.batch_size
        reports: list[dict[str, Any]] = []
        for start in range(0, len(pending), batch_size):
            seq = start // batch_size + 1
            chunk = pending[start:start + batch_size]
            reports.append(self.run_batch(f"batch-{seq:04d}", chunk, seq=seq))
        return {
            "ok": True,
            "request_ids": requested,
            "pending": pending,
            "batches": reports,
            "max_observed_global": self.max_observed_global,
            "max_observed_hop": self.max_observed_hop,
        }

    def rerun_unknown_batches(self) -> list[dict[str, Any]]:
        reports = []
        for batch in self.ledger.unknown_batches():
            reports.append(self.run_batch(
                batch["batch_id"],
                batch["request_ids"],
                seq=batch["seq"],
            ))
        return reports

    def invalidation_plan(self) -> dict[str, Any]:
        request_ids = self.ledger.committed_request_ids()
        caliber_changed = (
            int(self.ledger.get_meta("scale_caliber_version", "0"))
            != self.bundle.diagnostic.scale_caliber_version
        )
        stored_session_scale = self.ledger.get_meta("session_scale")
        session_scale_changed = (
            stored_session_scale is not None
            and int(stored_session_scale) != self.bundle.diagnostic.session_scale
        )
        return plan_invalidation(
            request_ids=request_ids,
            chain_node_ids=self.bundle.chain.node_ids,
            stored_receipt_versions=self.ledger.receipt_version_index(),
            current_config_versions=self.bundle.chain.config_versions()
            | {self.bundle.origin.origin_id: self.bundle.origin.config_version},
            current_scale_factors=self.bundle.chain.scale_factors()
            | {self.bundle.origin.origin_id: self.bundle.diagnostic.session_scale},
            scale_caliber_changed=caliber_changed or session_scale_changed,
        )

    def _replace_receipt(
        self,
        request_id: str,
        node_id: str,
    ) -> dict[str, Any]:
        if self.manager is None:
            raise HopLedgerError("process manager is not running", code="E_NO_MANAGER")
        message = self._message_for(request_id, node_id, 1)
        response = self.manager.call_node(node_id, message)
        receipt = response["receipt"]
        if not response.get("ok", False):
            raise HopLedgerError(
                response.get("error", {}).get("message", "targeted redispatch failed"),
                code=response.get("error", {}).get("code", "E_FORWARD"),
                request_id=request_id,
                hop_id=node_id,
            )
        return receipt

    def apply_invalidation(self, plan: dict[str, Any]) -> dict[str, Any]:
        before_digests = self.ledger.digest_index()
        redispatch_by_request: dict[str, set[str]] = {}
        for node in plan["redispatch_nodes"]:
            kind, request_id, node_id = node.split(":", 2)
            redispatch_by_request.setdefault(request_id, set()).add(node_id)

        affected = list(plan["affected_request_ids"])
        receipts_to_write: dict[str, list[dict[str, Any]]] = {}
        conclusions: dict[str, dict[str, Any]] = {}
        attributions: dict[str, dict[str, Any]] = {}
        events: dict[str, list[dict[str, Any]]] = {}

        for request_id in affected:
            stored_receipts = self.ledger.get_receipts(request_id)
            by_node = {receipt["node_id"]: receipt for receipt in stored_receipts}
            new_receipts: list[dict[str, Any]] = []
            for node_id in sorted(redispatch_by_request.get(request_id, set())):
                receipt = self._replace_receipt(request_id, node_id)
                by_node[node_id] = receipt
                new_receipts.append(receipt)
            if new_receipts:
                receipts_to_write[request_id] = new_receipts

            ordered_receipts = [
                by_node[node_id]
                for node_id in self.receipt_node_ids
                if node_id in by_node
            ]
            conclusion = build_conclusion(
                request_id=request_id,
                node_ids=self.bundle.chain.node_ids,
                receipts=ordered_receipts,
                session_scale=self.bundle.diagnostic.session_scale,
                scale_caliber_version=self.bundle.diagnostic.scale_caliber_version,
                hop_set_version=self.bundle.chain.hop_set_version,
            )
            attribution = attribute_errors(
                conclusion,
                self.bundle.diagnostic.error_model,
                failure_ratio_bp_by_node=self._failure_maps(),
                expected_config_versions=self.bundle.chain.config_versions()
                | {self.bundle.origin.origin_id: self.bundle.origin.config_version},
                expected_scale_factors=self.bundle.chain.scale_factors()
                | {self.bundle.origin.origin_id: self.bundle.diagnostic.session_scale},
                expected_scale_caliber_version=self.bundle.diagnostic.scale_caliber_version,
            )
            conclusions[request_id] = conclusion
            attributions[request_id] = attribution
            events[request_id] = [{
                "type": "targeted_invalidation_recompute",
                "request_id": request_id,
                "redispatched_nodes": sorted(redispatch_by_request.get(request_id, set())),
            }]

        plan_digest = str(plan.get("schema", "plan"))
        batch_id = "evolution-" + __import__("hashlib").sha256(
            __import__("hopledger.canonical", fromlist=["canonical_json"]).canonical_json(plan)
        ).hexdigest()[:12]
        seq = int(self.ledger.cursor().get("last_batch_seq", 0)) + 1
        self.ledger.commit_batch(
            batch_id=batch_id,
            seq=seq,
            request_ids=affected,
            receipts_by_request=receipts_to_write,
            conclusions=conclusions,
            attributions=attributions,
            events_by_request=events,
            hop_set_version=self.bundle.chain.hop_set_version,
            scale_caliber_version=self.bundle.diagnostic.scale_caliber_version,
            session_scale=self.bundle.diagnostic.session_scale,
            last_request_id=affected[-1] if affected else None,
        )
        after_digests = self.ledger.digest_index()
        allowed = set(plan["redispatch_nodes"])
        for request_id in affected:
            allowed.add(f"conclusion:{request_id}")
            allowed.add(f"attribution:{request_id}")
        violations = verify_digests_unchanged(
            before_digests,
            after_digests,
            allowed_changed=sorted(allowed),
        )
        return {
            "batch_id": batch_id,
            "affected_request_ids": affected,
            "redispatched": sorted(plan["redispatch_nodes"]),
            "digest_violations": violations,
            "unchanged_byte_check_passed": not violations,
        }
