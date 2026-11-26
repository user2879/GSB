from __future__ import annotations

import copy
from typing import Any

from .cache import (
    ContentCache,
    ReuseLedger,
    conclusion_cache_key,
    hit_cache_key,
    scope_cache_key,
    settlement_cache_key,
)
from .composition import compose_ticket, order_conclusions
from .edits import dry_apply_command
from .hashing import fingerprint
from .pipeline import config_hash, summarize_errors
from .reference import compare_ticket_error, reference_ticket
from .scanner import scan_ticket
from .settlement import assemble_settlement, settle_scope


class IncrementalEngine:
    def __init__(self, bundle: dict[str, Any], *, warm: bool = True):
        self.bundle = copy.deepcopy(bundle)
        self.cache = ContentCache()
        self.ledger = ReuseLedger()
        self.last_edit_report: dict[str, Any] | None = None
        self.gear = f"Q{self.bundle['config']['fixed_point']['decimal_width']}"
        self.scans: dict[str, dict[str, Any]] = {}
        self.settlements: dict[str, dict[str, Any]] = {}
        self.conclusions_by_id: dict[str, dict[str, Any]] = {}
        self.references: dict[str, dict[str, Any]] = {}
        self.errors: list[dict[str, Any]] = []
        if warm:
            self.rebuild_all()

    @property
    def ticket_ids(self) -> list[str]:
        return sorted(ticket["ticket_id"] for ticket in self.bundle["tickets"])

    def _ticket(self, ticket_id: str) -> dict[str, Any]:
        return next(ticket for ticket in self.bundle["tickets"] if ticket["ticket_id"] == ticket_id)

    def _scope_keys(self, ticket_id: str) -> tuple[dict[str, Any], dict[str, str], str, str]:
        ticket = self._ticket(ticket_id)
        hit_key = hit_cache_key(self.bundle, ticket, self.gear)
        scan_key = scope_cache_key(self.bundle, ticket, hit_key, self.gear)
        scan = self.scans.get(ticket_id) or scan_ticket(ticket, self.bundle)
        keys: dict[str, str] = {}
        for scope in scan["scopes"]:
            keys[scope["scope_id"]] = settlement_cache_key(self.bundle, ticket, scope, scan, self.gear)
        conclusion_key = conclusion_cache_key(self.bundle, ticket, list(keys.values()), self.gear)
        return scan, keys, conclusion_key, scan_key

    def rebuild_all(self) -> dict[str, Any]:
        self.ledger = ReuseLedger()
        for ticket_id in self.ticket_ids:
            self._materialize_ticket(ticket_id, affected=True, record_initial=True)
        self._refresh_result()
        return self.result

    def _materialize_ticket(
        self, ticket_id: str, *, affected: bool, ledger: ReuseLedger | None = None,
        record_initial: bool = False,
    ) -> None:
        ledger = ledger or self.ledger
        ticket = self._ticket(ticket_id)
        hit_key = hit_cache_key(self.bundle, ticket, self.gear)
        scan_key = scope_cache_key(self.bundle, ticket, hit_key, self.gear)
        cached_scan = self.cache.get(scan_key)
        if cached_scan is not None:
            scan = cached_scan
            if not record_initial:
                ledger.record(entity_type="hit_scan", entity_id=ticket_id, status="reused",
                              reason="stable hit/scope cache key", cache_key=scan_key,
                              basis={"ticket_id": ticket_id})
        else:
            scan = scan_ticket(ticket, self.bundle)
            self.cache.put(hit_key, scan["hits"])
            self.cache.put(scan_key, scan)
            if not record_initial:
                ledger.record(entity_type="hit_scan", entity_id=ticket_id, status="recomputed",
                              reason="hit/scope cache miss", new_cache_key=scan_key,
                              basis={"ticket_id": ticket_id})

        scope_results: list[dict[str, Any]] = []
        settlement_keys: list[str] = []
        for raw_scope in scan["scopes"]:
            skey = settlement_cache_key(self.bundle, ticket, raw_scope, scan, self.gear)
            settlement_keys.append(skey)
            cached_scope = self.cache.get(skey)
            if cached_scope is not None:
                scope_result = cached_scope
                if not record_initial:
                    ledger.record(entity_type="scope_settlement", entity_id=raw_scope["scope_id"],
                                  status="reused", reason="stable scope fingerprint", cache_key=skey,
                                  basis={"ticket_id": ticket_id, "scope_id": raw_scope["scope_id"]})
            else:
                scope_result = settle_scope(scan, raw_scope, ticket, self.bundle)
                self.cache.put(skey, scope_result)
                if not record_initial:
                    ledger.record(entity_type="scope_settlement", entity_id=raw_scope["scope_id"],
                                  status="recomputed", reason="scope cache miss", new_cache_key=skey,
                                  basis={"ticket_id": ticket_id, "scope_id": raw_scope["scope_id"]})
            scope_results.append(scope_result)

        settlement = assemble_settlement(scan, scope_results, self.bundle)
        ckey = conclusion_cache_key(self.bundle, ticket, settlement_keys, self.gear)
        cached_conclusion = self.cache.get(ckey)
        if cached_conclusion is not None:
            conclusion = cached_conclusion
            if not record_initial:
                ledger.record(entity_type="conclusion", entity_id=ticket_id, status="reused",
                              reason="stable conclusion fingerprint", cache_key=ckey,
                              basis={"ticket_id": ticket_id})
        else:
            conclusion = compose_ticket(ticket, scan, settlement, self.bundle)
            self.cache.put(ckey, conclusion)
            if not record_initial:
                ledger.record(entity_type="conclusion", entity_id=ticket_id, status="recomputed",
                              reason="conclusion cache miss", new_cache_key=ckey,
                              basis={"ticket_id": ticket_id})

        self.scans[ticket_id] = scan
        self.settlements[ticket_id] = settlement
        self.conclusions_by_id[ticket_id] = conclusion
        self.references[ticket_id] = reference_ticket(ticket, scan, self.bundle)

    def _invalidate_key(self, key: str) -> None:
        self.cache.invalidate(key)

    def _old_ticket_keys(self, ticket_id: str, old_bundle: dict[str, Any]) -> dict[str, Any]:
        old_ticket = next(t for t in old_bundle["tickets"] if t["ticket_id"] == ticket_id)
        old_hit_key = hit_cache_key(old_bundle, old_ticket, self.gear)
        old_scan_key = scope_cache_key(old_bundle, old_ticket, old_hit_key, self.gear)
        old_scan = self.scans[ticket_id]
        old_scope_keys = {
            scope["scope_id"]: settlement_cache_key(old_bundle, old_ticket, scope, old_scan, self.gear)
            for scope in old_scan["scopes"]
        }
        old_conclusion_key = conclusion_cache_key(
            old_bundle, old_ticket, list(old_scope_keys.values()), self.gear
        )
        return {
            "hit": old_hit_key,
            "scan": old_scan_key,
            "scopes": old_scope_keys,
            "conclusion": old_conclusion_key,
        }

    def _invalidate_ticket_all(self, ticket_id: str, old_bundle: dict[str, Any], reason: str, command: dict[str, Any]) -> set[str]:
        keys = self._old_ticket_keys(ticket_id, old_bundle)
        self._invalidate_key(keys["hit"])
        self._invalidate_key(keys["scan"])
        self._invalidate_key(keys["conclusion"])
        self.ledger.record(entity_type="hit_set", entity_id=ticket_id, status="invalidated",
                           reason=reason, old_cache_key=keys["hit"],
                           basis={"ticket_id": ticket_id, "command_seq": command.get("seq")})
        for scope_id, key in keys["scopes"].items():
            self._invalidate_key(key)
            self.ledger.record(entity_type="scope_settlement", entity_id=scope_id, status="invalidated",
                               reason=reason, old_cache_key=key,
                               basis={"ticket_id": ticket_id, "scope_id": scope_id, "command_seq": command.get("seq")})
        self.ledger.record(entity_type="conclusion", entity_id=ticket_id, status="invalidated",
                           reason=reason, old_cache_key=keys["conclusion"],
                           basis={"ticket_id": ticket_id, "command_seq": command.get("seq")})
        return {ticket_id}

    def _plan_invalidation(self, command: dict[str, Any], old_bundle: dict[str, Any]) -> set[str]:
        op = command["op"]
        affected: set[str] = set()

        if op == "replace_ticket":
            ticket_id = command["ticket"]["ticket_id"]
            if ticket_id in self.ticket_ids:
                affected |= self._invalidate_ticket_all(
                    ticket_id, old_bundle, "ticket_content_replaced", command
                )
            else:
                affected.add(ticket_id)
                self.ledger.record(
                    entity_type="ticket",
                    entity_id=ticket_id,
                    status="invalidated",
                    reason="ticket_upserted",
                    basis={"ticket_id": ticket_id, "command_seq": command.get("seq")},
                )
            return affected

        if op in {"upsert_neg_trigger", "remove_neg_trigger"}:
            entry = command.get("entry") or command.get("trigger")
            trigger_id = entry["trigger_id"] if entry else command["entry_id"]
            old_entry = old_bundle["trigger_ids"].get(trigger_id)
            new_term = entry.get("term") if entry else None
            old_term = old_entry.get("term") if old_entry else None
            for ticket_id in self.ticket_ids:
                scan = self.scans[ticket_id]
                ticket = self._ticket(ticket_id)
                matching_scopes = [
                    scope for scope in scan["scopes"]
                    if scope["trigger_id"] == trigger_id
                    or (old_term and scope["trigger_term"] == old_term)
                    or (new_term and new_term in ticket["tokens"] and scope["start"] == ticket["tokens"].index(new_term))
                ]
                term_present = (old_term and old_term in ticket["tokens"]) or (new_term and new_term in ticket["tokens"])
                if not matching_scopes and not term_present:
                    continue
                keys = self._old_ticket_keys(ticket_id, old_bundle)
                self._invalidate_key(keys["scan"])
                self._invalidate_key(keys["conclusion"])
                affected.add(ticket_id)
                if matching_scopes:
                    for scope in matching_scopes:
                        skey = keys["scopes"].get(scope["scope_id"])
                        if skey:
                            self._invalidate_key(skey)
                        self.ledger.record(
                            entity_type="scope_settlement", entity_id=scope["scope_id"], status="invalidated",
                            reason="negation_trigger_referenced", old_cache_key=skey,
                            basis={"ticket_id": ticket_id, "scope_id": scope["scope_id"],
                                   "trigger_id": trigger_id, "command_seq": command.get("seq")},
                        )
                else:
                    self.ledger.record(
                        entity_type="scope_index", entity_id=ticket_id, status="invalidated",
                        reason="new_trigger_term_occurrence", old_cache_key=keys["scan"],
                        basis={"ticket_id": ticket_id, "term": new_term, "command_seq": command.get("seq")},
                    )
                self.ledger.record(entity_type="conclusion", entity_id=ticket_id, status="invalidated",
                                   reason="depends_on_negation_scope", old_cache_key=keys["conclusion"],
                                   basis={"ticket_id": ticket_id, "command_seq": command.get("seq")})
            return affected

        if op in {"upsert_conflict", "remove_conflict"}:
            entry = command.get("entry") or command.get("conflict")
            conflict_id = entry["conflict_id"] if entry else command["entry_id"]
            old_entry = old_bundle["conflict_ids"].get(conflict_id)
            new_term = entry.get("term") if entry else None
            old_term = old_entry.get("term") if old_entry else None
            for ticket_id in self.ticket_ids:
                keys = self._old_ticket_keys(ticket_id, old_bundle)
                affected_scopes: list[str] = []
                for scope_result in self.settlements[ticket_id]["scope_results"]:
                    event_ids = {event["conflict_id"] for event in scope_result["conflict_events"]}
                    event_terms = {event["term"] for event in scope_result["conflict_events"]}
                    raw_scope = next(s for s in self.scans[ticket_id]["scopes"] if s["scope_id"] == scope_result["scope_id"])
                    ticket = self._ticket(ticket_id)
                    raw_terms = {ticket["tokens"][pos] for pos in raw_scope["token_positions"]}
                    if conflict_id in event_ids or old_term in event_terms or new_term in raw_terms or old_term in raw_terms:
                        affected_scopes.append(scope_result["scope_id"])
                if not affected_scopes:
                    continue
                self._invalidate_key(keys["conclusion"])
                affected.add(ticket_id)
                for scope_id in affected_scopes:
                    skey = keys["scopes"].get(scope_id)
                    self._invalidate_key(skey)
                    self.ledger.record(entity_type="scope_settlement", entity_id=scope_id, status="invalidated",
                                       reason="polarity_conflict_referenced", old_cache_key=skey,
                                       basis={"ticket_id": ticket_id, "scope_id": scope_id,
                                              "conflict_id": conflict_id, "command_seq": command.get("seq")})
                self.ledger.record(entity_type="conclusion", entity_id=ticket_id, status="invalidated",
                                   reason="depends_on_conflict_settlement", old_cache_key=keys["conclusion"],
                                   basis={"ticket_id": ticket_id, "command_seq": command.get("seq")})
            return affected

        if op == "set_channel_weight":
            channel_id = command["channel_id"]
            for ticket_id in self.ticket_ids:
                if self._ticket(ticket_id)["channel_id"] != channel_id:
                    continue
                keys = self._old_ticket_keys(ticket_id, old_bundle)
                self._invalidate_key(keys["conclusion"])
                self.ledger.record(entity_type="conclusion", entity_id=ticket_id, status="invalidated",
                                   reason="channel_weight_changed", old_cache_key=keys["conclusion"],
                                   basis={"ticket_id": ticket_id, "channel_id": channel_id,
                                          "command_seq": command.get("seq")})
                affected.add(ticket_id)
            return affected

        if op == "update_config":
            patch = command["patch"]
            if "fixed_point" in patch and "decimal_width" in patch.get("fixed_point", {}):
                for ticket_id in self.ticket_ids:
                    affected |= self._invalidate_ticket_all(ticket_id, old_bundle, "quantization_width_changed", command)
                return affected
            conclusion_only = {"positive_threshold_points", "negative_threshold_points",
                               "risk_thresholds_points", "boundary_half_step_decimal_width"}
            cancellation_fields = {"cancellation_strength_points", "score_lower_bound_points"}
            penalty_fields = {"length_penalty_bp_per_token", "length_min_retention_bp"}
            for ticket_id in self.ticket_ids:
                keys = self._old_ticket_keys(ticket_id, old_bundle)
                invalidate_conclusion = bool(set(patch) & conclusion_only)
                invalidate_all_scopes = "scope_max_tokens" in patch
                if invalidate_all_scopes:
                    affected |= self._invalidate_ticket_all(ticket_id, old_bundle, "scope_parameter_changed", command)
                    continue
                for raw_scope in self.scans[ticket_id]["scopes"]:
                    reason = None
                    if set(patch) & cancellation_fields and any(
                        event for event in self.settlements[ticket_id]["scope_results"]
                        if event["scope_id"] == raw_scope["scope_id"] and event["conflict_events"]
                    ):
                        reason = "cancellation_parameter_changed"
                    elif set(patch) & penalty_fields and raw_scope["length"] > 1:
                        reason = "length_penalty_parameter_changed"
                    if reason:
                        skey = keys["scopes"][raw_scope["scope_id"]]
                        self._invalidate_key(skey)
                        invalidate_conclusion = True
                        self.ledger.record(entity_type="scope_settlement", entity_id=raw_scope["scope_id"],
                                           status="invalidated", reason=reason, old_cache_key=skey,
                                           basis={"ticket_id": ticket_id, "scope_id": raw_scope["scope_id"],
                                                  "command_seq": command.get("seq")})
                if invalidate_conclusion:
                    self._invalidate_key(keys["conclusion"])
                    affected.add(ticket_id)
                    self.ledger.record(entity_type="conclusion", entity_id=ticket_id, status="invalidated",
                                       reason="composition_parameter_changed", old_cache_key=keys["conclusion"],
                                       basis={"ticket_id": ticket_id, "command_seq": command.get("seq")})
            return affected

        return affected

    def apply(self, command: dict[str, Any]) -> dict[str, Any]:
        old_bundle = copy.deepcopy(self.bundle)
        candidate = dry_apply_command(old_bundle, command)
        ledger = ReuseLedger()
        previous_ledger = self.ledger
        self.ledger = ledger
        affected = self._plan_invalidation(command, self.bundle)
        self.bundle = candidate
        self.gear = f"Q{self.bundle['config']['fixed_point']['decimal_width']}"

        for ticket_id in sorted(affected):
            self._materialize_ticket(ticket_id, affected=True, ledger=ledger)

        reused_tickets = [ticket_id for ticket_id in self.ticket_ids if ticket_id not in affected]
        for ticket_id in reused_tickets:
            ckey = conclusion_cache_key(
                self.bundle,
                self._ticket(ticket_id),
                [
                    settlement_cache_key(
                        self.bundle, self._ticket(ticket_id), scope, self.scans[ticket_id], self.gear
                    )
                    for scope in self.scans[ticket_id]["scopes"]
                ],
                self.gear,
            )
            # Unchanged parameters and dependencies produce the same key. Record a direct reuse hit.
            ledger.record(entity_type="conclusion", entity_id=ticket_id, status="reused",
                          reason="ticket_not_in_invalidation_set", cache_key=ckey,
                          basis={"ticket_id": ticket_id, "reused_previous_result": True})

        self._refresh_result()
        report = {
            "command_seq": command.get("seq"),
            "op": command["op"],
            "affected_ticket_ids": sorted(affected),
            "reused_ticket_ids": reused_tickets,
            "affected_ticket_count": len(affected),
            "reused_ticket_count": len(reused_tickets),
            "ledger": ledger.sorted_entries(),
            "ledger_counts": ledger.counts(),
        }
        self.last_edit_report = report
        self.ledger = ledger
        return report

    def _refresh_result(self) -> None:
        conclusions = [self.conclusions_by_id[ticket_id] for ticket_id in self.ticket_ids]
        ordered = order_conclusions(conclusions)
        tolerances = self.bundle["config"]["error_tolerances_points"]
        self.errors = [
            compare_ticket_error(
                ticket_id,
                self.references[ticket_id],
                self.settlements[ticket_id],
                self.conclusions_by_id[ticket_id],
                tolerances,
            )
            for ticket_id in self.ticket_ids
        ]
        self.result = {
            "global_version": self.bundle["config"]["global_version"],
            "config_hash": config_hash(self.bundle),
            "decimal_width": int(self.bundle["config"]["fixed_point"]["decimal_width"]),
            "gear": self.gear,
            "scans": self.scans,
            "settlements": self.settlements,
            "conclusions": conclusions,
            "conclusions_by_id": self.conclusions_by_id,
            "ordered_conclusions": ordered,
            "references": self.references,
            "errors": self.errors,
            "error_summary": summarize_errors(self.errors),
            "complete": True,
            "scheduler_report": None,
        }
