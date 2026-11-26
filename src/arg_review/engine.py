from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .adjudication import adjudicate_event, decision_key_parts
from .arguments import argument_key_parts, extract_event_arguments
from .canonical import canonical_dumps, fingerprint
from .injector import (
    apply_argument_stage_injection,
    apply_pre_validation_injection,
    apply_reference_stage_injection,
)
from .ledger import FingerprintLedger
from .references import build_reference_merges
from .scheduler import BudgetScheduler
from .snapshots import SnapshotStore
from .schema_context import build_schema_context
from .state import WorkspaceState
from .triggers import match_event_triggers, trigger_key_parts
from .validation import validate_workspace


class ReviewEngine:
    def __init__(
        self,
        state: WorkspaceState,
        state_dir: str | Path = "artifacts",
        *,
        injection: dict[str, Any] | None = None,
        persist_snapshot: bool = True,
    ) -> None:
        self.state = state.clone()
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = FingerprintLedger(self.state_dir / "ledger.json")
        self.snapshots = SnapshotStore(self.state_dir / "snapshots")
        self.injection = injection
        self.persist_snapshot = persist_snapshot
        self.manifest: dict[str, dict[str, Any]] = {}
        self.stage_faults: list[dict[str, Any]] = []
        self.business_counts: Counter[str] = Counter()
        self.injection_hits: list[dict[str, Any]] = []

    def count(self, code: str | None, amount: int = 1) -> None:
        if code:
            self.business_counts[code] += amount

    def add_fault(
        self,
        *,
        stage: str,
        business_code: str,
        stage_code: str,
        message: str,
        event_id: str | None = None,
        injection_id: str | None = None,
        fault_type: str | None = None,
    ) -> dict[str, Any]:
        fault = {
            "stage": stage,
            "event_id": event_id,
            "business_code": business_code,
            "stage_code": stage_code,
            "message": message,
            "injection_id": injection_id,
            "fault_type": fault_type,
        }
        self.stage_faults.append(fault)
        self.count(business_code)
        if injection_id:
            self.injection_hits.append(
                {
                    "injection_id": injection_id,
                    "hit": True,
                    "stage": stage,
                    "event_id": event_id,
                    "business_code": business_code,
                    "stage_code": stage_code,
                }
            )
        return fault

    def note_transition(
        self, artifact_id: str, cache_key: str, deps_fp: dict[str, str]
    ) -> None:
        old = self.ledger.old_artifacts.get(artifact_id)
        if old and old.get("cache_key") != cache_key:
            self.ledger.invalidated.append(
                {
                    "artifact_id": artifact_id,
                    "old_cache_key": old.get("cache_key"),
                    "new_cache_key": cache_key,
                    "status": "cache_key_changed",
                    "reasons": self.ledger.change_reasons(artifact_id, deps_fp),
                }
            )

    def reuse_or_none(
        self,
        artifact_id: str,
        cache_key: str,
        dependencies: list[str] | None = None,
        dependency_fingerprints: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        hit = self.ledger.inspect(artifact_id, cache_key)
        if hit is None:
            return None
        self.ledger.record_reuse(hit)
        if dependencies:
            dependency_list = sorted(dependencies)
        elif dependency_fingerprints:
            dependency_list = sorted(dependency_fingerprints)
        else:
            dependency_list = hit["basis"]
        self.manifest[artifact_id] = {
            "cache_key": cache_key,
            "dependencies": dependency_list,
            "dependency_fingerprints": dependency_fingerprints or {},
        }
        return self.ledger.get_result(cache_key)

    @staticmethod
    def normalize_trigger_revision(
        result: dict[str, Any], revision: dict[str, Any], scope: dict[str, Any]
    ) -> dict[str, Any]:
        result["schema_revision_id"] = revision["revision_id"]
        result["schema_version"] = revision["version"]
        result["rule_slice_revision_id"] = scope["trigger_rule_revision_id"]
        basis = [
            item
            for item in result.get("basis", [])
            if not str(item).startswith("schema_revision=")
        ]
        basis.append(
            f"schema_revision={revision['revision_id']}@v{revision['version']}"
        )
        result["basis"] = basis
        return result

    def run(self) -> dict[str, Any]:
        recovery = self.snapshots.recover() if self.persist_snapshot else {
            "decision": "snapshot_persistence_disabled"
        }
        injection = self.injection
        if injection and injection.get("stage") in {"validation", "schema"}:
            apply_pre_validation_injection(self.state, injection)

        parse_errors = getattr(self.state, "parse_errors", [])
        validation = validate_workspace(
            self.state.events,
            self.state.vocab_terms,
            self.state.revisions_doc,
            self.state.active_revision_id,
            parse_errors=parse_errors,
        )
        for invalid_entry in validation["invalid_entries"]:
            self.count("E_SCHEMA_INVALID")
            if (
                injection
                and injection.get("stage") == "validation"
                and invalid_entry.get("event_id") == injection.get("event_id")
            ):
                self.add_fault(
                    stage="validation",
                    event_id=injection.get("event_id"),
                    business_code="E_SCHEMA_INVALID",
                    stage_code="STAGE_VALIDATION_REJECTED_ENTRY",
                    message=";".join(invalid_entry.get("errors", [])),
                    injection_id=injection["injection_id"],
                    fault_type=injection.get("fault_type"),
                )

        chain = validation["revision_chain"]
        if not chain["valid"]:
            injected_chain = injection and injection.get("fault_type") == "broken_revision_chain"
            self.add_fault(
                stage="schema_revision",
                business_code="E_SCHEMA_INVALID",
                stage_code="STAGE_REVISION_CHAIN_INVALID",
                message=";".join(chain["errors"]),
                injection_id=injection["injection_id"] if injected_chain else None,
                fault_type="broken_revision_chain" if injected_chain else None,
            )

        revision = chain["revision_map"].get(self.state.active_revision_id)
        valid_events = validation["valid_events"]
        triggers: dict[str, dict[str, Any]] = {}
        trigger_failures: set[str] = set()
        scheduler = BudgetScheduler(self.state.config)
        schema_context = (
            build_schema_context(chain["revision_map"], self.state.active_revision_id)
            if revision is not None
            else {}
        )

        if revision is None:
            self.add_fault(
                stage="trigger",
                business_code="E_SCHEMA_INVALID",
                stage_code="STAGE_ACTIVE_REVISION_MISSING",
                message=self.state.active_revision_id,
            )
        else:
            trigger_plans: dict[str, dict[str, Any]] = {}
            for event in valid_events:
                event_id = event["event_id"]
                schema_type = revision["event_types"].get(event["event_type"])
                if schema_type is None:
                    trigger_failures.add(event_id)
                    self.add_fault(
                        stage="trigger",
                        event_id=event_id,
                        business_code="E_SCHEMA_INVALID",
                        stage_code="STAGE_EVENT_TYPE_MISSING_IN_REVISION",
                        message=event["event_type"],
                    )
                    continue
                scope = schema_context[event["event_type"]]
                parts = trigger_key_parts(event, schema_type, revision, scope)
                deps_fp = {
                    f"event_content:{event_id}": parts["content_fingerprint"],
                    f"trigger_rule_revision:{event['event_type']}": scope[
                        "trigger_rule_fingerprint"
                    ],
                    f"trigger_rules:{event['event_type']}": parts["rule_pack_fingerprint"],
                }
                self.note_transition(event_id and f"trigger:{event_id}", parts["cache_key"], deps_fp)
                cached = self.reuse_or_none(
                    f"trigger:{event_id}",
                    parts["cache_key"],
                    dependencies=sorted(deps_fp),
                    dependency_fingerprints=deps_fp,
                )
                trigger_plans[event_id] = {
                    "schema_type": schema_type,
                    "parts": parts,
                    "deps_fp": deps_fp,
                    "cached": cached,
                    "scope": scope,
                }
                if cached is not None:
                    triggers[event_id] = self.normalize_trigger_revision(
                        cached, revision, scope
                    )

            scheduler.enqueue_trigger_events(sorted(trigger_plans))
            computed_triggers: dict[str, dict[str, Any]] = {}

            def run_trigger_job(event_id: str) -> dict[str, Any]:
                plan = trigger_plans[event_id]
                if plan["cached"] is not None:
                    return plan["cached"]
                result = match_event_triggers(
                    next(e for e in valid_events if e["event_id"] == event_id),
                    plan["schema_type"],
                    revision,
                    plan["scope"],
                )
                computed_triggers[event_id] = result
                return result

            trigger_batches: list[dict[str, Any]] = []
            while scheduler.trigger_queue:
                batch = scheduler.run_trigger_batch(run_trigger_job)
                trigger_batches.append(batch)
                for event_id, message in batch.get("failures", {}).items():
                    trigger_failures.add(event_id)
                    self.add_fault(
                        stage="trigger",
                        event_id=event_id,
                        business_code="E_TRIGGER_STAGE_FAILED",
                        stage_code="STAGE_TRIGGER_EXCEPTION",
                        message=message,
                    )

            for event_id, plan in trigger_plans.items():
                if plan["cached"] is not None or event_id in trigger_failures:
                    continue
                result = computed_triggers[event_id]
                self.normalize_trigger_revision(result, revision, plan["scope"])
                artifact_id = f"trigger:{event_id}"
                self.ledger.store_entry(
                    artifact_id,
                    "trigger",
                    plan["parts"]["cache_key"],
                    result,
                    dependencies=list(plan["deps_fp"]),
                    dependency_fingerprints=plan["deps_fp"],
                )
                self.manifest[artifact_id] = {
                    "cache_key": plan["parts"]["cache_key"],
                    "dependencies": list(plan["deps_fp"]),
                    "dependency_fingerprints": plan["deps_fp"],
                }
                triggers[event_id] = result

            event_candidates: dict[str, list[dict[str, Any]]] = {
                event["event_id"]: [] for event in valid_events
            }
            argument_results: dict[str, dict[str, Any]] = {}
            failed_argument_events: set[str] = set()

            units: list[dict[str, Any]] = []
            unit_context: dict[str, dict[str, Any]] = {}
            for event in valid_events:
                event_id = event["event_id"]
                if event_id in trigger_failures or event_id not in triggers:
                    continue
                event_injection = (
                    injection
                    if injection
                    and injection.get("stage") in {"argument", "adjudication"}
                    and injection.get("event_id") == event_id
                    else None
                )
                staged_event = apply_argument_stage_injection(event, event_injection)
                schema_type = revision["event_types"][event["event_type"]]
                event_scope = schema_context[event["event_type"]]
                trigger_result = triggers[event_id]
                roles: dict[str, list[dict[str, Any]]] = {}
                cached_by_arg: dict[str, dict[str, Any]] = {}
                for argument in staged_event.get("arguments", []):
                    arg_id = argument["arg_id"]
                    role = str(argument.get("role"))
                    roles.setdefault(role, []).append(argument)
                    parts = argument_key_parts(
                        staged_event,
                        argument,
                        trigger_result,
                        revision,
                        self.state.config,
                        event_scope,
                    )
                    deps_fp = {
                        f"argument_content:{event_id}/{arg_id}": parts[
                            "arg_content_fingerprint"
                        ],
                        f"trigger_result:{event_id}": trigger_result["cache_key"],
                        f"schema_slice:{event_id}": event_scope[
                            "schema_slice_fingerprint"
                        ],
                        f"role_weight:{role}": parts["rule_pack_fingerprint"],
                        "quantization": parts["quantization_fingerprint"],
                    }
                    artifact_id = f"arg:{event_id}:{arg_id}"
                    self.note_transition(artifact_id, parts["cache_key"], deps_fp)
                    cached = self.reuse_or_none(
                        artifact_id,
                        parts["cache_key"],
                        dependencies=sorted(deps_fp),
                        dependency_fingerprints=deps_fp,
                    )
                    if cached is not None:
                        cached_by_arg[arg_id] = cached
                        event_candidates[event_id].append(cached)

                for role in sorted(roles):
                    role_args = sorted(
                        roles[role],
                        key=lambda arg: (
                            arg.get("start", 0),
                            arg.get("end", 0),
                            arg.get("arg_id", ""),
                        ),
                    )
                    unit = {
                        "event_id": event_id,
                        "role": role,
                        "candidate_count": len(role_args),
                        "point_cost": len(role_args),
                    }
                    unit_key = BudgetScheduler.unit_key(unit)
                    unit["unit_key"] = unit_key
                    units.append(unit)
                    unit_context[unit_key] = {
                        "event": staged_event,
                        "schema_type": schema_type,
                        "trigger_result": trigger_result,
                        "scope": event_scope,
                        "role_args": role_args,
                        "cached_by_arg": cached_by_arg,
                        "event_injection": event_injection,
                    }

            scheduler.enqueue_argument_units(units)
            unit_outputs: dict[str, dict[str, Any]] = {}

            def score_unit(unit: dict[str, Any]) -> dict[str, Any]:
                context = unit_context[unit["unit_key"]]
                outputs: list[dict[str, Any]] = []
                faults: list[dict[str, Any]] = []
                for argument in context["role_args"]:
                    arg_id = argument["arg_id"]
                    cached = context["cached_by_arg"].get(arg_id)
                    if cached is not None:
                        outputs.append(cached)
                        continue
                    try:
                        one_arg_event = {
                            **context["event"],
                            "arguments": [argument],
                        }
                        result = extract_event_arguments(
                            one_arg_event,
                            context["trigger_result"],
                            context["schema_type"],
                            revision,
                            self.state.config,
                            context["scope"],
                        )
                        outputs.append(result["candidates"][0])
                    except Exception as exc:
                        faults.append(
                            {
                                "arg_id": arg_id,
                                "error": exc,
                                "business_code": getattr(
                                    exc, "business_code", "E_ARGUMENT_STAGE_FAILED"
                                ),
                                "stage_code": getattr(
                                    exc, "stage_code", "STAGE_ARGUMENT_EXCEPTION"
                                ),
                            }
                        )
                output = {"candidates": outputs, "faults": faults}
                unit_outputs[unit["unit_key"]] = output
                return output

            argument_batch = scheduler.run_argument_batch(score_unit)
            for unit_key in argument_batch["admitted_unit_keys"]:
                unit = next(unit for unit in units if unit["unit_key"] == unit_key)
                context = unit_context[unit_key]
                output = unit_outputs[unit_key]
                event_id = unit["event_id"]
                for candidate in output["candidates"]:
                    arg_id = candidate["arg_id"]
                    artifact_id = candidate["artifact_id"]
                    if artifact_id in self.manifest:
                        continue
                    deps_fp = {
                        f"argument_content:{event_id}/{arg_id}": candidate[
                            "argument_content_fingerprint"
                        ],
                        f"trigger_result:{event_id}": context["trigger_result"]["cache_key"],
                        f"schema_slice:{event_id}": context["scope"][
                            "schema_slice_fingerprint"
                        ],
                        f"role_weight:{candidate['role']}": candidate["rule_pack_fingerprint"],
                        "quantization": candidate["quantization_fingerprint"],
                    }
                    self.ledger.store_entry(
                        artifact_id,
                        "argument",
                        candidate["cache_key"],
                        candidate,
                        dependencies=sorted(deps_fp),
                        dependency_fingerprints=deps_fp,
                    )
                    self.manifest[artifact_id] = {
                        "cache_key": candidate["cache_key"],
                        "dependencies": sorted(deps_fp),
                        "dependency_fingerprints": deps_fp,
                    }
                    event_candidates[event_id].append(candidate)

                for fault in output["faults"]:
                    failed_argument_events.add(event_id)
                    event_injection = context["event_injection"]
                    self.add_fault(
                        stage="argument_extraction",
                        event_id=event_id,
                        business_code=fault["business_code"],
                        stage_code=fault["stage_code"],
                        message=f"{fault['arg_id']}:{fault['error']}",
                        injection_id=(
                            event_injection["injection_id"] if event_injection else None
                        ),
                        fault_type=(
                            event_injection.get("fault_type")
                            if event_injection
                            else None
                        ),
                    )

            if argument_batch.get("budget_exceeded"):
                self.count("E_ARG_BUDGET_EXCEEDED")
            for pending in argument_batch.get("pending_units", []):
                self.count("E_ARG_PENDING")

            pending_event_ids = {
                pending["event_id"] for pending in argument_batch.get("pending_units", [])
            }
            for event_id, candidates in event_candidates.items():
                candidates.sort(
                    key=lambda item: (
                        item["role"],
                        item["start"],
                        item["end"],
                        item["arg_id"],
                    )
                )
                argument_results[event_id] = {
                    "artifact_id": f"arguments:{event_id}",
                    "event_id": event_id,
                    "candidates": candidates,
                    "candidate_count": len(candidates),
                }

            decisions: dict[str, dict[str, Any]] = {}
            for event in valid_events:
                event_id = event["event_id"]
                if event_id not in argument_results:
                    continue
                if event_id in failed_argument_events or event_id in pending_event_ids:
                    continue
                schema_type = revision["event_types"][event["event_type"]]
                candidates = argument_results[event_id]["candidates"]
                decision_scope = schema_context[event["event_type"]]
                key_parts = decision_key_parts(
                    event,
                    candidates,
                    schema_type,
                    revision,
                    self.state.config,
                    decision_scope,
                )
                artifact_id = f"decision:{event_id}"
                deps_fp = {
                    "decision_rules": key_parts["rule_pack_fingerprint"],
                    **{
                        f"argument:{candidate['arg_id']}": candidate["cache_key"]
                        for candidate in candidates
                    },
                }
                self.note_transition(artifact_id, key_parts["cache_key"], deps_fp)
                cached_decision = self.reuse_or_none(
                    artifact_id,
                    key_parts["cache_key"],
                    dependencies=sorted(deps_fp),
                    dependency_fingerprints=deps_fp,
                )
                if cached_decision is not None:
                    decision = cached_decision
                else:
                    decision = adjudicate_event(
                        event,
                        argument_results[event_id],
                        schema_type,
                        revision,
                        self.state.config,
                        decision_scope,
                    )
                    self.ledger.store_entry(
                        artifact_id,
                        "adjudication",
                        key_parts["cache_key"],
                        decision,
                        dependencies=sorted(deps_fp),
                        dependency_fingerprints=deps_fp,
                    )
                    self.manifest[artifact_id] = {
                        "cache_key": key_parts["cache_key"],
                        "dependencies": sorted(deps_fp),
                        "dependency_fingerprints": deps_fp,
                    }
                decisions[event_id] = decision
                if decision["status"] == "incomplete":
                    self.count("E_EVENT_INCOMPLETE")
                    self.count("E_REQUIRED_ROLE_MISSING", len(decision["missing_required_roles"]))
                self.count("E_MUTEX_SUPPRESSED", len(decision["suppressed_candidates"]))
                self.count("E_BOUNDARY_CANDIDATE", len(decision["boundary_candidates"]))

            reference_injection = (
                injection
                if injection and injection.get("stage") == "reference"
                else None
            )
            reference_events = apply_reference_stage_injection(
                validation["valid_events"], reference_injection
            )
            references = build_reference_merges(
                reference_events,
                argument_results,
                decisions,
                self.state.config,
                validation["raw_arg_map"],
            )

            final_groups: list[dict[str, Any]] = []
            for group in references["merge_groups"]:
                artifact_id = group["artifact_id"]
                deps_fp = {
                    "merge_rules": group["rule_pack_fingerprint"],
                    **{
                        f"argument_cache_key:{key[:12]}": key
                        for key in group["argument_cache_keys"]
                    },
                }
                self.note_transition(artifact_id, group["cache_key"], deps_fp)
                cached_group = self.reuse_or_none(
                    artifact_id,
                    group["cache_key"],
                    dependencies=sorted(group["argument_ids"]),
                    dependency_fingerprints=deps_fp,
                )
                if cached_group is not None:
                    cached_group["merge_id"] = group["merge_id"]
                    final_groups.append(cached_group)
                    continue
                self.ledger.store_entry(
                    artifact_id,
                    "reference_merge",
                    group["cache_key"],
                    group,
                    dependencies=sorted(group["argument_ids"]),
                    dependency_fingerprints=deps_fp,
                )
                self.manifest[artifact_id] = {
                    "cache_key": group["cache_key"],
                    "dependencies": sorted(group["argument_ids"]),
                    "dependency_fingerprints": deps_fp,
                }
                final_groups.append(group)
            references["merge_groups"] = final_groups

            for edge in references["edge_decisions"]:
                if edge.get("status") == "fault":
                    injection_id = edge.get("injection_id")
                    self.add_fault(
                        stage="reference_merge",
                        business_code=edge["business_code"],
                        stage_code=edge["stage_code"],
                        message=edge.get("reason", "reference fault"),
                        event_id=edge.get("source_event_id"),
                        injection_id=injection_id,
                        fault_type="missing_relation_arg" if injection_id else None,
                    )
                else:
                    self.count(edge.get("business_code"))
            self.count("E_REF_MERGED", len(final_groups))

        self.ledger.commit_manifest(self.manifest)
        ledger_lists = self.ledger.run_lists()

        trigger_output = {
            "results": [triggers[eid] for eid in sorted(triggers)],
            "scheduler_batches": scheduler.batch_reports if revision else [],
        }
        argument_output = {
            "results": [argument_results[eid] for eid in sorted(argument_results)]
            if revision
            else []
        }
        adjudication_output = {
            "decisions": [decisions[eid] for eid in sorted(decisions)] if revision else []
        }
        validation_snapshot = {
            key: value
            for key, value in validation.items()
            if key not in {"raw_event_map", "raw_arg_map"}
        }
        scheduler_summary = scheduler.budget_summary()
        report = self.build_report(
            validation=validation_snapshot,
            trigger_output=trigger_output,
            argument_output=argument_output,
            adjudication_output=adjudication_output,
            references=references if revision else {"merge_groups": [], "edge_decisions": []},
            scheduler_summary=scheduler_summary,
            ledger_lists=ledger_lists,
            recovery=recovery,
        )

        stage_outputs = {
            "validation": validation_snapshot,
            "triggers": trigger_output,
            "arguments": argument_output,
            "adjudication": adjudication_output,
            "references": references if revision else {"merge_groups": [], "edge_decisions": []},
            "scheduler": scheduler_summary,
            "ledger": {"lists": ledger_lists, "manifest": self.manifest},
            "report": report,
        }
        snapshot_commit = None
        if self.persist_snapshot:
            snapshot_commit = self.snapshots.commit(
                stage_outputs,
                self.state.global_version,
                self.state.config_hash(),
            )
            report["snapshot_commit"]["committed_snapshot_id"] = snapshot_commit[
                "snapshot_id"
            ]
            report["snapshot_commit"]["version_changed"] = snapshot_commit[
                "version_changed"
            ]
            report["snapshot_commit"]["previous_snapshot"] = snapshot_commit.get(
                "previous_snapshot"
            )
            report_path = self.state_dir / "diagnostic_report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

        projection = canonical_projection(stage_outputs, self.state)
        return {
            "report": report,
            "projection": projection,
            "stage_outputs": stage_outputs,
            "ledger_lists": ledger_lists,
            "manifest": self.manifest,
            "snapshot_commit": snapshot_commit,
            "state_dir": str(self.state_dir),
        }

    def build_report(self, **kwargs: Any) -> dict[str, Any]:
        validation = kwargs["validation"]
        trigger_output = kwargs["trigger_output"]
        argument_output = kwargs["argument_output"]
        adjudication_output = kwargs["adjudication_output"]
        references = kwargs["references"]
        scheduler_summary = kwargs["scheduler_summary"]
        ledger_lists = kwargs["ledger_lists"]
        recovery = kwargs["recovery"]
        invalid_count = len(validation["invalid_entries"])
        report = {
            "status": "partial" if self.stage_faults or invalid_count else "ok",
            "global_version": self.state.global_version,
            "config_hash": self.state.config_hash(),
            "state_fingerprint": self.state.state_fingerprint(),
            "recovery": recovery,
            "snapshot_commit": {
                "overall_replace": True,
                "global_version": self.state.global_version,
                "config_hash": self.state.config_hash(),
            },
            "stage_completion": {
                "validation": {
                    "completed_valid_events": len(validation["valid_events"]),
                    "skipped_invalid_entries": invalid_count,
                    "revision_chain_valid": validation["revision_chain"]["valid"],
                },
                "trigger_match": {
                    "completed_events": len(trigger_output["results"]),
                    "total_matches": sum(
                        item["match_count"] for item in trigger_output["results"]
                    ),
                    "suppressed": sum(
                        len(item["suppressed"]) for item in trigger_output["results"]
                    ),
                },
                "argument_extraction": {
                    "event_results": len(argument_output["results"]),
                    "candidates": sum(
                        item["candidate_count"] for item in argument_output["results"]
                    ),
                },
                "role_adjudication": {
                    "completed_decisions": len(adjudication_output["decisions"]),
                    "incomplete_events": sum(
                        1
                        for item in adjudication_output["decisions"]
                        if item["status"] == "incomplete"
                    ),
                },
                "reference_merge": {
                    "merge_groups": references.get("merge_count", 0),
                    "edge_decisions": len(references.get("edge_decisions", [])),
                },
            },
            "restricted_recompute_set": ledger_lists["restricted_recompute_set"],
            "reuse_hits": ledger_lists["reuse_hits"],
            "invalidated": ledger_lists["invalidated"],
            "reuse_hit_entries": ledger_lists["reuse_hits"],
            "budget": {
                "consumed_points": scheduler_summary["consumed_points"],
                "global_limit": scheduler_summary["global_limit"],
                "per_event_consumed": scheduler_summary["per_event_consumed"],
                "pending_unit_count": scheduler_summary["pending_unit_count"],
                "pending_units": scheduler_summary["pending_units"],
                "peak_concurrency": scheduler_summary["peak_concurrency"],
                "max_workers": scheduler_summary["max_workers"],
                "concurrency_limit_respected": scheduler_summary[
                    "concurrency_limit_respected"
                ],
                "trigger_batch_limit": scheduler_summary["trigger_batch_limit"],
            },
            "business_code_counts": dict(sorted(self.business_counts.items())),
            "stage_fault_code_counts": dict(
                sorted(Counter(fault["stage_code"] for fault in self.stage_faults).items())
            ),
            "stage_faults": self.stage_faults,
            "injection_hits": sorted(
                self.injection_hits, key=lambda item: item["injection_id"]
            ),
            "ledger_summary": {
                "reuse_count": len(ledger_lists["reuse_hits"]),
                "recomputed_count": len(ledger_lists["recomputed"]),
                "invalidated_count": len(ledger_lists["invalidated"]),
            },
        }
        return report


def canonical_projection(stage_outputs: dict[str, Any], state: WorkspaceState) -> dict[str, Any]:
    validation = stage_outputs["validation"]
    triggers = stage_outputs["triggers"]["results"]
    arguments = stage_outputs["arguments"]["results"]
    decisions = stage_outputs["adjudication"]["decisions"]
    references = stage_outputs["references"]
    manifest = stage_outputs["ledger"]["manifest"]

    trigger_ids = [
        match["match_id"]
        for result in triggers
        for match in sorted(result["matches"], key=lambda m: m["match_id"])
    ]
    argument_ids = [
        candidate["artifact_id"]
        for result in arguments
        for candidate in sorted(
            result["candidates"], key=lambda c: (c["role"], c["start"], c["arg_id"])
        )
    ]
    suppressed_ids = [
        item["candidate_id"]
        for decision in decisions
        for item in sorted(
            decision["suppressed_candidates"], key=lambda x: x["candidate_id"]
        )
    ]
    decision_ids = [item["artifact_id"] for item in decisions]
    merge_ids = [item["artifact_id"] for item in references.get("merge_groups", [])]
    ledger_ids = sorted(manifest)
    projection = {
        "active_revision_id": state.active_revision_id,
        "validation": {
            "valid_event_ids": [event["event_id"] for event in validation["valid_events"]],
            "invalid_entries": validation["invalid_entries"],
        },
        "triggers": triggers,
        "arguments": arguments,
        "decisions": decisions,
        "references": {
            "edge_decisions": references.get("edge_decisions", []),
            "merge_groups": references.get("merge_groups", []),
        },
        "ledger_manifest": manifest,
        "orders": {
            "event_ids": [event["event_id"] for event in validation["valid_events"]],
            "trigger_match_ids": trigger_ids,
            "argument_ids": argument_ids,
            "decision_ids": decision_ids,
            "suppressed_candidate_ids": suppressed_ids,
            "merge_artifact_ids": merge_ids,
            "ledger_artifact_ids": ledger_ids,
        },
    }
    projection["projection_fingerprint"] = fingerprint(
        canonical_dumps(projection), 16
    )
    return projection


def projections_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return canonical_dumps(left) == canonical_dumps(right)
