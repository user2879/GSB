"""Arbitration engine: incremental recompute, concurrency, snapshots."""

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

from .cache import StageCache
from .errors import (E_EVIDENCE_INVALID, E_FIXED_WIDTH, E_RULE_CYCLE,
                     E_RULE_UNKNOWN, Failure, FailureList)
from .fingerprint import combine, fingerprint
from .injection import (INJ_SELF_REFERENCE, INJ_TIMESTAMP_ORDER,
                        INJ_TYPE_UNKNOWN, INJ_WIDTH_OVERFLOW)
from .loader import validate_ticket_evidence
from .models import STAGES, RelationEntry, RulePackage, TicketDecision
from .references import (affected_for_relation, affected_for_role,
                         affected_for_type, build_references)
from .serialization import attach_hints, serialize
from .stages import (run_arbitration_stage, run_decay_stage,
                     run_relation_stage)

GLOBAL_VERSION_START = 1


class EngineState:
    def __init__(self, tickets, rules, config, injections, work_dir):
        self.tickets = {t.tid: t for t in tickets}
        self.rules = rules
        self.config = config
        self.injections = list(injections)
        self.work_dir = work_dir
        self.cache_dir = os.path.join(work_dir, "cache")
        self.snapshot_dir = os.path.join(work_dir, "snapshot")
        self.tmp_dir = os.path.join(work_dir, "tmp")
        for path in (self.cache_dir, self.snapshot_dir, self.tmp_dir):
            os.makedirs(path, exist_ok=True)
        self.cache = StageCache(self.cache_dir)
        self.decisions = {}
        self.failures = FailureList()
        self.references = build_references(list(self.tickets.values()), rules)
        self.reused = []
        self.recomputed = []
        self.global_version = GLOBAL_VERSION_START
        self.exchange_version = "v2"
        self.command_count = 0
        self.last_summary = None
        self.snapshot_meta = None
        self.injection_hits = []
        self.budget_exceeded = False

    def ordered_tids(self):
        return sorted(self.tickets.keys())

    def config_fp(self):
        return fingerprint(self.config.to_dict())

    def rules_fp(self):
        return fingerprint({
            "type_weights": self.rules.type_weights,
            "role_cred": self.rules.role_cred,
            "mutex": [e.to_dict() for e in self.rules.mutex],
            "support": [e.to_dict() for e in self.rules.support],
            "tiers": [t.to_dict() for t in self.rules.tiers],
        })

    def reason_fp(self):
        return fingerprint(dict(sorted(self.rules.reason_dict.items())))

    def ticket_fp(self, tid):
        return fingerprint(self.tickets[tid].to_dict())

    def stage_key(self, tid, stage):
        if stage == "decay":
            return combine([self.ticket_fp(tid), self.config_fp(),
                            self.rules_fp(), "decay"])
        if stage == "relation":
            return combine([self.stage_key(tid, "decay"), self.rules_fp(),
                            self.config_fp(), "relation"])
        if stage == "arbitration":
            return combine([self.stage_key(tid, "relation"), self.rules_fp(),
                            self.reason_fp(), "arbitration"])
        if stage == "conversion":
            return combine([self.stage_key(tid, "arbitration"),
                            self.reason_fp(), self.exchange_version,
                            "conversion"])
        raise ValueError(f"unknown stage {stage}")


def _injection_for(state, tid, stage):
    return [inj for inj in state.injections if inj.tid == tid]


def _apply_injections(state, tid, stage, ticket, rules, config):
    """Build per-ticket working copies to trigger real input faults.

    The canonical ticket/rules/config objects are never mutated here so
    injected faults cannot leak into later commands or other tickets.
    """
    import copy
    hits = []
    local_rules = copy.deepcopy(rules)
    local_config = copy.deepcopy(config)
    local_ticket = copy.deepcopy(ticket)
    for inj in _injection_for(state, tid, stage):
        hits.append(inj)
        if inj.kind == INJ_TIMESTAMP_ORDER:
            ordered = local_ticket.sorted_evidence()
            if len(ordered) >= 2:
                first, second = ordered[0], ordered[1]
                swapped = []
                for ev in local_ticket.evidences:
                    if ev.seq == first.seq:
                        swapped.append(_replace_ev(ev, ts=second.ts))
                    elif ev.seq == second.seq:
                        swapped.append(_replace_ev(ev, ts=first.ts))
                    else:
                        swapped.append(ev)
                local_ticket.evidences = swapped
        elif inj.kind == INJ_TYPE_UNKNOWN and inj.target_seq is not None:
            local_ticket.evidences = [
                _replace_ev(ev, etype="GHOST_TYPE")
                if ev.seq == inj.target_seq else ev
                for ev in local_ticket.evidences
            ]
        elif inj.kind == INJ_SELF_REFERENCE:
            etype = local_ticket.evidences[0].etype
            local_rules = _clone_rules_with_self_relation(
                local_rules, etype, inj.inj_id)
        elif inj.kind == INJ_WIDTH_OVERFLOW:
            local_config = _clone_config(local_config, decimals=99)
    return local_ticket, local_rules, local_config, hits


def _clone_rules_with_self_relation(rules, etype, rid):
    return RulePackage(
        type_weights=dict(rules.type_weights),
        role_cred=dict(rules.role_cred),
        mutex=list(rules.mutex) + [RelationEntry(
            rid=rid, kind="mutex", left=etype, right=etype)],
        support=list(rules.support),
        tiers=list(rules.tiers),
        reason_dict=dict(rules.reason_dict),
    )

def _replace_ev(ev, **changes):
    data = ev.to_dict()
    data.update(changes)
    from .models import Evidence
    return Evidence.from_dict(data)



class _TicketTask:
    def __init__(self, tid, stage, sort_key):
        self.tid = tid
        self.stage = stage
        self.sort_key = sort_key

    @staticmethod
    def for_ticket(tid):
        return _TicketTask(tid, "full", f"{tid}|full")


def _failure_for_injection(inj, detail):
    mapping = {
        INJ_TIMESTAMP_ORDER: E_EVIDENCE_INVALID,
        INJ_TYPE_UNKNOWN: E_RULE_UNKNOWN,
        INJ_SELF_REFERENCE: E_RULE_CYCLE,
        INJ_WIDTH_OVERFLOW: E_FIXED_WIDTH,
    }
    return Failure(mapping[inj.kind], "injection", inj.tid,
                   f"{inj.inj_id}: {detail}", inj_id=inj.inj_id)


def compute_one_ticket(state, tid):
    """Run all stages for one ticket, honoring per-stage cache and injections."""
    ticket = state.tickets[tid]
    working_ticket, working_rules, working_config, hits = _apply_injections(
        state, tid, "decay", ticket, state.rules, state.config)

    def record_hits(failure):
        for inj in hits:
            state.injection_hits.append({
                "inj_id": inj.inj_id, "kind": inj.kind, "tid": tid,
                "stage": inj.stage, "code": failure.code,
            })
        state.failures.add(failure.code, failure.scope, failure.ref,
                           failure.detail, inj_id=failure.inj_id)

    validation_problem = validate_ticket_evidence(
        working_ticket, working_rules, working_config)
    if validation_problem is not None:
        code, detail = validation_problem
        failure = Failure(code, "ticket", tid, detail,
                          inj_id=hits[0].inj_id if hits else None)
        record_hits(failure)
        return tid, None, failure

    for inj in hits:
        if inj.kind == INJ_SELF_REFERENCE:
            failure = _failure_for_injection(
                inj, "self-referential relation endpoint")
            record_hits(failure)
            return tid, None, failure
        if inj.kind == INJ_WIDTH_OVERFLOW:
            failure = _failure_for_injection(
                inj, f"decimal width {working_config.decimals} out of range")
            record_hits(failure)
            return tid, None, failure

    stage_keys = {stage: state.stage_key(tid, stage) for stage in STAGES}

    decay_payload = state.cache.get(tid, "decay", stage_keys["decay"])
    if decay_payload is not None:
        from .models import EvidenceScore
        scored = [EvidenceScore.from_dict(row)
                  for row in decay_payload["payload"]["evidence_scores"]]
        base_ts = decay_payload["payload"]["base_ts"]
    else:
        base_ts, scored = run_decay_stage(
            working_ticket, working_rules, working_config)
        state.cache.put(tid, "decay", stage_keys["decay"], {
            "base_ts": base_ts,
            "evidence_scores": [row.to_dict() for row in scored],
        })

    relation_payload = state.cache.get(
        tid, "relation", stage_keys["relation"])
    if relation_payload is not None:
        from .models import RelationContribution
        final_scores = {int(k): int(v) for k, v in
                        relation_payload["payload"]["final_scores"].items()}
        contributions = [RelationContribution.from_dict(row) for row in
                         relation_payload["payload"]["contributions"]]
        reason_codes = list(relation_payload["payload"]["reason_codes"])
    else:
        final_scores, contributions, reason_codes = run_relation_stage(
            scored, working_rules, working_config)
        state.cache.put(tid, "relation", stage_keys["relation"], {
            "final_scores": {str(k): v for k, v in final_scores.items()},
            "contributions": [row.to_dict() for row in contributions],
            "reason_codes": reason_codes,
        })

    arbitration_payload = state.cache.get(
        tid, "arbitration", stage_keys["arbitration"])
    if arbitration_payload is not None:
        decision = TicketDecision.from_dict(arbitration_payload["payload"])
    else:
        decision = run_arbitration_stage(
            tid, base_ts, scored, final_scores, contributions, reason_codes,
            working_rules, working_config)
        state.cache.put(tid, "arbitration", stage_keys["arbitration"],
                        decision.to_dict())

    conversion_payload = state.cache.get(
        tid, "conversion", stage_keys["conversion"])
    if conversion_payload is None:
        state.cache.put(tid, "conversion", stage_keys["conversion"],
                        decision.to_dict())
    decision.stage_keys = stage_keys
    return tid, decision, None


class _TicketTask:
    def __init__(self, tid, stage, sort_key):
        self.tid = tid
        self.stage = stage
        self.sort_key = sort_key

    @staticmethod
    def for_ticket(tid):
        return _TicketTask(tid, "full", f"{tid}|full")


def _failure_for_injection(inj, detail):
    mapping = {
        INJ_TIMESTAMP_ORDER: E_EVIDENCE_INVALID,
        INJ_TYPE_UNKNOWN: E_RULE_UNKNOWN,
        INJ_SELF_REFERENCE: E_RULE_CYCLE,
        INJ_WIDTH_OVERFLOW: E_FIXED_WIDTH,
    }
    return Failure(mapping[inj.kind], "injection", inj.tid,
                   f"{inj.inj_id}: {detail}", inj_id=inj.inj_id)


def compute_one_ticket(state, tid):
    """Run all stages for one ticket, honoring cache and injections."""
    ticket = state.tickets[tid]
    config = state.config
    rules = state.rules
    working_ticket, working_rules, working_config, hits = _apply_injections(
        state, tid, "decay", ticket, rules, config)

    def record_hits(failure):
        for inj in hits:
            state.injection_hits.append({
                "inj_id": inj.inj_id, "kind": inj.kind, "tid": tid,
                "stage": inj.stage, "code": failure.code,
            })
        state.failures.add(failure.code, failure.scope, failure.ref,
                           failure.detail, inj_id=failure.inj_id)

    validation_problem = validate_ticket_evidence(
        working_ticket, working_rules, working_config)
    if validation_problem is not None:
        code, detail = validation_problem
        failure = Failure(code, "ticket", tid, detail,
                          inj_id=hits[0].inj_id if hits else None)
        record_hits(failure)
        return tid, None, failure

    for inj in hits:
        if inj.kind == INJ_SELF_REFERENCE:
            failure = _failure_for_injection(
                inj, "self-referential relation endpoint")
            record_hits(failure)
            return tid, None, failure
        if inj.kind == INJ_WIDTH_OVERFLOW:
            failure = _failure_for_injection(
                inj, f"decimal width {working_config.decimals} out of range")
            record_hits(failure)
            return tid, None, failure

    stage_keys = {stage: state.stage_key(tid, stage) for stage in STAGES}

    cached = state.cache.load_decision(
        tid, "conversion", stage_keys["conversion"])
    if cached is not None:
        cached.stage_keys = stage_keys
        return tid, cached, None

    base_ts, scored = run_decay_stage(working_ticket, working_rules,
                                      working_config)
    final_scores, contributions, reason_codes = run_relation_stage(
        scored, working_rules, working_config)
    decision = run_arbitration_stage(
        tid, base_ts, scored, final_scores, contributions, reason_codes,
        working_rules, working_config)
    decision.stage_keys = stage_keys

    state.cache.store_decision(decision, "decay", stage_keys["decay"])
    state.cache.store_decision(decision, "relation", stage_keys["relation"])
    state.cache.store_decision(decision, "arbitration",
                               stage_keys["arbitration"])
    state.cache.store_decision(decision, "conversion",
                               stage_keys["conversion"])
    return tid, decision, None


def run_pool(state, tids):
    """Thread-pool execution with stable per-ticket sort keys."""
    tids = sorted(tids)
    results = {}
    failures = {}
    max_workers = min(max(1, state.config.concurrency), 16)
    tasks = [_TicketTask.for_ticket(tid) for tid in tids]
    tasks.sort(key=lambda task: task.sort_key)
    if max_workers == 1:
        for task in tasks:
            tid, decision, failure = compute_one_ticket(state, task.tid)
            results[tid], failures[tid] = decision, failure
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(compute_one_ticket, state, task.tid)
                       for task in tasks]
            for future in futures:
                tid, decision, failure = future.result()
                results[tid], failures[tid] = decision, failure
    return results, failures


def _all_ticket_ids(state):
    return set(state.ordered_tids())


def determine_recompute_set(state, command):
    """Return the exact set of tids invalidated by one command."""
    op = command["op"]
    refs = state.references
    if op == "set_type_weight":
        return affected_for_type(refs, command["etype"])
    if op == "set_role_cred":
        return affected_for_role(refs, command["role"])
    if op in ("add_relation", "remove_relation"):
        rid = command["rid"]
        affected = affected_for_relation(refs, rid)
        if op == "add_relation":
            affected |= _tids_with_both_types(
                state, command["left"], command["right"])
        return affected
    if op in ("add_reason_code", "remove_reason_code"):
        return _all_ticket_ids(state)
    if op in ("set_tier_limit", "set_half_life", "set_decimals"):
        return _all_ticket_ids(state)
    if op == "replace_evidence":
        return {command["evidence"]["ticket_id"]}
    if op == "set_version":
        return set()
    if op in ("snapshot_now", "rebuild"):
        return _all_ticket_ids(state)
    return set()


def _tids_with_both_types(state, left, right):
    out = set()
    for tid, ticket in state.tickets.items():
        types = {ev.etype for ev in ticket.evidences}
        if left in types and right in types:
            out.add(tid)
    return out


def apply_command(state, command):
    """Mutate in-memory rules/config/tickets for a validated command."""
    op = command["op"]
    if op == "set_type_weight":
        state.rules.type_weights[command["etype"]] = int(command["weight"])
    elif op == "set_role_cred":
        state.rules.role_cred[command["role"]] = int(command["credibility"])
    elif op == "add_relation":
        entry = RelationEntry(rid=command["rid"], kind=command["kind"],
                              left=command["left"], right=command["right"])
        bucket = (state.rules.mutex if command["kind"] == "mutex"
                  else state.rules.support)
        if not any(existing.rid == entry.rid for existing in bucket):
            bucket.append(entry)
        state.references = build_references(
            list(state.tickets.values()), state.rules)
    elif op == "remove_relation":
        for bucket in (state.rules.mutex, state.rules.support):
            bucket[:] = [e for e in bucket if e.rid != command["rid"]]
        state.references = build_references(
            list(state.tickets.values()), state.rules)
    elif op == "add_reason_code":
        state.rules.reason_dict[command["code"]] = command["name"]
    elif op == "remove_reason_code":
        state.rules.reason_dict.pop(command["code"], None)
    elif op == "set_tier_limit":
        for tier in state.rules.tiers:
            if tier.level == int(command["level"]):
                object.__setattr__(tier, "limit", int(command["limit"]))
    elif op == "set_half_life":
        object.__setattr__(state.config, "half_life_ms", int(command["half_life_ms"]))
    elif op == "set_decimals":
        object.__setattr__(state.config, "decimals", int(command["decimals"]))
    elif op == "replace_evidence":
        from .models import Evidence
        ev = Evidence.from_dict(command["evidence"])
        ticket = state.tickets[ev.ticket_id]
        ticket.evidences = [e for e in ticket.evidences if e.seq != ev.seq]
        ticket.evidences.append(ev)
        state.references = build_references(
            list(state.tickets.values()), state.rules)
    elif op == "set_version":
        state.exchange_version = command["version"]
    state.command_count += 1
    state.global_version += 1


def _reason_distribution(decisions):
    dist = {}
    for decision in decisions:
        for code in decision.reason_codes:
            dist[code] = dist.get(code, 0) + 1
    return dict(sorted(dist.items()))


def build_summary(state, affected, stale=False, budget_hit=False):
    decisions = [state.decisions[tid] for tid in state.ordered_tids()
                 if state.decisions.get(tid) is not None]
    escalated = sum(1 for d in decisions if d.escalate)
    return {
        "global_version": state.global_version,
        "completed_tickets": len(decisions),
        "total_tickets": len(state.tickets),
        "escalated": escalated,
        "reason_distribution": _reason_distribution(decisions),
        "affected_tickets": sorted(affected),
        "reused": list(state.reused),
        "recomputed": list(state.recomputed),
        "failures": state.failures.to_list(),
        "progress": round(len(decisions) / max(1, len(state.tickets)), 6),
        "parameters_stale": stale,
        "budget_exceeded": budget_hit,
        "exchange_version": state.exchange_version,
    }


def execute_command(state, command, compute_delay_ms=0):
    affected = determine_recompute_set(state, command)
    apply_command(state, command)
    if command["op"] == "set_version":
        affected_after = set()
    else:
        affected_after = determine_recompute_set(state, command)
    targets = sorted(affected | affected_after)

    started = time.perf_counter()
    state.reused = []
    state.recomputed = []
    state.failures = FailureList()
    state.injection_hits = []
    state.budget_exceeded = False

    recompute_set = set(targets)
    for tid in state.ordered_tids():
        if tid in recompute_set:
            continue
        key = state.stage_key(tid, "conversion")
        cached = state.cache.load_decision(tid, "conversion", key)
        if cached is not None:
            cached.stage_keys = {stage: state.stage_key(tid, stage)
                                 for stage in STAGES}
            state.decisions[tid] = cached
            state.reused.append(tid)
        else:
            recompute_set.add(tid)

    ordered_targets = sorted(recompute_set)
    if compute_delay_ms:
        time.sleep(compute_delay_ms / 1000.0)
    results, failures = run_pool(state, ordered_targets)
    for tid in ordered_targets:
        if results.get(tid) is not None:
            state.decisions[tid] = results[tid]
            state.recomputed.append(tid)
    state.recomputed.sort()
    state.reused.sort()

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    budget_hit = elapsed_ms > state.config.budget_ms
    stale = False
    if budget_hit:
        stale = True
        state.budget_exceeded = True
        state.failures.add(
            "E_INTERACT_BUDGET_EXCEEDED", "interaction", command["op"],
            f"elapsed {elapsed_ms:.3f}ms > budget "
            f"{state.config.budget_ms}ms")
    summary = build_summary(state, targets, stale=stale,
                            budget_hit=budget_hit)
    summary["elapsed_ms"] = round(elapsed_ms, 3)
    state.last_summary = summary

    if state.command_count % max(1, state.config.snapshot_interval) == 0:
        write_snapshot(state)
    return summary


def full_rebuild(state):
    state.decisions = {}
    state.reused = []
    state.failures = FailureList()
    state.injection_hits = []
    # Preserve load-time validation failures across a forced rebuild while
    # allowing compute-time failures (including injections) to be regenerated.
    load_failures = [f for f in state.failures.items]
    results, failures = run_pool(state, state.ordered_tids())
    state.failures.items = load_failures + state.failures.items
    for tid in state.ordered_tids():
        if results.get(tid) is not None:
            state.decisions[tid] = results[tid]
    state.recomputed = sorted(tid for tid, decision in results.items()
                              if decision is not None)
    state.reused = []
    write_snapshot(state)
    return build_summary(state, set(state.ordered_tids()))


def write_snapshot(state):
    """Stage artifacts in tmp then atomically replace the snapshot dir."""
    tmp_root = os.path.join(state.tmp_dir,
                            f"snapshot_{state.global_version}")
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)
    artifacts_dir = os.path.join(tmp_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    decisions = [state.decisions[tid] for tid in state.ordered_tids()
                 if state.decisions.get(tid) is not None]
    attach_hints(state.rules, state.config, artifacts_dir)
    data_path, side_path = serialize(state.exchange_version, decisions,
                                     state.rules, state.config,
                                     artifacts_dir)
    manifest = {
        "global_version": state.global_version,
        "config_hash": state.config_fp(),
        "rules_fingerprint": state.rules_fp(),
        "reason_fingerprint": state.reason_fp(),
        "exchange_version": state.exchange_version,
        "completed_tickets": sorted(d.tid for d in decisions),
        "data_file": os.path.basename(data_path),
        "sidecar_file": os.path.basename(side_path),
        "injection_hits": list(state.injection_hits),
        "failures": state.failures.to_list(),
    }
    stage_versions = {}
    for tid in state.ordered_tids():
        decision = state.decisions.get(tid)
        if decision is None:
            continue
        stage_versions[tid] = {stage: state.stage_key(tid, stage)
                               for stage in STAGES}
    manifest["stage_versions"] = stage_versions
    with open(os.path.join(tmp_root, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2,
                  sort_keys=True)

    target = state.snapshot_dir
    backup = target + ".old"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    if os.path.exists(target):
        os.replace(target, backup)
    os.replace(tmp_root, target)
    if os.path.exists(backup):
        shutil.rmtree(backup)
    state.snapshot_meta = manifest
    return manifest


def load_snapshot_manifest(state):
    path = os.path.join(state.snapshot_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def restore_or_rebuild(state):
    """Resume only when version + config hash match; else stale rebuild."""
    manifest = load_snapshot_manifest(state)
    if manifest is None:
        return {"restored": False, "code": None}
    if manifest.get("config_hash") != state.config_fp():
        state.failures.add(
            "E_SNAPSHOT_STALE", "snapshot", "manifest",
            f"config hash mismatch "
            f"{manifest.get('config_hash')} != {state.config_fp()}")
        full_rebuild(state)
        return {"restored": False, "code": "E_SNAPSHOT_STALE"}
    if manifest.get("global_version") != state.global_version:
        state.failures.add(
            "E_SNAPSHOT_STALE", "snapshot", "manifest",
            "global version mismatch")
        full_rebuild(state)
        return {"restored": False, "code": "E_SNAPSHOT_STALE"}
    artifacts_dir = os.path.join(state.snapshot_dir, "artifacts")
    decisions = _deserialize_snapshot(manifest, artifacts_dir)
    for decision in decisions:
        state.decisions[decision.tid] = decision
    state.reused = sorted(d.tid for d in decisions)
    state.recomputed = []
    return {"restored": True, "code": None,
            "tickets": state.reused}


def _deserialize_snapshot(manifest, artifacts_dir):
    from .serialization import deserialize
    return deserialize(manifest["exchange_version"], artifacts_dir)


def diagnostic_report(state, round_trip=None):
    decisions = [state.decisions[tid] for tid in state.ordered_tids()
                 if state.decisions.get(tid) is not None]
    sidecar_hits = []
    for version in ("v1", "v2", "v3"):
        side_path = os.path.join(
            state.snapshot_dir, "artifacts", f"sidecar_{version}.json")
        if os.path.exists(side_path):
            with open(side_path, "r", encoding="utf-8") as fh:
                sidecar = json.load(fh)
            sidecar_hits.append({"version": version,
                                 "loss_count": len(sidecar.get("losses", [])),
                                 "losses": sidecar.get("losses", [])})
    return {
        "global_version": state.global_version,
        "config_hash": state.config_fp(),
        "rules_fingerprint": state.rules_fp(),
        "reason_fingerprint": state.reason_fp(),
        "ticket_count": len(state.tickets),
        "completed_tickets": sorted(d.tid for d in decisions),
        "escalated": sum(1 for d in decisions if d.escalate),
        "boundary_tickets": sorted(d.tid for d in decisions if d.boundary),
        "reused": list(state.reused),
        "recomputed": list(state.recomputed),
        "failures": state.failures.to_list(),
        "injection_hits": list(state.injection_hits),
        "sidecar_losses": sidecar_hits,
        "round_trip": round_trip or {},
    }

def _clone_config(config, **changes):
    data = config.to_dict()
    data.update(changes)
    from .models import ArbConfig
    return ArbConfig.from_dict(data)

