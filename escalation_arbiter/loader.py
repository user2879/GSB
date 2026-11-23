"""Loading and validation layer for tickets, rules, config and commands."""

import json
import os

from .errors import (E_COMMAND_INVALID, E_CONFIG_INVALID, E_DICT_DUPLICATE,
                     E_EVIDENCE_INVALID, E_FIXED_WIDTH, E_RULE_CYCLE,
                     E_RULE_UNKNOWN, E_THRESHOLD_ORDER, FailureList)
from .fixedpoint import MAX_DECIMAL_WIDTH
from .injection import Injection
from .models import (ArbConfig, Evidence, PriorityTier, RelationEntry,
                     RulePackage, Ticket)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_config(path: str) -> tuple:
    failures = FailureList()
    try:
        data = _read_json(path)
        cfg = ArbConfig.from_dict(data)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        failures.add(E_CONFIG_INVALID, "config", os.path.basename(path),
                     f"config unreadable: {exc}")
        return None, failures
    if not (0 <= cfg.decimals <= MAX_DECIMAL_WIDTH):
        failures.add(E_FIXED_WIDTH, "config", "decimals",
                     f"decimal width {cfg.decimals} outside "
                     f"[0, {MAX_DECIMAL_WIDTH}]")
    if cfg.quantum < 0:
        failures.add(E_CONFIG_INVALID, "config", "quantum",
                     "quantum must be non-negative")
    if cfg.half_life_ms <= 0:
        failures.add(E_CONFIG_INVALID, "config", "half_life_ms",
                     "half_life_ms must be positive")
    if cfg.fusion not in ("sum", "mean"):
        failures.add(E_CONFIG_INVALID, "config", "fusion",
                     f"unknown fusion {cfg.fusion!r}")
    if not (0 < cfg.concurrency <= 16):
        failures.add(E_CONFIG_INVALID, "config", "concurrency",
                     "concurrency must be in [1, 16]")
    if cfg.budget_ms < 0 or cfg.snapshot_interval <= 0:
        failures.add(E_CONFIG_INVALID, "config", "budget",
                     "budget_ms non-negative and snapshot_interval positive")
    if not (0 <= cfg.conf_min_permille <= cfg.conf_max_permille <= 1000):
        failures.add(E_CONFIG_INVALID, "config", "confidence_range",
                     "confidence range must sit inside [0, 1000] per-mille")
    return cfg, failures


def load_rule_package(path: str) -> tuple:
    failures = FailureList()
    data = _read_json(path)
    type_weights = {k: int(v) for k, v in data["type_weights"].items()}
    role_cred = {k: int(v) for k, v in data["role_credibility"].items()}

    def load_relations(rows):
        entries = []
        for row in rows:
            entry = RelationEntry(rid=row["rid"], kind=row["kind"],
                                  left=row["left"], right=row["right"])
            if entry.left == entry.right:
                failures.add(E_RULE_CYCLE, "relation", entry.rid,
                             f"self reference on type {entry.left!r}")
                continue
            missing = [t for t in (entry.left, entry.right)
                       if t not in type_weights]
            if missing:
                failures.add(E_RULE_UNKNOWN, "relation", entry.rid,
                             f"unknown endpoint type(s): {missing}")
                continue
            entries.append(entry)
        return entries

    mutex = load_relations(data.get("mutex", []))
    support = load_relations(data.get("support", []))

    tiers = []
    for row in data["priority_tiers"]:
        tiers.append(PriorityTier(limit=int(row["limit"]),
                                  level=int(row["level"]),
                                  label=row["label"]))
    for prev, nxt in zip(tiers, tiers[1:]):
        if prev.limit >= nxt.limit:
            failures.add(E_THRESHOLD_ORDER, "priority_tiers",
                         f"{prev.limit}->{nxt.limit}",
                         "tier limits must be strictly increasing")
            break

    reason_dict = {}
    seen = set()
    for row in data["reason_codes"]:
        code, name = row["code"], row["name"]
        if code in seen:
            failures.add(E_DICT_DUPLICATE, "reason_codes", code,
                         "duplicate reason code in dictionary")
            continue
        seen.add(code)
        reason_dict[code] = name

    rules = RulePackage(type_weights=type_weights, role_cred=role_cred,
                        mutex=mutex, support=support, tiers=tiers,
                        reason_dict=reason_dict)
    return rules, failures


def _validate_evidence(raw: dict, rules: RulePackage, cfg: ArbConfig) -> tuple:
    required = ("ticket_id", "seq", "etype", "role", "ts", "terms",
                "confidence")
    for key in required:
        if key not in raw:
            return None, (E_EVIDENCE_INVALID, f"missing field {key}")
    try:
        ev = Evidence.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        return None, (E_EVIDENCE_INVALID, f"malformed evidence: {exc}")
    if ev.ts < 0:
        return None, (E_EVIDENCE_INVALID, f"negative timestamp {ev.ts}")
    if not ev.terms or any(not str(t).strip() for t in ev.terms):
        return None, (E_EVIDENCE_INVALID, "terms must be non-empty")
    if not (cfg.conf_min_permille <= ev.confidence <=
            cfg.conf_max_permille):
        return None, (E_EVIDENCE_INVALID,
                      f"confidence {ev.confidence} outside declared range")
    if ev.etype not in rules.type_weights:
        return None, (E_RULE_UNKNOWN,
                      f"evidence type {ev.etype!r} not in weight table")
    if ev.role not in rules.role_cred:
        return None, (E_RULE_UNKNOWN,
                      f"role {ev.role!r} not in credibility table")
    return ev, None


def load_tickets(path: str, rules: RulePackage, cfg: ArbConfig) -> tuple:
    """Parse JSONL evidence rows into tickets; return (tickets, failures)."""
    failures = FailureList()
    tickets = {}
    seen_pairs = set()
    parse_failures = []

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_failures.append(
                    ("", lineno, E_EVIDENCE_INVALID,
                     f"line {lineno}: bad JSON ({exc})"))
                continue
            ev, problem = _validate_evidence(raw, rules, cfg)
            if problem is not None:
                code, detail = problem
                parse_failures.append(
                    (str(raw.get("ticket_id", "")), lineno, code, detail))
                continue
            pair = (ev.ticket_id, ev.seq)
            if pair in seen_pairs:
                parse_failures.append(
                    (ev.ticket_id, lineno, E_EVIDENCE_INVALID,
                     f"duplicate (ticket_id, seq)=({ev.ticket_id},{ev.seq})"))
                continue
            seen_pairs.add(pair)
            tickets.setdefault(ev.ticket_id, Ticket(ev.ticket_id)).evidences.append(ev)

    failed_tickets = {}
    for tid, _lineno, code, detail in parse_failures:
        failed_tickets.setdefault(tid, []).append((code, detail))
    for tid, problems in failed_tickets.items():
        for code, detail in sorted(problems):
            failures.add(code, "ticket", tid or "<unknown>", detail)

    valid_tickets = [t for tid, t in sorted(tickets.items())
                     if tid not in failed_tickets]
    for ticket in valid_tickets:
        ordered = ticket.sorted_evidence()
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt.ts < prev.ts:
                failures.add(E_EVIDENCE_INVALID, "ticket", ticket.tid,
                             f"timestamp inversion seq {prev.seq}->{nxt.seq}")
                failed_tickets[ticket.tid] = True
                break
    valid_tickets = [t for t in valid_tickets
                     if t.tid not in failed_tickets]
    return valid_tickets, failures


def validate_ticket_evidence(ticket: Ticket, rules: RulePackage,
                             cfg: ArbConfig) -> object:
    """Re-run evidence validation for a single ticket; Failure or None."""
    seen = set()
    for ev in ticket.evidences:
        _ev, problem = _validate_evidence(ev.to_dict(), rules, cfg)
        if problem is not None:
            return problem
        if (ev.ticket_id, ev.seq) in seen:
            return E_EVIDENCE_INVALID, "duplicate evidence sequence"
        seen.add((ev.ticket_id, ev.seq))
    ordered = ticket.sorted_evidence()
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt.ts < prev.ts:
            return E_EVIDENCE_INVALID, "timestamp inversion"
    return None


def load_commands(path: str) -> tuple:
    failures = FailureList()
    try:
        data = _read_json(path)
    except json.JSONDecodeError as exc:
        failures.add(E_COMMAND_INVALID, "commands", os.path.basename(path),
                     f"bad JSON: {exc}")
        return [], failures
    commands = data.get("commands", []) if isinstance(data, dict) else data
    known = {
        "set_type_weight", "set_role_cred", "add_relation",
        "remove_relation", "add_reason_code", "remove_reason_code",
        "set_tier_limit", "set_half_life", "set_decimals",
        "replace_evidence", "set_version", "snapshot_now", "rebuild",
    }
    valid = []
    for idx, cmd in enumerate(commands):
        if not isinstance(cmd, dict) or "op" not in cmd:
            failures.add(E_COMMAND_INVALID, "command", str(idx),
                         "command must be an object with an op")
            continue
        if cmd["op"] not in known:
            failures.add(E_COMMAND_INVALID, "command", str(idx),
                         f"unknown op {cmd['op']!r}")
            continue
        valid.append(cmd)
    return valid, failures


def load_injections(path: str) -> list:
    if not path or not os.path.exists(path):
        return []
    data = _read_json(path)
    return [Injection.from_dict(row) for row in data.get("injections", [])]
