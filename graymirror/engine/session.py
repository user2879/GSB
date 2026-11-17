"""Rule versioning, whole view replacement, staleness guard and snapshots."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_VERSION, GrayError
from .jsonio import RationalEncoder
from .fixedpoint import scale_from_exponent
from .loader import load_rules, load_stream
from .models import Request, Rule
from .router import RoutingEngine
from .statistics import compute_statistics

SNAPSHOT_FORMAT = "graymirror-snapshot/1"


def rules_fingerprint(rules: list[Rule], scale_exponent: int) -> str:
    canonical = json.dumps(
        {
            "scale_exponent": scale_exponent,
            "rules": [rule.__dict__ for rule in sorted(rules, key=lambda r: r.rule_id)],
        },
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stream_digest(requests: list[Request]) -> dict[str, Any]:
    canonical = "\n".join(
        json.dumps(req.__dict__, sort_keys=True, ensure_ascii=False)
        for req in requests
    )
    return {
        "count": len(requests),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "first_request_id": requests[0].request_id if requests else None,
        "last_request_id": requests[-1].request_id if requests else None,
    }


@dataclass
class RuleView:
    version: int
    scale_exponent: int
    scale: int
    rules: list[Rule]
    fingerprint: str
    stream_summary: dict[str, Any]
    quantized_report: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    mirrors: list[dict[str, Any]]
    statistics: dict[str, Any]
    stale: bool = False
    change_ledger: list[dict[str, Any]] = field(default_factory=list)


def _diff_rules(old: list[Rule], new: list[Rule],
                old_exp: int, new_exp: int) -> list[dict[str, Any]]:
    old_by_id = {rule.rule_id: rule for rule in old}
    new_by_id = {rule.rule_id: rule for rule in new}
    changes: list[dict[str, Any]] = []
    for rule_id in sorted(set(old_by_id) | set(new_by_id)):
        before, after = old_by_id.get(rule_id), new_by_id.get(rule_id)
        if before is None:
            changes.append({"rule_id": rule_id, "kind": "added"})
            continue
        if after is None:
            changes.append({"rule_id": rule_id, "kind": "removed"})
            continue
        for attr in ("gray_weight", "mirror_rate", "gray_retries",
                     "mirror_timeout_ms", "prefix", "stable_target",
                     "gray_target", "mirror_target"):
            if getattr(before, attr) != getattr(after, attr):
                entry = {"rule_id": rule_id, "kind": "changed",
                         "field": attr, "old": getattr(before, attr),
                         "new": getattr(after, attr)}
                if attr in ("gray_weight", "mirror_rate"):
                    entry["delta_per_10000"] = getattr(after, attr) - getattr(before, attr)
                changes.append(entry)
    if old_exp != new_exp:
        changes.append({"rule_id": "*", "kind": "grid_changed",
                        "field": "scale_exponent", "old": old_exp, "new": new_exp})
    return changes


def build_view(version: int, rules: list[Rule], scale_exponent: int,
               requests: list[Request],
               previous: RuleView | None = None) -> RuleView:
    scale = scale_from_exponent(scale_exponent)
    engine = RoutingEngine(rules, scale)
    result = engine.run(requests)
    stats = compute_statistics(
        rules, result["quantized_rules"],
        result["decisions"], result["mirrors"])
    ledger: list[dict[str, Any]] = []
    if previous is not None:
        ledger = _diff_rules(previous.rules, rules,
                             previous.scale_exponent, scale_exponent)
        old_bounds = {
            item["rule_id"]: item.get("bounds", {})
            for item in previous.statistics.get("per_rule", {}).values()
        }
        for item in stats["per_rule"].values():
            previous_bound = old_bounds.get(item["rule_id"], {})
            new_bound = item["bounds"]
            for key, new_value in new_bound.items():
                old_value = previous_bound.get(key)
                if old_value is not None and old_value != new_value:
                    ledger.append({
                        "rule_id": item["rule_id"], "kind": "bound_changed",
                        "field": key, "old": old_value, "new": new_value,
                    })
    return RuleView(
        version=version, scale_exponent=scale_exponent, scale=scale, rules=rules,
        fingerprint=rules_fingerprint(rules, scale_exponent),
        stream_summary=stream_digest(requests),
        quantized_report=[
            result["quantized_rules"][rule.rule_id].to_report()
            for rule in sorted(rules, key=lambda r: r.rule_id)
        ],
        decisions=[decision.to_report() for decision in result["decisions"]],
        mirrors=[mirror.to_report() for mirror in result["mirrors"]],
        statistics=stats, stale=False, change_ledger=ledger,
    )


class Session:
    """Owns the current view; replacement marks every prior view stale."""

    def __init__(self) -> None:
        self.views: list[RuleView] = []

    @property
    def current(self) -> RuleView:
        if not self.views:
            raise GrayError(E_VERSION, "session has no active rule view")
        return self.views[-1]

    def replace(self, rules: list[Rule], scale_exponent: int,
                requests: list[Request]) -> RuleView:
        previous = self.views[-1] if self.views else None
        version = (previous.version + 1) if previous is not None else 1
        view = build_view(version, rules, scale_exponent, requests, previous)
        for old in self.views:
            old.stale = True
        self.views.append(view)
        return view

    # ---- snapshot persistence ---------------------------------------
    def snapshot(self, path: str | Path) -> Path:
        view = self.current
        payload = {
            "format": SNAPSHOT_FORMAT,
            "version": view.version,
            "scale_exponent": view.scale_exponent,
            "rule_fingerprint": view.fingerprint,
            "stream_summary": view.stream_summary,
            "quantized_rules": view.quantized_report,
            "statistics": view.statistics,
            "decisions": view.decisions,
            "mirrors": view.mirrors,
            "change_ledger": view.change_ledger,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, cls=RationalEncoder, ensure_ascii=False,
                          indent=2, sort_keys=True),
            encoding="utf-8")
        return path

    @staticmethod
    def load_snapshot(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GrayError(E_VERSION, f"cannot read snapshot: {exc}",
                            field_path=str(path))
        if payload.get("format") != SNAPSHOT_FORMAT:
            raise GrayError(E_VERSION, "unsupported snapshot format",
                            field_path="format",
                            details={"got": payload.get("format")})
        required = ("version", "stream_summary", "statistics",
                    "decisions", "mirrors", "quantized_rules")
        missing = [name for name in required if name not in payload]
        if missing:
            raise GrayError(E_VERSION, "snapshot missing required field(s)",
                            field_path=missing[0], details={"missing": missing})
        if "scale_exponent" not in payload:
            raise GrayError(E_VERSION, "snapshot scale exponent is missing",
                            field_path="scale_exponent")
        return payload


def restore_session(rule_path: str | Path, stream_path: str | Path,
                    snapshot_path: str | Path) -> tuple[Session, dict[str, Any], list[Request]]:
    """Whole state replacement restore; mismatch raises E_VERSION, never merges."""
    snapshot = Session.load_snapshot(snapshot_path)
    rules = load_rules(rule_path)
    requests = load_stream(stream_path, rules)

    current_fingerprint = rules_fingerprint(rules, snapshot["scale_exponent"])
    current_summary = stream_digest(requests)
    differences: list[dict[str, Any]] = []
    if current_fingerprint != snapshot.get("rule_fingerprint"):
        differences.append({
            "field": "rule_fingerprint",
            "snapshot": snapshot.get("rule_fingerprint"),
            "current": current_fingerprint,
        })
    if current_summary != snapshot.get("stream_summary"):
        differences.append({
            "field": "stream_summary",
            "snapshot": snapshot.get("stream_summary"),
            "current": current_summary,
        })
    if differences:
        raise GrayError(
            E_VERSION,
            "snapshot cannot align with current session; whole view replacement required",
            field_path=",".join(item["field"] for item in differences),
            details={"differences": differences})
    session = Session()
    view = build_view(snapshot["version"], rules,
                      snapshot["scale_exponent"], requests)
    session.views.append(view)
    return session, snapshot, requests
