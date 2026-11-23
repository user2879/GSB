"""Root-cause rules and silence-window specification (strongly validated)."""
from __future__ import annotations

from dataclasses import dataclass

from .errors import SpecError


@dataclass(frozen=True)
class RootCauseRule:
    rule_id: str
    match_service: str          # "*" matches every service
    include_upstream: bool
    include_downstream: bool
    max_layer: int


@dataclass(frozen=True)
class RuleSpec:
    declared_version: int
    schema_version: int
    default_silence_ms: int
    window_version: int
    rule_version: int
    silence_by_fingerprint: dict[str, int]
    silence_by_service: dict[str, int]
    rules: tuple[RootCauseRule, ...]

    def effective_silence_ms(self, service_id: str, fingerprint: str,
                             record_silence_ms: int) -> int:
        if fingerprint in self.silence_by_fingerprint:
            return self.silence_by_fingerprint[fingerprint]
        if service_id in self.silence_by_service:
            return self.silence_by_service[service_id]
        return record_silence_ms

    def matching_rules(self, service_id: str) -> list[RootCauseRule]:
        return [r for r in self.rules
                if r.match_service in ("*", service_id)]


def _nonneg_int(doc: dict, field: str) -> int:
    value = doc.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpecError(f"{field} must be a non-negative integer", 0, field,
                        repr(value))
    return value


def load_rule_spec(doc: dict, known_services: set[str]) -> RuleSpec:
    if not isinstance(doc, dict):
        raise SpecError("rule document must be a JSON object", 0, "document")
    declared_version = _nonneg_int(doc, "declared_version")
    schema_version = _nonneg_int(doc, "schema_version")
    default_silence = _nonneg_int(doc, "default_silence_window_ms")
    window_version = _nonneg_int(doc, "window_version")
    rule_version = _nonneg_int(doc, "rule_version")

    windows = doc.get("silence_overrides", {})
    if not isinstance(windows, dict):
        raise SpecError("silence_overrides must be an object", 0,
                        "silence_overrides")
    by_fp = windows.get("by_fingerprint", {})
    by_svc = windows.get("by_service", {})
    if not isinstance(by_fp, dict) or not isinstance(by_svc, dict):
        raise SpecError("silence override maps must be objects", 0,
                        "silence_overrides")
    for fp, ms in by_fp.items():
        if isinstance(ms, bool) or not isinstance(ms, int) or ms < 0:
            raise SpecError("silence override must be a non-negative integer",
                            0, f"silence_overrides.by_fingerprint.{fp}")
    for svc, ms in by_svc.items():
        if svc not in known_services:
            raise SpecError("silence override references unknown service", 0,
                            f"silence_overrides.by_service.{svc}")
        if isinstance(ms, bool) or not isinstance(ms, int) or ms < 0:
            raise SpecError("silence override must be a non-negative integer",
                            0, f"silence_overrides.by_service.{svc}")

    raw_rules = doc.get("root_cause_rules", [])
    if not isinstance(raw_rules, list):
        raise SpecError("root_cause_rules must be an array", 0,
                        "root_cause_rules")
    rules: list[RootCauseRule] = []
    seen_rule_ids: set[str] = set()
    for index, raw in enumerate(raw_rules, start=1):
        if not isinstance(raw, dict) or "rule_id" not in raw:
            raise SpecError("rule entry missing rule_id", index, "rule_id")
        rid = raw["rule_id"]
        if rid in seen_rule_ids:
            raise SpecError("duplicate rule_id", index, "rule_id", rid)
        seen_rule_ids.add(rid)
        match = raw.get("match_service", "*")
        if match != "*" and match not in known_services:
            raise SpecError("rule references unknown service", index,
                            "match_service", match)
        max_layer = raw.get("max_layer", 2 ** 31 - 1)
        if isinstance(max_layer, bool) or not isinstance(max_layer, int) \
                or max_layer < 0:
            raise SpecError("max_layer must be a non-negative integer", index,
                            "max_layer")
        rules.append(RootCauseRule(
            rule_id=rid,
            match_service=match,
            include_upstream=bool(raw.get("include_upstream", True)),
            include_downstream=bool(raw.get("include_downstream", False)),
            max_layer=max_layer,
        ))
    # Stable rule order independent of file order.
    rules.sort(key=lambda r: r.rule_id)
    return RuleSpec(
        declared_version=declared_version,
        schema_version=schema_version,
        default_silence_ms=default_silence,
        window_version=window_version,
        rule_version=rule_version,
        silence_by_fingerprint=dict(by_fp),
        silence_by_service=dict(by_svc),
        rules=tuple(rules),
    )
