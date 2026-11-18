"""Load and strongly validate the local JSON strategy manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

from .errors import SpecError
from .models import Manifest, Obj, Rule, Tier


def _is_int(value: Any) -> bool:
    # ``bool`` is a subclass of int but is never a valid manifest integer.
    return isinstance(value, int) and not isinstance(value, bool)


def _nonneg_int(errors: List[str], path: str, value: Any) -> bool:
    if not _is_int(value):
        errors.append(f"{path}: expected non-negative integer, got {value!r}")
        return False
    if value < 0:
        errors.append(f"{path}: expected non-negative integer, got {value}")
        return False
    return True


def _positive_int(errors: List[str], path: str, value: Any) -> bool:
    if not _is_int(value) or value <= 0:
        errors.append(f"{path}: expected positive integer, got {value!r}")
        return False
    return True


def _validate(raw: Any) -> Tuple[List[Tier], List[Obj], Rule, str, str, List[str]]:
    errors: List[str] = []
    if not isinstance(raw, dict):
        raise SpecError("manifest root must be a JSON object", {"path": "$"})

    policy_version = raw.get("policy_version")
    tier_param_version = raw.get("tier_param_version")
    for name, value in (
        ("policy_version", policy_version),
        ("tier_param_version", tier_param_version),
    ):
        if not isinstance(value, str) or not value:
            errors.append(f"$.{name}: expected non-empty string, got {value!r}")

    raw_tiers = raw.get("tiers")
    tiers: List[Tier] = []
    if not isinstance(raw_tiers, list) or len(raw_tiers) < 2:
        errors.append("$.tiers: expected a list of at least 2 tiers")
    else:
        seen_ids = set()
        seen_roles = set()
        for index, raw_tier in enumerate(raw_tiers):
            base = f"$.tiers[{index}]"
            if not isinstance(raw_tier, dict):
                errors.append(f"{base}: expected tier object")
                continue
            tier_id = raw_tier.get("tier_id")
            if not isinstance(tier_id, str) or not tier_id:
                errors.append(f"{base}.tier_id: expected non-empty string, got {tier_id!r}")
            elif tier_id in seen_ids:
                errors.append(f"{base}.tier_id: duplicate tier identifier {tier_id!r}")
            else:
                seen_ids.add(tier_id)
            role = raw_tier.get("role")
            if role is None:
                role = "hot" if index == 0 else "cold" if index == 1 else f"tier{index}"
            if not isinstance(role, str) or role not in ("hot", "cold"):
                errors.append(f"{base}.role: expected 'hot' or 'cold', got {role!r}")
            elif role in seen_roles:
                errors.append(f"{base}.role: duplicate tier role {role!r}")
            else:
                seen_roles.add(role)
            capacity = raw_tier.get("capacity_bytes")
            unit_cost = raw_tier.get("unit_cost")
            read_weight = raw_tier.get("read_weight")
            write_weight = raw_tier.get("write_weight")
            evict_priority = raw_tier.get("evict_priority")
            _nonneg_int(errors, f"{base}.capacity_bytes", capacity)
            _nonneg_int(errors, f"{base}.unit_cost", unit_cost)
            _nonneg_int(errors, f"{base}.read_weight", read_weight)
            _nonneg_int(errors, f"{base}.write_weight", write_weight)
            _nonneg_int(errors, f"{base}.evict_priority", evict_priority)
            if all(
                _is_int(v) and v >= 0
                for v in (capacity, unit_cost, read_weight, write_weight, evict_priority)
            ) and isinstance(tier_id, str) and tier_id and isinstance(role, str):
                tiers.append(
                    Tier(
                        tier_id=tier_id,
                        role=role,
                        capacity_bytes=capacity,
                        unit_cost=unit_cost,
                        read_weight=read_weight,
                        write_weight=write_weight,
                        evict_priority=evict_priority,
                    )
                )
        for required_role in ("hot", "cold"):
            if required_role not in seen_roles:
                errors.append(f"$.tiers: missing tier with role {required_role!r}")

    raw_objects = raw.get("objects")
    objects: List[Obj] = []
    if not isinstance(raw_objects, list):
        errors.append("$.objects: expected a list")
    else:
        seen_keys = set()
        for index, raw_obj in enumerate(raw_objects):
            base = f"$.objects[{index}]"
            if not isinstance(raw_obj, dict):
                errors.append(f"{base}: expected object entry")
                continue
            key = raw_obj.get("key")
            if not isinstance(key, str) or not key:
                errors.append(f"{base}.key: expected non-empty string, got {key!r}")
            elif key in seen_keys:
                errors.append(f"{base}.key: duplicate object key {key!r}")
            else:
                seen_keys.add(key)
            size_bytes = raw_obj.get("size_bytes")
            last_access_ms = raw_obj.get("last_access_ms")
            access_count = raw_obj.get("access_count")
            digest = raw_obj.get("digest")
            _positive_int(errors, f"{base}.size_bytes", size_bytes)
            _nonneg_int(errors, f"{base}.last_access_ms", last_access_ms)
            _nonneg_int(errors, f"{base}.access_count", access_count)
            if not isinstance(digest, str) or not digest:
                errors.append(f"{base}.digest: expected non-empty string, got {digest!r}")
            if (
                isinstance(key, str)
                and key
                and _is_int(size_bytes)
                and size_bytes > 0
                and _is_int(last_access_ms)
                and last_access_ms >= 0
                and _is_int(access_count)
                and access_count >= 0
                and isinstance(digest, str)
                and digest
            ):
                objects.append(
                    Obj(
                        key=key,
                        size_bytes=size_bytes,
                        last_access_ms=last_access_ms,
                        access_count=access_count,
                        digest=digest,
                    )
                )

    rule: Rule | None = None
    raw_rule = raw.get("rule")
    if not isinstance(raw_rule, dict):
        errors.append("$.rule: expected a rule object")
    else:
        hot_wm = raw_rule.get("hot_watermark_bytes")
        cold_reclaim = raw_rule.get("cold_reclaim_bytes")
        batch_size = raw_rule.get("batch_size")
        _nonneg_int(errors, "$.rule.hot_watermark_bytes", hot_wm)
        _nonneg_int(errors, "$.rule.cold_reclaim_bytes", cold_reclaim)
        _positive_int(errors, "$.rule.batch_size", batch_size)
        if (
            _is_int(hot_wm)
            and _is_int(cold_reclaim)
            and _is_int(batch_size)
            and batch_size > 0
        ):
            if hot_wm > cold_reclaim:
                errors.append(
                    "$.rule: hot_watermark_bytes must be <= cold_reclaim_bytes "
                    f"(got {hot_wm} > {cold_reclaim})"
                )
            rule = Rule(
                hot_watermark_bytes=hot_wm,
                cold_reclaim_bytes=cold_reclaim,
                batch_size=batch_size,
            )

    injection_sequence = raw.get("injection_sequence", [])
    if not isinstance(injection_sequence, list) or not all(
        isinstance(item, str) for item in injection_sequence
    ):
        errors.append("$.injection_sequence: expected a list of strings")
        injection_sequence = []

    # Cross-field capacity checks only when both structures parsed cleanly.
    if not errors and rule is not None:
        tier_by_role = {tier.role: tier for tier in tiers}
        if rule.hot_watermark_bytes > tier_by_role["hot"].capacity_bytes:
            errors.append(
                "$.rule.hot_watermark_bytes: exceeds hot tier capacity_bytes "
                f"({rule.hot_watermark_bytes} > {tier_by_role['hot'].capacity_bytes})"
            )
        if rule.cold_reclaim_bytes > tier_by_role["cold"].capacity_bytes:
            errors.append(
                "$.rule.cold_reclaim_bytes: exceeds cold tier capacity_bytes "
                f"({rule.cold_reclaim_bytes} > {tier_by_role['cold'].capacity_bytes})"
            )

    if errors:
        raise SpecError(
            f"manifest validation failed with {len(errors)} issue(s)",
            {"issues": errors},
        )

    return (
        tiers,
        objects,
        rule,  # type: ignore[return-value]
        policy_version,  # type: ignore[arg-type]
        tier_param_version,  # type: ignore[arg-type]
        list(injection_sequence),
    )


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"manifest file not found: {path}", {"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"manifest is not valid JSON: {exc.msg}",
            {"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    tiers, objects, rule, policy_version, tier_param_version, injections = _validate(raw)
    return Manifest(
        policy_version=policy_version,
        tier_param_version=tier_param_version,
        tiers=tiers,
        objects=objects,
        rule=rule,
        injection_sequence=injections,
    )
