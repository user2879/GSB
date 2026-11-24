"""Replay verification and resident-vs-rescan byte comparison."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .canonical import canonical_json, object_digest
from .engine import MaterializationEngine
from .errors import ColmvError
from .ledger import LedgerStore
from .model import Change, RuntimeConfig
from .schema import load_config_with_views_object


def split_changes(changes: Sequence[Change], batch_sizes: Sequence[int]) -> list[list[Change]]:
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ColmvError("E_USAGE", "batch sizes must be positive integers", field="batch_sizes")
    if sum(batch_sizes) != len(changes):
        raise ColmvError(
            "E_USAGE",
            "batch plan must consume the complete change stream",
            field="batch_sizes",
            details={"planned_changes": sum(batch_sizes), "stream_changes": len(changes)},
        )
    batches: list[list[Change]] = []
    cursor = 0
    for size in batch_sizes:
        batches.append(list(changes[cursor : cursor + size]))
        cursor += size
    return batches


def split_uniform(changes: Sequence[Change], batch_size: int) -> list[list[Change]]:
    if batch_size <= 0:
        raise ColmvError("E_USAGE", "batch_size must be positive", field="batch_size")
    return [list(changes[offset : offset + batch_size]) for offset in range(0, len(changes), batch_size)]


def _group_identity(group: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(group.get("view_id", "")),
        canonical_json(group.get("group_key_typed", group.get("group_key", []))),
    )


def _first_value_diff(expected: Any, actual: Any, field: str = "") -> tuple[str, Any, Any] | None:
    if type(expected) is not type(actual):
        return (field or "$", expected, actual)
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys | actual_keys):
            child_field = f"{field}.{key}" if field else key
            if key not in expected:
                return (child_field, "<missing>", actual[key])
            if key not in actual:
                return (child_field, expected[key], "<missing>")
            diff = _first_value_diff(expected[key], actual[key], child_field)
            if diff is not None:
                return diff
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return (f"{field}.length" if field else "length", len(expected), len(actual))
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            child_field = f"{field}[{index}]" if field else f"[{index}]"
            diff = _first_value_diff(expected_item, actual_item, child_field)
            if diff is not None:
                return diff
        return None
    if expected != actual:
        return (field or "$", expected, actual)
    return None


def first_group_difference(
    expected_groups: Sequence[Mapping[str, Any]],
    actual_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    expected_map = {_group_identity(group): group for group in expected_groups}
    actual_map = {_group_identity(group): group for group in actual_groups}
    for identity in sorted(set(expected_map) | set(actual_map)):
        expected = expected_map.get(identity)
        actual = actual_map.get(identity)
        if expected is None:
            return {
                "view_id": identity[0],
                "group_key": json.loads(identity[1]),
                "field": "<group>",
                "expected": "<missing>",
                "actual": actual,
            }
        if actual is None:
            return {
                "view_id": identity[0],
                "group_key": json.loads(identity[1]),
                "field": "<group>",
                "expected": expected,
                "actual": "<missing>",
            }
        diff = _first_value_diff(expected, actual)
        if diff is not None:
            field, expected_value, actual_value = diff
            return {
                "view_id": identity[0],
                "group_key": json.loads(identity[1]),
                "field": field,
                "expected": expected_value,
                "actual": actual_value,
            }
    return None


def first_event_difference(
    expected_events: Sequence[Mapping[str, Any]],
    actual_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    expected_by_seq = {int(event["event_seq"]): event for event in expected_events}
    actual_by_seq = {int(event["event_seq"]): event for event in actual_events}
    for event_seq in sorted(set(expected_by_seq) | set(actual_by_seq)):
        expected = expected_by_seq.get(event_seq)
        actual = actual_by_seq.get(event_seq)
        if expected is None or actual is None:
            source = actual or expected or {}
            return {
                "event_seq": event_seq,
                "view_id": source.get("view_id"),
                "group_key": source.get("group_key", source.get("evicted_group_key")),
                "field": "<event>",
                "expected": expected if expected is not None else "<missing>",
                "actual": actual if actual is not None else "<missing>",
            }
        diff = _first_value_diff(expected, actual)
        if diff is not None:
            field, expected_value, actual_value = diff
            return {
                "event_seq": event_seq,
                "view_id": actual.get("view_id", expected.get("view_id")),
                "group_key": actual.get(
                    "group_key", actual.get("evicted_group_key", expected.get("group_key"))
                ),
                "field": field,
                "expected": expected_value,
                "actual": actual_value,
            }
    return None


def config_from_ledger(
    ledger: LedgerStore,
    catalog_path: str,
    scheme_version: int,
) -> RuntimeConfig:
    scheme = ledger.scheme(scheme_version)
    if scheme is None:
        raise ColmvError(
            "E_REPLAY",
            "ledger scheme version does not exist",
            details={"scheme_version": scheme_version},
        )
    views_object = json.loads(scheme["views_json"])
    config = load_config_with_views_object(catalog_path, views_object)
    if config.catalog_digest != scheme["catalog_digest"]:
        raise ColmvError(
            "E_REPLAY",
            "catalog digest differs from the ledger scheme",
            field="catalog_digest",
            details={
                "scheme_version": scheme_version,
                "expected": scheme["catalog_digest"],
                "actual": config.catalog_digest,
            },
        )
    if config.views_digest != scheme["views_digest"]:
        raise ColmvError(
            "E_REPLAY",
            "views digest differs from the ledger scheme",
            field="views_digest",
            details={
                "scheme_version": scheme_version,
                "expected": scheme["views_digest"],
                "actual": config.views_digest,
            },
        )
    expected_fingerprint = config.scheme_fingerprint(int(scheme["budget"]))
    if expected_fingerprint != scheme["fingerprint"]:
        raise ColmvError(
            "E_REPLAY",
            "scheme fingerprint differs after reconstruction",
            field="scheme_fingerprint",
            details={"expected": scheme["fingerprint"], "actual": expected_fingerprint},
        )
    return config


def _changes_for_batch(
    changes_by_seq: Mapping[int, Change],
    start_seq: int,
    end_seq: int,
) -> list[Change]:
    return [
        changes_by_seq[seq]
        for seq in sorted(changes_by_seq)
        if start_seq <= seq <= end_seq
    ]


def replay_and_verify(
    ledger: LedgerStore,
    *,
    catalog_path: str,
    changes: Sequence[Change],
    scheme_version: int | None = None,
) -> tuple[MaterializationEngine, dict[str, Any]]:
    scheme = ledger.current_scheme() if scheme_version is None else ledger.scheme(scheme_version)
    if scheme is None:
        raise ColmvError("E_REPLAY", "ledger has no current scheme", field="scheme_version")
    version = int(scheme["scheme_version"])
    config = config_from_ledger(ledger, catalog_path, version)
    stored_batches = ledger.committed_batches(version)
    if not stored_batches:
        raise ColmvError(
            "E_REPLAY",
            "ledger has no committed batches to replay",
            details={"scheme_version": version},
        )
    changes_by_seq = {change.seq: change for change in changes}
    engine = MaterializationEngine(
        config,
        budget=int(scheme["budget"]),
        path_mode="auto",
        scheme_version=version,
    )
    verified = 0
    for stored in stored_batches:
        batch_changes = _changes_for_batch(changes_by_seq, int(stored["start_seq"]), int(stored["end_seq"]))
        if len(batch_changes) != int(stored["change_count"]):
            raise ColmvError(
                "E_REPLAY",
                "batch change count differs from the ledger",
                field="change_count",
                details={
                    "batch_no": stored["batch_no"],
                    "expected": stored["change_count"],
                    "actual": len(batch_changes),
                },
            )
        actual = engine.process_batch(
            batch_changes,
            batch_no=int(stored["batch_no"]),
            force_path=str(stored["path"]),
        )
        expected_groups = json.loads(stored["groups_json"])
        expected_events = json.loads(stored["events_json"])
        group_diff = first_group_difference(expected_groups, actual["groups"])
        if group_diff is not None:
            raise ColmvError(
                "E_REPLAY",
                "replayed group outcome differs from the ledger",
                view_id=group_diff.get("view_id"),
                field=group_diff.get("field"),
                details={
                    "batch_no": stored["batch_no"],
                    "group_key": group_diff.get("group_key"),
                    "expected": group_diff.get("expected"),
                    "actual": group_diff.get("actual"),
                    "basis": "compare group outcome, aggregate estimate, and error bound",
                },
            )
        event_diff = first_event_difference(expected_events, actual["events"])
        if event_diff is not None:
            raise ColmvError(
                "E_REPLAY",
                "replayed replacement/rescan event differs from the ledger",
                view_id=event_diff.get("view_id"),
                field=event_diff.get("field"),
                details={
                    "batch_no": stored["batch_no"],
                    "event_seq": event_diff.get("event_seq"),
                    "group_key": event_diff.get("group_key"),
                    "expected": event_diff.get("expected"),
                    "actual": event_diff.get("actual"),
                    "basis": "compare event sequence, group key, page id, and reason",
                },
            )
        actual_materialized = engine.materialized_result()
        actual_materialized_digest = object_digest(actual_materialized)
        if actual_materialized_digest != stored["materialized_digest"]:
            expected_materialized = json.loads(stored["materialized_json"])
            expected_groups_flat = [
                group
                for view in expected_materialized.get("views", [])
                for group in view.get("groups", [])
            ]
            actual_groups_flat = [
                group
                for view in actual_materialized.get("views", [])
                for group in view.get("groups", [])
            ]
            group_diff = first_group_difference(expected_groups_flat, actual_groups_flat)
            raise ColmvError(
                "E_REPLAY",
                "replayed materialized result digest differs from the ledger",
                view_id=group_diff.get("view_id") if group_diff else None,
                field=group_diff.get("field") if group_diff else "materialized_digest",
                details={
                    "batch_no": stored["batch_no"],
                    "group_key": group_diff.get("group_key") if group_diff else None,
                    "expected": stored["materialized_digest"],
                    "actual": actual_materialized_digest,
                    "basis": "compare canonical materialized bytes",
                },
            )
        if int(actual["cursor"]) != int(stored["cursor"]):
            raise ColmvError(
                "E_REPLAY",
                "replayed cursor differs from the ledger",
                field="cursor",
                details={
                    "batch_no": stored["batch_no"],
                    "expected": stored["cursor"],
                    "actual": actual["cursor"],
                },
            )
        verified += 1

    consumed_max_seq = max(change.seq for change in changes)
    remaining = [change for change in changes if change.seq > engine.last_seq] if engine.last_seq is not None else list(changes)
    result = {
        "ok": True,
        "scheme_version": version,
        "verified_batches": verified,
        "cursor": engine.last_seq,
        "materialized_digest": engine.materialized_digest(),
        "materialized": engine.materialized_result(),
        "remaining_changes": len(remaining),
        "stream_complete": engine.last_seq == consumed_max_seq,
    }
    return engine, result


def verify_paths(
    config: RuntimeConfig,
    changes: Sequence[Change],
    *,
    budget: int,
    batch_sizes: Sequence[int],
    scheme_version: int = 1,
) -> dict[str, Any]:
    batches = split_changes(changes, batch_sizes)
    resident_engine = MaterializationEngine(
        config,
        budget=budget,
        path_mode="resident",
        scheme_version=scheme_version,
    )
    resident_batches: list[dict[str, Any]] = []
    for batch_no, batch_changes in enumerate(batches, start=1):
        resident_batches.append(resident_engine.process_batch(batch_changes, batch_no=batch_no))
    rescan_engine = MaterializationEngine(
        config,
        budget=budget,
        path_mode="rescan",
        scheme_version=scheme_version,
    )
    rescan_batches: list[dict[str, Any]] = []
    for batch_no, batch_changes in enumerate(batches, start=1):
        rescan_batches.append(rescan_engine.process_batch(batch_changes, batch_no=batch_no))

    resident_bytes = resident_engine.materialized_bytes()
    rescan_bytes = rescan_engine.materialized_bytes()
    if resident_bytes != rescan_bytes:
        resident_groups = [
            group
            for view in resident_engine.materialized_result()["views"]
            for group in view["groups"]
        ]
        rescan_groups = [
            group
            for view in rescan_engine.materialized_result()["views"]
            for group in view["groups"]
        ]
        diff = first_group_difference(resident_groups, rescan_groups)
        raise ColmvError(
            "E_DIVERGE",
            "resident and affected-page rescan paths produced different bytes",
            view_id=diff.get("view_id") if diff else None,
            field=diff.get("field") if diff else "materialized_bytes",
            details={
                "first_divergence": diff,
                "resident_bytes": len(resident_bytes),
                "rescan_bytes": len(rescan_bytes),
                "resident_digest": resident_engine.materialized_digest(),
                "rescan_digest": rescan_engine.materialized_digest(),
            },
        )

    partition_checks: list[dict[str, Any]] = []
    partitions_complementary = True
    for resident_batch, rescan_batch in zip(resident_batches, rescan_batches):
        resident_partition = resident_batch["slot_partition"]
        rescan_partition = rescan_batch["slot_partition"]
        resident_ids = {
            (group["view_id"], canonical_json(group["group_key"]))
            for group in resident_partition["resident_groups"]
        }
        rescan_ids = {
            (group["view_id"], canonical_json(group["group_key"]))
            for group in resident_partition["rescan_affected_groups"]
        }
        affected_ids = {
            (group["view_id"], canonical_json(group["group_key"]))
            for group in resident_batch["affected_groups"]
        }
        complementary = (
            resident_ids | rescan_ids == affected_ids
            and not (resident_ids & rescan_ids)
        )
        same_rescan_set = {
            (group["view_id"], canonical_json(group["group_key"]))
            for group in rescan_partition["rescan_affected_groups"]
        } == rescan_ids
        replacement_victim_ids = {
            (event["view_id"], canonical_json(event["evicted_group_key"]))
            for event in resident_batch["replacement_events"]
        }
        all_denied_ids = {
            (group["view_id"], canonical_json(group["group_key"]))
            for group in resident_partition["rescan_groups"]
        }
        rescan_event_ids = {
            (event["view_id"], canonical_json(event["group_key"]))
            for event in rescan_batch["rescan_events"]
        }
        victims_match_denied = replacement_victim_ids == all_denied_ids
        rescans_cover_victims = rescan_event_ids == replacement_victim_ids
        partitions_complementary = (
            partitions_complementary
            and complementary
            and same_rescan_set
            and victims_match_denied
            and rescans_cover_victims
        )
        partition_checks.append(
            {
                "batch_no": resident_batch["batch_no"],
                "resident_count": len(resident_ids),
                "rescan_count": len(rescan_ids),
                "affected_count": len(affected_ids),
                "resident_plus_rescan_equals_affected": resident_ids | rescan_ids == affected_ids,
                "disjoint": not (resident_ids & rescan_ids),
                "resident_and_rescan_paths_share_rescan_partition": same_rescan_set,
                "replacement_victims_equal_all_slot_denied_groups": victims_match_denied,
                "rescan_events_cover_replacement_victims": rescans_cover_victims,
            }
        )

    return {
        "ok": True,
        "equal_bytes": True,
        "byte_length": len(resident_bytes),
        "materialized_digest": resident_engine.materialized_digest(),
        "budget": budget,
        "batch_sizes": list(batch_sizes),
        "strict_slot_partition": {
            "valid": partitions_complementary,
            "basis": "for each batch, final resident groups + slot-denied groups = affected groups and are disjoint",
            "batches": partition_checks,
        },
        "resident": {
            "hits": resident_engine.total_hits,
            "replacements": resident_engine.total_replacements,
            "rescans": resident_engine.total_rescans,
            "final_resident_groups": [ref.to_dict() for ref in resident_engine.cache.snapshot()],
        },
        "rescan": {
            "hits": rescan_engine.total_hits,
            "replacements": rescan_engine.total_replacements,
            "rescans": rescan_engine.total_rescans,
            "final_resident_groups": [ref.to_dict() for ref in rescan_engine.cache.snapshot()],
        },
        "materialized": resident_engine.materialized_result(),
    }
