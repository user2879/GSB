"""Input/config loading and per-record quarantine.

The validator is deliberately tolerant at record scope: a malformed record is
reported as ``E_SAMPLE_FORMAT`` and never stops the rest of a batch or stream.
A malformed batch cursor quarantines that batch only.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .canonical import is_finite_number, read_json, stable_number
from .models import AcceptedRecord, Failure, ValidationResult

E_SAMPLE_FORMAT = "E_SAMPLE_FORMAT"
EXCHANGE_VERSIONS = {"p1", "p2", "p3"}
ALLOWED_DECIMALS = {2, 3, 4}
ALLOWED_TIERS = {"coarse", "fine"}


def load_json_file(path: str | Path, label: str) -> tuple[Any | None, list[Failure]]:
    try:
        return read_json(path), []
    except FileNotFoundError:
        return None, [
            Failure(
                code=E_SAMPLE_FORMAT,
                message=f"{label} file not found: {path}",
                scope="file",
                field="path",
            )
        ]
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, [
            Failure(
                code=E_SAMPLE_FORMAT,
                message=f"{label} JSON is invalid: {exc}",
                scope="file",
                field="json",
            )
        ]


def load_config(path: str | Path) -> tuple[dict[str, Any], list[Failure]]:
    payload, issues = load_json_file(path, "config")
    if issues:
        return {}, issues
    if not isinstance(payload, dict):
        return {}, [
            Failure(
                code=E_SAMPLE_FORMAT,
                message="config root must be an object",
                scope="config",
                field="$",
            )
        ]
    return payload, []


def load_batches(path: str | Path) -> tuple[list[dict[str, Any]], list[Failure]]:
    payload, issues = load_json_file(path, "batches")
    if issues:
        return [], issues
    batches = payload.get("batches") if isinstance(payload, dict) else payload
    if not isinstance(batches, list):
        return [], [
            Failure(
                code=E_SAMPLE_FORMAT,
                message="batches must be a list or an object with a batches list",
                scope="file",
            )
        ]
    return batches, []


def _issue(
    message: str,
    *,
    scope: str,
    field: str | None = None,
    batch_seq: int | None = None,
    log_id: str | None = None,
    injection_id: str | None = None,
) -> Failure:
    return Failure(
        code=E_SAMPLE_FORMAT,
        message=message,
        scope=scope,
        field=field,
        batch_seq=batch_seq,
        log_id=log_id,
        injection_id=injection_id,
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def active_profile_name(config: dict[str, Any]) -> str:
    return str(config.get("active_profile", "balanced"))


def active_profile(config: dict[str, Any]) -> dict[str, Any]:
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        return {}
    profile = profiles.get(active_profile_name(config))
    return profile if isinstance(profile, dict) else {}


def resolve_sampling(config: dict[str, Any]) -> dict[str, Any]:
    profile = active_profile(config)
    quantization = profile.get("quantization", {})
    return {
        "profile": active_profile_name(config),
        "global_budget": profile.get("global_budget"),
        "default_bucket_capacity": profile.get("bucket_capacity"),
        "default_priority": config.get("default_bucket_priority", 10),
        "quantization": (
            quantization if isinstance(quantization, dict) else {}
        ),
        "seed": config.get("seed"),
        "evidence_format": config.get("evidence_format", "p3"),
        "max_workers": config.get("max_workers", 4),
    }


def validate_config(config: dict[str, Any]) -> list[Failure]:
    issues: list[Failure] = []

    def config_issue(message: str, field: str) -> None:
        issues.append(_issue(message, scope="config", field=field))

    if not _non_negative_int(config.get("seed")):
        config_issue("seed must be a non-negative integer", "seed")

    devices = config.get("devices")
    if not isinstance(devices, list) or not devices:
        config_issue("devices must be a non-empty list", "devices")
    else:
        seen_devices: set[str] = set()
        for index, device in enumerate(devices):
            if not isinstance(device, dict):
                config_issue(
                    f"devices[{index}] must be an object", f"devices[{index}]"
                )
                continue
            device_id = device.get("device_id")
            if not isinstance(device_id, str) or not device_id:
                config_issue(
                    f"devices[{index}].device_id must be a non-empty string",
                    f"devices[{index}].device_id",
                )
                continue
            if (
                "|" in device_id
                or len(device_id) > 31
                or not device_id.isascii()
            ):
                config_issue(
                    "device_id must be at most 31 ASCII characters and cannot contain '|'",
                    f"devices[{index}].device_id",
                )
            if device_id in seen_devices:
                config_issue(
                    f"duplicate device_id {device_id}",
                    f"devices[{index}].device_id",
                )
            seen_devices.add(device_id)
            severities = device.get("severities")
            if not isinstance(severities, list) or not severities:
                config_issue(
                    f"devices[{index}].severities must be a non-empty enum list",
                    f"devices[{index}].severities",
                )
            elif any(
                not isinstance(level, str) or not level for level in severities
            ):
                config_issue(
                    "severity enum entries must be non-empty strings",
                    f"devices[{index}].severities",
                )
            elif len(set(severities)) != len(severities):
                config_issue(
                    f"devices[{index}].severities contains duplicates",
                    f"devices[{index}].severities",
                )
            elif any(
                "|" in level or len(level) > 31 or not level.isascii()
                for level in severities
            ):
                config_issue(
                    "severity enum entries must be at most 31 ASCII characters and cannot contain '|'",
                    f"devices[{index}].severities",
                )

    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        config_issue("profiles must be a non-empty object", "profiles")
    else:
        if active_profile_name(config) not in profiles:
            config_issue(
                f"active_profile {active_profile_name(config)} is not declared",
                "active_profile",
            )
        for name, profile in profiles.items():
            if not isinstance(name, str) or not name:
                config_issue("profile names must be non-empty strings", "profiles")
                continue
            if not isinstance(profile, dict):
                config_issue(f"profiles.{name} must be an object", f"profiles.{name}")
                continue
            if not _positive_int(profile.get("global_budget")):
                config_issue(
                    f"profiles.{name}.global_budget must be a positive integer",
                    f"profiles.{name}.global_budget",
                )
            if not _positive_int(profile.get("bucket_capacity")):
                config_issue(
                    f"profiles.{name}.bucket_capacity must be a positive integer",
                    f"profiles.{name}.bucket_capacity",
                )
            quantization = profile.get("quantization")
            if not isinstance(quantization, dict):
                config_issue(
                    f"profiles.{name}.quantization must be an object",
                    f"profiles.{name}.quantization",
                )
            else:
                allowed = config.get("allowed_quantization")
                if not isinstance(allowed, dict):
                    config_issue(
                        "allowed_quantization must be an object",
                        "allowed_quantization",
                    )
                    allowed_decimals = set(ALLOWED_DECIMALS)
                    allowed_tiers = set(ALLOWED_TIERS)
                else:
                    declared_decimals = allowed.get(
                        "decimals", list(ALLOWED_DECIMALS)
                    )
                    declared_tiers = allowed.get("tiers", list(ALLOWED_TIERS))
                    if not isinstance(declared_decimals, list) or not all(
                        _non_negative_int(value) for value in declared_decimals
                    ):
                        config_issue(
                            "allowed_quantization.decimals must be an integer list",
                            "allowed_quantization.decimals",
                        )
                        declared_decimals = []
                    if not isinstance(declared_tiers, list) or not all(
                        isinstance(value, str) for value in declared_tiers
                    ):
                        config_issue(
                            "allowed_quantization.tiers must be a string list",
                            "allowed_quantization.tiers",
                        )
                        declared_tiers = []
                    allowed_decimals = set(declared_decimals) & ALLOWED_DECIMALS
                    allowed_tiers = set(declared_tiers) & ALLOWED_TIERS
                decimals = quantization.get("decimals")
                tier = quantization.get("tier")
                if decimals not in allowed_decimals:
                    config_issue(
                        f"profiles.{name}.quantization.decimals is outside the allowed enum",
                        f"profiles.{name}.quantization.decimals",
                    )
                if tier not in allowed_tiers:
                    config_issue(
                        f"profiles.{name}.quantization.tier is outside the allowed enum",
                        f"profiles.{name}.quantization.tier",
                    )

    default_priority = config.get("default_bucket_priority", 10)
    if not _non_negative_int(default_priority):
        config_issue(
            "default_bucket_priority must be a non-negative integer",
            "default_bucket_priority",
        )

    overrides = config.get("bucket_overrides", {})
    if not isinstance(overrides, dict):
        config_issue("bucket_overrides must be an object", "bucket_overrides")
    else:
        device_map = _device_map(config)
        for bucket_id, override in overrides.items():
            if not isinstance(bucket_id, str) or "|" not in bucket_id:
                config_issue(
                    "bucket override keys must be '<device>|<severity>'",
                    f"bucket_overrides.{bucket_id}",
                )
                continue
            device_id, severity = bucket_id.split("|", 1)
            if (
                device_id not in device_map
                or severity not in device_map.get(device_id, set())
            ):
                config_issue(
                    f"bucket override {bucket_id} is not in the declared device/severity enum",
                    f"bucket_overrides.{bucket_id}",
                )
            elif len(bucket_id) > 32 or not bucket_id.isascii():
                config_issue(
                    f"bucket override {bucket_id} must be at most 32 ASCII characters",
                    f"bucket_overrides.{bucket_id}",
                )
            if not isinstance(override, dict):
                config_issue(
                    f"bucket_overrides.{bucket_id} must be an object",
                    f"bucket_overrides.{bucket_id}",
                )
                continue
            if "capacity" in override and not _positive_int(override["capacity"]):
                config_issue(
                    f"bucket_overrides.{bucket_id}.capacity must be a positive integer",
                    f"bucket_overrides.{bucket_id}.capacity",
                )
            if "priority" in override and not _non_negative_int(override["priority"]):
                config_issue(
                    f"bucket_overrides.{bucket_id}.priority must be a non-negative integer",
                    f"bucket_overrides.{bucket_id}.priority",
                )

    max_workers = config.get("max_workers", 4)
    if not _positive_int(max_workers):
        config_issue("max_workers must be a positive integer", "max_workers")

    evidence_format = config.get("evidence_format", "p3")
    if evidence_format not in EXCHANGE_VERSIONS:
        config_issue(
            "evidence_format must be one of p1, p2, p3", "evidence_format"
        )

    return issues


def _device_map(config: dict[str, Any]) -> dict[str, set[str]]:
    devices = config.get("devices", [])
    result: dict[str, set[str]] = {}
    if not isinstance(devices, list):
        return result
    for device in devices:
        if not isinstance(device, dict):
            continue
        device_id = device.get("device_id")
        severities = device.get("severities")
        if isinstance(device_id, str) and isinstance(severities, list):
            result[device_id] = {
                level for level in severities if isinstance(level, str)
            }
    return result


def bucket_capacity(config: dict[str, Any], bucket_id: str) -> int:
    override = config.get("bucket_overrides", {}).get(bucket_id, {})
    if isinstance(override, dict) and "capacity" in override:
        return int(override["capacity"])
    return int(active_profile(config)["bucket_capacity"])


def bucket_priority(config: dict[str, Any], bucket_id: str) -> int:
    override = config.get("bucket_overrides", {}).get(bucket_id, {})
    if isinstance(override, dict) and "priority" in override:
        return int(override["priority"])
    return int(config.get("default_bucket_priority", 10))


def _injection_applies(
    injection: dict[str, Any], batch_seq: int, record: dict[str, Any]
) -> bool:
    if injection.get("batch_seq") != batch_seq:
        return False
    target_log = injection.get("log_id")
    return target_log is None or target_log == record.get("log_id")


def _apply_record_injection(record: dict[str, Any], kind: str) -> None:
    injection_ids = list(record.get("__injection_ids__", []))
    if kind == "non_finite_metric":
        record["metric"] = math.nan
    elif kind == "unknown_bucket":
        record["device_id"] = "__injected_unknown_device__"
    elif kind == "missing_random_digest":
        record["random_digest"] = None
    elif kind == "batch_order_regression":
        record["__arrival_seq_regression__"] = True


def _valid_injection(injection: Any) -> bool:
    if not isinstance(injection, dict):
        return False
    if not isinstance(injection.get("id"), str) or not injection["id"]:
        return False
    if injection.get("type") not in {
        "non_finite_metric",
        "unknown_bucket",
        "missing_random_digest",
        "batch_order_regression",
    }:
        return False
    return isinstance(injection.get("batch_seq"), int)


def apply_injections(
    batches: list[dict[str, Any]], injections: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Failure]]:
    effective = copy.deepcopy(batches)
    applied: list[dict[str, Any]] = []
    issues: list[Failure] = []
    for injection in injections or []:
        if not _valid_injection(injection):
            issues.append(
                _issue(
                    "injection declaration is malformed",
                    scope="injection",
                    field="injections",
                    injection_id=(
                        str(injection.get("id"))
                        if isinstance(injection, dict)
                        and injection.get("id") is not None
                        else None
                    ),
                )
            )
            continue
        kind = injection["type"]
        batch_seq = injection["batch_seq"]
        matched = False
        for batch in effective:
            if not isinstance(batch, dict) or batch.get("arrival_seq") != batch_seq:
                continue
            records = batch.get("records")
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                if not _injection_applies(injection, batch_seq, record):
                    continue
                matched = True
                _apply_record_injection(record, kind)
                injection_ids = list(record.get("__injection_ids__", []))
                if injection["id"] not in injection_ids:
                    injection_ids.append(injection["id"])
                record["__injection_ids__"] = injection_ids
                if kind == "batch_order_regression":
                    batch["arrival_seq"] = 0
                    batch_ids = list(batch.get("__injection_ids__", []))
                    if injection["id"] not in batch_ids:
                        batch_ids.append(injection["id"])
                    batch["__injection_ids__"] = batch_ids
        applied.append(
            {
                "id": injection["id"],
                "type": kind,
                "batch_seq": batch_seq,
                "log_id": injection.get("log_id"),
                "matched": matched,
            }
        )
        if not matched:
            issues.append(
                _issue(
                    "injection target was not found",
                    scope="injection",
                    batch_seq=batch_seq,
                    log_id=injection.get("log_id"),
                    injection_id=injection["id"],
                )
            )
    return effective, applied, issues


def validate_input(
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    injections: list[dict[str, Any]] | None = None,
) -> ValidationResult:
    """Validate config and stream, returning only clean records.

    Batch order is evaluated after canonical sorting by ``arrival_seq``. Any
    duplicate or non-positive cursor is rejected as a batch-level failure.
    """
    config_issues = validate_config(config)
    effective_batches, applied, injection_issues = apply_injections(
        batches, injections
    )
    failures: list[Failure] = [*config_issues, *injection_issues]
    if config_issues:
        return ValidationResult(
            accepted_records=[],
            failures=failures,
            accepted_batches=[],
            rejected_batches=[],
            completed_batch_cursor=0,
            injections_applied=applied,
            config_issues=config_issues,
        )

    ordered_batches = list(effective_batches)
    accepted: list[AcceptedRecord] = []
    accepted_batches: list[int] = []
    rejected_batches: list[int] = []
    seen_log_ids: set[str] = set()
    previous_seq = 0
    cursor = 0
    device_map = _device_map(config)

    for batch in ordered_batches:
        if not isinstance(batch, dict):
            failures.append(
                _issue(
                    "batch must be an object",
                    scope="batch",
                    field="$",
                )
            )
            continue
        arrival_seq = batch.get("arrival_seq")
        if not isinstance(arrival_seq, int) or isinstance(arrival_seq, bool):
            failures.append(
                _issue(
                    "batch arrival_seq must be a positive integer",
                    scope="batch",
                    field="arrival_seq",
                )
            )
            rejected_batches.append(-1)
            continue
        if arrival_seq <= previous_seq:
            failures.append(
                _issue(
                    f"batch arrival_seq {arrival_seq} is not strictly increasing after {previous_seq}",
                    scope="batch",
                    field="arrival_seq",
                    batch_seq=arrival_seq,
                    injection_id=(
                        batch["__injection_ids__"][0]
                        if batch.get("__injection_ids__")
                        else None
                    ),
                )
            )
            rejected_batches.append(arrival_seq)
            continue
        previous_seq = arrival_seq
        records = batch.get("records")
        if not isinstance(records, list):
            failures.append(
                _issue(
                    "batch records must be a list",
                    scope="batch",
                    field="records",
                    batch_seq=arrival_seq,
                )
            )
            rejected_batches.append(arrival_seq)
            continue

        batch_accepted = 0
        for record_index, raw_record in enumerate(records):
            tag_ids = (
                tuple(raw_record.get("__injection_ids__", []))
                if isinstance(raw_record, dict)
                else ()
            )
            injected_ids = tuple(
                dict.fromkeys(
                    [
                        *tag_ids,
                        *(
                            injection["id"]
                            for injection in applied
                            if injection["batch_seq"] == arrival_seq
                            and injection["log_id"] == (
                                raw_record.get("log_id")
                                if isinstance(raw_record, dict)
                                else None
                            )
                        ),
                    ]
                )
            )
            record, record_failures = _validate_record(
                raw_record,
                arrival_seq,
                record_index,
                device_map,
                seen_log_ids,
                injected_ids,
            )
            failures.extend(record_failures)
            if record is not None:
                accepted.append(record)
                batch_accepted += 1

        if batch_accepted:
            accepted_batches.append(arrival_seq)
            cursor = arrival_seq
        else:
            rejected_batches.append(arrival_seq)

    accepted.sort(key=lambda record: record.order_key)
    return ValidationResult(
        accepted_records=accepted,
        failures=sorted(
            failures,
            key=lambda item: (
                item.batch_seq or -1,
                item.scope,
                item.log_id or "",
                item.message,
            ),
        ),
        accepted_batches=accepted_batches,
        rejected_batches=rejected_batches,
        completed_batch_cursor=cursor,
        injections_applied=applied,
        config_issues=config_issues,
    )


def _validate_record(
    raw: Any,
    arrival_seq: int,
    record_index: int,
    device_map: dict[str, set[str]],
    seen_log_ids: set[str],
    injected_ids: tuple[str, ...],
) -> tuple[AcceptedRecord | None, list[Failure]]:
    failures: list[Failure] = []
    log_id = raw.get("log_id") if isinstance(raw, dict) else None
    injection_id = injected_ids[0] if injected_ids else None

    def fail(message: str, field: str) -> None:
        failures.append(
            _issue(
                message,
                scope="record",
                field=field,
                batch_seq=arrival_seq,
                log_id=log_id if isinstance(log_id, str) else None,
                injection_id=injection_id,
            )
        )

    if not isinstance(raw, dict):
        fail("record must be an object", "$")
        return None, failures
    if not isinstance(log_id, str) or not log_id:
        fail("log_id must be a non-empty string", "log_id")
    elif len(log_id) > 32 or "|" in log_id or not log_id.isascii():
        fail("log_id must be at most 32 ASCII characters and cannot contain '|'", "log_id")
    elif log_id in seen_log_ids:
        fail(f"log_id {log_id} is not globally unique", "log_id")

    timestamp = raw.get("timestamp")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        fail("timestamp must be an integer", "timestamp")

    device_id = raw.get("device_id")
    if not isinstance(device_id, str) or not device_id:
        fail("device_id must be a non-empty string", "device_id")
    elif device_id not in device_map:
        fail(f"device_id {device_id} is not declared", "device_id")

    severity = raw.get("severity")
    if not isinstance(severity, str) or not severity:
        fail("severity must be a non-empty string", "severity")
    elif isinstance(device_id, str) and device_id in device_map:
        if severity not in device_map[device_id]:
            fail(
                f"severity {severity} is outside declared enum for device {device_id}",
                "severity",
            )

    template_id = raw.get("template_id")
    if not isinstance(template_id, str) or not template_id:
        fail("template_id must be a non-empty string", "template_id")

    metric = raw.get("metric")
    if not is_finite_number(metric):
        fail("metric must be a finite non-negative number", "metric")
    elif float(metric) < 0:
        fail("metric must be a finite non-negative number", "metric")

    if "random_digest" in raw and not (
        isinstance(raw.get("random_digest"), str)
        and _is_hex_digest(raw.get("random_digest"))
    ):
        fail("random_digest must be a 32-byte hexadecimal digest", "random_digest")

    if "__arrival_seq_regression__" in raw:
        fail(
            "injected batch cursor regression quarantined this record",
            "arrival_seq",
        )

    if failures:
        return None, failures

    assert isinstance(log_id, str)
    assert isinstance(timestamp, int)
    assert isinstance(device_id, str)
    assert isinstance(severity, str)
    assert isinstance(template_id, str)
    seen_log_ids.add(log_id)
    return (
        AcceptedRecord(
            log_id=log_id,
            timestamp=timestamp,
            device_id=device_id,
            severity=severity,
            template_id=template_id,
            metric=stable_number(float(metric)),
            arrival_seq=arrival_seq,
            record_index=record_index,
            injection_ids=injected_ids,
        ),
        failures,
    )


def _is_hex_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True
