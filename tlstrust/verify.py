"""Deterministic offline certificate-chain validation and flip accounting."""

from __future__ import annotations

import base64
from typing import Any, Iterable

from .canonical import canonical_json, digest_obj, digest_without_field
from .errors import E_CHAIN, E_SNAPSHOT, WorkbenchError
from .trust import Revision, TrustDocument
from .x509 import CertRecord, parse_certificate


VALIDATOR_VERSION = 1
FAILURE_CATEGORIES = (
    "UNTRUSTED_ANCHOR",
    "BAD_SIGNATURE",
    "PATH_LEN",
    "EXPIRED",
    "REVOKED",
)


def result_index(chain_fingerprints: list[str], revision_digest: str, failure_category: str | None) -> str:
    payload = {
        "chain_fingerprints": list(chain_fingerprints),
        "revision_digest": revision_digest,
        "failure_category": failure_category if failure_category is not None else "NONE",
    }
    digest = digest_obj(payload)
    return "idx:" + digest.split(":", 1)[1]


def snapshot_chain_records(snapshot: dict[str, Any]) -> list[CertRecord]:
    if not isinstance(snapshot, dict):
        raise WorkbenchError(E_SNAPSHOT, "input snapshot must be an object", field_path="snapshot")
    if snapshot.get("format_version") != 1 or snapshot.get("kind") != "tlstrust.input_snapshot":
        raise WorkbenchError(
            E_SNAPSHOT,
            "unsupported input snapshot format",
            field_path="snapshot.format_version",
        )
    digest = snapshot.get("snapshot_digest")
    if not isinstance(digest, str) or digest != digest_without_field(snapshot, "snapshot_digest"):
        raise WorkbenchError(
            E_SNAPSHOT,
            "input snapshot content digest mismatch",
            field_path="snapshot.snapshot_digest",
        )
    chain_raw = snapshot.get("chain")
    if not isinstance(chain_raw, list) or not chain_raw:
        raise WorkbenchError(E_CHAIN, "input snapshot has no certificate chain", field_path="snapshot.chain")
    records: list[CertRecord] = []
    for index, item in enumerate(chain_raw):
        if not isinstance(item, dict):
            raise WorkbenchError(E_CHAIN, "chain entry must be an object", layer=index, field_path=f"snapshot.chain[{index}]")
        encoded = item.get("der_base64")
        if not isinstance(encoded, str):
            raise WorkbenchError(E_CHAIN, "chain entry has no DER", layer=index, field_path=f"snapshot.chain[{index}].der_base64")
        try:
            der = base64.b64decode(encoded, validate=True)
            record = parse_certificate(der)
        except Exception as exc:
            raise WorkbenchError(
                E_CHAIN,
                "certificate in input snapshot cannot be parsed",
                layer=index,
                field_path=f"snapshot.chain[{index}].der_base64",
            ) from exc
        if item.get("fingerprint") != record.fingerprint:
            raise WorkbenchError(
                E_CHAIN,
                "certificate fingerprint in snapshot does not match DER content",
                layer=index,
                field_path=f"snapshot.chain[{index}].fingerprint",
                details={"expected": item.get("fingerprint"), "actual": record.fingerprint},
            )
        if item.get("serial_number") != record.serial_number:
            raise WorkbenchError(
                E_CHAIN,
                "certificate serial in snapshot does not match DER content",
                layer=index,
                field_path=f"snapshot.chain[{index}].serial_number",
                details={"expected": item.get("serial_number"), "actual": record.serial_number},
            )
        if item.get("position") not in (None, index):
            raise WorkbenchError(
                E_CHAIN,
                "certificate chain positions are not contiguous",
                layer=index,
                field_path=f"snapshot.chain[{index}].position",
            )
        records.append(record)
    actual_fingerprints = [record.fingerprint for record in records]
    declared = snapshot.get("chain_fingerprints")
    if not isinstance(declared, list) or declared != actual_fingerprints:
        raise WorkbenchError(
            E_CHAIN,
            "snapshot chain fingerprint sequence does not match certificate DER",
            layer=0,
            field_path="snapshot.chain_fingerprints",
            details={"expected": actual_fingerprints, "actual": declared},
        )
    if snapshot.get("chain_digest") not in (None, digest_obj(actual_fingerprints)):
        raise WorkbenchError(
            E_CHAIN,
            "snapshot chain digest does not match its certificate fingerprints",
            layer=0,
            field_path="snapshot.chain_digest",
        )
    actual_serials = [record.serial_number for record in records]
    declared_serials = snapshot.get("chain_serial_numbers")
    if not isinstance(declared_serials, list) or declared_serials != actual_serials:
        raise WorkbenchError(
            E_CHAIN,
            "snapshot chain serial sequence does not match certificate DER",
            layer=0,
            field_path="snapshot.chain_serial_numbers",
            details={"expected": actual_serials, "actual": declared_serials},
        )
    return records


def _check(
    order: int,
    layer: int,
    fingerprint: str,
    check_name: str,
    passed: bool,
    category: str | None,
    expected: Any,
    actual: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "order": order,
        "layer": layer,
        "certificate_fingerprint": fingerprint,
        "check": check_name,
        "status": "PASS" if passed else "FAIL",
        "category": None if passed else category,
        "expected": expected,
        "actual": actual,
        "reason": reason,
    }


def validate_snapshot(snapshot: dict[str, Any], revision: Revision) -> dict[str, Any]:
    records = snapshot_chain_records(snapshot)
    revision_content = revision.content()
    revision_digest = revision.digest()
    chain_fingerprints = [record.fingerprint for record in records]
    checks: list[dict[str, Any]] = []
    order = 0

    def add(*args: Any, **kwargs: Any) -> None:
        nonlocal order
        checks.append(_check(order, *args, **kwargs))
        order += 1

    root = records[-1]
    trusted_roots = sorted(revision.trusted_root_fingerprints)
    anchor_ok = root.fingerprint in set(trusted_roots)
    add(
        len(records) - 1,
        root.fingerprint,
        "anchor",
        anchor_ok,
        "UNTRUSTED_ANCHOR",
        trusted_roots,
        root.fingerprint,
        "trust_anchor_present" if anchor_ok else "trust_anchor_not_in_revision",
    )

    for index in range(len(records) - 1):
        child = records[index]
        parent = records[index + 1]
        issuer_ok = child.issuer_der == parent.subject_der
        add(
            index,
            child.fingerprint,
            "issuer_match",
            issuer_ok,
            "BAD_SIGNATURE",
            parent.subject,
            child.issuer,
            "issuer_matches_parent" if issuer_ok else "issuer_does_not_match_parent",
        )
        signature_ok = child.verify_signature(parent)
        add(
            index,
            child.fingerprint,
            "signature",
            signature_ok,
            "BAD_SIGNATURE",
            parent.fingerprint,
            child.fingerprint,
            "signature_valid" if signature_ok else "signature_invalid",
        )

    leaf_ca_ok = not records[0].is_ca
    add(
        0,
        records[0].fingerprint,
        "basic_constraints.leaf_not_ca",
        leaf_ca_ok,
        "BAD_SIGNATURE",
        False,
        records[0].is_ca,
        "leaf_is_end_entity" if leaf_ca_ok else "leaf_is_ca",
    )

    for index in range(1, len(records)):
        cert = records[index]
        ca_ok = cert.is_ca
        add(
            index,
            cert.fingerprint,
            "basic_constraints.ca",
            ca_ok,
            "BAD_SIGNATURE",
            True,
            cert.is_ca,
            "issuer_is_ca" if ca_ok else "issuer_is_not_ca",
        )
        key_cert_sign_ok = not cert.key_usage or "keyCertSign" in cert.key_usage
        add(
            index,
            cert.fingerprint,
            "basic_constraints.key_cert_sign",
            key_cert_sign_ok,
            "BAD_SIGNATURE",
            "keyCertSign when keyUsage is present",
            list(cert.key_usage),
            "key_cert_sign_allowed" if key_cert_sign_ok else "key_cert_sign_not_allowed",
        )

    root_self_ok = root.is_self_signed()
    add(
        len(records) - 1,
        root.fingerprint,
        "root_self_signature",
        root_self_ok,
        "BAD_SIGNATURE",
        True,
        root_self_ok,
        "root_self_signature_valid" if root_self_ok else "root_self_signature_invalid",
    )

    override_map = revision.path_len_map()
    for index in range(len(records) - 1, 0, -1):
        cert = records[index]
        subordinate_cas = sum(1 for item in records[1:index] if item.is_ca)
        if cert.fingerprint in override_map:
            effective = override_map[cert.fingerprint]
            source = "revision_override"
        else:
            effective = cert.path_len
            source = "certificate"
        path_ok = effective is None or subordinate_cas <= effective
        add(
            index,
            cert.fingerprint,
            "path_length",
            path_ok,
            "PATH_LEN",
            {
                "max_intermediate_cas": effective,
                "source": source,
            },
            {"intermediate_cas_below": subordinate_cas},
            "path_length_satisfied" if path_ok else "path_length_exceeded",
        )

    evaluation_ms = revision.validity_evaluation_epoch_ms
    for index, cert in enumerate(records):
        not_before_ok = evaluation_ms >= cert.not_before_ms
        add(
            index,
            cert.fingerprint,
            "validity.not_before",
            not_before_ok,
            "EXPIRED",
            cert.not_before_ms,
            evaluation_ms,
            "certificate_is_current" if not_before_ok else "certificate_not_yet_valid",
        )
        not_after_ok = evaluation_ms <= cert.not_after_ms
        add(
            index,
            cert.fingerprint,
            "validity.not_after",
            not_after_ok,
            "EXPIRED",
            cert.not_after_ms,
            evaluation_ms,
            "certificate_not_expired" if not_after_ok else "certificate_expired",
        )

    revoked = revision.revoked_set()
    for index, cert in enumerate(records):
        not_revoked = cert.serial_number not in revoked
        add(
            index,
            cert.fingerprint,
            "revocation",
            not_revoked,
            "REVOKED",
            "serial_not_revoked",
            cert.serial_number,
            "serial_not_revoked" if not_revoked else "serial_revoked",
        )

    first_failure = next((item for item in checks if item["status"] == "FAIL"), None)
    failure_category = first_failure["category"] if first_failure else None
    conclusion = "PASS" if first_failure is None else "FAIL"
    index_value = result_index(chain_fingerprints, revision_digest, failure_category)
    result: dict[str, Any] = {
        "format_version": 1,
        "validator_version": VALIDATOR_VERSION,
        "snapshot_digest": snapshot["snapshot_digest"],
        "chain_fingerprints": chain_fingerprints,
        "revision": revision.revision,
        "revision_content": revision_content,
        "revision_digest": revision_digest,
        "conclusion": conclusion,
        "failure_category": failure_category,
        "first_failure": (
            {
                "layer": first_failure["layer"],
                "check": first_failure["check"],
                "reason": first_failure["reason"],
                "category": first_failure["category"],
            }
            if first_failure
            else None
        ),
        "checks": checks,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for item in checks if item["status"] == "PASS"),
        },
        "index": index_value,
    }
    result["result_digest"] = digest_obj(result)
    return result


def validate_many(snapshot: dict[str, Any], document: TrustDocument) -> list[dict[str, Any]]:
    results = [validate_snapshot(snapshot, revision) for revision in document.revisions]
    return sorted(results, key=lambda item: item["revision"])


def compute_flips(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(results, key=lambda item: item["revision"])
    flips: list[dict[str, Any]] = []
    for before, after in zip(ordered, ordered[1:]):
        if before["conclusion"] != after["conclusion"]:
            kind = "CONCLUSION"
        elif before["failure_category"] != after["failure_category"]:
            kind = "CATEGORY"
        else:
            continue
        flips.append(
            {
                "from_revision": before["revision"],
                "to_revision": after["revision"],
                "kind": kind,
                "from_conclusion": before["conclusion"],
                "to_conclusion": after["conclusion"],
                "from_failure_category": before["failure_category"],
                "to_failure_category": after["failure_category"],
                "from_first_failure_layer": (before.get("first_failure") or {}).get("layer"),
                "to_first_failure_layer": (after.get("first_failure") or {}).get("layer"),
                "reason": (
                    f"conclusion {before['conclusion']} -> {after['conclusion']}"
                    if kind == "CONCLUSION"
                    else f"failure category {before['failure_category']} -> {after['failure_category']}"
                ),
            }
        )
    return sorted(flips, key=lambda item: (item["from_revision"], item["to_revision"], item["kind"]))


def _top_level_layer(expected: dict[str, Any], actual: dict[str, Any]) -> int | None:
    for result in (expected, actual):
        failure = result.get("first_failure")
        if isinstance(failure, dict) and isinstance(failure.get("layer"), int):
            return failure["layer"]
    return None


def compare_results(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    if canonical_json(expected) == canonical_json(actual):
        return {
            "equal": True,
            "expected_index": expected.get("index"),
            "actual_index": actual.get("index"),
        }
    for key in ("conclusion", "failure_category", "revision_digest", "snapshot_digest"):
        if expected.get(key) != actual.get(key):
            return {
                "equal": False,
                "layer": _top_level_layer(expected, actual),
                "item": key,
                "field": key,
                "expected": expected.get(key),
                "actual": actual.get(key),
            }
    expected_failure = expected.get("first_failure") or {}
    actual_failure = actual.get("first_failure") or {}
    if expected_failure != actual_failure:
        layer = expected_failure.get("layer", actual_failure.get("layer"))
        for key in ("layer", "check", "reason", "category"):
            if expected_failure.get(key) != actual_failure.get(key):
                return {
                    "equal": False,
                    "layer": layer,
                    "item": "first_failure",
                    "field": f"first_failure.{key}",
                    "expected": expected_failure.get(key),
                    "actual": actual_failure.get(key),
                }
    expected_chain = expected.get("chain_fingerprints") or []
    actual_chain = actual.get("chain_fingerprints") or []
    if expected_chain != actual_chain:
        for index in range(max(len(expected_chain), len(actual_chain))):
            left = expected_chain[index] if index < len(expected_chain) else None
            right = actual_chain[index] if index < len(actual_chain) else None
            if left != right:
                return {
                    "equal": False,
                    "layer": index,
                    "item": "chain_fingerprints",
                    "field": f"chain_fingerprints[{index}]",
                    "expected": left,
                    "actual": right,
                }
    if expected.get("index") != actual.get("index"):
        return {
            "equal": False,
            "layer": _top_level_layer(expected, actual),
            "item": "result_index",
            "field": "index",
            "expected": expected.get("index"),
            "actual": actual.get("index"),
        }
    expected_checks = expected.get("checks") or []
    actual_checks = actual.get("checks") or []
    for index in range(max(len(expected_checks), len(actual_checks))):
        if index >= len(expected_checks) or index >= len(actual_checks):
            base = expected_checks[index] if index < len(expected_checks) else actual_checks[index]
            return {
                "equal": False,
                "layer": base.get("layer"),
                "item": base.get("check"),
                "field": f"checks[{index}]",
                "expected": expected_checks[index] if index < len(expected_checks) else None,
                "actual": actual_checks[index] if index < len(actual_checks) else None,
            }
        left = expected_checks[index]
        right = actual_checks[index]
        if left == right:
            continue
        for key in ("layer", "check", "status", "category", "expected", "actual", "reason"):
            if left.get(key) != right.get(key):
                return {
                    "equal": False,
                    "layer": left.get("layer"),
                    "item": left.get("check"),
                    "field": f"checks[{index}].{key}",
                    "expected": left.get(key),
                    "actual": right.get(key),
                }
    return {
        "equal": False,
        "layer": _top_level_layer(expected, actual),
        "item": "result",
        "field": "result_digest",
        "expected": expected.get("result_digest"),
        "actual": actual.get("result_digest"),
    }
