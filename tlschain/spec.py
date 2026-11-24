from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import digest_hex
from .errors import DomainError, DuplicateJsonKey
from .models import (
    DECISIONS,
    DIRECTIONS,
    SUPPORTED_SIGNATURE_ALGORITHMS,
    Certificate,
    FixedParams,
    ScriptMessage,
    Spec,
)


CERTIFICATE_FIELDS = {
    "id",
    "issuer",
    "serial",
    "not_before_ms",
    "not_after_ms",
    "modulus",
    "signature",
    "algorithm",
}
PROGRAMME_FIELDS = {"format_version", "trust_anchors", "fixed", "certificates"}
FIXED_FIELDS = {
    "scale",
    "max_bits",
    "time_quantum_ms",
    "time_epoch_ms",
    "max_path_length",
    "path_weight",
    "overflow_policy",
    "division_policy",
}
HANDSHAKE_FIELDS = {"format_version", "socket_timeout_ms", "messages"}
MESSAGE_FIELDS = {
    "seq",
    "sender",
    "client_seq",
    "server_seq",
    "arrival_ms",
    "type",
    "cert_ids",
    "expected",
    "alternative_cert_ids",
}


class _DuplicateDetector(dict):
    pass


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(key)
        result[key] = value
    return result


def _read_json(path: str | Path, label: str) -> tuple[Any, str]:
    json_path = Path(path)
    try:
        raw = json_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DomainError("E_SPEC", f"cannot read {label} file: {exc}", cert_id=None, field_path="$") from exc
    try:
        value = json.loads(raw, object_pairs_hook=_object_pairs_hook)
    except DuplicateJsonKey as exc:
        raise DomainError(
            "E_SPEC",
            f"duplicate JSON key in {label}: {exc.key}",
            cert_id=None,
            field_path=f"$.{exc.key}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise DomainError(
            "E_SPEC",
            f"corrupted JSON in {label}: {exc.msg}",
            cert_id=None,
            field_path=f"$[line={exc.lineno},column={exc.colno}]",
        ) from exc
    return value, raw


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, field_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DomainError("E_SPEC", f"{field_path} must be an object", cert_id=None, field_path=field_path)
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], field_path: str, cert_id: str | None) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if extra:
            parts.append("extra=" + ",".join(extra))
        raise DomainError(
            "E_SPEC",
            f"invalid fields at {field_path}: {'; '.join(parts)}",
            cert_id=cert_id,
            field_path=field_path,
        )


def _require_int(value: Any, field_path: str, cert_id: str | None) -> int:
    if not _is_int(value):
        raise DomainError("E_SPEC", f"{field_path} must be an integer", cert_id=cert_id, field_path=field_path)
    return value


def _require_positive_int(value: Any, field_path: str, cert_id: str | None) -> int:
    result = _require_int(value, field_path, cert_id)
    if result <= 0:
        raise DomainError("E_SPEC", f"{field_path} must be a positive integer", cert_id=cert_id, field_path=field_path)
    return result


def _require_string(value: Any, field_path: str, cert_id: str | None) -> str:
    if not isinstance(value, str) or not value:
        raise DomainError("E_SPEC", f"{field_path} must be a non-empty string", cert_id=cert_id, field_path=field_path)
    return value


def _parse_certificate(raw: Any, index: int) -> Certificate:
    path = f"$.certificates[{index}]"
    item = _require_object(raw, path)
    cert_id_value = item.get("id")
    cert_id = cert_id_value if isinstance(cert_id_value, str) and cert_id_value else None
    _require_exact_keys(item, CERTIFICATE_FIELDS, path, cert_id)
    cert_id = _require_string(item["id"], f"{path}.id", cert_id)
    issuer = _require_string(item["issuer"], f"{path}.issuer", cert_id)
    serial = _require_positive_int(item["serial"], f"{path}.serial", cert_id)
    not_before = _require_int(item["not_before_ms"], f"{path}.not_before_ms", cert_id)
    not_after = _require_int(item["not_after_ms"], f"{path}.not_after_ms", cert_id)
    if not_before >= not_after:
        raise DomainError(
            "E_SPEC",
            "certificate validity window must start before it ends",
            cert_id=cert_id,
            field_path=f"{path}.not_before_ms",
        )
    modulus = _require_positive_int(item["modulus"], f"{path}.modulus", cert_id)
    signature = _require_int(item["signature"], f"{path}.signature", cert_id)
    if signature < 0:
        raise DomainError("E_SPEC", "signature must be non-negative", cert_id=cert_id, field_path=f"{path}.signature")
    algorithm = _require_string(item["algorithm"], f"{path}.algorithm", cert_id)
    if algorithm not in SUPPORTED_SIGNATURE_ALGORITHMS:
        raise DomainError(
            "E_SPEC",
            f"unsupported signature algorithm: {algorithm}",
            cert_id=cert_id,
            field_path=f"{path}.algorithm",
        )
    return Certificate(cert_id, issuer, serial, not_before, not_after, modulus, signature, algorithm)


def _parse_fixed(raw: Any) -> FixedParams:
    path = "$.fixed"
    item = _require_object(raw, path)
    _require_exact_keys(item, FIXED_FIELDS, path, None)
    scale = _require_positive_int(item["scale"], f"{path}.scale", None)
    max_bits = _require_positive_int(item["max_bits"], f"{path}.max_bits", None)
    time_quantum = _require_positive_int(item["time_quantum_ms"], f"{path}.time_quantum_ms", None)
    time_epoch = _require_int(item["time_epoch_ms"], f"{path}.time_epoch_ms", None)
    max_path = _require_positive_int(item["max_path_length"], f"{path}.max_path_length", None)
    path_weight = _require_positive_int(item["path_weight"], f"{path}.path_weight", None)
    overflow = _require_string(item["overflow_policy"], f"{path}.overflow_policy", None)
    division = _require_string(item["division_policy"], f"{path}.division_policy", None)
    if overflow != "saturate":
        raise DomainError("E_SPEC", "overflow_policy must be 'saturate'", field_path=f"{path}.overflow_policy")
    if division != "floor":
        raise DomainError("E_SPEC", "division_policy must be 'floor'", field_path=f"{path}.division_policy")
    if scale >= (1 << max_bits):
        raise DomainError(
            "E_SPEC",
            "scale must fit within max_bits",
            field_path=f"{path}.scale",
        )
    return FixedParams(scale, max_bits, time_quantum, time_epoch, max_path, path_weight, overflow, division)


def _parse_message(
    raw: Any,
    index: int,
    cert_map: dict[str, Certificate],
    expected_seq: int,
    expected_client_seq: int,
    expected_server_seq: int,
) -> tuple[ScriptMessage, int, int]:
    path = f"$.messages[{index}]"
    item = _require_object(raw, path)
    first_cert = None
    if isinstance(item.get("cert_ids"), list) and item["cert_ids"]:
        candidate = item["cert_ids"][0]
        first_cert = candidate if isinstance(candidate, str) else None
    required_message_fields = MESSAGE_FIELDS - {"alternative_cert_ids"}
    missing = sorted(required_message_fields - set(item))
    unknown = sorted(set(item) - MESSAGE_FIELDS)
    if missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if unknown:
            parts.append("extra=" + ",".join(unknown))
        raise DomainError(
            "E_SPEC",
            f"invalid fields at {path}: {'; '.join(parts)}",
            cert_id=first_cert,
            field_path=path,
        )
    seq = _require_int(item["seq"], f"{path}.seq", first_cert)
    if seq != expected_seq:
        raise DomainError(
            "E_SPEC",
            f"message sequence must be continuous; expected {expected_seq}, got {seq}",
            cert_id=first_cert,
            field_path=f"{path}.seq",
            message_seq=seq,
        )
    sender = _require_string(item["sender"], f"{path}.sender", first_cert)
    if sender not in DIRECTIONS:
        raise DomainError(
            "E_SPEC",
            "sender must be 'client' or 'server'",
            cert_id=first_cert,
            field_path=f"{path}.sender",
            message_seq=seq,
        )
    client_seq = _require_int(item["client_seq"], f"{path}.client_seq", first_cert)
    server_seq = _require_int(item["server_seq"], f"{path}.server_seq", first_cert)
    if sender == "client":
        if client_seq != expected_client_seq:
            raise DomainError(
                "E_SPEC",
                f"client_seq must advance continuously; expected {expected_client_seq}",
                cert_id=first_cert,
                field_path=f"{path}.client_seq",
                message_seq=seq,
            )
        if server_seq != expected_server_seq - 1:
            raise DomainError(
                "E_SPEC",
                f"server_seq before a client message must be {expected_server_seq - 1}",
                cert_id=first_cert,
                field_path=f"{path}.server_seq",
                message_seq=seq,
            )
        expected_client_seq += 1
    else:
        if server_seq != expected_server_seq:
            raise DomainError(
                "E_SPEC",
                f"server_seq must advance continuously; expected {expected_server_seq}",
                cert_id=first_cert,
                field_path=f"{path}.server_seq",
                message_seq=seq,
            )
        if client_seq != expected_client_seq - 1:
            raise DomainError(
                "E_SPEC",
                f"client_seq before a server message must be {expected_client_seq - 1}",
                cert_id=first_cert,
                field_path=f"{path}.client_seq",
                message_seq=seq,
            )
        expected_server_seq += 1
    arrival = _require_int(item["arrival_ms"], f"{path}.arrival_ms", first_cert)
    message_type = _require_string(item["type"], f"{path}.type", first_cert)
    expected = _require_string(item["expected"], f"{path}.expected", first_cert)
    if expected not in DECISIONS:
        raise DomainError(
            "E_SPEC",
            "expected must be PASS, FAIL, or INDETERMINATE",
            cert_id=first_cert,
            field_path=f"{path}.expected",
            message_seq=seq,
        )
    cert_ids_raw = item["cert_ids"]
    if not isinstance(cert_ids_raw, list) or not cert_ids_raw:
        raise DomainError(
            "E_SPEC",
            "cert_ids must be a non-empty array",
            cert_id=first_cert,
            field_path=f"{path}.cert_ids",
            message_seq=seq,
        )
    cert_ids: list[str] = []
    for cert_index, cert_value in enumerate(cert_ids_raw):
        value = _require_string(cert_value, f"{path}.cert_ids[{cert_index}]", first_cert)
        if value not in cert_map:
            raise DomainError(
                "E_SPEC",
                f"unknown certificate reference: {value}",
                cert_id=value,
                field_path=f"{path}.cert_ids[{cert_index}]",
                message_seq=seq,
            )
        if value in cert_ids:
            raise DomainError(
                "E_SPEC",
                f"duplicate certificate in chain: {value}",
                cert_id=value,
                field_path=f"{path}.cert_ids[{cert_index}]",
                message_seq=seq,
            )
        cert_ids.append(value)
    alternative: tuple[str, ...] | None = None
    if "alternative_cert_ids" in item:
        alt_raw = item["alternative_cert_ids"]
        if not isinstance(alt_raw, list) or not alt_raw:
            raise DomainError(
                "E_SPEC",
                "alternative_cert_ids must be a non-empty array",
                cert_id=first_cert,
                field_path=f"{path}.alternative_cert_ids",
                message_seq=seq,
            )
        alt: list[str] = []
        for cert_index, cert_value in enumerate(alt_raw):
            value = _require_string(cert_value, f"{path}.alternative_cert_ids[{cert_index}]", first_cert)
            if value not in cert_map:
                raise DomainError(
                    "E_SPEC",
                    f"unknown certificate reference: {value}",
                    cert_id=value,
                    field_path=f"{path}.alternative_cert_ids[{cert_index}]",
                    message_seq=seq,
                )
            if value in alt:
                raise DomainError(
                    "E_SPEC",
                    f"duplicate certificate in alternative chain: {value}",
                    cert_id=value,
                    field_path=f"{path}.alternative_cert_ids[{cert_index}]",
                    message_seq=seq,
                )
            alt.append(value)
        alternative = tuple(alt)
    return (
        ScriptMessage(
            seq,
            sender,
            client_seq,
            server_seq,
            arrival,
            message_type,
            tuple(cert_ids),
            expected,
            alternative,
        ),
        expected_client_seq,
        expected_server_seq,
    )


def _detect_cycle(certificates: list[Certificate], trust_anchors: set[str]) -> None:
    by_id = {cert.cert_id: cert for cert in certificates}
    state: dict[str, int] = {cert.cert_id: 0 for cert in certificates}
    stack: list[str] = []

    def visit(cert_id: str) -> None:
        if state[cert_id] == 1:
            start = stack.index(cert_id)
            cycle = stack[start:] + [cert_id]
            raise DomainError(
                "E_SPEC",
                f"issuer cycle detected: {' -> '.join(cycle)}",
                cert_id=cert_id,
                field_path="$.certificates.issuer",
                details={"cycle": cycle},
            )
        if state[cert_id] == 2:
            return
        state[cert_id] = 1
        stack.append(cert_id)
        issuer = by_id[cert_id].issuer
        if issuer in by_id and issuer != cert_id:
            visit(issuer)
        elif issuer == cert_id and cert_id not in trust_anchors:
            raise DomainError(
                "E_SPEC",
                f"self-issued certificate is not a declared trust anchor: {cert_id}",
                cert_id=cert_id,
                field_path="$.certificates.issuer",
                details={"cycle": [cert_id, cert_id]},
            )
        stack.pop()
        state[cert_id] = 2

    for cert in certificates:
        visit(cert.cert_id)


def _build_spec(cert_doc: Any, handshake_doc: Any) -> Spec:
    cert_root = _require_object(cert_doc, "$")
    _require_exact_keys(cert_root, PROGRAMME_FIELDS, "$", None)
    format_version = _require_int(cert_root["format_version"], "$.format_version", None)
    if format_version != 1:
        raise DomainError("E_SPEC", "format_version must be 1", field_path="$.format_version")
    raw_certificates = cert_root["certificates"]
    if not isinstance(raw_certificates, list) or not raw_certificates:
        raise DomainError("E_SPEC", "certificates must be a non-empty array", field_path="$.certificates")
    certificates = [_parse_certificate(raw, index) for index, raw in enumerate(raw_certificates)]
    seen: set[str] = set()
    for cert in certificates:
        if cert.cert_id in seen:
            raise DomainError(
                "E_SPEC",
                f"duplicate certificate id: {cert.cert_id}",
                cert_id=cert.cert_id,
                field_path="$.certificates.id",
            )
        seen.add(cert.cert_id)
    cert_map = {cert.cert_id: cert for cert in certificates}
    for cert in certificates:
        if cert.issuer not in cert_map:
            raise DomainError(
                "E_SPEC",
                f"issuer reference does not exist: {cert.issuer}",
                cert_id=cert.cert_id,
                field_path="$.certificates.issuer",
            )
    anchors_raw = cert_root["trust_anchors"]
    if not isinstance(anchors_raw, list) or not anchors_raw:
        raise DomainError("E_SPEC", "trust_anchors must be a non-empty array", field_path="$.trust_anchors")
    declared_anchors = {
        value
        for value in anchors_raw
        if isinstance(value, str) and value
    }
    _detect_cycle(certificates, declared_anchors)
    anchors: list[str] = []
    for index, anchor_value in enumerate(anchors_raw):
        anchor = _require_string(anchor_value, f"$.trust_anchors[{index}]", None)
        if anchor not in cert_map:
            raise DomainError(
                "E_SPEC",
                f"trust anchor does not exist: {anchor}",
                cert_id=anchor,
                field_path=f"$.trust_anchors[{index}]",
            )
        if anchor in anchors:
            raise DomainError(
                "E_SPEC",
                f"duplicate trust anchor: {anchor}",
                cert_id=anchor,
                field_path=f"$.trust_anchors[{index}]",
            )
        anchors.append(anchor)
    fixed = _parse_fixed(cert_root["fixed"])

    handshake_root = _require_object(handshake_doc, "$")
    _require_exact_keys(handshake_root, HANDSHAKE_FIELDS, "$", None)
    handshake_version = _require_int(handshake_root["format_version"], "$.format_version", None)
    if handshake_version != 1:
        raise DomainError("E_SPEC", "handshake format_version must be 1", field_path="$.format_version")
    timeout_ms = _require_positive_int(handshake_root["socket_timeout_ms"], "$.socket_timeout_ms", None)
    messages_raw = handshake_root["messages"]
    if not isinstance(messages_raw, list) or not messages_raw:
        raise DomainError("E_SPEC", "messages must be a non-empty array", field_path="$.messages")
    messages: list[ScriptMessage] = []
    expected_seq = 1
    expected_client_seq = 1
    expected_server_seq = 1
    previous_arrival: int | None = None
    for index, raw in enumerate(messages_raw):
        message, expected_client_seq, expected_server_seq = _parse_message(
            raw,
            index,
            cert_map,
            expected_seq,
            expected_client_seq,
            expected_server_seq,
        )
        if index == 0 and message.sender != "client":
            raise DomainError(
                "E_SPEC",
                "handshake must begin with a client message",
                cert_id=message.cert_ids[0],
                field_path="$.messages[0].sender",
                message_seq=message.seq,
            )
        if previous_arrival is not None and message.arrival_ms < previous_arrival:
            raise DomainError(
                "E_SPEC",
                "arrival_ms must be monotonically non-decreasing",
                cert_id=message.cert_ids[0],
                field_path=f"$.messages[{index}].arrival_ms",
                message_seq=message.seq,
            )
        previous_arrival = message.arrival_ms
        messages.append(message)
        expected_seq += 1
    return Spec(
        format_version,
        tuple(certificates),
        tuple(anchors),
        fixed,
        timeout_ms,
        tuple(messages),
        digest_hex(cert_doc),
        digest_hex(handshake_doc),
    )


def load_spec(
    cert_path: str | Path,
    handshake_path: str | Path,
    anomaly: str = "none",
) -> Spec:
    if anomaly not in {"none", "corrupted-json", "self-cycle"}:
        raise DomainError("E_SPEC", f"unknown anomaly: {anomaly}", field_path="$")
    if anomaly == "corrupted-json":
        raw = Path(cert_path).read_text(encoding="utf-8")
        corrupted = raw.rstrip() + "\nBROKEN\n"
        try:
            cert_doc = json.loads(corrupted, object_pairs_hook=_object_pairs_hook)
        except json.JSONDecodeError as exc:
            raise DomainError(
                "E_SPEC",
                f"corrupted certificate JSON: {exc.msg}",
                cert_id=None,
                field_path=f"$[line={exc.lineno},column={exc.colno}]",
                details={"anomaly": "corrupted-json"},
            ) from exc
        handshake_doc, _ = _read_json(handshake_path, "handshake")
        return _build_spec(cert_doc, handshake_doc)
    cert_doc, _ = _read_json(cert_path, "certificate")
    handshake_doc, _ = _read_json(handshake_path, "handshake")
    if anomaly == "self-cycle":
        root = _require_object(cert_doc, "$")
        raw_certificates = root.get("certificates")
        if isinstance(raw_certificates, list):
            for item in raw_certificates:
                if isinstance(item, dict) and item.get("issuer") != item.get("id"):
                    item["issuer"] = item.get("id")
                    break
    return _build_spec(cert_doc, handshake_doc)


def chain_order_errors(spec: Spec, cert_ids: tuple[str, ...] | list[str]) -> list[dict[str, str]]:
    by_id = spec.cert_map
    errors: list[dict[str, str]] = []
    for index in range(len(cert_ids) - 1):
        current = by_id[cert_ids[index]]
        following = by_id[cert_ids[index + 1]]
        if current.issuer != following.cert_id:
            errors.append(
                {
                    "cert_id": current.cert_id,
                    "field": "issuer",
                    "message": f"expected next certificate {current.issuer}",
                }
            )
    return errors
