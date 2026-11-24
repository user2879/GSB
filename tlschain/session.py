from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_text, digest_hex, first_difference
from .fixed import build_certificate_base, verify_message
from .models import Certificate, Spec


CERT_COMPARE_FIELDS = (
    "issuer",
    "serial",
    "not_before_ms",
    "not_after_ms",
    "modulus",
    "signature",
    "algorithm",
)


@dataclass
class SessionResult:
    spec: Spec
    cert_bases: dict[str, dict[str, Any]]
    message_records: dict[int, dict[str, Any]]
    cert_invalidation: list[dict[str, Any]]
    message_invalidation: list[dict[str, Any]]
    invalidated_cert_ids: list[str]
    invalidated_message_seqs: list[int]
    mode: str

    def decision_counts(self) -> dict[str, int]:
        counts = {"PASS": 0, "FAIL": 0, "INDETERMINATE": 0}
        for record in self.message_records.values():
            counts[record["decision"]] += 1
        return counts

    def affected_count(self) -> int:
        return sum(1 for record in self.message_records.values() if record["affected_by_error"])

    def expected_mismatch_seqs(self) -> list[int]:
        return sorted(seq for seq, record in self.message_records.items() if not record["expected_match"])

    def certificate_hashes(self) -> dict[str, str]:
        return {cert_id: digest_hex(value) for cert_id, value in sorted(self.cert_bases.items())}

    def message_hashes(self) -> dict[str, str]:
        return {str(seq): digest_hex(value) for seq, value in sorted(self.message_records.items())}

    def verification_snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "certificates": self.cert_bases,
            "messages": {str(seq): record for seq, record in sorted(self.message_records.items())},
            "decision_counts": self.decision_counts(),
            "affected_by_error_count": self.affected_count(),
            "expected_mismatch_seqs": self.expected_mismatch_seqs(),
        }


def build_full_session(spec: Spec) -> SessionResult:
    cert_bases = {
        cert.cert_id: build_certificate_base(spec, cert)
        for cert in spec.certificates
    }
    message_records = {
        message.seq: verify_message(spec, cert_bases, message)
        for message in spec.messages
    }
    cert_invalidation = [
        {
            "order": index + 1,
            "kind": "certificate",
            "id": cert.cert_id,
            "reason": "full_build",
            "field": "all",
        }
        for index, cert in enumerate(spec.certificates)
    ]
    message_invalidation = [
        {
            "order": len(spec.certificates) + index + 1,
            "kind": "message",
            "id": f"message:{message.seq}",
            "seq": message.seq,
            "reason": "full_build",
            "field": "all",
        }
        for index, message in enumerate(spec.messages)
    ]
    return SessionResult(
        spec=spec,
        cert_bases=cert_bases,
        message_records=message_records,
        cert_invalidation=cert_invalidation,
        message_invalidation=message_invalidation,
        invalidated_cert_ids=[cert.cert_id for cert in spec.certificates],
        invalidated_message_seqs=[message.seq for message in spec.messages],
        mode="full",
    )


def _changed_certificate_fields(old: Certificate, new: Certificate) -> list[str]:
    changed: list[str] = []
    for field in CERT_COMPARE_FIELDS:
        if getattr(old, field) != getattr(new, field):
            changed.append(field)
    return changed


def _issuer_children(spec: Spec) -> dict[str, set[str]]:
    children: dict[str, set[str]] = {cert.cert_id: set() for cert in spec.certificates}
    for cert in spec.certificates:
        if cert.issuer in children and cert.issuer != cert.cert_id:
            children[cert.issuer].add(cert.cert_id)
    return children


def _stable_topological_cert_order(spec: Spec, affected: set[str]) -> list[str]:
    children = _issuer_children(spec)
    indegree = {cert_id: 0 for cert_id in affected}
    for cert_id in affected:
        for child in children.get(cert_id, set()):
            if child in affected:
                indegree[child] += 1
    ready = [cert_id for cert_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        cert_id = heapq.heappop(ready)
        order.append(cert_id)
        for child in sorted(children.get(cert_id, set())):
            if child not in indegree:
                continue
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(affected):
        order.extend(sorted(affected - set(order)))
    return order


def _affected_certificates(
    old: Spec,
    new: Spec,
    direct: dict[str, list[str]],
    fixed_changed: bool,
) -> tuple[set[str], dict[str, str]]:
    affected: set[str] = set()
    reasons: dict[str, str] = {}
    if fixed_changed:
        affected.update(cert.cert_id for cert in new.certificates)
        for cert_id in affected:
            reasons[cert_id] = "fixed_parameters_changed"
    for cert_id, fields in direct.items():
        affected.add(cert_id)
        reasons[cert_id] = "direct_change:" + ",".join(fields)
    old_children = _issuer_children(old)
    new_children = _issuer_children(new)
    queue = list(affected)
    while queue:
        parent = queue.pop(0)
        for children in (old_children, new_children):
            for child in sorted(children.get(parent, set())):
                if child not in affected:
                    affected.add(child)
                    reasons[child] = f"issuer_descendant:{parent}"
                    queue.append(child)
    return affected, reasons


def _message_references(message: Any) -> set[str]:
    refs = set(message.cert_ids)
    if message.alternative_cert_ids is not None:
        refs.update(message.alternative_cert_ids)
    return refs


def _message_changed(old_message: Any, new_message: Any) -> bool:
    return canonical_text(old_message.as_dict()) != canonical_text(new_message.as_dict())


def recompute_session(previous: SessionResult, new_spec: Spec) -> SessionResult:
    old_spec = previous.spec
    old_certs = old_spec.cert_map
    new_certs = new_spec.cert_map
    old_ids = set(old_certs)
    new_ids = set(new_certs)
    direct: dict[str, list[str]] = {}
    for cert_id in sorted(old_ids & new_ids):
        fields = _changed_certificate_fields(old_certs[cert_id], new_certs[cert_id])
        if fields:
            direct[cert_id] = fields
    for cert_id in sorted(new_ids - old_ids):
        direct[cert_id] = ["added"]
    for cert_id in sorted(old_ids - new_ids):
        direct[cert_id] = ["removed"]
    fixed_changed = old_spec.fixed != new_spec.fixed
    trust_anchors_changed = old_spec.trust_anchors != new_spec.trust_anchors
    affected_certs, cert_reasons = _affected_certificates(old_spec, new_spec, direct, fixed_changed)
    cert_order = _stable_topological_cert_order(new_spec, affected_certs)

    cert_bases: dict[str, dict[str, Any]] = {}
    for cert in new_spec.certificates:
        if cert.cert_id in affected_certs or cert.cert_id not in previous.cert_bases:
            cert_bases[cert.cert_id] = build_certificate_base(new_spec, cert)
        else:
            cert_bases[cert.cert_id] = previous.cert_bases[cert.cert_id]

    message_affected: set[int] = set()
    message_reasons: dict[int, str] = {}
    old_messages = old_spec.message_map
    for message in new_spec.messages:
        refs = _message_references(message) & affected_certs
        if fixed_changed:
            message_affected.add(message.seq)
            message_reasons[message.seq] = "fixed_parameters_changed"
        elif trust_anchors_changed:
            message_affected.add(message.seq)
            message_reasons[message.seq] = "trust_anchors_changed"
        elif message.seq in old_messages and _message_changed(old_messages[message.seq], message):
            message_affected.add(message.seq)
            message_reasons[message.seq] = "direct_message_change"
        elif refs:
            message_affected.add(message.seq)
            message_reasons[message.seq] = "carries_affected_certificate:" + ",".join(sorted(refs))
    previous_affected = False
    for message in new_spec.messages:
        if previous_affected and message.seq not in message_affected:
            message_affected.add(message.seq)
            message_reasons[message.seq] = f"follows_affected_message:{message.seq - 1}"
        previous_affected = message.seq in message_affected

    message_records: dict[int, dict[str, Any]] = {}
    for message in new_spec.messages:
        if message.seq in message_affected or message.seq not in previous.message_records:
            message_records[message.seq] = verify_message(new_spec, cert_bases, message)
        else:
            message_records[message.seq] = previous.message_records[message.seq]

    cert_invalidation: list[dict[str, Any]] = []
    order_counter = 1
    for cert_id in cert_order:
        if cert_id not in new_certs:
            continue
        cert_invalidation.append(
            {
                "order": order_counter,
                "kind": "certificate",
                "id": cert_id,
                "reason": cert_reasons.get(cert_id, "dependent"),
                "field": "all",
            }
        )
        order_counter += 1
    message_invalidation: list[dict[str, Any]] = []
    for message in new_spec.messages:
        if message.seq not in message_affected:
            continue
        message_invalidation.append(
            {
                "order": order_counter,
                "kind": "message",
                "id": f"message:{message.seq}",
                "seq": message.seq,
                "reason": message_reasons.get(message.seq, "dependent"),
                "field": "all",
            }
        )
        order_counter += 1
    return SessionResult(
        spec=new_spec,
        cert_bases=cert_bases,
        message_records=message_records,
        cert_invalidation=cert_invalidation,
        message_invalidation=message_invalidation,
        invalidated_cert_ids=[entry["id"] for entry in cert_invalidation],
        invalidated_message_seqs=[entry["seq"] for entry in message_invalidation],
        mode="incremental",
    )


def compare_incremental_with_full(incremental: SessionResult, full: SessionResult) -> dict[str, Any]:
    cert_differences: list[dict[str, Any]] = []
    for cert_id in sorted(set(incremental.cert_bases) | set(full.cert_bases)):
        left = incremental.cert_bases.get(cert_id)
        right = full.cert_bases.get(cert_id)
        difference = first_difference(left, right, f"$.certificates.{cert_id}")
        if difference is not None:
            difference["cert_id"] = cert_id
            cert_differences.append(difference)
    message_differences: list[dict[str, Any]] = []
    for seq in sorted(set(incremental.message_records) | set(full.message_records)):
        left = incremental.message_records.get(seq)
        right = full.message_records.get(seq)
        difference = first_difference(left, right, f"$.messages.{seq}")
        if difference is not None:
            difference["seq"] = seq
            message_differences.append(difference)
    return {
        "equal": not cert_differences and not message_differences,
        "certificate_differences": cert_differences,
        "message_differences": message_differences,
        "incremental_certificate_hashes": incremental.certificate_hashes(),
        "full_certificate_hashes": full.certificate_hashes(),
        "incremental_message_hashes": incremental.message_hashes(),
        "full_message_hashes": full.message_hashes(),
    }


def ensure_retained_bytes(previous: SessionResult, incremental: SessionResult, affected_messages: Iterable[int]) -> dict[str, Any]:
    affected = set(affected_messages)
    unchanged_certificates = sorted(set(previous.cert_bases) - set(incremental.invalidated_cert_ids))
    unchanged_messages = sorted(set(previous.message_records) - affected)
    cert_equal = all(
        canonical_text(previous.cert_bases[cert_id]) == canonical_text(incremental.cert_bases[cert_id])
        for cert_id in unchanged_certificates
    )
    message_equal = all(
        canonical_text(previous.message_records[seq]) == canonical_text(incremental.message_records[seq])
        for seq in unchanged_messages
    )
    return {
        "certificate_bytes_unchanged": cert_equal,
        "message_bytes_unchanged": message_equal,
        "retained_certificate_ids": unchanged_certificates,
        "retained_message_seqs": unchanged_messages,
    }
