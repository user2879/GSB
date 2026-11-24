from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_SIGNATURE_ALGORITHMS: dict[str, int] = {
    "TOY_RSA_E3_SHA256": 3,
    "TOY_RSA_E5_SHA256": 5,
    "TOY_RSA_E17_SHA256": 17,
    "TOY_RSA_E65537_SHA256": 65537,
}

DECISIONS = ("PASS", "FAIL", "INDETERMINATE")
DIRECTIONS = ("client", "server")


@dataclass(frozen=True)
class Certificate:
    cert_id: str
    issuer: str
    serial: int
    not_before_ms: int
    not_after_ms: int
    modulus: int
    signature: int
    algorithm: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.cert_id,
            "issuer": self.issuer,
            "serial": self.serial,
            "not_before_ms": self.not_before_ms,
            "not_after_ms": self.not_after_ms,
            "modulus": self.modulus,
            "signature": self.signature,
            "algorithm": self.algorithm,
        }

    def signed_payload(self) -> dict[str, Any]:
        return {
            "id": self.cert_id,
            "issuer": self.issuer,
            "serial": self.serial,
            "not_before_ms": self.not_before_ms,
            "not_after_ms": self.not_after_ms,
            "modulus": self.modulus,
            "algorithm": self.algorithm,
        }


@dataclass(frozen=True)
class FixedParams:
    scale: int
    max_bits: int
    time_quantum_ms: int
    time_epoch_ms: int
    max_path_length: int
    path_weight: int
    overflow_policy: str
    division_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "max_bits": self.max_bits,
            "time_quantum_ms": self.time_quantum_ms,
            "time_epoch_ms": self.time_epoch_ms,
            "max_path_length": self.max_path_length,
            "path_weight": self.path_weight,
            "overflow_policy": self.overflow_policy,
            "division_policy": self.division_policy,
        }


@dataclass(frozen=True)
class ScriptMessage:
    seq: int
    sender: str
    client_seq: int
    server_seq: int
    arrival_ms: int
    message_type: str
    cert_ids: tuple[str, ...]
    expected: str
    alternative_cert_ids: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        value = {
            "seq": self.seq,
            "sender": self.sender,
            "client_seq": self.client_seq,
            "server_seq": self.server_seq,
            "arrival_ms": self.arrival_ms,
            "type": self.message_type,
            "cert_ids": list(self.cert_ids),
            "expected": self.expected,
        }
        if self.alternative_cert_ids is not None:
            value["alternative_cert_ids"] = list(self.alternative_cert_ids)
        return value

    def wire_dict(self) -> dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True)
class Spec:
    format_version: int
    certificates: tuple[Certificate, ...]
    trust_anchors: tuple[str, ...]
    fixed: FixedParams
    socket_timeout_ms: int
    messages: tuple[ScriptMessage, ...]
    cert_source_digest: str
    handshake_source_digest: str

    @property
    def cert_map(self) -> dict[str, Certificate]:
        return {cert.cert_id: cert for cert in self.certificates}

    @property
    def message_map(self) -> dict[int, ScriptMessage]:
        return {message.seq: message for message in self.messages}

    def spec_digest_input(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "certificates": [cert.as_dict() for cert in self.certificates],
            "trust_anchors": list(self.trust_anchors),
            "fixed": self.fixed.as_dict(),
            "socket_timeout_ms": self.socket_timeout_ms,
            "messages": [message.as_dict() for message in self.messages],
        }

    def message_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for message in self.messages:
            if message.sender == "client":
                current = {"request": message, "replies": []}
                groups.append(current)
            else:
                if current is None:
                    raise ValueError("server message appears before a client message")
                current["replies"].append(message)
        return groups
