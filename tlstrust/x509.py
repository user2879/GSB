"""Minimal X.509 v3 certificate construction and parsing."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any

from . import asn1
from .canonical import fingerprint_der, normalize_serial
from .rsa import (
    RSA_ENCRYPTION_OID,
    SHA256_WITH_RSA_OID,
    RSAPrivateKey,
    RSAPublicKey,
    sign_sha256,
    verify_sha256,
)


OID_BASIC_CONSTRAINTS = "2.5.29.19"
OID_KEY_USAGE = "2.5.29.15"
OID_EXT_KEY_USAGE = "2.5.29.37"
OID_SUBJECT_ALT_NAME = "2.5.29.17"
OID_SUBJECT_KEY_IDENTIFIER = "2.5.29.14"
OID_AUTHORITY_KEY_IDENTIFIER = "2.5.29.35"
OID_SERVER_AUTH = "1.3.6.1.5.5.7.3.1"

KEY_USAGE_NAMES = [
    "digitalSignature",
    "nonRepudiation",
    "keyEncipherment",
    "dataEncipherment",
    "keyAgreement",
    "keyCertSign",
    "cRLSign",
    "encipherOnly",
    "decipherOnly",
]


def _extension(oid_value: str, critical: bool, value: bytes) -> bytes:
    parts = [asn1.oid(oid_value)]
    if critical:
        parts.append(asn1.boolean(True))
    parts.append(asn1.octet_string(value))
    return asn1.sequence(*parts)


def _key_usage_extension(usage: list[str], critical: bool = True) -> bytes:
    bits = 0
    highest = -1
    for name in usage:
        index = KEY_USAGE_NAMES.index(name)
        bits |= 0x80 >> index
        highest = max(highest, index)
    byte_count = highest // 8 + 1
    unused = byte_count * 8 - (highest + 1)
    value = bits.to_bytes(byte_count, "big")
    return _extension(OID_KEY_USAGE, critical, asn1.bit_string(value, unused))


def _basic_constraints_extension(is_ca: bool, path_len: int | None) -> bytes:
    if not is_ca:
        value = asn1.sequence()
    else:
        parts = [asn1.boolean(True)]
        if path_len is not None:
            parts.append(asn1.integer(path_len))
        value = asn1.sequence(*parts)
    return _extension(OID_BASIC_CONSTRAINTS, True, value)


def _key_identifier(public_der: bytes) -> bytes:
    return hashlib.sha256(public_der).digest()[:20]


def _subject_key_id_extension(key_id: bytes) -> bytes:
    return _extension(OID_SUBJECT_KEY_IDENTIFIER, False, asn1.octet_string(key_id))


def _authority_key_id_extension(key_id: bytes) -> bytes:
    return _extension(
        OID_AUTHORITY_KEY_IDENTIFIER,
        False,
        asn1.sequence(asn1.tlv(0x80, key_id)),
    )


def _subject_alt_name_extension(hostname: str) -> bytes:
    return _extension(
        OID_SUBJECT_ALT_NAME,
        False,
        asn1.sequence(asn1.tlv(0x82, hostname.encode("ascii"))),
    )


def _extended_key_usage_extension() -> bytes:
    return _extension(
        OID_EXT_KEY_USAGE,
        False,
        asn1.sequence(asn1.oid(OID_SERVER_AUTH)),
    )


def build_certificate(
    *,
    subject_cn: str,
    issuer_cn: str,
    subject_key: RSAPrivateKey,
    issuer_key: RSAPrivateKey,
    serial_number: str,
    not_before_ms: int,
    not_after_ms: int,
    is_ca: bool,
    path_len: int | None = None,
    server_name: str | None = None,
    corrupt_signature: bool = False,
) -> bytes:
    """Build a SHA-256/RSA X.509 v3 certificate and return DER bytes."""

    serial_value = int(normalize_serial(serial_number), 16)
    subject_der = asn1.name(subject_cn)
    issuer_der = asn1.name(issuer_cn)
    subject_public_der = subject_key.public_der()
    issuer_key_id = _key_identifier(issuer_key.public_der())
    subject_key_id = _key_identifier(subject_public_der)
    signature_alg = asn1.algorithm_identifier(SHA256_WITH_RSA_OID)
    spki = asn1.sequence(
        asn1.algorithm_identifier(RSA_ENCRYPTION_OID),
        asn1.bit_string(subject_public_der),
    )
    validity = asn1.sequence(
        asn1.utc_time(not_before_ms),
        asn1.utc_time(not_after_ms),
    )

    extensions = [
        _basic_constraints_extension(is_ca, path_len),
        _key_usage_extension(
            ["keyCertSign", "cRLSign", "digitalSignature"]
            if is_ca
            else ["digitalSignature", "keyEncipherment"]
        ),
        _subject_key_id_extension(subject_key_id),
        _authority_key_id_extension(issuer_key_id),
    ]
    if not is_ca:
        extensions.append(_extended_key_usage_extension())
        extensions.append(_subject_alt_name_extension(server_name or "localhost"))

    tbs = asn1.sequence(
        asn1.explicit(0, asn1.integer(2)),
        asn1.integer(serial_value),
        signature_alg,
        issuer_der,
        validity,
        subject_der,
        spki,
        asn1.explicit(3, asn1.sequence(*extensions)),
    )
    signature = bytearray(sign_sha256(issuer_key, tbs))
    if corrupt_signature:
        signature[-1] ^= 0x01
    return asn1.sequence(tbs, signature_alg, asn1.bit_string(bytes(signature)))


@dataclass(frozen=True)
class CertRecord:
    der: bytes
    fingerprint: str
    serial_number: str
    subject: str
    issuer: str
    not_before_ms: int
    not_after_ms: int
    is_ca: bool
    path_len: int | None
    key_usage: tuple[str, ...]
    extended_key_usage: tuple[str, ...]
    subject_alt_names: tuple[str, ...]
    subject_key_id: str | None
    authority_key_id: str | None
    public_key: RSAPublicKey
    tbs_der: bytes
    signature: bytes
    signature_algorithm: str
    issuer_der: bytes
    subject_der: bytes

    @property
    def ca(self) -> bool:
        return self.is_ca

    def verify_signature(self, issuer: "CertRecord") -> bool:
        if self.signature_algorithm != SHA256_WITH_RSA_OID:
            return False
        return verify_sha256(issuer.public_key, self.tbs_der, self.signature)

    def is_self_signed(self) -> bool:
        return self.issuer_der == self.subject_der and self.verify_signature(self)

    def to_snapshot_dict(self) -> dict[str, Any]:
        return {
            "position": None,
            "fingerprint": self.fingerprint,
            "serial_number": self.serial_number,
            "subject": self.subject,
            "issuer": self.issuer,
            "not_before_ms": self.not_before_ms,
            "not_after_ms": self.not_after_ms,
            "basic_constraints": {
                "ca": self.is_ca,
                "path_len": self.path_len,
            },
            "key_usage": list(self.key_usage),
            "extended_key_usage": list(self.extended_key_usage),
            "subject_alt_names": list(self.subject_alt_names),
            "signature_algorithm": self.signature_algorithm,
            "der_base64": base64.b64encode(self.der).decode("ascii"),
        }


def _raw_child(parent: asn1.TLV, child: asn1.TLV) -> bytes:
    return parent.value[child.start : child.end]


def _parse_extensions(node: asn1.TLV) -> dict[str, tuple[bool, bytes]]:
    result: dict[str, tuple[bool, bytes]] = {}
    if not node.value:
        return result
    inner = asn1.parse_one(node.value)
    asn1.require_tag(inner, 0x30, "Extensions")
    for extension in asn1.children(inner):
        parts = asn1.children(extension)
        if len(parts) not in (2, 3):
            raise asn1.DERError("invalid extension")
        oid_value = asn1.decode_oid(parts[0])
        index = 1
        critical = False
        if parts[index].tag == 0x01:
            critical = parts[index].value != b"\x00"
            index += 1
        if index >= len(parts) or parts[index].tag != 0x04:
            raise asn1.DERError("extension value is not OCTET STRING")
        result[oid_value] = (critical, parts[index].value)
    return result


def _parse_basic_constraints(value: bytes) -> tuple[bool, int | None]:
    if not value:
        return False, None
    node = asn1.parse_one(value)
    asn1.require_tag(node, 0x30, "BasicConstraints")
    parts = asn1.children(node)
    is_ca = False
    path_len: int | None = None
    index = 0
    if index < len(parts) and parts[index].tag == 0x01:
        is_ca = parts[index].value != b"\x00"
        index += 1
    if index < len(parts) and parts[index].tag == 0x02:
        path_len = asn1.decode_integer(parts[index])
    return is_ca, path_len


def _parse_key_usage(value: bytes) -> tuple[str, ...]:
    node = asn1.parse_one(value)
    unused, payload = asn1.decode_bit_string(node)
    names: list[str] = []
    for byte_index, byte in enumerate(payload):
        for bit_index in range(8):
            absolute = byte_index * 8 + bit_index
            if absolute >= len(KEY_USAGE_NAMES):
                continue
            if byte & (0x80 >> bit_index):
                names.append(KEY_USAGE_NAMES[absolute])
    return tuple(names)


def _parse_oid_sequence(value: bytes) -> tuple[str, ...]:
    node = asn1.parse_one(value)
    return tuple(asn1.decode_oid(part) for part in asn1.children(node))


def _parse_alt_names(value: bytes) -> tuple[str, ...]:
    node = asn1.parse_one(value)
    names: list[str] = []
    for part in asn1.children(node):
        if part.tag == 0x82:
            names.append(part.value.decode("ascii", "replace"))
    return tuple(names)


def _parse_key_identifier(value: bytes) -> str:
    node = asn1.parse_one(value)
    if node.tag == 0x04:
        return node.value.hex()
    if node.tag == 0x30:
        for child in asn1.children(node):
            if child.tag == 0x80:
                return child.value.hex()
    return node.value.hex()


def parse_certificate(der: bytes) -> CertRecord:
    root = asn1.parse_one(der)
    asn1.require_tag(root, 0x30, "Certificate")
    outer = asn1.children(root)
    if len(outer) != 3:
        raise asn1.DERError("certificate must contain tbs, algorithm and signature")
    tbs_node = asn1.require_tag(outer[0], 0x30, "TBSCertificate")
    asn1.require_tag(outer[1], 0x30, "signatureAlgorithm")
    signature_bits = asn1.decode_bit_string(outer[2])[1]
    tbs_parts = asn1.children(tbs_node)
    index = 0
    if tbs_parts[index].tag == 0xA0:
        index += 1
    serial_node = tbs_parts[index]
    index += 1
    serial_value = asn1.decode_integer(serial_node)
    if serial_value <= 0:
        raise asn1.DERError("certificate serial number must be positive")
    serial_number = format(serial_value, "X")
    if len(serial_number) % 2:
        serial_number = "0" + serial_number
    alg_node = asn1.require_tag(tbs_parts[index], 0x30, "tbs signature")
    index += 1
    issuer_node = tbs_parts[index]
    index += 1
    validity_node = tbs_parts[index]
    index += 1
    subject_node = tbs_parts[index]
    index += 1
    spki_node = tbs_parts[index]
    index += 1
    ext_node = None
    while index < len(tbs_parts):
        if tbs_parts[index].tag == 0xA3:
            ext_node = tbs_parts[index]
        index += 1

    validity_parts = asn1.children(validity_node)
    if len(validity_parts) != 2:
        raise asn1.DERError("invalid validity")
    not_before = asn1.decode_time(validity_parts[0])
    not_after = asn1.decode_time(validity_parts[1])

    spki_parts = asn1.children(spki_node)
    if len(spki_parts) != 2:
        raise asn1.DERError("invalid SubjectPublicKeyInfo")
    public_der = asn1.decode_bit_string(spki_parts[1])[1]
    public_key = RSAPublicKey.from_der(public_der)
    signature_algorithm = asn1.decode_oid(asn1.children(alg_node)[0])

    extensions = _parse_extensions(ext_node) if ext_node else {}
    is_ca, path_len = (False, None)
    if OID_BASIC_CONSTRAINTS in extensions:
        is_ca, path_len = _parse_basic_constraints(
            extensions[OID_BASIC_CONSTRAINTS][1]
        )
    key_usage = (
        _parse_key_usage(extensions[OID_KEY_USAGE][1])
        if OID_KEY_USAGE in extensions
        else tuple()
    )
    eku = (
        _parse_oid_sequence(extensions[OID_EXT_KEY_USAGE][1])
        if OID_EXT_KEY_USAGE in extensions
        else tuple()
    )
    alt_names = (
        _parse_alt_names(extensions[OID_SUBJECT_ALT_NAME][1])
        if OID_SUBJECT_ALT_NAME in extensions
        else tuple()
    )
    ski = (
        _parse_key_identifier(extensions[OID_SUBJECT_KEY_IDENTIFIER][1])
        if OID_SUBJECT_KEY_IDENTIFIER in extensions
        else None
    )
    aki = (
        _parse_key_identifier(extensions[OID_AUTHORITY_KEY_IDENTIFIER][1])
        if OID_AUTHORITY_KEY_IDENTIFIER in extensions
        else None
    )

    return CertRecord(
        der=der,
        fingerprint=fingerprint_der(der),
        serial_number=serial_number,
        subject=asn1.decode_name_cn(subject_node),
        issuer=asn1.decode_name_cn(issuer_node),
        not_before_ms=not_before,
        not_after_ms=not_after,
        is_ca=is_ca,
        path_len=path_len,
        key_usage=key_usage,
        extended_key_usage=eku,
        subject_alt_names=alt_names,
        subject_key_id=ski,
        authority_key_id=aki,
        public_key=public_key,
        tbs_der=_raw_child(root, tbs_node),
        signature=signature_bits,
        signature_algorithm=signature_algorithm,
        issuer_der=_raw_child(tbs_node, issuer_node),
        subject_der=_raw_child(tbs_node, subject_node),
    )


def pem_to_der(pem: str) -> bytes:
    lines = [line.strip() for line in pem.splitlines()]
    body: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("-----BEGIN") and line.endswith("-----"):
            inside = True
            continue
        if line.startswith("-----END") and line.endswith("-----"):
            break
        if inside and line:
            body.append(line)
    if not body:
        raise ValueError("no PEM block found")
    return base64.b64decode("".join(body), validate=True)


def der_to_pem(der: bytes) -> str:
    body = base64.b64encode(der).decode("ascii")
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return "-----BEGIN CERTIFICATE-----\n" + "\n".join(lines) + "\n-----END CERTIFICATE-----\n"
