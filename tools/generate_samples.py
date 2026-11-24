from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
SEED = 20261125
E = 65537
ALGORITHM = "TOY_RSA_E65537_SHA256"
T0 = 1_700_000_000_000


def is_probable_prime(value: int, rng: random.Random, rounds: int = 24) -> bool:
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
    for prime in small_primes:
        if value == prime:
            return True
        if value % prime == 0:
            return False
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for _ in range(rounds):
        base = rng.randrange(2, value - 2)
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int, rng: random.Random) -> int:
    while True:
        value = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        value |= 1
        if value % E == 1:
            continue
        if is_probable_prime(value, rng):
            return value


def generate_key(rng: random.Random) -> dict[str, int]:
    p = generate_prime(128, rng)
    q = generate_prime(128, rng)
    while p == q:
        q = generate_prime(128, rng)
    modulus = p * q
    phi = (p - 1) * (q - 1)
    private = pow(E, -1, phi)
    return {"modulus": modulus, "private": private, "public_exponent": E}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def signed_payload(cert: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": cert["id"],
        "issuer": cert["issuer"],
        "serial": cert["serial"],
        "not_before_ms": cert["not_before_ms"],
        "not_after_ms": cert["not_after_ms"],
        "modulus": cert["modulus"],
        "algorithm": cert["algorithm"],
    }


def sign_certificate(cert: dict[str, Any], issuer_key: dict[str, int]) -> dict[str, Any]:
    result = copy.deepcopy(cert)
    digest = int.from_bytes(hashlib.sha256(canonical(signed_payload(result))).digest(), "big")
    result["signature"] = pow(digest % issuer_key["modulus"], issuer_key["private"], issuer_key["modulus"])
    return result


def cert_declaration(
    cert_id: str,
    issuer: str,
    serial: int,
    not_before: int,
    not_after: int,
    own_key: dict[str, int],
    issuer_key: dict[str, int],
) -> dict[str, Any]:
    cert = {
        "id": cert_id,
        "issuer": issuer,
        "serial": serial,
        "not_before_ms": not_before,
        "not_after_ms": not_after,
        "modulus": own_key["modulus"],
        "algorithm": ALGORITHM,
        "signature": 0,
    }
    return sign_certificate(cert, issuer_key)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_certificate_document() -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    rng = random.Random(SEED)
    keys = {
        "root": generate_key(rng),
        "intermediate": generate_key(rng),
        "intermediate_b1": generate_key(rng),
        "intermediate_b2": generate_key(rng),
        "leaf_server": generate_key(rng),
        "leaf_expired": generate_key(rng),
        "leaf_edge": generate_key(rng),
        "leaf_b1": generate_key(rng),
        "leaf_b2": generate_key(rng),
    }
    certificates = [
        cert_declaration("root-ca", "root-ca", 1001, T0 - 100_000, T0 + 100_000_000, keys["root"], keys["root"]),
        cert_declaration("intermediate-ca", "root-ca", 2001, T0 - 50_000, T0 + 50_000_000, keys["intermediate"], keys["root"]),
        cert_declaration("intermediate-b-1", "root-ca", 2101, T0 - 50_000, T0 + 50_000_000, keys["intermediate_b1"], keys["root"]),
        cert_declaration("intermediate-b-2", "root-ca", 2102, T0 - 50_000, T0 + 50_000_000, keys["intermediate_b2"], keys["root"]),
        cert_declaration("leaf-server", "intermediate-ca", 3001, T0 - 1_000, T0 + 1_000_000, keys["leaf_server"], keys["intermediate"]),
        cert_declaration("leaf-expired", "intermediate-ca", 3002, T0 - 100_000, T0 - 1, keys["leaf_expired"], keys["intermediate"]),
        cert_declaration("leaf-edge", "intermediate-ca", 3003, T0 + 20, T0 + 1_000_000, keys["leaf_edge"], keys["intermediate"]),
        cert_declaration("leaf-b-1", "intermediate-b-1", 3101, T0 - 1_000, T0 + 1_000_000, keys["leaf_b1"], keys["intermediate_b1"]),
        cert_declaration("leaf-b-2", "intermediate-b-2", 3102, T0 - 1_000, T0 + 1_000_000, keys["leaf_b2"], keys["intermediate_b2"]),
    ]
    document = {
        "format_version": 1,
        "trust_anchors": ["root-ca"],
        "fixed": {
            "scale": 1_000_000,
            "max_bits": 256,
            "time_quantum_ms": 1,
            "time_epoch_ms": T0,
            "max_path_length": 5,
            "path_weight": 1,
            "overflow_policy": "saturate",
            "division_policy": "floor",
        },
        "certificates": certificates,
    }
    return document, keys


def build_handshake_document() -> dict[str, Any]:
    server_chain = ["leaf-server", "intermediate-ca", "root-ca"]
    expired_chain = ["leaf-expired", "intermediate-ca", "root-ca"]
    edge_chain = ["leaf-edge", "intermediate-ca", "root-ca"]
    primary_tie = ["leaf-b-1", "intermediate-b-1", "root-ca"]
    alternative_tie = ["leaf-b-2", "intermediate-b-2", "root-ca"]
    messages = [
        {
            "seq": 1,
            "sender": "client",
            "client_seq": 1,
            "server_seq": 0,
            "arrival_ms": T0,
            "type": "CLIENT_HELLO",
            "cert_ids": server_chain,
            "expected": "PASS",
        },
        {
            "seq": 2,
            "sender": "server",
            "client_seq": 1,
            "server_seq": 1,
            "arrival_ms": T0,
            "type": "SERVER_HELLO",
            "cert_ids": server_chain,
            "expected": "PASS",
        },
        {
            "seq": 3,
            "sender": "client",
            "client_seq": 2,
            "server_seq": 1,
            "arrival_ms": T0 + 10,
            "type": "CERTIFICATE",
            "cert_ids": server_chain,
            "expected": "PASS",
        },
        {
            "seq": 4,
            "sender": "server",
            "client_seq": 2,
            "server_seq": 2,
            "arrival_ms": T0 + 10,
            "type": "CERTIFICATE",
            "cert_ids": expired_chain,
            "expected": "FAIL",
        },
        {
            "seq": 5,
            "sender": "client",
            "client_seq": 3,
            "server_seq": 2,
            "arrival_ms": T0 + 20,
            "type": "CERTIFICATE_VERIFY",
            "cert_ids": edge_chain,
            "expected": "INDETERMINATE",
        },
        {
            "seq": 6,
            "sender": "server",
            "client_seq": 3,
            "server_seq": 3,
            "arrival_ms": T0 + 20,
            "type": "SERVER_HELLO",
            "cert_ids": primary_tie,
            "alternative_cert_ids": alternative_tie,
            "expected": "INDETERMINATE",
        },
        {
            "seq": 7,
            "sender": "client",
            "client_seq": 4,
            "server_seq": 3,
            "arrival_ms": T0 + 30,
            "type": "FINISHED",
            "cert_ids": server_chain,
            "expected": "PASS",
        },
        {
            "seq": 8,
            "sender": "server",
            "client_seq": 4,
            "server_seq": 4,
            "arrival_ms": T0 + 30,
            "type": "FINISHED",
            "cert_ids": server_chain,
            "expected": "PASS",
        },
    ]
    return {
        "format_version": 1,
        "socket_timeout_ms": 750,
        "messages": messages,
    }


def build_modified_certificates(document: dict[str, Any], keys: dict[str, dict[str, int]]) -> dict[str, Any]:
    modified = copy.deepcopy(document)
    for cert in modified["certificates"]:
        if cert["id"] == "leaf-expired":
            cert["not_before_ms"] = T0 - 100_000
            cert["not_after_ms"] = T0 + 500_000
            updated = sign_certificate(cert, keys["intermediate"])
            cert.clear()
            cert.update(updated)
            break
    return modified


def main() -> int:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    certificates, keys = build_certificate_document()
    handshake = build_handshake_document()
    write_json(SAMPLES / "certificates.json", certificates)
    write_json(SAMPLES / "handshake.json", handshake)
    write_json(SAMPLES / "certificates_modified.json", build_modified_certificates(certificates, keys))
    self_cycle = copy.deepcopy(certificates)
    for cert in self_cycle["certificates"]:
        if cert["id"] == "intermediate-ca":
            cert["issuer"] = "intermediate-ca"
            break
    write_json(SAMPLES / "certificates_self_cycle.json", self_cycle)
    bad_algorithm = copy.deepcopy(certificates)
    for cert in bad_algorithm["certificates"]:
        if cert["id"] == "leaf-server":
            cert["algorithm"] = "UNSUPPORTED"
            break
    write_json(SAMPLES / "certificates_bad_algorithm.json", bad_algorithm)
    raw = (SAMPLES / "certificates.json").read_text(encoding="utf-8")
    (SAMPLES / "corrupted.json").write_text(raw.rstrip() + "\nBROKEN\n", encoding="utf-8")
    bad_sequence = copy.deepcopy(handshake)
    bad_sequence["messages"][1]["seq"] = 4
    write_json(SAMPLES / "handshake_bad_seq.json", bad_sequence)
    print(json.dumps({"samples": str(SAMPLES), "certificates": len(certificates["certificates"]), "messages": len(handshake["messages"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
