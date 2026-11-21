"""Deterministic RSA key generation and PKCS#1 v1.5 SHA-256 signing.

The implementation is intentionally small and dependency-free.  It is suitable
for local diagnostic certificates, not for protecting production secrets.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Iterator

from . import asn1


SHA256_WITH_RSA_OID = "1.2.840.113549.1.1.11"
SHA256_OID = "2.16.840.1.101.3.4.2.1"
RSA_ENCRYPTION_OID = "1.2.840.113549.1.1.1"

_SMALL_PRIMES = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
    149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251,
]


class DeterministicRandom:
    def __init__(self, seed: str | bytes):
        raw = seed.encode("utf-8") if isinstance(seed, str) else seed
        self._seed = hashlib.sha256(raw).digest()
        self._counter = 0
        self._buffer = bytearray()

    def read(self, count: int) -> bytes:
        while len(self._buffer) < count:
            block = hashlib.sha256(
                self._seed + self._counter.to_bytes(8, "big")
            ).digest()
            self._counter += 1
            self._buffer.extend(block)
        result = bytes(self._buffer[:count])
        del self._buffer[:count]
        return result

    def randbelow(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        size = max(1, (bound.bit_length() + 7) // 8)
        limit = 256 ** size - (256 ** size % bound)
        while True:
            value = int.from_bytes(self.read(size), "big")
            if value < limit:
                return value % bound


def is_probable_prime(value: int, rounds: int = 24) -> bool:
    if value < 2:
        return False
    for prime in _SMALL_PRIMES:
        if value == prime:
            return True
        if value % prime == 0:
            return False
    d = value - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    rng = DeterministicRandom(f"miller-rabin:{value}")
    for _ in range(rounds):
        base = 2 + rng.randbelow(value - 3)
        x = pow(base, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, value)
            if x == value - 1:
                break
        else:
            return False
    return True


def _candidate(bits: int, rng: DeterministicRandom) -> int:
    raw = bytearray(rng.read((bits + 7) // 8))
    raw[0] |= 0xC0
    raw[-1] |= 1
    candidate = int.from_bytes(raw, "big")
    mask = (1 << bits) - 1
    return candidate & mask | (1 << (bits - 1)) | 1


def _generate_prime(bits: int, exponent: int, rng: DeterministicRandom) -> int:
    candidate = _candidate(bits, rng)
    while True:
        if candidate.bit_length() == bits and candidate % exponent != 1:
            if is_probable_prime(candidate):
                return candidate
        candidate += 2
        if candidate.bit_length() > bits:
            candidate = _candidate(bits, rng)


@dataclass(frozen=True)
class RSAPrivateKey:
    n: int
    e: int
    d: int
    p: int
    q: int

    @property
    def key_size_bytes(self) -> int:
        return (self.n.bit_length() + 7) // 8

    def public_der(self) -> bytes:
        return asn1.sequence(asn1.integer(self.n), asn1.integer(self.e))

    def private_der(self) -> bytes:
        dp = self.d % (self.p - 1)
        dq = self.d % (self.q - 1)
        qinv = pow(self.q, -1, self.p)
        return asn1.sequence(
            asn1.integer(0),
            asn1.integer(self.n),
            asn1.integer(self.e),
            asn1.integer(self.d),
            asn1.integer(self.p),
            asn1.integer(self.q),
            asn1.integer(dp),
            asn1.integer(dq),
            asn1.integer(qinv),
        )

    def private_pem(self) -> str:
        body = base64.b64encode(self.private_der()).decode("ascii")
        lines = [body[i : i + 64] for i in range(0, len(body), 64)]
        return "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join(lines) + "\n-----END RSA PRIVATE KEY-----\n"


@dataclass(frozen=True)
class RSAPublicKey:
    n: int
    e: int

    @classmethod
    def from_der(cls, der: bytes) -> "RSAPublicKey":
        root = asn1.parse_one(der)
        parts = asn1.children(root)
        if len(parts) != 2:
            raise asn1.DERError("invalid RSA public key")
        return cls(asn1.decode_integer(parts[0]), asn1.decode_integer(parts[1]))


def generate_key(seed: str, bits: int = 2048, exponent: int = 65537) -> RSAPrivateKey:
    if bits < 512 or bits % 16:
        raise ValueError("RSA bits must be a multiple of 16 and at least 512")
    if bits % 2:
        raise ValueError("RSA bits must be even")
    rng = DeterministicRandom("key:" + seed)
    half = bits // 2
    while True:
        p = _generate_prime(half, exponent, rng)
        q = _generate_prime(half, exponent, rng)
        if p == q:
            continue
        n = p * q
        if n.bit_length() != bits:
            continue
        phi = (p - 1) * (q - 1)
        try:
            d = pow(exponent, -1, phi)
        except ValueError:
            continue
        key = RSAPrivateKey(n=n, e=exponent, d=d, p=max(p, q), q=min(p, q))
        if pow(pow(42, key.e, key.n), key.d, key.n) == 42:
            return key


def _digest_info_sha256(digest: bytes) -> bytes:
    return asn1.sequence(
        asn1.algorithm_identifier(SHA256_OID),
        asn1.octet_string(digest),
    )


def sign_sha256(key: RSAPrivateKey, message: bytes) -> bytes:
    digest_info = _digest_info_sha256(hashlib.sha256(message).digest())
    k = key.key_size_bytes
    if len(digest_info) + 11 > k:
        raise ValueError("RSA key too small for SHA-256 signature")
    encoded = b"\x00\x01" + b"\xff" * (k - len(digest_info) - 3) + b"\x00" + digest_info
    signature = pow(int.from_bytes(encoded, "big"), key.d, key.n)
    return signature.to_bytes(k, "big")


def verify_sha256(public_key: RSAPublicKey, message: bytes, signature: bytes) -> bool:
    k = (public_key.n.bit_length() + 7) // 8
    if len(signature) != k:
        return False
    digest_info = _digest_info_sha256(hashlib.sha256(message).digest())
    expected = b"\x00\x01" + b"\xff" * (k - len(digest_info) - 3) + b"\x00" + digest_info
    actual = pow(int.from_bytes(signature, "big"), public_key.e, public_key.n).to_bytes(k, "big")
    return hmac.compare_digest(actual, expected)


def iter_prime_candidates(rng: DeterministicRandom, bits: int) -> Iterator[int]:
    candidate = _candidate(bits, rng)
    while True:
        yield candidate
        candidate += 2
