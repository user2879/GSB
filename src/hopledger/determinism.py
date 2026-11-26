from __future__ import annotations

import hashlib


def deterministic_body(seed: int, seed_offset: int, length: int) -> bytes:
    """Generate fixed-length bytes with a counter-mode SHA-256 KDF."""

    output = bytearray()
    counter = 0
    prefix = f"hopledger-body/v1/{seed}/{seed_offset}/".encode("ascii")
    while len(output) < length:
        output.extend(hashlib.sha256(prefix + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(output[:length])


def failure_selected(hop_id: str, request_id: str, ratio_bp: int) -> bool:
    if ratio_bp <= 0:
        return False
    if ratio_bp >= 10_000:
        return True
    digest = hashlib.sha256(f"hopledger-failure/v1/{hop_id}/{request_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return bucket < ratio_bp


def request_digest(seed: int, seed_offset: int, length: int) -> str:
    return hashlib.sha256(deterministic_body(seed, seed_offset, length)).hexdigest()
