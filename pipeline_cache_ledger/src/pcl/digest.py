"""Deterministic content digest rule used across every cache format.

Rule (task-declared, platform independent):
    digest = "sha256:" + sha256_hex(artifact file bytes)

Byte-exact raw file bytes are hashed so Windows CRLF / Linux LF inputs keep
the same digest as long as the stored bytes are identical.
"""

import hashlib

DIGEST_ALGO = "sha256"
DIGEST_PREFIX = "sha256:"


def digest_bytes(data: bytes) -> str:
    return DIGEST_PREFIX + hashlib.sha256(data).hexdigest()


def digest_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return DIGEST_PREFIX + h.hexdigest()
