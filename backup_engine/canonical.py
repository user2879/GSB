"""Canonical serialization, digest, and path validation helpers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from typing import Any


SUPPORTED_ALGORITHMS = ("sha256", "hmac-sha256")
DIGEST_PREFIX_LENGTH = 12

_HEX_64 = re.compile(r"^[0-9a-fA-F]{64}$")
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_CHAIN_KEYS = {
    "fingerprint",
    "previous_fingerprint",
    "prev_fingerprint",
    "previous_snapshot_fingerprint",
    "parent_fingerprint",
    "previous_version",
    "prev_version",
    "parent_version",
}
_ALGORITHM_KEYS = {"algorithm", "algo", "digest_algorithm"}


class CanonicalError(ValueError):
    """Raised when canonical input cannot be serialized."""


class StrictJsonError(ValueError):
    """Raised when JSON contains duplicate keys or non-finite constants."""


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise StrictJsonError(f"non-finite JSON constant is not allowed: {value}")


def strict_json_loads(data: bytes | str) -> Any:
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = data
    return json.loads(
        text,
        object_pairs_hook=_object_pairs,
        parse_constant=_reject_constant,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes with recursively sorted object keys."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CanonicalError(str(exc)) from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_prefix(digest: str) -> str:
    return digest[:DIGEST_PREFIX_LENGTH]


def normalize_algorithm(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "sha256": "sha256",
        "sha-256": "sha256",
        "hmacsha256": "hmac-sha256",
        "hmac-sha256": "hmac-sha256",
        "hmac-sha-256": "hmac-sha256",
    }
    return aliases.get(normalized)


def normalize_hex_fingerprint(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if candidate.startswith(("sha256:", "hmac-sha256:")):
        candidate = candidate.split(":", 1)[1]
    return candidate if _HEX_64.fullmatch(candidate) else None


def normalize_digest(value: Any, fallback_algorithm: str) -> tuple[str | None, str | None, str | None]:
    """Return ``(algorithm, hex_digest, error_code)`` for a digest field."""

    if not isinstance(value, str):
        return None, None, "E_DIGEST_FORMAT"
    candidate = value.strip()
    algorithm = fallback_algorithm
    body = candidate
    if ":" in candidate:
        prefix, body = candidate.split(":", 1)
        algorithm = normalize_algorithm(prefix) or ""
        if algorithm not in SUPPORTED_ALGORITHMS:
            return None, None, "E_ALGORITHM"
    body = body.strip().lower()
    if algorithm not in SUPPORTED_ALGORITHMS:
        return None, None, "E_ALGORITHM"
    if not _HEX_64.fullmatch(body):
        return None, None, "E_DIGEST_FORMAT"
    return algorithm, body, None


def compute_content_digest(algorithm: str, content: bytes, hmac_key: bytes | None) -> bytes:
    if algorithm == "sha256":
        return hashlib.sha256(content).digest()
    if algorithm == "hmac-sha256":
        if hmac_key is None:
            raise ValueError("HMAC-SHA256 requires an HMAC key")
        return hmac.new(hmac_key, content, hashlib.sha256).digest()
    raise ValueError(f"unsupported digest algorithm: {algorithm}")


def fingerprint_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build the content-addressed snapshot fingerprint payload.

    Chain-link fields and an optional self fingerprint are deliberately excluded.
    This makes fingerprints independent of their position in a chain and makes
    self-reference/cycle detection meaningful.
    """

    version = snapshot.get("version")
    algorithm_value = snapshot.get("algorithm")
    if algorithm_value is None:
        for key in ("algo", "digest_algorithm"):
            if key in snapshot:
                algorithm_value = snapshot[key]
                break
    algorithm = normalize_algorithm(algorithm_value) or "sha256"
    schema_version = snapshot.get("schema_version", 1)

    payload: dict[str, Any] = {
        "algorithm": algorithm,
        "entries": snapshot.get("entries"),
        "schema_version": schema_version if isinstance(schema_version, int) else schema_version,
        "version": version,
    }
    for key, value in snapshot.items():
        if key in _CHAIN_KEYS or key in _ALGORITHM_KEYS:
            continue
        if key in {"schema_version", "version", "entries"}:
            continue
        payload[key] = value
    return payload


def compute_snapshot_fingerprint(
    snapshot: dict[str, Any], algorithm: str, hmac_key: bytes | None = None
) -> str:
    payload_bytes = canonical_json_bytes(fingerprint_payload(snapshot))
    if algorithm == "sha256":
        return hashlib.sha256(payload_bytes).hexdigest()
    if algorithm == "hmac-sha256":
        if hmac_key is None:
            raise ValueError("HMAC-SHA256 requires an HMAC key")
        return hmac.new(hmac_key, payload_bytes, hashlib.sha256).hexdigest()
    raise ValueError(f"unsupported fingerprint algorithm: {algorithm}")


def decode_content(entry: dict[str, Any]) -> tuple[bytes | None, str | None]:
    """Decode entry content from canonical base64 or a UTF-8 convenience field."""

    has_b64 = "content_b64" in entry
    has_text = "content" in entry
    if has_b64 and has_text:
        return None, "E_TAMPER"
    if has_b64:
        value = entry.get("content_b64")
        if not isinstance(value, str):
            return None, "E_TAMPER"
        try:
            return base64.b64decode(value.encode("ascii"), validate=True), None
        except (UnicodeEncodeError, binascii.Error, ValueError):
            return None, "E_TAMPER"
    if has_text:
        value = entry.get("content")
        if not isinstance(value, str):
            return None, "E_TAMPER"
        try:
            return value.encode("utf-8"), None
        except UnicodeEncodeError:
            return None, "E_TAMPER"
    return None, "E_TAMPER"


def encode_content(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def normalize_relative_path(value: Any) -> tuple[str | None, str | None]:
    """Validate and normalize a Windows-safe relative path to POSIX form."""

    if not isinstance(value, str) or not value:
        return None, "E_PATH"
    if "\x00" in value or any(ord(char) < 32 for char in value):
        return None, "E_PATH"
    candidate = value.replace("\\", "/")
    if candidate.startswith("/") or _DRIVE_PREFIX.match(candidate):
        return None, "E_PATH"
    if candidate.endswith("/"):
        return None, "E_PATH"
    parts = candidate.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None, "E_PATH"
    for part in parts:
        if part != part.rstrip(" ."):
            return None, "E_PATH"
        if any(char in '<>:"|?*' for char in part):
            return None, "E_PATH"
        stem = part.split(".", 1)[0].casefold()
        if stem in _WINDOWS_RESERVED:
            return None, "E_PATH"
    normalized = "/".join(parts)
    if len(normalized) > 4096:
        return None, "E_PATH"
    return normalized, None


def path_sort_key(path: str) -> tuple[str, str]:
    return (path.casefold(), path)
