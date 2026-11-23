"""Signature authentication with key-version compatibility.

Explicit reproducible signature rule (no dict/hash iteration anywhere):

    expected = sha256(SECRET|CLIENT|SERVICE|SEQ|CRED_VERSION).hexdigest()

The expected signature is compared byte-for-byte with the credential's
signature string. Version compatibility rule:
  * cred_version == current version: verify against current secret.
  * cred_version <  current version: the credential stays valid while
    arrival_ms <= retire_ms(version) + signature_window_ms; inside the
    window it is verified against that version's secret, outside the
    window it is rejected as credential_expired.
  * unknown version: rejected as unknown_version.
"""

import hashlib

REASON_OK = "ok"
REASON_MISMATCH = "signature_mismatch"
REASON_EXPIRED = "credential_expired"
REASON_UNKNOWN_VERSION = "unknown_version"


def expected_signature(secret, client_key, service_id, seq, cred_version):
    msg = f"{secret}|{client_key}|{service_id}|{seq}|{cred_version}"
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


def verify_signature(key_versions, signature_window_ms, cred_version,
                     signature, client_key, service_id, seq, arrival_ms):
    """Return (ok, reason). key_versions: tuple of KeyVersion, ascending."""
    current = key_versions[-1]
    target = None
    for kv in key_versions:  # tuple order is the declared ascending order
        if kv.version == cred_version:
            target = kv
            break
    if target is None:
        return False, REASON_UNKNOWN_VERSION
    if target.version != current.version:
        if target.retire_ms < 0:
            # Non-current version without retirement date cannot happen
            # per spec validation, but stay total.
            return False, REASON_UNKNOWN_VERSION
        if arrival_ms > target.retire_ms + signature_window_ms:
            return False, REASON_EXPIRED
    expected = expected_signature(target.secret, client_key, service_id,
                                  seq, cred_version)
    if expected != signature:
        return False, REASON_MISMATCH
    return True, REASON_OK
