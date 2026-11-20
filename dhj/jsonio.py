"""Deterministic canonical JSON and content digests.

Canonical bytes are the byte-identity basis for row summaries, bucket
fingerprints, ledgers and snapshots. Rules: UTF-8, sort keys, no whitespace
separators, non-ASCII allowed, NaN/Infinity rejected (must never appear in a
canonical payload).
"""

import hashlib
import json
import os
import tempfile


class CanonicalError(ValueError):
    pass


def dumps_canonical(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_hex(obj):
    return hashlib.sha256(dumps_canonical(obj)).hexdigest()


def dumps_pretty(obj):
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def write_json(path, obj, atomic=False, mode=0o644):
    data = dumps_pretty(obj).encode("utf-8")
    if atomic:
        write_atomic(path, data, mode=mode)
    else:
        with open(path, "wb") as handle:
            handle.write(data)
        os.chmod(path, mode)


def read_json(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)


def _reject_constant(value):
    raise CanonicalError(f"non-finite JSON constant is not allowed: {value}")


def write_atomic(path, data, mode=0o644):
    """Write ``data`` to ``path`` atomically (same-directory temp + replace)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()
