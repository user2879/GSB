"""r1/r2/r3 attribution receipt codec and normalization.

Historical formats:

* ``r1``: batch_id, request_id, output_digest
* ``r2``: r1 + param_version, duration_ms, worker_id
* ``r3``: r2 + rank (stable position inside the batch), duplicate_ref
  (superseded duplicate delivery) and param_digest snapshot

Decoding normalizes every receipt onto one record shape and records the
source format, every field that had to be filled and *why* it was filled, so
the unified ledger is independently reviewable.
"""

from dataclasses import dataclass, field

from .errors import ChainError, FieldError
from .jsonlocate import load

SUPPORTED_FORMATS = ("r1", "r2", "r3")

_FIELDS = {
    "r1": ("batch_id", "request_id", "output_digest"),
    "r2": ("batch_id", "request_id", "output_digest", "param_version",
           "duration_ms", "worker_id"),
    "r3": ("batch_id", "request_id", "output_digest", "param_version",
           "duration_ms", "worker_id", "rank", "duplicate_ref",
           "param_digest"),
}

_NORMALIZED_FIELDS = (
    "batch_id", "request_id", "output_digest", "worker_id",
    "param_version", "duration_ms", "rank", "duplicate_ref",
    "param_digest",
)


@dataclass
class NormalizedReceipt:
    batch_id: str
    request_id: str
    output_digest: str
    worker_id: object
    param_version: object
    duration_ms: object
    rank: object
    duplicate_ref: object
    param_digest: object
    source_format: str
    filled_fields: list = field(default_factory=list)
    source_line: int = None

    def to_dict(self):
        data = {name: getattr(self, name) for name in _NORMALIZED_FIELDS}
        data["source_format"] = self.source_format
        data["filled_fields"] = self.filled_fields
        data["source_line"] = self.source_line
        return data


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _batch_worker(batch_id):
    parts = str(batch_id).split(":")
    if len(parts) != 4 or parts[0] != "B":
        return None
    return parts[1]


def _fill_basis(fmt, name, raw, receipt):
    if name == "worker_id":
        derived = _batch_worker(raw.get("batch_id", ""))
        return derived, f"derived:batch_id_prefix:{derived}"
    if name == "param_version":
        return 1, "default:param_version=1"
    if name == "duration_ms":
        return 0, "default:duration_ms=0"
    return None, f"absent:{fmt}:{name}"


def decode_receipt(raw, fmt=None, known_request_ids=None,
                   batch_sizes=None, line_map=None, source=None,
                   path=()):
    """Decode and validate one raw receipt into a normalized record."""
    if not isinstance(raw, dict):
        raise FieldError("receipt must be an object",
                         field=".".join(str(p) for p in path),
                         line=line_map.line_of(*path) if line_map else None,
                         source=source)
    fmt = fmt or raw.get("format")
    if fmt not in SUPPORTED_FORMATS:
        raise FieldError(f"unsupported receipt format {fmt!r}",
                         field=_join(path, "format"),
                         line=line_map.line_of(*(path + ("format",)))
                         if line_map else None, source=source)

    def fail(name, reason, code=FieldError):
        key_path = path + (name,)
        line = line_map.line_of(*key_path) if line_map else None
        raise code(reason, field=".".join(str(p) for p in key_path),
                   line=line, source=source)

    for name in _FIELDS[fmt]:
        if name not in raw:
            fail(name, f"required field {name!r} missing for format {fmt}")

    for name in ("batch_id", "request_id", "output_digest"):
        if not isinstance(raw[name], str) or not raw[name]:
            fail(name, f"{name} must be a non-empty string")

    if _batch_worker(raw["batch_id"]) is None:
        fail("batch_id",
             "batch_id must follow B:<worker_id>:<batch_seq>:<hash>")

    if fmt in ("r2", "r3"):
        if not _is_int(raw["param_version"]) or raw["param_version"] < 1:
            fail("param_version", "param_version must be a positive integer")
        if not _is_int(raw["duration_ms"]) or raw["duration_ms"] < 0:
            fail("duration_ms", "duration_ms must be a non-negative integer")
        if not isinstance(raw["worker_id"], str) or not raw["worker_id"]:
            fail("worker_id", "worker_id must be a non-empty string")
        prefix_worker = _batch_worker(raw["batch_id"])
        if raw["worker_id"] != prefix_worker:
            fail("worker_id",
                 f"worker_id {raw['worker_id']!r} disagrees with batch_id "
                 f"prefix {prefix_worker!r}")

    rank = None
    duplicate_ref = None
    param_digest = None
    if fmt == "r3":
        rank = raw["rank"]
        if not _is_int(rank) or rank < 0:
            fail("rank", "rank must be a non-negative integer")
        size = (batch_sizes or {}).get(raw["batch_id"])
        if size is not None and rank >= size:
            fail("rank",
                 f"rank {rank} is out of bounds for batch of size {size}")
        duplicate_ref = raw["duplicate_ref"]
        if duplicate_ref is not None and not isinstance(duplicate_ref, str):
            fail("duplicate_ref",
                 "duplicate_ref must be null or a request id string")
        if not isinstance(raw["param_digest"], str) \
                or not raw["param_digest"]:
            fail("param_digest",
                 "param_digest must be a non-empty string")

    normalized = {
        "batch_id": raw["batch_id"],
        "request_id": raw["request_id"],
        "output_digest": raw["output_digest"],
        "worker_id": raw.get("worker_id"),
        "param_version": raw.get("param_version"),
        "duration_ms": raw.get("duration_ms"),
        "rank": rank,
        "duplicate_ref": duplicate_ref,
        "param_digest": param_digest,
    }

    if known_request_ids is not None:
        if raw["request_id"] not in known_request_ids:
            fail("request_id",
                 f"request_id {raw['request_id']!r} is not present in the "
                 "request stream", code=ChainError)
        if duplicate_ref is not None \
                and duplicate_ref not in known_request_ids:
            fail("duplicate_ref",
                 f"duplicate_ref {duplicate_ref!r} does not point at an "
                 "existing request", code=ChainError)
        if duplicate_ref == raw["request_id"]:
            fail("duplicate_ref",
                 "duplicate_ref must not reference the receipt itself",
                 code=ChainError)

    filled = []
    for name in _NORMALIZED_FIELDS:
        if name in _FIELDS[fmt]:
            continue
        value, basis = _fill_basis(fmt, name, raw, normalized)
        normalized[name] = value
        filled.append({"field": name, "basis": basis, "filled_value": value})

    return NormalizedReceipt(
        source_format=fmt,
        filled_fields=filled,
        source_line=line_map.line_of(*path) if line_map else None,
        **normalized,
    )


def _join(path, name):
    return ".".join(str(p) for p in tuple(path) + (name,))


def decode_receipts_file(path, known_request_ids=None, batch_sizes=None):
    """Load a JSON array receipt file and validate every receipt."""
    document, line_map = load(str(path))
    if not isinstance(document, list) or not document:
        raise FieldError("receipt document must be a non-empty array",
                         source=str(path))
    records = []
    ranks_seen = {}
    for index, raw in enumerate(document):
        record = decode_receipt(
            raw, known_request_ids=known_request_ids,
            batch_sizes=batch_sizes, line_map=line_map, source=str(path),
            path=(index,),
        )
        if record.rank is not None:
            seen = ranks_seen.setdefault(record.batch_id, set())
            if record.rank in seen:
                key_path = (index, "rank")
                raise FieldError(
                    f"duplicate rank {record.rank} inside batch "
                    f"{record.batch_id!r}",
                    field=".".join(str(p) for p in key_path),
                    line=line_map.line_of(*key_path), source=str(path))
            seen.add(record.rank)
        records.append(record)
    return records


def encode_receipt(record, fmt):
    """Serialize a normalized record back into a raw *fmt* payload."""
    if fmt not in SUPPORTED_FORMATS:
        raise FieldError(f"unsupported receipt format {fmt!r}")
    if isinstance(record, NormalizedReceipt):
        record = record.to_dict()
    payload = {"format": fmt}
    for name in _FIELDS[fmt]:
        if name not in record:
            raise FieldError(f"cannot encode {fmt}: field {name!r} missing")
        payload[name] = record[name]
    return payload
