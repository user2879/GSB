"""Encode / decode / normalize the r1, r2 and r3 receipt formats.

Field sets per format:

* r1: ``batch_id``, ``request_ids``, ``output_digest``
* r2: r1 + ``param_version``, ``elapsed_ms``, ``worker_id``
* r3: r2 + ``ranks``, ``superseded_refs``, ``param_digest``

Normalization fills missing historical fields from explicit bases and returns
the source format version, the set of filled fields and the filling basis for
every filled field.
"""

from .digest import digest
from .errors import FieldError, RankError, RefError

R1_FIELDS = ("batch_id", "request_ids", "output_digest")
R2_FIELDS = R1_FIELDS + ("param_version", "elapsed_ms", "worker_id")
R3_FIELDS = R2_FIELDS + ("ranks", "superseded_refs", "param_digest")

FORMAT_FIELDS = {"r1": R1_FIELDS, "r2": R2_FIELDS, "r3": R3_FIELDS}


def detect_format(record):
    if not isinstance(record, dict):
        raise FieldError("receipt must be a JSON object", field="<record>")
    if all(key in record for key in R3_FIELDS):
        return "r3"
    if all(key in record for key in R2_FIELDS):
        return "r2"
    if all(key in record for key in R1_FIELDS):
        return "r1"
    missing = next(
        key for key in R1_FIELDS if key not in record
    )
    raise FieldError(
        "receipt does not satisfy any known format", field=missing
    )


def encode(normalized, fmt):
    """Project a normalized receipt onto a historical format."""
    if fmt not in FORMAT_FIELDS:
        raise FieldError(f"unknown receipt format {fmt!r}", field="format")
    return {key: normalized["fields"][key] for key in FORMAT_FIELDS[fmt]}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def decode(record, known_request_ids=None, *, line=None):
    """Validate a raw receipt and return its normalized representation.

    ``known_request_ids`` enables cross-request reference checks (``E_REF``);
    structural and scalar checks raise ``E_FIELD`` / ``E_RANK``.
    """
    fmt = detect_format(record)
    known = set(known_request_ids or [])

    batch_id = record.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise FieldError("batch_id must be a non-empty string",
                         field="batch_id", line=line)
    request_ids = record.get("request_ids")
    if not isinstance(request_ids, list) or not request_ids:
        raise FieldError("request_ids must be a non-empty list",
                         field="request_ids", line=line)
    if not all(isinstance(rid, str) and rid for rid in request_ids):
        raise FieldError("request_ids entries must be non-empty strings",
                         field="request_ids", line=line)
    if len(set(request_ids)) != len(request_ids):
        raise FieldError("request_ids must be unique",
                         field="request_ids", line=line)
    output_digest = record.get("output_digest")
    if not isinstance(output_digest, str) or not output_digest:
        raise FieldError("output_digest must be a non-empty string",
                         field="output_digest", line=line)

    if known:
        for rid in request_ids:
            if rid not in known:
                raise RefError(
                    f"receipt references unknown request {rid!r}",
                    field="request_ids", line=line,
                )

    fields = {
        "batch_id": batch_id,
        "request_ids": list(request_ids),
        "output_digest": output_digest,
    }

    if fmt in ("r2", "r3"):
        param_version = record.get("param_version")
        if not _is_int(param_version) or param_version < 0:
            raise FieldError(
                "param_version must be a non-negative integer",
                field="param_version", line=line,
            )
        elapsed_ms = record.get("elapsed_ms")
        if not _is_int(elapsed_ms) or elapsed_ms < 0:
            raise FieldError("elapsed_ms must be a non-negative integer",
                             field="elapsed_ms", line=line)
        worker_id = record.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            raise FieldError("worker_id must be a non-empty string",
                             field="worker_id", line=line)
        fields.update(
            param_version=param_version,
            elapsed_ms=elapsed_ms,
            worker_id=worker_id,
        )

    if fmt == "r3":
        ranks = record.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != len(request_ids):
            raise FieldError(
                "ranks length must match request_ids length",
                field="ranks", line=line,
            )
        if not all(_is_int(rank) for rank in ranks):
            raise FieldError("ranks entries must be integers",
                             field="ranks", line=line)
        size = len(request_ids)
        for rank in ranks:
            if rank < 0 or rank >= size:
                raise RankError(
                    f"rank {rank} out of bounds for batch size {size}",
                    field="ranks", line=line,
                )
        if len(set(ranks)) != len(ranks):
            raise RankError("ranks must be unique within a batch",
                            field="ranks", line=line)
        superseded_refs = record.get("superseded_refs")
        if not isinstance(superseded_refs, list) or not all(
            isinstance(ref, str) and ref for ref in superseded_refs
        ):
            raise FieldError(
                "superseded_refs must be a list of non-empty strings",
                field="superseded_refs", line=line,
            )
        if known:
            for ref in superseded_refs:
                if ref not in known:
                    raise RefError(
                        f"superseded reference points to unknown request "
                        f"{ref!r}",
                        field="superseded_refs", line=line,
                    )
        param_digest = record.get("param_digest")
        if not isinstance(param_digest, str) or not param_digest:
            raise FieldError("param_digest must be a non-empty string",
                             field="param_digest", line=line)
        fields.update(
            ranks=list(ranks),
            superseded_refs=list(superseded_refs),
            param_digest=param_digest,
        )

    return {"format": fmt, "fields": fields}


def normalize(decoded, *, worker_id, param_version, param_digest,
              elapsed_ms=0, superseded_refs=None):
    """Fill historical-format gaps using explicit filling bases.

    Returns the full r3 field set plus provenance metadata:
    ``source_format``, ``filled_fields`` (sorted) and ``fill_basis`` mapping
    each filled field to the basis used.
    """
    fmt = decoded["format"]
    fields = dict(decoded["fields"])
    fill_basis = {}

    def fill(key, value, basis):
        if key not in fields:
            fields[key] = value
            fill_basis[key] = basis

    fill("param_version", int(param_version),
         "merge_rules.default_param_version")
    fill("elapsed_ms", int(elapsed_ms), "default:zero_elapsed_ms")
    fill("worker_id", worker_id, "plan.worker_id")

    request_ids = fields["request_ids"]
    fill("ranks", list(range(len(request_ids))),
         "stable_order:request_ids_sorted")
    fill("superseded_refs", list(superseded_refs or []),
         "default:empty_superseded_refs")
    fill("param_digest", param_digest,
         "fixedpoint.parameter_digest(worker,version,seed)")

    normalized_digest = digest(
        {"kind": "normalized_receipt", "fields": fields}
    )
    return {
        "source_format": fmt,
        "fields": fields,
        "filled_fields": sorted(fill_basis),
        "fill_basis": fill_basis,
        "normalized_digest": normalized_digest,
    }
