"""Codecs and normalization for the three historical receipt formats.

Wire forms (one line per receipt):

* x1 - fixed width, no delimiter::

      <batch_id:32 hex><fixed_point_sum:21 signed zero-padded>   # 53 chars

* x2 - pipe delimited, adds subscriber, completion millis and format version::

      x2|<batch_id>|<subscriber_id>|<completion_ms>|<fixed_point_sum>|2

* x3 - adds the in-batch event seq set, covered redelivery refs and caliber::

      x3|<batch_id>|<subscriber_id>|<completion_ms>|<fixed_point_sum>\
|<event_seqs csv>|<covered_refs csv>|<caliber_version>|3

x1 cannot carry the event seq set; normalization fills every missing field
from a ledger dispatch record and records exactly which fields were backfilled
and the traceable basis used.
"""

from __future__ import annotations

from collections.abc import Callable

from .aggregation import CALIBER_VERSION
from .digest import canonical_text, sha256_hex
from .errors import BackfillError, FormatError

BATCH_ID_WIDTH = 32
SUMMARY_WIDTH = 21
X1_LINE_WIDTH = BATCH_ID_WIDTH + SUMMARY_WIDTH
DELIMITER = "|"
FORMAT_ORDER = {"x1": 1, "x2": 2, "x3": 3}


def _summary_field(value: int) -> str:
    if value < 0:
        return "-" + str(abs(value)).zfill(SUMMARY_WIDTH - 1)
    return str(value).zfill(SUMMARY_WIDTH)


def _parse_summary(text: str, entry: str) -> int:
    if len(text) != SUMMARY_WIDTH or any(ch not in "0123456789-" for ch in text):
        raise FormatError(
            "fixed-width summary field has wrong width",
            entry=entry,
            field="fixed_point_sum",
        )
    try:
        return int(text)
    except ValueError:
        raise FormatError(
            "fixed-width summary is not an integer",
            entry=entry,
            field="fixed_point_sum",
        )


def encode_x1(batch_id: str, fixed_point_sum: int) -> str:
    if len(batch_id) != BATCH_ID_WIDTH:
        raise FormatError("batch_id width mismatch", entry=batch_id, field="batch_id")
    return batch_id + _summary_field(fixed_point_sum)


def encode_x2(batch_id, subscriber_id, completion_ms, fixed_point_sum) -> str:
    return DELIMITER.join(
        [
            "x2",
            batch_id,
            subscriber_id,
            str(int(completion_ms)),
            str(int(fixed_point_sum)),
            "2",
        ]
    )


def encode_x3(
    batch_id,
    subscriber_id,
    completion_ms,
    fixed_point_sum,
    event_seqs,
    covered_refs,
    caliber_version,
) -> str:
    seqs = ",".join(str(seq) for seq in event_seqs)
    refs = ",".join(str(ref) for ref in covered_refs)
    return DELIMITER.join(
        [
            "x3",
            batch_id,
            subscriber_id,
            str(int(completion_ms)),
            str(int(fixed_point_sum)),
            seqs,
            refs,
            caliber_version,
            "3",
        ]
    )


def _parse_int(token, field, entry):
    try:
        return int(token)
    except (TypeError, ValueError):
        raise FormatError(f"field {field} must be an integer", entry=entry, field=field)


def _parse_seq_csv(token, field, entry):
    if token == "":
        return []
    values = []
    for part in token.split(","):
        values.append(_parse_int(part, field, entry))
    return values


def decode(receipt_text: str) -> dict:
    """Decode any of the three wire forms into a raw field dictionary."""
    line = receipt_text.strip()
    if not line:
        raise FormatError("empty receipt line", entry="<receipt>", field="<line>")
    entry = line[:16]

    if DELIMITER in line:
        parts = line.split(DELIMITER)
        version = parts[0]
        if version == "x2":
            if len(parts) != 6:
                raise FormatError(
                    "x2 receipt must have 6 pipe fields", entry=entry, field="<fields>"
                )
            _, batch_id, subscriber_id, completion, summary, ver = parts
            if ver != "2":
                raise FormatError(
                    "x2 format version token must be 2",
                    entry=entry,
                    field="format_version",
                )
            return {
                "format_version": "x2",
                "batch_id": batch_id,
                "subscriber_id": subscriber_id,
                "completion_ms": _parse_int(completion, "completion_ms", entry),
                "fixed_point_sum": _parse_int(summary, "fixed_point_sum", entry),
            }
        if version == "x3":
            if len(parts) != 9:
                raise FormatError(
                    "x3 receipt must have 9 pipe fields", entry=entry, field="<fields>"
                )
            (
                _,
                batch_id,
                subscriber_id,
                completion,
                summary,
                seqs,
                refs,
                caliber,
                ver,
            ) = parts
            if ver != "3":
                raise FormatError(
                    "x3 format version token must be 3",
                    entry=entry,
                    field="format_version",
                )
            return {
                "format_version": "x3",
                "batch_id": batch_id,
                "subscriber_id": subscriber_id,
                "completion_ms": _parse_int(completion, "completion_ms", entry),
                "fixed_point_sum": _parse_int(summary, "fixed_point_sum", entry),
                "event_seqs": _parse_seq_csv(seqs, "event_seqs", entry),
                "covered_refs": _parse_seq_csv(refs, "covered_refs", entry),
                "caliber_version": caliber,
            }
        raise FormatError(
            f"unknown receipt version '{version}'", entry=entry, field="format_version"
        )

    if len(line) < X1_LINE_WIDTH:
        raise FormatError(
            f"fixed-width receipt too short: {len(line)} < {X1_LINE_WIDTH}",
            entry=entry,
            field="<line-width>",
        )
    if len(line) > X1_LINE_WIDTH:
        raise FormatError(
            f"unknown receipt version or over-wide fixed line ({len(line)} chars)",
            entry=entry,
            field="format_version",
        )
    batch_id = line[:BATCH_ID_WIDTH]
    if any(ch not in "0123456789abcdef" for ch in batch_id):
        raise FormatError(
            "x1 batch_id must be 32 lowercase hex chars", entry=entry, field="batch_id"
        )
    return {
        "format_version": "x1",
        "batch_id": batch_id,
        "fixed_point_sum": _parse_summary(line[BATCH_ID_WIDTH:], entry),
    }


def _backfill_record(batch_id, ledger_lookup):
    try:
        record = ledger_lookup(batch_id)
    except BackfillError:
        raise
    except Exception as exc:  # ledger unavailable is not a traceable basis
        raise BackfillError(
            f"backfill basis not traceable for {batch_id}: {exc}",
            entry=batch_id,
            field="<ledger>",
        )
    if not record:
        raise BackfillError(
            f"no registered dispatch record to backfill batch {batch_id}",
            entry=batch_id,
            field="<ledger>",
        )
    required = ("dispatch_id", "subscriber_id", "event_seqs", "digest")
    missing = [key for key in required if key not in record]
    if missing:
        raise BackfillError(
            f"ledger record for {batch_id} lacks traceable fields {missing}",
            entry=batch_id,
            field=",".join(missing),
        )
    return record


def normalize(receipt_text: str, ledger_lookup: Callable) -> dict:
    """Normalize a raw receipt into the full canonical field set.

    ``ledger_lookup(batch_id)`` must return the registered dispatch record
    (or a falsey value / raise BackfillError when untraceable).
    """
    raw = decode(receipt_text)
    source_format = raw["format_version"]
    batch_id = raw["batch_id"]
    backfilled: list[str] = []
    basis = None

    record = None
    needs_ledger = source_format in ("x1", "x2")
    if needs_ledger:
        record = _backfill_record(batch_id, ledger_lookup)
        basis = f"ledger:dispatch:{record['dispatch_id']}:{record['digest'][:12]}"

    def fill(field_name, ledger_value):
        if field_name not in raw or raw.get(field_name) is None:
            backfilled.append(field_name)
            return ledger_value
        return raw[field_name]

    if source_format == "x1":
        subscriber_id = fill("subscriber_id", record["subscriber_id"])
        completion_ms = fill("completion_ms", record.get("completion_ms"))
        event_seqs = fill("event_seqs", list(record["event_seqs"]))
        covered_refs = fill("covered_refs", list(record.get("covered_refs", [])))
        caliber = fill(
            "caliber_version", record.get("caliber_version", CALIBER_VERSION)
        )
    elif source_format == "x2":
        subscriber_id = raw["subscriber_id"]
        completion_ms = raw["completion_ms"]
        event_seqs = fill("event_seqs", list(record["event_seqs"]))
        covered_refs = fill("covered_refs", list(record.get("covered_refs", [])))
        caliber = fill(
            "caliber_version", record.get("caliber_version", CALIBER_VERSION)
        )
    else:
        subscriber_id = raw["subscriber_id"]
        completion_ms = raw["completion_ms"]
        event_seqs = list(raw["event_seqs"])
        covered_refs = list(raw["covered_refs"])
        caliber = raw["caliber_version"]

    if completion_ms is None:
        raise BackfillError(
            f"completion_ms not traceable for batch {batch_id}",
            entry=batch_id,
            field="completion_ms",
        )

    normalized = {
        "batch_id": batch_id,
        "subscriber_id": subscriber_id,
        "completion_ms": int(completion_ms),
        "event_seqs": sorted(int(seq) for seq in event_seqs),
        "covered_refs": sorted(int(ref) for ref in covered_refs),
        "caliber_version": caliber,
        "fixed_point_sum": int(raw["fixed_point_sum"]),
        "format_version": source_format,
        "source_format": source_format,
        "backfilled_fields": sorted(backfilled),
        "backfill_basis": basis,
        "format_version_int": FORMAT_ORDER[source_format],
    }
    normalized["normalized_digest"] = sha256_hex(_adoption_core(normalized))
    return normalized


def _adoption_core(normalized: dict) -> dict:
    """Fields that must be byte-identical regardless of source format."""
    return {
        "batch_id": normalized["batch_id"],
        "subscriber_id": normalized["subscriber_id"],
        "completion_ms": normalized["completion_ms"],
        "event_seqs": normalized["event_seqs"],
        "covered_refs": normalized["covered_refs"],
        "caliber_version": normalized["caliber_version"],
        "fixed_point_sum": normalized["fixed_point_sum"],
    }


def canonical_normalized_text(normalized: dict) -> str:
    return canonical_text(_adoption_core(normalized))
