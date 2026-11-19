"""Independent ledger auditor.

This module intentionally reimplements feed ordering/grouping, assignment ID
derivation, and MLP inference.  It does not call the scheduler, feed parser, or
forward module from the execution path.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import numpy as np

from .canonical import canonical_json_bytes, sha256_hex
from .constants import (
    ARRAY_DIGEST_DOMAIN,
    ASSIGNMENT_DIGEST_DOMAIN,
    AUDIT_DIGEST_DOMAIN,
    BATCH_OUTPUT_DIGEST_DOMAIN,
    FEED_DIGEST_DOMAIN,
    INPUT_DIGEST_DOMAIN,
    LEDGER_FORMAT_VERSION,
    MODEL_ARRAY_ORDER,
    MODEL_DIGEST_DOMAIN,
    SAMPLE_OUTPUT_DIGEST_DOMAIN,
)


class AuditMismatch(Exception):
    code = "E_AUDIT"

    def __init__(
        self,
        message: str,
        *,
        sample_id: str | None,
        batch_slot: int | None,
        field: str,
        expected: Any,
        actual: Any,
        generation: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.sample_id = sample_id
        self.batch_slot = batch_slot
        self.field = field
        self.expected = expected
        self.actual = actual
        self.generation = generation

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "generation": self.generation,
            "first_mismatch": {
                "sample_id": self.sample_id,
                "batch_slot": self.batch_slot,
                "field": self.field,
                "expected": self.expected,
                "actual": self.actual,
            },
        }


def run_audit(*, ledger_path: Path, feed_path: Path) -> dict[str, Any]:
    ledger_path = Path(ledger_path)
    feed_path = Path(feed_path)
    conn = _open_read_only(ledger_path)
    try:
        header = conn.execute("SELECT * FROM ledger_header WHERE singleton = 1").fetchone()
        if header is None:
            raise AuditMismatch(
                "ledger header is missing",
                sample_id=None,
                batch_slot=None,
                field="ledger_header",
                expected="format_version=1 row",
                actual=None,
            )
        if int(header["format_version"]) != LEDGER_FORMAT_VERSION:
            raise AuditMismatch(
                "ledger format version is not supported",
                sample_id=None,
                batch_slot=None,
                field="ledger_format_version",
                expected=LEDGER_FORMAT_VERSION,
                actual=int(header["format_version"]),
            )
        generation_row = conn.execute(
            "SELECT * FROM generations ORDER BY generation_id DESC LIMIT 1"
        ).fetchone()
        if generation_row is None:
            raise AuditMismatch(
                "ledger has no generation",
                sample_id=None,
                batch_slot=None,
                field="generation",
                expected="one committed generation",
                actual=None,
            )
        generation = _decode_generation(generation_row)
        generation_id = int(generation["generation_id"])
        config = generation["config"]
        slots = tuple(sorted(int(item) for item in config["supported_slots"]))
        sample_ids = tuple(str(item) for item in generation["sample_ids"])
        arrays = _read_arrays(conn, generation_id, generation)
        inputs = arrays["inputs"]
        model = {name: arrays[name] for name in MODEL_ARRAY_ORDER}
        if inputs.shape[0] != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
            raise AuditMismatch(
                "ledger sample ids and input matrix are inconsistent",
                sample_id=None,
                batch_slot=None,
                field="sample_ids",
                expected=f"{len(sample_ids)} unique ids",
                actual=list(inputs.shape),
                generation=generation_id,
            )

        feed = _independent_load_feed(
            feed_path,
            supported_slots=slots,
            known_sample_ids=sample_ids,
            generation=generation_id,
        )
        if feed["feed_digest"] != generation["feed_digest"]:
            raise AuditMismatch(
                "feed digest does not match the active ledger generation",
                sample_id=None,
                batch_slot=None,
                field="feed_digest",
                expected=generation["feed_digest"],
                actual=feed["feed_digest"],
                generation=generation_id,
            )

        input_digest = _independent_input_digest(sample_ids, inputs)
        if input_digest != generation["input_digest"]:
            raise AuditMismatch(
                "input digest mismatch",
                sample_id=None,
                batch_slot=None,
                field="input_digest",
                expected=generation["input_digest"],
                actual=input_digest,
                generation=generation_id,
            )
        param_digest = _independent_model_digest(model)
        if param_digest != generation["param_digest"]:
            raise AuditMismatch(
                "model parameter digest mismatch",
                sample_id=None,
                batch_slot=None,
                field="param_digest",
                expected=generation["param_digest"],
                actual=param_digest,
                generation=generation_id,
            )

        slot_rows = {
            int(row["batch_slot"]): _decode_slot(row)
            for row in conn.execute(
                "SELECT * FROM slots WHERE generation_id = ? ORDER BY batch_slot",
                (generation_id,),
            ).fetchall()
        }
        assignment_rows: dict[int, list[dict[str, Any]]] = {}
        for row in conn.execute(
            "SELECT * FROM assignments WHERE generation_id = ? ORDER BY batch_slot, rank",
            (generation_id,),
        ).fetchall():
            assignment_rows.setdefault(int(row["batch_slot"]), []).append(dict(row))

        expected_slots = {group["batch_slot"]: group for group in feed["groups"]}
        auditable_slots: list[int] = []
        unavailable: list[dict[str, Any]] = []
        recomputed_slots: list[dict[str, Any]] = []
        for slot in slots:
            row = slot_rows.get(slot)
            if row is None:
                unavailable.append(
                    {
                        "batch_slot": slot,
                        "status": "missing",
                        "reason": "NO_COMMITTED_SLOT_ROW",
                    }
                )
                continue
            if row["status"] not in {"committed", "empty"}:
                unavailable.append(
                    {
                        "batch_slot": slot,
                        "status": row["status"],
                        "reason": "NOT_COMMITTED_BEFORE_TERMINATION_OR_NOT_YET_EXECUTED",
                    }
                )
                continue
            auditable_slots.append(slot)
            expected = _recompute_slot(
                slot=slot,
                group=expected_slots[slot],
                sample_ids=sample_ids,
                inputs=inputs,
                model=model,
                numbering_version=generation["numbering_version"],
                param_digest=generation["param_digest"],
            )
            _compare_slot(
                generation=generation_id,
                expected=expected,
                slot_row=row,
                assignments=assignment_rows.get(slot, []),
            )
            recomputed_slots.append(expected)

        expected_digest = _audit_digest(
            generation=generation,
            slots=recomputed_slots,
            unavailable=unavailable,
        )
        computed_ledger_digest = _independent_ledger_content_digest(conn)
        if computed_ledger_digest != str(header["content_digest"]):
            raise AuditMismatch(
                "ledger content digest mismatch",
                sample_id=None,
                batch_slot=None,
                field="ledger_content_digest",
                expected=str(header["content_digest"]),
                actual=computed_ledger_digest,
                generation=generation_id,
            )

        complete = not unavailable
        return {
            "code": "AUDIT_OK" if complete else "AUDIT_PARTIAL",
            "ok": True,
            "generation": generation_id,
            "version": generation_id,
            "audit_scope": "full" if complete else "partial",
            "auditable_slots": auditable_slots,
            "not_auditable_slots": unavailable,
            "not_auditable_rule": (
                "Only slots with ledger status committed or empty are auditable. "
                "Pending/running/unknown/missing slots were not transactionally committed "
                "and are never inferred from an aborted process."
            ),
            "recomputed_digest": expected_digest,
            "comparisons": {
                "slots_compared": len(auditable_slots),
                "assignments_compared": sum(
                    int(item["assignment_count"]) for item in recomputed_slots
                ),
                "ledger_content_digest": computed_ledger_digest,
            },
        }
    finally:
        conn.close()


def _open_read_only(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise AuditMismatch(
            f"ledger does not exist: {path}",
            sample_id=None,
            batch_slot=None,
            field="ledger_path",
            expected=str(path),
            actual=None,
        )
    uri = "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _decode_generation(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "generation_id": int(row["generation_id"]),
        "feed_digest": str(row["feed_digest"]),
        "input_digest": str(row["input_digest"]),
        "param_digest": str(row["param_digest"]),
        "numbering_version": str(row["numbering_version"]),
        "config": json.loads(row["config_json"]),
        "sample_ids": json.loads(row["sample_ids_json"]),
        "status": str(row["status"]),
        "replacement_reason": row["replacement_reason"],
        "supersedes_generation": row["supersedes_generation"],
        "affected_samples": int(row["affected_samples"]),
        "affected_slots": json.loads(row["affected_slots_json"]),
        "progress_cursor": int(row["progress_cursor"]),
        "completed_slot_count": int(row["completed_slot_count"]),
        "created_at": str(row["created_at"]),
        "completed_at": row["completed_at"],
    }


def _decode_slot(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "generation_id": int(row["generation_id"]),
        "batch_slot": int(row["batch_slot"]),
        "status": str(row["status"]),
        "cursor": int(row["cursor"]),
        "members": json.loads(row["members_json"]),
        "overwritten": json.loads(row["overwritten_json"]),
        "failed": json.loads(row["failed_json"]),
        "output_digest": row["output_digest"],
        "assignment_count": int(row["assignment_count"]),
        "error_code": row["error_code"],
        "updated_at": str(row["updated_at"]),
    }


def _read_arrays(
    conn: sqlite3.Connection,
    generation_id: int,
    generation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for row in conn.execute(
        "SELECT name, dtype, shape_json, digest, data FROM arrays "
        "WHERE generation_id = ? ORDER BY name",
        (generation_id,),
    ).fetchall():
        name = str(row["name"])
        array = np.frombuffer(row["data"], dtype=np.dtype(str(row["dtype"]))).reshape(
            tuple(json.loads(row["shape_json"]))
        ).copy()
        actual = sha256_hex(ARRAY_DIGEST_DOMAIN + _array_record(name, array) + b"\n")
        if actual != str(row["digest"]):
            raise AuditMismatch(
                "ledger array digest mismatch",
                sample_id=None,
                batch_slot=None,
                field=f"arrays[{name}].digest",
                expected=str(row["digest"]),
                actual=actual,
                generation=generation_id,
            )
        arrays[name] = array
    required = {"inputs", *MODEL_ARRAY_ORDER}
    missing = sorted(required - set(arrays))
    if missing:
        raise AuditMismatch(
            "ledger is missing arrays required for audit recomputation",
            sample_id=None,
            batch_slot=None,
            field="arrays",
            expected=sorted(required),
            actual=sorted(arrays),
            generation=generation_id,
        )
    return arrays


def _array_record(name: str, array: np.ndarray) -> bytes:
    data = np.ascontiguousarray(array, dtype="<f8")
    if data.size:
        data = data.copy()
        data[data == 0.0] = 0.0
    header = canonical_json_bytes(
        {"dtype": "<f8", "name": name, "order": "C", "shape": list(data.shape)}
    )
    return header + b"\n" + data.tobytes(order="C")


def _independent_load_feed(
    path: Path,
    *,
    supported_slots: Sequence[int],
    known_sample_ids: Sequence[str],
    generation: int,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditMismatch(
            "cannot read the original feed file",
            sample_id=None,
            batch_slot=None,
            field="feed",
            expected="valid JSON feed",
            actual=str(exc),
            generation=generation,
        ) from exc
    schema: Any = None
    if isinstance(value, list):
        raw_entries = value
    elif isinstance(value, dict):
        schema = value.get("schema")
        raw_entries = value.get("entries")
    else:
        raw_entries = None
    if not isinstance(raw_entries, list):
        raise AuditMismatch(
            "feed root does not contain an entries array",
            sample_id=None,
            batch_slot=None,
            field="feed.entries",
            expected="array",
            actual=type(raw_entries).__name__,
            generation=generation,
        )

    slot_set = set(int(item) for item in supported_slots)
    known = set(str(item) for item in known_sample_ids)
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise _feed_mismatch(generation, f"entries[{index}]", "object", item)
        sample_id = item.get("sample_id")
        arrival = item.get("arrival_seq")
        slot = item.get("batch_slot")
        admitted = item.get("admitted")
        reason = item.get("reject_reason", item.get("reason"))
        if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
            raise _feed_mismatch(generation, f"entries[{index}].sample_id", "non-empty string", sample_id)
        if type(arrival) is not int:
            raise _feed_mismatch(generation, f"entries[{index}].arrival_seq", "integer", arrival)
        if type(slot) is not int or slot not in slot_set:
            raise _feed_mismatch(generation, f"entries[{index}].batch_slot", sorted(slot_set), slot)
        if type(admitted) is not bool:
            raise _feed_mismatch(generation, f"entries[{index}].admitted", "boolean", admitted)
        if admitted and sample_id not in known:
            raise _feed_mismatch(generation, f"entries[{index}].sample_id", "known sample id", sample_id)
        if reason is not None and not isinstance(reason, str):
            raise _feed_mismatch(generation, f"entries[{index}].reject_reason", "string or null", reason)
        if admitted:
            reason = None
        elif reason is None or not reason.strip():
            reason = "admission_rejected"
        entries.append(
            {
                "sample_id": sample_id,
                "arrival_seq": arrival,
                "batch_slot": slot,
                "admitted": admitted,
                "reject_reason": reason,
                "source_index": index,
            }
        )

    arrivals: set[int] = set()
    for item in sorted(entries, key=lambda entry: (entry["arrival_seq"], entry["source_index"])):
        if item["arrival_seq"] in arrivals:
            raise _feed_mismatch(
                generation,
                f"entries[{item['source_index']}].arrival_seq",
                "unique integer",
                item["arrival_seq"],
            )
        arrivals.add(item["arrival_seq"])
    ordered = sorted(entries, key=lambda entry: (entry["arrival_seq"], entry["source_index"]))

    first_success: dict[str, dict[str, Any]] = {}
    success_slots: dict[str, set[int]] = {}
    for item in ordered:
        if item["admitted"]:
            success_slots.setdefault(item["sample_id"], set()).add(item["batch_slot"])
            first_success.setdefault(item["sample_id"], item)
    for item in ordered:
        if item["admitted"] and len(success_slots[item["sample_id"]]) > 1:
            primary = first_success[item["sample_id"]]["batch_slot"]
            if item["batch_slot"] != primary:
                raise _feed_mismatch(
                    generation,
                    f"entries[{item['source_index']}].batch_slot",
                    f"single slot for sample {item['sample_id']}",
                    item["batch_slot"],
                )

    groups: list[dict[str, Any]] = []
    for slot in sorted(slot_set):
        successes = [item for item in ordered if item["admitted"] and item["batch_slot"] == slot]
        failed = [
            {
                "sample_id": item["sample_id"],
                "arrival_seq": item["arrival_seq"],
                "batch_slot": item["batch_slot"],
                "reason": item["reject_reason"],
            }
            for item in ordered
            if (not item["admitted"]) and item["batch_slot"] == slot
        ]
        by_sample: dict[str, list[dict[str, Any]]] = {}
        for item in successes:
            by_sample.setdefault(item["sample_id"], []).append(item)
        winners: list[dict[str, Any]] = []
        overwritten: list[dict[str, Any]] = []
        for sample_id in sorted(by_sample):
            candidates = sorted(
                by_sample[sample_id], key=lambda entry: (entry["arrival_seq"], entry["source_index"])
            )
            winner = candidates[0]
            winners.append(winner)
            for extra in candidates[1:]:
                overwritten.append(
                    {
                        "sample_id": extra["sample_id"],
                        "arrival_seq": extra["arrival_seq"],
                        "batch_slot": slot,
                        "kept_arrival_seq": winner["arrival_seq"],
                    }
                )
        winners.sort(key=lambda entry: (entry["arrival_seq"], entry["sample_id"], entry["source_index"]))
        overwritten.sort(key=lambda entry: (entry["arrival_seq"], entry["sample_id"]))
        groups.append(
            {
                "batch_slot": slot,
                "members": [
                    {
                        "sample_id": item["sample_id"],
                        "arrival_seq": item["arrival_seq"],
                        "rank": rank,
                        "source_index": item["source_index"],
                    }
                    for rank, item in enumerate(winners, start=1)
                ],
                "overwritten": overwritten,
                "failed": failed,
            }
        )

    normalized = [
        {
            "sample_id": item["sample_id"],
            "arrival_seq": item["arrival_seq"],
            "batch_slot": item["batch_slot"],
            "admitted": item["admitted"],
            "reject_reason": item["reject_reason"],
        }
        for item in ordered
    ]
    digest = sha256_hex(
        FEED_DIGEST_DOMAIN
        + canonical_json_bytes({"schema": schema, "entries": normalized})
        + b"\n"
    )
    return {"feed_digest": digest, "groups": groups, "ordered": ordered}


def _feed_mismatch(generation: int, field: str, expected: Any, actual: Any) -> AuditMismatch:
    sample_id = None
    if actual is not None and isinstance(actual, str):
        sample_id = actual
    return AuditMismatch(
        "the original feed fails the recomputed validation contract",
        sample_id=sample_id,
        batch_slot=None,
        field=f"feed.{field}",
        expected=expected,
        actual=actual,
        generation=generation,
    )


def _independent_input_digest(sample_ids: Sequence[str], inputs: np.ndarray) -> str:
    ordered_ids = sorted(str(item) for item in sample_ids)
    index = {str(item): i for i, item in enumerate(sample_ids)}
    rows = np.asarray([inputs[index[item]] for item in ordered_ids], dtype="<f8")
    h = hashlib.sha256()
    h.update(INPUT_DIGEST_DOMAIN)
    h.update(canonical_json_bytes({"sample_ids": ordered_ids, "shape": list(inputs.shape)}))
    h.update(b"\n")
    h.update(_array_record("inputs", rows))
    h.update(b"\n")
    return h.hexdigest()


def _independent_model_digest(model: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(MODEL_DIGEST_DOMAIN)
    for name in MODEL_ARRAY_ORDER:
        h.update(_array_record(name, model[name]))
        h.update(b"\n")
    return h.hexdigest()


def _recompute_slot(
    *,
    slot: int,
    group: Mapping[str, Any],
    sample_ids: Sequence[str],
    inputs: np.ndarray,
    model: Mapping[str, np.ndarray],
    numbering_version: str,
    param_digest: str,
) -> dict[str, Any]:
    row_index = {str(item): index for index, item in enumerate(sample_ids)}
    if not group["members"]:
        return {
            "batch_slot": int(slot),
            "status": "empty",
            "error_code": "E_EMPTY",
            "members": [],
            "member_snapshot": [],
            "overwritten": list(group["overwritten"]),
            "failed": list(group["failed"]),
            "output_digest": None,
            "assignment_count": 0,
        }

    sample_digests: dict[str, str] = {}
    for member in group["members"]:
        sample_id = str(member["sample_id"])
        x = np.asarray(inputs[row_index[sample_id]], dtype="<f8")
        h1 = np.tanh(_independent_linear(x, model["W1"], model["b1"]))
        h2 = np.tanh(_independent_linear(h1, model["W2"], model["b2"]))
        output = np.tanh(_independent_linear(h2, model["W3"], model["b3"]))
        output = np.round(output, 12).astype("<f8")
        if output.size:
            output[output == 0.0] = 0.0
        sample_digests[sample_id] = _sample_output_digest(sample_id, output)
    batch_digest = _batch_output_digest(sample_digests)

    overwrite_counts: dict[str, int] = {}
    for item in group["overwritten"]:
        sample_id = str(item["sample_id"])
        overwrite_counts[sample_id] = overwrite_counts.get(sample_id, 0) + 1
    members_out: list[dict[str, Any]] = []
    for member in group["members"]:
        sample_id = str(member["sample_id"])
        assignment_id, assignment_digest = _expected_assignment_id(
            numbering_version=numbering_version,
            batch_slot=slot,
            rank=int(member["rank"]),
            sample_id=sample_id,
            param_digest=param_digest,
        )
        members_out.append(
            {
                "sample_id": sample_id,
                "arrival_seq": int(member["arrival_seq"]),
                "rank": int(member["rank"]),
                "assignment_id": assignment_id,
                "assignment_digest": assignment_digest,
                "sample_output_digest": sample_digests[sample_id],
                "batch_output_digest": batch_digest,
                "overwritten_count": overwrite_counts.get(sample_id, 0),
            }
        )
    return {
        "batch_slot": int(slot),
        "status": "committed",
        "error_code": None,
        "members": members_out,
        "member_snapshot": [
            {
                "sample_id": item["sample_id"],
                "arrival_seq": item["arrival_seq"],
                "rank": item["rank"],
            }
            for item in members_out
        ],
        "overwritten": list(group["overwritten"]),
        "failed": list(group["failed"]),
        "output_digest": batch_digest,
        "assignment_count": len(members_out),
    }


def _independent_linear(
    vector: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    result = np.asarray(bias, dtype="<f8").copy()
    for index in range(vector.shape[0]):
        result = result + vector[index] * weights[index]
    return result


def _sample_output_digest(sample_id: str, output: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(SAMPLE_OUTPUT_DIGEST_DOMAIN)
    h.update(
        canonical_json_bytes(
            {"sample_id": sample_id, "dtype": "<f8", "shape": list(output.shape)}
        )
    )
    h.update(b"\n")
    h.update(output.tobytes(order="C"))
    h.update(b"\n")
    return h.hexdigest()


def _batch_output_digest(sample_digests: Mapping[str, str]) -> str:
    h = hashlib.sha256()
    h.update(BATCH_OUTPUT_DIGEST_DOMAIN)
    h.update(canonical_json_bytes({"members": sorted(sample_digests.items())}))
    h.update(b"\n")
    return h.hexdigest()


def _expected_assignment_id(
    *,
    numbering_version: str,
    batch_slot: int,
    rank: int,
    sample_id: str,
    param_digest: str,
) -> tuple[str, str]:
    payload = canonical_json_bytes(
        {
            "batch_slot": int(batch_slot),
            "model_param_digest": str(param_digest),
            "numbering_version": str(numbering_version),
            "rank": int(rank),
            "sample_id": str(sample_id),
        }
    )
    digest = sha256_hex(ASSIGNMENT_DIGEST_DOMAIN + payload + b"\n")
    return f"{numbering_version}:S{int(batch_slot)}:R{int(rank):06d}:{digest[:20]}", digest


def _compare_slot(
    *,
    generation: int,
    expected: Mapping[str, Any],
    slot_row: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
) -> None:
    slot = int(expected["batch_slot"])
    if str(slot_row["status"]) != expected["status"]:
        raise AuditMismatch(
            "slot status mismatch",
            sample_id=None,
            batch_slot=slot,
            field="status",
            expected=expected["status"],
            actual=slot_row["status"],
            generation=generation,
        )
    if (slot_row["output_digest"] or None) != expected["output_digest"]:
        raise AuditMismatch(
            "slot batch output digest mismatch",
            sample_id=None,
            batch_slot=slot,
            field="output_digest",
            expected=expected["output_digest"],
            actual=slot_row["output_digest"],
            generation=generation,
        )
    actual_by_rank = {int(item["rank"]): item for item in assignments}
    expected_members = expected["members"]
    expected_ranks = {int(item["rank"]) for item in expected_members}
    actual_ranks = set(actual_by_rank)
    extra_ranks = sorted(actual_ranks - expected_ranks)
    if extra_ranks:
        extra = actual_by_rank[extra_ranks[0]]
        raise AuditMismatch(
            "ledger has an extra assignment rank",
            sample_id=str(extra["sample_id"]),
            batch_slot=slot,
            field="rank",
            expected=sorted(expected_ranks),
            actual=sorted(actual_ranks),
            generation=generation,
        )
    for item in expected_members:
        rank = int(item["rank"])
        actual = actual_by_rank.get(rank)
        if actual is None:
            raise AuditMismatch(
                "ledger assignment is missing",
                sample_id=item["sample_id"],
                batch_slot=slot,
                field="rank",
                expected=rank,
                actual=None,
                generation=generation,
            )
        field_values = (
            ("sample_id", item["sample_id"], actual["sample_id"]),
            ("arrival_seq", item["arrival_seq"], actual["arrival_seq"]),
            ("assignment_id", item["assignment_id"], actual["assignment_id"]),
            ("assignment_digest", item["assignment_digest"], actual["assignment_digest"]),
            ("sample_output_digest", item["sample_output_digest"], actual["sample_output_digest"]),
            ("batch_output_digest", item["batch_output_digest"], actual["batch_output_digest"]),
            ("overwritten_count", item["overwritten_count"], actual["overwritten_count"]),
        )
        for field, expected_value, actual_value in field_values:
            if expected_value != actual_value:
                raise AuditMismatch(
                    "assignment field mismatch",
                    sample_id=item["sample_id"],
                    batch_slot=slot,
                    field=field,
                    expected=expected_value,
                    actual=actual_value,
                    generation=generation,
                )
    if slot_row["members"] != expected["member_snapshot"]:
        raise AuditMismatch(
            "slot member sequence/rank snapshot mismatch",
            sample_id=_first_member_sample(slot_row["members"], expected["member_snapshot"]),
            batch_slot=slot,
            field="members",
            expected=expected["member_snapshot"],
            actual=slot_row["members"],
            generation=generation,
        )
    actual_overwritten = _normalize_overwritten(slot_row["overwritten"])
    expected_overwritten = _normalize_overwritten(expected["overwritten"])
    if actual_overwritten != expected_overwritten:
        raise AuditMismatch(
            "overwritten delivery snapshot mismatch",
            sample_id=_first_overwritten_sample(actual_overwritten, expected_overwritten),
            batch_slot=slot,
            field="overwritten",
            expected=expected_overwritten,
            actual=actual_overwritten,
            generation=generation,
        )
    if slot_row["failed"] != expected["failed"]:
        raise AuditMismatch(
            "failed admission snapshot mismatch",
            sample_id=_first_failed_sample(slot_row["failed"], expected["failed"]),
            batch_slot=slot,
            field="failed",
            expected=expected["failed"],
            actual=slot_row["failed"],
            generation=generation,
        )
    if int(slot_row["assignment_count"]) != int(expected["assignment_count"]):
        raise AuditMismatch(
            "assignment count mismatch",
            sample_id=None,
            batch_slot=slot,
            field="assignment_count",
            expected=expected["assignment_count"],
            actual=int(slot_row["assignment_count"]),
            generation=generation,
        )
    if (slot_row["error_code"] or None) != expected["error_code"]:
        raise AuditMismatch(
            "slot error code mismatch",
            sample_id=None,
            batch_slot=slot,
            field="error_code",
            expected=expected["error_code"],
            actual=slot_row["error_code"],
            generation=generation,
        )


def _first_member_sample(actual: Any, expected: Any) -> str | None:
    for index in range(max(len(actual or []), len(expected or []))):
        if index >= len(actual or []):
            return str(expected[index]["sample_id"])
        if index >= len(expected or []):
            return str(actual[index]["sample_id"])
        if actual[index].get("sample_id") != expected[index].get("sample_id"):
            return str(expected[index]["sample_id"])
    return None


def _first_overwritten_sample(actual: Any, expected: Any) -> str | None:
    return _first_member_sample(actual, expected)


def _first_failed_sample(actual: Any, expected: Any) -> str | None:
    return _first_member_sample(actual, expected)


def _normalize_overwritten(rows: Any) -> list[dict[str, Any]]:
    """Strip legacy file-position metadata absent from the feed digest contract."""

    normalized: list[dict[str, Any]] = []
    for row in rows or []:
        normalized.append(
            {
                "sample_id": str(row["sample_id"]),
                "arrival_seq": int(row["arrival_seq"]),
                "batch_slot": int(row["batch_slot"]),
                "kept_arrival_seq": int(row["kept_arrival_seq"]),
            }
        )
    return normalized


def _audit_digest(
    *,
    generation: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    unavailable: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "generation": generation["generation_id"],
        "feed_digest": generation["feed_digest"],
        "input_digest": generation["input_digest"],
        "param_digest": generation["param_digest"],
        "numbering_version": generation["numbering_version"],
        "slots": list(slots),
        "not_auditable_slots": list(unavailable),
    }
    return sha256_hex(AUDIT_DIGEST_DOMAIN + canonical_json_bytes(payload) + b"\n")


def _independent_ledger_content_digest(conn: sqlite3.Connection) -> str:
    domain = b"LEDGER-CONTENT-V1\n"
    generations = [
        _decode_generation(row)
        for row in conn.execute("SELECT * FROM generations ORDER BY generation_id").fetchall()
    ]
    arrays = [
        {
            "generation_id": int(row["generation_id"]),
            "name": str(row["name"]),
            "dtype": str(row["dtype"]),
            "shape": json.loads(row["shape_json"]),
            "digest": str(row["digest"]),
        }
        for row in conn.execute(
            "SELECT generation_id, name, dtype, shape_json, digest FROM arrays "
            "ORDER BY generation_id, name"
        ).fetchall()
    ]
    slots = [
        {
            "generation_id": int(row["generation_id"]),
            "batch_slot": int(row["batch_slot"]),
            "status": str(row["status"]),
            "cursor": int(row["cursor"]),
            "members": json.loads(row["members_json"]),
            "overwritten": json.loads(row["overwritten_json"]),
            "failed": json.loads(row["failed_json"]),
            "output_digest": row["output_digest"],
            "assignment_count": int(row["assignment_count"]),
            "error_code": row["error_code"],
            "updated_at": str(row["updated_at"]),
        }
        for row in conn.execute(
            "SELECT * FROM slots ORDER BY generation_id, batch_slot"
        ).fetchall()
    ]
    assignments = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM assignments ORDER BY generation_id, batch_slot, rank"
        ).fetchall()
    ]
    attempts = [
        dict(row)
        for row in conn.execute("SELECT * FROM attempts ORDER BY attempt_id").fetchall()
    ]
    payload = {
        "format_version": LEDGER_FORMAT_VERSION,
        "generations": generations,
        "arrays": arrays,
        "slots": slots,
        "assignments": assignments,
        "attempts": attempts,
    }
    return sha256_hex(domain + canonical_json_bytes(payload) + b"\n")
