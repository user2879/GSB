"""Independent sequential replay oracle and E_REPLAY difference localisation."""

from __future__ import annotations

import json
from typing import Any

from .content import deterministic_page_bytes, digest_bytes, digest_file, digest_json
from .errors import BusinessError
from .storage import Storage
from .validation import load_sample


def _ledger_record(storage: Storage, lsn: int) -> dict[str, Any] | None:
    with storage.read_connection() as conn:
        row = conn.execute(
            "SELECT batch_id, run_id, start_lsn, end_lsn, checkpoint_id, ledger_hash "
            "FROM replay_batches WHERE start_lsn <= ? AND end_lsn >= ? "
            "ORDER BY batch_id LIMIT 1",
            (lsn, lsn),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT batch_id, run_id, start_lsn, end_lsn, checkpoint_id, ledger_hash "
                "FROM replay_batches WHERE end_lsn <= ? ORDER BY end_lsn DESC LIMIT 1",
                (lsn,),
            ).fetchone()
        if row is not None:
            return {"event_type": "REPLAY_BATCH", **dict(row)}
        load_row = conn.execute(
            "SELECT event_id, event_type, cursor, event_hash FROM ledger_events "
            "WHERE event_type='LOAD' ORDER BY event_id LIMIT 1"
        ).fetchone()
        return dict(load_row) if load_row else None


def _raise_diff(
    storage: Storage,
    *,
    lsn: int,
    page_id: str | None,
    field: str,
    expected: Any,
    actual: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    context: dict[str, Any] = {
        "first_mismatch_lsn": lsn,
        "lsn": lsn,
        "page_id": page_id,
        "field": field,
        "expected": expected,
        "actual": actual,
        "ledger_record": _ledger_record(storage, lsn),
        "baseline": "independent_contiguous_single_thread_replay",
    }
    if extra:
        context.update(extra)
    raise BusinessError("E_REPLAY", f"replay mismatch at LSN {lsn}, field {field}", context)


def _expected_page_state(sample: Any, cursor: int) -> dict[str, dict[str, Any]]:
    versions = {page.page_id: page.initial_version for page in sample.pages}
    digests = {}
    page_by_id = {page.page_id: page for page in sample.pages}
    for page in sample.pages:
        base = deterministic_page_bytes(page.page_id, page.size_bytes, page.initial_version, 0)
        digests[page.page_id] = digest_bytes(base)
    for entry in sample.entries:
        if entry.lsn > cursor:
            break
        if not entry.visible:
            continue
        versions[entry.page_id] += 1
        digests[entry.page_id] = entry.generated_digest
    return {
        page_id: {
            "page_id": page_id,
            "version": versions[page_id],
            "digest": digests[page_id],
            "size_bytes": page_by_id[page_id].size_bytes,
        }
        for page_id in sorted(versions)
    }


def verify_replay(storage: Storage) -> dict[str, Any]:
    """Compare committed storage against a fresh in-memory sequential replay.

    The oracle is deliberately independent of scheduler completion order.  It
    checks every page version, disk content digest, visible transaction set,
    crash point, and checkpoint state.  The first difference is reported with
    the ledger batch that would have covered it.
    """

    if not storage.is_initialized():
        raise BusinessError("E_STATE", "workspace is not initialized")

    ledger_check = storage.verify_ledger()
    if not ledger_check.get("ok"):
        event_id = ledger_check.get("event_id") or ledger_check.get("batch_id")
        _raise_diff(
            storage,
            lsn=storage.ledger_cursor(),
            page_id=None,
            field=ledger_check.get("field", "ledger_hash"),
            expected=ledger_check.get("expected"),
            actual=ledger_check.get("actual"),
            extra={"ledger_check": ledger_check, "ledger_event_id": event_id},
        )

    sample_path = storage.inputs_dir / "sample.json"
    if not sample_path.exists():
        raise BusinessError(
            "E_REPLAY",
            "workspace input sample is missing; independent baseline unavailable",
            {"ledger_record": None, "path": str(sample_path)},
        )
    oracle = load_sample(sample_path)
    expected_active_log = storage.get_meta("active_wal_digest")
    if expected_active_log != oracle.wal_digest:
        _raise_diff(
            storage,
            lsn=0,
            page_id=None,
            field="log_digest",
            expected=oracle.wal_digest,
            actual=expected_active_log,
            extra={"path": str(sample_path)},
        )

    active_budget_digest = storage.get_meta("active_budget_digest")
    table_budget = storage.budget()
    ledger_budget, ledger_budget_source = storage.ledger_budget()
    if table_budget.as_dict() != oracle.budget.as_dict():
        _raise_diff(
            storage,
            lsn=0,
            page_id=None,
            field="budget_params",
            expected=oracle.budget.as_dict(),
            actual=table_budget.as_dict(),
        )
    if ledger_budget.as_dict() != oracle.budget.as_dict():
        _raise_diff(
            storage,
            lsn=0,
            page_id=None,
            field="ledger_budget_params",
            expected=oracle.budget.as_dict(),
            actual=ledger_budget.as_dict(),
            extra={"budget_source": ledger_budget_source},
        )
    if active_budget_digest != oracle.budget_digest:
        _raise_diff(
            storage,
            lsn=0,
            page_id=None,
            field="budget_digest",
            expected=oracle.budget_digest,
            actual=active_budget_digest,
        )

    with storage.read_connection() as conn:
        db_pages = {
            row["page_id"]: dict(row)
            for row in conn.execute(
                "SELECT page_id, size_bytes, initial_version, initial_digest FROM pages ORDER BY page_id"
            ).fetchall()
        }
    for page in sorted(oracle.pages, key=lambda item: item.page_id):
        actual = db_pages.get(page.page_id)
        if actual is None:
            _raise_diff(
                storage,
                lsn=0,
                page_id=page.page_id,
                field="page_exists",
                expected=True,
                actual=False,
            )
        if actual["size_bytes"] != page.size_bytes:
            _raise_diff(
                storage,
                lsn=0,
                page_id=page.page_id,
                field="size_bytes",
                expected=page.size_bytes,
                actual=actual["size_bytes"],
            )
        if actual["initial_version"] != page.initial_version:
            _raise_diff(
                storage,
                lsn=0,
                page_id=page.page_id,
                field="initial_version",
                expected=page.initial_version,
                actual=actual["initial_version"],
            )
        expected_base = digest_bytes(
            deterministic_page_bytes(page.page_id, page.size_bytes, page.initial_version, 0)
        )
        if actual["initial_digest"] != expected_base:
            _raise_diff(
                storage,
                lsn=0,
                page_id=page.page_id,
                field="initial_digest",
                expected=expected_base,
                actual=actual["initial_digest"],
            )
    for page_id in sorted(set(db_pages) - {page.page_id for page in oracle.pages}):
        _raise_diff(
            storage,
            lsn=0,
            page_id=page_id,
            field="page_exists",
            expected=False,
            actual=True,
        )

    expected_by_lsn = {entry.lsn: entry for entry in oracle.entries}
    with storage.read_connection() as conn:
        wal_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT lsn, txid, page_id, target_version, generated_digest, visible, "
                "commit_flag, persist_flag FROM wal_entries ORDER BY lsn"
            ).fetchall()
        ]
    actual_wal_by_lsn = {row["lsn"]: row for row in wal_rows}
    for lsn in sorted(set(actual_wal_by_lsn) | set(expected_by_lsn)):
        expected = expected_by_lsn.get(lsn)
        actual = actual_wal_by_lsn.get(lsn)
        if expected is None or actual is None:
            _raise_diff(
                storage,
                lsn=lsn,
                page_id=(expected.page_id if expected else actual["page_id"]),
                field="wal_entry_exists",
                expected=expected is not None,
                actual=actual is not None,
            )
        for field, expected_value, actual_value in (
            ("page_id", expected.page_id, actual["page_id"]),
            ("txid", expected.txid, actual["txid"]),
            ("target_version", expected.target_version, actual["target_version"]),
            ("generated_digest", expected.generated_digest, actual["generated_digest"]),
            ("visible", expected.visible, bool(actual["visible"])),
            ("commit", expected.commit, bool(actual["commit_flag"])),
            ("persist", expected.persist, bool(actual["persist_flag"])),
        ):
            if expected_value != actual_value:
                _raise_diff(
                    storage,
                    lsn=lsn,
                    page_id=expected.page_id,
                    field=f"wal_{field}",
                    expected=expected_value,
                    actual=actual_value,
                )

    with storage.read_connection() as conn:
        actual_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT page_id, version, lsn, txid, digest, content_path, visible, kind, batch_id "
                "FROM page_versions ORDER BY lsn, page_id, version"
            ).fetchall()
        ]
    actual_by_lsn = {row["lsn"]: row for row in actual_rows if row["kind"] == "wal"}
    base_rows = {row["page_id"]: row for row in actual_rows if row["kind"] == "base"}
    for page in oracle.pages:
        base = base_rows.get(page.page_id)
        if base is None:
            _raise_diff(
                storage,
                lsn=0,
                page_id=page.page_id,
                field="base_page_version",
                expected=True,
                actual=False,
            )
        try:
            base_disk_digest = digest_file(storage.path_for(base["content_path"]))
        except OSError as exc:
            _raise_diff(
                storage,
                lsn=0,
                page_id=page.page_id,
                field="base_page_file",
                expected=base["digest"],
                actual=f"missing: {exc}",
            )
        if base_disk_digest != base["digest"]:
            _raise_diff(
                storage,
                lsn=0,
                page_id=page.page_id,
                field="base_page_file_digest",
                expected=base["digest"],
                actual=base_disk_digest,
            )
    all_lsns = sorted(set(actual_by_lsn) | set(expected_by_lsn))
    for lsn in all_lsns:
        expected = expected_by_lsn.get(lsn)
        actual = actual_by_lsn.get(lsn)
        if expected is None:
            _raise_diff(
                storage,
                lsn=lsn,
                page_id=actual["page_id"],
                field="page_version_exists",
                expected=False,
                actual=True,
            )
        if actual is None:
            _raise_diff(
                storage,
                lsn=lsn,
                page_id=expected.page_id,
                field="page_version_exists",
                expected=True,
                actual=False,
            )
        checks = (
            ("page_id", expected.page_id, actual["page_id"]),
            ("txid", expected.txid, actual["txid"]),
            ("version", expected.target_version, actual["version"]),
            ("content_digest", expected.generated_digest, actual["digest"]),
            ("visible", expected.visible, bool(actual["visible"])),
        )
        for field, expected_value, actual_value in checks:
            if expected_value != actual_value:
                _raise_diff(
                    storage,
                    lsn=lsn,
                    page_id=expected.page_id,
                    field=field,
                    expected=expected_value,
                    actual=actual_value,
                )
        try:
            disk_digest = digest_file(storage.path_for(actual["content_path"]))
        except OSError as exc:
            _raise_diff(
                storage,
                lsn=lsn,
                page_id=expected.page_id,
                field="page_file",
                expected=expected.generated_digest,
                actual=f"missing: {exc}",
            )
        if disk_digest != expected.generated_digest:
            _raise_diff(
                storage,
                lsn=lsn,
                page_id=expected.page_id,
                field="page_file_digest",
                expected=expected.generated_digest,
                actual=disk_digest,
            )

    expected_transactions = list(oracle.committed_txids)
    actual_transactions = sorted(
        {row["txid"] for row in actual_rows if row["kind"] == "wal" and row["visible"]}
    )
    if sorted(expected_transactions) != actual_transactions:
        missing = sorted(set(expected_transactions) - set(actual_transactions))
        extra = sorted(set(actual_transactions) - set(expected_transactions))
        focus_tx = (missing or extra)[0]
        focus_lsn = next(
            (entry.lsn for entry in oracle.entries if entry.txid == focus_tx),
            storage.ledger_cursor(),
        )
        _raise_diff(
            storage,
            lsn=focus_lsn,
            page_id=None,
            field="committed_transactions",
            expected=sorted(expected_transactions),
            actual=actual_transactions,
            extra={"missing": missing, "unexpected": extra},
        )

    actual_crash_point = storage.get_meta_int("crash_point_lsn", 0)
    if actual_crash_point != oracle.crash_point_lsn:
        focus_lsn = min(actual_crash_point, oracle.crash_point_lsn) + 1
        _raise_diff(
            storage,
            lsn=focus_lsn,
            page_id=None,
            field="crash_point_lsn",
            expected=oracle.crash_point_lsn,
            actual=actual_crash_point,
        )

    for checkpoint in storage.checkpoint_rows():
        cursor = checkpoint["ledger_cursor"]
        expected_state = _expected_page_state(oracle, cursor)
        try:
            actual_state_payload = json.loads(checkpoint["state_json"])
        except json.JSONDecodeError:
            _raise_diff(
                storage,
                lsn=cursor,
                page_id=None,
                field="checkpoint_state_json",
                expected="valid JSON",
                actual=checkpoint["state_json"],
            )
        actual_state = {
            item["page_id"]: item for item in actual_state_payload.get("pages", [])
        }
        if actual_state_payload.get("cursor") != cursor:
            _raise_diff(
                storage,
                lsn=cursor,
                page_id=None,
                field="checkpoint_internal_cursor",
                expected=cursor,
                actual=actual_state_payload.get("cursor"),
                extra={"checkpoint_id": checkpoint["checkpoint_id"]},
            )
        computed_state_digest = digest_json(actual_state_payload)
        if computed_state_digest != checkpoint["state_digest"]:
            _raise_diff(
                storage,
                lsn=cursor,
                page_id=None,
                field="checkpoint_state_digest",
                expected=computed_state_digest,
                actual=checkpoint["state_digest"],
                extra={"checkpoint_id": checkpoint["checkpoint_id"]},
            )
        for page_id in sorted(set(expected_state) | set(actual_state)):
            expected = expected_state.get(page_id)
            actual = actual_state.get(page_id)
            if expected is None or actual is None:
                _raise_diff(
                    storage,
                    lsn=cursor,
                    page_id=page_id,
                    field="checkpoint_page_set",
                    expected=expected,
                    actual=actual,
                    extra={"checkpoint_id": checkpoint["checkpoint_id"]},
                )
            if expected["version"] != actual.get("version"):
                _raise_diff(
                    storage,
                    lsn=cursor,
                    page_id=page_id,
                    field="checkpoint_version",
                    expected=expected["version"],
                    actual=actual.get("version"),
                    extra={"checkpoint_id": checkpoint["checkpoint_id"]},
                )
            if expected["digest"] != actual.get("digest"):
                _raise_diff(
                    storage,
                    lsn=cursor,
                    page_id=page_id,
                    field="checkpoint_digest",
                    expected=expected["digest"],
                    actual=actual.get("digest"),
                    extra={"checkpoint_id": checkpoint["checkpoint_id"]},
                )

    return {
        "ok": True,
        "baseline": "independent_contiguous_single_thread_replay",
        "checked_entries": len(oracle.entries),
        "checked_page_versions": len(actual_by_lsn),
        "cursor": storage.ledger_cursor(),
        "crash_point_lsn": oracle.crash_point_lsn,
        "committed_txids": sorted(expected_transactions),
        "checkpoint_count": len(storage.checkpoint_rows()),
        "ledger": ledger_check,
    }
