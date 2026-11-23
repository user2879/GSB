"""Strict, all-or-nothing loading of page declarations and WAL entries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .content import deterministic_page_bytes, digest_bytes, digest_json
from .errors import BusinessError
from .models import PageDecl, PreparedEntry, ReplayBudget, SampleSpec, WalEntry


MIN_PAGE_SIZE = 64
MAX_PAGE_SIZE = 1024 * 1024
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _fail(code: str, message: str, **context: Any) -> None:
    raise BusinessError(code, message, context)


def _require_mapping(value: Any, label: str, code: str, **context: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code, f"{label} must be a JSON object", field=label, **context)
    return value


def _as_identifier(value: Any, field: str, code: str, **context: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        _fail(code, f"{field} must be a non-empty string or integer", field=field, **context)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            _fail(code, f"{field} must not be empty", field=field, **context)
        return normalized
    if value < 0:
        _fail(code, f"{field} integer must be non-negative", field=field, **context)
    return str(value)


def _as_int(value: Any, field: str, code: str, minimum: int | None = None, **context: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(code, f"{field} must be an integer", field=field, **context)
    if minimum is not None and value < minimum:
        _fail(code, f"{field} must be >= {minimum}", field=field, **context)
    return value


def _as_bool(value: Any, field: str, code: str, **context: Any) -> bool:
    if not isinstance(value, bool):
        _fail(code, f"{field} must be a JSON boolean", field=field, **context)
    return value


def _page_size_supported(size: int) -> bool:
    return MIN_PAGE_SIZE <= size <= MAX_PAGE_SIZE and (size & (size - 1)) == 0


def _resolve_external_file(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate)


def _read_ndjson(path: Path) -> tuple[list[Any], str]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        _fail("E_LOG", "cannot read WAL file", field="log_file", path=str(path), error=str(exc))
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail("E_LOG", "WAL file must be UTF-8", field="log_file", path=str(path), error=str(exc))
    records: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _fail(
                "E_LOG",
                "invalid JSON in WAL line",
                line=line_number,
                field="wal_line",
                error=str(exc),
            )
    return records, text


def _normalise_wal_objects(raw_entries: Iterable[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_entries, start=1):
        if not isinstance(raw, dict):
            _fail("E_LOG", "WAL entry must be an object", line=index, field="wal_entry")
        normalized.append(raw)
    return normalized


def _group_by_transaction(entries: list[WalEntry]) -> list[list[WalEntry]]:
    groups: list[list[WalEntry]] = []
    closed: set[str] = set()
    current: list[WalEntry] = []
    for entry in entries:
        if current and current[-1].txid != entry.txid:
            closed.add(current[-1].txid)
            groups.append(current)
            current = []
        if entry.txid in closed:
            _fail(
                "E_LOG",
                "transaction identifiers must form one contiguous WAL group",
                first_invalid_lsn=entry.lsn,
                lsn=entry.lsn,
                field="txid",
                txid=entry.txid,
            )
        current.append(entry)
    if current:
        groups.append(current)
    return groups


def _validate_transaction_groups(groups: list[list[WalEntry]]) -> None:
    for group_index, group in enumerate(groups):
        is_last_group = group_index == len(groups) - 1
        for entry_index, entry in enumerate(group):
            is_last_entry = entry_index == len(group) - 1
            if entry.commit and not is_last_entry:
                _fail(
                    "E_LOG",
                    "commit marker is only valid on the final entry of a transaction group",
                    first_invalid_lsn=entry.lsn,
                    lsn=entry.lsn,
                    field="commit",
                    txid=entry.txid,
                )
        final = group[-1]
        if not final.commit and not is_last_group:
            _fail(
                "E_LOG",
                "only the final transaction group may be uncommitted",
                first_invalid_lsn=final.lsn,
                lsn=final.lsn,
                field="commit",
                txid=final.txid,
            )


def _compute_crash_point(groups: list[list[WalEntry]]) -> tuple[int, tuple[str, ...]]:
    """Return the durable committed prefix endpoint and its transaction IDs.

    A transaction is durable only when it is committed and its WAL group
    contains a persist marker.  The crash point advances to the end of such a
    complete transaction group; processing stops at the first incomplete
    trailing group.  Any committed group after the last durable group is not
    part of the crash-visible prefix.
    """

    crash_point = 0
    committed: list[str] = []
    for group in groups:
        if not group[-1].commit:
            break
        if any(entry.persist for entry in group):
            crash_point = group[-1].lsn
    for group in groups:
        if group[-1].lsn <= crash_point and group[-1].commit:
            committed.append(group[0].txid)
    return crash_point, tuple(committed)


def load_sample(path: str | Path) -> SampleSpec:
    """Load and validate one sample without mutating any workspace state."""

    sample_path = Path(path)
    try:
        raw_document = json.loads(sample_path.read_text(encoding="utf-8"))
    except OSError as exc:
        _fail("E_LOG", "cannot read sample JSON", field="sample", path=str(sample_path), error=str(exc))
    except json.JSONDecodeError as exc:
        _fail("E_LOG", "sample file is not valid JSON", field="sample", path=str(sample_path), error=str(exc))

    document = _require_mapping(raw_document, "sample", "E_LOG")
    raw_pages = document.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        _fail("E_PAGE", "pages must be a non-empty array", field="pages")

    pages: list[PageDecl] = []
    page_ids: set[str] = set()
    for page_index, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, dict):
            _fail("E_PAGE", "page declaration must be an object", page_index=page_index, field="page")
        page_context = {"page_index": page_index}
        page_id = _as_identifier(raw_page.get("page_id"), "page_id", "E_PAGE", **page_context)
        page_context["page_id"] = page_id
        size_bytes = _as_int(raw_page.get("size_bytes"), "size_bytes", "E_PAGE", 1, **page_context)
        if not _page_size_supported(size_bytes):
            _fail(
                "E_PAGE",
                "page size must be a supported power of two between 64 and 1048576 bytes",
                **page_context,
                field="size_bytes",
                value=size_bytes,
            )
        initial_version = _as_int(
            raw_page.get("initial_version"), "initial_version", "E_PAGE", 0, **page_context
        )
        if page_id in page_ids:
            _fail("E_PAGE", "page identifiers must be unique", **page_context, field="page_id")
        page_ids.add(page_id)
        pages.append(PageDecl(page_id=page_id, size_bytes=size_bytes, initial_version=initial_version))

    raw_budget = document.get("budget")
    if not isinstance(raw_budget, dict):
        _fail("E_BUDGET", "budget must be a JSON object", field="budget")
    budget = ReplayBudget(
        global_slots=_as_int(raw_budget.get("global_slots"), "global_slots", "E_BUDGET", 1),
        per_tx_quota=_as_int(raw_budget.get("per_tx_quota"), "per_tx_quota", "E_BUDGET", 1),
        page_byte_budget=_as_int(
            raw_budget.get("page_byte_budget"), "page_byte_budget", "E_BUDGET", 1
        ),
        checkpoint_interval=_as_int(
            raw_budget.get("checkpoint_interval"), "checkpoint_interval", "E_BUDGET", 1
        ),
    )
    largest_page = max(page.size_bytes for page in pages)
    if budget.page_byte_budget < largest_page:
        _fail(
            "E_BUDGET",
            "page_byte_budget must be at least the largest declared page size",
            field="page_byte_budget",
            page_byte_budget=budget.page_byte_budget,
            largest_page_size=largest_page,
        )

    log_file_value = document.get("log_file")
    wal_text = ""
    if log_file_value is not None:
        if not isinstance(log_file_value, str) or not log_file_value.strip():
            _fail("E_LOG", "log_file must be a non-empty relative or absolute path", field="log_file")
        log_path = _resolve_external_file(sample_path.parent, log_file_value)
        raw_wal, wal_text = _read_ndjson(log_path)
        log_file = str(log_path)
    elif "wal" in document:
        raw_wal_value = document["wal"]
        if not isinstance(raw_wal_value, list):
            _fail("E_LOG", "embedded wal must be an array", field="wal")
        raw_wal = list(raw_wal_value)
        wal_text = "\n".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in raw_wal
        )
        if wal_text:
            wal_text += "\n"
        log_file = "<embedded:wal>"
    else:
        _fail("E_LOG", "sample must provide log_file or an embedded wal array", field="log_file")

    raw_entries = _normalise_wal_objects(raw_wal)
    entries: list[WalEntry] = []
    previous_lsn: int | None = None
    seen_lsns: set[int] = set()
    page_by_id = {page.page_id: page for page in pages}
    for line_index, raw in enumerate(raw_entries, start=1):
        if "lsn" not in raw:
            _fail(
                "E_LOG",
                "WAL entry is missing lsn",
                line=line_index,
                first_invalid_lsn=line_index,
                field="lsn",
            )
        lsn = _as_int(
            raw.get("lsn"),
            "lsn",
            "E_LOG",
            1,
            line=line_index,
            first_invalid_lsn=line_index,
        )
        expected_lsn = 1 if previous_lsn is None else previous_lsn + 1
        if lsn in seen_lsns or (previous_lsn is not None and lsn != expected_lsn):
            _fail(
                "E_ORDER",
                "WAL sequence numbers must be unique and contiguous after the first entry",
                first_invalid_lsn=lsn,
                lsn=lsn,
                expected_lsn=expected_lsn,
                field="lsn",
                line=line_index,
            )
        line_context = {"line": line_index, "first_invalid_lsn": lsn, "lsn": lsn}
        txid = _as_identifier(raw.get("txid"), "txid", "E_LOG", **line_context)
        page_id = _as_identifier(raw.get("page_id"), "page_id", "E_LOG", **line_context)
        if page_id not in page_by_id:
            _fail(
                "E_LOG",
                "WAL entry refers to an undeclared page",
                **line_context,
                field="page_id",
                page_id=page_id,
            )
        content_digest = raw.get("content_digest")
        if not isinstance(content_digest, str) or not _DIGEST_RE.fullmatch(content_digest):
            _fail(
                "E_LOG",
                "content_digest must be a 64-character hexadecimal SHA-256 value",
                **line_context,
                field="content_digest",
            )
        commit = _as_bool(raw.get("commit"), "commit", "E_LOG", **line_context)
        persist = _as_bool(raw.get("persist"), "persist", "E_LOG", **line_context)
        entries.append(
            WalEntry(
                lsn=lsn,
                txid=txid,
                page_id=page_id,
                content_digest=content_digest.lower(),
                commit=commit,
                persist=persist,
            )
        )
        seen_lsns.add(lsn)
        previous_lsn = lsn

    groups = _group_by_transaction(entries)
    _validate_transaction_groups(groups)
    crash_point, committed_txids = _compute_crash_point(groups)

    versions = {page.page_id: page.initial_version for page in pages}
    prepared: list[PreparedEntry] = []
    for entry in entries:
        versions[entry.page_id] += 1
        target_version = versions[entry.page_id]
        page = page_by_id[entry.page_id]
        generated = deterministic_page_bytes(
            entry.page_id, page.size_bytes, target_version, entry.lsn
        )
        generated_digest = digest_bytes(generated)
        if generated_digest != entry.content_digest:
            _fail(
                "E_LOG",
                "WAL content_digest does not match the deterministic page content",
                first_invalid_lsn=entry.lsn,
                lsn=entry.lsn,
                field="content_digest",
                page_id=entry.page_id,
                expected_digest=generated_digest,
                supplied_digest=entry.content_digest,
            )
        prepared.append(
            PreparedEntry(
                lsn=entry.lsn,
                txid=entry.txid,
                page_id=entry.page_id,
                content_digest=entry.content_digest,
                commit=entry.commit,
                persist=entry.persist,
                target_version=target_version,
                generated_digest=generated_digest,
                visible=entry.lsn <= crash_point,
            )
        )

    pages_payload = [
        {
            "page_id": page.page_id,
            "size_bytes": page.size_bytes,
            "initial_version": page.initial_version,
        }
        for page in sorted(pages, key=lambda item: item.page_id)
    ]
    pages_digest = digest_json(pages_payload)
    budget_digest = digest_json(budget.as_dict())
    wal_digest = digest_bytes(wal_text.encode("utf-8"))
    sample_digest = digest_json(
        {
            "pages_digest": pages_digest,
            "budget_digest": budget_digest,
            "wal_digest": wal_digest,
        }
    )

    return SampleSpec(
        pages=tuple(pages),
        budget=budget,
        entries=tuple(prepared),
        crash_point_lsn=crash_point,
        committed_txids=committed_txids,
        source_sample=str(sample_path),
        log_file=log_file,
        wal_text=wal_text,
        wal_digest=wal_digest,
        pages_digest=pages_digest,
        budget_digest=budget_digest,
        sample_digest=sample_digest,
        raw_document=document,
    )
