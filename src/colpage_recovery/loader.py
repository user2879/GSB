from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .errors import CommitMarkerConflictError, RangeOutOfBoundsError, WalValidationError
from .models import Budget, Bundle, ColumnSpec, PageSpec, SUPPORTED_DTYPES, WalEntry


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_bool_marker(value: Any, *, lsn: int | None, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1) and _is_int(value):
        return bool(value)
    raise WalValidationError(
        f"{field_name} must be boolean or 0/1",
        field=field_name,
        lsn=lsn,
        details={"actual_type": type(value).__name__},
    )


def load_json_file(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise WalValidationError(f"JSON file not found: {path}", field=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise WalValidationError(
            f"invalid JSON in {path.name}: {exc.msg}",
            field=f"{path.name}:{exc.lineno}:{exc.colno}",
        ) from exc


def parse_null_bitmap(raw: Any, row_count: int) -> tuple[bool, ...]:
    if raw is None:
        return tuple(False for _ in range(row_count))
    if isinstance(raw, list):
        if len(raw) != row_count:
            raise WalValidationError(
                "null_bitmap length must equal row_count",
                field="null_bitmap",
                details={"expected": row_count, "actual": len(raw)},
            )
        try:
            return tuple(bool(value) for value in raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise WalValidationError("null_bitmap must contain booleans", field="null_bitmap") from exc
    if isinstance(raw, str):
        try:
            packed = bytes.fromhex(raw.removeprefix("0x"))
        except ValueError as exc:
            raise WalValidationError("null_bitmap hex is invalid", field="null_bitmap") from exc
        bits: list[bool] = []
        for byte in packed:
            for bit in range(8):
                bits.append(bool((byte >> (7 - bit)) & 1))
        if len(bits) < row_count:
            raise WalValidationError("null_bitmap hex has too few bits", field="null_bitmap")
        return tuple(bits[:row_count])
    raise WalValidationError("null_bitmap must be a list or hex string", field="null_bitmap")


def parse_page(raw: Any, index: int) -> PageSpec:
    where = f"pages[{index}]"
    if not isinstance(raw, dict):
        raise WalValidationError("page declaration must be an object", field=where)
    page_id = raw.get("page_id") or raw.get("id")
    if not isinstance(page_id, str) or not page_id:
        raise WalValidationError("page_id must be a non-empty string", field=f"{where}.page_id")
    row_count = raw.get("row_count")
    if not _is_int(row_count) or row_count <= 0:
        raise WalValidationError(
            "row_count must be a positive integer",
            field=f"{where}.row_count",
            details={"page_id": page_id, "actual": row_count},
        )
    raw_columns = raw.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise WalValidationError(
            "columns must be a non-empty list", field=f"{where}.columns"
        )
    columns: list[ColumnSpec] = []
    seen_names: set[str] = set()
    for col_index, raw_col in enumerate(raw_columns):
        col_field = f"{where}.columns[{col_index}]"
        if not isinstance(raw_col, dict):
            raise WalValidationError("column must be an object", field=col_field)
        name = raw_col.get("name")
        dtype = raw_col.get("dtype") or raw_col.get("type")
        if not isinstance(name, str) or not name:
            raise WalValidationError("column name must be non-empty string", field=f"{col_field}.name")
        if name in seen_names:
            raise WalValidationError("duplicate column name", field=f"{col_field}.name")
        if dtype not in SUPPORTED_DTYPES:
            raise WalValidationError(
                f"unsupported dtype {dtype!r}",
                field=f"{col_field}.dtype",
                details={"supported": sorted(SUPPORTED_DTYPES)},
            )
        seen_names.add(name)
        columns.append(ColumnSpec(name=name, dtype=dtype))
    null_bitmap = parse_null_bitmap(raw.get("null_bitmap"), row_count)
    return PageSpec(
        page_id=page_id,
        row_count=row_count,
        columns=tuple(columns),
        null_bitmap=null_bitmap,
    )


def parse_pages_payload(payload: Any) -> list[PageSpec]:
    if isinstance(payload, dict):
        raw_pages = payload.get("pages")
    else:
        raw_pages = payload
    if not isinstance(raw_pages, list) or not raw_pages:
        raise WalValidationError("pages must be a non-empty list", field="pages")
    pages = [parse_page(raw, index) for index, raw in enumerate(raw_pages)]
    seen: set[str] = set()
    for page in pages:
        if page.page_id in seen:
            raise WalValidationError(
                "duplicate page_id", field="page_id", details={"page_id": page.page_id}
            )
        seen.add(page.page_id)
    return pages


def _raw_lsn(raw: dict[str, Any]) -> int | None:
    lsn = raw.get("lsn", raw.get("log_sequence_number", raw.get("sequence")))
    return lsn if _is_int(lsn) else None


def parse_wal_entry(raw: Any, page_map: dict[str, PageSpec]) -> WalEntry:
    if not isinstance(raw, dict):
        raise WalValidationError("WAL entry must be an object", field="wal_entry")
    lsn_value = raw.get("lsn", raw.get("log_sequence_number", raw.get("sequence")))
    lsn = lsn_value if _is_int(lsn_value) else None
    if lsn is None or lsn <= 0:
        raise WalValidationError(
            "lsn must be a positive integer", field="lsn", lsn=lsn
        )
    tx_id = raw.get("tx_id", raw.get("transaction_id"))
    if not isinstance(tx_id, (str, int)) or tx_id == "":
        raise WalValidationError("tx_id must be non-empty", field="tx_id", lsn=lsn)
    tx_id = str(tx_id)
    page_id = raw.get("page_id")
    if not isinstance(page_id, str) or page_id not in page_map:
        raise WalValidationError(
            "page_id must reference a declared page",
            field="page_id",
            lsn=lsn,
            details={"page_id": page_id, "known_pages": sorted(page_map)},
        )
    if "row_range" in raw:
        row_range = raw["row_range"]
        if not isinstance(row_range, (list, tuple)) or len(row_range) != 2:
            raise WalValidationError(
                "row_range must be [start, end)", field="row_range", lsn=lsn
            )
        row_start, row_end = row_range
    else:
        row_start = raw.get("row_start")
        row_end = raw.get("row_end", raw.get("row_index"))
        if "row_index" in raw and "row_end" not in raw:
            # A scalar row_index updates exactly one row.
            row_end = row_start = raw["row_index"]
            row_end = row_start + 1
    if not _is_int(row_start) or not _is_int(row_end):
        raise WalValidationError(
            "row_start and row_end must be integers",
            field="row_range",
            lsn=lsn,
        )
    page = page_map[page_id]
    if row_start < 0 or row_end <= row_start or row_end > page.row_count:
        raise RangeOutOfBoundsError(
            "row range is outside page or empty",
            field="row_range",
            lsn=lsn,
            details={
                "page_id": page_id,
                "row_count": page.row_count,
                "row_start": row_start,
                "row_end": row_end,
            },
        )
    seed = raw.get("new_value_seed", raw.get("seed"))
    if not _is_int(seed):
        raise WalValidationError(
            "new_value_seed must be an integer", field="new_value_seed", lsn=lsn
        )
    commit_flag = _as_bool_marker(
        raw.get("commit_flag", raw.get("committed", raw.get("commit_marker"))),
        lsn=lsn,
        field_name="commit_flag",
    )
    persist_flag = _as_bool_marker(
        raw.get("persist_flag", raw.get("persisted", raw.get("persistence_marker"))),
        lsn=lsn,
        field_name="persist_flag",
    )
    return WalEntry(
        lsn=lsn,
        tx_id=tx_id,
        page_id=page_id,
        row_start=row_start,
        row_end=row_end,
        seed=seed,
        commit_flag=commit_flag,
        persist_flag=persist_flag,
    )


def validate_transaction_rules(entries: list[WalEntry], protocol: str) -> None:
    commit_by_tx: dict[str, tuple[bool, int]] = {}
    for entry in entries:
        previous = commit_by_tx.get(entry.tx_id)
        if previous is not None and previous[0] != entry.commit_flag:
            raise CommitMarkerConflictError(
                f"transaction {entry.tx_id} has conflicting commit markers",
                field="commit_flag",
                lsn=entry.lsn,
                details={
                    "tx_id": entry.tx_id,
                    "first_lsn": previous[1],
                    "first_commit_flag": previous[0],
                    "current_commit_flag": entry.commit_flag,
                },
            )
        if previous is None:
            commit_by_tx[entry.tx_id] = (entry.commit_flag, entry.lsn)
        if protocol == "wal-strict" and not entry.commit_flag and entry.persist_flag:
            raise WalValidationError(
                "persist_flag=true for uncommitted entry is illegal under wal-strict",
                field="persist_flag",
                lsn=entry.lsn,
                details={"allowed_protocol": "steal-no-force permits dirty page persistence"},
            )

    finished_tx: set[str] = set()
    active_tx: str | None = None
    for entry in entries:
        if active_tx is not None and entry.tx_id == active_tx:
            continue
        if entry.tx_id in finished_tx:
            raise WalValidationError(
                "transaction log entries must form one contiguous group",
                field="tx_id",
                lsn=entry.lsn,
                details={"tx_id": entry.tx_id},
            )
        if active_tx is not None:
            finished_tx.add(active_tx)
        active_tx = entry.tx_id


def parse_budget(payload: Any) -> Budget:
    if payload is None:
        return Budget()
    if not isinstance(payload, dict):
        raise WalValidationError("budget must be an object", field="budget")
    raw = payload.get("recovery_budget", payload)
    if not isinstance(raw, dict):
        raise WalValidationError("recovery_budget must be an object", field="recovery_budget")

    def positive_int(name: str, default: int) -> int:
        value = raw.get(name, default)
        if not _is_int(value) or value <= 0:
            raise WalValidationError(
                f"budget.{name} must be a positive integer", field=name, details={"actual": value}
            )
        return value

    budget = Budget(
        global_slots=positive_int("global_parallel_slots", 2),
        per_tx_quota=positive_int("per_transaction_replay_quota", 1),
        page_byte_budget=positive_int("single_page_replay_byte_budget", 65536),
        checkpoint_interval=positive_int("checkpoint_interval_entries", 3),
        ledger_batch_entries=positive_int("ledger_batch_entries", 2),
        marker_protocol=raw.get("marker_protocol", "steal-no-force"),
        work_delay_ms=raw.get("work_delay_ms", 0),
    )
    if budget.marker_protocol not in {"steal-no-force", "wal-strict"}:
        raise WalValidationError(
            "marker_protocol must be steal-no-force or wal-strict",
            field="marker_protocol",
        )
    if not _is_int(budget.work_delay_ms) or budget.work_delay_ms < 0:
        raise WalValidationError("work_delay_ms must be non-negative", field="work_delay_ms")
    return budget


def build_bundle(
    pages_payload: Any,
    wal_payload: Any,
    budget_payload: Any | None = None,
) -> Bundle:
    pages = parse_pages_payload(pages_payload)
    page_map = {page.page_id: page for page in pages}
    if isinstance(wal_payload, dict):
        raw_entries = wal_payload.get("entries", wal_payload.get("wal", []))
    else:
        raw_entries = wal_payload
    if not isinstance(raw_entries, list):
        raise WalValidationError("WAL entries must be a list", field="wal")

    seen_lsn: set[int] = set()
    normalized_raw: list[Any] = []
    for raw_entry in raw_entries:
        lsn = _raw_lsn(raw_entry) if isinstance(raw_entry, dict) else None
        if lsn is not None:
            if lsn in seen_lsn:
                raise WalValidationError(
                    "duplicate lsn", field="lsn", lsn=lsn
                )
            seen_lsn.add(lsn)
        normalized_raw.append(raw_entry)

    entries = [parse_wal_entry(raw, page_map) for raw in normalized_raw]
    entries.sort(key=lambda entry: entry.lsn)
    lsns = [entry.lsn for entry in entries]
    if len(lsns) != len(set(lsns)):  # defensive; checked above.
        raise WalValidationError("duplicate lsn", field="lsn")
    budget = parse_budget(budget_payload)
    validate_transaction_rules(entries, budget.marker_protocol)
    return Bundle(
        pages=pages,
        entries=entries,
        budget=budget,
        pages_payload=pages_payload if isinstance(pages_payload, dict) else {"pages": pages_payload},
        wal_payload=wal_payload if isinstance(wal_payload, dict) else {"entries": wal_payload},
        budget_payload=budget_payload if isinstance(budget_payload, dict) else {},
    )


def load_bundle(sample_dir: str | Path) -> Bundle:
    sample_dir = Path(sample_dir)
    pages_payload = load_json_file(sample_dir / "pages.json")
    wal_payload = load_json_file(sample_dir / "wal.json")
    budget_path = sample_dir / "budget.json"
    budget_payload = load_json_file(budget_path) if budget_path.exists() else None
    return build_bundle(pages_payload, wal_payload, budget_payload)


def inject_payload_fault(
    pages_payload: dict[str, Any],
    wal_payload: dict[str, Any],
    injection: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return mutated copies for explicit diagnostic fault injection."""
    pages = copy.deepcopy(pages_payload)
    wal = copy.deepcopy(wal_payload)
    if not injection or injection == "none":
        return pages, wal
    entries = wal.get("entries", wal if isinstance(wal, list) else [])
    if injection == "commit-conflict":
        tx_positions: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            tx_positions.setdefault(str(entry.get("tx_id")), []).append(entry)
        for group in tx_positions.values():
            if len(group) >= 2:
                target = group[1]
                target["commit_flag"] = not bool(target.get("commit_flag", target.get("committed")))
                return pages, wal
        if entries:
            entries[0]["commit_flag"] = not bool(entries[0].get("commit_flag", True))
        return pages, wal
    if injection == "range-overflow":
        if not entries:
            raise WalValidationError("cannot inject range-overflow into empty WAL", field="wal")
        target = entries[0]
        page_id = target.get("page_id")
        page = next((item for item in pages.get("pages", []) if item.get("page_id") == page_id), None)
        row_count = int(page["row_count"]) if page else 1
        if "row_range" in target:
            target["row_range"] = [0, row_count + 7]
        else:
            target["row_start"] = 0
            target["row_end"] = row_count + 7
        return pages, wal
    raise WalValidationError(f"unknown injection {injection!r}", field="inject")
