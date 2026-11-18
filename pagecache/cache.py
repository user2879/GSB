from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import EngineError, ErrorCode
from .manifest import read_manifest
from .model import CURRENT_FORMAT_VERSION, Manifest, PageData, PageEntry
from .pagefile import page_checksum, page_path, read_page_file
from .segment import read_segment_file, segment_inventory, segment_path, write_segment


class PageCache:
    """Deterministic LRU page cache backed by immutable page files.

    Eviction does not depend on dictionary traversal order: the victim is the
    minimum over the total ordering ``(last_access_clock, page_id)``. Every
    decision is also materialized in the event stream for replay comparison.
    """

    def __init__(
        self,
        *,
        manifest: Manifest | str | Path,
        pages_dir: str | Path,
        overflow_dir: str | Path,
        budget_bytes: int,
        page_size: int,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            manifest = read_manifest(manifest)
        self.manifest = manifest
        self.pages_dir = Path(pages_dir)
        self.overflow_dir = Path(overflow_dir)
        self.overflow_dir.mkdir(parents=True, exist_ok=True)
        self.budget_bytes = int(budget_bytes)
        self.page_size = int(page_size)
        if self.page_size <= 0:
            raise EngineError(
                ErrorCode.E_BUDGET,
                "page size must be a positive integer",
                path="$cache.page_size",
                details={"page_size": self.page_size},
            )
        if self.budget_bytes <= 0:
            raise EngineError(
                ErrorCode.E_BUDGET,
                "memory budget is zero or negative",
                path="$cache.budget_bytes",
                details={"budget_bytes": self.budget_bytes, "basis": "memory_budget_zero"},
            )
        self.capacity = self.budget_bytes // self.page_size
        if self.capacity <= 0:
            raise EngineError(
                ErrorCode.E_BUDGET,
                "memory budget cannot hold one declared page",
                path="$cache.budget_bytes",
                details={
                    "budget_bytes": self.budget_bytes,
                    "page_size": self.page_size,
                    "capacity_frames": self.capacity,
                    "basis": "budget_below_page_size",
                },
            )
        if manifest.header_version != CURRENT_FORMAT_VERSION or any(
            entry.record_version != CURRENT_FORMAT_VERSION for entry in manifest.records
        ):
            raise EngineError(
                ErrorCode.E_FORMAT,
                "page cache requires an upgraded v3 manifest",
                path="$cache.manifest.record_version",
                details={
                    "header_version": manifest.header_version,
                    "record_versions": list(manifest.format_versions),
                    "required_version": CURRENT_FORMAT_VERSION,
                },
            )
        self._entries: dict[str, PageEntry] = {entry.page_id: entry for entry in manifest.records}
        if len(self._entries) != len(manifest.records):
            raise EngineError(ErrorCode.E_FORMAT, "manifest contains duplicate page identifiers", path="$cache.manifest")

        self._resident: dict[str, PageData] = {}
        self._last_access: dict[str, int] = {}
        self._overflow: dict[str, str] = {}
        for entry in sorted(manifest.records, key=lambda item: item.page_id):
            if entry.overflow_ref is not None:
                self._overflow[entry.page_id] = entry.overflow_ref
        self._access_clock = 0
        self._swap_in_count = 0
        self._page_load_count = 0
        self._eviction_count = 0
        self._overflow_sequence: list[str] = []
        self._swap_in_sequence: list[str] = []
        self._events: list[dict[str, Any]] = []

    def get_page(self, page_id: str) -> PageData:
        entry = self._entries.get(page_id)
        if entry is None:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "page is not declared in the manifest",
                path="$cache.page_id",
                page_id=page_id,
            )
        self._access_clock += 1
        access = self._access_clock
        if page_id in self._resident:
            self._last_access[page_id] = access
            return self._resident[page_id]

        self._evict_until_room(incoming_page_id=page_id, access=access)
        if page_id in self._overflow:
            page = self._swap_in(entry, access)
        else:
            page = self._load_from_page_file(entry, access)
        self._resident[page_id] = page
        self._last_access[page_id] = access
        return page

    def _evict_until_room(self, *, incoming_page_id: str, access: int) -> None:
        while len(self._resident) >= self.capacity:
            victim = sorted(
                self._resident,
                key=lambda candidate: (self._last_access.get(candidate, -1), candidate),
            )[0]
            self._evict(victim, incoming_page_id=incoming_page_id, access=access)

    def _evict(self, page_id: str, *, incoming_page_id: str, access: int) -> None:
        entry = self._entries[page_id]
        page = self._resident.pop(page_id)
        self._last_access.pop(page_id, None)
        ref = entry.overflow_ref or f"{page_id}.seg"
        target = segment_path(self.overflow_dir, ref, page_id)
        write_segment(
            target,
            page_id=page_id,
            row_count=entry.row_count,
            payload=page.raw_bytes,
            compressed=entry.compressed,
            version=CURRENT_FORMAT_VERSION,
        )
        self._overflow[page_id] = target.name
        self._eviction_count += 1
        self._overflow_sequence.append(page_id)
        self._events.append(
            {
                "event_sequence": len(self._events) + 1,
                "access_sequence": access,
                "type": "evict",
                "page_id": page_id,
                "victim_page_id": page_id,
                "incoming_page_id": incoming_page_id,
                "overflow_ref": target.name,
                "compressed": entry.compressed,
            }
        )

    def _swap_in(self, entry: PageEntry, access: int) -> PageData:
        ref = self._overflow[entry.page_id]
        target = segment_path(self.overflow_dir, ref, entry.page_id)
        if not target.exists():
            raise EngineError(
                ErrorCode.E_INTEGRITY,
                "overflow segment required for swap-in is missing",
                path=str(target),
                page_id=entry.page_id,
                details={"overflow_ref": ref, "basis": "missing_overflow_segment"},
            )
        segment = read_segment_file(target, expected_page_id=entry.page_id)
        if segment.compressed != entry.compressed:
            raise EngineError(
                ErrorCode.E_PAYLOAD,
                "manifest compression flag does not match the segment payload format",
                path=f"$cache.records[{entry.page_id}].compressed",
                page_id=entry.page_id,
                details={
                    "manifest_compressed": entry.compressed,
                    "segment_compressed": segment.compressed,
                },
            )
        actual_checksum = page_checksum(segment.payload)
        if entry.payload_checksum != actual_checksum:
            raise EngineError(
                ErrorCode.E_INTEGRITY,
                "swapped-in payload checksum does not match the manifest",
                path=f"$cache.records[{entry.page_id}].payload_checksum",
                page_id=entry.page_id,
                details={"expected": entry.payload_checksum, "actual": actual_checksum},
            )
        if segment.row_count != entry.row_count:
            raise EngineError(
                ErrorCode.E_INTEGRITY,
                "swapped-in segment row count does not match the manifest",
                path=f"$cache.records[{entry.page_id}].row_count",
                page_id=entry.page_id,
                details={"manifest_rows": entry.row_count, "segment_rows": segment.row_count},
            )
        self._overflow.pop(entry.page_id, None)
        self._swap_in_count += 1
        self._swap_in_sequence.append(entry.page_id)
        self._events.append(
            {
                "event_sequence": len(self._events) + 1,
                "access_sequence": access,
                "type": "swap_in",
                "page_id": entry.page_id,
                "incoming_page_id": entry.page_id,
                "overflow_ref": target.name,
                "compressed": segment.compressed,
            }
        )
        return segment.page()

    def _load_from_page_file(self, entry: PageEntry, access: int) -> PageData:
        target = page_path(self.pages_dir, entry.page_id)
        page = read_page_file(target, expected_page_id=entry.page_id)
        actual_checksum = page_checksum(page.raw_bytes)
        if entry.payload_checksum != actual_checksum:
            raise EngineError(
                ErrorCode.E_INTEGRITY,
                "page file checksum does not match the manifest",
                path=f"$cache.records[{entry.page_id}].payload_checksum",
                page_id=entry.page_id,
                details={"expected": entry.payload_checksum, "actual": actual_checksum},
            )
        if page.row_count != entry.row_count:
            raise EngineError(
                ErrorCode.E_INTEGRITY,
                "page row count does not match the manifest",
                path=f"$cache.records[{entry.page_id}].row_count",
                page_id=entry.page_id,
                details={"manifest_rows": entry.row_count, "page_rows": page.row_count},
            )
        self._page_load_count += 1
        self._events.append(
            {
                "event_sequence": len(self._events) + 1,
                "access_sequence": access,
                "type": "load",
                "page_id": entry.page_id,
                "incoming_page_id": entry.page_id,
            }
        )
        return page

    def state(self, *, include_events: bool = True) -> dict[str, Any]:
        resident = [
            {
                "page_id": page_id,
                "raw_sha256": hashlib.sha256(self._resident[page_id].raw_bytes).hexdigest(),
            }
            for page_id in sorted(self._resident)
        ]
        overflow = []
        for page_id in sorted(self._overflow):
            entry = self._entries[page_id]
            overflow.append(
                {
                    "page_id": page_id,
                    "overflow_ref": self._overflow[page_id],
                    "compressed": entry.compressed,
                }
            )
        result: dict[str, Any] = {
            "capacity_frames": self.capacity,
            "budget_bytes": self.budget_bytes,
            "page_size": self.page_size,
            "access_clock": self._access_clock,
            "resident_pages": resident,
            "overflow_pages": overflow,
            "last_access": [
                {"page_id": page_id, "clock": self._last_access[page_id]} for page_id in sorted(self._last_access)
            ],
            "overflow_sequence": list(self._overflow_sequence),
            "swap_in_sequence": list(self._swap_in_sequence),
            "swap_in_count": self._swap_in_count,
            "page_load_count": self._page_load_count,
            "eviction_count": self._eviction_count,
            "events": list(self._events) if include_events else [],
        }
        return result

    def restore_state(self, state: dict[str, Any]) -> None:
        """Wholesale replacement of in-memory cache state from a snapshot."""

        try:
            budget_bytes = int(state["budget_bytes"])
            page_size = int(state["page_size"])
            capacity = int(state["capacity_frames"])
            access_clock = int(state["access_clock"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(ErrorCode.E_REPLAY, "cache snapshot header is malformed", path="$snapshot.cache") from exc
        if budget_bytes != self.budget_bytes or page_size != self.page_size or capacity != self.capacity:
            raise EngineError(
                ErrorCode.E_REPLAY,
                "cache budget semantics changed; snapshot cannot be replayed",
                path="$snapshot.cache.budget_bytes",
                details={
                    "snapshot_budget_bytes": budget_bytes,
                    "current_budget_bytes": self.budget_bytes,
                    "snapshot_page_size": page_size,
                    "current_page_size": self.page_size,
                    "basis": "cache_configuration_changed",
                },
            )
        resident_rows = state.get("resident_pages", [])
        overflow_rows = state.get("overflow_pages", [])
        last_access_rows = state.get("last_access", [])
        resident_pages: dict[str, PageData] = {}
        resident_ids: set[str] = set()
        for index, row in enumerate(resident_rows):
            try:
                page_id = str(row["page_id"])
                expected_sha = str(row["raw_sha256"])
            except (KeyError, TypeError) as exc:
                raise EngineError(ErrorCode.E_REPLAY, "resident page row is malformed", path=f"$snapshot.cache.resident_pages[{index}]") from exc
            entry = self._entries.get(page_id)
            if entry is None:
                raise EngineError(ErrorCode.E_REPLAY, "resident page is not in the current manifest", page_id=page_id, path=f"$snapshot.cache.resident_pages[{index}]")
            page = read_page_file(page_path(self.pages_dir, page_id), expected_page_id=page_id)
            actual_sha = hashlib.sha256(page.raw_bytes).hexdigest()
            if actual_sha != expected_sha:
                raise EngineError(
                    ErrorCode.E_REPLAY,
                    "resident page bytes differ from the snapshot",
                    page_id=page_id,
                    path=f"$snapshot.cache.resident_pages[{index}].raw_sha256",
                    details={"expected": expected_sha, "actual": actual_sha, "field": "raw_sha256"},
                )
            resident_pages[page_id] = page
            resident_ids.add(page_id)

        overflow_pages: dict[str, str] = {}
        for index, row in enumerate(overflow_rows):
            try:
                page_id = str(row["page_id"])
                ref = str(row["overflow_ref"])
            except (KeyError, TypeError) as exc:
                raise EngineError(ErrorCode.E_REPLAY, "overflow page row is malformed", path=f"$snapshot.cache.overflow_pages[{index}]") from exc
            if page_id in resident_ids:
                raise EngineError(ErrorCode.E_REPLAY, "page is both resident and overflowed", page_id=page_id, path="$snapshot.cache")
            if page_id not in self._entries:
                raise EngineError(ErrorCode.E_REPLAY, "overflow page is not in the current manifest", page_id=page_id, path=f"$snapshot.cache.overflow_pages[{index}]")
            overflow_pages[page_id] = ref

        last_access: dict[str, int] = {}
        for index, row in enumerate(last_access_rows):
            try:
                page_id = str(row["page_id"])
                clock = int(row["clock"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EngineError(ErrorCode.E_REPLAY, "last-access row is malformed", path=f"$snapshot.cache.last_access[{index}]") from exc
            if page_id not in resident_ids:
                raise EngineError(ErrorCode.E_REPLAY, "last-access entry is not resident", page_id=page_id, path=f"$snapshot.cache.last_access[{index}]")
            last_access[page_id] = clock
        if len(last_access) != len(resident_ids):
            raise EngineError(ErrorCode.E_REPLAY, "resident pages and last-access clocks differ", path="$snapshot.cache.last_access")

        self._resident = resident_pages
        self._overflow = overflow_pages
        self._last_access = last_access
        self._access_clock = access_clock
        try:
            self._swap_in_count = int(state["swap_in_count"])
            self._swap_in_sequence = [str(value) for value in state.get("swap_in_sequence", [])]
            self._overflow_sequence = [str(value) for value in state.get("overflow_sequence", [])]
            self._page_load_count = int(state.get("page_load_count", 0))
            self._eviction_count = int(state.get("eviction_count", 0))
            self._events = [dict(value) for value in state.get("events", [])]
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(ErrorCode.E_REPLAY, "cache counters are malformed", path="$snapshot.cache") from exc

    def disk_inventory(self) -> dict[str, Any]:
        manifest_fingerprint = hashlib.sha256(
            canonical_json([entry.to_dict() for entry in self.manifest.records]).encode("utf-8")
        ).hexdigest()
        return {
            "manifest_versions": list(self.manifest.format_versions),
            "manifest_header_version": self.manifest.header_version,
            "manifest_sha256": manifest_fingerprint,
            "segments": segment_inventory(self.overflow_dir),
        }

    def state_digest(self) -> str:
        return hashlib.sha256(canonical_json(self.state()).encode("utf-8")).hexdigest()
