"""Read-once preview cache separated from authoritative recomputation."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PreviewError
from .models import AllocationResult, MinuteDecision

_PREVIEW_FIELDS = ("allocated", "rejected", "trim_reason")


@dataclass(frozen=True)
class PreviewEntry:
    minute: int
    version: int
    allocations: tuple[AllocationResult, ...]
    total_allocated: int

    def allocation_map(self) -> dict[str, AllocationResult]:
        return {item.service_id: item for item in self.allocations}


class PreviewCache:
    """In-memory cache populated by the incremental path only.

    A :meth:`query` is exactly one in-memory read; it never invokes the full
    estimator or touches the ledger write path.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int], PreviewEntry] = {}
        self._faults: dict[int, dict[str, object]] = {}

    def populate(self, decision: MinuteDecision) -> None:
        entry = PreviewEntry(
            minute=decision.minute,
            version=decision.version,
            allocations=decision.allocations,
            total_allocated=decision.total_allocated,
        )
        self._entries[(decision.version, decision.minute)] = entry

    def inject_fault(self, version: int, minute: int, service_id: str,
                     field: str, value: object) -> None:
        """Test hook: corrupt one cached scalar without touching the ledger."""

        self._faults[(version, minute)] = {
            "service_id": service_id,
            "field": field,
            "value": value,
        }

    def _effective_entry(self, version: int, minute: int) -> PreviewEntry:
        entry = self._entries.get((version, minute))
        if entry is None:
            raise KeyError((version, minute))
        fault = self._faults.get((version, minute))
        if not fault:
            return entry
        allocations = []
        for allocation in entry.allocations:
            if allocation.service_id == fault["service_id"]:
                allocation = AllocationResult(
                    **{**allocation.__dict__, fault["field"]: fault["value"]}
                )
            allocations.append(allocation)
        return PreviewEntry(
            minute=entry.minute,
            version=entry.version,
            allocations=tuple(allocations),
            total_allocated=entry.total_allocated,
        )

    def query(self, version: int, minute: int) -> PreviewEntry:
        return self._effective_entry(version, minute)

    def contains(self, version: int, minute: int) -> bool:
        return (version, minute) in self._entries

    def verify_against_authority(self, authoritative: list[MinuteDecision]) -> None:
        """Compare cached allocations/rejections/reasons; raise E_PREVIEW."""

        for decision in sorted(authoritative, key=lambda item: item.minute):
            if not self.contains(decision.version, decision.minute):
                raise PreviewError(
                    "preview missing minute present in ledger",
                    details={"minute": decision.minute, "service_id": None,
                             "field": "minute"},
                )
            preview = self.query(decision.version, decision.minute)
            preview_map = preview.allocation_map()
            for allocation in sorted(decision.allocations, key=lambda item: item.service_id):
                cached = preview_map.get(allocation.service_id)
                if cached is None:
                    raise PreviewError(
                        "preview missing service allocation",
                        details={"minute": decision.minute,
                                 "service_id": allocation.service_id,
                                 "field": "service_id"},
                    )
                for field_name in _PREVIEW_FIELDS:
                    expected = getattr(allocation, field_name)
                    actual = getattr(cached, field_name)
                    if expected != actual:
                        raise PreviewError(
                            "preview diverged from authoritative ledger",
                            details={
                                "minute": decision.minute,
                                "service_id": allocation.service_id,
                                "field": field_name,
                                "authoritative": expected,
                                "preview": actual,
                            },
                        )
