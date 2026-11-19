"""Fixed-size grayscale atlas pages, allocation and reproducible degradation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import atlas_full, budget_denied
from .raster import RasterResult, bitmap_hash


@dataclass(frozen=True)
class AtlasAsset:
    key: str
    width: int
    height: int
    pixels: bytes
    content_hash: str
    size: int
    fill_rule: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise budget_denied("atlas asset key must be a non-empty string")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.width, self.height, self.size)
        ):
            raise budget_denied(
                "atlas asset dimensions and size must be integers", key=self.key
            )
        if self.width <= 0 or self.height <= 0:
            raise budget_denied("atlas asset dimensions must be positive")
        if len(self.pixels) != self.width * self.height:
            raise budget_denied(
                "atlas asset pixel byte length does not match dimensions",
                key=self.key,
                expected=self.width * self.height,
                actual=len(self.pixels),
            )
        actual_hash = bitmap_hash(self.width, self.height, self.pixels)
        if self.content_hash != actual_hash:
            raise budget_denied(
                "atlas asset content hash mismatch",
                key=self.key,
                expected=self.content_hash,
                actual=actual_hash,
            )

    @classmethod
    def from_raster(cls, result: RasterResult, *, key: str | None = None) -> AtlasAsset:
        return cls(
            key=key or f"{result.glyph_id}|{result.size}|{result.fill_rule}",
            width=result.width,
            height=result.height,
            pixels=result.pixels,
            content_hash=result.content_hash,
            size=result.size,
            fill_rule=result.fill_rule,
        )

    @property
    def bitmap(self) -> np.ndarray:
        array = np.frombuffer(self.pixels, dtype=np.uint8).reshape(
            (self.height, self.width)
        )
        array.setflags(write=False)
        return array

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "width": self.width,
            "height": self.height,
            "content_hash": self.content_hash,
            "size": self.size,
            "fill_rule": self.fill_rule,
            "pixels": self.width * self.height,
        }


@dataclass(frozen=True)
class AtlasBudget:
    page_size: int = 512
    max_pages: int = 2
    max_occupied_pixels: int = 2 * 512 * 512
    padding: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.page_size, bool)
            or not isinstance(self.page_size, int)
            or self.page_size <= 0
        ):
            raise budget_denied("atlas page_size must be a positive integer")
        if (
            isinstance(self.max_pages, bool)
            or not isinstance(self.max_pages, int)
            or self.max_pages < 0
        ):
            raise budget_denied("atlas max_pages must be a non-negative integer")
        if (
            isinstance(self.max_occupied_pixels, bool)
            or not isinstance(self.max_occupied_pixels, int)
            or self.max_occupied_pixels < 0
        ):
            raise budget_denied(
                "atlas max_occupied_pixels must be a non-negative integer"
            )
        if (
            isinstance(self.padding, bool)
            or not isinstance(self.padding, int)
            or self.padding < 0
        ):
            raise budget_denied("atlas padding must be a non-negative integer")


@dataclass(frozen=True, order=True)
class AtlasRegion:
    key: str
    page_index: int
    x: int
    y: int
    width: int
    height: int
    content_hash: str
    source_size: int
    fill_rule: str
    slot_id: str
    generation: int
    reused_slot: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "page_index": self.page_index,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "content_hash": self.content_hash,
            "source_size": self.source_size,
            "fill_rule": self.fill_rule,
            "slot_id": self.slot_id,
            "generation": self.generation,
            "reused_slot": self.reused_slot,
        }


@dataclass(order=True)
class _FreeRect:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class _Page:
    index: int
    width: int
    height: int
    generation: int = 0
    bitmap: np.ndarray = field(init=False)
    regions: dict[str, AtlasRegion] = field(default_factory=dict)
    free_rects: list[_FreeRect] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bitmap = np.zeros((self.height, self.width), dtype=np.uint8)
        self.free_rects = [_FreeRect(0, 0, self.width, self.height)]

    @property
    def occupied_pixels(self) -> int:
        return sum(region.width * region.height for region in self.regions.values())

    def bitmap_bytes(self) -> bytes:
        return self.bitmap.tobytes(order="C")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "generation": self.generation,
            "occupied_pixels": self.occupied_pixels,
            "bitmap_hash": bitmap_hash(self.width, self.height, self.bitmap),
            "regions": [self.regions[key].to_dict() for key in sorted(self.regions)],
            "free_rectangles": [
                rect.to_dict()
                for rect in sorted(
                    self.free_rects, key=lambda r: (r.y, r.x, r.width, r.height)
                )
            ],
        }


def _rects_mergeable(a: _FreeRect, b: _FreeRect) -> tuple[bool, _FreeRect | None]:
    if a.x == b.x and a.width == b.width:
        if a.y + a.height == b.y:
            return True, _FreeRect(a.x, a.y, a.width, a.height + b.height)
        if b.y + b.height == a.y:
            return True, _FreeRect(b.x, b.y, b.width, b.height + a.height)
    if a.y == b.y and a.height == b.height:
        if a.x + a.width == b.x:
            return True, _FreeRect(a.x, a.y, a.width + b.width, a.height)
        if b.x + b.width == a.x:
            return True, _FreeRect(b.x, b.y, b.width + a.width, b.height)
    return False, None


def _coalesce(rectangles: Iterable[_FreeRect]) -> list[_FreeRect]:
    result = sorted(
        rectangles, key=lambda rect: (rect.y, rect.x, rect.width, rect.height)
    )
    changed = True
    while changed:
        changed = False
        for left_index in range(len(result)):
            for right_index in range(left_index + 1, len(result)):
                mergeable, merged = _rects_mergeable(
                    result[left_index], result[right_index]
                )
                if mergeable and merged is not None:
                    result = [
                        rect
                        for index, rect in enumerate(result)
                        if index not in (left_index, right_index)
                    ]
                    result.append(merged)
                    result.sort(
                        key=lambda rect: (rect.y, rect.x, rect.width, rect.height)
                    )
                    changed = True
                    break
            if changed:
                break
    return result


class AtlasManager:
    """Deterministic best-short-side-fit allocator over fixed-size pages."""

    def __init__(self, budget: AtlasBudget | None = None) -> None:
        self.budget = budget or AtlasBudget()
        self.pages: list[_Page] = []
        self.assets: dict[str, AtlasAsset] = {}
        self.regions: dict[str, AtlasRegion] = {}
        self.recycle_count = 0
        self.slot_reuse_count = 0
        self._next_page_index = 0

    def _page_by_index(self, index: int) -> _Page:
        for page in self.pages:
            if page.index == index:
                return page
        raise atlas_full("atlas page index is unavailable", page_index=index)

    @property
    def occupied_pixels(self) -> int:
        return sum(page.occupied_pixels for page in self.pages)

    def _find_allocation(
        self, width: int, height: int
    ) -> tuple[_Page, _FreeRect] | None:
        best: tuple[tuple[int, int, int, int, int], _Page, _FreeRect] | None = None
        for page in self.pages:
            for rect in page.free_rects:
                if rect.width < width or rect.height < height:
                    continue
                leftover_x = rect.width - width
                leftover_y = rect.height - height
                score = (
                    min(leftover_x, leftover_y),
                    max(leftover_x, leftover_y),
                    page.index,
                    rect.y,
                    rect.x,
                )
                if best is None or score < best[0]:
                    best = (score, page, rect)
        return None if best is None else (best[1], best[2])

    def _split_free_rect(
        self, page: _Page, rect: _FreeRect, width: int, height: int
    ) -> None:
        page.free_rects.remove(rect)
        padding = self.budget.padding
        if rect.width > width + padding:
            page.free_rects.append(
                _FreeRect(
                    rect.x + width + padding,
                    rect.y,
                    rect.width - width - padding,
                    height,
                )
            )
        if rect.height > height + padding:
            page.free_rects.append(
                _FreeRect(
                    rect.x,
                    rect.y + height + padding,
                    rect.width,
                    rect.height - height - padding,
                )
            )
        page.free_rects = _coalesce(page.free_rects)

    def _new_page(self) -> _Page:
        if len(self.pages) >= self.budget.max_pages:
            raise atlas_full(
                "atlas page budget exhausted",
                pages=len(self.pages),
                max_pages=self.budget.max_pages,
            )
        if not self.pages:
            self._next_page_index = 0
        page = _Page(
            index=self._next_page_index,
            width=self.budget.page_size,
            height=self.budget.page_size,
        )
        self._next_page_index += 1
        self.pages.append(page)
        return page

    def pack(self, asset: AtlasAsset) -> AtlasRegion:
        if not isinstance(asset, AtlasAsset):
            raise budget_denied("AtlasManager.pack requires an AtlasAsset")
        if asset.width > self.budget.page_size or asset.height > self.budget.page_size:
            raise atlas_full(
                "asset is larger than an atlas page",
                key=asset.key,
                width=asset.width,
                height=asset.height,
                page_size=self.budget.page_size,
            )
        existing = self.regions.get(asset.key)
        if existing is not None:
            if existing.width == asset.width and existing.height == asset.height:
                page = self._page_by_index(existing.page_index)
                page.bitmap[
                    existing.y : existing.y + asset.height,
                    existing.x : existing.x + asset.width,
                ] = asset.bitmap
                page.generation += 1
                region = AtlasRegion(
                    key=asset.key,
                    page_index=existing.page_index,
                    x=existing.x,
                    y=existing.y,
                    width=asset.width,
                    height=asset.height,
                    content_hash=asset.content_hash,
                    source_size=asset.size,
                    fill_rule=asset.fill_rule,
                    slot_id=existing.slot_id,
                    generation=page.generation,
                    reused_slot=True,
                )
                page.regions[asset.key] = region
                self.regions[asset.key] = region
                self.assets[asset.key] = asset
                self.slot_reuse_count += 1
                return region
            self.remove(asset.key)

        new_area = asset.width * asset.height
        if self.occupied_pixels + new_area > self.budget.max_occupied_pixels:
            raise atlas_full(
                "atlas occupied-pixel budget exhausted",
                key=asset.key,
                occupied_pixels=self.occupied_pixels,
                requested_pixels=new_area,
                max_occupied_pixels=self.budget.max_occupied_pixels,
            )

        found = self._find_allocation(asset.width, asset.height)
        if found is None:
            page = self._new_page()
            rect = page.free_rects[0]
        else:
            page, rect = found

        self._split_free_rect(page, rect, asset.width, asset.height)
        page.bitmap[rect.y : rect.y + asset.height, rect.x : rect.x + asset.width] = (
            asset.bitmap
        )
        page.generation += 1
        region = AtlasRegion(
            key=asset.key,
            page_index=page.index,
            x=rect.x,
            y=rect.y,
            width=asset.width,
            height=asset.height,
            content_hash=asset.content_hash,
            source_size=asset.size,
            fill_rule=asset.fill_rule,
            slot_id=f"p{page.index}:x{rect.x}:y{rect.y}",
            generation=page.generation,
            reused_slot=False,
        )
        page.regions[asset.key] = region
        self.regions[asset.key] = region
        self.assets[asset.key] = asset
        return region

    def remove(self, key: str) -> AtlasRegion | None:
        region = self.regions.pop(key, None)
        self.assets.pop(key, None)
        if region is None:
            return None
        page = self._page_by_index(region.page_index)
        page.bitmap[
            region.y : region.y + region.height, region.x : region.x + region.width
        ] = 0
        page.regions.pop(key, None)
        page.free_rects.append(
            _FreeRect(region.x, region.y, region.width, region.height)
        )
        page.free_rects = _coalesce(page.free_rects)
        page.generation += 1
        return region

    def recycle_unreferenced(self, referenced_keys: Iterable[str]) -> tuple[int, ...]:
        references = set(referenced_keys)
        recycled: list[int] = []
        for page in self.pages:
            if not page.regions or not any(key in references for key in page.regions):
                recycled.append(page.index)
        if not recycled:
            return ()
        recycled_set = set(recycled)
        removed_keys: list[str] = []
        for page in self.pages:
            if page.index in recycled_set:
                removed_keys.extend(page.regions)
        self.pages = [page for page in self.pages if page.index not in recycled_set]
        for key in removed_keys:
            self.regions.pop(key, None)
            self.assets.pop(key, None)
        if not self.pages:
            self._next_page_index = 0
        self.recycle_count += len(recycled)
        return tuple(sorted(recycled))

    def clear(self) -> None:
        self.pages.clear()
        self.assets.clear()
        self.regions.clear()

    def page_pixels(self, page_index: int) -> bytes:
        return self._page_by_index(page_index).bitmap_bytes()

    def validate_region(self, key: str) -> None:
        region = self.regions[key]
        asset = self.assets[key]
        page = self._page_by_index(region.page_index)
        patch = page.bitmap[
            region.y : region.y + region.height, region.x : region.x + region.width
        ]
        if bitmap_hash(region.width, region.height, patch) != region.content_hash:
            raise atlas_full(
                "atlas region differs from its recorded bitmap hash",
                key=key,
                region_hash=region.content_hash,
                page_hash=bitmap_hash(region.width, region.height, patch),
            )
        if asset.content_hash != region.content_hash:
            raise atlas_full("atlas asset mapping hash mismatch", key=key)

    def ledger(self) -> dict[str, Any]:
        return {
            "budget": {
                "page_size": self.budget.page_size,
                "max_pages": self.budget.max_pages,
                "max_occupied_pixels": self.budget.max_occupied_pixels,
                "padding": self.budget.padding,
            },
            "occupied_pixels": self.occupied_pixels,
            "recycle_count": self.recycle_count,
            "slot_reuse_count": self.slot_reuse_count,
            "pages": [
                page.to_dict()
                for page in sorted(self.pages, key=lambda item: item.index)
            ],
            "mappings": [self.regions[key].to_dict() for key in sorted(self.regions)],
        }


@dataclass(frozen=True)
class PackOutcome:
    request_key: str
    status: str
    requested_size: int
    packed_size: int | None
    region: AtlasRegion | None
    reason: str
    sequence: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_key": self.request_key,
            "status": self.status,
            "requested_size": self.requested_size,
            "packed_size": self.packed_size,
            "reason": self.reason,
            "sequence": self.sequence,
            "region": self.region.to_dict() if self.region is not None else None,
        }


class AtlasBudgetManager:
    """Reproducible atlas fallback pipeline.

    Order on pressure is fixed: recycle fully unreferenced pages, retry the
    requested asset, try lower-size fallback assets in descending size order,
    and finally delay packing.  Every decision gets a monotonic sequence number.
    """

    def __init__(self, manager: AtlasManager | None = None) -> None:
        self.manager = manager or AtlasManager()
        self._outcomes: dict[str, PackOutcome] = {}
        self._pending: dict[str, tuple[AtlasAsset, tuple[AtlasAsset, ...]]] = {}
        self._decisions: list[dict[str, Any]] = []
        self._sequence = 0

    def _record(
        self, action: str, *, request_key: str, reason: str, **details: Any
    ) -> None:
        self._sequence += 1
        self._decisions.append(
            {
                "sequence": self._sequence,
                "action": action,
                "request_key": request_key,
                "reason": reason,
                **details,
            }
        )

    def _invalidate_recycled_outcomes(
        self,
        pages_before: Mapping[int, tuple[str, ...]],
        recycled: Iterable[int],
    ) -> None:
        removed_keys = {
            key for page_index in recycled for key in pages_before.get(page_index, ())
        }
        for key in sorted(removed_keys):
            old = self._outcomes.get(key)
            if old is None:
                continue
            self._sequence += 1
            self._outcomes[key] = PackOutcome(
                request_key=key,
                status="delayed",
                requested_size=old.requested_size,
                packed_size=None,
                region=None,
                reason="page_recycled",
                sequence=self._sequence,
            )

    def _finish(
        self,
        *,
        request_key: str,
        status: str,
        requested_size: int,
        region: AtlasRegion | None,
        reason: str,
    ) -> PackOutcome:
        self._sequence += 1
        outcome = PackOutcome(
            request_key=request_key,
            status=status,
            requested_size=requested_size,
            packed_size=region.source_size if region is not None else None,
            region=region,
            reason=reason,
            sequence=self._sequence,
        )
        self._outcomes[request_key] = outcome
        return outcome

    def publish(
        self,
        request_key: str,
        primary: AtlasAsset,
        *,
        fallbacks: Sequence[AtlasAsset] = (),
        referenced_keys: Iterable[str] | None = None,
    ) -> PackOutcome:
        if primary.key != request_key:
            raise budget_denied("primary asset key must equal request key")
        pending_candidates = (primary, tuple(fallbacks))
        self._pending.pop(request_key, None)
        try:
            region = self.manager.pack(primary)
            return self._finish(
                request_key=request_key,
                status="packed",
                requested_size=primary.size,
                region=region,
                reason="direct_fit",
            )
        except Exception as exc:
            from .errors import ErrorCode, GlyphForgeError

            if (
                not isinstance(exc, GlyphForgeError)
                or exc.code is not ErrorCode.ATLAS_FULL
            ):
                raise

        if referenced_keys is not None:
            pages_before = {
                page.index: tuple(sorted(page.regions)) for page in self.manager.pages
            }
            recycled = self.manager.recycle_unreferenced(referenced_keys)
            if recycled:
                self._invalidate_recycled_outcomes(pages_before, recycled)
                for page_index in recycled:
                    self._record(
                        "recycle_page",
                        request_key=request_key,
                        reason="atlas_pressure",
                        page_index=page_index,
                    )
                try:
                    region = self.manager.pack(primary)
                    return self._finish(
                        request_key=request_key,
                        status="packed",
                        requested_size=primary.size,
                        region=region,
                        reason="after_page_recycle",
                    )
                except Exception as exc:
                    from .errors import ErrorCode, GlyphForgeError

                    if (
                        not isinstance(exc, GlyphForgeError)
                        or exc.code is not ErrorCode.ATLAS_FULL
                    ):
                        raise

        ordered_fallbacks = sorted(
            [
                asset
                for asset in fallbacks
                if asset.size < primary.size and asset.key != primary.key
            ],
            key=lambda asset: (-asset.size, asset.key),
        )
        for fallback in ordered_fallbacks:
            if (
                fallback.width > self.manager.budget.page_size
                or fallback.height > self.manager.budget.page_size
            ):
                continue
            try:
                # Keep the requested key as the public mapping.  The region
                # records the actual lower size and the fallback bitmap hash.
                fallback_under_request = AtlasAsset(
                    key=request_key,
                    width=fallback.width,
                    height=fallback.height,
                    pixels=fallback.pixels,
                    content_hash=fallback.content_hash,
                    size=fallback.size,
                    fill_rule=fallback.fill_rule,
                )
                region = self.manager.pack(fallback_under_request)
            except Exception as exc:
                from .errors import ErrorCode, GlyphForgeError

                if (
                    not isinstance(exc, GlyphForgeError)
                    or exc.code is not ErrorCode.ATLAS_FULL
                ):
                    raise
                continue
            self._record(
                "fallback_size",
                request_key=request_key,
                reason="atlas_pressure",
                from_size=primary.size,
                to_size=fallback.size,
            )
            return self._finish(
                request_key=request_key,
                status="fallback",
                requested_size=primary.size,
                region=region,
                reason="lower_size_tier",
            )

        self._record(
            "delay_pack",
            request_key=request_key,
            reason="atlas_pressure",
            requested_size=primary.size,
        )
        self._pending[request_key] = pending_candidates
        return self._finish(
            request_key=request_key,
            status="delayed",
            requested_size=primary.size,
            region=None,
            reason="atlas_pressure",
        )

    def recycle(self, referenced_keys: Iterable[str]) -> tuple[int, ...]:
        """Recycle fully unreferenced pages and invalidate those outcomes."""

        before = {
            page.index: tuple(sorted(page.regions)) for page in self.manager.pages
        }
        recycled = self.manager.recycle_unreferenced(referenced_keys)
        if not recycled:
            return ()
        self._invalidate_recycled_outcomes(before, recycled)
        for page_index in recycled:
            self._record(
                "recycle_page",
                request_key="*",
                reason="explicit_recycle",
                page_index=page_index,
            )
        return recycled

    @property
    def pending_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending))

    def retry_pending(
        self,
        *,
        referenced_keys: Iterable[str] | None = None,
    ) -> tuple[PackOutcome, ...]:
        results: list[PackOutcome] = []
        for key in sorted(self._pending):
            primary, fallbacks = self._pending[key]
            results.append(
                self.publish(
                    key,
                    primary,
                    fallbacks=fallbacks,
                    referenced_keys=referenced_keys,
                )
            )
        return tuple(results)

    def remove(self, key: str) -> AtlasRegion | None:
        self._pending.pop(key, None)
        self._outcomes.pop(key, None)
        return self.manager.remove(key)

    @property
    def outcomes(self) -> tuple[PackOutcome, ...]:
        return tuple(self._outcomes[key] for key in sorted(self._outcomes))

    @property
    def decisions(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._decisions)

    def ledger(self) -> dict[str, Any]:
        return {
            "atlas": self.manager.ledger(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "pending_keys": list(self.pending_keys),
            "degrade_decisions": list(self.decisions),
        }
