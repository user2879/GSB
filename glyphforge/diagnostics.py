"""Versioned immutable diagnostic snapshots and read-only view state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .application import ChangeReport, GlyphApplication, RenderKey
from .atlas import AtlasRegion, PackOutcome
from .errors import snapshot_stale
from .raster import RasterResult, bitmap_hash
from .scheduler import ThreadBudgetReport


@dataclass(frozen=True)
class SnapshotAtlasPage:
    index: int
    width: int
    height: int
    generation: int
    pixels: bytes
    bitmap_hash: str
    regions: tuple[AtlasRegion, ...]
    free_rectangles: tuple[tuple[int, int, int, int], ...]
    occupied_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "generation": self.generation,
            "bitmap_hash": self.bitmap_hash,
            "occupied_pixels": self.occupied_pixels,
            "regions": [region.to_dict() for region in self.regions],
            "free_rectangles": [
                {"x": x, "y": y, "width": width, "height": height}
                for x, y, width, height in self.free_rectangles
            ],
        }


@dataclass(frozen=True)
class DiagnosticSnapshot:
    version: int
    glyph_ids: tuple[str, ...]
    active_sizes: tuple[tuple[str, tuple[int, ...]], ...]
    fill_rules: tuple[tuple[str, str], ...]
    bitmaps: tuple[tuple[RenderKey, RasterResult], ...]
    glyph_hashes: tuple[tuple[str, str], ...]
    atlas_pages: tuple[SnapshotAtlasPage, ...]
    atlas_regions: tuple[AtlasRegion, ...]
    pack_outcomes: tuple[PackOutcome, ...]
    pending_keys: tuple[str, ...]
    thread_budget: ThreadBudgetReport | None
    degrade_decisions_json: str
    last_change_json: str
    change_closure: tuple[str, ...]

    @property
    def bitmap_map(self) -> Mapping[RenderKey, RasterResult]:
        return MappingProxyType(dict(self.bitmaps))

    def bitmap_for(self, key: RenderKey) -> RasterResult:
        try:
            return dict(self.bitmaps)[key]
        except KeyError as exc:
            raise snapshot_stale(
                "snapshot does not contain the requested bitmap", key=key.display()
            ) from exc

    def to_dict(self, *, include_pixels: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": self.version,
            "glyph_ids": list(self.glyph_ids),
            "active_sizes": [
                {"glyph_id": glyph_id, "sizes": list(sizes)}
                for glyph_id, sizes in self.active_sizes
            ],
            "fill_rules": [
                {"glyph_id": glyph_id, "fill_rule": fill_rule}
                for glyph_id, fill_rule in self.fill_rules
            ],
            "bitmaps": [
                {"key": key.to_dict(), **bitmap.to_dict()}
                for key, bitmap in self.bitmaps
            ],
            "glyph_hashes": [
                {"key": key, "bitmap_hash": content_hash}
                for key, content_hash in self.glyph_hashes
            ],
            "atlas_pages": [page.to_dict() for page in self.atlas_pages],
            "atlas_regions": [region.to_dict() for region in self.atlas_regions],
            "pack_outcomes": [outcome.to_dict() for outcome in self.pack_outcomes],
            "pending_keys": list(self.pending_keys),
            "thread_budget": self.thread_budget.to_dict()
            if self.thread_budget
            else None,
            "degrade_decisions": json.loads(self.degrade_decisions_json),
            "last_change": json.loads(self.last_change_json),
            "change_closure": list(self.change_closure),
        }
        if include_pixels:
            import base64

            for bitmap in result["bitmaps"]:
                bitmap["pixels_base64"] = base64.b64encode(
                    next(
                        item.pixels
                        for key, item in self.bitmaps
                        if key.display()
                        == f"{bitmap['key']['glyph_id']}|{bitmap['key']['size']}|{bitmap['key']['fill_rule']}"
                    )
                ).decode("ascii")
            for page in result["atlas_pages"]:
                source = next(
                    item for item in self.atlas_pages if item.index == page["index"]
                )
                page["pixels_base64"] = base64.b64encode(source.pixels).decode("ascii")
        return result


class DiagnosticService:
    """Builds full snapshots from application state; never mutates or reruns it."""

    def __init__(self, application: GlyphApplication) -> None:
        self.application = application

    @property
    def current_version(self) -> int:
        return self.application.version

    def snapshot(self) -> DiagnosticSnapshot:
        app = self.application
        active = app.active_keys()
        bitmaps = tuple(
            (key, app.bitmap_cache[key]) for key in active if key in app.bitmap_cache
        )
        glyph_hashes = tuple(
            (
                key.display(),
                app.bitmap_cache[key].content_hash if key in app.bitmap_cache else "",
            )
            for key in active
        )
        manager = app.atlas_budget_manager.manager
        pages: list[SnapshotAtlasPage] = []
        for page in sorted(manager.pages, key=lambda item: item.index):
            pages.append(
                SnapshotAtlasPage(
                    index=page.index,
                    width=page.width,
                    height=page.height,
                    generation=page.generation,
                    pixels=page.bitmap_bytes(),
                    bitmap_hash=bitmap_hash(page.width, page.height, page.bitmap),
                    regions=tuple(page.regions[key] for key in sorted(page.regions)),
                    free_rectangles=tuple(
                        sorted(
                            (rect.x, rect.y, rect.width, rect.height)
                            for rect in page.free_rects
                        )
                    ),
                    occupied_pixels=page.occupied_pixels,
                )
            )
        last_report: ChangeReport | None = app.reports[-1] if app.reports else None
        return DiagnosticSnapshot(
            version=app.version,
            glyph_ids=tuple(sorted(app.glyphs)),
            active_sizes=tuple(
                (glyph_id, app.active_sizes[glyph_id])
                for glyph_id in sorted(app.glyphs)
            ),
            fill_rules=tuple(
                (glyph_id, app.fill_rules[glyph_id]) for glyph_id in sorted(app.glyphs)
            ),
            bitmaps=bitmaps,
            glyph_hashes=glyph_hashes,
            atlas_pages=tuple(pages),
            atlas_regions=tuple(
                manager.regions[key] for key in sorted(manager.regions)
            ),
            pack_outcomes=app.atlas_budget_manager.outcomes,
            pending_keys=app.atlas_budget_manager.pending_keys,
            thread_budget=app.last_thread_report,
            degrade_decisions_json=json.dumps(
                list(app.atlas_budget_manager.decisions),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            last_change_json=json.dumps(
                last_report.to_dict() if last_report is not None else {},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            change_closure=last_report.recomputed_glyph_ids
            if last_report is not None
            else (),
        )

    def require_current(self, snapshot: DiagnosticSnapshot) -> None:
        if snapshot.version != self.application.version:
            raise snapshot_stale(
                "diagnostic snapshot is stale",
                snapshot_version=snapshot.version,
                current_version=self.application.version,
            )


@dataclass
class DiagnosticViewState:
    """Small adapter enforcing whole-snapshot replacement semantics."""

    current_version: int | None = None

    def apply(self, snapshot: DiagnosticSnapshot) -> bool:
        if not isinstance(snapshot, DiagnosticSnapshot):
            raise snapshot_stale("view accepts only DiagnosticSnapshot instances")
        if self.current_version is not None and snapshot.version < self.current_version:
            raise snapshot_stale(
                "refusing to replace a newer snapshot with an older version",
                snapshot_version=snapshot.version,
                current_version=self.current_version,
            )
        if self.current_version == snapshot.version:
            return False
        self.current_version = snapshot.version
        return True
