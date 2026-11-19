"""Incremental editing session tying rasterization and atlas publication together."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .atlas import (
    AtlasAsset,
    AtlasBudget,
    AtlasBudgetManager,
    AtlasManager,
    AtlasRegion,
    PackOutcome,
)
from .errors import bitmap_hash_error, budget_denied, task_key_error
from .model import SUPPORTED_SIZES, Glyph, Point, validate_glyphs
from .raster import RasterConfig, RasterResult, normalize_fill_rule
from .scheduler import (
    RasterScheduler,
    RasterTask,
    ThreadBudgetReport,
)


@dataclass(frozen=True, order=True)
class RenderKey:
    glyph_id: str
    size: int
    fill_rule: str

    def __post_init__(self) -> None:
        if not isinstance(self.glyph_id, str) or not self.glyph_id:
            raise task_key_error("render key glyph id must be non-empty")
        if self.size not in SUPPORTED_SIZES:
            raise task_key_error(
                "render key size is not a supported tier",
                size=self.size,
                allowed=list(SUPPORTED_SIZES),
            )
        object.__setattr__(self, "fill_rule", normalize_fill_rule(self.fill_rule))

    def display(self) -> str:
        return f"{self.glyph_id}|{self.size}|{self.fill_rule}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "glyph_id": self.glyph_id,
            "size": self.size,
            "fill_rule": self.fill_rule,
        }


@dataclass(frozen=True)
class ChangeReport:
    step: str
    version: int
    requested_keys: tuple[RenderKey, ...]
    recomputed_keys: tuple[RenderKey, ...]
    recomputed_glyph_ids: tuple[str, ...]
    removed_keys: tuple[RenderKey, ...]
    atlas_modified_keys: tuple[str, ...]
    fallback_keys: tuple[RenderKey, ...]
    delayed_keys: tuple[RenderKey, ...]
    bitmap_hashes: tuple[tuple[RenderKey, str], ...]
    unchanged_bitmap_hashes: tuple[tuple[RenderKey, str], ...]
    thread_budget: ThreadBudgetReport | None
    atlas_decisions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "version": self.version,
            "requested_keys": [key.to_dict() for key in self.requested_keys],
            "recomputed_keys": [key.to_dict() for key in self.recomputed_keys],
            "recomputed_glyph_ids": list(self.recomputed_glyph_ids),
            "change_closure": list(self.recomputed_glyph_ids),
            "removed_keys": [key.to_dict() for key in self.removed_keys],
            "atlas_modified_keys": list(self.atlas_modified_keys),
            "fallback_keys": [key.to_dict() for key in self.fallback_keys],
            "delayed_keys": [key.to_dict() for key in self.delayed_keys],
            "bitmap_hashes": [
                {"key": key.to_dict(), "bitmap_hash": content_hash}
                for key, content_hash in self.bitmap_hashes
            ],
            "unchanged_bitmap_hashes": [
                {"key": key.to_dict(), "bitmap_hash": content_hash}
                for key, content_hash in self.unchanged_bitmap_hashes
            ],
            "thread_budget": self.thread_budget.to_dict()
            if self.thread_budget
            else None,
            "atlas_decisions": list(self.atlas_decisions),
        }


def _normalize_sizes(value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        sizes = (value,)
    else:
        sizes = tuple(value)
    if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
        raise budget_denied("active sizes must be integers")
    invalid = [size for size in sizes if size not in SUPPORTED_SIZES]
    if invalid:
        raise budget_denied(
            "active sizes must use supported pixel tiers",
            sizes=invalid,
            allowed=list(SUPPORTED_SIZES),
        )
    if len(set(sizes)) != len(sizes):
        raise budget_denied("active sizes must be unique")
    return tuple(sorted(sizes))


class GlyphApplication:
    """Incremental application service.

    A report's ``recomputed_glyph_ids`` is derived exclusively from raster
    results returned by the scheduler.  Atlas repacks without rasterization are
    visible in ``atlas_modified_keys`` but never inflate the change closure.
    """

    def __init__(
        self,
        glyphs: Mapping[str, Glyph] | Iterable[Glyph],
        *,
        active_sizes: Mapping[str, int | Sequence[int]] | None = None,
        fill_rules: Mapping[str, str] | None = None,
        atlas_budget: AtlasBudget | None = None,
        atlas_manager: AtlasManager | None = None,
        worker_budget: int = 4,
        memory_budget_bytes: int = 8 * 1024 * 1024,
        raster_config: RasterConfig | None = None,
    ) -> None:
        self._glyphs = validate_glyphs(glyphs)
        if (
            isinstance(worker_budget, bool)
            or not isinstance(worker_budget, int)
            or worker_budget < 1
        ):
            raise budget_denied("worker_budget must be a positive integer")
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0
        ):
            raise budget_denied("memory_budget_bytes must be a positive integer")
        self._worker_budget = worker_budget
        self._memory_budget_bytes = memory_budget_bytes
        self._raster_config = raster_config or RasterConfig()
        self._scheduler = RasterScheduler(self._raster_config)

        if atlas_manager is not None:
            self._atlas_budget_manager = AtlasBudgetManager(atlas_manager)
        else:
            self._atlas_budget_manager = AtlasBudgetManager(AtlasManager(atlas_budget))

        self._fill_rules: dict[str, str] = {}
        for glyph_id in self._glyphs:
            raw_rule = (fill_rules or {}).get(glyph_id, "nonzero")
            self._fill_rules[glyph_id] = normalize_fill_rule(raw_rule)

        self._active_sizes: dict[str, tuple[int, ...]] = {}
        for glyph_id, glyph in self._glyphs.items():
            if active_sizes is not None and glyph_id in active_sizes:
                sizes = _normalize_sizes(active_sizes[glyph_id])
            else:
                sizes = (glyph.suggested_sizes[0],)
            self._active_sizes[glyph_id] = sizes

        self._revisions = {glyph_id: 1 for glyph_id in self._glyphs}
        self._cache: dict[RenderKey, RasterResult] = {}
        self._cache_revision: dict[RenderKey, int] = {}
        self._version = 0
        self._reports: list[ChangeReport] = []
        self._last_thread_report: ThreadBudgetReport | None = None

    @property
    def version(self) -> int:
        return self._version

    @property
    def glyphs(self) -> Mapping[str, Glyph]:
        return MappingProxyType(self._glyphs)

    @property
    def active_sizes(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType(self._active_sizes)

    @property
    def fill_rules(self) -> Mapping[str, str]:
        return MappingProxyType(self._fill_rules)

    @property
    def bitmap_cache(self) -> Mapping[RenderKey, RasterResult]:
        return MappingProxyType(self._cache)

    @property
    def atlas_budget_manager(self) -> AtlasBudgetManager:
        return self._atlas_budget_manager

    @property
    def reports(self) -> tuple[ChangeReport, ...]:
        return tuple(self._reports)

    @property
    def last_thread_report(self) -> ThreadBudgetReport | None:
        return self._last_thread_report

    def active_keys(self, glyph_id: str | None = None) -> tuple[RenderKey, ...]:
        glyph_ids = sorted(self._glyphs) if glyph_id is None else [glyph_id]
        keys: list[RenderKey] = []
        for current_id in glyph_ids:
            if current_id not in self._glyphs:
                raise task_key_error("unknown glyph id", glyph_id=current_id)
            for size in self._active_sizes[current_id]:
                keys.append(RenderKey(current_id, size, self._fill_rules[current_id]))
        return tuple(sorted(keys))

    def bitmap_hashes(self, glyph_id: str | None = None) -> dict[str, str]:
        result: dict[str, str] = {}
        for key in self.active_keys(glyph_id):
            result[key.display()] = (
                self._cache[key].content_hash if key in self._cache else ""
            )
        return dict(sorted(result.items()))

    def region_for(self, key: RenderKey) -> AtlasRegion | None:
        return self._atlas_budget_manager.manager.regions.get(key.display())

    def _fallback_assets(
        self,
        key: RenderKey,
        cache: Mapping[RenderKey, RasterResult],
        cache_revisions: Mapping[RenderKey, int] | None = None,
    ) -> tuple[AtlasAsset, ...]:
        current_revision = self._revisions[key.glyph_id]
        revisions = (
            cache_revisions if cache_revisions is not None else self._cache_revision
        )
        candidates: list[RasterResult] = []
        for size in sorted(SUPPORTED_SIZES, reverse=True):
            if size >= key.size:
                continue
            candidate_key = RenderKey(key.glyph_id, size, key.fill_rule)
            result = cache.get(candidate_key)
            if result is not None and revisions.get(candidate_key) == current_revision:
                candidates.append(result)
        # Coordinator renames each fallback to the public request key.
        return tuple(AtlasAsset.from_raster(result) for result in candidates)

    def _publish_order(self, keys: Iterable[RenderKey]) -> tuple[RenderKey, ...]:
        return tuple(
            sorted(keys, key=lambda key: (-key.size, key.glyph_id, key.fill_rule))
        )

    def _baseline_regions(
        self,
        keys: Iterable[RenderKey],
    ) -> dict[RenderKey, tuple[int, int, int, int, int, str, bytes]]:
        baseline: dict[RenderKey, tuple[int, int, int, int, int, str, bytes]] = {}
        manager = self._atlas_budget_manager.manager
        for key in keys:
            region = manager.regions.get(key.display())
            if region is None:
                continue
            page = manager._page_by_index(region.page_index)
            patch = page.bitmap[
                region.y : region.y + region.height,
                region.x : region.x + region.width,
            ].tobytes(order="C")
            baseline[key] = (
                region.page_index,
                region.x,
                region.y,
                region.width,
                region.height,
                region.content_hash,
                patch,
            )
        return baseline

    def _verify_baseline(
        self,
        baseline: Mapping[RenderKey, tuple[int, int, int, int, int, str, bytes]],
    ) -> dict[RenderKey, str]:
        manager = self._atlas_budget_manager.manager
        unchanged: dict[RenderKey, str] = {}
        for key, record in baseline.items():
            page_index, x, y, width, height, content_hash, pixels = record
            region = manager.regions.get(key.display())
            if region is None:
                raise bitmap_hash_error(
                    "unaffected atlas region disappeared",
                    key=key.display(),
                    expected_hash=content_hash,
                )
            actual_tuple = (
                region.page_index,
                region.x,
                region.y,
                region.width,
                region.height,
                region.content_hash,
            )
            if actual_tuple != (page_index, x, y, width, height, content_hash):
                raise bitmap_hash_error(
                    "unaffected atlas region metadata changed",
                    key=key.display(),
                    expected=record[:6],
                    actual=list(actual_tuple),
                )
            page = manager._page_by_index(region.page_index)
            actual_pixels = page.bitmap[
                region.y : region.y + region.height,
                region.x : region.x + region.width,
            ].tobytes(order="C")
            if actual_pixels != pixels:
                raise bitmap_hash_error(
                    "unaffected atlas region pixels changed",
                    key=key.display(),
                    expected_hash=content_hash,
                )
            cached = self._cache.get(key)
            # Fallback publication can legitimately use a lower-size bitmap; a
            # primary cached result's hash must still match only when that
            # primary bitmap was the published content.
            if (
                cached is not None
                and cached.content_hash != content_hash
                and cached.size == height
                and cached.width == width
            ):
                raise bitmap_hash_error(
                    "unaffected cached bitmap hash changed",
                    key=key.display(),
                    expected_hash=cached.content_hash,
                    actual_hash=content_hash,
                )
            unchanged[key] = content_hash
        return unchanged

    def _report_without_raster(
        self,
        *,
        step: str,
        removed_keys: Iterable[RenderKey] = (),
        atlas_modified_keys: Iterable[str] = (),
        unchanged: Mapping[RenderKey, str] | None = None,
    ) -> ChangeReport:
        report = ChangeReport(
            step=step,
            version=self._version,
            requested_keys=(),
            recomputed_keys=(),
            recomputed_glyph_ids=(),
            removed_keys=tuple(sorted(removed_keys)),
            atlas_modified_keys=tuple(sorted(set(atlas_modified_keys))),
            fallback_keys=(),
            delayed_keys=(),
            bitmap_hashes=(),
            unchanged_bitmap_hashes=tuple(sorted((unchanged or {}).items())),
            thread_budget=None,
            atlas_decisions=self._atlas_budget_manager.decisions,
        )
        self._reports.append(report)
        return report

    def _execute(
        self,
        *,
        step: str,
        new_glyphs: Mapping[str, Glyph],
        new_fill_rules: Mapping[str, str],
        new_active_sizes: Mapping[str, tuple[int, ...]],
        recompute_keys: Iterable[RenderKey],
        removed_keys: Iterable[RenderKey],
    ) -> ChangeReport:
        recompute = tuple(sorted(set(recompute_keys)))
        removed = tuple(sorted(set(removed_keys)))
        desired = tuple(
            sorted(
                RenderKey(glyph_id, size, new_fill_rules[glyph_id])
                for glyph_id in new_glyphs
                for size in new_active_sizes[glyph_id]
            )
        )
        if not set(recompute).issubset(desired):
            raise task_key_error("recompute set contains a key outside the active set")

        tasks = [
            RasterTask(
                new_glyphs[key.glyph_id],
                key.size,
                key.fill_rule,
                revision=self._revisions[key.glyph_id],
            )
            for key in recompute
        ]
        scheduled = (
            self._scheduler.run(
                tasks,
                worker_budget=self._worker_budget,
                memory_budget_bytes=self._memory_budget_bytes,
            )
            if tasks
            else None
        )
        raster_results = (
            {
                RenderKey(task_key.glyph_id, task_key.size, task_key.fill_rule): result
                for task_key, result in scheduled.results
            }
            if scheduled
            else {}
        )

        new_cache = dict(self._cache)
        new_cache_revision = dict(self._cache_revision)
        for key, result in raster_results.items():
            new_cache[key] = result
            new_cache_revision[key] = self._revisions[key.glyph_id]

        affected = set(recompute) | set(removed)
        baseline_keys = [
            key
            for key in desired
            if key not in affected
            and key in self._cache
            and key.display() in self._atlas_budget_manager.manager.regions
        ]
        baseline = self._baseline_regions(baseline_keys)

        for key in sorted(affected, key=lambda item: item.display()):
            self._atlas_budget_manager.remove(key.display())

        references = {key.display() for key in desired}
        outcomes: list[PackOutcome] = []
        for key in self._publish_order(recompute):
            result = new_cache[key]
            primary = AtlasAsset.from_raster(result, key=key.display())
            fallbacks = self._fallback_assets(key, new_cache, new_cache_revision)
            outcome = self._atlas_budget_manager.publish(
                key.display(),
                primary,
                fallbacks=fallbacks,
                referenced_keys=references,
            )
            outcomes.append(outcome)

        unchanged_hashes = self._verify_baseline(baseline)
        for key in recompute:
            if key not in new_cache:
                raise bitmap_hash_error("missing recomputed bitmap", key=key.display())
        # Removed mappings can leave a page empty even when no new request put
        # pressure on the allocator.  Reclaim only pages with zero references.
        self._atlas_budget_manager.recycle(references)

        self._glyphs = dict(sorted(new_glyphs.items()))
        self._fill_rules = {
            glyph_id: new_fill_rules[glyph_id] for glyph_id in sorted(new_glyphs)
        }
        self._active_sizes = {
            glyph_id: new_active_sizes[glyph_id] for glyph_id in sorted(new_glyphs)
        }
        self._cache = new_cache
        self._cache_revision = new_cache_revision
        self._last_thread_report = scheduled.report if scheduled else None
        self._version += 1

        outcome_by_key = {outcome.request_key: outcome for outcome in outcomes}
        fallback_keys = tuple(
            key
            for key in recompute
            if outcome_by_key.get(key.display()) is not None
            and outcome_by_key[key.display()].status == "fallback"
        )
        delayed_keys = tuple(
            key
            for key in recompute
            if outcome_by_key.get(key.display()) is not None
            and outcome_by_key[key.display()].status == "delayed"
        )
        bitmap_hashes = tuple(
            (key, raster_results[key].content_hash)
            for key in recompute
            if key in raster_results
        )
        report = ChangeReport(
            step=step,
            version=self._version,
            requested_keys=recompute,
            recomputed_keys=tuple(sorted(raster_results)),
            recomputed_glyph_ids=tuple(
                sorted({key.glyph_id for key in raster_results})
            ),
            removed_keys=removed,
            atlas_modified_keys=tuple(
                sorted(
                    {key.display() for key in affected}
                    | {item.request_key for item in outcomes}
                )
            ),
            fallback_keys=fallback_keys,
            delayed_keys=delayed_keys,
            bitmap_hashes=bitmap_hashes,
            unchanged_bitmap_hashes=tuple(sorted(unchanged_hashes.items())),
            thread_budget=scheduled.report if scheduled else None,
            atlas_decisions=self._atlas_budget_manager.decisions,
        )
        self._reports.append(report)
        return report

    def render_all(self, *, step: str = "render_all") -> ChangeReport:
        return self._execute(
            step=step,
            new_glyphs=self._glyphs,
            new_fill_rules=self._fill_rules,
            new_active_sizes=self._active_sizes,
            recompute_keys=self.active_keys(),
            removed_keys=(),
        )

    def render_keys(
        self, keys: Iterable[RenderKey], *, step: str = "render_keys"
    ) -> ChangeReport:
        return self._execute(
            step=step,
            new_glyphs=self._glyphs,
            new_fill_rules=self._fill_rules,
            new_active_sizes=self._active_sizes,
            recompute_keys=tuple(keys),
            removed_keys=(),
        )

    def edit_point(
        self,
        glyph_id: str,
        *,
        contour_index: int,
        segment_index: int,
        role: str,
        point: Point,
        step: str = "edit_point",
    ) -> ChangeReport:
        if glyph_id not in self._glyphs:
            raise task_key_error("unknown glyph id", glyph_id=glyph_id)
        old_glyph = self._glyphs[glyph_id]
        new_glyph = old_glyph.with_point(
            contour_index=contour_index,
            segment_index=segment_index,
            role=role,
            point=point,
        )
        new_glyphs = dict(self._glyphs)
        new_glyphs[glyph_id] = new_glyph
        self._revisions[glyph_id] += 1
        try:
            return self._execute(
                step=step,
                new_glyphs=new_glyphs,
                new_fill_rules=self._fill_rules,
                new_active_sizes=self._active_sizes,
                recompute_keys=self.active_keys(glyph_id),
                removed_keys=(),
            )
        except Exception:
            self._revisions[glyph_id] -= 1
            raise

    def set_fill_rule(
        self, glyph_id: str, fill_rule: str, *, step: str = "set_fill_rule"
    ) -> ChangeReport:
        if glyph_id not in self._glyphs:
            raise task_key_error("unknown glyph id", glyph_id=glyph_id)
        normalized = normalize_fill_rule(fill_rule)
        if self._fill_rules[glyph_id] == normalized:
            return self._report_without_raster(step=step)
        old_keys = set(self.active_keys(glyph_id))
        new_fill_rules = dict(self._fill_rules)
        new_fill_rules[glyph_id] = normalized
        new_keys = {
            RenderKey(glyph_id, size, normalized)
            for size in self._active_sizes[glyph_id]
        }
        return self._execute(
            step=step,
            new_glyphs=self._glyphs,
            new_fill_rules=new_fill_rules,
            new_active_sizes=self._active_sizes,
            recompute_keys=new_keys,
            removed_keys=old_keys - new_keys,
        )

    def set_global_fill_rule(
        self, fill_rule: str, *, step: str = "set_global_fill_rule"
    ) -> ChangeReport:
        normalized = normalize_fill_rule(fill_rule)
        changed = [
            glyph_id
            for glyph_id in self._glyphs
            if self._fill_rules[glyph_id] != normalized
        ]
        if not changed:
            return self._report_without_raster(step=step)
        old_keys = {key for key in self.active_keys() if key.glyph_id in set(changed)}
        new_fill_rules = {glyph_id: normalized for glyph_id in self._glyphs}
        new_keys = {
            RenderKey(glyph_id, size, normalized)
            for glyph_id in changed
            for size in self._active_sizes[glyph_id]
        }
        return self._execute(
            step=step,
            new_glyphs=self._glyphs,
            new_fill_rules=new_fill_rules,
            new_active_sizes=self._active_sizes,
            recompute_keys=new_keys,
            removed_keys=old_keys - new_keys,
        )

    def set_sizes(
        self,
        glyph_id: str,
        sizes: int | Sequence[int],
        *,
        step: str = "set_sizes",
    ) -> ChangeReport:
        if glyph_id not in self._glyphs:
            raise task_key_error("unknown glyph id", glyph_id=glyph_id)
        normalized = _normalize_sizes(sizes)
        if normalized == self._active_sizes[glyph_id]:
            return self._report_without_raster(step=step)
        old_keys = set(self.active_keys(glyph_id))
        new_active = dict(self._active_sizes)
        new_active[glyph_id] = normalized
        new_keys = {
            RenderKey(glyph_id, size, self._fill_rules[glyph_id]) for size in normalized
        }
        return self._execute(
            step=step,
            new_glyphs=self._glyphs,
            new_fill_rules=self._fill_rules,
            new_active_sizes=new_active,
            recompute_keys=new_keys - old_keys,
            removed_keys=old_keys - new_keys,
        )

    def switch_size(
        self, glyph_id: str, size: int, *, step: str = "switch_size"
    ) -> ChangeReport:
        return self.set_sizes(glyph_id, size, step=step)

    def unpublish_all(
        self, *, recycle: bool = True, step: str = "unpublish_all"
    ) -> ChangeReport:
        desired = self.active_keys()
        baseline = self._baseline_regions([])
        removed: list[RenderKey] = []
        for key in desired:
            if self._atlas_budget_manager.remove(key.display()) is not None:
                removed.append(key)
        if recycle:
            recycled = self._atlas_budget_manager.recycle(())
        else:
            recycled = ()
        if removed or recycled:
            self._version += 1
        return self._report_without_raster(
            step=step,
            removed_keys=removed,
            atlas_modified_keys=[key.display() for key in removed],
            unchanged=baseline,
        )

    def republish_cached(self, *, step: str = "republish_cached") -> ChangeReport:
        desired = self.active_keys()
        missing_cache = [key for key in desired if key not in self._cache]
        if missing_cache:
            raise bitmap_hash_error(
                "cannot republish missing cached bitmaps",
                missing_keys=[key.display() for key in missing_cache],
            )
        baseline = self._baseline_regions(
            [
                key
                for key in desired
                if key.display() in self._atlas_budget_manager.manager.regions
            ]
        )
        references = {key.display() for key in desired}
        modified: list[str] = []
        for key in self._publish_order(desired):
            if key.display() in self._atlas_budget_manager.manager.regions:
                continue
            result = self._cache[key]
            self._atlas_budget_manager.publish(
                key.display(),
                AtlasAsset.from_raster(result, key=key.display()),
                fallbacks=self._fallback_assets(key, self._cache),
                referenced_keys=references,
            )
            modified.append(key.display())
        unchanged = self._verify_baseline(baseline)
        self._version += 1
        return self._report_without_raster(
            step=step,
            atlas_modified_keys=modified,
            unchanged=unchanged,
        )

    def retry_pending(self, *, step: str = "retry_pending") -> ChangeReport:
        if not self._atlas_budget_manager.pending_keys:
            return self._report_without_raster(step=step)
        baseline = self._baseline_regions(
            [
                key
                for key in self.active_keys()
                if key.display() in self._atlas_budget_manager.manager.regions
            ]
        )
        references = {key.display() for key in self.active_keys()}
        outcomes = self._atlas_budget_manager.retry_pending(referenced_keys=references)
        unchanged = self._verify_baseline(baseline)
        self._version += 1
        return self._report_without_raster(
            step=step,
            atlas_modified_keys=[outcome.request_key for outcome in outcomes],
            unchanged=unchanged,
        )
