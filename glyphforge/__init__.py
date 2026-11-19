"""GlyphForge: deterministic outline rasterization and atlas diagnostics."""

__version__ = "0.1.0"

from .application import ChangeReport, GlyphApplication, RenderKey
from .atlas import (
    AtlasAsset,
    AtlasBudget,
    AtlasBudgetManager,
    AtlasManager,
    AtlasRegion,
    PackOutcome,
)
from .demo import run_demo
from .diagnostics import DiagnosticService, DiagnosticSnapshot, DiagnosticViewState
from .errors import ErrorCode, GlyphForgeError
from .geometry import (
    FlattenedContour,
    FlattenedGlyph,
    FlattenedSegment,
    flatten_contour,
    flatten_glyph,
    flatten_tolerance_for_size,
)
from .model import (
    Contour,
    Glyph,
    GlyphDocument,
    Point,
    Segment,
    dumps_glyphs,
    loads_glyphs,
    validate_glyphs,
)
from .raster import RasterConfig, RasterResult, bitmap_hash, rasterize_glyph
from .samples import build_sample_document, sample_glyphs, write_sample_json
from .scheduler import (
    RasterScheduler,
    RasterTask,
    RasterTaskKey,
    ScheduledRasterResult,
    ThreadBudgetReport,
)

__all__ = [
    "AtlasAsset",
    "AtlasBudget",
    "AtlasBudgetManager",
    "AtlasManager",
    "AtlasRegion",
    "ChangeReport",
    "Contour",
    "DiagnosticService",
    "DiagnosticSnapshot",
    "DiagnosticViewState",
    "ErrorCode",
    "FlattenedContour",
    "FlattenedGlyph",
    "FlattenedSegment",
    "Glyph",
    "GlyphApplication",
    "GlyphDocument",
    "GlyphForgeError",
    "PackOutcome",
    "Point",
    "RasterConfig",
    "RasterResult",
    "RasterScheduler",
    "RasterTask",
    "RasterTaskKey",
    "RenderKey",
    "ScheduledRasterResult",
    "Segment",
    "ThreadBudgetReport",
    "bitmap_hash",
    "build_sample_document",
    "dumps_glyphs",
    "flatten_contour",
    "flatten_glyph",
    "flatten_tolerance_for_size",
    "loads_glyphs",
    "rasterize_glyph",
    "run_demo",
    "sample_glyphs",
    "validate_glyphs",
    "write_sample_json",
]
