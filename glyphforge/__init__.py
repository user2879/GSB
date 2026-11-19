"""GlyphForge: deterministic outline booleans, stroke expansion, and coverage."""

from .boolean import BooleanOp, BooleanResult, ResolutionOrder, boolean_operation
from .errors import ErrorCode, GlyphError
from .flatten import FlattenOptions, FlattenedOutline, flatten_contours
from .model import Contour, GlyphCase, GlyphSuite, OutlineNode, Point, SegmentType
from .raster import (
    BitmapConsistency,
    RasterBitmap,
    Viewport,
    check_bitmap_consistency,
    rasterize_boolean,
    rasterize_loops,
    rasterize_stroke,
)
from .samples import generate_suite
from .stroke import CapStyle, JoinStyle, StrokeOptions, StrokeResult, stroke_polyline

__all__ = [
    "BitmapConsistency",
    "BooleanOp",
    "BooleanResult",
    "CapStyle",
    "Contour",
    "ErrorCode",
    "FlattenOptions",
    "FlattenedOutline",
    "GlyphCase",
    "GlyphError",
    "GlyphSuite",
    "JoinStyle",
    "OutlineNode",
    "Point",
    "RasterBitmap",
    "ResolutionOrder",
    "SegmentType",
    "StrokeOptions",
    "StrokeResult",
    "Viewport",
    "boolean_operation",
    "check_bitmap_consistency",
    "flatten_contours",
    "generate_suite",
    "rasterize_boolean",
    "rasterize_loops",
    "rasterize_stroke",
    "stroke_polyline",
]
