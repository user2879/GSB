"""Byte-order and index-width compatible binary stroke skeleton format.

The wire format is documented in ``docs/FORMAT.md``. Parsing is deliberately
transactional: a function-local document is fully validated before any caller
receives geometry, so malformed input cannot leave a half-loaded scene.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .errors import NprError, require
from .geometry import GeometryEngine, StrokeSkeleton, flatten_shape
from .migration import migrate_document
from .model import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_STROKE_SAMPLE,
    DEFAULT_STYLE,
    Contour,
    Project,
    Segment,
    bytes_hash,
    canonical_hash,
    project_from_dict,
)

MAGIC = b"NPBS"
ENDIAN_MARKER = 0x01020304
OLD_FORMAT_VERSION = 1
CURRENT_FORMAT_VERSION = 2
OLD_WIDTH_ENCODING = 0
NEW_WIDTH_ENCODING = 1
FLAG_PROJECT_METADATA = 0x01
KNOWN_FLAGS = FLAG_PROJECT_METADATA


@dataclass
class BinaryContourRange:
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int
    closed: bool


@dataclass
class BinaryShapeSkeleton:
    shape_id: str
    layer: str
    stroke_width: float
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int
    contours: list[BinaryContourRange] = field(default_factory=list)


@dataclass
class ParsedBinary:
    format_version: int
    byte_order: str
    index_width: int
    width_encoding: int
    flags: int
    vertex_count: int
    index_count: int
    shape_count: int
    shapes: list[BinaryShapeSkeleton]
    vertices: np.ndarray
    indices: np.ndarray
    metadata: dict[str, Any] | None
    content_hash: str

    @property
    def metadata_project_hash(self) -> str | None:
        if self.metadata is None:
            return None
        return canonical_hash(self.metadata)

    def indices_for_shape(self, shape: BinaryShapeSkeleton) -> np.ndarray:
        return self.indices[shape.index_offset : shape.index_offset + shape.index_count]


@dataclass
class BinaryEncodeResult:
    data: bytes
    content_hash: str
    format_version: int
    byte_order: str
    index_width: int
    width_encoding: int
    vertex_count: int
    index_count: int
    shape_count: int

    def report(self) -> dict[str, Any]:
        return {
            "binary_hash": f"bin://sha256:{self.content_hash}",
            "format_version": self.format_version,
            "byte_order": self.byte_order,
            "index_width": self.index_width,
            "width_encoding": self.width_encoding,
            "vertex_count": self.vertex_count,
            "index_count": self.index_count,
            "shape_count": self.shape_count,
        }


@dataclass
class BinaryDecodeResult:
    parsed: ParsedBinary
    project: Project
    project_hash: str

    def report(self) -> dict[str, Any]:
        return {
            "binary_hash": f"bin://sha256:{self.parsed.content_hash}",
            "project_hash": f"json://sha256:{self.project_hash}",
            "format_version": self.parsed.format_version,
            "byte_order": self.parsed.byte_order,
            "index_width": self.parsed.index_width,
            "width_encoding": self.parsed.width_encoding,
            "vertex_count": self.parsed.vertex_count,
            "index_count": self.parsed.index_count,
            "shape_count": self.parsed.shape_count,
        }


def _byte_order_name(value: str) -> tuple[str, str]:
    if value in ("little", "le", "<"):
        return "little", "<"
    if value in ("big", "be", ">"):
        return "big", ">"
    raise NprError("E_FORMAT_ENDIAN", "unknown requested byte order", byte_order=value)


def _encode_width(width: float, encoding: int, endian: str) -> bytes:
    require(
        np.isfinite(width) and width > 0.0,
        "E_FORMAT_INDEX",
        "stroke width must be finite and positive",
        width=width,
    )
    if encoding == NEW_WIDTH_ENCODING:
        return struct.pack(endian + "f", np.float32(width))
    fixed = int(round(width * 256.0))
    require(
        1 <= fixed <= 0xFFFF,
        "E_FORMAT_INDEX",
        "stroke width cannot be represented by legacy uint16 encoding",
        width=width,
        fixed=fixed,
    )
    return struct.pack(endian + "H", fixed)


def _decode_width(data: bytes, offset: int, encoding: int, endian: str) -> tuple[float, int]:
    if encoding == NEW_WIDTH_ENCODING:
        value = struct.unpack_from(endian + "f", data, offset)[0]
        require(
            np.isfinite(value) and value > 0.0,
            "E_FORMAT_INDEX",
            "decoded float32 width is invalid",
            width=value,
        )
        return float(value), offset + 4
    fixed = struct.unpack_from(endian + "H", data, offset)[0]
    require(fixed > 0, "E_FORMAT_INDEX", "decoded legacy width is zero")
    return fixed / 256.0, offset + 2


def _normalize_options(
    format_version: int, width_encoding: int | None, index_width: int, byte_order: str
) -> tuple[str, str, int, int]:
    endian_name, endian = _byte_order_name(byte_order)
    require(
        format_version in (OLD_FORMAT_VERSION, CURRENT_FORMAT_VERSION),
        "E_FORMAT_VERSION",
        "unsupported binary format version",
        format_version=format_version,
    )
    require(
        index_width in (2, 4),
        "E_FORMAT_INDEX",
        "index width must be 16 or 32 bits",
        index_width=index_width,
    )
    if width_encoding is None:
        width_encoding = OLD_WIDTH_ENCODING if format_version == 1 else NEW_WIDTH_ENCODING
    require(
        width_encoding in (OLD_WIDTH_ENCODING, NEW_WIDTH_ENCODING),
        "E_FORMAT_INDEX",
        "unknown width encoding",
        width_encoding=width_encoding,
    )
    expected = OLD_WIDTH_ENCODING if format_version == OLD_FORMAT_VERSION else NEW_WIDTH_ENCODING
    require(
        width_encoding == expected,
        "E_FORMAT_INDEX",
        "format version and width encoding conflict",
        format_version=format_version,
        width_encoding=width_encoding,
        expected_width_encoding=expected,
    )
    return endian_name, endian, index_width, width_encoding


def encode_binary(
    project: Project,
    *,
    byte_order: str = "little",
    index_width: int = 4,
    format_version: int = CURRENT_FORMAT_VERSION,
    width_encoding: int | None = None,
    include_project_metadata: bool = True,
    skeletons: Mapping[str, StrokeSkeleton] | None = None,
) -> BinaryEncodeResult:
    endian_name, endian, index_width, width_encoding = _normalize_options(
        format_version, width_encoding, index_width, byte_order
    )

    if skeletons is None:
        engine = GeometryEngine(project)
        ordered_skeletons = [engine.ensure(shape.id) for shape in project.ordered_shapes()]
    else:
        ordered_skeletons = []
        for shape in project.ordered_shapes():
            try:
                ordered_skeletons.append(skeletons[shape.id])
            except KeyError as exc:
                raise NprError(
                    "E_FORMAT_INDEX",
                    "missing skeleton for project shape",
                    shape=shape.id,
                ) from exc

    vertex_count = sum(skeleton.vertex_count for skeleton in ordered_skeletons)
    index_count = sum(skeleton.index_count for skeleton in ordered_skeletons)
    require(
        vertex_count <= np.iinfo(np.uint32).max,
        "E_FORMAT_INDEX",
        "vertex count exceeds 32-bit header capacity",
    )
    if index_width == 2:
        require(
            vertex_count <= 0xFFFF,
            "E_FORMAT_INDEX",
            "16-bit index width cannot address the vertex buffer",
            vertex_count=vertex_count,
        )

    metadata = b""
    flags = 0
    if include_project_metadata:
        metadata = json.dumps(
            project.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        flags |= FLAG_PROJECT_METADATA

    output = bytearray()
    output.extend(MAGIC)
    output.extend(struct.pack(endian + "I", ENDIAN_MARKER))
    output.extend(
        struct.pack(
            endian + "HBBBBIIII",
            format_version,
            index_width,
            width_encoding,
            flags,
            0,
            vertex_count,
            index_count,
            len(ordered_skeletons),
            len(metadata),
        )
    )
    output.extend(metadata)

    vertex_cursor = 0
    index_cursor = 0
    for skeleton in ordered_skeletons:
        shape_id = skeleton.shape_id.encode("utf-8")
        layer = skeleton.layer.encode("utf-8")
        require(
            len(shape_id) <= 0xFFFF and len(layer) <= 0xFFFF,
            "E_FORMAT_INDEX",
            "shape id or layer is too long for binary table",
        )
        require(
            len(skeleton.contour_ranges) <= 0xFFFF,
            "E_FORMAT_INDEX",
            "too many contours in one binary shape",
        )
        output.extend(struct.pack(endian + "H", len(shape_id)))
        output.extend(shape_id)
        output.extend(struct.pack(endian + "H", len(layer)))
        output.extend(layer)
        output.extend(_encode_width(skeleton.stroke_width, width_encoding, endian))
        output.extend(
            struct.pack(
                endian + "IIIIH",
                vertex_cursor,
                skeleton.vertex_count,
                index_cursor,
                skeleton.index_count,
                len(skeleton.contour_ranges),
            )
        )
        for contour in skeleton.contour_ranges:
            output.extend(
                struct.pack(
                    endian + "IIIIB",
                    contour.vertex_offset,
                    contour.vertex_count,
                    contour.index_offset,
                    contour.index_count,
                    1 if contour.closed else 0,
                )
            )
        vertex_cursor += skeleton.vertex_count
        index_cursor += skeleton.index_count

    require(
        vertex_cursor == vertex_count and index_cursor == index_count,
        "E_FORMAT_INDEX",
        "internal skeleton count mismatch",
    )

    for skeleton in ordered_skeletons:
        vertices = np.asarray(skeleton.vertices, dtype=np.float32)
        output.extend(vertices.astype(endian + "f4", copy=False).tobytes(order="C"))
    index_dtype = np.uint16 if index_width == 2 else np.uint32
    vertex_base = 0
    for skeleton in ordered_skeletons:
        indices = np.asarray(skeleton.indices, dtype=np.uint32) + np.uint32(vertex_base)
        indices = indices.astype(index_dtype)
        output.extend(indices.astype(endian + ("u2" if index_width == 2 else "u4"), copy=False).tobytes(order="C"))
        vertex_base += skeleton.vertex_count

    data = bytes(output)
    return BinaryEncodeResult(
        data=data,
        content_hash=bytes_hash(data),
        format_version=format_version,
        byte_order=endian_name,
        index_width=index_width,
        width_encoding=width_encoding,
        vertex_count=vertex_count,
        index_count=index_count,
        shape_count=len(ordered_skeletons),
    )


def _need(offset: int, size: int, total: int, what: str) -> None:
    require(
        size >= 0 and offset >= 0 and offset + size <= total,
        "E_FORMAT_INDEX",
        "binary payload is truncated or has an invalid count",
        section=what,
        offset=offset,
        size=size,
        total=total,
    )


def _read_u16(data: bytes, offset: int, endian: str) -> tuple[int, int]:
    _need(offset, 2, len(data), "u16")
    return struct.unpack_from(endian + "H", data, offset)[0], offset + 2


def _read_u32(data: bytes, offset: int, endian: str) -> tuple[int, int]:
    _need(offset, 4, len(data), "u32")
    return struct.unpack_from(endian + "I", data, offset)[0], offset + 4


def parse_binary(data: bytes) -> ParsedBinary:
    require(
        isinstance(data, (bytes, bytearray, memoryview)),
        "E_FORMAT_INDEX",
        "binary input must be bytes",
    )
    data = bytes(data)
    require(len(data) >= 8, "E_FORMAT_INDEX", "binary input is shorter than its magic")
    require(data[:4] == MAGIC, "E_FORMAT_INDEX", "binary magic does not match NPBS")
    marker = data[4:8]
    if marker == b"\x01\x02\x03\x04":
        endian_name, endian = "big", ">"
    elif marker == b"\x04\x03\x02\x01":
        endian_name, endian = "little", "<"
    else:
        raise NprError(
            "E_FORMAT_ENDIAN",
            "unknown binary endian marker",
            marker=marker.hex(),
        )

    _need(8, 22, len(data), "header")
    (
        format_version,
        index_width,
        width_encoding,
        flags,
        reserved,
        vertex_count,
        index_count,
        shape_count,
        metadata_len,
    ) = struct.unpack_from(endian + "HBBBBIIII", data, 8)
    require(
        format_version in (OLD_FORMAT_VERSION, CURRENT_FORMAT_VERSION),
        "E_FORMAT_VERSION",
        "unsupported binary format version",
        format_version=format_version,
    )
    require(index_width in (2, 4), "E_FORMAT_INDEX", "invalid index width", index_width=index_width)
    require(
        width_encoding in (OLD_WIDTH_ENCODING, NEW_WIDTH_ENCODING),
        "E_FORMAT_INDEX",
        "invalid width encoding",
        width_encoding=width_encoding,
    )
    expected_encoding = (
        OLD_WIDTH_ENCODING if format_version == OLD_FORMAT_VERSION else NEW_WIDTH_ENCODING
    )
    require(
        width_encoding == expected_encoding,
        "E_FORMAT_INDEX",
        "format version and width encoding conflict",
        format_version=format_version,
        width_encoding=width_encoding,
    )
    require(
        flags & ~KNOWN_FLAGS == 0,
        "E_FORMAT_DATA",
        "unknown binary flags are set",
        flags=flags,
    )
    require(reserved == 0, "E_FORMAT_DATA", "reserved header byte must be zero")
    require(shape_count > 0, "E_FORMAT_INDEX", "binary contains no shapes")
    require(vertex_count > 0, "E_FORMAT_INDEX", "binary contains no vertices")
    require(index_count > 0, "E_FORMAT_INDEX", "binary contains no indices")

    offset = 30
    _need(offset, metadata_len, len(data), "metadata")
    metadata_bytes = data[offset : offset + metadata_len]
    offset += metadata_len
    metadata: dict[str, Any] | None = None
    if flags & FLAG_PROJECT_METADATA:
        require(metadata_len > 0, "E_FORMAT_DATA", "metadata flag set without bytes")
        try:
            parsed_metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise NprError("E_FORMAT_DATA", "project metadata is not valid UTF-8 JSON") from exc
        require(
            isinstance(parsed_metadata, dict),
            "E_FORMAT_DATA",
            "project metadata must be an object",
        )
        metadata = parsed_metadata

    shapes: list[BinaryShapeSkeleton] = []
    width_size = 4 if width_encoding == NEW_WIDTH_ENCODING else 2
    for _ in range(shape_count):
        id_len, offset = _read_u16(data, offset, endian)
        _need(offset, id_len, len(data), "shape id")
        try:
            shape_id = data[offset : offset + id_len].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NprError("E_FORMAT_DATA", "shape id is not UTF-8") from exc
        offset += id_len
        layer_len, offset = _read_u16(data, offset, endian)
        _need(offset, layer_len, len(data), "layer")
        try:
            layer = data[offset : offset + layer_len].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NprError("E_FORMAT_DATA", "layer is not UTF-8") from exc
        offset += layer_len
        require(bool(shape_id) and bool(layer), "E_FORMAT_DATA", "empty shape id or layer")
        _need(offset, 4 if width_encoding == NEW_WIDTH_ENCODING else 2, len(data), "shape width")
        stroke_width, offset = _decode_width(data, offset, width_encoding, endian)
        _need(offset, 18, len(data), "shape table")
        (
            vertex_offset,
            shape_vertex_count,
            index_offset,
            shape_index_count,
            contour_count,
        ) = struct.unpack_from(endian + "IIIIH", data, offset)
        offset += 18
        require(contour_count > 0, "E_FORMAT_INDEX", "shape has no contours", shape=shape_id)
        require(
            vertex_offset + shape_vertex_count <= vertex_count,
            "E_FORMAT_INDEX",
            "shape vertex range exceeds header vertex count",
            shape=shape_id,
        )
        require(
            index_offset + shape_index_count <= index_count,
            "E_FORMAT_INDEX",
            "shape index range exceeds header index count",
            shape=shape_id,
        )
        contours: list[BinaryContourRange] = []
        contour_vertex_sum = 0
        contour_index_sum = 0
        for _ in range(contour_count):
            _need(offset, 17, len(data), "contour table")
            (
                contour_vertex_offset,
                contour_vertex_count,
                contour_index_offset,
                contour_index_count,
                closed,
            ) = struct.unpack_from(endian + "IIIIB", data, offset)
            offset += 17
            require(closed in (0, 1), "E_FORMAT_INDEX", "closed flag must be 0 or 1")
            require(
                contour_vertex_offset + contour_vertex_count <= shape_vertex_count,
                "E_FORMAT_INDEX",
                "contour vertex range exceeds shape range",
                shape=shape_id,
            )
            require(
                contour_index_offset + contour_index_count <= shape_index_count,
                "E_FORMAT_INDEX",
                "contour index range exceeds shape range",
                shape=shape_id,
            )
            require(
                contour_vertex_count >= 2,
                "E_FORMAT_INDEX",
                "contour has fewer than two vertices",
                shape=shape_id,
            )
            expected_indices = contour_vertex_count + (1 if closed else 0)
            require(
                contour_index_count == expected_indices,
                "E_FORMAT_INDEX",
                "contour vertex/index count mismatch",
                shape=shape_id,
                expected=expected_indices,
                actual=contour_index_count,
            )
            contours.append(
                BinaryContourRange(
                    vertex_offset=contour_vertex_offset,
                    vertex_count=contour_vertex_count,
                    index_offset=contour_index_offset,
                    index_count=contour_index_count,
                    closed=bool(closed),
                )
            )
            contour_vertex_sum += contour_vertex_count
            contour_index_sum += contour_index_count
        require(
            contour_vertex_sum == shape_vertex_count,
            "E_FORMAT_INDEX",
            "shape vertex count does not match its contours",
            shape=shape_id,
            declared=shape_vertex_count,
            contours=contour_vertex_sum,
        )
        require(
            contour_index_sum == shape_index_count,
            "E_FORMAT_INDEX",
            "shape index count does not match its contours",
            shape=shape_id,
            declared=shape_index_count,
            contours=contour_index_sum,
        )
        shapes.append(
            BinaryShapeSkeleton(
                shape_id=shape_id,
                layer=layer,
                stroke_width=stroke_width,
                vertex_offset=vertex_offset,
                vertex_count=shape_vertex_count,
                index_offset=index_offset,
                index_count=shape_index_count,
                contours=contours,
            )
        )

    require(
        len(shapes) == shape_count,
        "E_FORMAT_INDEX",
        "shape table count mismatch",
    )
    require(
        sum(shape.vertex_count for shape in shapes) == vertex_count,
        "E_FORMAT_INDEX",
        "shape vertex counts do not sum to header vertex count",
    )
    require(
        sum(shape.index_count for shape in shapes) == index_count,
        "E_FORMAT_INDEX",
        "shape index counts do not sum to header index count",
    )
    running_vertex = 0
    running_index = 0
    for shape in shapes:
        require(
            shape.vertex_offset == running_vertex,
            "E_FORMAT_INDEX",
            "shape vertex offsets are not contiguous",
            shape=shape.shape_id,
        )
        require(
            shape.index_offset == running_index,
            "E_FORMAT_INDEX",
            "shape index offsets are not contiguous",
            shape=shape.shape_id,
        )
        running_vertex += shape.vertex_count
        running_index += shape.index_count

    vertex_bytes = vertex_count * 8
    _need(offset, vertex_bytes, len(data), "vertices")
    vertices = np.frombuffer(data, dtype=endian + "f4", count=vertex_count * 2, offset=offset).reshape(
        vertex_count, 2
    )
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    require(
        np.isfinite(vertices).all(),
        "E_FORMAT_INDEX",
        "vertex buffer contains non-finite coordinates",
    )
    offset += vertex_bytes
    index_size = 2 if index_width == 2 else 4
    index_bytes = index_count * index_size
    _need(offset, index_bytes, len(data), "indices")
    index_dtype = endian + ("u2" if index_width == 2 else "u4")
    indices = np.frombuffer(data, dtype=index_dtype, count=index_count, offset=offset).astype(
        np.uint32, copy=True
    )
    offset += index_bytes
    require(
        offset == len(data),
        "E_FORMAT_INDEX",
        "unexpected trailing bytes after binary skeleton",
        trailing=len(data) - offset,
    )
    require(
        indices.size == index_count,
        "E_FORMAT_INDEX",
        "index buffer count mismatch",
    )
    if indices.size:
        require(
            int(indices.max()) < vertex_count,
            "E_FORMAT_INDEX",
            "index is outside the global vertex buffer",
            max_index=int(indices.max()),
            vertex_count=vertex_count,
        )

    for shape in shapes:
        shape_vertex_start = shape.vertex_offset
        shape_vertex_stop = shape.vertex_offset + shape.vertex_count
        shape_index_start = shape.index_offset
        shape_index_stop = shape.index_offset + shape.index_count
        for contour in shape.contours:
            contour_vertex_start = shape_vertex_start + contour.vertex_offset
            contour_vertex_stop = contour_vertex_start + contour.vertex_count
            contour_index_start = shape_index_start + contour.index_offset
            contour_index_stop = contour_index_start + contour.index_count
            require(
                contour_vertex_stop <= shape_vertex_stop,
                "E_FORMAT_INDEX",
                "contour vertex range escapes its shape",
            )
            subset = indices[contour_index_start:contour_index_stop]
            require(
                subset.size == contour.index_count,
                "E_FORMAT_INDEX",
                "contour index range escapes its shape",
            )
            if subset.size:
                require(
                    int(subset.min()) >= contour_vertex_start
                    and int(subset.max()) < contour_vertex_stop,
                    "E_FORMAT_INDEX",
                    "contour index points outside its vertex range",
                    shape=shape.shape_id,
                )
                if contour.closed:
                    require(
                        int(subset[-1]) == contour_vertex_start,
                        "E_FORMAT_INDEX",
                        "closed contour does not repeat its first vertex",
                    )

    parsed = ParsedBinary(
        format_version=format_version,
        byte_order=endian_name,
        index_width=index_width,
        width_encoding=width_encoding,
        flags=flags,
        vertex_count=vertex_count,
        index_count=index_count,
        shape_count=shape_count,
        shapes=shapes,
        vertices=vertices,
        indices=indices,
        metadata=metadata,
        content_hash=bytes_hash(data),
    )
    return parsed


def _project_from_skeleton(parsed: ParsedBinary) -> Project:
    shapes = []
    layer_order: list[str] = []
    for shape in parsed.shapes:
        if shape.layer not in layer_order:
            layer_order.append(shape.layer)
        contours: list[Contour] = []
        for contour in shape.contours:
            index_start = shape.index_offset + contour.index_offset
            index_stop = index_start + contour.index_count
            indices = parsed.indices[index_start:index_stop]
            segments: list[Segment] = []
            for first, second in zip(indices[:-1], indices[1:]):
                p0 = parsed.vertices[int(first)]
                p1 = parsed.vertices[int(second)]
                segments.append(
                    Segment(
                        "line",
                        [
                            (float(np.float32(p0[0])), float(np.float32(p0[1]))),
                            (float(np.float32(p1[0])), float(np.float32(p1[1]))),
                        ],
                    )
                )
            require(
                bool(segments),
                "E_FORMAT_INDEX",
                "imported contour has no line segments",
                shape=shape.shape_id,
            )
            contours.append(Contour(closed=contour.closed, segments=segments))
        shapes.append(
            {
                "id": shape.shape_id,
                "layer": shape.layer,
                "stroke_width": float(shape.stroke_width),
                "style_tags": [],
                "contours": [contour.to_dict() for contour in contours],
            }
        )
    document = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "project_id": "binary-import",
        "version_stamp": "000000000001",
        "shapes": shapes,
        "style": DEFAULT_STYLE,
        "stroke_sample": DEFAULT_STROKE_SAMPLE,
        "layer_order": layer_order,
        "metadata": {"imported_from": "NPBS skeleton without project metadata"},
    }
    return project_from_dict(document)


def _validate_metadata_geometry(project: Project, parsed: ParsedBinary) -> None:
    engine = GeometryEngine(project)
    by_id = {shape.shape_id: shape for shape in parsed.shapes}
    require(
        len(by_id) == len(parsed.shapes),
        "E_FORMAT_INDEX",
        "binary contains duplicate shape ids",
    )
    project_ids = [shape.id for shape in project.ordered_shapes()]
    require(
        set(project_ids) == set(by_id),
        "E_FORMAT_INDEX",
        "binary and metadata shape sets differ",
        missing=sorted(set(project_ids) - set(by_id)),
        extra=sorted(set(by_id) - set(project_ids)),
    )
    for project_shape in project.ordered_shapes():
        binary_shape = by_id[project_shape.id]
        skeleton = engine.ensure(project_shape.id)
        expected_start = binary_shape.vertex_offset
        expected = parsed.vertices[
            expected_start : expected_start + binary_shape.vertex_count
        ]
        require(
            expected.shape == skeleton.vertices.shape and np.array_equal(expected, skeleton.vertices),
            "E_FORMAT_INDEX",
            "metadata geometry does not match skeleton vertices",
            shape=project_shape.id,
        )


def decode_binary(data: bytes) -> BinaryDecodeResult:
    parsed = parse_binary(data)
    if parsed.metadata is not None:
        migration = migrate_document(parsed.metadata)
        project = migration.project
        _validate_metadata_geometry(project, parsed)
    else:
        project = _project_from_skeleton(parsed)
    return BinaryDecodeResult(
        parsed=parsed,
        project=project,
        project_hash=project.content_hash(),
    )


def json_project_hash(project: Project) -> str:
    return f"json://sha256:{project.content_hash()}"


def binary_content_hash(data: bytes) -> str:
    return f"bin://sha256:{bytes_hash(data)}"


def roundtrip_project(
    project: Project,
    *,
    byte_order: str = "little",
    index_width: int = 4,
    format_version: int = CURRENT_FORMAT_VERSION,
) -> tuple[BinaryEncodeResult, BinaryDecodeResult]:
    encoded = encode_binary(
        project,
        byte_order=byte_order,
        index_width=index_width,
        format_version=format_version,
    )
    decoded = decode_binary(encoded.data)
    return encoded, decoded
