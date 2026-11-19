"""Project migration and the binary stroke-skeleton exchange format."""

from __future__ import annotations

import copy
import io
import json
import struct
from pathlib import Path
from typing import Any, Iterable

from .errors import (
    E_FORMAT_ENDIAN,
    E_FORMAT_INDEX,
    E_FORMAT_WIDTH,
    E_OUTLINE_DEGENERATE,
    E_SCHEMA_VERSION,
    E_STROKE_SAMPLE,
    NPRError,
)
from .geometry import flatten_element
from .model import SCHEMA_VERSION, Layer, Point, Scene, StrokeSampling, style_from_dict
from .util import atomic_write_bytes, canonical_json_bytes, read_json, sha256_bytes, sha256_json

BINARY_MAGIC = b"NPRSKEL\x00"
BINARY_VERSION = 1
HEADER_SIZE = 64
MIGRATION_DEFAULTS = {
    "style": {"tag": "ink", "tone": 0.42, "roughness": 0.18, "tint": [24, 31, 38]},
    "sampling": {
        "mode": "adaptive",
        "tolerance_px": 1.0,
        "min_subdivisions": 1,
        "max_subdivisions": 18,
        "width_scale": 1.0,
    },
    "layer_order": 0,
    "layer_name_prefix": "migrated-layer-",
    "canvas": {"width": 640, "height": 480},
    "generator": {"kind": "migrated-project", "seed": 0},
}
ENDIAN_MARKERS = {"little": b"LE", "big": b"BE"}
MARKER_ENDIAN = {value: key for key, value in ENDIAN_MARKERS.items()}
WIDTH_ENCODINGS = {"f32": 1, "u16_fixed": 2}
ENCODING_WIDTHS = {value: key for key, value in WIDTH_ENCODINGS.items()}
KIND_TO_CODE = {"line": 1, "quadratic": 2, "cubic": 3}
CODE_TO_KIND = {value: key for key, value in KIND_TO_CODE.items()}


def _legacy_layer_id(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return f"L{int(raw):02d}"
    except (TypeError, ValueError):
        return "L00"


def _ensure_missing_fields_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """Migration table 1 -> 2: fill all fields absent from early projects."""

    result = copy.deepcopy(raw)
    result.setdefault("canvas", copy.deepcopy(MIGRATION_DEFAULTS["canvas"]))
    result.setdefault("generator", copy.deepcopy(MIGRATION_DEFAULTS["generator"]))
    result.setdefault("sampling", copy.deepcopy(MIGRATION_DEFAULTS["sampling"]))
    elements = result.get("elements", [])
    if not isinstance(elements, list):
        raise NPRError(E_SCHEMA_VERSION, "legacy elements field must be a list")
    layers_raw = result.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        discovered: dict[str, int] = {}
        for index, element in enumerate(elements):
            if not isinstance(element, dict):
                continue
            layer_id = _legacy_layer_id(element.get("layer_id", element.get("layer", 0)))
            order_raw = element.get("layer_order", element.get("order", index))
            try:
                order = int(order_raw)
            except (TypeError, ValueError):
                order = index
            discovered.setdefault(layer_id, order)
        if not discovered:
            discovered["L00"] = 0
        layers_raw = [
            {"id": layer_id, "name": layer_id, "order": order, "visible": True}
            for layer_id, order in sorted(discovered.items(), key=lambda item: (item[1], item[0]))
        ]
        result["layers"] = layers_raw
    else:
        normalised_layers: list[dict[str, Any]] = []
        for index, layer in enumerate(layers_raw):
            if not isinstance(layer, dict):
                raise NPRError(E_SCHEMA_VERSION, "legacy layer must be an object")
            item = copy.deepcopy(layer)
            item.setdefault("name", str(item.get("id", f"L{index:02d}")))
            item.setdefault("order", index)
            item.setdefault("visible", True)
            item.setdefault("id", _legacy_layer_id(item.get("order", index)))
            normalised_layers.append(item)
        result["layers"] = normalised_layers
    layer_order_by_id = {str(layer["id"]): int(layer.get("order", index)) for index, layer in enumerate(result["layers"])}
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            raise NPRError(E_SCHEMA_VERSION, "legacy element must be an object")
        element.setdefault("layer_id", _legacy_layer_id(element.get("layer", 0)))
        element["layer_id"] = _legacy_layer_id(element["layer_id"])
        element.pop("layer", None)
        if "layer_order" not in element:
            element["layer_order"] = layer_order_by_id.get(str(element["layer_id"]), MIGRATION_DEFAULTS["layer_order"])
        if "style" not in element:
            element["style"] = copy.deepcopy(MIGRATION_DEFAULTS["style"])
            if "style_tag" in element:
                element["style"]["tag"] = str(element["style_tag"])
            element.pop("style_tag", None)
        if "sampling" not in element:
            element["sampling"] = copy.deepcopy(MIGRATION_DEFAULTS["sampling"])
        if "closed" not in element:
            element["closed"] = bool(element.get("is_closed", False))
        element.pop("is_closed", None)
    result["schema_version"] = 2
    return result


def _migrate_v2_to_v3(raw: dict[str, Any]) -> dict[str, Any]:
    """Migration table 2 -> 3: canonical layer/order representation."""

    result = copy.deepcopy(raw)
    layers = result.get("layers", [])
    order_by_id: dict[str, int] = {}
    for index, layer in enumerate(layers):
        layer.setdefault("name", str(layer.get("id", f"L{index:02d}")))
        layer.setdefault("visible", True)
        layer.setdefault("order", index)
        order_by_id[str(layer["id"])] = int(layer["order"])
    for index, element in enumerate(result.get("elements", [])):
        element["layer_id"] = _legacy_layer_id(element.get("layer_id", 0))
        if "layer_order" not in element:
            element["layer_order"] = order_by_id.get(str(element["layer_id"]), MIGRATION_DEFAULTS["layer_order"])
    result["schema_version"] = 3
    return result


def migrate_project(raw: Any, target_version: int = SCHEMA_VERSION) -> dict[str, Any]:
    """Apply the fixed migration table.  Latest projects are returned unchanged."""

    if not isinstance(raw, dict):
        raise NPRError(E_SCHEMA_VERSION, "project root must be an object")
    try:
        version = int(raw.get("schema_version"))
    except (TypeError, ValueError) as exc:
        raise NPRError(E_SCHEMA_VERSION, "missing or invalid schema_version") from exc
    if version > target_version:
        raise NPRError(
            E_SCHEMA_VERSION,
            "project schema is newer than this application",
            got=version,
            supported=target_version,
        )
    if version < 1:
        raise NPRError(E_SCHEMA_VERSION, "schema_version must be >= 1", got=version)
    result = copy.deepcopy(raw)
    if version == 1:
        result = _ensure_missing_fields_v1(result)
        version = 2
    if version == 2:
        result = _migrate_v2_to_v3(result)
        version = 3
    if version != target_version:
        raise NPRError(E_SCHEMA_VERSION, "no migration path to requested schema", got=version, target=target_version)
    result["schema_version"] = int(target_version)
    return result


def migrate_project_bytes(data: bytes) -> bytes:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NPRError(E_SCHEMA_VERSION, "project JSON is invalid") from exc
    return canonical_json_bytes(migrate_project(raw))


def load_project(path: Path) -> Scene:
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise NPRError(E_SCHEMA_VERSION, "cannot read project JSON", path=str(path)) from exc
    migrated = migrate_project(raw)
    from .model import scene_from_project_dict

    return scene_from_project_dict(migrated)


# ---------------------------------------------------------------------------
# Binary writer / reader


def _write_width(writer: "_BodyWriter", value: float, encoding: str) -> None:
    if encoding == "f32":
        writer.f32(value)
        return
    if encoding == "u16_fixed":
        encoded = int(round(float(value) * 256.0))
        if encoded < 0 or encoded > 0xFFFF:
            raise NPRError(E_FORMAT_WIDTH, "stroke width does not fit u16 fixed encoding", width=value)
        writer.u16(encoded)
        writer.u16(0)
        return
    raise NPRError(E_FORMAT_WIDTH, "unsupported line width encoding", encoding=encoding)


def _read_width(reader: "_BodyReader", encoding: str) -> float:
    if encoding == "f32":
        return float(reader.unpack("f")[0])
    if encoding == "u16_fixed":
        encoded = int(reader.unpack("H")[0])
        padding = int(reader.unpack("H")[0])
        if padding != 0:
            raise NPRError(E_FORMAT_WIDTH, "invalid u16 width padding")
        return encoded / 256.0
    raise NPRError(E_FORMAT_WIDTH, "unsupported line width encoding", encoding=encoding)


class _BodyWriter:
    def __init__(self, endian: str) -> None:
        self.endian = endian
        self.buffer = io.BytesIO()

    def pack(self, fmt: str, *values: Any) -> None:
        self.buffer.write(struct.pack(("<" if self.endian == "little" else ">") + fmt, *values))

    def u8(self, value: int) -> None:
        self.pack("B", int(value))

    def u16(self, value: int) -> None:
        self.pack("H", int(value))

    def u32(self, value: int) -> None:
        self.pack("I", int(value))

    def i64(self, value: int) -> None:
        self.pack("q", int(value))

    def f32(self, value: float) -> None:
        self.pack("f", float(value))

    def f64(self, value: float) -> None:
        self.pack("d", float(value))

    def blob16(self, value: bytes) -> None:
        if len(value) > 0xFFFF:
            raise NPRError(E_FORMAT_INDEX, "metadata string is too long")
        self.u16(len(value))
        self.buffer.write(value)

    def getvalue(self) -> bytes:
        return self.buffer.getvalue()


class _BodyReader:
    def __init__(self, data: bytes, endian: str) -> None:
        self.data = data
        self.endian = endian
        self.offset = 0

    def _take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.data):
            raise NPRError(E_FORMAT_INDEX, "truncated binary skeleton payload")
        chunk = self.data[self.offset : self.offset + count]
        self.offset += count
        return chunk

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        prefix = "<" if self.endian == "little" else ">"
        size = struct.calcsize(prefix + fmt)
        return struct.unpack(prefix + fmt, self._take(size))

    def u8(self) -> int:
        return int(self.unpack("B")[0])

    def u16(self) -> int:
        return int(self.unpack("H")[0])

    def u32(self) -> int:
        return int(self.unpack("I")[0])

    def i64(self) -> int:
        return int(self.unpack("q")[0])

    def f32(self) -> float:
        return float(self.unpack("f")[0])

    def f64(self) -> float:
        return float(self.unpack("d")[0])

    def blob16(self) -> bytes:
        count = self.u16()
        return self._take(count)

    def blob32(self) -> bytes:
        count = self.u32()
        if count > 64 * 1024 * 1024:
            raise NPRError(E_FORMAT_INDEX, "binary metadata blob is unreasonably large")
        return self._take(count)

    def text16(self) -> str:
        try:
            return self.blob16().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NPRError(E_FORMAT_INDEX, "binary string is not UTF-8") from exc

    def remaining(self) -> int:
        return len(self.data) - self.offset


def _sampling_to_writer(writer: _BodyWriter, sampling: Any) -> None:
    writer.blob16(sampling.mode.encode("utf-8"))
    writer.f64(sampling.tolerance_px)
    writer.u16(sampling.min_subdivisions)
    writer.u16(sampling.max_subdivisions)
    writer.f64(sampling.width_scale)


def _sampling_from_reader(reader: _BodyReader) -> StrokeSampling:
    mode = reader.text16()
    tolerance = reader.f64()
    minimum = reader.u16()
    maximum = reader.u16()
    width_scale = reader.f64()
    if tolerance <= 0.0 or minimum < 1 or maximum < minimum or width_scale <= 0.0:
        raise NPRError(E_STROKE_SAMPLE, "binary sampling configuration is invalid")
    return StrokeSampling(mode, tolerance, minimum, maximum, width_scale)


def _style_to_writer(writer: _BodyWriter, style: Any) -> None:
    writer.blob16(style.tag.encode("utf-8"))
    writer.f64(style.tone)
    writer.f64(style.roughness)
    writer.u8(style.tint[0])
    writer.u8(style.tint[1])
    writer.u8(style.tint[2])


def _style_from_reader(reader: _BodyReader) -> Any:
    return style_from_dict(
        {
            "tag": reader.text16(),
            "tone": reader.f64(),
            "roughness": reader.f64(),
            "tint": [reader.u8(), reader.u8(), reader.u8()],
        }
    )


def _write_indices(writer: _BodyWriter, indices: Iterable[int], index_width: int) -> None:
    fmt = "H" if index_width == 2 else "I"
    for index in indices:
        writer.pack(fmt, int(index))


def _read_indices(reader: _BodyReader, count: int, index_width: int) -> list[int]:
    fmt = "H" if index_width == 2 else "I"
    return [int(reader.unpack(fmt)[0]) for _ in range(count)]


def scene_to_skeleton_binary(
    scene: Scene,
    *,
    endian: str = "little",
    index_width: int = 4,
    width_encoding: str = "f32",
    viewport: tuple[int, int] | None = None,
    flat_geometry: Any | None = None,
) -> bytes:
    """Serialize a project and its *derived* skeleton into the NPRB format."""

    if endian not in ENDIAN_MARKERS:
        raise NPRError(E_FORMAT_ENDIAN, "unsupported byte order", endian=endian)
    if index_width not in (2, 4):
        raise NPRError(E_FORMAT_INDEX, "index width must be 2 or 4", index_width=index_width)
    if width_encoding not in WIDTH_ENCODINGS:
        raise NPRError(E_FORMAT_WIDTH, "line width encoding must be f32 or u16_fixed")
    target_viewport = viewport or (scene.canvas_width, scene.canvas_height)
    writer = _BodyWriter(endian)
    writer.u32(scene.canvas_width)
    writer.u32(scene.canvas_height)
    writer.i64(scene.seed)
    writer.u16(SCHEMA_VERSION)
    _sampling_to_writer(writer, scene.default_sampling)
    generator_bytes = canonical_json_bytes(scene.generator)
    writer.u32(len(generator_bytes))
    writer.buffer.write(generator_bytes)
    writer.u32(len(scene.layers))
    for layer in scene.layers:
        writer.blob16(layer.id.encode("utf-8"))
        writer.blob16(layer.name.encode("utf-8"))
        writer.u32(layer.order)
        writer.u8(1 if layer.visible else 0)
    writer.u32(len(scene.elements))
    total_vertices = 0
    total_indices = 0
    for element in scene.elements:
        if flat_geometry is not None and element.id in flat_geometry:
            cached = flat_geometry[element.id]
            if isinstance(cached, tuple) and len(cached) == 2:
                flat, skeleton = cached
            else:
                flat, skeleton = cached, None
            if skeleton is None:
                from .geometry import StrokeSkeleton

                skeleton = StrokeSkeleton(
                    element_id=element.id,
                    vertices=flat.points,
                    widths=tuple(flat.stroke_width for _ in flat.points),
                    indices=flat.indices,
                    content_hash=flat.skeleton_hash,
                )
        else:
            flat, skeleton = flatten_element(element, target_viewport=target_viewport)
        if index_width == 2 and len(flat.points) > 0xFFFF:
            raise NPRError(E_FORMAT_INDEX, "vertex count does not fit the declared index width", element_id=element.id)
        writer.blob16(element.id.encode("utf-8"))
        writer.blob16(element.layer_id.encode("utf-8"))
        writer.u32(element.layer_order)
        writer.u8(1 if element.closed else 0)
        _write_width(writer, element.stroke_width, width_encoding)
        _style_to_writer(writer, element.style)
        _sampling_to_writer(writer, element.sampling)
        writer.u16(len(element.segments))
        for segment in element.segments:
            writer.u8(KIND_TO_CODE[segment.kind])
            writer.u8(len(segment.points))
            for point in segment.points:
                writer.f64(point.x)
                writer.f64(point.y)
        writer.u32(len(flat.points))
        total_vertices += len(flat.points)
        for point in flat.points:
            writer.f64(point.x)
            writer.f64(point.y)
        writer.u32(len(skeleton.indices))
        total_indices += len(skeleton.indices)
        _write_indices(writer, skeleton.indices, index_width)
    payload = writer.getvalue()
    header = bytearray(HEADER_SIZE)
    header[0:8] = BINARY_MAGIC
    header[8:10] = ENDIAN_MARKERS[endian]
    header[10] = index_width
    header[11:13] = struct.pack(">H", BINARY_VERSION)
    header[13] = WIDTH_ENCODINGS[width_encoding]
    header[14] = 0
    header[15] = 0
    header[16:20] = struct.pack(">I", total_vertices)
    header[20:24] = struct.pack(">I", total_indices)
    header[24:28] = struct.pack(">I", len(scene.layers))
    header[28:32] = struct.pack(">I", len(scene.elements))
    digest = sha256_bytes(payload)
    header[32:64] = bytes.fromhex(digest)
    return bytes(header) + payload


def _parse_binary_header(data: bytes) -> dict[str, Any]:
    if len(data) < HEADER_SIZE:
        raise NPRError(E_FORMAT_INDEX, "binary file is shorter than its header")
    if data[0:8] != BINARY_MAGIC:
        raise NPRError(E_FORMAT_ENDIAN, "binary magic does not identify an NPR skeleton file")
    marker = data[8:10]
    if marker not in MARKER_ENDIAN:
        raise NPRError(E_FORMAT_ENDIAN, "invalid byte-order marker", marker=marker.hex())
    index_width = int(data[10])
    if index_width not in (2, 4):
        raise NPRError(E_FORMAT_INDEX, "declared index width must be 2 or 4", index_width=index_width)
    version = struct.unpack(">H", data[11:13])[0]
    if version != BINARY_VERSION:
        raise NPRError(E_FORMAT_INDEX, "unsupported binary skeleton version", version=version)
    encoding_code = int(data[13])
    if encoding_code not in ENCODING_WIDTHS:
        raise NPRError(E_FORMAT_WIDTH, "unsupported line-width encoding", code=encoding_code)
    return {
        "endian": MARKER_ENDIAN[marker],
        "index_width": index_width,
        "version": version,
        "width_encoding": ENCODING_WIDTHS[encoding_code],
        "vertex_count": struct.unpack(">I", data[16:20])[0],
        "index_count": struct.unpack(">I", data[20:24])[0],
        "layer_count": struct.unpack(">I", data[24:28])[0],
        "element_count": struct.unpack(">I", data[28:32])[0],
        "payload_hash": data[32:64].hex(),
    }


def inspect_skeleton_binary(data: bytes) -> dict[str, Any]:
    """Return the public header fields without constructing a scene."""

    header = _parse_binary_header(data)
    return {
        "magic": BINARY_MAGIC.rstrip(b"\x00").decode("ascii"),
        "endian": header["endian"],
        "index_width": header["index_width"],
        "version": header["version"],
        "line_width_encoding": header["width_encoding"],
        "vertex_count": header["vertex_count"],
        "index_count": header["index_count"],
        "layer_count": header["layer_count"],
        "element_count": header["element_count"],
        "payload_hash": header["payload_hash"],
        "file_bytes": len(data),
    }


def skeleton_binary_to_project(data: bytes) -> dict[str, Any]:
    header = _parse_binary_header(data)
    payload = data[HEADER_SIZE:]
    if sha256_bytes(payload) != header["payload_hash"]:
        raise NPRError(E_FORMAT_INDEX, "binary payload hash does not match the header")
    reader = _BodyReader(payload, header["endian"])
    width_encoding = header["width_encoding"]
    canvas_width = reader.u32()
    canvas_height = reader.u32()
    seed = reader.i64()
    schema_version = reader.u16()
    if schema_version != SCHEMA_VERSION:
        raise NPRError(E_SCHEMA_VERSION, "binary payload schema is unsupported", version=schema_version)
    default_sampling = _sampling_from_reader(reader)
    try:
        generator = json.loads(reader.blob32().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NPRError(E_FORMAT_INDEX, "generator metadata in binary payload is invalid") from exc
    layer_count = reader.u32()
    if layer_count != header["layer_count"] or layer_count > 10000:
        raise NPRError(E_FORMAT_INDEX, "binary layer count is inconsistent")
    layers: list[dict[str, Any]] = []
    for _ in range(layer_count):
        layer_id = reader.text16()
        name = reader.text16()
        order = reader.u32()
        visible = bool(reader.u8())
        layers.append({"id": layer_id, "name": name, "order": order, "visible": visible})
    element_count = reader.u32()
    if element_count != header["element_count"] or element_count > 10_000_000:
        raise NPRError(E_FORMAT_INDEX, "binary element count is inconsistent")
    elements: list[dict[str, Any]] = []
    computed_vertices = 0
    computed_indices = 0
    for _ in range(element_count):
        element_id = reader.text16()
        layer_id = reader.text16()
        layer_order = reader.u32()
        closed = bool(reader.u8())
        stroke_width = _read_width(reader, width_encoding)
        style = _style_from_reader(reader)
        sampling = _sampling_from_reader(reader)
        segment_count = reader.u16()
        if segment_count < 1:
            raise NPRError(E_OUTLINE_DEGENERATE, "binary element has no segments", element_id=element_id)
        segments: list[dict[str, Any]] = []
        for _segment_index in range(segment_count):
            kind_code = reader.u8()
            if kind_code not in CODE_TO_KIND:
                raise NPRError(E_OUTLINE_DEGENERATE, "binary segment type is invalid")
            kind = CODE_TO_KIND[kind_code]
            point_count = reader.u8()
            expected = {"line": 2, "quadratic": 3, "cubic": 4}[kind]
            if point_count != expected:
                raise NPRError(E_OUTLINE_DEGENERATE, "binary segment control-point count is invalid")
            points = [[reader.f64(), reader.f64()] for _ in range(point_count)]
            segments.append({"type": kind, "points": points})
        vertex_count = reader.u32()
        if vertex_count > 10_000_000:
            raise NPRError(E_FORMAT_INDEX, "binary element vertex count is unreasonable")
        computed_vertices += vertex_count
        for _vertex_index in range(vertex_count):
            reader.f64()
            reader.f64()
        index_count = reader.u32()
        if index_count > 40_000_000:
            raise NPRError(E_FORMAT_INDEX, "binary element index count is unreasonable")
        computed_indices += index_count
        indices = _read_indices(reader, index_count, header["index_width"])
        if any(index >= vertex_count for index in indices):
            raise NPRError(E_FORMAT_INDEX, "binary index points outside its vertex array", element_id=element_id)
        elements.append(
            {
                "id": element_id,
                "layer_id": layer_id,
                "layer_order": layer_order,
                "stroke_width": stroke_width,
                "style": style.as_dict(),
                "sampling": sampling.as_dict(),
                "closed": closed,
                "segments": segments,
            }
        )
    if reader.remaining() != 0:
        raise NPRError(E_FORMAT_INDEX, "unexpected bytes at end of binary payload", remaining=reader.remaining())
    if computed_vertices != header["vertex_count"]:
        raise NPRError(E_FORMAT_INDEX, "declared total vertex count does not match payload")
    if computed_indices != header["index_count"]:
        raise NPRError(E_FORMAT_INDEX, "declared total index count does not match payload")
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": generator,
        "canvas": {"width": canvas_width, "height": canvas_height},
        "sampling": default_sampling.as_dict(),
        "layers": layers,
        "elements": elements,
    }


def skeleton_binary_to_scene(data: bytes) -> Scene:
    from .model import scene_from_project_dict

    raw = skeleton_binary_to_project(data)
    if raw.get("schema_version") == SCHEMA_VERSION:
        return scene_from_project_dict(raw)
    return scene_from_project_dict(migrate_project(raw))


def write_skeleton_binary(path: Path, data: bytes) -> None:
    atomic_write_bytes(path, data)


def roundtrip_report(original: Scene, binary: bytes) -> dict[str, Any]:
    header = _parse_binary_header(binary)
    width_tolerance = 1.0 / 512.0 if header["width_encoding"] == "u16_fixed" else 1e-6
    recovered_raw = skeleton_binary_to_project(binary)
    recovered_elements = {str(element["id"]): element for element in recovered_raw.get("elements", [])}
    original_elements = original.element_map()
    max_point_error = 0.0
    max_width_error = 0.0
    missing = sorted(set(original_elements) ^ set(recovered_elements))
    segment_shape_error = False
    if not missing:
        for element_id, source in original_elements.items():
            target = recovered_elements[element_id]
            max_width_error = max(max_width_error, abs(source.stroke_width - float(target["stroke_width"])))
            target_segments = target.get("segments", [])
            if len(target_segments) != len(source.segments):
                segment_shape_error = True
                continue
            for source_segment, target_segment in zip(source.segments, target_segments, strict=True):
                if target_segment.get("type") != source_segment.kind:
                    segment_shape_error = True
                    continue
                target_points = target_segment.get("points", [])
                if len(target_points) != len(source_segment.points):
                    segment_shape_error = True
                    continue
                for source_point, target_point in zip(source_segment.points, target_points, strict=True):
                    max_point_error = max(
                        max_point_error,
                        abs(source_point.x - float(target_point[0])),
                        abs(source_point.y - float(target_point[1])),
                    )
    recovered_hash = sha256_json(recovered_raw)
    within = (
        not missing
        and not segment_shape_error
        and max_point_error <= 1e-12
        and max_width_error <= width_tolerance
    )
    return {
        "element_ids_match": not missing,
        "missing_ids": missing,
        "segment_shape_matches": not segment_shape_error,
        "max_control_point_error": max_point_error,
        "max_stroke_width_error": max_width_error,
        "point_tolerance": 1e-12,
        "width_tolerance": width_tolerance,
        "line_width_encoding": header["width_encoding"],
        "within_tolerance": within,
        "binary_bytes": len(binary),
        "binary_hash": sha256_bytes(binary),
        "roundtrip_scene_hash": recovered_hash,
        "roundtrip_exact_scene": recovered_hash == original.content_hash,
    }
