"""Binary mesh exchange format (BMESH).

Layout (all multi-byte header fields use the endianness declared by the
endian marker byte):

  offset  size  field
  0       4     magic "BMSH"
  4       1     format_version (1 or 2)
  5       1     endian marker: 0x01 = little-endian, 0x02 = big-endian
  6       1     index width in bits: 16 or 32
  7       1     flags: bit0 = normals present, bit1 = interleaved layout
  8       4     vertex_count (uint32)
  12      4     face_count (uint32)
  16      ..    payload

Payload, planar layout (bit1 clear):
  vertices : vertex_count * 3 * float64
  normals  : vertex_count * 3 * float64   (only when bit0 set)
  faces    : face_count * 3 * (uint16 | uint32)

Payload, interleaved layout (bit1 set, requires bit0):
  per vertex: 3 * float64 position followed by 3 * float64 normal
  faces    : face_count * 3 * (uint16 | uint32)

Optional trailing source-model chunk (allows lossless binary -> JSON
round trips of the originating parametric model):
  "JSRC" + uint32 length + canonical JSON model bytes

Format version 1 files are always little-endian, 16-bit indices, planar
layout with normals.  Version 2 allows every combination of the header
fields.  Conversion between versions rewrites the container only; vertex
and normal payloads stay float64, so conversion is bit-exact.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from .errors import CadError, ErrorCode

MAGIC = b"BMSH"
SOURCE_MAGIC = b"JSRC"
HEADER_SIZE = 16

ENDIAN_MARKERS = {"little": 0x01, "big": 0x02}
MARKER_ENDIANS = {0x01: "little", 0x02: "big"}
STRUCT_PREFIX = {"little": "<", "big": ">"}

FLAG_NORMALS = 0x01
FLAG_INTERLEAVED = 0x02

FORMAT_VERSION = 2


@dataclass
class MeshData:
    vertices: np.ndarray            # (N, 3) float64
    normals: np.ndarray | None      # (N, 3) float64 or None
    faces: np.ndarray               # (M, 3) int64
    source_model_bytes: bytes | None = None


def _check_faces_in_range(faces: np.ndarray, vertex_count: int, location: str) -> None:
    if faces.size == 0:
        return
    bad = np.argwhere(faces >= vertex_count)
    if bad.size:
        face_i, corner = (int(x) for x in bad[0])
        raise CadError(
            ErrorCode.E_INDEX_OVERFLOW,
            f"face index {int(faces[face_i, corner])} out of range "
            f"(vertex_count={vertex_count})",
            location=f"{location}.faces[{face_i}][{corner}]",
        )


def write_mesh(vertices, normals, faces, *, version: int = FORMAT_VERSION,
               endian: str = "little", index_width: int = 32,
               interleaved: bool = False,
               source_model_bytes: bytes | None = None) -> bytes:
    """Serialize a mesh.  All float payloads are float64 (lossless)."""
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise CadError(ErrorCode.E_FORMAT_HEADER, "vertices must be (N, 3)", "vertices")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise CadError(ErrorCode.E_FORMAT_HEADER, "faces must be (M, 3)", "faces")
    vertex_count = int(vertices.shape[0])
    _check_faces_in_range(faces, vertex_count, "mesh")
    if endian not in ENDIAN_MARKERS:
        raise CadError(ErrorCode.E_FORMAT_ENDIAN,
                       f"unknown endianness {endian!r}", "header.endian")
    if index_width not in (16, 32):
        raise CadError(ErrorCode.E_FORMAT_HEADER,
                       f"index width must be 16 or 32, got {index_width}",
                       "header.index_width")
    if index_width == 16 and vertex_count > 0x10000:
        raise CadError(
            ErrorCode.E_INDEX_OVERFLOW,
            f"vertex_count {vertex_count} does not fit 16-bit indices; "
            f"refusing lossy conversion",
            location="header.index_width",
        )
    if version == 1:
        # Declared v1 profile: little-endian, 16-bit, planar, with normals.
        endian, index_width, interleaved = "little", 16, False
        if vertex_count > 0x10000:
            raise CadError(ErrorCode.E_INDEX_OVERFLOW,
                           "vertex_count exceeds v1 16-bit index range",
                           "header.index_width")
        if normals is None:
            raise CadError(ErrorCode.E_NORMAL_COUNT,
                           "format version 1 requires vertex normals", "normals")
    has_normals = normals is not None
    if has_normals:
        normals = np.ascontiguousarray(normals, dtype=np.float64)
        if normals.shape != vertices.shape:
            raise CadError(
                ErrorCode.E_NORMAL_COUNT,
                f"normal count {normals.shape[0]} != vertex count {vertex_count}",
                location="normals",
            )
    if interleaved and not has_normals:
        raise CadError(ErrorCode.E_NORMAL_COUNT,
                       "interleaved layout requires normals", "header.flags")

    prefix = STRUCT_PREFIX[endian]
    flags = (FLAG_NORMALS if has_normals else 0) | (FLAG_INTERLEAVED if interleaved else 0)
    header = bytearray()
    header += MAGIC
    header += struct.pack("<BBBB", version, ENDIAN_MARKERS[endian], index_width, flags)
    header += struct.pack(prefix + "II", vertex_count, int(faces.shape[0]))

    body = bytearray()
    if interleaved:
        block = np.empty((vertex_count, 6), dtype=np.float64)
        block[:, 0:3] = vertices
        block[:, 3:6] = normals
        body += block.astype(prefix + "f8").tobytes()
    else:
        body += vertices.astype(prefix + "f8").tobytes()
        if has_normals:
            body += normals.astype(prefix + "f8").tobytes()
    index_dtype = prefix + ("u2" if index_width == 16 else "u4")
    body += faces.astype(index_dtype).tobytes()

    out = bytes(header) + bytes(body)
    if source_model_bytes is not None:
        out += SOURCE_MAGIC + struct.pack(prefix + "I", len(source_model_bytes))
        out += source_model_bytes
    return out


def read_mesh(data: bytes):
    """Parse BMESH bytes.  Returns (MeshData, summary dict)."""
    if len(data) < HEADER_SIZE:
        raise CadError(ErrorCode.E_FORMAT_TRUNCATED,
                       f"file too short for header ({len(data)} bytes)", "header")
    if data[0:4] != MAGIC:
        raise CadError(ErrorCode.E_FORMAT_MAGIC,
                       f"bad magic {data[0:4]!r}", "header.magic")
    version, endian_marker, index_width, flags = struct.unpack("<BBBB", data[4:8])
    if version not in (1, 2):
        raise CadError(ErrorCode.E_FORMAT_HEADER,
                       f"unsupported format version {version}", "header.format_version")
    if endian_marker not in MARKER_ENDIANS:
        raise CadError(ErrorCode.E_FORMAT_ENDIAN,
                       f"unrecognized endian marker 0x{endian_marker:02x}",
                       "header.endian")
    if index_width not in (16, 32):
        raise CadError(ErrorCode.E_FORMAT_HEADER,
                       f"unsupported index width {index_width}", "header.index_width")
    endian = MARKER_ENDIANS[endian_marker]
    prefix = STRUCT_PREFIX[endian]
    vertex_count, face_count = struct.unpack(prefix + "II", data[8:16])
    has_normals = bool(flags & FLAG_NORMALS)
    interleaved = bool(flags & FLAG_INTERLEAVED)

    offset = HEADER_SIZE
    f8 = np.dtype(prefix + "f8")

    def take(nbytes: int, location: str, code: str = ErrorCode.E_FORMAT_TRUNCATED):
        nonlocal offset
        if len(data) - offset < nbytes:
            raise CadError(code,
                           f"payload truncated: need {nbytes} bytes, "
                           f"have {len(data) - offset}", location)
        chunk = data[offset:offset + nbytes]
        offset += nbytes
        return chunk

    if interleaved:
        raw = take(vertex_count * 6 * 8, "vertices")
        block = np.frombuffer(raw, dtype=f8).reshape(vertex_count, 6).astype(np.float64)
        vertices = block[:, 0:3].copy()
        normals = block[:, 3:6].copy()
    else:
        raw = take(vertex_count * 3 * 8, "vertices")
        vertices = np.frombuffer(raw, dtype=f8).reshape(vertex_count, 3).astype(np.float64)
        normals = None
        if has_normals:
            raw = take(vertex_count * 3 * 8, "normals", ErrorCode.E_NORMAL_COUNT)
            normals = np.frombuffer(raw, dtype=f8).reshape(vertex_count, 3).astype(np.float64)
    if has_normals and normals is not None and normals.shape[0] != vertex_count:
        raise CadError(ErrorCode.E_NORMAL_COUNT,
                       f"normal count {normals.shape[0]} != vertex count {vertex_count}",
                       "normals")

    index_nbytes = 2 if index_width == 16 else 4
    raw = take(face_count * 3 * index_nbytes, "faces")
    index_dtype = np.dtype(prefix + ("u2" if index_width == 16 else "u4"))
    faces = np.frombuffer(raw, dtype=index_dtype).reshape(face_count, 3).astype(np.int64)
    _check_faces_in_range(faces, vertex_count, "mesh")

    source_model_bytes = None
    if len(data) > offset:
        if data[offset:offset + 4] != SOURCE_MAGIC:
            raise CadError(ErrorCode.E_FORMAT_MAGIC,
                           "unexpected trailing bytes after faces",
                           f"offset {offset}")
        (length,) = struct.unpack(prefix + "I", data[offset + 4:offset + 8])
        source_model_bytes = data[offset + 8:offset + 8 + length]
        if len(source_model_bytes) != length:
            raise CadError(ErrorCode.E_FORMAT_TRUNCATED,
                           "source model chunk truncated", "source_model")
        if len(data) != offset + 8 + length:
            raise CadError(ErrorCode.E_FORMAT_MAGIC,
                           "unexpected bytes after source model chunk",
                           f"offset {offset + 8 + length}")

    mesh = MeshData(vertices=vertices, normals=normals, faces=faces,
                    source_model_bytes=source_model_bytes)
    summary = {
        "format_version": version,
        "endianness": endian,
        "index_width": index_width,
        "interleaved": interleaved,
        "has_normals": has_normals,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "bbox_min": [float(x) for x in vertices.min(axis=0)] if vertex_count else [0.0] * 3,
        "bbox_max": [float(x) for x in vertices.max(axis=0)] if vertex_count else [0.0] * 3,
        "payload_sha256": hashlib.sha256(data[HEADER_SIZE:offset]).hexdigest(),
        "has_source_model": source_model_bytes is not None,
    }
    return mesh, summary


def convert_mesh(data: bytes, *, version: int = FORMAT_VERSION,
                 endian: str = "little", index_width: int = 32,
                 interleaved: bool = False) -> bytes:
    """Re-encode BMESH bytes into another container profile, losslessly."""
    mesh, _ = read_mesh(data)
    return write_mesh(mesh.vertices, mesh.normals, mesh.faces,
                      version=version, endian=endian, index_width=index_width,
                      interleaved=interleaved,
                      source_model_bytes=mesh.source_model_bytes)
