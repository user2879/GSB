"""Deterministic binary little-endian PLY writer and strict reader."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np

from .errors import BusinessError, ErrorCode


_HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "comment deterministic mesh repair benchmark\n"
    "element vertex {vertex_count}\n"
    "property double x\n"
    "property double y\n"
    "property double z\n"
    "element face {face_count}\n"
    "property list uchar int vertex_indices\n"
    "end_header\n"
)


def _validate_arrays(vertices: np.ndarray, faces: np.ndarray) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise BusinessError(ErrorCode.E_MESH_FORMAT, "vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise BusinessError(ErrorCode.E_MESH_FORMAT, "faces must have shape (M, 3)")
    if not np.issubdtype(vertices.dtype, np.floating):
        raise BusinessError(ErrorCode.E_MESH_FORMAT, "vertices must be floating point")
    if not np.issubdtype(faces.dtype, np.integer):
        raise BusinessError(ErrorCode.E_MESH_FORMAT, "faces must be integers")
    if not np.all(np.isfinite(vertices)):
        raise BusinessError(ErrorCode.E_MESH_FORMAT, "vertices contain NaN or infinity")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise BusinessError(
            ErrorCode.E_MESH_FORMAT,
            "face index is outside the vertex array",
            details={"vertex_count": int(len(vertices)), "face_count": int(len(faces))},
        )


def write_ply(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a canonical PLY file.

    The byte stream depends only on the arrays: no timestamps, platform paths or
    dictionary ordering are included.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    _validate_arrays(vertices, faces)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = _HEADER_TEMPLATE.format(
        vertex_count=len(vertices), face_count=len(faces)
    ).encode("ascii")

    vertex_dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    vertex_records = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_records["x"] = vertices[:, 0]
    vertex_records["y"] = vertices[:, 1]
    vertex_records["z"] = vertices[:, 2]
    face_records = np.empty(len(faces), dtype=face_dtype)
    face_records["count"] = 3
    face_records["indices"] = faces.astype("<i4", copy=False)

    with target.open("wb") as handle:
        handle.write(header)
        handle.write(vertex_records.tobytes(order="C"))
        handle.write(face_records.tobytes(order="C"))


def _read_exact(handle: BinaryIO, size: int, context: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise BusinessError(
            ErrorCode.E_MESH_FORMAT,
            f"truncated PLY while reading {context}",
            details={"expected_bytes": size, "actual_bytes": len(data)},
        )
    return data


def _read_header(handle: BinaryIO) -> tuple[int, int]:
    first = handle.readline()
    if first not in (b"ply\n", b"ply\r\n"):
        raise BusinessError(ErrorCode.E_MESH_FORMAT, "PLY magic header is missing")

    vertex_count: int | None = None
    face_count: int | None = None
    current_element: str | None = None
    vertex_properties: list[str] = []
    face_property: list[str] | None = None

    for _ in range(256):
        raw = handle.readline()
        if not raw:
            break
        try:
            line = raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise BusinessError(ErrorCode.E_MESH_FORMAT, "PLY header is not ASCII") from exc
        if line == "end_header":
            break
        if line.startswith("format "):
            if line != "format binary_little_endian 1.0":
                raise BusinessError(
                    ErrorCode.E_MESH_FORMAT,
                    "only binary_little_endian PLY version 1.0 is supported",
                    details={"format": line},
                )
        elif line.startswith("element "):
            parts = line.split()
            if len(parts) != 3:
                raise BusinessError(ErrorCode.E_MESH_FORMAT, "invalid PLY element line")
            current_element = parts[1]
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise BusinessError(ErrorCode.E_MESH_FORMAT, "invalid PLY element count") from exc
            if count < 0:
                raise BusinessError(ErrorCode.E_MESH_FORMAT, "negative PLY element count")
            if current_element == "vertex":
                vertex_count = count
            elif current_element == "face":
                face_count = count
        elif line.startswith("property "):
            parts = line.split()
            if current_element == "vertex":
                vertex_properties.append(" ".join(parts[1:]))
            elif current_element == "face":
                face_property = parts[1:]
        elif line.startswith("comment "):
            continue
        else:
            raise BusinessError(
                ErrorCode.E_MESH_FORMAT,
                "unsupported PLY header declaration",
                details={"line": line},
            )
    else:
        raise BusinessError(ErrorCode.E_MESH_FORMAT, "PLY header is too long")

    if vertex_count is None or face_count is None:
        raise BusinessError(
            ErrorCode.E_MESH_FORMAT, "PLY must declare vertex and face elements"
        )
    if vertex_properties != ["double x", "double y", "double z"]:
        raise BusinessError(
            ErrorCode.E_MESH_FORMAT,
            "PLY vertex properties must be double x/y/z",
            details={"properties": vertex_properties},
        )
    if face_property != ["list", "uchar", "int", "vertex_indices"]:
        raise BusinessError(
            ErrorCode.E_MESH_FORMAT,
            "PLY face property must be list uchar int vertex_indices",
            details={"property": face_property},
        )
    return vertex_count, face_count


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    target = Path(path)
    try:
        handle = target.open("rb")
    except OSError as exc:
        raise BusinessError(
            ErrorCode.E_MESH_FORMAT,
            "cannot open PLY file",
            details={"path": str(target), "error": str(exc)},
        ) from exc

    with handle:
        vertex_count, face_count = _read_header(handle)
        vertex_dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
        vertex_bytes = _read_exact(handle, vertex_count * vertex_dtype.itemsize, "vertices")
        face_bytes = _read_exact(handle, face_count * face_dtype.itemsize, "faces")
        trailing = handle.read(1)
        if trailing:
            raise BusinessError(ErrorCode.E_MESH_FORMAT, "trailing bytes after PLY payload")

    try:
        vertex_records = np.frombuffer(vertex_bytes, dtype=vertex_dtype, count=vertex_count)
        face_records = np.frombuffer(face_bytes, dtype=face_dtype, count=face_count)
    except ValueError as exc:
        raise BusinessError(ErrorCode.E_MESH_FORMAT, "invalid PLY payload size") from exc

    if face_count and np.any(face_records["count"] != 3):
        bad = int(np.nonzero(face_records["count"] != 3)[0][0])
        raise BusinessError(
            ErrorCode.E_MESH_FORMAT,
            "only triangular faces are supported",
            location={"face": bad},
        )

    vertices = np.column_stack(
        [vertex_records["x"], vertex_records["y"], vertex_records["z"]]
    ).astype(np.float64, copy=False)
    faces = face_records["indices"].astype(np.int64, copy=False)
    _validate_arrays(vertices, faces)
    return vertices, faces
