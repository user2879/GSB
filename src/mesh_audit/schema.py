from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import digest_json, sha256_hex
from .errors import BusinessError, ErrorCode


SCRIPT_SCHEMA_VERSION = 1
ALLOWED_OPERATORS = {
    "hole_triangulation",
    "remove_degenerate_faces",
    "reorient_normals",
    "laplacian_smooth",
}
TOP_SCRIPT_KEYS = {
    "schema_version",
    "coordinate_tolerance",
    "area_tolerance",
    "fixed_scale",
    "trace_radius",
    "concurrency",
    "memory_budget_nodes",
    "operators",
}
OPERATOR_PARAM_KEYS: dict[str, set[str]] = {
    "hole_triangulation": set(),
    "remove_degenerate_faces": set(),
    "reorient_normals": set(),
    "laplacian_smooth": {"iterations", "lambda"},
}


@dataclass(frozen=True, slots=True)
class MeshInput:
    vertices: list[list[float]]
    faces: list[list[int]]
    mesh_digest: str
    source_path: str
    source_format: str

    def content_dict(self) -> dict[str, Any]:
        return {"vertices": self.vertices, "faces": self.faces}


@dataclass(frozen=True, slots=True)
class RepairScript:
    coordinate_tolerance: float
    area_tolerance: float
    fixed_scale: int
    trace_radius: int
    concurrency: int
    memory_budget_nodes: int
    operators: tuple[dict[str, Any], ...]
    normalized: dict[str, Any]
    script_digest: str


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_positive(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value) > 0.0


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def load_json_file(path: str | Path) -> Any:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError(
            ErrorCode.E_IO, "cannot read JSON file", {"path": str(source), "reason": str(exc)}
        ) from exc
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "invalid JSON",
            {"path": str(source), "reason": str(exc)},
        ) from exc


def _validate_point(point: Any, vertex: int, source: str) -> list[float]:
    if not isinstance(point, list) or len(point) != 3:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "each vertex must be a three-coordinate array",
            {"source": source, "vertex": vertex},
        )
    result: list[float] = []
    for axis, value in enumerate(point):
        if not _is_number(value) or not math.isfinite(float(value)):
            raise BusinessError(
                ErrorCode.E_SCHEMA,
                "vertex coordinates must be finite numbers",
                {"source": source, "vertex": vertex, "axis": axis},
            )
        result.append(float(value))
    return result


def _validate_face(face: Any, face_index: int, vertex_count: int, source: str) -> list[int]:
    if not isinstance(face, list) or len(face) != 3:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "each face must contain exactly three indices",
            {"source": source, "face": face_index},
        )
    result: list[int] = []
    for corner, value in enumerate(face):
        if not isinstance(value, int) or isinstance(value, bool):
            raise BusinessError(
                ErrorCode.E_SCHEMA,
                "face indices must be integers",
                {"source": source, "face": face_index, "corner": corner},
            )
        if value < 0 or value >= vertex_count:
            raise BusinessError(
                ErrorCode.E_INDEX,
                "face index is outside the vertex array",
                {
                    "source": source,
                    "face": face_index,
                    "corner": corner,
                    "index": value,
                    "vertex_count": vertex_count,
                },
            )
        result.append(value)
    return result


def validate_mesh_payload(payload: Any, source: str = "<memory>") -> MeshInput:
    if not isinstance(payload, dict):
        raise BusinessError(
            ErrorCode.E_SCHEMA, "mesh root must be a JSON object", {"source": source}
        )
    if "vertices" not in payload or "faces" not in payload:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "mesh requires vertices and faces arrays",
            {"source": source},
        )
    raw_vertices = payload["vertices"]
    raw_faces = payload["faces"]
    if not isinstance(raw_vertices, list) or len(raw_vertices) < 3:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "vertices must be an array with at least three entries",
            {"source": source},
        )
    if not isinstance(raw_faces, list) or len(raw_faces) < 1:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "faces must be a non-empty array",
            {"source": source},
        )
    vertices = [
        _validate_point(point, index, source) for index, point in enumerate(raw_vertices)
    ]
    faces = [
        _validate_face(face, index, len(vertices), source)
        for index, face in enumerate(raw_faces)
    ]
    content = {"vertices": vertices, "faces": faces}
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_digest=digest_json(content),
        source_path=source,
        source_format="json",
    )


def _parse_obj_face(token: str, vertex_count: int, face_index: int, source: str) -> int:
    head = token.split("/", 1)[0]
    try:
        raw = int(head)
    except ValueError as exc:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "OBJ face index must be an integer",
            {"source": source, "face": face_index, "token": token},
        ) from exc
    index = raw - 1 if raw > 0 else vertex_count + raw
    if index < 0 or index >= vertex_count:
        raise BusinessError(
            ErrorCode.E_INDEX,
            "OBJ face index is outside the vertex array",
            {"source": source, "face": face_index, "index": raw, "vertex_count": vertex_count},
        )
    return index


def _parse_obj_vertex(parts: list[str], vertex_index: int, source: str) -> list[float]:
    try:
        values = [float(parts[index]) for index in range(1, 4)]
    except ValueError as exc:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "OBJ vertex coordinates must be finite numbers",
            {"source": source, "vertex": vertex_index},
        ) from exc
    return _validate_point(values, vertex_index, source)


def load_obj_mesh(path: str | Path) -> MeshInput:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BusinessError(
            ErrorCode.E_IO, "cannot read OBJ mesh", {"path": str(source), "reason": str(exc)}
        ) from exc
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        keyword = parts[0]
        if keyword == "v":
            if len(parts) < 4:
                raise BusinessError(
                    ErrorCode.E_SCHEMA,
                    "OBJ vertex requires three coordinates",
                    {"source": str(source), "line": line_number},
                )
            vertices.append(_parse_obj_vertex(parts, len(vertices), str(source)))
        elif keyword == "f":
            if len(parts) != 4:
                raise BusinessError(
                    ErrorCode.E_SCHEMA,
                    "only triangular OBJ faces are accepted",
                    {"source": str(source), "line": line_number, "corners": len(parts) - 1},
                )
            face_index = len(faces)
            faces.append(
                [
                    _parse_obj_face(token, len(vertices), face_index, str(source))
                    for token in parts[1:4]
                ]
            )
    return validate_mesh_payload({"vertices": vertices, "faces": faces}, str(source))


def load_mesh_file(path: str | Path) -> MeshInput:
    source = Path(path)
    if source.suffix.lower() == ".obj":
        mesh = load_obj_mesh(source)
    else:
        mesh = validate_mesh_payload(load_json_file(source), str(source))
    return MeshInput(
        vertices=mesh.vertices,
        faces=mesh.faces,
        mesh_digest=mesh.mesh_digest,
        source_path=str(source),
        source_format=source.suffix.lower().lstrip(".") or "json",
    )


def validate_repair_script(payload: Any, source: str = "<memory>") -> RepairScript:
    if not isinstance(payload, dict):
        raise BusinessError(
            ErrorCode.E_SCHEMA, "repair script root must be a JSON object", {"source": source}
        )
    unknown = sorted(set(payload) - TOP_SCRIPT_KEYS)
    if unknown:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "repair script contains unknown fields",
            {"source": source, "unknown": unknown},
        )
    missing = sorted({"coordinate_tolerance", "operators"} - set(payload))
    if missing:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "repair script is missing required fields",
            {"source": source, "missing": missing},
        )
    schema_version = payload.get("schema_version", SCRIPT_SCHEMA_VERSION)
    if schema_version != SCRIPT_SCHEMA_VERSION:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "unsupported repair-script schema version",
            {"source": source, "schema_version": schema_version},
        )
    if not _finite_positive(payload["coordinate_tolerance"]):
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "coordinate_tolerance must be a finite positive number",
            {"source": source},
        )
    coordinate_tolerance = float(payload["coordinate_tolerance"])
    area_tolerance_value = payload.get(
        "area_tolerance", coordinate_tolerance * coordinate_tolerance
    )
    if not _finite_positive(area_tolerance_value):
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "area_tolerance must be a finite positive number",
            {"source": source},
        )
    fixed_scale = payload.get("fixed_scale", 1_000_000)
    if not _positive_int(fixed_scale) or fixed_scale > 1_000_000_000_000:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "fixed_scale must be a positive integer no larger than 1e12",
            {"source": source},
        )
    trace_radius = payload.get("trace_radius", 4)
    if not _positive_int(trace_radius) or trace_radius > 10_000:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "trace_radius must be a positive integer no larger than 10000",
            {"source": source},
        )
    concurrency = payload.get("concurrency", 4)
    if not _positive_int(concurrency) or concurrency > 64:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "concurrency must be an integer from 1 through 64",
            {"source": source},
        )
    memory_budget = payload.get("memory_budget_nodes", 1_000_000)
    if not _positive_int(memory_budget):
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "memory_budget_nodes must be a positive integer",
            {"source": source},
        )
    raw_operators = payload["operators"]
    if not isinstance(raw_operators, list) or not raw_operators:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "operators must be a non-empty array",
            {"source": source},
        )
    operators: list[dict[str, Any]] = []
    for index, raw_operator in enumerate(raw_operators):
        if not isinstance(raw_operator, dict):
            raise BusinessError(
                ErrorCode.E_SCHEMA,
                "operator entry must be an object",
                {"source": source, "operator_index": index},
            )
        operator_type = raw_operator.get("type")
        if operator_type not in ALLOWED_OPERATORS:
            raise BusinessError(
                ErrorCode.E_SCHEMA,
                "operator type is not in the allowed set",
                {
                    "source": source,
                    "operator_index": index,
                    "type": operator_type,
                    "allowed": sorted(ALLOWED_OPERATORS),
                },
            )
        unknown_params = sorted(set(raw_operator) - {"type"} - OPERATOR_PARAM_KEYS[operator_type])
        if unknown_params:
            raise BusinessError(
                ErrorCode.E_SCHEMA,
                "operator contains unknown parameters",
                {
                    "source": source,
                    "operator_index": index,
                    "unknown": unknown_params,
                },
            )
        normalized_operator: dict[str, Any] = {"type": operator_type}
        if operator_type == "laplacian_smooth":
            iterations = raw_operator.get("iterations")
            lam = raw_operator.get("lambda")
            if not _positive_int(iterations):
                raise BusinessError(
                    ErrorCode.E_SCHEMA,
                    "laplacian iterations must be a positive integer",
                    {"source": source, "operator_index": index},
                )
            if not _finite_positive(lam) or float(lam) > 1.0:
                raise BusinessError(
                    ErrorCode.E_SCHEMA,
                    "laplacian lambda must be a finite number in (0, 1]",
                    {"source": source, "operator_index": index},
                )
            normalized_operator["iterations"] = iterations
            normalized_operator["lambda"] = float(lam)
        operators.append(normalized_operator)

    normalized = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "coordinate_tolerance": coordinate_tolerance,
        "area_tolerance": float(area_tolerance_value),
        "fixed_scale": fixed_scale,
        "trace_radius": trace_radius,
        "concurrency": concurrency,
        "memory_budget_nodes": memory_budget,
        "operators": operators,
    }
    return RepairScript(
        coordinate_tolerance=coordinate_tolerance,
        area_tolerance=float(area_tolerance_value),
        fixed_scale=fixed_scale,
        trace_radius=trace_radius,
        concurrency=concurrency,
        memory_budget_nodes=memory_budget,
        operators=tuple(operators),
        normalized=normalized,
        script_digest=digest_json(normalized),
    )


def load_repair_script(path: str | Path) -> RepairScript:
    return validate_repair_script(load_json_file(path), str(path))


def make_version_id(mesh_digest: str, script_digest: str) -> str:
    return sha256_hex(f"MESH-AUDIT-V1\0{mesh_digest}\0{script_digest}".encode("utf-8"))
