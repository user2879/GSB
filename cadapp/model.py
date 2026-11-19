"""Model data structures and canonical hashing.

Canonical JSON bytes (sorted keys, tight separators, Python float repr)
give a deterministic content hash for every model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

from . import SCHEMA_VERSION
from .errors import CadError, ErrorCode
from .geometry import validate_knot_vector


@dataclass
class SurfacePatch:
    patch_id: str
    degree_u: int
    degree_v: int
    knots_u: list
    knots_v: list
    control_points: list  # list[list[[x, y, z]]]
    weights: list         # list[list[float]]
    material_channel: str = "default"
    tangent_space: bool = False

    def validate(self, location: str) -> None:
        nu = len(self.control_points)
        nv = len(self.control_points[0])
        validate_knot_vector(self.knots_u, self.degree_u, nu, f"{location}.knots_u")
        validate_knot_vector(self.knots_v, self.degree_v, nv, f"{location}.knots_v")
        for row in self.control_points:
            if len(row) != nv:
                raise CadError(
                    ErrorCode.E_FORMAT_HEADER,
                    "control point rows have inconsistent lengths",
                    location=f"{location}.control_points",
                )
        if len(self.weights) != nu or any(len(r) != nv for r in self.weights):
            raise CadError(
                ErrorCode.E_FORMAT_HEADER,
                "weight grid shape does not match control point grid",
                location=f"{location}.weights",
            )


@dataclass
class Model:
    model_id: str
    patches: list = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        for i, patch in enumerate(self.patches):
            patch.validate(f"patches[{i}]({patch.patch_id})")

    def patch_by_id(self, patch_id: str) -> SurfacePatch:
        for patch in self.patches:
            if patch.patch_id == patch_id:
                return patch
        raise CadError(ErrorCode.E_FORMAT_HEADER, f"unknown patch id {patch_id!r}",
                       location="patches")

    def clone(self) -> "Model":
        return copy.deepcopy(self)


def model_to_dict(model: Model) -> dict:
    return {
        "schema_version": model.schema_version,
        "model_id": model.model_id,
        "patches": [
            {
                "id": p.patch_id,
                "degree_u": p.degree_u,
                "degree_v": p.degree_v,
                "knots_u": list(p.knots_u),
                "knots_v": list(p.knots_v),
                "control_points": [[list(pt) for pt in row] for row in p.control_points],
                "weights": [list(row) for row in p.weights],
                "material_channel": p.material_channel,
                "tangent_space": p.tangent_space,
            }
            for p in model.patches
        ],
    }


def model_from_dict(doc: dict) -> Model:
    if not isinstance(doc, dict):
        raise CadError(ErrorCode.E_FORMAT_HEADER, "model document must be an object", "<root>")
    version = doc.get("schema_version")
    if not isinstance(version, int):
        raise CadError(ErrorCode.E_SCHEMA_VERSION, "missing or invalid schema_version",
                       "schema_version")
    patches = []
    for i, pdoc in enumerate(doc.get("patches", [])):
        loc = f"patches[{i}]"
        try:
            patch = SurfacePatch(
                patch_id=pdoc["id"],
                degree_u=int(pdoc["degree_u"]),
                degree_v=int(pdoc["degree_v"]),
                knots_u=[float(x) for x in pdoc["knots_u"]],
                knots_v=[float(x) for x in pdoc["knots_v"]],
                control_points=[
                    [[float(c) for c in pt] for pt in row]
                    for row in pdoc["control_points"]
                ],
                weights=[[float(w) for w in row] for row in pdoc["weights"]],
                material_channel=str(pdoc["material_channel"]),
                tangent_space=bool(pdoc["tangent_space"]),
            )
        except KeyError as exc:
            raise CadError(ErrorCode.E_FORMAT_HEADER,
                           f"missing required field {exc}", loc) from exc
        patch.validate(f"{loc}({patch.patch_id})")
        patches.append(patch)
    model = Model(model_id=str(doc.get("model_id", "model")),
                  patches=patches, schema_version=version)
    return model


def canonical_bytes(model: Model) -> bytes:
    return json.dumps(model_to_dict(model), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def content_hash(model: Model) -> str:
    return hashlib.sha256(canonical_bytes(model)).hexdigest()
