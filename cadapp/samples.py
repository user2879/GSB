"""Deterministic procedural sample generation.

All sample models are synthesized from closed-form math (sin/cos grids);
no external CAD files, no network, no randomness.  The same call always
produces byte-identical models.
"""

from __future__ import annotations

import json
import math

from .model import Model, SurfacePatch


def _clamped_uniform_knots(ctrl_count: int, degree: int) -> list[float]:
    interior_count = ctrl_count - degree - 1
    knots = [0.0] * (degree + 1)
    for k in range(1, interior_count + 1):
        knots.append(k / (interior_count + 1))
    knots += [1.0] * (degree + 1)
    return knots


def _make_patch(patch_id: str, degree_u: int, degree_v: int,
                nu: int, nv: int, height_fn, offset,
                weight_fn=None) -> SurfacePatch:
    control_points = []
    weights = []
    for i in range(nu):
        row = []
        wrow = []
        for j in range(nv):
            x = offset[0] + (i / (nu - 1) - 0.5) * 1.8
            y = offset[1] + (j / (nv - 1) - 0.5) * 1.8
            z = offset[2] + height_fn(i / (nu - 1), j / (nv - 1))
            row.append([x, y, z])
            wrow.append(weight_fn(i, j) if weight_fn else 1.0)
        control_points.append(row)
        weights.append(wrow)
    return SurfacePatch(
        patch_id=patch_id,
        degree_u=degree_u,
        degree_v=degree_v,
        knots_u=_clamped_uniform_knots(nu, degree_u),
        knots_v=_clamped_uniform_knots(nv, degree_v),
        control_points=control_points,
        weights=weights,
        material_channel="default",
        tangent_space=False,
    )


def make_sample_model() -> Model:
    """Three-patch NURBS model: wave sheet, rational dome, saddle."""
    wave = _make_patch(
        "patch_wave", 3, 3, 4, 4,
        lambda u, v: 0.25 * math.sin(2.0 * math.pi * u) * math.cos(2.0 * math.pi * v),
        offset=(-1.4, 0.0, 0.0),
    )
    dome = _make_patch(
        "patch_dome", 3, 3, 5, 5,
        lambda u, v: 0.6 * math.cos(math.pi * (u - 0.5)) * math.cos(math.pi * (v - 0.5)),
        offset=(0.6, 0.0, 0.2),
        weight_fn=lambda i, j: 1.0 + 0.25 * math.sin(i + j),
    )
    saddle = _make_patch(
        "patch_saddle", 3, 2, 4, 3,
        lambda u, v: 0.5 * ((u - 0.5) ** 2 - (v - 0.5) ** 2),
        offset=(0.0, 1.6, -0.1),
    )
    model = Model(model_id="sample_model", patches=[wave, dome, saddle])
    model.validate()
    return model


def make_legacy_v1_document() -> dict:
    """A schema-v1 JSON document: same geometry, new fields stripped."""
    model = make_sample_model()
    doc = {
        "schema_version": 1,
        "model_id": model.model_id,
        "patches": [],
    }
    for p in model.patches:
        doc["patches"].append({
            "id": p.patch_id,
            "knots_u": list(p.knots_u),
            "knots_v": list(p.knots_v),
            "control_points": [[list(pt) for pt in row] for row in p.control_points],
            "weights": [list(row) for row in p.weights],
        })
    return doc


def legacy_v1_json_text() -> str:
    return json.dumps(make_legacy_v1_document(), sort_keys=True,
                      separators=(",", ":"))
