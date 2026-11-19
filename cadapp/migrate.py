"""Schema migration rules.

Migration table v1 -> v2 (applied per patch, in document order):
  degree_u / degree_v : inferred as len(knots) - len(control_points) - 1.
                        This is exact for every well-formed v1 file, so the
                        inferred degree reproduces the original geometry
                        bit-for-bit (isocurves and shaded shape unchanged).
  material_channel    : constant default "default" (display-only field).
  tangent_space       : constant default false (display-only field).
  schema_version      : set to 2.

Migration is idempotent: a document already at the current schema version
is returned unchanged (no actions, identical bytes and hash).
"""

from __future__ import annotations

import copy

from . import SCHEMA_VERSION
from .errors import CadError, ErrorCode


def _infer_degree(knots, ctrl_count: int) -> int:
    return len(knots) - ctrl_count - 1


def _migrate_1_to_2(doc: dict, actions: list) -> None:
    for i, pdoc in enumerate(doc.get("patches", [])):
        loc = f"patches[{i}]({pdoc.get('id', '?')})"
        if "degree_u" not in pdoc:
            value = _infer_degree(pdoc["knots_u"], len(pdoc["control_points"]))
            pdoc["degree_u"] = value
            actions.append({"patch": pdoc.get("id"), "field": "degree_u",
                            "value": value, "rule": "infer_from_knot_count",
                            "path": f"patches[{i}].degree_u",
                            "location": f"{loc}.degree_u"})
        if "degree_v" not in pdoc:
            value = _infer_degree(pdoc["knots_v"], len(pdoc["control_points"][0]))
            pdoc["degree_v"] = value
            actions.append({"patch": pdoc.get("id"), "field": "degree_v",
                            "value": value, "rule": "infer_from_knot_count",
                            "path": f"patches[{i}].degree_v",
                            "location": f"{loc}.degree_v"})
        if "material_channel" not in pdoc:
            pdoc["material_channel"] = "default"
            actions.append({"patch": pdoc.get("id"), "field": "material_channel",
                            "value": "default", "rule": "default_constant",
                            "path": f"patches[{i}].material_channel",
                            "location": f"{loc}.material_channel"})
        if "tangent_space" not in pdoc:
            pdoc["tangent_space"] = False
            actions.append({"patch": pdoc.get("id"), "field": "tangent_space",
                            "value": False, "rule": "default_constant",
                            "path": f"patches[{i}].tangent_space",
                            "location": f"{loc}.tangent_space"})
    doc["schema_version"] = 2


_MIGRATION_STEPS = {1: _migrate_1_to_2}


def migrate_document(doc: dict):
    """Migrate a raw JSON document to the current schema version.

    Returns (migrated_doc, actions).  Raises E_SCHEMA_VERSION when the
    document is newer than this application or has no usable version.
    """
    version = doc.get("schema_version")
    if not isinstance(version, int):
        raise CadError(ErrorCode.E_SCHEMA_VERSION,
                       "missing or invalid schema_version", "schema_version")
    if version > SCHEMA_VERSION:
        raise CadError(ErrorCode.E_SCHEMA_VERSION,
                       f"schema version {version} is newer than supported "
                       f"version {SCHEMA_VERSION}", "schema_version")
    result = copy.deepcopy(doc)
    actions: list = []
    while result["schema_version"] < SCHEMA_VERSION:
        step = _MIGRATION_STEPS.get(result["schema_version"])
        if step is None:
            raise CadError(ErrorCode.E_SCHEMA_VERSION,
                           f"no migration path from schema version "
                           f"{result['schema_version']}", "schema_version")
        step(result, actions)
    return result, actions
