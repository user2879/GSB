"""Version comparison with difference attribution.

diff_models produces a deterministic, per-patch list of differences
between two versions of a model.  attribute_diffs then explains each
difference using the application's operation log:

  source "edit"               control-point change matching an edit op
  source "migration"          field added/changed by a migration action
  source "format_conversion"  declared effect of a conversion op
  source "suspicious"         no operation explains the difference

Unattributable differences are always reported, never dropped.
"""

from __future__ import annotations

_MIGRATED_FIELDS = {"degree_u", "degree_v", "material_channel", "tangent_space"}


def diff_models(model_a, model_b) -> list[dict]:
    diffs: list[dict] = []
    if model_a.schema_version != model_b.schema_version:
        diffs.append({"kind": "schema_version",
                      "old": model_a.schema_version, "new": model_b.schema_version})
    ids_a = [p.patch_id for p in model_a.patches]
    ids_b = [p.patch_id for p in model_b.patches]
    for pid in ids_a:
        if pid not in ids_b:
            diffs.append({"kind": "patch_removed", "patch": pid})
    for pid in ids_b:
        if pid not in ids_a:
            diffs.append({"kind": "patch_added", "patch": pid})
    for pa in model_a.patches:
        if pa.patch_id not in ids_b:
            continue
        pb = model_b.patch_by_id(pa.patch_id)
        for field_name in ("degree_u", "degree_v", "material_channel", "tangent_space"):
            va = getattr(pa, field_name)
            vb = getattr(pb, field_name)
            if va != vb:
                diffs.append({"kind": "field", "patch": pa.patch_id,
                              "field": field_name, "old": va, "new": vb})
        if pa.knots_u != pb.knots_u or pa.knots_v != pb.knots_v:
            diffs.append({"kind": "knots", "patch": pa.patch_id})
        if pa.weights != pb.weights:
            diffs.append({"kind": "weights", "patch": pa.patch_id})
        rows_a = len(pa.control_points)
        cols_a = len(pa.control_points[0])
        rows_b = len(pb.control_points)
        cols_b = len(pb.control_points[0])
        if rows_a != rows_b or cols_a != cols_b:
            diffs.append({"kind": "control_grid_shape", "patch": pa.patch_id,
                          "old": [rows_a, cols_a], "new": [rows_b, cols_b]})
            continue
        for i in range(rows_a):
            for j in range(cols_a):
                if pa.control_points[i][j] != pb.control_points[i][j]:
                    diffs.append({"kind": "control_point", "patch": pa.patch_id,
                                  "index": [i, j],
                                  "old": list(pa.control_points[i][j]),
                                  "new": list(pb.control_points[i][j])})
    return diffs


def attribute_diffs(diffs: list[dict], op_log: list[dict]) -> list[dict]:
    edits = [op for op in op_log if op.get("op") == "edit"]
    migrations = [a for op in op_log if op.get("op") == "migrate"
                  for a in op.get("actions", [])]
    conversions = [op for op in op_log if op.get("op") == "format_conversion"]
    report: list[dict] = []
    for diff in diffs:
        entry = dict(diff)
        source = None
        if diff["kind"] == "control_point":
            for op in edits:
                if (op["patch"] == diff["patch"] and op["index"] == diff["index"]
                        and op.get("new_value") == diff["new"]):
                    source = "edit"
                    break
        elif diff["kind"] == "field" and diff["field"] in _MIGRATED_FIELDS:
            for action in migrations:
                if (action.get("patch") == diff["patch"]
                        and action.get("field") == diff["field"]
                        and action.get("value") == diff["new"]):
                    source = "migration"
                    break
        elif diff["kind"] == "schema_version" and migrations:
            source = "migration"
        if source is None and conversions:
            declared = {e for op in conversions for e in op.get("effects", [])}
            if diff["kind"] in declared:
                source = "format_conversion"
        entry["source"] = source if source is not None else "suspicious"
        report.append(entry)
    return report


def compare_models(model_a, model_b, op_log: list[dict]) -> dict:
    diffs = diff_models(model_a, model_b)
    attributed = attribute_diffs(diffs, op_log)
    return {
        "model_a": model_a.model_id,
        "model_b": model_b.model_id,
        "diff_count": len(attributed),
        "suspicious_count": sum(1 for d in attributed if d["source"] == "suspicious"),
        "diffs": attributed,
    }


def diff_documents(doc_a, doc_b, path: str = "") -> list[dict]:
    """Recursive diff of two raw JSON documents (e.g. pre/post migration)."""
    diffs: list[dict] = []
    if isinstance(doc_a, dict) and isinstance(doc_b, dict):
        for key in sorted(set(doc_a) | set(doc_b)):
            sub = f"{path}.{key}" if path else key
            if key not in doc_a:
                diffs.append({"kind": "document", "path": sub,
                              "old": None, "new": doc_b[key]})
            elif key not in doc_b:
                diffs.append({"kind": "document", "path": sub,
                              "old": doc_a[key], "new": None})
            else:
                diffs.extend(diff_documents(doc_a[key], doc_b[key], sub))
    elif (isinstance(doc_a, list) and isinstance(doc_b, list)
          and len(doc_a) == len(doc_b)):
        for i, (xa, xb) in enumerate(zip(doc_a, doc_b)):
            diffs.extend(diff_documents(xa, xb, f"{path}[{i}]"))
    elif doc_a != doc_b:
        diffs.append({"kind": "document", "path": path,
                      "old": doc_a, "new": doc_b})
    return diffs


def attribute_document_diffs(diffs: list[dict], op_log: list[dict]) -> list[dict]:
    """Attribute document-level diffs; migration actions carry a 'path'."""
    migrations = [a for op in op_log if op.get("op") == "migrate"
                  for a in op.get("actions", [])]
    report: list[dict] = []
    for diff in diffs:
        entry = dict(diff)
        source = None
        if diff["kind"] == "document":
            if diff["path"] == "schema_version" and migrations:
                source = "migration"
            else:
                for action in migrations:
                    if (action.get("path") == diff["path"]
                            and action.get("value") == diff["new"]):
                        source = "migration"
                        break
        entry["source"] = source if source is not None else "suspicious"
        report.append(entry)
    return report


def compare_documents(doc_a, doc_b, op_log: list[dict]) -> dict:
    diffs = diff_documents(doc_a, doc_b)
    attributed = attribute_document_diffs(diffs, op_log)
    return {
        "diff_count": len(attributed),
        "suspicious_count": sum(1 for d in attributed if d["source"] == "suspicious"),
        "diffs": attributed,
    }
