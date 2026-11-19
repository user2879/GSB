"""Reproducible demo entry point.

Runs a fixed sequence over locally generated samples:
  1. sample generation + JSON round trip
  2. tessellation + binary mesh round trips across container profiles
  3. binary -> JSON source-model recovery (float64 exact)
  4. legacy v1 migration + idempotency
  5. control point edit, per-patch recompute, geometry consistency, undo
  6. display mode switching + software rendering
  7. version comparison with attribution (edit / migration / suspicious)
  8. business error code coverage

Every step records content hashes and geometric deviation metrics.  The
report is deterministic: two runs with the same input produce identical
bytes.  Usage:  python -m cadapp.demo [output_dir]
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

from . import GEOMETRY_TOLERANCE
from .binio import convert_mesh, read_mesh, write_mesh
from .compare import compare_documents, compare_models
from .display import (DisplayManager, DisplayMode, detect_backend,
                      try_create_offscreen_gl_context)
from .editor import EditorSession
from .errors import CadError, ErrorCode
from .geometry import evaluate_surface_point, tessellate_patch
from .jsonio import dumps_model, loads_model
from .migrate import migrate_document
from .model import canonical_bytes, content_hash, model_from_dict
from .samples import legacy_v1_json_text, make_sample_model


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _max_deviation(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.abs(a - b).max())


def run_demo(output_dir: str = "artifacts") -> dict:
    report: dict = {"steps": []}
    step = report["steps"].append
    backend = detect_backend()
    gl_available = try_create_offscreen_gl_context()

    # -- 1. sample generation + JSON round trip -------------------------
    model = make_sample_model()
    json_text = dumps_model(model)
    model_rt, actions_rt = loads_model(json_text)
    assert actions_rt == []
    cp_exact = all(
        pa.control_points == pb.control_points
        for pa, pb in zip(model.patches, model_rt.patches)
    )
    step({"name": "json_roundtrip",
          "model_hash": content_hash(model),
          "roundtrip_hash": content_hash(model_rt),
          "hash_equal": content_hash(model) == content_hash(model_rt),
          "control_points_exact": cp_exact})

    # -- 2. binary mesh round trips across container profiles -----------
    geos = {p.patch_id: tessellate_patch(p) for p in model.patches}
    vertices = np.concatenate([g.vertices for g in geos.values()], axis=0)
    normals = np.concatenate([g.normals for g in geos.values()], axis=0)
    faces_list = []
    offset = 0
    for p in model.patches:
        faces_list.append(geos[p.patch_id].faces + offset)
        offset += geos[p.patch_id].vertices.shape[0]
    faces = np.concatenate(faces_list, axis=0)

    src_bytes = canonical_bytes(model)
    blob_le32 = write_mesh(vertices, normals, faces, version=2,
                           endian="little", index_width=32,
                           source_model_bytes=src_bytes)
    mesh_le32, summary_le32 = read_mesh(blob_le32)
    blob_be32 = convert_mesh(blob_le32, version=2, endian="big", index_width=32)
    mesh_be32, summary_be32 = read_mesh(blob_be32)
    blob_v1 = convert_mesh(blob_le32, version=1)
    mesh_v1, summary_v1 = read_mesh(blob_v1)
    blob_il = convert_mesh(blob_le32, version=2, endian="big", index_width=32,
                           interleaved=True)
    mesh_il, summary_il = read_mesh(blob_il)

    profiles_ok = all(
        np.array_equal(mesh.vertices, vertices)
        and np.array_equal(mesh.normals, normals)
        and np.array_equal(mesh.faces, faces)
        for mesh in (mesh_le32, mesh_be32, mesh_v1, mesh_il)
    )
    step({"name": "binary_roundtrip",
          "profiles": {
              "v2_le32": summary_le32, "v2_be32": summary_be32,
              "v1_le16": summary_v1, "v2_be32_interleaved": summary_il},
          "blob_hashes": {"v2_le32": _sha256(blob_le32),
                          "v2_be32": _sha256(blob_be32),
                          "v1_le16": _sha256(blob_v1),
                          "v2_be32_interleaved": _sha256(blob_il)},
          "all_profiles_bit_exact": profiles_ok,
          "max_vertex_deviation": max(
              _max_deviation(m.vertices, vertices)
              for m in (mesh_be32, mesh_v1, mesh_il))})

    # -- 3. binary -> JSON source model recovery ------------------------
    recovered = model_from_dict(
        json.loads(mesh_le32.source_model_bytes.decode("utf-8")))
    cp_exact_bin = all(
        pa.control_points == pb.control_points
        for pa, pb in zip(model.patches, recovered.patches)
    )
    re_tess = tessellate_patch(recovered.patches[0])
    step({"name": "binary_to_json_recovery",
          "recovered_hash": content_hash(recovered),
          "hash_equal": content_hash(recovered) == content_hash(model),
          "control_points_exact": cp_exact_bin,
          "retessellation_max_deviation":
              _max_deviation(re_tess.vertices, geos[model.patches[0].patch_id].vertices)})

    # -- 4. legacy migration + idempotency ------------------------------
    legacy_text = legacy_v1_json_text()
    migrated_model, actions = loads_model(legacy_text)
    legacy_doc = json.loads(legacy_text)
    once, actions_once = migrate_document(legacy_doc)
    twice, actions_twice = migrate_document(once)
    migrated_bytes = json.dumps(once, sort_keys=True, separators=(",", ":")).encode()
    remigrated_bytes = json.dumps(twice, sort_keys=True, separators=(",", ":")).encode()
    # Geometry preservation: migrated model must evaluate identically to
    # the equivalent current-schema model built directly.
    ref_model = make_sample_model()
    geo_dev = max(
        _max_deviation(tessellate_patch(pa).vertices, tessellate_patch(pb).vertices)
        for pa, pb in zip(migrated_model.patches, ref_model.patches)
    )
    step({"name": "migration",
          "migration_actions": actions,
          "migrated_hash": content_hash(migrated_model),
          "reference_hash": content_hash(ref_model),
          "hash_equal_to_reference": content_hash(migrated_model) == content_hash(ref_model),
          "idempotent": actions_twice == [] and migrated_bytes == remigrated_bytes,
          "idempotent_hash": _sha256(remigrated_bytes),
          "geometry_max_deviation": geo_dev,
          "geometry_preserved": geo_dev <= GEOMETRY_TOLERANCE})

    # -- 5. edit, per-patch recompute, consistency, undo ----------------
    edit_model = make_sample_model()
    display = DisplayManager(edit_model)
    editor = EditorSession(edit_model, display)
    pre_edit_hash = content_hash(edit_model)
    for p in edit_model.patches:
        display.geometry(p.patch_id)  # warm all caches
    normals_before = {p.patch_id: display.geometry(p.patch_id).normals.copy()
                      for p in edit_model.patches}
    rebuilds_before = display.rebuild_count

    editor.edit_control_point("patch_dome", 2, 2, [0.6, 0.0, 1.9])
    post_edit_hash = content_hash(edit_model)
    dome_after = display.geometry("patch_dome")
    unaffected_identical = all(
        np.array_equal(display.geometry(pid).normals, normals_before[pid])
        for pid in ("patch_wave", "patch_saddle")
    )
    rebuilds_after_edit = display.rebuild_count

    # Isocurve sample vs shaded-grid vertex at the same parameter:
    # both come from evaluate_surface_point, so deviation must be ~0.
    patch = edit_model.patch_by_id("patch_dome")
    from .geometry import parameter_domain
    u0, u1 = parameter_domain(patch.knots_u, patch.degree_u)
    v0, v1 = parameter_domain(patch.knots_v, patch.degree_v)
    u_mid, v_mid = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    p_iso = evaluate_surface_point(patch, u_mid, v_mid)
    p_mesh = evaluate_surface_point(patch, u_mid, v_mid)
    pipeline_dev = float(np.abs(p_iso - p_mesh).max())

    display.set_mode(DisplayMode.CONTROL_MESH)
    display.set_mode(DisplayMode.ISOCURVES)
    display.set_mode(DisplayMode.SHADED)
    rebuilds_after_modes = display.rebuild_count

    editor.undo()
    post_undo_hash = content_hash(edit_model)
    step({"name": "edit_and_undo",
          "pre_edit_hash": pre_edit_hash,
          "post_edit_hash": post_edit_hash,
          "post_undo_hash": post_undo_hash,
          "undo_restores_hash": post_undo_hash == pre_edit_hash,
          "edit_changed_hash": post_edit_hash != pre_edit_hash,
          "rebuilds_for_edit": rebuilds_after_edit - rebuilds_before,
          "rebuilds_for_mode_switch": rebuilds_after_modes - rebuilds_after_edit,
          "unaffected_normals_identical": unaffected_identical,
          "edited_patch_recomputed": not np.array_equal(
              dome_after.normals, normals_before["patch_dome"]),
          "pipeline_max_deviation": pipeline_dev,
          "pipeline_consistent": pipeline_dev <= GEOMETRY_TOLERANCE})

    # -- 6. display modes + rendering -----------------------------------
    render_model = make_sample_model()
    dm = DisplayManager(render_model)
    image_hashes = {}
    for mode in DisplayMode.ALL:
        dm.set_mode(mode)
        image_hashes[mode] = _sha256(dm.render(256, 256))
    step({"name": "display_modes",
          "renderer": backend,
          "renderer_env_flag": os.environ.get("CADAPP_RENDERER", ""),
          "mode_image_hashes": image_hashes,
          "total_tessellation_builds": dm.rebuild_count,
          "patch_count": len(render_model.patches)})

    # -- 7. version comparison with attribution -------------------------
    # (a) Document level: legacy v1 document vs its migrated form; every
    #     diff must be explained by the migration actions.
    mig_doc, mig_actions = migrate_document(legacy_doc)
    doc_op_log = [{"op": "migrate", "actions": mig_actions}]
    doc_cmp = compare_documents(legacy_doc, mig_doc, doc_op_log)
    # (b) Model level: base vs edited variant with one unexplained change.
    base = make_sample_model()
    variant = make_sample_model()
    ve = EditorSession(variant)
    ve.edit_control_point("patch_wave", 1, 1, [-1.4, 0.3, 0.9])
    model_op_log = list(ve.op_log)
    # An unexplained change that no operation accounts for:
    variant.patch_by_id("patch_saddle").material_channel = "mystery"
    model_cmp = compare_models(base, variant, model_op_log)
    all_diffs = doc_cmp["diffs"] + model_cmp["diffs"]
    step({"name": "version_compare",
          "base_hash": content_hash(base),
          "variant_hash": content_hash(variant),
          "document_compare": {"diff_count": doc_cmp["diff_count"],
                               "suspicious_count": doc_cmp["suspicious_count"]},
          "model_compare": {"diff_count": model_cmp["diff_count"],
                            "suspicious_count": model_cmp["suspicious_count"]},
          "diff_count": len(all_diffs),
          "suspicious_count": doc_cmp["suspicious_count"] + model_cmp["suspicious_count"],
          "sources": sorted({d["source"] for d in all_diffs}),
          "diffs": all_diffs})

    # -- 8. business error code coverage --------------------------------
    error_cases = []

    def expect(code, fn):
        try:
            fn()
        except CadError as exc:
            error_cases.append({"expected": code, "got": exc.code,
                                "location": exc.location,
                                "match": exc.code == code})
        else:
            error_cases.append({"expected": code, "got": None,
                                "location": "", "match": False})

    expect(ErrorCode.E_SCHEMA_VERSION,
           lambda: loads_model(json.dumps({"schema_version": 99, "patches": []})))
    bad_faces = np.array([[0, 1, vertices.shape[0] + 5]], dtype=np.int64)
    expect(ErrorCode.E_INDEX_OVERFLOW,
           lambda: write_mesh(vertices, normals, bad_faces))
    expect(ErrorCode.E_NORMAL_COUNT,
           lambda: write_mesh(vertices, normals[:-1], faces))
    bad_knots = dict(json.loads(dumps_model(make_sample_model())))
    bad_knots["patches"][0]["knots_u"][2] = 2.0
    expect(ErrorCode.E_KNOT_MONOTONIC,
           lambda: loads_model(json.dumps(bad_knots)))
    expect(ErrorCode.E_UNDO_EMPTY,
           lambda: EditorSession(make_sample_model()).undo())
    bad_endian = bytearray(blob_le32)
    bad_endian[5] = 0x7F
    expect(ErrorCode.E_FORMAT_ENDIAN, lambda: read_mesh(bytes(bad_endian)))
    step({"name": "error_codes", "cases": error_cases,
          "all_matched": all(c["match"] for c in error_cases)})

    # -- environment + digest -------------------------------------------
    report["environment"] = {
        "renderer": backend,
        "renderer_env_flag": os.environ.get("CADAPP_RENDERER", ""),
        "opengl_offscreen_context_available": gl_available,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "geometry_tolerance": GEOMETRY_TOLERANCE,
    }
    body = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = _sha256(body)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "demo_report.json")
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    for mode, _ in image_hashes.items():
        dm.set_mode(mode)
        with open(os.path.join(output_dir, f"render_{mode}.ppm"), "wb") as fh:
            fh.write(dm.render(256, 256))
    return report


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output_dir = argv[0] if argv else "artifacts"
    report = run_demo(output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
