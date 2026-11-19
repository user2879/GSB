"""Format interoperability: JSON round trip, binary profiles, migration."""
import json

import numpy as np
import pytest

from cadapp import GEOMETRY_TOLERANCE
from cadapp.binio import convert_mesh, read_mesh, write_mesh
from cadapp.geometry import tessellate_patch
from cadapp.jsonio import dumps_model, loads_model
from cadapp.migrate import migrate_document
from cadapp.model import canonical_bytes, content_hash, model_from_dict
from cadapp.samples import legacy_v1_json_text, make_sample_model


def _whole_mesh(model):
    geos = [tessellate_patch(p) for p in model.patches]
    vertices = np.concatenate([g.vertices for g in geos])
    normals = np.concatenate([g.normals for g in geos])
    faces = []
    offset = 0
    for g in geos:
        faces.append(g.faces + offset)
        offset += g.vertices.shape[0]
    return vertices, normals, np.concatenate(faces)


def test_json_roundtrip_control_points_exact():
    model = make_sample_model()
    restored, actions = loads_model(dumps_model(model))
    assert actions == []
    assert content_hash(restored) == content_hash(model)
    for pa, pb in zip(model.patches, restored.patches):
        assert pa.control_points == pb.control_points  # float64 exact


@pytest.mark.parametrize("endian,index_width,interleaved", [
    ("little", 32, False),
    ("big", 32, False),
    ("little", 16, False),
    ("big", 16, False),
    ("little", 32, True),
    ("big", 32, True),
])
def test_binary_profiles_bit_exact(endian, index_width, interleaved):
    model = make_sample_model()
    vertices, normals, faces = _whole_mesh(model)
    blob = write_mesh(vertices, normals, faces, version=2, endian=endian,
                      index_width=index_width, interleaved=interleaved,
                      source_model_bytes=canonical_bytes(model))
    mesh, summary = read_mesh(blob)
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.normals, normals)
    assert np.array_equal(mesh.faces, faces)
    assert summary["endianness"] == endian
    assert summary["index_width"] == index_width
    assert summary["interleaved"] == interleaved
    assert summary["has_source_model"]


def test_binary_v1_profile_and_conversion():
    model = make_sample_model()
    vertices, normals, faces = _whole_mesh(model)
    blob_v2 = write_mesh(vertices, normals, faces, version=2, endian="big",
                         index_width=32, interleaved=True)
    blob_v1 = convert_mesh(blob_v2, version=1)
    mesh, summary = read_mesh(blob_v1)
    assert summary["format_version"] == 1
    assert summary["endianness"] == "little"
    assert summary["index_width"] == 16
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.normals, normals)
    back = convert_mesh(blob_v1, version=2, endian="little", index_width=32)
    mesh2, _ = read_mesh(back)
    assert np.array_equal(mesh2.vertices, vertices)


def test_binary_to_json_recovers_model_exactly():
    model = make_sample_model()
    vertices, normals, faces = _whole_mesh(model)
    blob = write_mesh(vertices, normals, faces,
                      source_model_bytes=canonical_bytes(model))
    mesh, _ = read_mesh(blob)
    recovered = model_from_dict(json.loads(mesh.source_model_bytes.decode()))
    assert content_hash(recovered) == content_hash(model)
    for pa, pb in zip(model.patches, recovered.patches):
        assert pa.control_points == pb.control_points
    # Triangulation of the recovered model matches the exchanged mesh.
    v2, n2, f2 = _whole_mesh(recovered)
    assert np.abs(v2 - vertices).max() <= GEOMETRY_TOLERANCE
    assert np.array_equal(f2, faces)


def test_migration_idempotent_and_geometry_preserving():
    legacy_doc = json.loads(legacy_v1_json_text())
    once, actions1 = migrate_document(legacy_doc)
    twice, actions2 = migrate_document(once)
    assert actions1  # v1 -> v2 applied defaults
    assert actions2 == []  # idempotent: no further actions
    assert json.dumps(once, sort_keys=True) == json.dumps(twice, sort_keys=True)
    migrated = model_from_dict(once)
    reference = make_sample_model()
    assert content_hash(migrated) == content_hash(reference)
    for pa, pb in zip(migrated.patches, reference.patches):
        da = tessellate_patch(pa).vertices
        db = tessellate_patch(pb).vertices
        assert np.abs(da - db).max() <= GEOMETRY_TOLERANCE


def test_migration_defaults_do_not_move_isocurves():
    legacy_doc = json.loads(legacy_v1_json_text())
    once, _ = migrate_document(legacy_doc)
    migrated = model_from_dict(once)
    reference = make_sample_model()
    for pa, pb in zip(migrated.patches, reference.patches):
        ga = tessellate_patch(pa)
        gb = tessellate_patch(pb)
        for ca, cb in zip(ga.isocurves, gb.isocurves):
            assert np.abs(ca - cb).max() <= GEOMETRY_TOLERANCE
