"""Business error code coverage."""
import json

import numpy as np
import pytest

from cadapp.binio import read_mesh, write_mesh
from cadapp.editor import EditorSession
from cadapp.errors import CadError, ErrorCode
from cadapp.jsonio import dumps_model, loads_model
from cadapp.samples import make_sample_model


def _expect(code, fn):
    with pytest.raises(CadError) as excinfo:
        fn()
    assert excinfo.value.code == code
    assert excinfo.value.location  # every error carries a location


def test_schema_version_too_new():
    _expect(ErrorCode.E_SCHEMA_VERSION,
            lambda: loads_model(json.dumps({"schema_version": 99, "patches": []})))


def test_schema_version_missing():
    _expect(ErrorCode.E_SCHEMA_VERSION,
            lambda: loads_model(json.dumps({"patches": []})))


def test_index_overflow_on_write():
    vertices = np.zeros((4, 3))
    normals = np.zeros((4, 3))
    faces = np.array([[0, 1, 7]], dtype=np.int64)
    _expect(ErrorCode.E_INDEX_OVERFLOW,
            lambda: write_mesh(vertices, normals, faces))


def test_index_overflow_on_read():
    vertices = np.zeros((4, 3))
    normals = np.zeros((4, 3))
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    blob = bytearray(write_mesh(vertices, normals, faces))
    # Corrupt the first face index to 500 (little-endian uint32 payload).
    face_offset = 16 + 4 * 3 * 8 * 2
    blob[face_offset:face_offset + 4] = (500).to_bytes(4, "little")
    _expect(ErrorCode.E_INDEX_OVERFLOW, lambda: read_mesh(bytes(blob)))


def test_normal_count_mismatch_on_write():
    vertices = np.zeros((4, 3))
    normals = np.zeros((3, 3))
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    _expect(ErrorCode.E_NORMAL_COUNT,
            lambda: write_mesh(vertices, normals, faces))


def test_normal_count_mismatch_on_read():
    vertices = np.zeros((4, 3))
    normals = np.zeros((4, 3))
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    blob = write_mesh(vertices, normals, faces)
    truncated = blob[:16 + 4 * 3 * 8 + 5 * 8]  # cut inside the normal block
    _expect(ErrorCode.E_NORMAL_COUNT, lambda: read_mesh(truncated))


def test_knot_vector_non_monotonic():
    doc = json.loads(dumps_model(make_sample_model()))
    doc["patches"][0]["knots_u"][2] = 2.0
    _expect(ErrorCode.E_KNOT_MONOTONIC, lambda: loads_model(json.dumps(doc)))


def test_undo_empty_stack():
    session = EditorSession(make_sample_model())
    _expect(ErrorCode.E_UNDO_EMPTY, session.undo)


def test_endian_marker_unrecognized():
    vertices = np.zeros((4, 3))
    normals = np.zeros((4, 3))
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    blob = bytearray(write_mesh(vertices, normals, faces))
    blob[5] = 0x7F
    _expect(ErrorCode.E_FORMAT_ENDIAN, lambda: read_mesh(bytes(blob)))
