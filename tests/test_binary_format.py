from __future__ import annotations

import struct

import numpy as np
import pytest

from npr_editor.binary_format import (
    CURRENT_FORMAT_VERSION,
    OLD_FORMAT_VERSION,
    decode_binary,
    encode_binary,
    parse_binary,
)
from npr_editor.errors import NprError
from npr_editor.geometry import flatten_shape
from npr_editor.model import bytes_hash


@pytest.mark.parametrize("byte_order", ["little", "big"])
@pytest.mark.parametrize("index_width", [2, 4])
@pytest.mark.parametrize("format_version", [OLD_FORMAT_VERSION, CURRENT_FORMAT_VERSION])
def test_binary_roundtrip_all_endian_index_version_combinations(
    sample_project, byte_order, index_width, format_version
):
    encoded = encode_binary(
        sample_project,
        byte_order=byte_order,
        index_width=index_width,
        format_version=format_version,
    )
    parsed = parse_binary(encoded.data)
    decoded = decode_binary(encoded.data)
    assert parsed.byte_order == byte_order
    assert parsed.index_width == index_width
    assert parsed.format_version == format_version
    assert parsed.width_encoding == (0 if format_version == 1 else 1)
    assert parsed.content_hash == bytes_hash(encoded.data) == encoded.content_hash
    assert decoded.project_hash == sample_project.content_hash()
    assert decoded.project.content_hash() == sample_project.content_hash()
    for shape in sample_project.ordered_shapes():
        expected = flatten_shape(shape, sample_project.stroke_sample).vertices
        parsed_shape = next(item for item in parsed.shapes if item.shape_id == shape.id)
        actual = parsed.vertices[
            parsed_shape.vertex_offset : parsed_shape.vertex_offset + parsed_shape.vertex_count
        ]
        assert np.array_equal(actual, expected)


def test_old_width_encoding_is_auditable(sample_project):
    encoded = encode_binary(sample_project, format_version=1, byte_order="big", index_width=2)
    parsed = parse_binary(encoded.data)
    assert parsed.format_version == 1
    assert parsed.width_encoding == 0
    assert parsed.byte_order == "big"
    assert parsed.index_width == 2


def test_skeleton_without_metadata_reconstructs_same_flattened_geometry(sample_project):
    encoded = encode_binary(
        sample_project,
        byte_order="little",
        index_width=2,
        include_project_metadata=False,
    )
    decoded = decode_binary(encoded.data)
    assert decoded.parsed.metadata is None
    for original in sample_project.ordered_shapes():
        imported = decoded.project.shape_by_id(original.id)
        expected = flatten_shape(original, sample_project.stroke_sample).vertices
        actual = flatten_shape(imported, decoded.project.stroke_sample).vertices
        assert np.array_equal(actual, expected)


def test_unknown_endian_marker_is_reported(sample_project):
    data = bytearray(encode_binary(sample_project).data)
    data[4:8] = b"\xde\xad\xbe\xef"
    with pytest.raises(NprError) as caught:
        decode_binary(bytes(data))
    assert caught.value.code == "E_FORMAT_ENDIAN"


def test_out_of_range_index_is_rejected(sample_project):
    data = bytearray(encode_binary(sample_project, byte_order="little", index_width=4).data)
    struct.pack_into("<I", data, len(data) - 4, 0xFFFF)
    with pytest.raises(NprError) as caught:
        decode_binary(bytes(data))
    assert caught.value.code == "E_FORMAT_INDEX"


def test_index_count_mismatch_is_rejected(sample_project):
    data = bytearray(encode_binary(sample_project, byte_order="little", index_width=4).data)
    declared = struct.unpack_from("<I", data, 18)[0]
    struct.pack_into("<I", data, 18, declared + 1)
    with pytest.raises(NprError) as caught:
        decode_binary(bytes(data))
    assert caught.value.code == "E_FORMAT_INDEX"


def test_unknown_binary_version_is_rejected(sample_project):
    data = bytearray(encode_binary(sample_project).data)
    struct.pack_into("<H", data, 8, 99)
    with pytest.raises(NprError) as caught:
        decode_binary(bytes(data))
    assert caught.value.code == "E_FORMAT_VERSION"


def test_failed_parse_does_not_mutate_existing_project(sample_project):
    before = sample_project.content_hash()
    data = bytearray(encode_binary(sample_project).data)
    data[4:8] = b"\x00\x00\x00\x00"
    with pytest.raises(NprError):
        decode_binary(bytes(data))
    assert sample_project.content_hash() == before
