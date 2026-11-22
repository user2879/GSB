from __future__ import annotations

from pathlib import Path

import pytest

from snapdiag.errors import DiagnosticError, E_GEOM
from snapdiag.geometry import (
    decode_hex_block,
    generate_volume,
    validate_geometry,
)


def test_unsupported_block_size_is_first_geometry_violation(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticError) as caught:
        validate_geometry(tmp_path / "missing.bin", 777, -1)
    assert caught.value.code == E_GEOM
    assert caught.value.field == "block_size"
    assert caught.value.to_dict()["first_violation"]["field"] == "block_size"


def test_image_length_must_be_exact(tmp_path: Path) -> None:
    image = tmp_path / "volume.bin"
    generate_volume(image, 512, 4)
    with open(image, "ab") as handle:
        handle.write(b"x")
    with pytest.raises(DiagnosticError) as caught:
        validate_geometry(image, 512, 4)
    assert caught.value.code == E_GEOM
    assert caught.value.field == "image_length"
    assert caught.value.details["expected"] == 2048
    assert caught.value.details["actual"] == 2049


def test_block_number_is_checked_before_payload(tmp_path: Path) -> None:
    image = tmp_path / "volume.bin"
    geometry = generate_volume(image, 512, 4)
    with pytest.raises(DiagnosticError) as caught:
        geometry.read_block(4)
    assert caught.value.code == E_GEOM
    assert caught.value.field == "block_index"
    with pytest.raises(DiagnosticError) as caught:
        geometry.write_block(-1, b"wrong")
    assert caught.value.field == "block_index"


def test_hex_payload_must_exactly_fill_one_block() -> None:
    payload = decode_hex_block("0x" + "ab" * 512, 512)
    assert len(payload) == 512
    assert payload[:2] == b"\xab\xab"
    with pytest.raises(DiagnosticError) as caught:
        decode_hex_block("0x1234", 512)
    assert caught.value.code == E_GEOM
    assert caught.value.field == "hex_data"
    with pytest.raises(DiagnosticError) as caught:
        decode_hex_block("zz", 512)
    assert caught.value.field == "hex_data"


def test_deterministic_volume_generation(tmp_path: Path) -> None:
    first = generate_volume(tmp_path / "a.bin", 512, 8, seed=123)
    second = generate_volume(tmp_path / "b.bin", 512, 8, seed=123)
    assert first.image_path.read_bytes() == second.image_path.read_bytes()

