from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from mdet_render.cli import main
from mdet_render.errors import BusinessError
from mdet_render.png import PNG_SIGNATURE
from mdet_render.workspace import Workspace

from conftest import edit_config, load_workspace


def _chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(body, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def _write_rgba_png(path: Path, width: int, height: int) -> None:
    raw = b"".join(b"\x00" + bytes([10, 20, 30, 255]) * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.write_bytes(PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw))
                     + _chunk(b"IEND", b""))


def test_valid_sample_loads(sample_root: Path) -> None:
    workspace = load_workspace(sample_root)
    assert workspace.config.condition_dim == 40
    assert workspace.vocabulary.dimension == 40
    assert workspace.motion.values.shape == (4, 5, 2)


def test_unknown_prompt_term_is_rejected(sample_root: Path) -> None:
    prompt_path = sample_root / "prompts" / "frame0.json"
    raw = json.loads(prompt_path.read_text())
    raw["entities"] = ["not_in_vocabulary"]
    prompt_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BusinessError) as caught:
        load_workspace(sample_root)
    assert caught.value.code == "E_PROMPT_TERM"


def test_bad_png_signature_is_rejected(sample_root: Path) -> None:
    (sample_root / "images" / "frame0.png").write_bytes(b"not-a-png")
    with pytest.raises(BusinessError) as caught:
        load_workspace(sample_root)
    assert caught.value.code == "E_PNG_HEADER"


def test_bad_png_pixel_mode_is_rejected(sample_root: Path) -> None:
    _write_rgba_png(sample_root / "images" / "frame0.png", 32, 24)
    with pytest.raises(BusinessError) as caught:
        load_workspace(sample_root)
    assert caught.value.code == "E_PNG_PIXEL_MODE"


def test_bad_png_resolution_is_rejected(sample_root: Path) -> None:
    def change(config: dict) -> None:
        config["resolution"] = {"width": 64, "height": 48}
        config["execution"]["point_budget"] = 64 * 48 * 3 * 3

    edit_config(sample_root, change)
    with pytest.raises(BusinessError) as caught:
        load_workspace(sample_root)
    assert caught.value.code == "E_PNG_RESOLUTION"


def test_bad_motion_rows_are_rejected(sample_root: Path) -> None:
    path = sample_root / "motion.json"
    raw = json.loads(path.read_text())
    raw["block_rows"] = 3
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BusinessError) as caught:
        load_workspace(sample_root)
    assert caught.value.code == "E_MOTION_ROWS"


def test_quantization_enums_are_enforced(sample_root: Path) -> None:
    def change(config: dict) -> None:
        config["fixed_point"]["fraction_bits"] = 7

    edit_config(sample_root, change)
    with pytest.raises(BusinessError) as caught:
        load_workspace(sample_root)
    assert caught.value.code == "E_ENUM_FRACTION_BITS"


def test_cli_validation_returns_nonzero_on_business_error(sample_root: Path, capsys) -> None:
    def change(config: dict) -> None:
        config["fixed_point"]["quant_level"] = 9

    edit_config(sample_root, change)
    code = main(["validate", "--config", str(sample_root / "config.json")])
    output = json.loads(capsys.readouterr().out)
    assert code != 0
    assert output["ok"] is False
    assert output["code"] == "E_ENUM_QUANT_LEVEL"
