from __future__ import annotations

import copy
import json

import pytest

from mmfm.cli import main
from mmfm.config import load_render_config, load_vocabulary, validate_config
from mmfm.conditions import _validate_motion
from mmfm.errors import BusinessError
from mmfm.exchange import StageRecord, serialize_v1
from mmfm.png import decode_png_rgb


def expect_code(code: str, function, *args, **kwargs) -> BusinessError:
    with pytest.raises(BusinessError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code
    return caught.value


def test_unknown_prompt_term_and_dimension(workspace, repo_root):
    from mmfm.config import validate_prompt

    expect_code(
        "E_UNKNOWN_TERM",
        validate_prompt,
        {"entities": ["not-in-vocab"], "attributes": ["soft"]},
        workspace.conditions.vocab,
    )
    expect_code(
        "E_DIMENSION",
        load_vocabulary,
        repo_root / "configs" / "vocab.json",
        8,
    )


def test_png_header_resolution_and_mode(workspace, repo_root):
    png = (repo_root / "samples" / "images" / "geometry_a.png").read_bytes()
    expect_code("E_PNG_SIGNATURE", decode_png_rgb, b"not-a-png", width=48, height=32)
    expect_code("E_PNG_DIMENSION", decode_png_rgb, png, width=47, height=32)
    expect_code("E_PNG_MODE", decode_png_rgb, png, width=48, height=32, mode="RGBA")


def test_motion_grid_compatibility(workspace, repo_root):
    config = workspace.config
    motion = json.loads((repo_root / "samples" / "motion" / "motion_a.json").read_text())
    motion["vectors"] = motion["vectors"][:-1]
    expect_code(
        "E_MOTION_DIMENSION",
        _validate_motion,
        json.dumps(motion).encode(),
        config=config,
        asset_id="broken",
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("frame_count", 0, "E_FRAME_COUNT"),
        ("denoise_steps", 0, "E_DENOISE_STEPS"),
        ("chunk_rows", 0, "E_CHUNK_ROWS"),
        ("quant_profile", "", "E_ENUM_MISSING"),
        ("quant_profile", "q99", "E_ENUM_INVALID"),
        ("exchange_version", "", "E_EXCHANGE_VERSION"),
    ],
)
def test_config_enum_and_positive_integer_validation(workspace, field, value, code):
    raw = copy.deepcopy(workspace.config.raw)
    raw[field] = value
    expect_code(code, validate_config, raw)


def test_missing_vocab_and_fixed_enum(workspace):
    raw = copy.deepcopy(workspace.config.raw)
    raw.pop("vocab")
    expect_code("E_ENUM_MISSING", validate_config, raw)
    raw = copy.deepcopy(workspace.config.raw)
    raw["fixed_point"]["milli_scale"] = 100
    expect_code("E_FIXED_SCALE", validate_config, raw)


def test_v1_fixed_width_overflow(workspace):
    record = StageRecord(
        stage="denoise",
        frame=10000,
        guidance_permille=1000,
        noise_code=1,
        quant_bits=8,
    )
    expect_code("E_V1_WIDTH", serialize_v1, [record])


def test_cli_validation_uses_nonzero_exit(tmp_path, capsys):
    rc = main(["validate", "--config", str(tmp_path / "missing.json"), "--conditions", "missing.json"])
    assert rc != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "E_CONFIG_MISSING"


def test_edit_stream_rejects_unknown_command(workspace):
    expect_code("E_UNKNOWN_COMMAND", workspace.apply_commands, [{"op": "rewrite_everything"}])
    expect_code(
        "E_UNKNOWN_TERM",
        workspace.apply_commands,
        [{"op": "set_prompt", "id": "text_a", "prompt": {"entities": ["alien"], "attributes": ["soft"]}}],
    )
