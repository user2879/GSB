from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fixedvideo.config import load_command_stream, load_render_config
from fixedvideo.errors import (
    E_CONFIG_INVALID,
    E_NOISE_TABLE_MISMATCH,
    E_PATCH_OUT_OF_RANGE,
    E_SEED_OUT_OF_RANGE,
    RenderError,
)
from fixedvideo.state import apply_command, initial_state


def _rewrite_config(sample_dir: Path, **changes: object) -> Path:
    path = sample_dir / "render_config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(changes)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_seed_out_of_range_config(sample_dir: Path) -> None:
    path = _rewrite_config(sample_dir, seed=-1)
    with pytest.raises(RenderError) as exc_info:
        load_render_config(path)
    assert exc_info.value.code == E_SEED_OUT_OF_RANGE


def test_fixed_point_enum_validation(sample_dir: Path) -> None:
    path = _rewrite_config(sample_dir, fixed_point_bits=11)
    with pytest.raises(RenderError) as exc_info:
        load_render_config(path)
    assert exc_info.value.code == E_CONFIG_INVALID


def test_noise_table_version_declared_set(sample_dir: Path) -> None:
    path = sample_dir / "render_config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["noise_table"]["version"] = "nt-v9"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RenderError) as exc_info:
        load_render_config(path)
    assert exc_info.value.code == E_NOISE_TABLE_MISMATCH


def test_noise_table_file_version_mismatch(sample_dir: Path) -> None:
    path = sample_dir / "render_config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["noise_table"]["path"] = "noise_tables/nt-v2.npz"
    raw["noise_table"]["version"] = "nt-v1"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RenderError) as exc_info:
        load_render_config(path)
    assert exc_info.value.code == E_NOISE_TABLE_MISMATCH


def test_initial_latent_descriptor_must_match_data_length(sample_dir: Path) -> None:
    bad = sample_dir / "bad_initial.npz"
    np.savez(
        bad,
        latent=np.zeros((3, 8, 8), dtype=np.float32),
        shape=np.asarray([3, 8, 7], dtype=np.int64),
        dtype=np.asarray("float32"),
    )
    path = sample_dir / "render_config.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["initial_image"]["path"] = "bad_initial.npz"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RenderError) as exc_info:
        load_render_config(path)
    assert exc_info.value.code == E_CONFIG_INVALID


def test_command_field_type_validation(sample_dir: Path) -> None:
    path = sample_dir / "commands_bad_type.json"
    path.write_text(
        json.dumps({"commands": [{"command_id": "bad", "op": "set_seed", "seed": "123"}]}),
        encoding="utf-8",
    )
    with pytest.raises(RenderError) as exc_info:
        load_command_stream(path)
    assert exc_info.value.code == E_CONFIG_INVALID


def test_command_seed_range_and_patch_bounds(sample_dir: Path) -> None:
    config = load_render_config(sample_dir / "render_config.json")
    state = initial_state(config)
    with pytest.raises(RenderError) as seed_error:
        apply_command(
            config,
            state,
            {"command_id": "s", "op": "set_seed", "seed": 2**40},
            command_index=0,
        )
    assert seed_error.value.code == E_SEED_OUT_OF_RANGE
    with pytest.raises(RenderError) as patch_error:
        apply_command(
            config,
            state,
            {
                "command_id": "p",
                "op": "add_patch",
                "patch": {
                    "patch_id": "oob",
                    "step": 4,
                    "frame": 0,
                    "x": 7,
                    "y": 0,
                    "width": 4,
                    "height": 2,
                    "delta": 0.2,
                },
            },
            command_index=1,
        )
    assert patch_error.value.code == E_PATCH_OUT_OF_RANGE
