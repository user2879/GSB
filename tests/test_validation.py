from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from gsscene import codes
from gsscene.sample import generate_sample
from gsscene.util import atomic_write_json, read_json
from gsscene.validation import load_project

from conftest import copy_sample, update_config, write_edits


def _codes(project) -> set[str]:
    return {failure.code for failure in project.failures}


def test_sample_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first = generate_sample(tmp_path / "one", width=12, height=12, views=2, force=True)
    second = generate_sample(tmp_path / "two", width=12, height=12, views=2, force=True)
    for relative in sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file()):
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative


def test_valid_project_and_edit_plan(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    project = load_project(sample)
    assert project.config is not None
    assert sorted(project.views) == [0, 1, 2, 3, 4, 5]
    assert project.failures == []
    assert [edit["seq"] for edit in project.edits] == list(range(8))


def test_truncated_and_wrong_mode_images(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "truncated")
    image = sample / "images" / "view_0000.png"
    image.write_bytes(image.read_bytes()[:50])
    assert codes.E_IMAGE_TRUNCATED in _codes(load_project(sample))

    sample = copy_sample(base_sample, tmp_path / "mode")
    Image.new("L", (16, 16), color=128).save(sample / "images" / "view_0001.png")
    assert codes.E_IMAGE_MODE in _codes(load_project(sample))


def test_resolution_and_camera_checks(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "resolution")
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(sample / "images" / "view_0000.png")
    assert codes.E_IMAGE_RESOLUTION in _codes(load_project(sample))

    sample = copy_sample(base_sample, tmp_path / "camera")
    cameras = read_json(sample / "cameras.json")
    cameras[0]["intrinsic"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    cameras[1]["rotation"] = [[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    atomic_write_json(sample / "cameras.json", cameras)
    project_codes = _codes(load_project(sample))
    assert codes.E_CAMERA_INTRINSIC in project_codes
    assert codes.E_CAMERA_ROTATION in project_codes


def test_missing_and_orphan_cameras(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "missing")
    cameras = read_json(sample / "cameras.json")
    atomic_write_json(sample / "cameras.json", [item for item in cameras if item["view_index"] != 2])
    project = load_project(sample)
    assert codes.E_CAMERA_MISSING in _codes(project)
    assert 2 in project.excluded_views

    sample = copy_sample(base_sample, tmp_path / "orphan")
    cameras = read_json(sample / "cameras.json")
    extra = dict(cameras[0])
    extra["view_index"] = 99
    cameras.append(extra)
    atomic_write_json(sample / "cameras.json", cameras)
    assert codes.E_VIEW_CAMERA_ORPHAN in _codes(load_project(sample))


def test_config_business_codes(base_sample: Path, tmp_path: Path) -> None:
    cases = [
        ({"scale_decay": 0.0}, codes.E_CONFIG_SCALE_DECAY),
        ({"quant_gear": "BOGUS"}, codes.E_CONFIG_QUANT_GEAR),
        ({"frac_bits": 7}, codes.E_CONFIG_FRAC_BITS),
        ({"levels": 0}, codes.E_CONFIG_LEVELS),
        ({"kernel_budgets": [1, 0, 1]}, codes.E_CONFIG_KERNEL_BUDGET),
        ({"concurrency": 0}, codes.E_CONFIG_CONCURRENCY),
    ]
    for index, (changes, expected) in enumerate(cases):
        sample = copy_sample(base_sample, tmp_path / f"config-{index}")
        update_config(sample, **changes)
        project_codes = _codes(load_project(sample))
        assert expected in project_codes, (changes, project_codes)


def test_edit_plan_codes(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "edits")
    write_edits(
        sample,
        [
            {"seq": 0, "type": "unknown_edit", "params": {}},
            {"seq": 0, "type": "camera_tweak", "params": {"view_index": 0}},
            {"seq": 2, "type": "camera_tweak", "params": {"view_index": 999}},
            {
                "seq": 3,
                "type": "sh_modify",
                "params": {
                    "view_index": 0,
                    "levels": [0],
                    "kernel_index": 0,
                    "coefficient": 99,
                    "channel": 0,
                    "delta": 0.1,
                },
            },
        ],
    )
    project_codes = _codes(load_project(sample))
    assert codes.E_EDIT_TYPE in project_codes
    assert codes.E_EDIT_SEQUENCE in project_codes
    assert codes.E_EDIT_VIEW_MISSING in project_codes
    assert codes.E_EDIT_SH_COEFF in project_codes


def test_injection_spec_validation(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "injection")
    atomic_write_json(
        sample / "injections.json",
        [
            {"injection_id": "I1", "kind": "not_a_kind"},
            {"injection_id": "I1", "kind": "quant_gear_out_of_range"},
        ],
    )
    assert codes.E_INJECTION_SPEC in _codes(load_project(sample))
