import json
from pathlib import Path

import pytest

from mlimgedit.errors import (
    E_BBOX_OUT_OF_RANGE,
    E_LABEL_UNKNOWN,
    E_SAMPLE_FORMAT,
)
from mlimgedit.injection import load_injections
from mlimgedit.validation import validate_directory


def _write_bad_pair(directory: Path, name: str, png_bytes: bytes, annotation: dict):
    (directory / f"{name}.png").write_bytes(png_bytes)
    (directory / f"{name}.json").write_text(json.dumps(annotation), encoding="utf-8")


def test_good_dataset_passes(config, sample_dir):
    result = validate_directory(sample_dir, config)
    assert not result.failures
    assert result.valid_names == sorted(result.valid_names)
    assert len(result.valid) == 10


def test_truncated_png_maps_to_sample_format(config, sample_dir):
    target = sample_dir / "sample_000.png"
    raw = target.read_bytes()
    target.write_bytes(raw[: len(raw) // 2])
    result = validate_directory(sample_dir, config)
    failure = next(item for item in result.failures if item.sample == "sample_000")
    assert failure.code == E_SAMPLE_FORMAT
    assert len(result.valid) == 9


def test_unknown_label_code(config, sample_dir):
    path = sample_dir / "sample_002.json"
    annotation = json.loads(path.read_text())
    annotation["labels"].append("not_a_real_label")
    annotation["boxes"]["not_a_real_label"] = [0, 0, 3, 3]
    path.write_text(json.dumps(annotation), encoding="utf-8")
    result = validate_directory(sample_dir, config)
    failure = next(item for item in result.failures if item.sample == "sample_002")
    assert failure.code == E_LABEL_UNKNOWN


def test_bbox_out_of_range_code(config, sample_dir):
    path = sample_dir / "sample_002.json"
    annotation = json.loads(path.read_text())
    label = annotation["labels"][0]
    annotation["boxes"][label] = [0, 0, config.image_size[0] + 5, config.image_size[1] + 5]
    path.write_text(json.dumps(annotation), encoding="utf-8")
    result = validate_directory(sample_dir, config)
    failure = next(item for item in result.failures if item.sample == "sample_002")
    assert failure.code == E_BBOX_OUT_OF_RANGE


def test_non_integer_bbox_is_sample_format(config, sample_dir):
    path = sample_dir / "sample_002.json"
    annotation = json.loads(path.read_text())
    label = annotation["labels"][0]
    annotation["boxes"][label] = [0.1, 0, 3, 3]
    path.write_text(json.dumps(annotation), encoding="utf-8")
    result = validate_directory(sample_dir, config)
    failure = next(item for item in result.failures if item.sample == "sample_002")
    assert failure.code == E_SAMPLE_FORMAT


def test_injection_ids_are_reported_and_isolated(config, sample_dir, injections_path):
    injections = load_injections(injections_path)
    result = validate_directory(sample_dir, config, injections)
    by_id = {failure.injection_id: failure for failure in result.failures}
    assert set(by_id) == {"INJ-001", "INJ-002", "INJ-003", "INJ-004"}
    assert by_id["INJ-001"].code == E_SAMPLE_FORMAT
    assert by_id["INJ-002"].code == E_LABEL_UNKNOWN
    assert by_id["INJ-003"].code == E_BBOX_OUT_OF_RANGE
    assert by_id["INJ-004"].code == E_SAMPLE_FORMAT
    # Remaining samples are still validated despite the failures.
    assert len(result.valid) == 10 - 4


def test_failure_order_is_stable(config, sample_dir, injections_path):
    injections = load_injections(injections_path)
    first = validate_directory(sample_dir, config, injections).failures
    second = validate_directory(sample_dir, config, injections).failures
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]


def test_config_validation_rejects_bad_ratios_and_threshold(config_dict):
    from mlimgedit.config import load_config_dict
    from mlimgedit.errors import ConfigError

    bad = dict(config_dict)
    bad["positive_ratios"] = {label: 1.5 for label in config_dict["labels"]}
    with pytest.raises(ConfigError):
        load_config_dict(bad)
    bad = dict(config_dict)
    bad["threshold"] = float("nan")
    with pytest.raises(ConfigError):
        load_config_dict(bad)
