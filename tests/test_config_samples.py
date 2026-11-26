from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from gsb.config import RunConfig, load_config, parse_config, validate_config_file
from gsb.constants import E_COMMAND_INVALID, E_CONFIG_INVALID, E_SAMPLE_FORMAT
from gsb.samples import (
    load_annotation_version,
    load_dataset,
    load_manifest,
    scan_png_directory,
    validate_png_bytes,
)


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_default_project_validates(project, project_root):
    assert not project.template_issues
    assert not project.dataset_issues
    assert project.dataset.sample_order == ["page_a", "page_b"]
    assert not scan_png_directory(project_root / "samples", config=project.config)


def test_config_rejects_non_divisible_window_and_bad_quantization(tmp_path):
    payload = RunConfig().to_dict()
    payload["window"]["width"] = 30
    payload["quantization"] = {"level": "Q7", "fixed_bits": 7}
    path = _write_yaml(tmp_path / "invalid.yaml", payload)

    issues = validate_config_file(path)

    assert issues
    assert all(issue.code == E_CONFIG_INVALID for issue in issues)
    assert any("divide layout.width" in issue.message for issue in issues)
    assert any("quantization.level" in issue.message for issue in issues)
    with pytest.raises(Exception):
        load_config(path)


def test_parse_config_requires_declared_template_version():
    payload = RunConfig().to_dict()
    payload["recognition"]["template_version"] = ""

    config, issues = parse_config(payload, source="<test>")

    assert config is None
    assert any(issue.code == E_CONFIG_INVALID for issue in issues)
    assert any("template_version" in issue.message for issue in issues)


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (lambda raw: raw["keypoints"][0].update({"x": 1.5}), "must be integers"),
        (lambda raw: raw["columns"][0].update({"x_min": 250}), "intervals overlap"),
        (lambda raw: raw["keypoints"][1].update({"reading_index": 99}), "continuous from zero"),
        (lambda raw: raw.update({"template_version": "unknown-v9"}), "does not match"),
    ],
)
def test_manifest_validation_rejects_invalid_truth(
    tmp_path,
    project,
    mutation,
    expected_message,
):
    raw = json.loads((project.root / "samples" / "page_a.manifest.json").read_text(encoding="utf-8"))
    mutation(raw)
    path = tmp_path / "page_a.manifest.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    manifest, issues = load_manifest(path, config=project.config)

    assert manifest is None
    assert issues
    assert any(expected_message in issue.message for issue in issues)
    assert all(issue.code in {E_SAMPLE_FORMAT, E_CONFIG_INVALID} for issue in issues)


def test_annotation_rejects_fractional_coordinates(tmp_path, project):
    record = project.dataset.samples["page_a"]
    raw = json.loads((project.root / "samples" / "annotations" / "page_a.truth-v1.json").read_text(encoding="utf-8"))
    raw["keypoints"][0]["x"] = float(raw["keypoints"][0]["x"]) + 0.5
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    annotation, issues = load_annotation_version(
        path,
        manifest=record.manifest,
        config=project.config,
    )

    assert annotation is None
    assert any(issue.code == E_SAMPLE_FORMAT for issue in issues)
    assert any("must be integers" in issue.message for issue in issues)


def test_truncated_png_is_rejected_and_other_sample_survives(tmp_path, project):
    samples = tmp_path / "samples"
    shutil.copytree(project.root / "samples", samples)
    image_path = samples / "page_a.png"
    payload = image_path.read_bytes()
    image_path.write_bytes(payload[: len(payload) // 2])

    dataset, issues = load_dataset(samples / "dataset.yaml", config=project.config)

    assert all(issue.sample_id != "page_b" for issue in issues)
    assert any(issue.sample_id == "page_a" and issue.code == E_SAMPLE_FORMAT for issue in issues)
    assert dataset.sample_order == ["page_b"]
    assert not validate_png_bytes(
        (project.root / "samples" / "page_b.png").read_bytes(),
        expected_width=project.config.layout.width,
        expected_height=project.config.layout.height,
        expected_mode="L",
        source="page_b.png",
    )


def test_command_stream_requires_continuous_sequence(tmp_path):
    from gsb.commands import load_command_stream

    path = tmp_path / "commands.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"seq": 1, "op": "noop"}),
                json.dumps({"seq": 3, "op": "noop"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    commands, issues = load_command_stream(path)

    assert len(commands) == 2
    assert any(issue.code == E_COMMAND_INVALID for issue in issues)
    assert any("continuous" in issue.message for issue in issues)
