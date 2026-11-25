from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ins_diag.config import ConfigValidationError, load_config


def test_valid_example_config_loads(examples_dir: Path) -> None:
    config = load_config(examples_dir / "config.yaml")
    assert config.window_length == 64
    assert config.window_shift == 32
    assert config.families == 3
    assert config.fixed_point.fractional_bits == 12
    assert config.lease.ttl_ms == 5000
    assert len(config.waveform.devices) == 2


@pytest.mark.parametrize(
    ("mutator", "expected_path"),
    [
        (lambda c: c["window"].__setitem__("shift", 65), "$.window.shift"),
        (
            lambda c: c["clustering"].__setitem__("family_limit", 7),
            "$.clustering.family_limit",
        ),
        (lambda c: c["kde"].__setitem__("bandwidth", "0.17"), "$.kde.bandwidth"),
        (
            lambda c: c["clustering"]["fixed_point"].__setitem__("fractional_bits", 9),
            "$.clustering.fixed_point.fractional_bits",
        ),
        (
            lambda c: c["clustering"]["fixed_point"].__setitem__("quantization_units", 3),
            "$.clustering.fixed_point.quantization_units",
        ),
    ],
)
def test_config_field_errors_have_codes_and_paths(
    tmp_path: Path,
    examples_dir: Path,
    base_config: dict,
    mutator,
    expected_path: str,
) -> None:
    value = copy.deepcopy(base_config)
    value["waveform_path"] = str((examples_dir / "waveform.yaml").resolve())
    value["lease_declaration_path"] = str((examples_dir / "lease.yaml").resolve())
    value["edits_path"] = str((examples_dir / "edits.yaml").resolve())
    mutator(value)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigValidationError) as captured:
        load_config(path)
    issues = captured.value.issues
    assert any(issue.code == "E_CONFIG_INVALID" and issue.path == expected_path for issue in issues)


def test_duplicate_device_and_invalid_lease_are_rejected(
    tmp_path: Path, examples_dir: Path, base_config: dict
) -> None:
    waveform = yaml.safe_load((examples_dir / "waveform.yaml").read_text(encoding="utf-8"))
    waveform["devices"][1]["id"] = waveform["devices"][0]["id"]
    waveform_path = tmp_path / "waveform.yaml"
    waveform_path.write_text(yaml.safe_dump(waveform, sort_keys=False), encoding="utf-8")
    lease_path = tmp_path / "lease.yaml"
    lease_path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "worker_id": "x", "ttl_ms": 0, "restore_generation": 0},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = copy.deepcopy(base_config)
    config["waveform_path"] = str(waveform_path)
    config["lease_declaration_path"] = str(lease_path)
    config.pop("edits_path", None)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigValidationError) as captured:
        load_config(path)
    paths = {issue.path for issue in captured.value.issues}
    assert "waveform.devices[1].id" in paths
    assert "lease.ttl_ms" in paths

