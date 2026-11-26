from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdet_render.cli import main
from mdet_render.errors import BusinessError
from mdet_render.formats import (
    NOISE_CODE_TO_NAME,
    V1_LINE_WIDTH,
    StageRecord,
    parse_v1,
    parse_v2,
    to_v1_lines,
    v1_to_v2,
    v2_to_v1,
)


def _v2_document() -> dict:
    return {
        "format_version": 2,
        "records": [
            {
                "stage": "denoise",
                "frame": 2,
                "step": 5,
                "block": 1,
                "guide_strength": 1.23456,
                "noise_scheme": "perlin",
                "precision_tag": "v2-high-precision",
                "quantization_bits": 10,
                "quantization_level": 2,
                "residual": -13,
                "digest": "ab" * 32,
                "extra": {"lease_generation": 7, "trace": [1, 2, 3]},
            },
            {
                "stage": "block",
                "frame": 0,
                "guide_strength": 0.0004,
                "noise_scheme": "uniform",
                "precision_tag": "v2-block",
                "quantization_bits": 8,
                "quantization_level": 1,
                "residual": 9,
                "digest": "cd" * 32,
            },
        ],
    }


def test_v2_to_v1_is_fixed_width_and_uses_permille() -> None:
    document = _v2_document()
    v1, sidecar = v2_to_v1(document)
    lines = v1.splitlines()
    assert len(lines) == 2
    assert all(len(line) == V1_LINE_WIDTH for line in lines)
    first = lines[0]
    assert first.split("|")[4].strip() == "1235"
    assert first.split("|")[5].strip() == "2"
    assert sidecar["records"][0]["guide_rounding_residual"] == pytest.approx(1.23456 - 1.235)
    assert sidecar["records"][0]["precision_tag"] == "v2-high-precision"
    assert sidecar["records"][0]["overwritten_fields"]["extra"]["extra"] == {
        "lease_generation": 7,
        "trace": [1, 2, 3],
    }


def test_v2_v1_v2_roundtrip_restores_exact_v2_from_sidecar() -> None:
    document = _v2_document()
    v1, sidecar = v2_to_v1(document)
    assert v1_to_v2(v1, sidecar) == document


def test_v1_upgrade_maps_noise_and_generates_precision_tag() -> None:
    record = StageRecord(
        stage="denoise",
        frame=1,
        step=3,
        guide_strength=1.125,
        noise_scheme="uniform",
        quantization_bits=12,
        quantization_level=3,
        residual=4,
        digest="ef" * 32,
    )
    v1 = to_v1_lines([record])
    upgraded = v1_to_v2(v1)
    raw = upgraded["records"][0]
    assert raw["guide_strength"] == pytest.approx(1.125)
    assert raw["noise_scheme"] == NOISE_CODE_TO_NAME[1]
    assert "noise_code=1" in raw["precision_tag"]
    assert raw["_source_version"] == 1


def test_missing_v1_fields_are_marked_and_remain_missing() -> None:
    record = StageRecord(stage="denoise", guide_strength=None, noise_scheme=None,
                         digest="00" * 32)
    v1 = to_v1_lines([record])
    parsed = parse_v1(v1)[0]
    assert "guide_strength" in parsed.missing_fields
    assert "noise_scheme" in parsed.missing_fields
    upgraded = v1_to_v2(v1)
    raw = upgraded["records"][0]
    assert "guide_strength" not in raw
    assert "noise_scheme" not in raw
    assert "_missing_fields" in raw


def test_unknown_noise_scheme_fails_loudly() -> None:
    record = StageRecord(stage="denoise", noise_scheme="mystery", digest="00" * 32)
    with pytest.raises(BusinessError) as caught:
        to_v1_lines([record])
    assert caught.value.code == "E_FORMAT_NOISE_SCHEME"


def test_parse_v2_retains_unknown_fields() -> None:
    parsed = parse_v2(_v2_document())
    assert parsed[0].extra["extra"]["lease_generation"] == 7
    assert parsed[0].source_version == 2
    assert parsed[0].as_v2_dict()["extra"]["trace"] == [1, 2, 3]


def test_conversion_cli_roundtrip(tmp_path: Path, capsys) -> None:
    source = tmp_path / "v2.json"
    v1 = tmp_path / "records.v1"
    sidecar = tmp_path / "records.sidecar.json"
    output = tmp_path / "restored.json"
    source.write_text(json.dumps(_v2_document()), encoding="utf-8")
    assert main(["v2-to-v1", "--input", str(source), "--output", str(v1),
                 "--sidecar", str(sidecar)]) == 0
    capsys.readouterr()
    assert main(["v1-to-v2", "--input", str(v1), "--output", str(output),
                 "--sidecar", str(sidecar)]) == 0
    capsys.readouterr()
    assert json.loads(output.read_text()) == _v2_document()
