from __future__ import annotations

import copy
import shutil
from pathlib import Path

from ancient_vlayout.engine import RecognitionEngine
from ancient_vlayout.util import atomic_write_json, read_json


def _first_char(result):
    return result.recognition["images"][0]["columns"][0]["chars"][0]


def test_bit_width_changes_fixed_value_numeric_value_and_digits(
    samples_copy: Path, tmp_path: Path
) -> None:
    base_config = read_json(samples_copy / "config.json")
    values = {}
    recognitions = {}
    for bits in (8, 12, 16):
        config = copy.deepcopy(base_config)
        config["fixed_point_bits"] = bits
        atomic_write_json(samples_copy / "config.json", config)
        result = RecognitionEngine(samples_copy, state_dir=tmp_path / f"bits-{bits}").settle()
        values[bits] = _first_char(result)
        recognitions[bits] = result.recognition
    assert recognitions[8]["images"][0]["columns"][0]["chars"][0]["recognized_char"] == (
        recognitions[12]["images"][0]["columns"][0]["chars"][0]["recognized_char"]
    )
    assert values[8]["confidence_fixed"] != values[12]["confidence_fixed"]
    assert values[12]["confidence_fixed"] != values[16]["confidence_fixed"]
    assert values[8]["confidence_value"] != values[12]["confidence_value"]
    assert values[12]["confidence_value"] != values[16]["confidence_value"]
    assert [values[8]["confidence_digits"], values[12]["confidence_digits"], values[16]["confidence_digits"]] == [3, 4, 5]
    assert values[8]["confidence_quantum"] != values[12]["confidence_quantum"]


def test_changing_only_bit_width_invalidates_only_match(samples_copy: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    assert len(RecognitionEngine(samples_copy, state_dir=state).settle().pending) == 0
    config = read_json(samples_copy / "config.json")
    config["fixed_point_bits"] = 8
    atomic_write_json(samples_copy / "config.json", config)
    result = RecognitionEngine(samples_copy, state_dir=state).settle()
    assert {item["stage"] for item in result.report["cache"]["recomputed"]} == {"match"}
    assert {item["stage"] for item in result.report["cache"]["hits"]} == {
        "columns",
        "split",
        "border",
    }
