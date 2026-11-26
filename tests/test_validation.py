from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixed_lab.config import normalize_config
from fixed_lab.dataset import load_samples_jsonl
from fixed_lab.errors import InputValidationError


COLUMNS = ["a", "b"]


def write_jsonl(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_jsonl_collects_invalid_rows_without_discarding_valid(tmp_path: Path) -> None:
    valid = {"id": "S1", "seed": 1, "features": [1.0, 2.0], "target": 3.0, "weight": 1.0}
    duplicate = dict(valid)
    bad_target = {"id": "S2", "seed": 2, "features": [1.0, 2.0], "target": "nan", "weight": 1.0}
    wrong_dim = {"id": "S3", "seed": 3, "features": [1.0], "target": 2.0, "weight": 1.0}
    path = write_jsonl(tmp_path / "samples.jsonl", [
        json.dumps(valid),
        json.dumps(duplicate),
        json.dumps(bad_target),
        json.dumps(wrong_dim),
        "{not json",
    ])

    records, errors = load_samples_jsonl(path, COLUMNS)

    assert [record.id for record in records] == ["S1"]
    fields = {error["field"] for error in errors}
    assert any("id" in field for field in fields)
    assert any("target" in field for field in fields)
    assert any("features" in field for field in fields)
    assert any("json" in field for field in fields)
    assert all(error["code"] == "E_INPUT_INVALID" for error in errors)


def minimal_config() -> dict:
    return {
        "feature_columns": COLUMNS,
        "penalty": "ridge",
        "alphas": [0.1, 1.0],
        "min_alpha": 0,
        "max_alpha": 10,
        "fixed_point": {"decimal_places": 6, "quant_level": 1},
    }


def test_config_rejects_bad_penalty_alpha_order_and_gear() -> None:
    bad_penalty = minimal_config() | {"penalty": "elastic_net"}
    with pytest.raises(InputValidationError) as exc:
        normalize_config(bad_penalty)
    assert any("penalty" in error["field"] for error in exc.value.errors)

    bad_alphas = minimal_config() | {"alphas": [1.0, 0.5]}
    with pytest.raises(InputValidationError) as exc:
        normalize_config(bad_alphas)
    assert any("strictly increasing" in error["message"] for error in exc.value.errors)

    bad_gear = minimal_config() | {"fixed_point": {"decimal_places": 4, "quant_level": 5}}
    with pytest.raises(InputValidationError) as exc:
        normalize_config(bad_gear)
    assert any("quant_level" in error["field"] for error in exc.value.errors)


def test_config_accepts_valid_decimal_path() -> None:
    config = normalize_config(minimal_config())
    assert config["paths"]["default"]["penalty"] == "ridge"
    assert [str(alpha) for alpha in config["paths"]["default"]["alphas"]] == ["0.1", "1.0"]
