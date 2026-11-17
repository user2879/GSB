from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from softmax_layernorm.errors import BusinessError
from softmax_layernorm.loader import generate_tensor, load_run


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _paths(tmp_path: Path, *, calls, tensor=None, config=None):
    tensor = tensor or {
        "shape": [3, 4],
        "seed": 9,
        "distribution": {"kind": "normal", "loc": 0, "scale": 1},
        "special_values": [],
    }
    config = config or {
        "interaction_budget_ms": 10,
        "thread_pool_workers": 2,
        "snapshot_interval_commands": 2,
        "snapshot_root": str(tmp_path / "out"),
    }
    return (
        _write(tmp_path / "calls.json", {"calls": calls}),
        _write(tmp_path / "tensor.json", tensor),
        _write(tmp_path / "config.json", config),
    )


def _call(**overrides):
    value = {
        "call_id": "c",
        "sequence": 1,
        "rows": 3,
        "cols": 4,
        "softmax_axis": 1,
        "layernorm_axis": 0,
        "epsilon": 1e-5,
        "stabilize": True,
    }
    value.update(overrides)
    return value


def test_invalid_json_returns_config_invalid(tmp_path: Path) -> None:
    calls = tmp_path / "calls.json"
    calls.write_text("{not-json", encoding="utf-8")
    tensor = _write(tmp_path / "tensor.json", {"shape": [3, 4], "seed": 1, "distribution": "normal"})
    config = _write(
        tmp_path / "config.json",
        {"interaction_budget_ms": 10, "thread_pool_workers": 1, "snapshot_interval_commands": 1},
    )
    with pytest.raises(BusinessError) as error:
        load_run(calls, tensor, config)
    assert error.value.failures[0].code == "E_CONFIG_INVALID"


def test_shape_mismatch_code(tmp_path: Path) -> None:
    with pytest.raises(BusinessError) as error:
        load_run(*_paths(tmp_path, calls=[_call(rows=2)]))
    assert error.value.failures[0].code == "E_SHAPE_MISMATCH"


def test_axis_out_of_range_code(tmp_path: Path) -> None:
    with pytest.raises(BusinessError) as error:
        load_run(*_paths(tmp_path, calls=[_call(softmax_axis=2)]))
    assert error.value.failures[0].code == "E_AXIS_OUT_OF_RANGE"


@pytest.mark.parametrize("epsilon", [0, -1.0])
def test_invalid_epsilon_code(tmp_path: Path, epsilon: float) -> None:
    with pytest.raises(BusinessError) as error:
        load_run(*_paths(tmp_path, calls=[_call(epsilon=epsilon)]))
    assert error.value.failures[0].code == "E_EPS_INVALID"


def test_non_finite_epsilon_code(tmp_path: Path) -> None:
    calls_path, tensor_path, config_path = _paths(tmp_path, calls=[_call()])
    calls_text = calls_path.read_text(encoding="utf-8").replace("1e-05", "1e309")
    calls_path.write_text(calls_text, encoding="utf-8")
    with pytest.raises(BusinessError) as error:
        load_run(calls_path, tensor_path, config_path)
    assert error.value.failures[0].code == "E_EPS_INVALID"


def test_non_positive_concurrency_is_config_invalid(tmp_path: Path) -> None:
    config = {
        "interaction_budget_ms": 10,
        "thread_pool_workers": 0,
        "snapshot_interval_commands": 1,
    }
    with pytest.raises(BusinessError) as error:
        load_run(*_paths(tmp_path, calls=[_call()], config=config))
    assert error.value.failures[0].code == "E_CONFIG_INVALID"


def test_tensor_generation_is_repeatable_and_preserves_special_values(tmp_path: Path) -> None:
    loaded = load_run(
        *_paths(
            tmp_path,
            calls=[_call()],
            tensor={
                "shape": [3, 4],
                "seed": 42,
                "distribution": {"kind": "uniform", "low": -2, "high": 2},
                "special_values": [
                    {"row": 0, "col": 1, "value": 123.5},
                    {"row": 2, "col": 3, "value": "-inf"},
                ],
            },
        )
    )
    generated = generate_tensor(loaded.tensor_spec)
    assert np.array_equal(generated, loaded.tensor)
    assert generated[0, 1] == np.float32(123.5)
    assert np.isneginf(generated[2, 3])
