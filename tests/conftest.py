from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from softmax_layernorm.loader import load_run


@pytest.fixture
def make_run(tmp_path: Path):
    def factory(
        *,
        rows: int = 5,
        cols: int = 4,
        calls: list[dict[str, Any]] | None = None,
        tensor_spec: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        name: str = "run",
    ):
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True, exist_ok=True)
        if calls is None:
            calls = [
                {
                    "call_id": "call-1",
                    "sequence": 1,
                    "rows": rows,
                    "cols": cols,
                    "softmax_axis": 1,
                    "layernorm_axis": 1,
                    "epsilon": 1e-5,
                    "stabilize": True,
                }
            ]
        if tensor_spec is None:
            tensor_spec = {
                "shape": [rows, cols],
                "seed": 12345,
                "distribution": {"kind": "normal", "loc": 0.0, "scale": 1.5},
                "special_values": [],
            }
        if config is None:
            config = {
                "interaction_budget_ms": 2000.0,
                "thread_pool_workers": 4,
                "snapshot_interval_commands": 100,
                "layernorm_scale": 1.0,
                "snapshot_root": str(run_dir / "output"),
                "resume": True,
            }
        tensor_path = run_dir / "tensor.json"
        calls_path = run_dir / "calls.json"
        config_path = run_dir / "config.json"
        tensor_path.write_text(json.dumps(tensor_spec), encoding="utf-8")
        calls_path.write_text(json.dumps({"calls": calls}), encoding="utf-8")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        loaded = load_run(calls_path, tensor_path, config_path)
        return loaded, calls_path, tensor_path, config_path

    return factory
