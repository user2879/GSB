from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import repair_studio  # noqa: F401,E402
from repair_studio.imaging import LoadedSample, load_sample  # noqa: E402
from repair_studio.samplegen import generate_sample  # noqa: E402


def create_sample(
    root: Path,
    name: str = "sample",
    width: int = 64,
    height: int = 48,
    mask: dict[str, int] | None = None,
    commands: list[dict[str, Any]] | None = None,
    config_overrides: dict[str, Any] | None = None,
    seed: int = 777,
) -> LoadedSample:
    directory = root / name
    if mask is None:
        mask = {"x": 16, "y": 12, "width": 24, "height": 20}
    if commands is None:
        commands = [
            {"op": "set_repair_radius", "value": 3},
            {"op": "set_style_weights", "weights": [0.2, 0.7, 0.5]},
            {"op": "set_quant_levels", "value": 8},
            {"op": "set_mask_rect", "rect": {"x": 12, "y": 10, "width": 30, "height": 22}},
        ]
    generate_sample(
        directory,
        width=width,
        height=height,
        seed=seed,
        mask=mask,
        command_overrides=commands,
    )
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mask"] = mask
    config["repair"] = {"radius": 2, "iterations": 16}
    config["style"] = {
        "quant_levels": 8,
        "channel_weights": [0.35, 0.7, 0.45],
        "preserve_mask": False,
    }
    config["interaction"] = {
        "latency_budget_ms": 5,
        "chunk_rows": min(4, height),
        "thread_workers": 4,
        "snapshot_interval": 2,
    }
    config["fixed_bits"] = 16
    if config_overrides:
        for key, value in config_overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return load_sample(directory)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

