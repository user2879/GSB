from __future__ import annotations

import json
import shutil
from pathlib import Path

from layout_diag.generator import generate_samples, write_default_config


def make_dataset(root: Path, specs=None):
    root.mkdir(parents=True, exist_ok=True)
    if specs is None:
        generate_samples(root, config_path=root / "config.json")
    else:
        generate_samples(root, specs=specs, config_path=root / "config.json")
    return root


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_workspace(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def make_default_config(path: Path, **overrides):
    write_default_config(path)
    config = read_config(path)
    for key, value in overrides.items():
        config[key] = value
    write_config(path, config)
    return config
