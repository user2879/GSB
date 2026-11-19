from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import read_json
from .config import RecognitionConfig, apply_config_update
from .loading import ImageSource, scan_image_sources


def load_command_stream(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, dict):
        commands = value.get("commands")
    else:
        commands = value
    if not isinstance(commands, list):
        raise ValueError("command stream must be a list or an object with a commands list")
    normalized: list[dict[str, Any]] = []
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            raise ValueError(f"command {index} must be an object")
        if not command.get("type"):
            raise ValueError(f"command {index} is missing type")
        item = copy.deepcopy(command)
        item.setdefault("id", f"cmd_{index:03d}")
        normalized.append(item)
    return normalized


@dataclass
class CommandState:
    config: RecognitionConfig
    sources: dict[str, ImageSource]
    base_dir: Path
    initial_config: RecognitionConfig = field(init=False)

    def __post_init__(self) -> None:
        self.initial_config = self.config

    def ordered_sources(self) -> list[ImageSource]:
        return [self.sources[name] for name in sorted(self.sources)]

    def apply(self, command: dict[str, Any]) -> dict[str, Any]:
        kind = command.get("type")
        payload = copy.deepcopy(command.get("payload", {}))
        if not isinstance(payload, dict):
            raise ValueError(f"{command.get('id')}: payload must be an object")
        if kind in {"replace_image", "delete_image"}:
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{command.get('id')}: {kind} requires image name")
            if kind == "delete_image":
                if name not in self.sources:
                    raise ValueError(f"{command.get('id')}: image {name!r} does not exist")
                del self.sources[name]
                return {"effect": "deleted", "image": name}
            png_value = payload.get("source_png")
            truth_value = payload.get("source_truth")
            if not png_value:
                raise ValueError(f"{command.get('id')}: replace_image requires source_png")
            png_path = Path(png_value)
            if not png_path.is_absolute():
                png_path = self.base_dir / png_path
            if truth_value:
                truth_path = Path(truth_value)
                if not truth_path.is_absolute():
                    truth_path = self.base_dir / truth_path
            else:
                truth_path = png_path.with_suffix(".truth.json")
            if not png_path.is_file():
                raise ValueError(f"{command.get('id')}: replacement PNG not found: {png_path}")
            if not truth_path.is_file():
                raise ValueError(f"{command.get('id')}: replacement truth not found: {truth_path}")
            self.sources[name] = ImageSource(name=name, png_path=png_path, truth_path=truth_path)
            return {"effect": "replaced", "image": name, "source_png": str(png_path)}
        updated = apply_config_update(self.config, command)
        changed = updated != self.config
        self.config = updated
        return {
            "effect": "config_updated" if changed else "config_unchanged",
            "config_hash": updated.config_hash,
        }


def initial_command_state(data_dir: Path, config: RecognitionConfig, base_dir: Path) -> CommandState:
    return CommandState(
        config=config,
        sources={source.name: source for source in scan_image_sources(data_dir)},
        base_dir=base_dir,
    )

