from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_command(root: str | Path, command: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    op = command["op"]
    if op == "set_config":
        path = root / "config" / "reconstruction.json"
        config = _read_json(path)
        key = command["key"]
        if key not in config:
            raise KeyError(f"unknown config key {key}")
        config[key] = command["value"]
        _write_json(path, config)
        return {"applied": op, "key": key, "value": command["value"]}

    if op == "set_budget":
        path = root / "config" / "reconstruction.json"
        config = _read_json(path)
        for key, value in command["budget"].items():
            if key not in config:
                raise KeyError(f"unknown budget/config key {key}")
            config[key] = value
        _write_json(path, config)
        return {"applied": op, "budget": command["budget"]}

    if op == "set_extrinsic":
        path = root / "data" / "cameras.json"
        cameras = _read_json(path)
        index = int(command["view_index"])
        for camera in cameras["views"]:
            if int(camera["index"]) == index:
                camera["t"] = command["t"]
                if "R" in command:
                    camera["R"] = command["R"]
                _write_json(path, cameras)
                return {"applied": op, "view_index": index, "t": command["t"]}
        raise KeyError(f"view {index} not found")

    if op == "replace_view":
        index = int(command["view_index"])
        cameras_path = root / "data" / "cameras.json"
        cameras = _read_json(cameras_path)
        target = None
        for camera in cameras["views"]:
            if int(camera["index"]) == index:
                target = root / "data" / camera["image"]
                break
        if target is None:
            raise KeyError(f"view {index} not found")
        source = (root / "data" / command["source"]).resolve()
        data_root = (root / "data").resolve()
        try:
            source.relative_to(data_root)
        except ValueError:
            raise ValueError("replacement source must remain below data/")
        shutil.copyfile(source, target)
        return {"applied": op, "view_index": index, "target": str(target.relative_to(root))}

    raise ValueError(f"unsupported command op {op}")


def apply_commands(root: str | Path, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [apply_command(root, command) for command in commands]
