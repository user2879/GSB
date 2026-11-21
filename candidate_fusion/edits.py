"""Apply deterministic sample edit commands to a workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import atomic_copy, atomic_write_json, read_json


CONFIG_SETTERS = {
    "set_channel_capacity": ("channel", "capacity"),
    "set_channel_quota": ("root", "channel_quota"),
    "set_global_budget": ("root", "global_budget"),
    "set_max_workers": ("root", "max_workers"),
    "set_normalization": ("root", "normalization"),
    "set_quant_bits": ("root", "quant_bits"),
    "set_rounding": ("root", "rounding"),
    "set_trunc_k": ("root", "trunc_k"),
    "set_weight": ("channel_weight", "weight"),
}


def _apply_config_edit(config: dict[str, Any], edit: dict[str, Any]) -> None:
    edit_type = edit["type"]
    if edit_type not in CONFIG_SETTERS:
        raise ValueError(f"unsupported config edit type: {edit_type}")
    target, field = CONFIG_SETTERS[edit_type]
    if "value" not in edit:
        raise ValueError(f"edit {edit.get('id')} is missing value")
    if target == "root":
        config[field] = edit["value"]
    else:
        channel = edit.get("channel")
        if not isinstance(channel, str) or channel not in config.get("channels", {}):
            raise ValueError(f"edit {edit.get('id')} targets unknown channel: {channel!r}")
        config["channels"][channel][field] = edit["value"]


def apply_edit_sequence(
    workspace: Path | str,
    sequence_path: Path | str,
    *,
    edit_ids: set[str] | None = None,
) -> list[str]:
    workspace = Path(workspace)
    sequence_path = Path(sequence_path)
    sequence = read_json(sequence_path)
    if not isinstance(sequence, dict) or not isinstance(sequence.get("edits"), list):
        raise ValueError("edit sequence must contain an edits list")
    config_path = workspace / "config.json"
    config = read_json(config_path)
    applied: list[str] = []
    config_dirty = False
    for edit in sequence["edits"]:
        if not isinstance(edit, dict) or not isinstance(edit.get("type"), str):
            raise ValueError("each edit must be an object with a type")
        edit_id = str(edit.get("id") or edit["type"])
        if edit_ids is not None and edit_id not in edit_ids:
            continue
        edit_type = edit["type"]
        if edit_type in {"append_fragment", "replace_fragment"}:
            channel = edit.get("channel")
            source = edit.get("source")
            destination = edit.get("destination")
            if not all(isinstance(value, str) and value for value in (channel, source, destination)):
                raise ValueError(f"fragment edit {edit.get('id')} requires channel/source/destination")
            source_path = (sequence_path.parent / source).resolve()
            destination_path = (workspace / destination).resolve()
            if workspace.resolve() not in destination_path.parents:
                raise ValueError(f"fragment destination escapes workspace: {destination}")
            expected_prefix = f"fragments/{channel}/"
            if not destination.startswith(expected_prefix):
                raise ValueError(f"fragment destination must be under {expected_prefix}")
            if edit_type == "replace_fragment" and not destination_path.exists():
                raise ValueError(f"replace target does not exist: {destination}")
            atomic_copy(source_path, destination_path)
        else:
            _apply_config_edit(config, edit)
            config_dirty = True
        applied.append(edit_id)
    if config_dirty:
        atomic_write_json(config_path, config)
    return applied
