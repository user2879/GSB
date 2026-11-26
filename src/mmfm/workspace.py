"""Editable render workspace and validated edit-command stream."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import content_digest, load_json
from .config import RenderConfig, load_render_config, require_mapping, validate_config
from .conditions import ResolvedConditions, resolve_conditions, validate_prompt
from .errors import (
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
    E_CONDITION_REF,
    E_UNKNOWN_COMMAND,
    fail,
)


@dataclass
class Workspace:
    config: RenderConfig
    conditions: ResolvedConditions
    config_raw: dict[str, Any]
    manifest_raw: dict[str, Any]

    @property
    def input_hash(self) -> str:
        return content_digest(
            {
                "config": self.config_raw,
                "manifest": self.manifest_raw,
                "frames": self.conditions.serialisable()["frames"],
            }
        )

    @property
    def config_hash(self) -> str:
        return content_digest(self.config_raw)

    def serialisable(self) -> dict[str, Any]:
        return {
            "config_raw": self.config_raw,
            "config_source_path": self.config.source_path,
            "conditions": self.conditions.serialisable(),
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
        }

    @classmethod
    def from_serialised(cls, value: dict[str, Any]) -> "Workspace":
        config_raw = copy.deepcopy(value["config_raw"])
        config = validate_config(config_raw, source_path=str(value.get("config_source_path", "<snapshot>")))
        conditions = ResolvedConditions.from_serialised(value["conditions"])
        return cls(
            config=config,
            conditions=conditions,
            config_raw=config_raw,
            manifest_raw=copy.deepcopy(conditions.manifest),
        )

    def apply_commands(self, commands: Any) -> tuple["Workspace", list[dict[str, Any]]]:
        parsed = load_command_stream(commands)
        config_raw = copy.deepcopy(self.config_raw)
        manifest = copy.deepcopy(self.manifest_raw)
        applied: list[dict[str, Any]] = []
        for index, command in enumerate(parsed):
            op = command.get("op")
            if not isinstance(op, str):
                fail(E_COMMAND_INVALID, "edit command op is missing", index=index)
            if op in {"set_denoise_steps", "set_chunk_rows", "set_quant_profile", "set_exchange_version"}:
                field = {
                    "set_denoise_steps": "denoise_steps",
                    "set_chunk_rows": "chunk_rows",
                    "set_quant_profile": "quant_profile",
                    "set_exchange_version": "exchange_version",
                }[op]
                if "value" not in command:
                    fail(E_COMMAND_INVALID, "edit command value is missing", index=index, op=op)
                config_raw[field] = command["value"]
                applied.append({"index": index, "op": op, "field": field, "value": command["value"]})
            elif op == "set_fusion_weight":
                condition_type = command.get("condition_type")
                if condition_type not in {"image", "motion", "text"}:
                    fail(
                        E_COMMAND_INVALID,
                        "set_fusion_weight condition_type must be image, motion, or text",
                        index=index,
                    )
                value = command.get("value_permille")
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    fail(E_COMMAND_INVALID, "set_fusion_weight value_permille must be non-negative", index=index)
                config_raw.setdefault("fusion_weights_permille", {})[condition_type] = value
                applied.append(
                    {
                        "index": index,
                        "op": op,
                        "condition_type": condition_type,
                        "value_permille": value,
                    }
                )
            elif op in {"set_image", "set_image_path", "set_motion", "set_motion_path"}:
                kind = "images" if op.startswith("set_image") else "motions"
                asset_id = command.get("id")
                path = command.get("path")
                if not isinstance(asset_id, str) or not asset_id or not isinstance(path, str) or not path:
                    fail(E_COMMAND_INVALID, f"{op} requires string id and path", index=index)
                manifest.setdefault(kind, {})[asset_id] = path
                applied.append({"index": index, "op": op, "kind": kind, "id": asset_id, "path": path})
            elif op == "set_prompt":
                text_id = command.get("id")
                prompt = command.get("prompt")
                if not isinstance(text_id, str) or not text_id or not isinstance(prompt, dict):
                    fail(E_COMMAND_INVALID, "set_prompt requires string id and prompt object", index=index)
                manifest.setdefault("texts", {})[text_id] = copy.deepcopy(prompt)
                applied.append({"index": index, "op": op, "id": text_id})
            elif op == "assign_frame":
                frame = command.get("frame")
                if isinstance(frame, bool) or not isinstance(frame, int) or frame <= 0:
                    fail(E_COMMAND_INVALID, "assign_frame frame must be a positive integer", index=index)
                updated = False
                for item in manifest.get("frames", []):
                    if item.get("frame") == frame:
                        for field in ("text", "image", "motion", "guidance_permille", "seed_offset"):
                            if field in command:
                                item[field] = command[field]
                                updated = True
                        break
                if not updated:
                    fail(E_CONDITION_REF, "assign_frame references an unknown frame", frame=frame, index=index)
                applied.append({"index": index, "op": op, "frame": frame})
            elif op == "set_guidance_permille":
                frame = command.get("frame")
                value = command.get("value")
                if isinstance(frame, bool) or not isinstance(frame, int) or frame <= 0:
                    fail(E_COMMAND_INVALID, "set_guidance_permille frame must be a positive integer", index=index)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    fail(E_COMMAND_INVALID, "set_guidance_permille value must be non-negative", index=index)
                found = False
                for item in manifest.get("frames", []):
                    if item.get("frame") == frame:
                        item["guidance_permille"] = value
                        found = True
                        break
                if not found:
                    fail(E_CONDITION_REF, "set_guidance_permille references an unknown frame", frame=frame)
                applied.append({"index": index, "op": op, "frame": frame, "value": value})
            elif op == "set_max_workers":
                value = command.get("value")
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    fail(E_COMMAND_INVALID, "set_max_workers value must be a positive integer", index=index)
                config_raw["max_workers"] = value
                applied.append({"index": index, "op": op, "value": value})
            elif op == "set_budget":
                value = command.get("value")
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    fail(E_COMMAND_INVALID, "set_budget value must be non-negative or null", index=index)
                config_raw["max_denoise_points"] = value
                applied.append({"index": index, "op": op, "value": value})
            else:
                fail(E_UNKNOWN_COMMAND, "unknown edit command", index=index, op=op)
        # Validate everything before exposing the new workspace.  Resolution
        # performs prompt/vocabulary, PNG, and motion validation as well.
        config = validate_config(config_raw, source_path=self.config.source_path)
        # resolve_conditions rereads a manifest file; patching the manifest in
        # memory is required for edit command execution.
        conditions = _resolve_from_memory(config, manifest, self.conditions.manifest_path, self.conditions.vocab)
        return (
            Workspace(
                config=config,
                conditions=conditions,
                config_raw=config_raw,
                manifest_raw=manifest,
            ),
            applied,
        )


def _resolve_from_memory(
    config: RenderConfig,
    manifest: dict[str, Any],
    manifest_path: str,
    vocab: Any,
) -> ResolvedConditions:
    """Resolve a mutated manifest without overwriting its source file."""

    import tempfile

    from .conditions import resolve_conditions

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=Path(manifest_path).parent, delete=False
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        temp_path = handle.name
    try:
        # Relative asset paths still resolve against the manifest's parent;
        # the temporary file is deliberately placed in that same directory.
        return resolve_conditions(config, temp_path, vocabulary=vocab)
    finally:
        try:
            Path(temp_path).unlink()
        except OSError:
            pass


def load_workspace(
    config_path: str | Path,
    conditions_path: str | Path,
    *,
    vocabulary: Any | None = None,
) -> Workspace:
    config = load_render_config(config_path)
    conditions = resolve_conditions(config, conditions_path, vocabulary=vocabulary)
    return Workspace(
        config=config,
        conditions=conditions,
        config_raw=copy.deepcopy(config.raw),
        manifest_raw=copy.deepcopy(conditions.manifest),
    )


def load_command_stream(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            value = load_json(path)
        except OSError as exc:
            fail(E_COMMAND_INVALID, "edit command file cannot be read", path=str(path), error=str(exc))
    if isinstance(value, dict):
        value = value.get("commands")
    if not isinstance(value, list):
        fail(E_COMMAND_INVALID, "edit command stream must be a list or an object with commands")
    parsed: list[dict[str, Any]] = []
    for index, command in enumerate(value):
        if not isinstance(command, dict):
            fail(E_COMMAND_INVALID, "edit command must be an object", index=index)
        parsed.append(command)
    return parsed
