from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .errors import Codes, Failure
from .validation import (
    load_json_file,
    load_yaml_file,
    validate_config,
    validate_dimensions,
    validate_motion_table,
    validate_noise_table,
    validate_png_bytes,
    validate_prompt_text,
)


IMAGE_FILES = {
    "condition_a": "samples/images/condition_a.png",
    "condition_b": "samples/images/condition_b.png",
    "condition_c": "samples/images/condition_c.png",
}


def load_bundle(root: str | Path) -> tuple[dict[str, Any], list[Failure]]:
    root = Path(root)
    failures: list[Failure] = []

    config_path = root / "samples/render_config.yaml"
    config = load_yaml_file(config_path, failures)
    if not isinstance(config, dict):
        failures.extend(validate_config({}, str(config_path)))
        return {"root": str(root), "config": config}, failures
    config_failures = validate_config(config, str(config_path))
    failures.extend(config_failures)
    if config_failures:
        return {"root": str(root), "config": config}, failures

    vocab_path = root / config.get("paths", {}).get("vocab", "samples/controlled_vocab.yaml")
    vocab = load_yaml_file(vocab_path, failures) or {}
    terms: list[str] = []
    if isinstance(vocab, dict):
        terms = list(vocab.get("entities", [])) + list(vocab.get("attributes", []))
    else:
        failures.append(Failure(Codes.CONFIG_INVALID, "controlled vocabulary must be a mapping",
                                path=str(vocab_path)))
    failures.extend(validate_dimensions(config, terms))

    prompts_path = root / config.get("paths", {}).get("prompts", "samples/prompt_table.yaml")
    prompt_table = load_yaml_file(prompts_path, failures) or {}
    frame_prompts = prompt_table.get("frame_prompts") if isinstance(prompt_table, dict) else None
    if not isinstance(frame_prompts, list) or len(frame_prompts) != config.get("frames"):
        failures.append(Failure(
            Codes.CONFIG_INVALID, "prompt table must provide one prompt per frame",
            path=str(prompts_path),
        ))
    else:
        for frame_index, text in enumerate(frame_prompts):
            failures.extend(validate_prompt_text(
                text, terms, path=str(prompts_path), frame=frame_index
            ))

    images: dict[str, bytes] = {}
    for image_id, relative_path in IMAGE_FILES.items():
        image_path = root / relative_path
        try:
            data = image_path.read_bytes()
            images[image_id] = data
            failures.extend(validate_png_bytes(data, config, str(image_path)))
        except FileNotFoundError:
            failures.append(Failure(Codes.FILE_NOT_FOUND, "condition PNG is missing",
                                    path=str(image_path)))

    motion_path = root / config.get("paths", {}).get("motion", "samples/motion_vectors.yaml")
    motion = load_yaml_file(motion_path, failures)
    if isinstance(motion, dict):
        failures.extend(validate_motion_table(motion, config, str(motion_path)))
    elif not any(f.path == str(motion_path) for f in failures):
        failures.append(Failure(Codes.MOTION_GRID, "motion table must be a mapping",
                                path=str(motion_path)))

    noise_path = root / config.get("noise", {}).get("table_path", "samples/noise_table.json")
    noise = load_json_file(noise_path, failures)
    if isinstance(noise, dict):
        failures.extend(validate_noise_table(noise, config, str(noise_path)))
    elif not any(f.path == str(noise_path) for f in failures):
        failures.append(Failure(Codes.NOISE_VERSION_MISSING,
                                "noise table must be a mapping with a declared version",
                                path=str(noise_path)))

    commands_path = root / config.get("paths", {}).get("commands", "samples/command_stream.yaml")
    commands = load_yaml_file(commands_path, failures)
    if commands is not None and (
        not isinstance(commands, dict) or not isinstance(commands.get("commands"), list)
    ):
        failures.append(Failure(Codes.COMMAND_INVALID,
                                "command stream must be a mapping with a commands list",
                                path=str(commands_path)))

    injections_path = root / config.get("paths", {}).get("injections", "samples/injections.yaml")
    injections = load_yaml_file(injections_path, failures)

    bundle = {
        "root": str(root),
        "config": config,
        "vocab": vocab,
        "terms": terms,
        "prompt_table": prompt_table,
        "frame_prompts": list(frame_prompts or []),
        "images": images,
        "motion": motion,
        "noise": noise,
        "commands": commands,
        "injections": injections,
    }
    if not failures and isinstance(commands, dict):
        probe_state = initial_state(bundle)
        for command in commands.get("commands", []):
            if not isinstance(command, dict):
                failures.append(Failure(
                    Codes.COMMAND_INVALID, "each command must be a mapping",
                    path=commands_path and str(commands_path),
                ))
                continue
            outcome = apply_command(probe_state, command)
            if not outcome.get("ok"):
                failure_dict = outcome["failure"]
                failures.append(Failure(
                    failure_dict["code"], failure_dict["message"],
                    path=str(commands_path), details=failure_dict.get("details"),
                ))
    return bundle, failures


def initial_state(bundle: dict[str, Any]) -> dict[str, Any]:
    config = bundle["config"]
    frames = int(config["frames"])
    image_ids = ["condition_a", "condition_a", "condition_b", "condition_b"]
    if len(image_ids) < frames:
        image_ids.extend(["condition_c"] * (frames - len(image_ids)))
    return {
        "config": copy.deepcopy(config),
        "terms": list(bundle["terms"]),
        "vocab": copy.deepcopy(bundle.get("vocab", {})),
        "prompts": list(bundle["frame_prompts"]),
        "image_refs": image_ids[:frames],
        "image_bytes": {key: value for key, value in bundle["images"].items()},
        "motion": copy.deepcopy(bundle["motion"]["vectors"]),
        "noise": copy.deepcopy(bundle["noise"]),
        "command_index": 0,
        "global_version": int(config.get("global_version", 1)),
    }


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def apply_command(state: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
    """Apply one command. Invalid commands return a Failure and leave state unchanged."""
    cmd = copy.deepcopy(command)
    op = cmd.get("op")
    config = state["config"]
    frames = int(config["frames"])

    def frame_ok() -> bool:
        frame = cmd.get("frame")
        return isinstance(frame, int) and not isinstance(frame, bool) and 0 <= frame < frames

    if op == "replace_prompt":
        if not frame_ok() or not isinstance(cmd.get("text"), str):
            return {"ok": False, "failure": Failure(
                Codes.COMMAND_INVALID, "replace_prompt requires valid frame and text",
                details={"command_id": cmd.get("id")},
            ).to_dict()}
        failures = validate_prompt_text(cmd["text"], state["terms"], frame=cmd["frame"])
        if failures:
            failures[0].details = {**(failures[0].details or {}), "command_id": cmd.get("id")}
            return {"ok": False, "failure": failures[0].to_dict()}
        state["prompts"][int(cmd["frame"])] = cmd["text"]

    elif op == "replace_image":
        if not frame_ok():
            return {"ok": False, "failure": Failure(
                Codes.COMMAND_INVALID, "replace_image requires a valid frame",
                details={"command_id": cmd.get("id")},
            ).to_dict()}
        image_id = cmd.get("image_id")
        if image_id not in state["image_bytes"]:
            return {"ok": False, "failure": Failure(
                Codes.FILE_NOT_FOUND, f"image condition {image_id!r} is not in local asset store",
                details={"command_id": cmd.get("id"), "image_id": image_id},
            ).to_dict()}
        state["image_refs"][int(cmd["frame"])] = image_id

    elif op == "delete_image":
        if not frame_ok():
            return {"ok": False, "failure": Failure(
                Codes.COMMAND_INVALID, "delete_image requires a valid frame",
                details={"command_id": cmd.get("id")},
            ).to_dict()}
        state["image_refs"][int(cmd["frame"])] = None

    elif op == "adjust_motion":
        frame, row, col = cmd.get("frame"), cmd.get("row"), cmd.get("col")
        component_name = cmd.get("component", "dx")
        component = 0 if component_name == "dx" else 1 if component_name == "dy" else None
        grid = config["motion_grid"]
        if not frame_ok() or component is None or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in (row, col, cmd.get("value"))
        ) or not (0 <= row < grid["rows"] and 0 <= col < grid["cols"]):
            return {"ok": False, "failure": Failure(
                Codes.COMMAND_INVALID, "adjust_motion coordinates/value are invalid",
                details={"command_id": cmd.get("id")},
            ).to_dict()}
        state["motion"][frame][row][col][component] = int(cmd["value"])

    elif op == "set_weights":
        weights = cmd.get("weights", {})
        if not isinstance(weights, dict) or set(weights) != {"text", "image", "motion"}:
            return {"ok": False, "failure": Failure(
                Codes.COMMAND_INVALID, "set_weights requires text/image/motion decimal weights",
                details={"command_id": cmd.get("id")},
            ).to_dict()}
        config["fusion"]["weights"] = {k: str(v) for k, v in weights.items()}

    elif op == "set_denoise_steps":
        steps = cmd.get("steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            return {"ok": False, "failure": Failure(
                Codes.POSITIVE_INT, "steps must be a positive integer",
                details={"command_id": cmd.get("id"), "actual": steps},
            ).to_dict()}
        if steps > int(config["max_noise_steps"]):
            return {"ok": False, "failure": Failure(
                Codes.COMMAND_INVALID, "steps exceeds deterministic noise table capacity",
                details={"command_id": cmd.get("id"), "max": config["max_noise_steps"]},
            ).to_dict()}
        config["denoise_steps"] = steps

    elif op == "set_tile_rows":
        rows = cmd.get("tile_rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            return {"ok": False, "failure": Failure(
                Codes.POSITIVE_INT, "tile_rows must be a positive integer",
                details={"command_id": cmd.get("id"), "actual": rows},
            ).to_dict()}
        if rows > int(config["height"]):
            return {"ok": False, "failure": Failure(
                Codes.TILE_ROW_RANGE, "tile_rows must not exceed frame height",
                details={"command_id": cmd.get("id"), "height": config["height"]},
            ).to_dict()}
        config["tile_rows"] = rows

    elif op == "set_guidance":
        try:
            from .fixedpoint import parse_decimal
            parse_decimal(cmd.get("guidance_scale"), "guidance_scale")
        except Exception:
            return {"ok": False, "failure": Failure(
                Codes.COMMAND_INVALID, "guidance_scale must be decimal",
                details={"command_id": cmd.get("id")},
            ).to_dict()}
        config["guidance_scale"] = str(cmd["guidance_scale"])

    elif op == "set_quant_profile":
        from .fixedpoint import PROFILE_FRAC_BITS
        profile = cmd.get("quant_profile")
        if profile not in PROFILE_FRAC_BITS:
            return {"ok": False, "failure": Failure(
                Codes.QUANT_ENUM, "unknown quant_profile",
                details={"command_id": cmd.get("id"), "allowed": sorted(PROFILE_FRAC_BITS)},
            ).to_dict()}
        config["fixed_point"]["quant_profile"] = profile
        config["fixed_point"]["frac_bits"] = PROFILE_FRAC_BITS[profile]

    else:
        return {"ok": False, "failure": Failure(
            Codes.COMMAND_INVALID, f"unknown command op {op!r}",
            details={"command_id": cmd.get("id")},
        ).to_dict()}

    state["command_index"] = int(state.get("command_index", 0)) + 1
    state["global_version"] = int(state.get("global_version", 1)) + 1
    return {"ok": True, "command": cmd}
