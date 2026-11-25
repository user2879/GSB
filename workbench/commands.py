from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .configuration import ALLOWED_FRACTIONAL_BITS, ALLOWED_QUANT_LEVELS, WorkbenchConfig, is_int, load_json
from .errors import WorkbenchError
from .hashing import canonical_hash
from .masks import Rect, parse_rect


@dataclass(frozen=True)
class RuntimeState:
    rectangles: dict[str, Rect]
    iterations: tuple[int, ...]
    guidance: tuple[float, ...]
    style_strength: tuple[float, ...]
    style_weights: tuple[tuple[float, float, float], ...]
    quant_levels: tuple[int, ...]
    fractional_bits: tuple[int, ...]

    @classmethod
    def initial(cls, config: WorkbenchConfig, rectangles: dict[str, Rect]) -> "RuntimeState":
        return cls(
            rectangles=dict(rectangles),
            iterations=config.iterations,
            guidance=config.guidance,
            style_strength=config.style_strength,
            style_weights=config.style_weights,
            quant_levels=config.quant_levels,
            fractional_bits=config.fractional_bits,
        )

    def copy(self) -> "RuntimeState":
        return RuntimeState(
            rectangles=dict(self.rectangles),
            iterations=self.iterations,
            guidance=self.guidance,
            style_strength=self.style_strength,
            style_weights=self.style_weights,
            quant_levels=self.quant_levels,
            fractional_bits=self.fractional_bits,
        )

    def fingerprint(self) -> str:
        return canonical_hash({
            "rectangles": [self.rectangles[key].to_json() for key in sorted(self.rectangles)],
            "iterations": list(self.iterations),
            "guidance": list(self.guidance),
            "style_strength": list(self.style_strength),
            "style_weights": [list(w) for w in self.style_weights],
            "quant_levels": list(self.quant_levels),
            "fractional_bits": list(self.fractional_bits),
        })


def _level(command: dict[str, Any], levels: int) -> int:
    value = command.get("level")
    if not is_int(value) or not (0 <= int(value) < levels):
        raise WorkbenchError("E_CONFIG_INVALID", "command level is out of range", {"level": value, "levels": levels})
    return int(value)


def _unit(command: dict[str, Any], field: str = "strength") -> float:
    value = command.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 <= float(value) <= 1):
        raise WorkbenchError("E_CONFIG_INVALID", f"command {field} must be in [0,1]", {"value": value})
    return float(value)


def validate_command_stream(
    path_or_data: str | Path | dict[str, Any],
    config: WorkbenchConfig,
    initial_rectangles: dict[str, Rect],
) -> list[dict[str, Any]]:
    if isinstance(path_or_data, (str, Path)):
        data = load_json(path_or_data, "E_CONFIG_INVALID")
    else:
        data = path_or_data
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise WorkbenchError("E_CONFIG_INVALID", "command stream must have schema_version=1")
    commands = data.get("commands")
    if not isinstance(commands, list):
        raise WorkbenchError("E_CONFIG_INVALID", "command stream must contain a commands list")

    simulated = dict(initial_rectangles)
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(commands):
        if not isinstance(raw, dict):
            raise WorkbenchError("E_CONFIG_INVALID", "each command must be an object", {"index": index})
        command = dict(raw)
        command.setdefault("id", f"cmd-{index + 1:03d}")
        op = command.get("op")
        try:
            if op == "upsert_rect":
                raw_rect = command.get("rectangle", command)
                rect = parse_rect(raw_rect, config.width, config.height)
                simulated[rect.id] = rect
                command["rectangle"] = rect.to_json()
            elif op in {"delete_rect", "remove_rect"}:
                rect_id = command.get("rectangle_id", command.get("id"))
                if not isinstance(rect_id, str) or not rect_id:
                    raise WorkbenchError("E_CONFIG_INVALID", "delete_rect requires rectangle_id")
                if rect_id not in simulated:
                    raise WorkbenchError("E_CONFIG_INVALID", "cannot delete unknown rectangle", {"rectangle_id": rect_id})
                del simulated[rect_id]
                command["op"] = "delete_rect"
                command["rectangle_id"] = rect_id
            elif op == "set_iterations":
                level = _level(command, config.levels)
                value = command.get("iterations")
                iterations = _positive_command_int(value, "iterations", maximum=200)
                command.update({"level": level, "iterations": iterations})
            elif op == "set_guidance":
                level = _level(command, config.levels)
                strength = _unit(command, "strength")
                command.update({"level": level, "strength": strength})
            elif op == "set_style_strength":
                level = _level(command, config.levels)
                strength = _unit(command, "strength")
                command.update({"level": level, "strength": strength})
            elif op == "set_style_weight":
                level = _level(command, config.levels)
                weights = command.get("weights")
                if not isinstance(weights, list) or len(weights) != 3:
                    raise WorkbenchError("E_CONFIG_INVALID", "weights must contain three numbers")
                checked = []
                for channel, value in enumerate(weights):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 <= float(value) <= 4):
                        raise WorkbenchError("E_CONFIG_INVALID", f"weight {channel} must be in [0,4]", {"value": value})
                    checked.append(float(value))
                if sum(checked) <= 0:
                    raise WorkbenchError("E_CONFIG_INVALID", "style weights must have positive sum")
                command.update({"level": level, "weights": checked})
            elif op == "set_quantization":
                level = _level(command, config.levels)
                levels = command.get("levels")
                if not is_int(levels) or levels not in ALLOWED_QUANT_LEVELS:
                    raise WorkbenchError("E_CONFIG_INVALID", "quantization levels are not in the allowed enum", {"value": levels})
                bits = command.get("fractional_bits")
                if bits is None:
                    bits = config.fractional_bits[level]
                if not is_int(bits) or bits not in ALLOWED_FRACTIONAL_BITS:
                    raise WorkbenchError("E_CONFIG_INVALID", "fractional bits are not in the allowed enum", {"value": bits})
                command.update({"level": level, "levels": int(levels), "fractional_bits": int(bits)})
            else:
                raise WorkbenchError("E_CONFIG_INVALID", "unknown command op", {"op": op, "index": index})
        except WorkbenchError as exc:
            exc.details["command_index"] = index
            exc.details["command_id"] = command.get("id")
            raise
        normalized.append(command)
    return normalized


def _positive_command_int(value: Any, name: str, maximum: int) -> int:
    if not is_int(value) or value <= 0 or value > maximum:
        raise WorkbenchError("E_CONFIG_INVALID", f"{name} must be a positive integer <= {maximum}", {"value": value})
    return int(value)


def apply_command(state: RuntimeState, command: dict[str, Any]) -> RuntimeState:
    op = command["op"]
    if op == "upsert_rect":
        rect = parse_rect(command["rectangle"], 10**9, 10**9)
        rectangles = dict(state.rectangles)
        rectangles[rect.id] = rect
        return replace(state, rectangles=rectangles)
    if op == "delete_rect":
        rectangles = dict(state.rectangles)
        rectangles.pop(command["rectangle_id"], None)
        return replace(state, rectangles=rectangles)

    level = int(command["level"])

    def update_tuple(values: tuple[Any, ...], value: Any) -> tuple[Any, ...]:
        items = list(values)
        items[level] = value
        return tuple(items)

    if op == "set_iterations":
        return replace(state, iterations=update_tuple(state.iterations, int(command["iterations"])))
    if op == "set_guidance":
        return replace(state, guidance=update_tuple(state.guidance, float(command["strength"])))
    if op == "set_style_strength":
        return replace(state, style_strength=update_tuple(state.style_strength, float(command["strength"])))
    if op == "set_style_weight":
        return replace(state, style_weights=update_tuple(state.style_weights, tuple(float(v) for v in command["weights"])))
    if op == "set_quantization":
        next_levels = update_tuple(state.quant_levels, int(command["levels"]))
        next_bits = update_tuple(state.fractional_bits, int(command["fractional_bits"]))
        return replace(state, quant_levels=next_levels, fractional_bits=next_bits)
    raise WorkbenchError("E_CONFIG_INVALID", "unsupported validated command", {"op": op})


def apply_commands(state: RuntimeState, commands: list[dict[str, Any]]) -> RuntimeState:
    for command in commands:
        state = apply_command(state, command)
    return state
