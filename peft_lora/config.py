from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .canonical import sha256_object
from .errors import E_ADAPTER_CONFIG, BusinessError


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if character in ('"', "'"):
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if character == "#" and quote is None:
            return line[:index]
    return line


def _split_inline(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if character in ('"', "'"):
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if quote is not None:
            continue
        if character in "[{(":
            depth += 1
        elif character in "]})":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in ("null", "none", "~"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_parse_scalar(part) for part in _split_inline(inner)]
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        result: dict[str, Any] = {}
        if inner:
            for part in _split_inline(inner):
                key, separator, raw_value = part.partition(":")
                if not separator:
                    raise ValueError(f"invalid inline mapping item: {part}")
                result[str(_parse_scalar(key))] = _parse_scalar(raw_value)
        return result
    if text[0:1] in ('"', "'") and text[-1:] == text[0]:
        if text[0] == '"':
            return json.loads(text)
        return text[1:-1].replace("''", "'")
    try:
        if any(marker in text.lower() for marker in (".", "e")):
            return float(text)
        return int(text, 0)
    except ValueError:
        return text


class _YamlSubsetParser:
    def __init__(self, source: str) -> None:
        self.lines: list[tuple[int, str, int]] = []
        for line_number, raw_line in enumerate(source.splitlines(), start=1):
            without_comment = _strip_comment(raw_line).rstrip()
            if not without_comment.strip():
                continue
            if "\t" in without_comment[: len(without_comment) - len(without_comment.lstrip())]:
                raise ValueError(f"tabs are not supported for indentation (line {line_number})")
            indent = len(without_comment) - len(without_comment.lstrip(" "))
            self.lines.append((indent, without_comment.strip(), line_number))

    def parse(self) -> Any:
        if not self.lines:
            return None
        value, index = self._parse_block(0, self.lines[0][0])
        if index != len(self.lines):
            _, _, line_number = self.lines[index]
            raise ValueError(f"unexpected content at line {line_number}")
        return value

    def _parse_block(self, index: int, indent: int) -> tuple[Any, int]:
        if index >= len(self.lines):
            return None, index
        current_indent, content, _ = self.lines[index]
        if current_indent != indent:
            raise ValueError("inconsistent indentation")
        if content.startswith("- "):
            return self._parse_list(index, indent)
        return self._parse_mapping(index, indent)

    def _parse_list(self, index: int, indent: int) -> tuple[list[Any], int]:
        result: list[Any] = []
        while index < len(self.lines):
            current_indent, content, line_number = self.lines[index]
            if current_indent < indent:
                break
            if current_indent != indent or not content.startswith("- "):
                break
            item = content[2:].strip()
            index += 1
            if item == "":
                if index < len(self.lines) and self.lines[index][0] > indent:
                    value, index = self._parse_block(index, self.lines[index][0])
                else:
                    value = None
            else:
                value = _parse_scalar(item)
            result.append(value)
        return result, index

    def _parse_mapping(self, index: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(self.lines):
            current_indent, content, line_number = self.lines[index]
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ValueError(f"unexpected indentation at line {line_number}")
            if content.startswith("- "):
                break
            key, separator, raw_value = content.partition(":")
            if not separator or not key.strip():
                raise ValueError(f"invalid mapping at line {line_number}")
            key = str(_parse_scalar(key.strip()))
            raw_value = raw_value.strip()
            index += 1
            if raw_value:
                value = _parse_scalar(raw_value)
            elif index < len(self.lines) and self.lines[index][0] > indent:
                value, index = self._parse_block(index, self.lines[index][0])
            else:
                value = None
            result[key] = value
        return result, index


def parse_yaml_subset(source: str) -> Any:
    return _YamlSubsetParser(source).parse()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not (number > 0.0) or number == float("inf"):
        raise ValueError(f"{name} must be positive and finite")
    return number


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} contains an invalid value")
        result.append(item.strip())
    return tuple(sorted(set(result)))


@dataclass(frozen=True)
class PeftConfig:
    target_layer_whitelist: tuple[str, ...]
    rank_candidates: tuple[int, ...]
    scale_candidates: tuple[float, ...]
    learning_rate: float
    epochs: int
    batch_size: int
    quantization_scale: int
    quantization_tolerance_steps: float
    concurrency: int
    merge_weight_budget: int | None
    merge_compute_budget: int | None
    snapshot_commit_interval: int
    base_dir: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        return sha256_object(self.as_dict())

    @property
    def quantization_step(self) -> float:
        return 1.0 / float(self.quantization_scale)

    @property
    def allowed_restore_error(self) -> float:
        return self.quantization_tolerance_steps * self.quantization_step


def parse_config(raw: Any) -> PeftConfig:
    root = _mapping(raw, "configuration")
    adapter = _mapping(root.get("adapter"), "adapter")
    quantization = _mapping(root.get("quantization"), "quantization")
    merge_budget = _mapping(root.get("merge_budget"), "merge_budget")

    whitelist = root.get("target_layer_whitelist", root.get("target_layers"))
    rank_candidates = adapter.get("rank_candidates", root.get("rank_candidates"))
    scale_candidates = adapter.get("scale_candidates", root.get("scale_candidates"))

    if not isinstance(rank_candidates, list) or not rank_candidates:
        raise ValueError("adapter.rank_candidates must be a non-empty list")
    ranks: list[int] = []
    for rank in rank_candidates:
        ranks.append(_positive_int(rank, "adapter.rank"))
    if not isinstance(scale_candidates, list) or not scale_candidates:
        raise ValueError("adapter.scale_candidates must be a non-empty list")
    scales = [_positive_float(scale, "adapter.scale") for scale in scale_candidates]

    quantization_scale = quantization.get("scale", root.get("quantization_scale", 1_000_000))
    tolerance = quantization.get(
        "tolerance_steps", root.get("quantization_tolerance_steps", 1.0)
    )
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("quantization.tolerance_steps must be a number")
    tolerance_float = float(tolerance)
    if tolerance_float < 0.0 or tolerance_float == float("inf"):
        raise ValueError("quantization.tolerance_steps must be finite and non-negative")

    weight_budget = merge_budget.get("weight_elements", root.get("merge_weight_budget"))
    compute_budget = merge_budget.get("compute_units", root.get("merge_compute_budget"))

    return PeftConfig(
        target_layer_whitelist=_string_list(whitelist, "target_layer_whitelist"),
        rank_candidates=tuple(sorted(set(ranks))),
        scale_candidates=tuple(sorted(set(scales))),
        learning_rate=_positive_float(root.get("learning_rate"), "learning_rate"),
        epochs=_positive_int(root.get("epochs"), "epochs"),
        batch_size=_positive_int(root.get("batch_size", 2), "batch_size"),
        quantization_scale=_positive_int(quantization_scale, "quantization.scale"),
        quantization_tolerance_steps=tolerance_float,
        concurrency=_positive_int(root.get("concurrency"), "concurrency"),
        merge_weight_budget=(
            None
            if weight_budget is None
            else _positive_int(weight_budget, "merge_budget.weight_elements", allow_zero=True)
        ),
        merge_compute_budget=(
            None
            if compute_budget is None
            else _positive_int(compute_budget, "merge_budget.compute_units", allow_zero=True)
        ),
        snapshot_commit_interval=_positive_int(
            root.get("snapshot_commit_interval"), "snapshot_commit_interval"
        ),
        base_dir=str(root.get("base_dir", "base")),
    )


def load_config(path: str | Path) -> PeftConfig:
    try:
        source = Path(path).read_text(encoding="utf-8")
        raw = parse_yaml_subset(source)
        return parse_config(raw)
    except (OSError, ValueError, TypeError) as exc:
        raise BusinessError(E_ADAPTER_CONFIG, f"invalid PEFT configuration: {exc}") from exc
