"""受控 YAML 子集解析器。

支持工作台样本所需的最小 YAML 方言：
- 缩进映射（key: value / 嵌套 key:）
- 块序列（"- " 前缀，可含映射项）
- 行内流序列（[1, 2, 3] / ["a", "b"]）
- 标量：整数、浮点、true/false、null、普通字符串、引号字符串
- # 注释与空行

不支持锚点、标签、多行字符串、流映射等高级特性；这是有意的受控子集，
样本与文档均只使用上述语法，从而无需第三方依赖即可复现实验。
"""

from __future__ import annotations

import re
from typing import Any

_FLOW_LIST_RE = re.compile(r"^\[(.*)\]$")


def _strip_comment(text: str) -> str:
    in_single = in_double = False
    for idx, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or text[idx - 1] in " \t":
                return text[:idx]
    return text


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token == "":
        return None
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    low = token.lower()
    if low in ("null", "~"):
        return None
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _split_flow(inner: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = in_double = False
    for ch in inner:
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
        elif ch in "[{" and not in_single and not in_double:
            depth += 1
            buf.append(ch)
        elif ch in "]}" and not in_single and not in_double:
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0 and not in_single and not in_double:
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf and "".join(buf).strip():
        items.append("".join(buf).strip())
    return items


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    flow = _FLOW_LIST_RE.match(raw)
    if flow:
        inner = flow.group(1).strip()
        if not inner:
            return []
        return [_parse_value(part) for part in _split_flow(inner)]
    return _parse_scalar(raw)


def _preprocess(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        cleaned = _strip_comment(raw_line).rstrip()
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append((indent, cleaned.strip()))
    return lines


def loads(text: str) -> Any:
    lines = _preprocess(text)
    if not lines:
        return {}
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError(
            f"yaml_lite: 无法解析的残余内容，第 {index + 1} 行附近: {lines[index]}"
        )
    return value


def load_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    is_sequence = lines[index][1].startswith("- ") or lines[index][1] == "-"
    if is_sequence:
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_sequence(
    lines: list[tuple[int, str]], index: int, indent: int
) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        cur_indent, content = lines[index]
        if cur_indent != indent or not (content.startswith("- ") or content == "-"):
            break
        after = content[1:].strip()
        if after == "":
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                child, index = _parse_block(lines, index, lines[index][0])
                result.append(child)
            else:
                result.append(None)
            continue
        if ":" in after and not _FLOW_LIST_RE.match(after):
            # "- key: value" 形式：把该项重写为同缩进的映射首行。
            virtual = [(indent + 2, after)]
            j = index + 1
            while j < len(lines) and lines[j][0] > indent:
                virtual.append(lines[j])
                j += 1
            child, consumed = _parse_mapping(virtual, 0, virtual[0][0])
            if consumed != len(virtual):
                raise ValueError("yaml_lite: 序列内映射解析失败")
            result.append(child)
            index = j
        else:
            result.append(_parse_value(after))
            index += 1
    return result, index


def _parse_mapping(
    lines: list[tuple[int, str]], index: int, indent: int
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        cur_indent, content = lines[index]
        if cur_indent != indent:
            if cur_indent < indent:
                break
            raise ValueError(
                f"yaml_lite: 缩进异常，第 {index + 1} 行: {content!r}"
            )
        if content.startswith("- ") or content == "-":
            break
        if ":" not in content:
            raise ValueError(f"yaml_lite: 期望映射条目，第 {index + 1} 行: {content!r}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            result[key] = _parse_value(raw_value)
            continue
        if index < len(lines) and lines[index][0] > indent:
            child, index = _parse_block(lines, index, lines[index][0])
            result[key] = child
        else:
            result[key] = None
    return result, index
