"""受限 YAML 子集解析器（无第三方依赖时的离线兜底）。

支持语法（工作台生成的配置只使用该子集）：
- 两空格缩进的嵌套映射 ``key: value``；
- 序列 ``- item``（item 可为标量或行内 ``[a, b]`` 列表）；
- 标量：整数、浮点、true/false/null、单/双引号字符串、纯字符串；
- 行内流式序列 ``[1, 2, 3]``；
- ``#`` 注释（引号外）与空行。

明确拒绝：Tab 缩进、锚点/别名、标签、流式映射、多行字符串等非子集语法，
避免静默解析歧义。
"""

from __future__ import annotations

from typing import Any


class MiniYamlError(ValueError):
    pass


try:  # pragma: no cover - 取决于运行环境
    import yaml as _yaml  # type: ignore

    def parse_yaml(text: str) -> dict[str, Any]:
        data = _yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise MiniYamlError("YAML 顶层必须是映射")
        return data

except ImportError:  # pragma: no cover

    def parse_yaml(text: str) -> dict[str, Any]:
        lines = []
        for raw in text.splitlines():
            if "\t" in raw[: len(raw) - len(raw.lstrip(" "))]:
                raise MiniYamlError("不允许 Tab 缩进")
            stripped = _strip_comment(raw).rstrip()
            if not stripped.strip():
                continue
            indent = len(stripped) - len(stripped.lstrip(" "))
            if indent % 2 != 0:
                raise MiniYamlError(f"缩进必须为 2 的倍数: {raw!r}")
            lines.append((indent, stripped.strip()))
        root, _ = _parse_block(lines, 0, 0)
        if not isinstance(root, dict):
            raise MiniYamlError("YAML 顶层必须是映射")
        return root


def _strip_comment(line: str) -> str:
    out = []
    quote = None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        return _parse_seq(lines, index, indent)
    return _parse_map(lines, index, indent)


def _parse_map(lines, index, indent):
    result: dict[str, Any] = {}
    while index < len(lines):
        cur_indent, content = lines[index]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise MiniYamlError(f"意外缩进: {content!r}")
        if content.startswith("- "):
            break
        if ":" not in content:
            raise MiniYamlError(f"映射项缺少 ':': {content!r}")
        key, _, raw_value = content.partition(":")
        key = _scalar(key.strip())
        if not isinstance(key, str):
            raise MiniYamlError(f"映射键必须是字符串: {content!r}")
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            result[key] = _scalar(raw_value)
        elif index < len(lines) and lines[index][0] > indent:
            child, index = _parse_block(lines, index, lines[index][0])
            result[key] = child
        else:
            result[key] = None
    return result, index


def _parse_seq(lines, index, indent):
    result: list[Any] = []
    while index < len(lines):
        cur_indent, content = lines[index]
        if cur_indent != indent or not content.startswith("- "):
            break
        item = content[2:].strip()
        index += 1
        if ":" in item and not item.startswith("["):
            # 序列项为映射：首键与 "-" 同行，其余键以更深缩进续写
            mapping: dict[str, Any] = {}
            key, _, raw_value = item.partition(":")
            key = _scalar(key.strip())
            raw_value = raw_value.strip()
            mapping[key] = _scalar(raw_value) if raw_value else None
            while index < len(lines) and lines[index][0] > indent:
                sub_indent, sub_content = lines[index]
                if sub_content.startswith("- "):
                    child, index = _parse_seq(lines, index, sub_indent)
                    mapping[list(mapping)[-1]] = child
                    continue
                sub_key, _, sub_value = sub_content.partition(":")
                sub_key = _scalar(sub_key.strip())
                sub_value = sub_value.strip()
                index += 1
                if sub_value:
                    mapping[sub_key] = _scalar(sub_value)
                elif index < len(lines) and lines[index][0] > sub_indent:
                    child, index = _parse_block(lines, index, lines[index][0])
                    mapping[sub_key] = child
                else:
                    mapping[sub_key] = None
            result.append(mapping)
        else:
            result.append(_scalar(item))
    return result, index


def _scalar(token: str) -> Any:
    token = token.strip()
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part) for part in _split_flow(inner)]
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    low = token.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~", "none"):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    if token.startswith(("&", "*", "!", "|", ">", "{")):
        raise MiniYamlError(f"迷你 YAML 不支持的语法: {token!r}")
    return token


def _split_flow(inner: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    quote = None
    buf: list[str] = []
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == "[":
            depth += 1
            buf.append(ch)
        elif ch == "]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts
