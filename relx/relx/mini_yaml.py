"""受控 YAML 子集解析器。

支持：两空格缩进的嵌套映射、块序列（'- ' 前缀）、整数/浮点/布尔/字符串
标量、空行与 # 注释、单双引号字符串。足以解析本项目抽取配置，避免引入
第三方依赖。不支持锚点、流式语法与多行折叠。
"""

from .errors import E_CONFIG_INVALID, RelxError


def _strip_comment(line: str) -> str:
    in_single = in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(text: str):
    text = text.strip()
    if text == "":
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _tokenize(text: str):
    tokens = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw.rstrip())
        if not line.strip():
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise RelxError(E_CONFIG_INVALID, f"第 {lineno} 行禁止使用 Tab 缩进")
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise RelxError(E_CONFIG_INVALID, f"第 {lineno} 行缩进必须为两空格倍数")
        content = line.strip()
        if content.startswith("- "):
            tokens.append((indent, "-", _parse_scalar(content[2:])))
        elif content == "-":
            tokens.append((indent, "-", None))
        elif ":" in content:
            key, _, value = content.partition(":")
            tokens.append((indent, key.strip(), _parse_scalar(value)))
        else:
            raise RelxError(E_CONFIG_INVALID, f"第 {lineno} 行无法解析: {content}")
    return tokens


def _parse_block(tokens, index, indent):
    if index >= len(tokens):
        return None, index
    first_indent, first_kind, _ = tokens[index]
    if first_indent < indent:
        return None, index
    if first_kind == "-":
        result = []
        while index < len(tokens):
            cur_indent, kind, value = tokens[index]
            if cur_indent != indent or kind != "-":
                break
            index += 1
            if value is not None:
                result.append(value)
            else:
                child, index = _parse_block(tokens, index, indent + 2)
                result.append(child)
        return result, index
    result = {}
    while index < len(tokens):
        cur_indent, kind, value = tokens[index]
        if cur_indent != indent or kind == "-":
            break
        index += 1
        if index < len(tokens) and tokens[index][0] > cur_indent:
            child, index = _parse_block(tokens, index, tokens[index][0])
            result[kind] = child
        else:
            result[kind] = value
    return result, index


def loads(text: str) -> dict:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    if tokens[0][0] != 0:
        raise RelxError(E_CONFIG_INVALID, "顶层键必须从第 0 列开始")
    data, end = _parse_block(tokens, 0, 0)
    if end != len(tokens):
        raise RelxError(E_CONFIG_INVALID, "YAML 缩进结构解析未完成")
    if not isinstance(data, dict):
        raise RelxError(E_CONFIG_INVALID, "配置根节点必须为映射")
    return data


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())


def _dump_value(value, indent: int) -> list:
    pad = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_dump_value(val, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_scalar_text(val)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                nested = _dump_value(item, indent + 2)
                first = nested[0].lstrip(" ")
                lines.append(f"{pad}- {first}")
                lines.extend(nested[1:])
            else:
                lines.append(f"{pad}- {_scalar_text(item)}")
        return lines
    return [f"{pad}{_scalar_text(value)}"]


def _scalar_text(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    if text == "" or any(ch in text for ch in ":#"):
        return f'"{text}"'
    return text


def dumps(data: dict) -> str:
    return "\n".join(_dump_value(data, 0)) + "\n"
