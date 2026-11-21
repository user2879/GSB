"""确定性 YAML 子集解析器（仅依赖标准库）。

支持：缩进映射、块序列（- 项）、标量（整数/浮点/布尔/null/字符串）、
行内流式列表 [a, b] 与映射 {k: v}、# 注释、引号字符串。
样本与配置均由本项目固定种子生成，语法严格受控。
"""
from __future__ import annotations


class YamlError(ValueError):
    pass


def _scalar(token: str):
    token = token.strip()
    if token == "" or token in ("null", "~", "Null", "NULL"):
        return None
    if token in ("true", "True", "TRUE"):
        return True
    if token in ("false", "False", "FALSE"):
        return False
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part) for part in _split_flow(inner)]
    if token.startswith("{") and token.endswith("}"):
        result = {}
        inner = token[1:-1].strip()
        if inner:
            for part in _split_flow(inner):
                if ":" not in part:
                    raise YamlError(f"非法流式映射项: {part}")
                key, val = part.split(":", 1)
                result[_scalar(key.strip())] = _scalar(val.strip())
        return result
    return token


def _split_flow(text: str) -> list[str]:
    parts, depth, buf = [], 0, []
    quote = None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "[{":
            depth += 1
            buf.append(ch)
        elif ch in "]}":
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


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def loads(text: str):
    raw_lines = []
    for line in text.splitlines():
        stripped = _strip_comment(line).rstrip()
        if stripped.strip():
            raw_lines.append(stripped)
    if not raw_lines:
        return {}
    idx = 0

    def parse_block(indent: int):
        nonlocal idx
        if idx >= len(raw_lines):
            return None
        first = raw_lines[idx]
        if _indent(first) < indent:
            return None
        if first.lstrip().startswith("- "):
            seq = []
            while idx < len(raw_lines):
                line = raw_lines[idx]
                if _indent(line) != indent or not line.lstrip().startswith("- "):
                    break
                content = line.lstrip()[2:].strip()
                if ":" in content and not content.startswith(("[", "{")):
                    # 序列项是映射：把 "- " 替换为同缩进空格后复用映射解析
                    raw_lines[idx] = " " * (indent + 2) + content
                    seq.append(parse_block(indent + 2))
                else:
                    seq.append(_scalar(content))
                    idx += 1
            return seq
        mapping = {}
        while idx < len(raw_lines):
            line = raw_lines[idx]
            cur_indent = _indent(line)
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise YamlError(f"缩进异常: {line}")
            body = line.strip()
            if body.startswith("- "):
                break
            if ":" not in body:
                raise YamlError(f"缺少键值分隔: {line}")
            key, val = body.split(":", 1)
            key = _scalar(key.strip())
            val = val.strip()
            idx += 1
            if val:
                mapping[key] = _scalar(val)
            else:
                if idx < len(raw_lines) and _indent(raw_lines[idx]) > indent:
                    mapping[key] = parse_block(_indent(raw_lines[idx]))
                else:
                    mapping[key] = None
        return mapping

    return parse_block(_indent(raw_lines[0]))


def _strip_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if ch in "\"'":
            if quote is None:
                quote = ch
            elif quote == ch:
                quote = None
        elif ch == "#" and quote is None and (i == 0 or line[i - 1] == " "):
            return line[:i]
    return line
