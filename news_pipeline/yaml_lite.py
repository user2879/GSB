"""A tiny deterministic YAML subset loader.

The pipeline only needs nested mappings, lists of mappings and scalar values.
Implementing the subset locally keeps the project dependency-light while
guaranteeing identical parsing on every machine.  Supported syntax:

* ``key: value`` mappings and nested indented mappings
* ``- key: value`` list items (each item is a mapping)
* scalars: int, float, true/false/null, and quoted/plain strings
* blank lines and ``#`` comments
"""


class YamlSyntaxError(ValueError):
    pass


def _scalar(text):
    text = text.strip()
    if text == "":
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
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


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def _parse_block(lines, index, indent):
    if index >= len(lines):
        return None, index
    if lines[index].lstrip().startswith("- "):
        result = []
        while index < len(lines):
            line = lines[index]
            cur_indent = _indent(line)
            if cur_indent != indent or not line.lstrip().startswith("- "):
                break
            body = line.lstrip()[2:]
            if body.strip() == "":
                index += 1
                item, index = _parse_block(lines, index, indent + 2)
                result.append(item)
                continue
            if ":" not in body:
                raise YamlSyntaxError("list item must be a mapping: %r" % line)
            key, value = body.split(":", 1)
            item = {}
            if value.strip():
                item[key.strip()] = _scalar(value)
                index += 1
            else:
                index += 1
                child, index = _parse_block(lines, index, cur_indent + 2)
                item[key.strip()] = child
            while index < len(lines):
                follow = lines[index]
                follow_indent = _indent(follow)
                if follow_indent <= indent or follow.lstrip().startswith("- "):
                    break
                sub_key, sub_value = follow.strip().split(":", 1)
                if sub_value.strip():
                    item[sub_key.strip()] = _scalar(sub_value)
                    index += 1
                else:
                    index += 1
                    child, index = _parse_block(lines, index, follow_indent + 2)
                    item[sub_key.strip()] = child
            result.append(item)
        return result, index

    result = {}
    while index < len(lines):
        line = lines[index]
        cur_indent = _indent(line)
        if cur_indent != indent:
            break
        stripped = line.strip()
        if ":" not in stripped:
            raise YamlSyntaxError("expected mapping line: %r" % line)
        key, value = stripped.split(":", 1)
        if value.strip():
            result[key.strip()] = _scalar(value)
            index += 1
        else:
            index += 1
            child, index = _parse_block(lines, index, indent + 2)
            result[key.strip()] = child
    return result, index


def loads(text):
    cleaned = []
    for raw in text.splitlines():
        if "\t" in raw[:len(raw) - len(raw.lstrip())]:
            raise YamlSyntaxError("tabs are not allowed for indentation")
        stripped_raw = raw.strip()
        if stripped_raw.startswith("#"):
            continue
        line = raw.split(" #", 1)[0].rstrip()
        if line.strip() == "":
            continue
        cleaned.append(line)
    if not cleaned:
        return {}
    data, _ = _parse_block(cleaned, 0, _indent(cleaned[0]))
    return data


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())
