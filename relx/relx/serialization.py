"""确定性 JSON 序列化：键排序、固定分隔符、UTF-8 中文直出。"""

import json


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=(",", ": "))


def write_json(path: str, value):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(dumps(value))
        handle.write("\n")


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")))
            handle.write("\n")
