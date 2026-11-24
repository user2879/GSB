from __future__ import annotations

from typing import Any

from .receipts import WIDTHS


def apply_receipt_injection(text: str, injection: dict[str, Any]) -> str:
    lines = text.splitlines(keepends=True)
    line_no = int(injection["line_no"]) - 1
    if not (0 <= line_no < len(lines)):
        raise ValueError(f"injection {injection['id']} line_no is out of range")
    line = lines[line_no]
    newline = "\n" if line.endswith("\n") else ""
    body = line.rstrip("\r\n")
    kind = injection["kind"]
    if kind == "truncate_line":
        drop = int(injection.get("drop_suffix", 1))
        body = body[: max(0, len(body) - drop)]
    elif kind == "set_field":
        version = body[:2].lower()
        version = {"r1": "r1", "r2": "r2", "r3": "r3"}.get(body[:2], version)
        specs = WIDTHS[version]
        field_name = injection["field"]
        field_index = [name for name, _width, _kind in specs].index(field_name) + 1
        parts = body.split("|")
        width = specs[field_index - 1][1]
        value = str(injection["value"])
        parts[field_index] = value[:width].ljust(width)
        body = "|".join(parts)
    else:
        raise ValueError(f"unsupported receipt injection kind: {kind}")
    lines[line_no] = body + newline
    return "".join(lines)


def apply_request_injection(payload: dict[str, Any], injection: dict[str, Any]) -> dict[str, Any]:
    if injection["kind"] != "duplicate_request_id":
        return payload
    result = dict(payload)
    result["requests"] = list(payload.get("requests", []))
    result["requests"].append({
        "seq": injection["seq"],
        "request_id": injection["request_id"],
        "arrived_ms": injection["arrived_ms"],
        "batch_id": injection["batch_id"],
        "worker_id": injection["worker_id"],
    })
    return result


def selected_injections(config: dict[str, Any], ids: list[str] | None = None) -> list[dict[str, Any]]:
    injections = list(config.get("injections", []))
    if not ids:
        return injections
    wanted = set(ids)
    return [item for item in injections if item["id"] in wanted]
