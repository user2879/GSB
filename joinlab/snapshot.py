from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import digest_json
from .errors import fail
from .logging_utils import atomic_write_text


SNAPSHOT_FORMAT = "joinlab-snapshot-v1"


def save_snapshot(path: str | Path, document: dict[str, Any]) -> None:
    payload = dict(document)
    payload["format"] = SNAPSHOT_FORMAT
    payload["conclusion_digest"] = digest_json(
        {
            key: payload.get(key)
            for key in (
                "params",
                "tables",
                "histograms",
                "samples",
                "estimate",
                "attribution",
                "truth",
                "validation",
            )
        }
    )
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def load_snapshot(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        fail("E_SNAP", "快照文件不存在", "snapshot", file=str(source))
    except json.JSONDecodeError as exc:
        fail("E_SNAP", "快照 JSON 无法解析", "snapshot", line=exc.lineno, column=exc.colno)
    except OSError as exc:
        fail("E_SNAP", "快照文件无法读取", "snapshot", reason=str(exc))
    if not isinstance(document, dict):
        fail("E_SNAP", "快照顶层必须是对象", "snapshot")
    if document.get("format") != SNAPSHOT_FORMAT:
        fail("E_SNAP", "快照格式不受支持", "snapshot.format", value=document.get("format"))
    if "params_version" not in document or document.get("params_version") is None:
        fail("E_SNAP", "快照缺少 params_version", "snapshot.params_version")
    version = document["params_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        fail("E_SNAP", "快照 params_version 无效", "snapshot.params_version", value=version)
    if not isinstance(document.get("params"), dict):
        fail("E_SNAP", "快照缺少 params", "snapshot.params")
    if not isinstance(document.get("tables"), dict):
        fail("E_SNAP", "快照缺少 tables", "snapshot.tables")
    return document


def first_difference(expected: Any, actual: Any, path: str = "$") -> dict[str, Any] | None:
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float)) and not isinstance(expected, bool)
        and isinstance(actual, (int, float)) and not isinstance(actual, bool)
    ):
        return {"path": path, "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return {
                "path": path,
                "expected_keys": sorted(expected),
                "actual_keys": sorted(actual),
            }
        for key in sorted(expected):
            difference = first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return {"path": path, "expected_length": len(expected), "actual_length": len(actual)}
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = first_difference(expected_item, actual_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return {"path": path, "expected": expected, "actual": actual}
    return None


def verify_replay(snapshot: dict[str, Any], recomputed: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    table_difference = first_difference(snapshot["tables"], recomputed["tables"], "$.tables")
    if table_difference:
        fail(
            "E_REPLAY",
            "当前表结构或数据摘要与快照不一致，无法复现",
            "snapshot.tables",
            **{"first_mismatch": table_difference},
        )
    checks.append({"name": "table_structure_and_data_digest", "status": "MATCH"})

    for section in ("histograms", "samples"):
        difference = first_difference(snapshot.get(section), recomputed.get(section), f"$.{section}")
        if difference:
            fail(
                "E_REPLAY",
                f"快照 {section} 摘要与重算结果不一致",
                f"snapshot.{section}",
                first_mismatch=difference,
            )
        checks.append({"name": section, "status": "MATCH"})

    for section in ("estimate", "attribution", "truth", "validation"):
        difference = first_difference(snapshot.get(section), recomputed.get(section), f"$.{section}")
        if difference:
            fail(
                "E_REPLAY",
                f"快照 {section} 结论与重算结果不一致",
                f"snapshot.{section}",
                first_mismatch=difference,
            )
        checks.append({"name": section, "status": "MATCH"})

    snapshot_digest = digest_json(
        {
            key: snapshot.get(key)
            for key in ("params", "tables", "histograms", "samples", "estimate", "attribution", "truth", "validation")
        }
    )
    if snapshot.get("conclusion_digest") != snapshot_digest:
        fail(
            "E_REPLAY",
            "快照 conclusion_digest 与快照内容不一致",
            "snapshot.conclusion_digest",
            first_mismatch={
                "path": "$.conclusion_digest",
                "expected": snapshot_digest,
                "actual": snapshot.get("conclusion_digest"),
            },
        )
    checks.append({"name": "conclusion_digest", "status": "MATCH"})

    return {
        "verified": True,
        "checks": checks,
        "snapshot_conclusion_digest": snapshot.get("conclusion_digest"),
        "recomputed_conclusion_digest": digest_json(
            {
                key: recomputed.get(key)
                for key in ("params", "tables", "histograms", "samples", "estimate", "attribution", "truth", "validation")
            }
        ),
    }
