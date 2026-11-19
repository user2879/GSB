"""Content-addressed snapshots for stale-input/output detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from . import __version__
from .canonical import pretty_json_bytes, sha256_bytes, sha256_value
from .errors import E_SNAPSHOT_STALE, BusinessError


def build_snapshot(artifacts: Mapping[str, bytes], input_names: tuple[str, ...]) -> dict[str, Any]:
    records = [
        {"path": name, "sha256": sha256_bytes(artifacts[name]), "size": len(artifacts[name])}
        for name in sorted(artifacts)
    ]
    content = {
        "version": 1,
        "engine_version": __version__,
        "inputs": list(input_names),
        "artifact_count": len(records),
        "artifacts": records,
    }
    content["content_sha256"] = sha256_value(content)
    return content


def verify_snapshot_value(
    snapshot: Mapping[str, Any],
    read_artifact: Callable[[str], bytes],
) -> dict[str, Any]:
    if snapshot.get("version") != 1:
        raise BusinessError(
            E_SNAPSHOT_STALE,
            "snapshot version is stale or unknown",
            {"snapshot_version": snapshot.get("version")},
        )
    if snapshot.get("engine_version") != __version__:
        raise BusinessError(
            E_SNAPSHOT_STALE,
            "snapshot engine version does not match the running engine",
            {"snapshot_engine_version": snapshot.get("engine_version"), "engine_version": __version__},
        )
    content = dict(snapshot)
    recorded_content_hash = content.pop("content_sha256", None)
    if recorded_content_hash != sha256_value(content):
        raise BusinessError(E_SNAPSHOT_STALE, "snapshot content hash is invalid")
    records = snapshot.get("artifacts")
    if not isinstance(records, list):
        raise BusinessError(E_SNAPSHOT_STALE, "snapshot artifact list is invalid")
    checked: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise BusinessError(E_SNAPSHOT_STALE, "snapshot contains an invalid artifact record")
        name = record["path"]
        expected = record.get("sha256")
        try:
            data = read_artifact(name)
        except OSError as exc:
            raise BusinessError(
                E_SNAPSHOT_STALE,
                "snapshot artifact is missing",
                {"path": name, "error": str(exc)},
            ) from exc
        actual = sha256_bytes(data)
        if actual != expected:
            raise BusinessError(
                E_SNAPSHOT_STALE,
                "snapshot artifact content has changed",
                {"path": name, "expected_sha256": expected, "actual_sha256": actual},
            )
        checked.append({"path": name, "sha256": actual})
    return {"business_code": "OK", "checked_artifacts": checked, "artifact_count": len(checked)}


def verify_snapshot_file(path: str | Path) -> dict[str, Any]:
    snapshot_path = Path(path).resolve()
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessError(
            E_SNAPSHOT_STALE,
            "cannot read snapshot",
            {"path": str(snapshot_path), "error": str(exc)},
        ) from exc
    base = snapshot_path.parent

    def read_artifact(name: str) -> bytes:
        return (base / name).read_bytes()

    return verify_snapshot_value(snapshot, read_artifact)


def snapshot_bytes(snapshot: Mapping[str, Any]) -> bytes:
    return pretty_json_bytes(snapshot)

