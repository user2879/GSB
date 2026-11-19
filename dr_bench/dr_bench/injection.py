"""Deterministic fault injection for scene, ratio and snapshot failures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import InjectError, SceneError, SnapshotError
from .loader import parse_scene

FAULT_CORRUPT_SCENE = "corrupt_scene"
FAULT_BAD_RATIO = "bad_inject_ratio"
FAULT_MISSING_SNAPSHOT_VERSION = "missing_snapshot_version"
FAULT_KINDS = (
    FAULT_CORRUPT_SCENE,
    FAULT_BAD_RATIO,
    FAULT_MISSING_SNAPSHOT_VERSION,
)


class StructuredLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, code: str, message: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": level,
            "code": code,
            "message": message,
            "fields": fields,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        return entry


def validate_ratio(ratio: Any) -> float:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise InjectError("injection ratio must be a number", "ratio", {"value": ratio})
    value = float(ratio)
    if value < 0.0 or value > 1.0:
        raise InjectError(
            "injection ratio out of allowed [0.0, 1.0] range",
            "ratio",
            {"value": value},
        )
    return value


def corrupt_scene_payload(scene_dict: dict[str, Any]) -> dict[str, Any]:
    """Introduce a deterministic validation error and prove all-or-nothing load."""

    corrupted = json.loads(json.dumps(scene_dict))
    if corrupted.get("observations"):
        corrupted["observations"][0]["result"] = "not_a_supported_result"
    else:
        corrupted.setdefault("observations", []).append(
            {
                "service_id": "ghost",
                "instance_id": "ghost-0",
                "round_index": 0,
                "observed_at_ms": 0,
                "result": OBS_GHOST,
            }
        )
    return corrupted


OBS_GHOST = "not_a_supported_result"


def load_with_corruption(corrupted: dict[str, Any]) -> None:
    try:
        parse_scene(corrupted, source="<injected>")
    except SceneError:
        raise
    raise SceneError("corruption injection did not trigger E_SCENE", "<injected>")


def strip_snapshot_version(snapshot_path: str | Path) -> Path:
    path = Path(snapshot_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("snapshot_format_version", None)
    payload.pop("ledger_format_version", None)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def exercise_fault(kind: str, context: dict[str, Any], logger: StructuredLogger) -> BenchResult:
    """Run one injected fault and always record a structured log line."""

    if kind not in FAULT_KINDS:
        raise InjectError(f"unknown fault kind {kind!r}", "kind", {"allowed": list(FAULT_KINDS)})
    try:
        if kind == FAULT_CORRUPT_SCENE:
            corrupted = corrupt_scene_payload(context["scene_dict"])
            load_with_corruption(corrupted)
        if kind == FAULT_BAD_RATIO:
            validate_ratio(context.get("ratio"))
        if kind == FAULT_MISSING_SNAPSHOT_VERSION:
            from .ledger import read_snapshot

            read_snapshot(context["snapshot_dir"], context["scene_id"])
    except (SceneError, InjectError, SnapshotError) as exc:
        logger.log("ERROR", exc.code, exc.message, path=getattr(exc, "path", ""), detail=exc.detail)
        raise


class BenchResult:
    def __init__(self, ok: bool, error: dict[str, Any] | None = None, payload: Any = None):
        self.ok = ok
        self.error = error
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "error": self.error, "payload": self.payload}
