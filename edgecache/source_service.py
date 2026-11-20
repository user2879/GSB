"""Deterministic content-source process implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .errors import EdgeError
from .model import ContentSnapshot, sha256_hex
from .protocol import PROTOCOL_VERSION
from .spec import load_spec, parse_content_records


class ContentSource:
    def __init__(self, spec_path: str | Path) -> None:
        loaded = load_spec(spec_path)
        self.spec_path = str(Path(spec_path).resolve())
        self.content = loaded.content
        self.policy = loaded.policy
        self.reload_count = 0

    def hello_payload(self) -> dict[str, Any]:
        return {
            "service": "content-source",
            "protocol": PROTOCOL_VERSION,
            "pid": os.getpid(),
            "content_version": self.content.version,
            "content_digest": self.content.digest,
            "spec_digest": sha256_hex(self.content.digest.encode("ascii")),
            "paths": len(self.content.records),
        }

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        op = message.get("op")
        if op == "hello":
            return {"ok": True, **self.hello_payload()}
        if op == "snapshot":
            return self._snapshot_response()
        if op == "status":
            return {"ok": True, **self.hello_payload(), "reload_count": self.reload_count}
        if op == "reload":
            return self._reload(message)
        if op == "shutdown":
            return {"ok": True, "service": "content-source", "shutdown": True, "close_connection": True}
        raise EdgeError(
            code="E_PROTOCOL",
            message="unsupported content-source operation",
            details={"op": op},
        )

    def _snapshot_response(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "content-source",
            "content_version": self.content.version,
            "content_digest": self.content.digest,
            "content": self.content.to_declared_list(),
            "graph": self.content.graph_dict(),
            "topological_order": self.content.topological_order(),
        }

    def _reload(self, message: dict[str, Any]) -> dict[str, Any]:
        raw_content = message.get("content")
        if not isinstance(raw_content, list):
            raise EdgeError(code="E_CONTENT", message="reload requires a content array", field_path="content")
        candidate = parse_content_records(raw_content)
        expected_version = message.get("content_version")
        expected_digest = message.get("content_digest")
        if expected_digest is not None and candidate.digest != expected_digest:
            raise EdgeError(
                code="E_CONTENT",
                message="reloaded content digest does not match the orchestrator snapshot",
                field_path="content_digest",
                details={"expected": expected_digest, "actual": candidate.digest},
            )
        if expected_version is not None and candidate.version != expected_version:
            raise EdgeError(
                code="E_CONTENT",
                message="reloaded content version does not match the orchestrator snapshot",
                field_path="content_version",
                details={"expected": expected_version, "actual": candidate.version},
            )
        self.content = candidate
        self.reload_count += 1
        return {"ok": True, **self.hello_payload(), "reload_count": self.reload_count}


__all__ = ["ContentSource", "ContentSnapshot"]
