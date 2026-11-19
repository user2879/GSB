from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from .content import ContentRepository
from .errors import PipelineError, e_spec
from .netproto import JsonLineServer, atomic_write_json, read_json_file
from .spec import SourceUpdateSpec, load_spec


def _update_from_payload(payload: dict[str, Any]) -> SourceUpdateSpec:
    at_ms = payload.get("at_ms")
    if isinstance(at_ms, bool) or not isinstance(at_ms, int) or at_ms < 0:
        raise e_spec("update at_ms must be a non-negative integer", "source_update", "at_ms")
    path = payload.get("path")
    if not isinstance(path, str) or not path:
        raise e_spec("update path must be a non-empty string", "source_update", "path")
    etag = payload.get("etag")
    if etag is not None and (not isinstance(etag, str) or not etag):
        raise e_spec("update etag must be a non-empty string", "source_update", "etag")
    body = None
    if "body_hex" in payload:
        try:
            body = bytes.fromhex(str(payload["body_hex"]))
        except ValueError as exc:
            raise e_spec("body_hex is invalid", "source_update", "body_hex") from exc
    elif "body" in payload:
        if not isinstance(payload["body"], str):
            raise e_spec("body must be a string", "source_update", "body")
        body = payload["body"].encode("utf-8")
    ttl = payload.get("ttl_seconds")
    if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0):
        raise e_spec("ttl_seconds must be a positive integer", "source_update", "ttl_seconds")
    dependencies = None
    if "derivation" in payload:
        derivation = payload["derivation"]
        if derivation is None:
            dependencies = ()
        else:
            if not isinstance(derivation, dict) or not isinstance(derivation.get("inputs"), list):
                raise e_spec("derivation requires inputs", "source_update", "derivation.inputs")
            deps = []
            for index, raw in enumerate(derivation["inputs"]):
                if not isinstance(raw, dict):
                    raise e_spec("derivation input must be an object", "source_update", f"derivation.inputs[{index}]")
                dep_path = raw.get("path")
                coefficient = raw.get("coefficient")
                if not isinstance(dep_path, str) or not dep_path:
                    raise e_spec("derivation input path is invalid", "source_update", "derivation.inputs.path")
                if isinstance(coefficient, bool) or not isinstance(coefficient, int) or coefficient <= 0:
                    raise e_spec(
                        "derivation coefficient must be a positive integer",
                        "source_update",
                        "derivation.inputs.coefficient",
                    )
                from .spec import Dependency

                deps.append(Dependency(dep_path, coefficient))
            dependencies = tuple(deps)
    if not any(key in payload for key in ("etag", "body", "body_hex", "ttl_seconds", "derivation")):
        raise e_spec("source update changes no fields", "source_update", "$")
    return SourceUpdateSpec(at_ms, path, etag, body, ttl, dependencies)


class SourceApp:
    def __init__(self, repository: ContentRepository, state_file: str | None = None) -> None:
        self.repository = repository
        self.state_file = state_file
        self.lock = threading.RLock()

    def _persist(self) -> None:
        if not self.state_file:
            return
        atomic_write_json(
            self.state_file,
            {
                "updates": [update.as_dict() for update in self.repository.applied_updates],
            },
        )

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        op = payload.get("op")
        if op == "ping":
            return {"ok": True, "service": "source", "pid": os.getpid()}
        if op == "shutdown":
            threading.Thread(target=lambda: self.shutdown_server(), daemon=True).start()
            return {"ok": True, "status": "stopping"}
        with self.lock:
            if op == "get":
                return self._get(payload)
            if op == "versions":
                return self._versions(payload)
            if op == "update":
                update = _update_from_payload(payload)
                signature = json.dumps(update.as_dict(), sort_keys=True, separators=(",", ":"))
                existing = {
                    json.dumps(item.as_dict(), sort_keys=True, separators=(",", ":"))
                    for item in self.repository.applied_updates
                }
                if signature in existing:
                    return {"ok": True, "applied": update.as_dict(), "idempotent": True}
                self.repository.apply_update(update)
                self._persist()
                return {"ok": True, "applied": update.as_dict(), "idempotent": False}
        raise PipelineError("E_PROTOCOL", "unsupported source operation", path="op", field="op", details={"op": op})

    def _get(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = payload.get("path")
        at_ms = payload.get("at_ms")
        if not isinstance(path, str) or not path:
            raise PipelineError("E_PROTOCOL", "get requires path", path="path", field="path")
        if isinstance(at_ms, bool) or not isinstance(at_ms, int) or at_ms < 0:
            raise PipelineError("E_PROTOCOL", "get requires non-negative at_ms", path="at_ms", field="at_ms")
        content = self.repository.get(path, at_ms)
        return {
            "ok": True,
            "service": "source",
            "path": path,
            "at_ms": at_ms,
            "etag": content.etag,
            "declared_etag": content.declared_etag,
            "body_b64": base64.b64encode(content.body).decode("ascii"),
            "body_sha256": content.body_sha256,
            "ttl_seconds": content.ttl_seconds,
            "derived": content.derived,
            "dependencies": [dep.as_dict() for dep in content.dependencies],
            "source": "origin_authoritative",
        }

    def _versions(self, payload: dict[str, Any]) -> dict[str, Any]:
        at_ms = payload.get("at_ms")
        if isinstance(at_ms, bool) or not isinstance(at_ms, int) or at_ms < 0:
            raise PipelineError("E_PROTOCOL", "versions requires non-negative at_ms", path="at_ms", field="at_ms")
        snapshot = self.repository.snapshot(at_ms)
        return {"ok": True, "service": "source", **snapshot}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback CDN origin simulator")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--error-file", required=True)
    parser.add_argument("--state-file")
    args = parser.parse_args(argv)

    def fail(error: PipelineError) -> int:
        try:
            atomic_write_json(args.error_file, error.to_dict())
        except OSError:
            pass
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return error.exit_code

    try:
        spec = load_spec(args.spec)
        if args.port < 1024 or args.port > 65535:
            raise PipelineError(
                "E_PORT", "source port must be in [1024, 65535]", path="source", field="port"
            )
        repository = ContentRepository(spec)
        if args.state_file and Path(args.state_file).exists():
            state = read_json_file(args.state_file)
            for raw_update in state.get("updates", []):
                repository.apply_update(_update_from_payload(raw_update))
        app = SourceApp(repository, args.state_file)
        try:
            server = JsonLineServer((args.host, args.port), app)
        except OSError as exc:
            raise PipelineError(
                "E_PORT",
                "source port is occupied or unavailable",
                path="source",
                field="port",
                details={"host": args.host, "port": args.port, "reason": str(exc)},
            ) from exc
        app.shutdown_server = server.shutdown
        atomic_write_json(
            args.ready_file,
            {
                "service": "source",
                "pid": os.getpid(),
                "host": args.host,
                "port": args.port,
                "content_digest_at_zero": repository.snapshot(0)["content_digest"],
            },
        )
        server.serve_forever(poll_interval=0.05)
        server.server_close()
        return 0
    except PipelineError as exc:
        return fail(exc)
    except Exception as exc:
        return fail(
            PipelineError(
                "E_PROTOCOL",
                "source worker failed",
                path="source",
                field="worker",
                details={"exception_type": type(exc).__name__, "reason": str(exc)},
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
