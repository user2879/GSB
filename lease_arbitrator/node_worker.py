from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .content import build_shard, shard_digest
from .protocol import monotonic_ns


class NodeState:
    def __init__(self, node_id: str, seed: int, artifact_dir: Path, shard_bytes: int, quota_version: int) -> None:
        self.node_id = node_id
        self.seed = seed
        self.artifact_dir = artifact_dir
        self.shard_bytes = shard_bytes
        self.quota_version = quota_version
        self.leases: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        artifact_dir.mkdir(parents=True, exist_ok=True)

    def shard_path(self, artifact_key: str, shard_index: int) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in artifact_key)
        digest_prefix = __import__("hashlib").sha256(artifact_key.encode("utf-8")).hexdigest()[:12]
        return self.artifact_dir / f"{safe[:48]}-{digest_prefix}.shard{shard_index}.bin"

    def make_shard(self, artifact_key: str, shard_index: int) -> dict[str, Any]:
        return build_shard(
            node_id=self.node_id,
            node_seed=self.seed,
            artifact_key=artifact_key,
            shard_index=shard_index,
            byte_count=self.shard_bytes,
        )


class LineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "NodeServer" = self.server  # type: ignore[assignment]
        for raw_line in self.rfile:
            started = monotonic_ns()
            try:
                request = json.loads(raw_line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                response = server.dispatch(request)
            except Exception as exc:  # defensive protocol boundary
                response = {
                    "ok": False,
                    "error": {"code": "E_FIELD", "message": str(exc)},
                    "elapsed_ms": max(0, (monotonic_ns() - started) // 1_000_000),
                }
            response.setdefault("node_id", server.state.node_id)
            response.setdefault("quota_version", server.state.quota_version)
            response.setdefault("elapsed_ms", max(0, (monotonic_ns() - started) // 1_000_000))
            self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()


class NodeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: NodeState) -> None:
        self.state = state
        super().__init__(address, LineHandler)

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action", "write")
        request_seq = request.get("request_seq")
        if action == "ping":
            return {"ok": True, "action": "ping", "request_seq": request_seq}
        if action == "shutdown":
            threading.Thread(target=self.shutdown, daemon=True).start()
            return {"ok": True, "action": "shutdown"}
        if action == "configure_quota":
            version = request.get("quota_version")
            if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
                return {"ok": False, "error": {"code": "E_FIELD", "field": "quota_version", "message": "quota_version must be positive integer"}}
            with self.state.lock:
                self.state.quota_version = int(version)
                self.state.leases.clear()
            return {"ok": True, "action": "configure_quota", "quota_version": version}
        if action == "rehydrate_lease":
            return self.rehydrate_lease(request)
        if action == "probe":
            missing = [name for name in ("artifact_key", "shard_index", "request_seq") if name not in request]
            if missing:
                return {"ok": False, "error": {"code": "E_FIELD", "field": missing[0], "message": f"missing {missing[0]}"}}
            shard = self.state.make_shard(str(request["artifact_key"]), int(request["shard_index"]))
            return {
                "ok": True,
                "action": "probe",
                "request_seq": request_seq,
                "digest": shard["digest"],
                "shard_bytes": shard["shard_bytes"],
                "source_version": shard["source_version"],
            }
        if action == "write":
            return self.write(request)
        if action == "read":
            return self.read(request)
        if action == "renew":
            return self.renew(request)
        if action == "release":
            return self.release(request)
        return {"ok": False, "error": {"code": "E_FIELD", "field": "action", "message": f"unknown action {action!r}"}}

    def _quota_error(self, request: dict[str, Any]) -> dict[str, Any] | None:
        version = request.get("quota_version")
        if version is not None and version != self.state.quota_version:
            return {
                "ok": False,
                "error": {
                    "code": "E_STALE_QUOTA",
                    "field": "quota_version",
                    "message": f"node quota version is {self.state.quota_version}, request used {version}",
                },
            }
        return None

    def write(self, request: dict[str, Any]) -> dict[str, Any]:
        missing = [
            name
            for name in ("artifact_key", "shard_index", "lease_ms", "request_seq", "quota_version")
            if name not in request
        ]
        if missing:
            return {"ok": False, "error": {"code": "E_FIELD", "field": missing[0], "message": f"missing {missing[0]}"}}
        quota_error = self._quota_error(request)
        if quota_error:
            return quota_error
        if request.get("fail_write") is True:
            return {"ok": False, "error": {"code": "E_WRITE", "field": "write", "message": "injected deterministic write failure"}}

        artifact_key = str(request["artifact_key"])
        shard_index = int(request["shard_index"])
        lease_ms = int(request["lease_ms"])
        shard = self.state.make_shard(artifact_key, shard_index)
        path = self.state.shard_path(artifact_key, shard_index)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(shard["content"])
        os.replace(temporary, path)

        lease_id = f"lease-{self.state.node_id}-{request['request_seq']}-{uuid.uuid4().hex[:8]}"
        now_ns = monotonic_ns()
        with self.state.lock:
            self.state.leases[lease_id] = {
                "lease_version": 1,
                "expires_ns": now_ns + lease_ms * 1_000_000,
                "artifact_key": artifact_key,
                "shard_index": shard_index,
            }
        return {
            "ok": True,
            "action": "write",
            "request_seq": request["request_seq"],
            "lease_id": lease_id,
            "lease_version": 1,
            "digest": shard["digest"],
            "shard_bytes": shard["shard_bytes"],
            "source_version": shard["source_version"],
        }

    def read(self, request: dict[str, Any]) -> dict[str, Any]:
        missing = [name for name in ("artifact_key", "shard_index", "request_seq") if name not in request]
        if missing:
            return {"ok": False, "error": {"code": "E_FIELD", "field": missing[0], "message": f"missing {missing[0]}"}}
        artifact_key = str(request["artifact_key"])
        shard_index = int(request["shard_index"])
        path = self.state.shard_path(artifact_key, shard_index)
        if not path.is_file():
            return {"ok": False, "error": {"code": "E_NOT_FOUND", "field": "artifact_file", "message": f"shard file does not exist: {path.name}"}}
        content = path.read_bytes()
        shard = build_shard(
            node_id=self.state.node_id,
            node_seed=self.state.seed,
            artifact_key=artifact_key,
            shard_index=shard_index,
            byte_count=len(content),
        )
        actual_digest = shard_digest(
            node_id=self.state.node_id,
            artifact_key=artifact_key,
            shard_index=shard_index,
            content=content,
        )
        if shard["digest"] != actual_digest:
            return {"ok": False, "error": {"code": "E_DIGEST", "field": "digest", "message": "stored shard digest mismatch"}}
        return {
            "ok": True,
            "action": "read",
            "request_seq": request["request_seq"],
            "digest": shard["digest"],
            "shard_bytes": len(content),
            "source_version": shard["source_version"],
        }

    def renew(self, request: dict[str, Any]) -> dict[str, Any]:
        missing = [name for name in ("lease_id", "lease_version", "lease_ms", "request_seq", "quota_version") if name not in request]
        if missing:
            return {"ok": False, "error": {"code": "E_FIELD", "field": missing[0], "message": f"missing {missing[0]}"}}
        quota_error = self._quota_error(request)
        if quota_error:
            return quota_error
        lease_id = str(request["lease_id"])
        expected_version = int(request["lease_version"])
        with self.state.lock:
            lease = self.state.leases.get(lease_id)
            if lease is None:
                return {"ok": False, "error": {"code": "E_LEASE", "field": "lease_id", "message": "lease is unknown to node"}}
            if lease["expires_ns"] < monotonic_ns():
                self.state.leases.pop(lease_id, None)
                return {"ok": False, "error": {"code": "E_LEASE", "field": "expiry", "message": "lease has really expired"}}
            if int(lease["lease_version"]) != expected_version:
                return {
                    "ok": False,
                    "error": {
                        "code": "E_LEASE",
                        "field": "lease_version",
                        "message": f"expected version {expected_version}, node has {lease['lease_version']}",
                    },
                    "current_version": int(lease["lease_version"]),
                }
            lease["lease_version"] = expected_version + 1
            lease["expires_ns"] = monotonic_ns() + int(request["lease_ms"]) * 1_000_000
            new_version = int(lease["lease_version"])
        return {"ok": True, "action": "renew", "lease_id": lease_id, "lease_version": new_version, "request_seq": request["request_seq"]}

    def rehydrate_lease(self, request: dict[str, Any]) -> dict[str, Any]:
        missing = [name for name in ("lease_id", "lease_version", "remaining_ms", "artifact_key", "shard_index") if name not in request]
        if missing:
            return {"ok": False, "error": {"code": "E_FIELD", "field": missing[0], "message": f"missing {missing[0]}"}}
        lease_id = str(request["lease_id"])
        remaining_ms = max(0, int(request["remaining_ms"]))
        with self.state.lock:
            self.state.leases[lease_id] = {
                "lease_version": int(request["lease_version"]),
                "expires_ns": monotonic_ns() + remaining_ms * 1_000_000,
                "artifact_key": str(request["artifact_key"]),
                "shard_index": int(request["shard_index"]),
            }
        return {"ok": True, "action": "rehydrate_lease", "lease_id": lease_id, "lease_version": int(request["lease_version"])}

    def release(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        if not lease_id:
            return {"ok": True, "action": "release", "existed": False}
        with self.state.lock:
            existed = self.state.leases.pop(str(lease_id), None) is not None
        return {"ok": True, "action": "release", "lease_id": lease_id, "existed": existed}


def start_parent_watchdog(parent_pid: int | None) -> None:
    def watch_stdin() -> None:
        try:
            while sys.stdin.buffer.read(1):
                pass
        except Exception:
            pass
        os._exit(0)

    def watch() -> None:
        if parent_pid and sys.platform != "win32":
            while True:
                try:
                    os.kill(parent_pid, 0)
                except OSError:
                    os._exit(0)
                time.sleep(0.25)
        # On Windows the stdin EOF watchdog is the portable parent-death path.
        while True:
            time.sleep(1.0)

    threading.Thread(target=watch_stdin, name="stdin-watchdog", daemon=True).start()
    if parent_pid:
        threading.Thread(target=watch, name="pid-watchdog", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lease arbitration cache node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--shard-bytes", type=int, default=64)
    parser.add_argument("--quota-version", type=int, default=1)
    parser.add_argument("--parent-pid", type=int, default=os.getppid())
    args = parser.parse_args(argv)

    start_parent_watchdog(args.parent_pid)
    state = NodeState(
        node_id=args.node_id,
        seed=args.seed,
        artifact_dir=Path(args.artifact_dir),
        shard_bytes=args.shard_bytes,
        quota_version=args.quota_version,
    )
    try:
        server = NodeServer((args.host, args.port), state)
    except OSError as exc:
        text = str(exc).lower()
        code = "E_PORT" if "address already in use" in text or "10048" in text else "E_HANDSHAKE"
        print(json.dumps({"ok": False, "code": code, "message": str(exc)}), file=sys.stderr, flush=True)
        return 2
    try:
        server.serve_forever(poll_interval=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
