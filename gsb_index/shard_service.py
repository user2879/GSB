"""Loopback TCP shard worker.

The process owns one port and one configured index revision.  Every accepted
connection carries exactly one JSON-line request; the response carries one
JSON-line receipt.  The concurrency semaphore is non-blocking: an extra request
is rejected immediately with ``E_SHARD_CONCURRENCY_EXCEEDED``.
"""

from __future__ import annotations

import argparse
import json
import signal
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json, read_json
from .codes import E_SHARD_CONCURRENCY_EXCEEDED, E_SHARD_PROTOCOL_ERROR
from .indexing import compute_shard_summary
from .models import CompressionProfile, Document, ShardSpec, Vocabulary
from .protocol import decode_message, encode_message


class ShardWorkerConfig:
    def __init__(self, raw: dict[str, Any]):
        self.shard = ShardSpec(
            shard_id=raw["shard_id"],
            port=int(raw["port"]),
            term_bucket_capacity=int(raw["term_bucket_capacity"]),
            concurrency_limit=int(raw["concurrency_limit"]),
            seed=int(raw["seed"]),
            revision=int(raw["revision"]),
        )
        profile_raw = raw["compression"]
        self.profile = CompressionProfile(
            bucket_count=int(profile_raw["bucket_count"]),
            bucket_weights=tuple(float(weight) for weight in profile_raw["bucket_weights"]),
            encoding=str(profile_raw["encoding"]),
        )
        self.vocabulary = Vocabulary(
            terms=tuple(sorted(raw["vocabulary"]["terms"])),
            stopwords=frozenset(raw["vocabulary"].get("stopwords", [])),
        )


def _parse_request_document(raw: Any, worker: ShardWorkerConfig) -> Document:
    if not isinstance(raw, dict):
        raise ValueError("document must be an object")
    doc_id = raw.get("doc_id")
    version = raw.get("version")
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("doc_id must be a non-empty string")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise ValueError("version must be a positive integer")
    fields_raw = raw.get("fields")
    flags_raw = raw.get("field_stopwords")
    if not isinstance(fields_raw, dict) or set(fields_raw) != {"title", "body"}:
        raise ValueError("fields must contain exactly title and body")
    if not isinstance(flags_raw, dict) or set(flags_raw) != {"title", "body"}:
        raise ValueError("field_stopwords must contain exactly title and body")
    if not all(isinstance(value, list) and all(isinstance(term, str) for term in value) for value in fields_raw.values()):
        raise ValueError("field terms must be string lists")
    if not all(isinstance(flags_raw[name], bool) for name in ("title", "body")):
        raise ValueError("field stopword markers must be booleans")
    unknown = sorted(
        (set(fields_raw["title"]) | set(fields_raw["body"])) - set(worker.vocabulary.terms)
    )
    if unknown:
        raise ValueError(f"terms outside controlled vocabulary: {unknown}")
    return Document(
        doc_id=doc_id,
        version=version,
        shard_id=worker.shard.shard_id,
        fields={name: tuple(fields_raw[name]) for name in ("title", "body")},
        field_stopwords={name: bool(flags_raw[name]) for name in ("title", "body")},
    )


class ShardRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: "ShardTCPServer" = self.server  # type: ignore[assignment]
        received = bytearray()
        self.request.settimeout(server.request_timeout_seconds)
        while not received.endswith(b"\n"):
            chunk = self.request.recv(65536)
            if not chunk:
                break
            received.extend(chunk)
            if len(received) > server.max_line_bytes:
                self.request.sendall(
                    encode_message(
                        {
                            "ok": False,
                            "code": E_SHARD_PROTOCOL_ERROR,
                            "message": "request exceeds line limit",
                            "shard_id": server.worker.shard.shard_id,
                        }
                    )
                )
                return
        if not received:
            return
        try:
            message = decode_message(bytes(received))
        except (ValueError, json.JSONDecodeError) as exc:
            self.request.sendall(
                encode_message(
                    {
                        "ok": False,
                        "code": E_SHARD_PROTOCOL_ERROR,
                        "message": f"invalid JSON line: {exc}",
                        "shard_id": server.worker.shard.shard_id,
                    }
                )
            )
            return

        acquired = server.slots.acquire(blocking=False)
        if not acquired:
            self.request.sendall(
                encode_message(
                    {
                        "ok": False,
                        "code": E_SHARD_CONCURRENCY_EXCEEDED,
                        "message": "shard concurrency limit is already occupied",
                        "shard_id": server.worker.shard.shard_id,
                        "concurrency_limit": server.worker.shard.concurrency_limit,
                        "served_revision": server.worker.shard.revision,
                        "request_id": message.get("request_id"),
                    }
                )
            )
            return

        try:
            response = server.process(message)
        except Exception as exc:  # pragma: no cover - final containment barrier
            response = {
                "ok": False,
                "code": E_SHARD_PROTOCOL_ERROR,
                "message": f"unhandled shard error: {exc}",
                "shard_id": server.worker.shard.shard_id,
                "served_revision": server.worker.shard.revision,
                "request_id": message.get("request_id"),
            }
        finally:
            server.slots.release()
        self.request.sendall(encode_message(response))


class ShardTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    max_line_bytes = 16 * 1024 * 1024
    request_timeout_seconds = 30.0

    def __init__(self, address: tuple[str, int], worker: ShardWorkerConfig):
        self.worker = worker
        self.slots = threading.BoundedSemaphore(worker.shard.concurrency_limit)
        super().__init__(address, ShardRequestHandler)

    def process(self, message: dict[str, Any]) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        request_id = message.get("request_id")
        if message.get("type") != "index_batch":
            return {
                "ok": False,
                "code": E_SHARD_PROTOCOL_ERROR,
                "message": "unsupported request type",
                "shard_id": self.worker.shard.shard_id,
                "served_revision": self.worker.shard.revision,
                "request_id": request_id,
                "elapsed_ms_monotonic": (time.monotonic_ns() - started_ns) // 1_000_000,
            }
        delay = message.get("debug_delay_ms", 0)
        if isinstance(delay, int) and delay > 0:
            time.sleep(delay / 1000.0)
        try:
            documents = [
                _parse_request_document(raw, self.worker)
                for raw in message.get("documents", [])
            ]
        except ValueError as exc:
            return {
                "ok": False,
                "code": E_SHARD_PROTOCOL_ERROR,
                "message": str(exc),
                "shard_id": self.worker.shard.shard_id,
                "served_revision": self.worker.shard.revision,
                "request_id": request_id,
                "elapsed_ms_monotonic": (time.monotonic_ns() - started_ns) // 1_000_000,
            }
        entry_limit = message.get("entry_limit")
        if not isinstance(entry_limit, int) or isinstance(entry_limit, bool) or entry_limit < 0:
            return {
                "ok": False,
                "code": E_SHARD_PROTOCOL_ERROR,
                "message": "entry_limit must be a non-negative integer",
                "shard_id": self.worker.shard.shard_id,
                "served_revision": self.worker.shard.revision,
                "request_id": request_id,
                "elapsed_ms_monotonic": (time.monotonic_ns() - started_ns) // 1_000_000,
            }
        try:
            summary = compute_shard_summary(
                self.worker.shard,
                self.worker.profile,
                self.worker.vocabulary,
                documents,
                self.worker.shard.revision,
                entry_limit=entry_limit,
            )
        except OverflowError as exc:
            required = getattr(exc, "required_entries", None)
            limit = getattr(exc, "entry_limit", None)
            code = "E_BUCKET_CAPACITY_EXCEEDED"
            if required is not None:
                code = "E_ENTRY_LIMIT_EXCEEDED"
            return {
                "ok": False,
                "code": code,
                "message": str(exc),
                "shard_id": self.worker.shard.shard_id,
                "served_revision": self.worker.shard.revision,
                "request_id": request_id,
                "required_entries": required,
                "entry_limit": limit,
                "compression": self.worker.profile.to_dict(),
                "elapsed_ms_monotonic": (time.monotonic_ns() - started_ns) // 1_000_000,
            }
        return {
            "ok": True,
            "type": "batch_result",
            "request_id": request_id,
            "shard_id": self.worker.shard.shard_id,
            "served_revision": self.worker.shard.revision,
            "compression": self.worker.profile.to_dict(),
            "elapsed_ms_monotonic": (time.monotonic_ns() - started_ns) // 1_000_000,
            "summary": summary,
        }


def serve(config_path: Path, ready_file: Path | None = None, stop_event: threading.Event | None = None) -> None:
    raw = read_json(config_path)
    worker = ShardWorkerConfig(raw)
    server = ShardTCPServer(("127.0.0.1", worker.shard.port), worker)
    if ready_file is not None:
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        ready_file.write_text(
            canonical_json({"ready": True, "port": worker.shard.port, "pid": __import__("os").getpid()}) + "\n",
            encoding="utf-8",
        )
    if stop_event is None:
        stop_event = threading.Event()

    def _stop(_signum: int, _frame: Any) -> None:
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    old_term = signal.signal(signal.SIGTERM, _stop)
    old_int = signal.signal(signal.SIGINT, _stop)
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one deterministic inverted-index shard")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)
    try:
        serve(args.config, args.ready_file)
    except Exception as exc:
        print(f"shard failed: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
