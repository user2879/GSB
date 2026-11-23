"""Deterministic loopback subscriber service.

The parent starts this module in a child process.  Every TCP connection is
greeted with one JSON ready line, then accepts one or more newline-delimited
batch requests and emits one confirmation line per request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
import time
from typing import Any, Iterable

from .canonical import digest_value, stable_serialize, wall_ms
from .logging_utils import StructuredLogger


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 16 * 1024 * 1024


def deterministic_processing_ms(
    *, subscriber_id: str, batch_id: str, event_ids: Iterable[int], budget_ms: int, seed: int
) -> int:
    payload = stable_serialize(
        {
            "subscriber_id": subscriber_id,
            "batch_id": batch_id,
            "event_ids": sorted(int(value) for value in event_ids),
            "seed": int(seed),
            "budget_ms": int(budget_ms),
        }
    )
    value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return int(value % max(1, int(budget_ms)))


def compute_ack(
    *,
    subscriber_id: str,
    batch_id: str,
    events: Iterable[dict[str, Any]],
    seed: int,
    budget_ms: int,
) -> dict[str, Any]:
    ordered = sorted(
        events,
        key=lambda item: (int(item.get("expected_order", 0)), int(item["event_id"])),
    )
    event_ids = sorted(int(item["event_id"]) for item in ordered)
    processed_order = [int(item["event_id"]) for item in ordered]
    processing_ms = deterministic_processing_ms(
        subscriber_id=subscriber_id,
        batch_id=batch_id,
        event_ids=event_ids,
        budget_ms=budget_ms,
        seed=seed,
    )
    summary = {
        "subscriber_id": subscriber_id,
        "batch_id": batch_id,
        "event_ids": event_ids,
        "processing_ms": processing_ms,
        "processed_order": processed_order,
        "protocol_version": PROTOCOL_VERSION,
    }
    summary["ack_digest"] = digest_value(summary)
    return summary


class SubscriberServer:
    def __init__(
        self,
        *,
        subscriber_id: str,
        host: str,
        port: int,
        processing_budget_ms: int,
        seed: int,
        logger: StructuredLogger,
        processing_delay_ms: int = 0,
        omit_ack_field: str | None = None,
    ) -> None:
        self.subscriber_id = subscriber_id
        self.host = host
        self.port = port
        self.processing_budget_ms = processing_budget_ms
        self.seed = seed
        self.logger = logger
        self.processing_delay_ms = max(0, int(processing_delay_ms))
        self.omit_ack_field = omit_ack_field
        self.stop_event = threading.Event()
        self._listener: socket.socket | None = None

    def serve_forever(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.settimeout(0.2)
        try:
            if os.name == "nt":
                exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
                if exclusive is not None:
                    listener.setsockopt(socket.SOL_SOCKET, exclusive, 1)
            else:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(128)
        except OSError as exc:
            listener.close()
            error = {
                "event": "worker_error",
                "code": "E_PORT",
                "message": f"cannot bind {self.host}:{self.port}: {exc}",
                "subscriber_id": self.subscriber_id,
                "port": self.port,
            }
            self.logger.error("E_PORT", error["message"], subscriber_id=self.subscriber_id, port=self.port)
            sys.stdout.write(stable_serialize(error) + "\n")
            sys.stdout.flush()
            raise

        self._listener = listener
        ready = {
            "event": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "subscriber_id": self.subscriber_id,
            "host": self.host,
            "port": self.port,
            "pid": os.getpid(),
            "at_wall_ms": wall_ms(),
        }
        sys.stdout.write(stable_serialize(ready) + "\n")
        sys.stdout.flush()
        self.logger.info(
            "SERVICE_READY",
            "subscriber service ready",
            subscriber_id=self.subscriber_id,
            port=self.port,
            pid=os.getpid(),
        )

        while not self.stop_event.is_set():
            try:
                connection, address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._handle_connection,
                args=(connection, address),
                name=f"worker-{self.subscriber_id}-{address[1]}",
                daemon=True,
            )
            thread.start()

        try:
            listener.close()
        except OSError:
            pass

    def stop(self) -> None:
        self.stop_event.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass

    def _handle_connection(self, connection: socket.socket, address: tuple[str, int]) -> None:
        connection.settimeout(30.0)
        try:
            greeting = {
                "event": "ready",
                "protocol_version": PROTOCOL_VERSION,
                "subscriber_id": self.subscriber_id,
                "port": self.port,
            }
            connection.sendall((stable_serialize(greeting) + "\n").encode("utf-8"))
            buffer = b""
            while not self.stop_event.is_set():
                chunk = connection.recv(65536)
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > MAX_LINE_BYTES:
                    self._send_error(connection, "E_ACK", "request line exceeds size limit")
                    break
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    if not raw_line.strip():
                        continue
                    response = self._process_request_line(raw_line)
                    connection.sendall((stable_serialize(response) + "\n").encode("utf-8"))
        except (OSError, ValueError) as exc:
            self.logger.error(
                "E_WORKER_SOCKET",
                f"worker connection failed: {exc}",
                subscriber_id=self.subscriber_id,
                port=self.port,
                remote_port=address[1],
            )
        finally:
            try:
                connection.close()
            except OSError:
                pass

    def _send_error(self, connection: socket.socket, code: str, message: str) -> None:
        response = {
            "event": "protocol_error",
            "code": code,
            "message": message,
            "subscriber_id": self.subscriber_id,
            "protocol_version": PROTOCOL_VERSION,
        }
        try:
            connection.sendall((stable_serialize(response) + "\n").encode("utf-8"))
        except OSError:
            pass

    def _process_request_line(self, raw_line: bytes) -> dict[str, Any]:
        try:
            request = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {
                "event": "protocol_error",
                "code": "E_ACK",
                "message": f"invalid request JSON: {exc}",
                "subscriber_id": self.subscriber_id,
            }
        if not isinstance(request, dict):
            return {
                "event": "protocol_error",
                "code": "E_ACK",
                "message": "request must be an object",
                "subscriber_id": self.subscriber_id,
            }
        if request.get("subscriber_id") != self.subscriber_id:
            return {
                "event": "protocol_error",
                "code": "E_ACK",
                "message": "request subscriber_id does not match service",
                "subscriber_id": self.subscriber_id,
            }
        batch_id = request.get("batch_id")
        events = request.get("events")
        if not isinstance(batch_id, str) or not isinstance(events, list):
            return {
                "event": "protocol_error",
                "code": "E_ACK",
                "message": "request requires batch_id and events",
                "subscriber_id": self.subscriber_id,
            }
        try:
            normalized_events = []
            for event in events:
                if not isinstance(event, dict):
                    raise ValueError("event must be an object")
                normalized_events.append(
                    {
                        "event_id": int(event["event_id"]),
                        "expected_order": int(event.get("expected_order", 0)),
                        "payload_digest": str(event.get("payload_digest", "")),
                    }
                )
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "event": "protocol_error",
                "code": "E_ACK",
                "message": f"invalid event payload: {exc}",
                "subscriber_id": self.subscriber_id,
            }

        if self.processing_delay_ms:
            time.sleep(self.processing_delay_ms / 1000.0)

        ack = compute_ack(
            subscriber_id=self.subscriber_id,
            batch_id=batch_id,
            events=normalized_events,
            seed=self.seed,
            budget_ms=self.processing_budget_ms,
        )
        if self.omit_ack_field and self.omit_ack_field in ack:
            del ack[self.omit_ack_field]
        self.logger.info(
            "ACK_EMITTED",
            "batch acknowledged",
            subscriber_id=self.subscriber_id,
            batch_id=batch_id,
            event_ids=ack.get("event_ids"),
            processing_ms=ack.get("processing_ms"),
        )
        return ack


def _watch_stdin(server: SubscriberServer) -> None:
    try:
        for line in sys.stdin:
            if line.strip() == "STOP":
                break
    except (OSError, ValueError):
        pass
    server.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one deterministic subscriber service")
    parser.add_argument("--subscriber-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--budget-ms", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--log-file")
    parser.add_argument("--processing-delay-ms", type=int, default=0)
    parser.add_argument(
        "--omit-ack-field",
        choices=["subscriber_id", "batch_id", "event_ids", "processing_ms"],
        help="test-only fault injection: omit a required acknowledgement field",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = StructuredLogger(args.log_file)
    server = SubscriberServer(
        subscriber_id=args.subscriber_id,
        host=args.host,
        port=args.port,
        processing_budget_ms=args.budget_ms,
        seed=args.seed,
        logger=logger,
        processing_delay_ms=args.processing_delay_ms,
        omit_ack_field=args.omit_ack_field,
    )
    stdin_thread = threading.Thread(
        target=_watch_stdin, args=(server,), name="parent-watchdog", daemon=True
    )
    stdin_thread.start()
    try:
        server.serve_forever()
    except OSError:
        return 2
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
