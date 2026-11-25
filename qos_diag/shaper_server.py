from __future__ import annotations

import argparse
import json
import os
import socketserver
import time
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, sha256_hex
from .engine import StageState, evaluate_stage_transition
from .fixedpoint import FixedPoint
from .model import HOST_LITERAL, load_bundle


class ShaperHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "ShaperTCPServer" = self.server  # type: ignore[assignment]
        for raw_line in self.rfile:
            try:
                payload = json.loads(raw_line.decode("utf-8"))
                response = server.handle_payload(payload)
            except Exception as exc:  # return a real socket response instead of dying
                response = {"ok": False, "error": "E_SHAPER", "message": str(exc)}
            self.wfile.write((json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
            self.wfile.flush()


class ShaperTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[ShaperHandler], shaper: dict[str, Any], delay_ms: int):
        super().__init__(address, handler)
        self.shaper = shaper
        self.delay_ms = delay_ms

    def handle_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("type") == "ping":
            return {"ok": True, "type": "pong", "shaper_id": self.shaper["shaper_id"], "pid": os.getpid()}
        if payload.get("type") != "judge":
            return {"ok": False, "error": "E_SHAPER", "message": "unsupported request type"}
        if self.delay_ms:
            time.sleep(max(0, self.delay_ms) / 1000.0)
        scale = int(payload.get("scale", 2))
        fp = FixedPoint(scale)
        event = payload["event"]
        previous = payload["previous_state"]
        state = StageState(int(previous["balance_q"]), int(previous["state_clock_ms"]))
        record, _ = evaluate_stage_transition(
            self.shaper,
            int(event["arrival_ms"]),
            int(event["packet_length"]),
            state,
            fp,
            window_start_ms=payload.get("window_start_ms"),
        )
        return {
            "ok": True,
            "type": "receipt",
            "request_id": payload.get("request_id"),
            "shaper_id": self.shaper["shaper_id"],
            "pid": os.getpid(),
            "result_sha256": sha256_hex(record),
            "record_sha256": sha256_hex(record),
            "result": record,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one QoS token-bucket shaper subprocess.")
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--shaper-id", required=True)
    parser.add_argument("--ready-file")
    parser.add_argument("--delay-ms", type=int, default=None)
    args = parser.parse_args(argv)

    bundle = load_bundle(args.spec_dir)
    shaper = next((item for item in bundle["shapers"] if item["shaper_id"] == args.shaper_id), None)
    if shaper is None:
        raise SystemExit(f"unknown shaper_id: {args.shaper_id}")
    if shaper["host"] != HOST_LITERAL:
        raise SystemExit("shaper host must be 127.0.0.1")
    delay_ms = int(args.delay_ms if args.delay_ms is not None else shaper.get("service_delay_ms", 0))
    with ShaperTCPServer((shaper["host"], int(shaper["port"])), ShaperHandler, shaper, delay_ms) as server:
        if args.ready_file:
            atomic_write_json(
                args.ready_file,
                {"ok": True, "pid": os.getpid(), "shaper_id": shaper["shaper_id"], "port": shaper["port"]},
            )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
