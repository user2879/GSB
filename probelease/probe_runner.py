"""Deterministic UDP probe process.

One process serves one declared session port and multiplexes all probe ids for
that session.  Delay behavior is implemented as a real delayed send; the
orchestrator never fabricates a timeout.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .protocol import decode_json_datagram, make_response, validate_request


class ParentWatcher(threading.Thread):
    def __init__(self, parent_pid: int) -> None:
        super().__init__(daemon=True, name="probe-parent-watcher")
        self.parent_pid = parent_pid
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(0.10):
            if not _parent_is_alive(self.parent_pid):
                os._exit(0)

    def stop(self) -> None:
        self._stop.set()


def _parent_is_alive(parent_pid: int) -> bool:
    if os.name == "nt":
        synchronize = 0x00100000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:
            return False
        try:
            result = kernel32.WaitForSingleObject(handle, 0)
            return result == 0x00000102  # WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return os.getppid() == parent_pid


def _serve(spec: dict[str, Any], ready_path: Path, parent_pid: int) -> int:
    port = int(spec["port"])
    probes = {item["probe_id"]: item for item in spec["probes"]}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.settimeout(0.20)
        temporary = ready_path.with_suffix(ready_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"pid": os.getpid(), "port": port, "status": "ready"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(temporary, ready_path)
        watcher = ParentWatcher(parent_pid)
        watcher.start()
        while True:
            try:
                payload, address = sock.recvfrom(65_535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                request = decode_json_datagram(payload)
                validate_request(request)
            except (ValueError, KeyError, TypeError):
                continue
            probe = probes.get(request["probe_id"])
            if probe is None:
                continue
            behavior = probe.get("behavior", "normal")
            if behavior == "drop":
                continue
            response = make_response(
                request,
                seed=int(probe["seed"]),
                sequence_offset=int(probe.get("sequence_offset", 0)),
            )
            encoded = json.dumps(
                response, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if behavior == "delay":
                delay_seconds = int(probe["delay_ms"]) / 1000.0
                threading.Thread(
                    target=_delayed_send,
                    args=(sock, encoded, address, delay_seconds),
                    daemon=True,
                ).start()
            else:
                try:
                    sock.sendto(encoded, address)
                except OSError:
                    pass
    finally:
        sock.close()
        try:
            ready_path.unlink()
        except FileNotFoundError:
            pass
    return 0


def _delayed_send(
    sock: socket.socket, payload: bytes, address: tuple[str, int], delay_seconds: float
) -> None:
    time.sleep(delay_seconds)
    try:
        sock.sendto(payload, address)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="probelease deterministic UDP probe")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    return _serve(spec, Path(args.ready_file), args.parent_pid)


if __name__ == "__main__":
    raise SystemExit(main())

