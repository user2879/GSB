"""Deterministic local cluster simulator speaking newline-delimited JSON.

Run one process per cluster.  The process binds only to 127.0.0.1 and computes
its group aggregates from a fixed request seed; elapsed time is advisory and is
measured with ``time.monotonic_ns``.
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
from typing import Any

import numpy as np

from .errors import E_PORT, E_REPLY
from .util import canonical_json, digest_json, is_int, monotonic_ms, stable_int


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 2 * 1024 * 1024


def _error_frame(code: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "code": code, "message": message}
    if details:
        result["details"] = details
    return result


def _validate_dispatch(request: dict[str, Any]) -> tuple[str, int, list[str], dict[str, int], int, int, int]:
    required_strings = ("shard_id", "cluster_id", "task_id")
    for field in required_strings:
        if not isinstance(request.get(field), str) or not request[field]:
            raise ValueError(f"missing or invalid {field}")
    for field in ("revision_version", "seed", "sample_count", "group_count", "shard_index"):
        if not is_int(request.get(field)):
            raise ValueError(f"missing or invalid integer {field}")
    revision = int(request["revision_version"])
    seed = int(request["seed"])
    sample_count = int(request["sample_count"])
    group_count = int(request["group_count"])
    if sample_count < 1 or sample_count > 1_000_000:
        raise ValueError("sample_count must be in [1, 1000000]")
    if group_count < 1 or group_count > sample_count:
        raise ValueError("group_count must be in [1, sample_count]")

    raw_keys = request.get("config_keys")
    raw_values = request.get("config_values")
    if not isinstance(raw_keys, list) or not all(isinstance(key, str) and key for key in raw_keys):
        raise ValueError("missing or invalid config_keys")
    if len(set(raw_keys)) != len(raw_keys):
        raise ValueError("config_keys contains duplicates")
    if not isinstance(raw_values, dict):
        raise ValueError("missing or invalid config_values")
    values: dict[str, int] = {}
    for key in raw_keys:
        if key not in raw_values or not is_int(raw_values[key]):
            raise ValueError(f"missing or invalid integer config_values.{key}")
        values[key] = int(raw_values[key])
    return (
        request["shard_id"],
        revision,
        list(raw_keys),
        values,
        seed,
        sample_count,
        group_count,
    )


def aggregate_samples(
    *,
    cluster_id: str,
    shard_id: str,
    revision_version: int,
    config_keys: list[str],
    config_values: dict[str, int],
    seed: int,
    sample_count: int,
    group_count: int,
) -> dict[str, Any]:
    """Create samples and aggregate them with NumPy's vectorised primitives."""

    # The sample stream is fixed by cluster + logical shard.  Revision/config
    # changes influence the offset values, not the random sample positions.
    shard_seed = (int(seed) ^ stable_int(f"{cluster_id}|{shard_id}")) & (
        (1 << 63) - 1
    )
    rng = np.random.default_rng(shard_seed)
    groups = rng.integers(0, group_count, size=sample_count, dtype=np.int64)
    unique_groups = np.unique(groups)

    series: dict[str, np.ndarray] = {}
    for key in sorted(config_keys):
        offsets = np.int64(config_values[key])
        series[key] = rng.integers(
            -16, 17, size=sample_count, dtype=np.int64
        ) + offsets

    group_rows: list[dict[str, Any]] = []
    for raw_group in unique_groups:
        group = int(raw_group)
        mask = groups == raw_group
        key_rows: dict[str, Any] = {}
        for key in sorted(series):
            values = series[key][mask]
            mean = round(float(np.mean(values, dtype=np.float64)), 6)
            key_rows[key] = {
                "count": int(values.size),
                "sum": int(np.sum(values, dtype=np.int64)),
                "min": int(np.min(values)),
                "max": int(np.max(values)),
                "mean": mean,
            }
        group_rows.append({"group": group, "keys": key_rows})

    return {
        "algorithm": "numpy-group-v1",
        "sample_count": sample_count,
        "group_count": group_count,
        "groups": group_rows,
    }


class _SimulatorServer:
    def __init__(self, cluster_id: str, host: str, port: int) -> None:
        self.cluster_id = cluster_id
        self.host = host
        self.port = port
        self.stop_event = threading.Event()
        self.listener: socket.socket | None = None
        self.threads: list[threading.Thread] = []
        self.lock = threading.Lock()

    @staticmethod
    def _parent_is_alive(parent_pid: int) -> bool:
        if parent_pid <= 0:
            return True
        if os.name == "nt":
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            synchronize = 0x00100000
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(synchronize, False, parent_pid)
            if not handle:
                return False
            try:
                # WAIT_OBJECT_0 means the process handle was signalled (exited).
                return kernel32.WaitForSingleObject(handle, 0) != 0
            finally:
                kernel32.CloseHandle(handle)
        try:
            if os.getppid() != parent_pid:
                return False
            os.kill(parent_pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # EPERM means the PID still exists but is not signalable.
            return True

    def _watch_parent(self, parent_pid: int) -> None:
        while not self.stop_event.wait(0.2):
            if not self._parent_is_alive(parent_pid):
                self.stop_event.set()
                if self.listener is not None:
                    try:
                        self.listener.close()
                    except OSError:
                        pass
                os._exit(0)

    def _listen(self) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name == "nt":
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(0.25)
        try:
            listener.bind((self.host, self.port))
            listener.listen(64)
        except OSError as exc:
            listener.close()
            raise OSError(f"{E_PORT}: cannot bind {self.host}:{self.port}: {exc}") from exc
        return listener

    def serve_forever(self) -> None:
        self.listener = self._listen()
        while not self.stop_event.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._handle_connection,
                args=(connection,),
                name=f"sim-{self.cluster_id}",
                daemon=True,
            )
            with self.lock:
                self.threads = [item for item in self.threads if item.is_alive()]
                self.threads.append(thread)
            thread.start()

    def _handle_connection(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(30.0)
            with connection.makefile("rwb") as stream:
                while not self.stop_event.is_set():
                    line = stream.readline(MAX_LINE_BYTES + 1)
                    if not line:
                        return
                    if len(line) > MAX_LINE_BYTES:
                        response = _error_frame(E_REPLY, "request line too large")
                    else:
                        response = self._handle_line(line)
                    stream.write(canonical_json(response).encode("utf-8") + b"\n")
                    stream.flush()
                    if response.get("shutdown"):
                        self.stop_event.set()
                        if self.listener is not None:
                            try:
                                self.listener.close()
                            except OSError:
                                pass
                        return
        except (OSError, ValueError):
            return
        finally:
            try:
                connection.close()
            except OSError:
                pass

    def _handle_line(self, line: bytes) -> dict[str, Any]:
        try:
            request = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            return _error_frame(E_REPLY, f"invalid JSON request: {exc}")
        if not isinstance(request, dict):
            return _error_frame(E_REPLY, "request must be a JSON object")

        request_type = request.get("type")
        if request_type == "hello":
            return {
                "ok": True,
                "service": "deterministic-cluster-sim",
                "protocol": PROTOCOL_VERSION,
                "cluster_id": self.cluster_id,
                "pid": os.getpid(),
                "port": self.port,
            }
        if request_type == "shutdown":
            return {
                "ok": True,
                "service": "deterministic-cluster-sim",
                "shutdown": True,
                "pid": os.getpid(),
            }
        if request_type != "dispatch":
            return _error_frame(E_REPLY, "unknown request type", request_type=request_type)

        started = monotonic_ms()
        try:
            (
                shard_id,
                revision,
                config_keys,
                config_values,
                seed,
                sample_count,
                group_count,
            ) = _validate_dispatch(request)
            if request.get("cluster_id") != self.cluster_id:
                raise ValueError("cluster_id does not match this simulator")
            summary = aggregate_samples(
                cluster_id=self.cluster_id,
                shard_id=shard_id,
                revision_version=revision,
                config_keys=config_keys,
                config_values=config_values,
                seed=seed,
                sample_count=sample_count,
                group_count=group_count,
            )
            elapsed_ms = monotonic_ms() - started
            return {
                "ok": True,
                "cluster_id": self.cluster_id,
                "shard_id": shard_id,
                "revision_version": revision,
                "aggregate_summary": summary,
                "aggregate_digest": digest_json(summary),
                "elapsed_ms": int(elapsed_ms),
            }
        except ValueError as exc:
            return _error_frame(E_REPLY, str(exc))

    def close(self) -> None:
        self.stop_event.set()
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass


def run_server(cluster_id: str, host: str, port: int, parent_pid: int = 0) -> int:
    server = _SimulatorServer(cluster_id=cluster_id, host=host, port=port)
    if parent_pid:
        threading.Thread(
            target=server._watch_parent,
            args=(parent_pid,),
            name=f"sim-parent-watch-{cluster_id}",
            daemon=True,
        ).start()
    try:
        server.serve_forever()
    except OSError as exc:
        sys.stderr.write(canonical_json(_error_frame(E_PORT, str(exc))) + "\n")
        sys.stderr.flush()
        return 2
    finally:
        server.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1" or not (1 <= args.port <= 65535):
        sys.stderr.write(
            canonical_json(
                _error_frame(E_PORT, "simulator requires literal 127.0.0.1 and port 1..65535")
            )
            + "\n"
        )
        return 2
    return run_server(args.cluster_id, args.host, args.port, args.parent_pid)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
