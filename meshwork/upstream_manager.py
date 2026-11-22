from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import MeshError
from .model import InstanceSpec, MeshSpec
from .protocol import check_port_free, send_ping


@dataclass
class UpstreamProcess:
    instance: InstanceSpec
    process: subprocess.Popen[str]


class UpstreamManager:
    """Starts one real OS subprocess per declared loopback instance."""

    def __init__(
        self,
        spec: MeshSpec,
        handshake_timeout_ms: int = 3000,
        startup_timeout_ms: int = 5000,
    ) -> None:
        self.spec = spec
        self.handshake_timeout_ms = handshake_timeout_ms
        self.startup_timeout_ms = startup_timeout_ms
        self.processes: list[UpstreamProcess] = []
        self._script = Path(__file__).resolve().parent.parent / "scripts" / "upstream_instance.py"

    def preflight(self) -> None:
        for service in self.spec.ordered_services():
            for instance in service.instances:
                try:
                    check_port_free(instance.host, instance.port)
                except MeshError as exc:
                    exc.details.update(
                        {
                            "service": service.id,
                            "instance": instance.id,
                        }
                    )
                    exc.service = service.id
                    raise

    def _popen(self, instance: InstanceSpec) -> subprocess.Popen[str]:
        behavior = instance.behavior.to_dict()
        command = [
            sys.executable,
            "-u",
            str(self._script),
            "--instance-id",
            instance.id,
            "--service-id",
            instance.service_id,
            "--host",
            instance.host,
            "--port",
            str(instance.port),
            "--seed",
            str(instance.seed),
            "--behavior-json",
            json.dumps(behavior, ensure_ascii=False, separators=(",", ":")),
        ]
        kwargs: dict[str, object] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "cwd": str(Path(__file__).resolve().parent.parent),
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        try:
            return subprocess.Popen(command, **kwargs)  # type: ignore[arg-type]
        except OSError as exc:
            raise MeshError(
                "E_HOST",
                f"cannot start upstream subprocess: {exc}",
                field="upstream.process",
                details={"instance_id": instance.id, "port": instance.port},
            ) from exc

    def _stderr_tail(self, process: subprocess.Popen[str]) -> str:
        if process.poll() is None:
            return ""
        try:
            output = process.stderr.read() if process.stderr is not None else ""
        except (OSError, ValueError):
            output = ""
        return (output or "").strip()

    def _wait_handshake(self, instance: InstanceSpec, process: subprocess.Popen[str]) -> None:
        deadline = time.monotonic() + self.startup_timeout_ms / 1000.0
        last_error: MeshError | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = self._stderr_tail(process)
                code = "E_PORT" if "E_PORT" in tail else "E_HOST"
                raise MeshError(
                    code,
                    f"upstream {instance.id!r} exited during startup (exit={process.returncode})",
                    field="upstream.startup",
                    details={
                        "instance_id": instance.id,
                        "port": instance.port,
                        "stderr": tail[-2000:],
                    },
                )
            try:
                send_ping(
                    instance.host,
                    instance.port,
                    instance.id,
                    self.handshake_timeout_ms,
                )
                return
            except MeshError as exc:
                last_error = exc
                if exc.code == "E_REPLY":
                    raise
                time.sleep(0.03)
        detail = last_error.message if last_error else "no handshake"
        raise MeshError(
            "E_HOST",
            f"handshake timed out for instance {instance.id!r}: {detail}",
            field="handshake.timeout_ms",
            details={"instance_id": instance.id, "port": instance.port},
        )

    def start(self) -> None:
        self.preflight()
        try:
            for service in self.spec.ordered_services():
                for instance in service.instances:
                    process = self._popen(instance)
                    self.processes.append(UpstreamProcess(instance, process))
                    self._wait_handshake(instance, process)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for item in reversed(self.processes):
            process = item.process
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except (OSError, ValueError):
                pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    pass
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass
        self.processes.clear()

    def assert_alive(self) -> None:
        for item in self.processes:
            code = item.process.poll()
            if code is not None:
                tail = self._stderr_tail(item.process)
                raise MeshError(
                    "E_HOST",
                    f"upstream {item.instance.id!r} is not running (exit={code})",
                    field="upstream.process",
                    details={"instance_id": item.instance.id, "stderr": tail[-2000:]},
                )

    def __enter__(self) -> "UpstreamManager":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()
