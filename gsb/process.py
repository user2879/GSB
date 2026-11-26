"""Child process lifecycle and startup handshakes."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .model import SessionSpec, ShardSpec
from .protocol import TcpShardClient


@dataclass
class WorkerHandle:
    shard: ShardSpec
    process: subprocess.Popen[bytes]
    client: TcpShardClient
    spec_path: Path
    stderr_lines: list[str] = field(default_factory=list)
    _stderr_thread: threading.Thread | None = None

    @property
    def pid(self) -> int:
        return int(self.process.pid)


class WorkerPool:
    def __init__(
        self,
        session: SessionSpec,
        *,
        startup_timeout_s: float = 8.0,
        handshake_timeout_s: float = 3.0,
        connect_timeout_s: float = 2.0,
        read_timeout_s: float = 30.0,
        work_dir: str | Path | None = None,
    ) -> None:
        self.session = session
        self.startup_timeout_s = startup_timeout_s
        self.handshake_timeout_s = handshake_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self._temp_dir = Path(tempfile.mkdtemp(prefix="gsb-workers-", dir=work_dir))
        self._own_temp_dir = True
        self.workers: dict[str, WorkerHandle] = {}

    def start(self) -> "WorkerPool":
        try:
            for shard in self.session.shards:
                self._start_shard(shard)
        except Exception:
            self.stop(force=True)
            raise
        return self

    def _start_shard(self, shard: ShardSpec) -> None:
        spec_path = self._temp_dir / f"{shard.shard_id}.json"
        spec_path.write_text(json.dumps(shard.to_dict(), indent=2) + "\n", encoding="utf-8")
        project_root = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ)
        env["PYTHONPATH"] = project_root + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "gsb.worker", "--spec", str(spec_path)],
            cwd=project_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        ready_queue: queue.Queue[bytes | None] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout is not None
            while True:
                line = process.stdout.readline()
                if not line:
                    ready_queue.put(None)
                    return
                ready_queue.put(line)

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stdout_thread.start()
        try:
            first = ready_queue.get(timeout=self.startup_timeout_s)
        except queue.Empty as exc:
            process.kill()
            process.wait(timeout=2)
            self._close_process_pipes(process)
            raise DiagnosticError(
                "E_TIMEOUT",
                f"worker {shard.shard_id} did not report readiness in time",
                details={"shard_id": shard.shard_id, "startup_timeout_s": self.startup_timeout_s},
            ) from exc
        if first is None:
            stderr = b""
            if process.stderr is not None:
                try:
                    stderr = process.stderr.read(65536)
                except Exception:
                    pass
            self._close_process_pipes(process)
            raise DiagnosticError(
                "E_PORT",
                f"worker {shard.shard_id} exited before listening: {stderr.decode('utf-8', 'replace').strip()}",
                details={"shard_id": shard.shard_id},
            )
        try:
            ready = json.loads(first.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            process.kill()
            process.wait(timeout=2)
            self._close_process_pipes(process)
            raise DiagnosticError(
                "E_PROTOCOL", f"worker {shard.shard_id} emitted an invalid ready line"
            ) from exc
        if ready.get("event") == "error":
            process.wait(timeout=2)
            self._close_process_pipes(process)
            raise DiagnosticError(
                str(ready.get("code", "E_PORT")),
                str(ready.get("message", f"worker {shard.shard_id} failed to start")),
                field=ready.get("field"),
                line=ready.get("line"),
                details={"shard_id": shard.shard_id},
            )
        if ready.get("event") != "ready" or ready.get("shard_id") != shard.shard_id:
            process.kill()
            process.wait(timeout=2)
            self._close_process_pipes(process)
            raise DiagnosticError(
                "E_PROTOCOL", f"worker {shard.shard_id} emitted an unexpected readiness record"
            )
        client = TcpShardClient(
            shard.host,
            shard.port,
            connect_timeout_s=self.connect_timeout_s,
            handshake_timeout_s=self.handshake_timeout_s,
            read_timeout_s=self.read_timeout_s,
        )
        try:
            client.handshake(shard.shard_id)
        except Exception:
            process.kill()
            process.wait(timeout=2)
            self._close_process_pipes(process)
            raise
        handle = WorkerHandle(
            shard=shard,
            process=process,
            client=client,
            spec_path=spec_path,
        )

        def collect_stderr() -> None:
            assert process.stderr is not None
            for raw in process.stderr:
                handle.stderr_lines.append(raw.decode("utf-8", "replace").rstrip())
                if len(handle.stderr_lines) > 200:
                    del handle.stderr_lines[:100]

        handle._stderr_thread = threading.Thread(target=collect_stderr, daemon=True)
        handle._stderr_thread.start()
        self.workers[shard.shard_id] = handle

    def client(self, shard_id: str) -> TcpShardClient:
        return self.workers[shard_id].client

    def handle(self, shard_id: str) -> WorkerHandle:
        return self.workers[shard_id]

    def kill(self, shard_id: str) -> int:
        handle = self.workers[shard_id]
        if handle.process.poll() is None:
            handle.process.kill()
        return handle.process.wait(timeout=5)

    def restart(self, shard_id: str) -> WorkerHandle:
        handle = self.workers.pop(shard_id, None)
        if handle is not None:
            if handle.process.poll() is None:
                handle.process.kill()
                handle.process.wait(timeout=5)
            self._close_process_pipes(handle.process)
        shard = self.session.shard(shard_id)
        self._start_shard(shard)
        return self.workers[shard_id]

    def stop(self, *, force: bool = False) -> None:
        for shard_id, handle in list(self.workers.items()):
            if handle.process.poll() is not None:
                continue
            if not force:
                try:
                    handle.client._call(
                        {"type": "shutdown", "shard_id": shard_id},
                        ("ok", "state"),
                        timeout_s=1.0,
                    )
                except DiagnosticError:
                    pass
                try:
                    handle.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    handle.process.terminate()
            else:
                handle.process.terminate()
        for handle in list(self.workers.values()):
            if handle.process.poll() is None:
                try:
                    handle.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    handle.process.kill()
                    handle.process.wait(timeout=2)
            self._close_process_pipes(handle.process)
        self.workers.clear()
        if self._own_temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __enter__(self) -> "WorkerPool":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    @staticmethod
    def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
