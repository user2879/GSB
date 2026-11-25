"""Subprocess lifecycle management for loopback shard workers."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .canonical import write_json
from .models import Project, ShardSpec


class ClusterStartError(RuntimeError):
    def __init__(self, message: str, *, shard_id: str, log_path: Path | None = None):
        super().__init__(message)
        self.shard_id = shard_id
        self.log_path = log_path


@dataclass
class RunningShard:
    shard: ShardSpec
    process: subprocess.Popen[bytes]
    log_path: Path
    ready_path: Path
    config_path: Path


class ShardCluster:
    def __init__(
        self,
        project: Project,
        workdir: Path,
        shard_ids: Iterable[str],
        *,
        startup_timeout: float = 10.0,
    ):
        self.project = project
        self.workdir = Path(workdir)
        self.shard_ids = tuple(sorted(set(shard_ids)))
        self.startup_timeout = startup_timeout
        self.processes: dict[str, RunningShard] = {}
        self.start_errors: dict[str, Exception] = {}
        self.project_root = Path(__file__).resolve().parent.parent

    def _worker_config(self, shard: ShardSpec) -> dict:
        try:
            profile = self.project.profile_for_revision(shard.revision)
        except KeyError:
            profile = self.project.profile_for_revision(self.project.current_revision)
        return {
            "shard_id": shard.shard_id,
            "port": shard.port,
            "term_bucket_capacity": shard.term_bucket_capacity,
            "concurrency_limit": shard.concurrency_limit,
            "seed": shard.seed,
            "revision": shard.revision,
            "compression": profile.to_dict(),
            "vocabulary": {
                "terms": list(self.project.vocabulary.terms),
                "stopwords": sorted(self.project.vocabulary.stopwords),
            },
        }

    def start(self, *, continue_on_error: bool = False) -> "ShardCluster":
        if self.processes:
            return self
        shard_map = self.project.shard_map()
        for shard_id in self.shard_ids:
            if shard_id not in shard_map:
                raise ClusterStartError(f"unknown shard {shard_id}", shard_id=shard_id)
        for shard_id in self.shard_ids:
            shard = shard_map[shard_id]
            service_dir = self.workdir / "services" / shard_id
            service_dir.mkdir(parents=True, exist_ok=True)
            config_path = service_dir / "worker.json"
            ready_path = service_dir / "ready.json"
            log_path = service_dir / "shard.log"
            write_json(config_path, self._worker_config(shard))
            if ready_path.exists():
                ready_path.unlink()
            log_file = log_path.open("ab")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(self.project_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
            command = [
                sys.executable,
                "-m",
                "gsb_index.shard_service",
                "--config",
                str(config_path),
                "--ready-file",
                str(ready_path),
            ]
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            except OSError as exc:
                log_file.close()
                error = ClusterStartError(f"unable to launch shard {shard_id}: {exc}", shard_id=shard_id, log_path=log_path)
                self.start_errors[shard_id] = error
                if not continue_on_error:
                    self.stop()
                    raise error
                continue
            log_file.close()
            running = RunningShard(shard, process, log_path, ready_path, config_path)
            self.processes[shard_id] = running
            deadline = time.monotonic() + self.startup_timeout
            error: ClusterStartError | None = None
            while time.monotonic() < deadline:
                if ready_path.exists():
                    break
                if process.poll() is not None:
                    error = ClusterStartError(
                        f"shard {shard_id} exited before readiness (rc={process.returncode}): {self._log_tail(log_path)}",
                        shard_id=shard_id,
                        log_path=log_path,
                    )
                    break
                time.sleep(0.02)
            else:
                error = ClusterStartError(
                    f"shard {shard_id} did not become ready within {self.startup_timeout}s: {self._log_tail(log_path)}",
                    shard_id=shard_id,
                    log_path=log_path,
                )
            if error is not None:
                self.start_errors[shard_id] = error
                if process.poll() is None:
                    process.terminate()
                self.processes.pop(shard_id, None)
                if not continue_on_error:
                    self.stop()
                    raise error
        return self

    @staticmethod
    def _log_tail(path: Path, limit: int = 2000) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-limit:]

    def stop(self, timeout: float = 3.0) -> None:
        for running in self.processes.values():
            if running.process.poll() is None:
                running.process.terminate()
        deadline = time.monotonic() + timeout
        for running in self.processes.values():
            if running.process.poll() is None:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    running.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    running.process.kill()
        for running in self.processes.values():
            try:
                running.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        self.processes.clear()

    def __enter__(self) -> "ShardCluster":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
