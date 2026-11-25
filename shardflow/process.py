"""Node child-process lifecycle management for Windows and POSIX."""

import json
import os
import subprocess
import sys
import tempfile

from .errors import PortError
from .protocol import NodeClient


class NodeProcess:
    def __init__(self, node, effective_config, handshake_timeout_ms, io_timeout_ms,
                 workdir=None):
        self.node = node
        self.effective_config = dict(effective_config)
        self.handshake_timeout_ms = handshake_timeout_ms
        self.io_timeout_ms = io_timeout_ms
        self.workdir = workdir or tempfile.gettempdir()
        self.config_path = None
        self.process = None
        self.killed = False

    def start(self):
        os.makedirs(self.workdir, exist_ok=True)
        self.config_path = os.path.join(
            self.workdir, "shardflow-node-%s.json" % self.node.id
        )
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "node": self.node.__dict__,
                    "effective_config": self.effective_config,
                },
                handle,
            )
        kwargs = {
            "cwd": os.getcwd(),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        else:
            kwargs["start_new_session"] = True
        self.process = subprocess.Popen(
            [sys.executable, "-m", "shardflow.node", self.config_path], **kwargs
        )
        client = NodeClient(
            self.node, self.handshake_timeout_ms, self.io_timeout_ms
        )
        try:
            return client.handshake()
        except PortError as exc:
            self.stop()
            raise PortError(
                "node %s did not accept connections: %s" % (self.node.id, exc),
                object_id=self.node.id,
                field="port",
            ) from exc

    def is_alive(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)
        self.killed = True

    def force_kill(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=3.0)
        self.killed = True
