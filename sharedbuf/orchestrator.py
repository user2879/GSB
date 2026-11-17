"""Node orchestration: port checks, Windows-compatible subprocess launch,
handshake with timeout and structured JSON logging.

Startup only changes launch parameters; samples and sources are never
modified by the orchestrator.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time

from .errors import (E_HANDSHAKE, E_NODE, E_PORT, E_SEQ, PlatformError)
from .protocol import decode, encode, validate_message


def make_logger(log_path):
    logger = logging.getLogger(f"sharedbuf.{id(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(logger, event, **fields):
    rec = {"event": event, "ts": time.time(), **fields}
    logger.info(json.dumps(rec, ensure_ascii=False))


class NodeConn:
    def __init__(self, proc, sock, fobj):
        self.proc = proc
        self.sock = sock
        self.fobj = fobj
        self.lock = threading.Lock()

    def send(self, msg):
        self.fobj.write(encode(msg).decode("utf-8"))
        self.fobj.flush()

    def recv(self, wl, timeout):
        self.sock.settimeout(timeout)
        try:
            line = self.fobj.readline()
        finally:
            self.sock.settimeout(None)
        if not line:
            raise PlatformError(E_NODE, reason="node closed connection")
        return decode(line, wl)


class Orchestrator:
    def __init__(self, platform, nodes_cfg, log_path="logs/platform.log",
                 handshake_timeout=5.0, op_timeout=5.0, corrupt_nodes=()):
        self.platform = platform
        self.nodes_cfg = nodes_cfg
        self.handshake_timeout = handshake_timeout
        self.op_timeout = op_timeout
        self.corrupt_nodes = set(corrupt_nodes)
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.logger = make_logger(log_path)
        self.conns = {}

    def _check_port(self, node_id, port):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            log_event(self.logger, "node_start", node=node_id, port=port,
                      result="error", code=E_PORT,
                      reason="loopback port occupied")
            raise PlatformError(E_PORT, node=node_id, port=port)
        finally:
            probe.close()

    def _spawn(self, node_id, ncfg):
        owned = [s for s, o in self.platform.owners.items() if o == node_id]
        sector_spec = ",".join(
            f"{s}={self.platform.sector_index[s]}" for s in sorted(owned))
        cmd = ncfg.get("cmd") or [sys.executable, "-m", "sharedbuf.node",
               "--node-id", node_id, "--port", str(ncfg["port"]),
               "--seed", str(self.platform.seed),
               "--sectors", sector_spec]
        if ncfg.get("cmd"):
            cmd = list(cmd)
        if node_id in self.corrupt_nodes:
            cmd.append("--corrupt")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        try:
            return subprocess.Popen(cmd, env=env,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except OSError as exc:
            log_event(self.logger, "node_start", node=node_id,
                      result="error", code=E_NODE, reason=str(exc))
            raise PlatformError(E_NODE, node=node_id, reason=str(exc))

    def _handshake(self, node_id, ncfg, proc):
        deadline = time.monotonic() + self.handshake_timeout
        sock = None
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                log_event(self.logger, "node_start", node=node_id,
                          result="error", code=E_NODE,
                          reason=f"process exited rc={proc.returncode}")
                raise PlatformError(E_NODE, node=node_id,
                                    reason="process exited during handshake",
                                    rc=proc.returncode)
            try:
                sock = socket.create_connection(("127.0.0.1", ncfg["port"]),
                                                timeout=0.5)
                break
            except OSError:
                time.sleep(0.05)
        if sock is None:
            if proc is not None:
                proc.kill()
            log_event(self.logger, "node_start", node=node_id,
                      result="error", code=E_HANDSHAKE,
                      reason="connect timeout")
            raise PlatformError(E_HANDSHAKE, node=node_id,
                                reason="connect timeout")
        fobj = sock.makefile("rw", encoding="utf-8", newline="\n")
        conn = NodeConn(proc, sock, fobj)
        try:
            conn.send({"type": "session", "wl": self.platform.wl,
                       "session": self.platform.session})
            hello = conn.recv(self.platform.wl,
                              max(deadline - time.monotonic(), 0.1))
            validate_message(hello, self.platform.wl)
            if hello["type"] != "hello" or hello["node_id"] != node_id:
                raise PlatformError(E_HANDSHAKE, node=node_id,
                                    reason="unexpected hello message")
        except (PlatformError, OSError) as exc:
            if proc is not None:
                proc.kill()
            log_event(self.logger, "node_start", node=node_id,
                      result="error", code=E_HANDSHAKE, reason=str(exc))
            if isinstance(exc, PlatformError) and exc.code == E_HANDSHAKE:
                raise
            raise PlatformError(E_HANDSHAKE, node=node_id,
                                reason=str(exc)) from exc
        log_event(self.logger, "node_start", node=node_id,
                  port=ncfg["port"], result="ok", session=self.platform.session,
                  wl=self.platform.wl)
        return conn

    def start(self):
        for node_id, ncfg in self.nodes_cfg.items():
            port = ncfg["port"]
            proc = None
            if ncfg.get("spawn", True):
                self._check_port(node_id, port)
                proc = self._spawn(node_id, ncfg)
            self.conns[node_id] = self._handshake(node_id, ncfg, proc)
        return list(self.conns)

    def node_runner(self):
        def run(platform, lease, req):
            conn = self.conns.get(lease.node_id)
            if conn is None:
                raise PlatformError(E_NODE, node=lease.node_id,
                                    reason="node not connected")
            msg = {"type": "op", "wl": platform.wl, "lease_id": lease.lease_id,
                   "node_id": lease.node_id, "op": req.op, "agg": req.agg,
                   "sectors": list(lease.sectors), "seq_lo": lease.seq_lo,
                   "seq_hi": lease.seq_hi, "delta": float(req.delta)}
            with conn.lock:
                conn.send(msg)
                resp = conn.recv(platform.wl, self.op_timeout)
            validate_message(resp, platform.wl)
            if resp["type"] == "error":
                raise PlatformError(resp["code"], node=lease.node_id,
                                    path=resp.get("path"))
            if resp["type"] != "op_result":
                raise PlatformError(E_NODE, node=lease.node_id,
                                    reason=f"unexpected reply {resp['type']}")
            if resp["status"] != "done":
                raise PlatformError(E_NODE, node=lease.node_id,
                                    reason=f"op status {resp['status']}")
            if resp["progress"] != lease.seq_hi:
                raise PlatformError(E_SEQ, node=lease.node_id,
                                    lease=lease.lease_id,
                                    progress=resp["progress"],
                                    seq_hi=lease.seq_hi)
            return resp["summary"]
        return run

    def shutdown(self):
        for node_id, conn in self.conns.items():
            try:
                with conn.lock:
                    conn.send({"type": "shutdown", "wl": self.platform.wl})
            except OSError:
                pass
            try:
                conn.fobj.close()
                conn.sock.close()
            except OSError:
                pass
            if conn.proc is not None:
                try:
                    conn.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    conn.proc.kill()
        self.conns.clear()
