"""Component node subprocess.

Binds a unique loopback port, speaks JSON-lines with the platform and
executes leased batch operations on its owned sectors with NumPy.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys

from . import buffer as buf
from .errors import E_FIELD, E_OWNER, PlatformError
from .protocol import decode, encode, validate_message


def parse_sectors(spec):
    out = {}
    for part in spec.split(","):
        sid, idx = part.rsplit("=", 1)
        out[sid] = int(idx)
    return out


class Node:
    def __init__(self, node_id, port, sector_map, seed, corrupt=False):
        self.node_id = node_id
        self.port = port
        self.corrupt = corrupt
        self.data = {sid: buf.initial_sector_data(seed, idx)
                     for sid, idx in sector_map.items()}
        self.versions = {sid: 0 for sid in sector_map}
        self.wl = None

    def send(self, fileobj, msg):
        fileobj.write(encode(msg).decode("utf-8"))
        fileobj.flush()

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.port))
        srv.listen(1)
        conn, _ = srv.accept()
        fobj = conn.makefile("rw", encoding="utf-8", newline="\n")
        line = fobj.readline()
        if not line:
            return 1
        try:
            hello = json.loads(line)
        except json.JSONDecodeError:
            return 1
        if hello.get("type") != "session":
            return 1
        self.wl = hello["wl"]
        self.send(fobj, {"type": "hello", "wl": self.wl, "node_id": self.node_id})
        while True:
            line = fobj.readline()
            if not line:
                break
            try:
                msg = decode(line, self.wl)
            except PlatformError as exc:
                self.send(fobj, {"type": "error", "wl": self.wl,
                                 "code": exc.code,
                                 "path": str(exc.detail.get("path", "$"))})
                continue
            mtype = msg["type"]
            if mtype == "shutdown":
                break
            if mtype == "op":
                self.handle_op(fobj, msg)
        conn.close()
        srv.close()
        return 0

    def handle_op(self, fobj, msg):
        sectors = msg["sectors"]
        missing = [s for s in sectors if s not in self.data]
        if missing:
            self.send(fobj, {"type": "error", "wl": self.wl, "code": E_OWNER,
                             "path": "$.sectors"})
            return
        count = msg["seq_hi"] - msg["seq_lo"] + 1
        pre_versions = {s: self.versions[s] for s in sectors}
        value, post = buf.run_batch([self.data[s] for s in sectors],
                                    msg["op"], msg["agg"], msg["delta"], count)
        if self.corrupt:
            value += 0.5
        if msg["op"] == "write":
            for idx, sid in enumerate(sectors):
                self.data[sid] = post[idx]
                self.versions[sid] += count
        summary = {"lease_id": msg["lease_id"], "sectors": list(sectors),
                   "seq_lo": msg["seq_lo"], "seq_hi": msg["seq_hi"],
                   "versions": pre_versions, "value": value}
        self.send(fobj, {"type": "op_result", "wl": self.wl,
                         "lease_id": msg["lease_id"], "node_id": self.node_id,
                         "status": "done", "summary": summary,
                         "progress": msg["seq_hi"]})


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sharedbuf.node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sectors", required=True,
                        help="comma list of sector_id=index")
    parser.add_argument("--corrupt", action="store_true")
    args = parser.parse_args(argv)
    node = Node(args.node_id, args.port, parse_sectors(args.sectors),
                args.seed, corrupt=args.corrupt)
    return node.serve()


if __name__ == "__main__":
    sys.exit(main())
