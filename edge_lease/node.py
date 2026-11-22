"""Simulated edge-cache node running as an independent local subprocess.

The node speaks one JSON object per line over loopback TCP::

    platform -> {"type":"handshake","node_id":...,"quant_version":...}
    node     -> {"type":"hello","node_id":...,"quant_version":...,"status":"ready"}
    platform -> {"type":"dispatch", ... full path table + target path ...}
    node     -> {"type":"receipt", node_id, version, path, derived_digest,
                 shard_hit_digest, cost_ms, quant_version}

Fail modes are explicit so tests can exercise E_PORT / E_HANDSHAKE /
E_REPLY / timeout / dead-process paths without touching any real network.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from typing import Any, Dict, List, Tuple

from . import fixedpoint as fp
from .content import shard_hash, shard_hit_digest, shard_intervals
from .manifest import topological_order
from .models import PathEntry


def _send(handle: socket.socket, obj: Dict[str, Any]) -> None:
    handle.sendall((json.dumps(obj, sort_keys=True) + "\n").encode("utf-8"))


def _recv_line(handle: socket.socket) -> Dict[str, Any]:
    buffer = bytearray()
    while True:
        chunk = handle.recv(4096)
        if not chunk:
            raise ConnectionError("peer closed connection mid-frame")
        buffer.extend(chunk)
        if b"\n" in buffer:
            line, _rest = buffer.split(b"\n", 1)
            return json.loads(line.decode("utf-8"))


def _entries_from_table(path_table: List[Dict[str, Any]]) -> Dict[str, PathEntry]:
    entries: Dict[str, PathEntry] = {}
    for doc in path_table:
        derived = None
        if doc.get("derived"):
            sources = tuple(
                (pair[0], pair[1]) for pair in doc["derived"]["sources"]
            )
            from .models import DerivedSpec

            derived = DerivedSpec(sources=sources)
        entries[doc["path"]] = PathEntry(
            path=doc["path"],
            length=int(doc["length"]),
            weight=doc["weight"],
            shards=tuple(dict(s) for s in doc["shards"]),
            targets=tuple(doc["targets"]),
            strategy=doc["strategy"],
            priority=int(doc["priority"]),
            derived=derived,
        )
    return entries


def compute_digests(
    path_table: List[Dict[str, Any]],
    target_path: str,
    content_seed: int,
    scale: int,
) -> Tuple[int, int, bool]:
    """Recompute target path derived digest and shard hit digest (fixed-point)."""
    entries = _entries_from_table(path_table)
    order = topological_order(entries)
    digests: Dict[str, int] = {}
    for name in order:
        entry = entries[name]
        weight_q, _single = fp.quantize_scalar(entry.weight, scale)
        if entry.derived is None:
            from .content import content_hash

            chash = content_hash(content_seed, name, entry.length)
            digest, _sat = fp.leaf_digest(chash, weight_q)
        else:
            terms = []
            for source, coefficient in entry.derived.sources:
                coeff_q, _single = fp.quantize_scalar(coefficient, scale)
                terms.append((coeff_q, digests[source]))
            digest, _sat, _raw = fp.derived_digest(terms, scale)
        digests[name] = digest
    target = entries[target_path]
    intervals = shard_intervals(target.shards, target.length)
    shard_hashes = [
        shard_hash(content_seed, target_path, start, end, target.length)
        for start, end in intervals
    ]
    return digests[target_path], shard_hit_digest(shard_hashes), False


def run_server(
    node_id: str,
    host: str,
    port: int,
    scale: int,
    fail_mode: str,
    startup_timeout_s: float,
) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        json.dump(
            {"type": "bind_error", "node_id": node_id, "error": str(exc)},
            sys.stderr,
        )
        sys.stderr.write("\n")
        return 3
    sock.listen(64)
    quant = fp.quant_version(scale)
    ready_line = {"type": "ready", "node_id": node_id, "port": port}
    print(json.dumps(ready_line), flush=True)
    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        try:
            sock.settimeout(0.25)
            conn, _addr = sock.accept()
        except socket.timeout:
            continue
        with conn:
            conn.settimeout(5.0)
            try:
                handshake = _recv_line(conn)
            except Exception:  # pragma: no cover - malformed client
                continue
            if handshake.get("type") != "handshake":
                continue
            _send(
                conn,
                {
                    "type": "hello",
                    "node_id": node_id,
                    "quant_version": quant,
                    "status": "ready",
                },
            )
            if fail_mode == "exit":
                conn.close()
                return 7
            dispatch = _recv_line(conn)
            started = time.monotonic()
            if fail_mode == "timeout":
                time.sleep(60.0)
            digest, sh_digest, _sat = compute_digests(
                dispatch["path_table"],
                dispatch["path"],
                int(dispatch["content_seed"]),
                scale,
            )
            cost_ms = max(0, int((time.monotonic() - started) * 1000))
            receipt = {
                "type": "receipt",
                "node_id": node_id,
                "version": int(dispatch["version"]),
                "path": dispatch["path"],
                "derived_digest": digest,
                "shard_hit_digest": sh_digest,
                "cost_ms": cost_ms,
                "quant_version": (
                    quant + "-MISMATCH" if fail_mode == "quant_mismatch" else quant
                ),
            }
            if fail_mode == "bad_reply":
                receipt.pop("derived_digest")
            _send(conn, receipt)
    sock.close()
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="simulated edge cache node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument(
        "--fail-mode",
        default="none",
        choices=["none", "timeout", "exit", "bad_reply", "quant_mismatch"],
    )
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    args = parser.parse_args(argv)
    return run_server(
        args.node_id,
        args.host,
        args.port,
        args.scale,
        args.fail_mode,
        args.startup_timeout_s,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
