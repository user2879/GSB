"""Observation sampling against a target service endpoint.

Samples are taken at a fixed interval for a bounded window; every request
is recorded individually (mono clock ms, ok flag, latency ms) and becomes
the ground truth stored in the ledger. Summaries are derived with integer
fixed-point arithmetic (see fixedpoint.py) and marked undecidable when the
sample count is below the declared threshold.
"""

import socket
import time

import numpy as np

from . import fixedpoint


def mono_ms():
    return int(time.monotonic() * 1000)


def do_request(port, timeout=5.0):
    """One real loopback request. Returns (ok, latency_ms, mono_ms)."""
    start = time.monotonic()
    stamp = mono_ms()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
            conn.sendall(b"REQ\n")
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(1024)
                if not chunk:
                    break
                buf += chunk
        ok = 1 if buf.strip() == b"OK" else 0
    except OSError:
        ok = 0
    latency = int((time.monotonic() - start) * 1000)
    return ok, latency, stamp


def observe_window(port, duration_ms, interval_ms):
    """Collect raw samples for one window. Returns list of dicts."""
    samples = []
    deadline = time.monotonic() + duration_ms / 1000.0
    idx = 0
    while time.monotonic() < deadline:
        ok, latency, stamp = do_request(port)
        samples.append({"req_idx": idx, "mono_ms": stamp, "ok": ok,
                        "latency_ms": latency})
        idx += 1
        next_tick = time.monotonic() + interval_ms / 1000.0
        if next_tick < deadline:
            time.sleep(max(0.0, next_tick - time.monotonic()))
    return samples


def summarize(samples, scale, min_samples):
    """Fixed-point summary of raw samples (numpy-backed sorting)."""
    lat = np.array([s["latency_ms"] for s in samples], dtype=np.int64)
    ok = np.array([s["ok"] for s in samples], dtype=np.int64)
    return fixedpoint.summarize(lat.tolist(), ok.tolist(), scale, min_samples)
